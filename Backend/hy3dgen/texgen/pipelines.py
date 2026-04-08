# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.


import logging
import numpy as np
import os
import torch
from PIL import Image
from typing import Union, Optional

from .differentiable_renderer.mesh_render import MeshRender
from .utils.dehighlight_utils import Light_Shadow_Remover
from .utils.multiview_utils import Multiview_Diffusion_Net
from hy3dgen.system_utils import empty_cache, get_device
from .utils.imagesuper_utils import Image_Super_Net
from .utils.uv_warp_utils import mesh_uv_wrap

logger = logging.getLogger(__name__)


class Hunyuan3DTexGenConfig:

    def __init__(self, light_remover_ckpt_path, multiview_ckpt_path, device=get_device()):
        self.device = device
        self.light_remover_ckpt_path = light_remover_ckpt_path
        self.multiview_ckpt_path = multiview_ckpt_path

        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

        self.render_size = 1024
        self.texture_size = 1024
        self.bake_exp = 4
        self.merge_method = 'fast'


class Hunyuan3DPaintPipeline:
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        original_model_path = model_path
        
        # 1. Check if model_path is a direct local directory
        if os.path.exists(model_path):
            delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(model_path, 'hunyuan3d-paint-v2-0')
            if os.path.exists(delight_model_path) and os.path.exists(multiview_model_path):
                return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, device=kwargs.get('device', get_device())))

        # 2. Check local cache directory
        base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
        local_model_path = os.path.expanduser(os.path.join(base_dir, original_model_path))
        
        delight_model_path = os.path.join(local_model_path, 'hunyuan3d-delight-v2-0')
        multiview_model_path = os.path.join(local_model_path, 'hunyuan3d-paint-v2-0')

        if os.path.exists(delight_model_path) and os.path.exists(multiview_model_path):
            return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, device=kwargs.get('device', get_device())))

        # 3. Try to download from Hugging Face
        try:
            import huggingface_hub
            logger.info(f"Attempting to download {original_model_path} from Hugging Face Hub...")
            # download from huggingface
            download_path = huggingface_hub.snapshot_download(
                repo_id=original_model_path,
                allow_patterns=["hunyuan3d-delight-v2-0/*", "hunyuan3d-paint-v2-0/*"]
            )
            delight_model_path = os.path.join(download_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(download_path, 'hunyuan3d-paint-v2-0')
            return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, device=kwargs.get('device', get_device())))
        except ImportError:
            logger.error("HuggingFace Hub not installed. Cannot download models.")
            raise RuntimeError("huggingface_hub is required for downloading models. Please install it.")
        except Exception as e:
            logger.error(f"Failed to download model from Hugging Face: {e}")
            raise FileNotFoundError(f"Model {original_model_path} not found locally and download failed.")

    def __init__(self, config):
        self.config = config
        self.models = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size)

        self.load_models()

    def load_models(self):
        # empty cache
        empty_cache()
        # Load model
        self.models['delight_model'] = Light_Shadow_Remover(self.config)
        self.models['multiview_model'] = Multiview_Diffusion_Net(self.config)
        # self.models['super_model'] = Image_Super_Net(self.config)

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = get_device()):
        self.models['delight_model'].pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
        self.models['multiview_model'].pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)

    def render_normal_multiview(self, camera_elevs, camera_azims, use_abs_coor=True):
        normal_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            normal_map = self.render.render_normal(
                elev, azim, use_abs_coor=use_abs_coor, return_type='pl')
            normal_maps.append(normal_map)

        return normal_maps

    def render_position_multiview(self, camera_elevs, camera_azims):
        position_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            position_map = self.render.render_position(
                elev, azim, return_type='pl')
            position_maps.append(position_map)

        return position_maps

    def bake_from_multiview(self, views, camera_elevs,
                            camera_azims, view_weights, method='graphcut'):
        project_textures, project_weighted_cos_maps = [], []
        project_boundary_maps = []
        for view, camera_elev, camera_azim, weight in zip(
            views, camera_elevs, camera_azims, view_weights):
            project_texture, project_cos_map, project_boundary_map = self.render.back_project(
                view, camera_elev, camera_azim)
            project_cos_map = weight * (project_cos_map ** self.config.bake_exp)
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_cos_map)
            project_boundary_maps.append(project_boundary_map)

        if method == 'fast':
            texture, ori_trust_map = self.render.fast_bake_texture(
                project_textures, project_weighted_cos_maps)
        else:
            raise f'no method {method}'
        return texture, ori_trust_map > 1E-8

    def texture_inpaint(self, texture, mask):

        texture_np = self.render.uv_inpaint(texture, mask)
        texture = torch.tensor(texture_np / 255).float().to(texture.device)

        return texture

    def recenter_image(self, image, border_ratio=0.2):
        if image.mode == 'RGB':
            return image
        elif image.mode == 'L':
            image = image.convert('RGB')
            return image

        alpha_channel = np.array(image)[:, :, 3]
        non_zero_indices = np.argwhere(alpha_channel > 0)
        if non_zero_indices.size == 0:
            raise ValueError("Image is fully transparent")

        min_row, min_col = non_zero_indices.min(axis=0)
        max_row, max_col = non_zero_indices.max(axis=0)

        cropped_image = image.crop((min_col, min_row, max_col + 1, max_row + 1))

        width, height = cropped_image.size
        border_width = int(width * border_ratio)
        border_height = int(height * border_ratio)

        new_width = width + 2 * border_width
        new_height = height + 2 * border_height

        square_size = max(new_width, new_height)

        new_image = Image.new('RGBA', (square_size, square_size), (255, 255, 255, 0))

        paste_x = (square_size - new_width) // 2 + border_width
        paste_y = (square_size - new_height) // 2 + border_height

        new_image.paste(cropped_image, (paste_x, paste_y))
        return new_image

    @torch.inference_mode()
    def __call__(self, mesh, image):
        import sys
        def trace(msg):
            sys.stderr.write(f"\n[TexGen Trace] {msg}\n")
            sys.stderr.flush()

        trace("Starting texture generation...")
        if isinstance(image, str):
            image_prompt = Image.open(image)
        else:
            image_prompt = image

        trace("Recentering image...")
        image_prompt = self.recenter_image(image_prompt)

        trace("Running light/shadow remover (delight_model)...")
        image_prompt = self.models['delight_model'](image_prompt)

        trace("Running UV wrap...")
        mesh = mesh_uv_wrap(mesh)

        trace("Loading mesh into renderer...")
        self.render.load_mesh(mesh)

        selected_camera_elevs, selected_camera_azims, selected_view_weights = \
            self.config.candidate_camera_elevs, self.config.candidate_camera_azims, self.config.candidate_view_weights

        trace("Rendering normal/position maps (combined)...")
        normal_maps = []
        position_maps = []
        for elev, azim in zip(selected_camera_elevs, selected_camera_azims):
            normal_maps.append(self.render.render_normal(elev, azim, use_abs_coor=True, return_type='pl'))
            position_maps.append(self.render.render_position(elev, azim, return_type='pl'))

        trace("Running multiview diffusion model (Inference)...")
        camera_info = [(((azim // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[
            elev] + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[elev] for azim, elev in
                       zip(selected_camera_azims, selected_camera_elevs)]
        
        trace(f"Multiview model starting with {len(camera_info)} views...")
        multiviews = self.models['multiview_model'](image_prompt, normal_maps + position_maps, camera_info)

        trace("Resizing multiviews...")
        for i in range(len(multiviews)):
            multiviews[i] = multiviews[i].resize(
                (self.config.render_size, self.config.render_size))

        trace("Baking from multiviews...")
        texture, mask = self.bake_from_multiview(multiviews,
                                                 selected_camera_elevs, selected_camera_azims, selected_view_weights,
                                                 method=self.config.merge_method)

        trace("Preparing mask...")
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        trace("Final inpainting...")
        texture = self.texture_inpaint(texture, mask_np)

        trace("Setting texture on renderer...")
        self.render.set_texture(texture)
        textured_mesh = self.render.save_mesh()
        trace("Texture generation complete.")

        return textured_mesh
