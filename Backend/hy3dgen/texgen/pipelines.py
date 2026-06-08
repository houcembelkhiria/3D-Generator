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
from .utils.uv_warp_utils import mesh_uv_wrap

logger = logging.getLogger(__name__)


class Hunyuan3DTexGenConfig:
    def __init__(self, light_remover_ckpt_path, multiview_ckpt_path, device=get_device()):
        self.device = device
        self.light_remover_ckpt_path = light_remover_ckpt_path
        self.multiview_ckpt_path = multiview_ckpt_path
        # 6 views matching working fork (Hunyuan3D-2GP). Top/bottom carry low
        # weight (0.05 each) but close coverage gaps so the back doesn't depend
        # entirely on inpainting from front-view colours.
        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]
        self.render_size = 2048
        self.texture_size = 2048
        self.bake_exp = 4
        self.merge_method = 'fast'


class Hunyuan3DPaintPipeline:
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        original_model_path = model_path
        if os.path.exists(model_path):
            delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(model_path, 'hunyuan3d-paint-v2-0')
            if os.path.exists(delight_model_path) and os.path.exists(multiview_model_path):
                return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, device=kwargs.get('device', get_device())))
        base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
        local_model_path = os.path.expanduser(os.path.join(base_dir, original_model_path))
        delight_model_path = os.path.join(local_model_path, 'hunyuan3d-delight-v2-0')
        multiview_model_path = os.path.join(local_model_path, 'hunyuan3d-paint-v2-0')
        if os.path.exists(delight_model_path) and os.path.exists(multiview_model_path):
            return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, device=kwargs.get('device', get_device())))
        try:
            import huggingface_hub
            download_path = huggingface_hub.snapshot_download(
                repo_id=original_model_path,
                allow_patterns=["hunyuan3d-delight-v2-0/*", "hunyuan3d-paint-v2-0/*"])
            delight_model_path = os.path.join(download_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(download_path, 'hunyuan3d-paint-v2-0')
            return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, device=kwargs.get('device', get_device())))
        except Exception as e:
            raise FileNotFoundError(f"Model {original_model_path} not found: {e}")

    def __init__(self, config):
        self.config = config
        self.models = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size)
        self.load_models()

    def load_models(self):
        empty_cache()
        self.models['delight_model'] = Light_Shadow_Remover(self.config)
        self.models['multiview_model'] = Multiview_Diffusion_Net(self.config)

    def enable_model_cpu_offload(self, gpu_id=None, device=get_device()):
        self.models['delight_model'].pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
        self.models['multiview_model'].pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)

    def bake_from_multiview(self, views, camera_elevs, camera_azims, view_weights, method='graphcut'):
        project_textures, project_weighted_cos_maps = [], []
        project_boundary_maps = []
        for _vi, (view, elev, azim, weight) in enumerate(zip(views, camera_elevs, camera_azims, view_weights)):
            project_texture, project_cos_map, project_boundary_map = self.render.back_project(view, elev, azim)
            # DEBUG: save per-view texture
            try:
                _dbg = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'generated', '_debug')
                os.makedirs(_dbg, exist_ok=True)
                _t = (project_texture.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(_t).save(os.path.join(_dbg, f'view_texture_{_vi}.png'))
            except Exception:
                pass
            project_cos_map = weight * (project_cos_map ** self.config.bake_exp)
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_cos_map)
            project_boundary_maps.append(project_boundary_map)
        if method == 'fast':
            texture, ori_trust_map = self.render.fast_bake_texture(project_textures, project_weighted_cos_maps)
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
            return image.convert('RGB')
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

    # Maps texgen camera azimuth (elev=0) → user view name.
    # Front is already the primary `image` input so we only substitute the others.
    _AZIM_TO_VIEW = {90: 'right', 180: 'back', 270: 'left'}

    @torch.inference_mode()
    def __call__(self, mesh, image, user_views=None):
        """
        Args:
            mesh: trimesh.Trimesh
            image: front-view PIL image (used for delight + multiview seed)
            user_views: optional dict of {view_name: PIL.Image} from a multiview
                        capture. When provided, generated views at matching azimuths
                        are replaced with the user's actual images, giving accurate
                        back/left/right texture instead of AI-hallucinated colours.
        """
        import sys, time as _time
        _t0 = _time.time()
        _last = [_t0]
        def trace(msg):
            now = _time.time()
            sys.stderr.write(f"\n[TexGen +{now - _t0:.1f}s D{now - _last[0]:.1f}s] {msg}\n")
            sys.stderr.flush()
            _last[0] = now

        trace("Starting texture generation...")
        image_prompt = Image.open(image) if isinstance(image, str) else image
        trace("Recentering image...")
        image_prompt = self.recenter_image(image_prompt)
        trace("Running delight (CPU + 50 steps, working fork config)...")
        image_prompt = self.models['delight_model'](image_prompt)
        try:
            _dbg = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'generated', '_debug')
            os.makedirs(_dbg, exist_ok=True)
            image_prompt.save(os.path.join(_dbg, 'delight_output.png'))
        except Exception:
            pass
        trace("Running UV wrap...")
        mesh = mesh_uv_wrap(mesh)
        trace("Loading mesh into renderer...")
        self.render.load_mesh(mesh)

        selected_camera_elevs = self.config.candidate_camera_elevs
        selected_camera_azims = self.config.candidate_camera_azims
        selected_view_weights = self.config.candidate_view_weights

        trace("Rendering normal/position maps...")
        normal_maps, position_maps = [], []
        for elev, azim in zip(selected_camera_elevs, selected_camera_azims):
            normal_maps.append(self.render.render_normal(elev, azim, use_abs_coor=True, return_type='pl'))
            position_maps.append(self.render.render_position(elev, azim, return_type='pl'))

        trace("Running multiview (CPU + float32 + 30 steps, like working fork)...")
        camera_info = [(((azim // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[
            elev] + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[elev] for azim, elev in
                       zip(selected_camera_azims, selected_camera_elevs)]
        multiviews = self.models['multiview_model'](image_prompt, normal_maps + position_maps, camera_info)

        trace("Resizing multiviews...")
        for i in range(len(multiviews)):
            multiviews[i] = multiviews[i].resize((self.config.render_size, self.config.render_size))

        # Substitute user-provided views where available (back/left/right).
        # This replaces AI-hallucinated texture with the user's actual images,
        # which matters most for the back (weight=0.5) and sides (weight=0.1).
        if user_views:
            for i, (azim, elev) in enumerate(zip(selected_camera_azims, selected_camera_elevs)):
                if elev != 0:
                    continue
                view_name = self._AZIM_TO_VIEW.get(azim)
                if view_name and view_name in user_views:
                    try:
                        user_img = self.recenter_image(user_views[view_name])
                        user_img = user_img.convert('RGB').resize(
                            (self.config.render_size, self.config.render_size))
                        multiviews[i] = user_img
                        trace(f"Substituted {view_name} view (azim={azim}°) with user image")
                    except Exception as _e:
                        trace(f"Warning: could not substitute {view_name} view: {_e}")

        trace("Baking from multiviews...")
        texture, mask = self.bake_from_multiview(multiviews,
            selected_camera_elevs, selected_camera_azims, selected_view_weights,
            method=self.config.merge_method)
        # DEBUG: save baked texture before inpaint
        try:
            _tex_np = (texture.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(_tex_np).save(os.path.join(_dbg, 'baked_texture_raw.png'))
            # Also save the first multiview input to compare
            multiviews[0].save(os.path.join(_dbg, 'multiview_input_0.png'))
            trace(f"DEBUG: saved baked texture + input to {_dbg}/")
        except Exception as _e:
            trace(f"DEBUG: save failed: {_e}")

        trace("Preparing mask...")
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        trace("Final inpainting...")
        texture = self.texture_inpaint(texture, mask_np)
        # DEBUG: save texture AFTER inpainting (this is what goes on the mesh)
        try:
            _tex_final = (texture.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(_tex_final).save(os.path.join(_dbg, 'texture_after_inpaint.png'))
            trace(f"DEBUG: saved final texture ({_tex_final.shape}) to {_dbg}/texture_after_inpaint.png")
        except Exception as _e:
            trace(f"DEBUG: save failed: {_e}")
        trace("Setting texture on renderer...")
        self.render.set_texture(texture)
        textured_mesh = self.render.save_mesh()
        trace("Texture generation complete.")
        return textured_mesh
