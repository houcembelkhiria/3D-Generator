import os
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from hy3dgen.device_utils import get_device_manager


def _delight_steps_env_default() -> int:
    """Default for the InstructPix2Pix delight pipeline.

    HY3D_DELIGHT_STEPS overrides; falls back to 20 (the prior hardcoded
    default of this util's __call__).
    """
    raw = os.environ.get("HY3D_DELIGHT_STEPS")
    if not raw:
        return 20
    try:
        n = int(raw)
        return max(1, n)
    except ValueError:
        return 20


def _texgen_device_override() -> str | None:
    """Same override mechanism as multiview_utils — see that file's docstring."""
    v = os.environ.get("HY3D_TEXGEN_DEVICE", "").strip().lower()
    if v in ("mps", "cuda", "cpu"):
        return v
    return None


class Light_Shadow_Remover:
    def __init__(self, config):
        delight_ckpt_path = config.light_remover_ckpt_path
        dm = get_device_manager()
        device_str = str(dm.device)

        override = _texgen_device_override()
        if override is not None:
            self.device = override
            if override == "mps":
                torch_dtype = torch.float32
            else:
                torch_dtype = dm.dtype if override != "cpu" else torch.float32
        elif device_str == 'mps':
            # Force CPU + float32 when device is MPS (matches working fork — see commit 4caf310).
            # Set HY3D_TEXGEN_DEVICE=mps to bypass and try MPS again.
            self.device = 'cpu'
            torch_dtype = torch.float32
        else:
            self.device = device_str
            torch_dtype = dm.dtype

        # Working fork values
        self.cfg_image = 1.5
        self.cfg_text = 1.0

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            delight_ckpt_path, torch_dtype=torch_dtype, safety_checker=None)
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
        pipeline.set_progress_bar_config(disable=True)
        pipeline = pipeline.to(self.device)

        self.pipeline = pipeline

    def recorrect_rgb(self, src_image, target_image, alpha_channel, scale=0.95):
        def flat_and_mask(bgr, a):
            mask = a[:, :, 0] > 0.5
            bgr_flat = bgr.reshape(-1, 3)
            mask_flat = mask.reshape(-1)
            return bgr_flat[mask_flat, :]
        src_flat = flat_and_mask(src_image, alpha_channel)
        target_flat = flat_and_mask(target_image, alpha_channel)
        corrected_bgr = torch.zeros_like(src_image)
        for i in range(3):
            src_mean, src_stddev = torch.mean(src_flat[:, i]), torch.std(src_flat[:, i])
            target_mean, target_stddev = torch.mean(target_flat[:, i]), torch.std(target_flat[:, i])
            corrected_bgr[:, :, i] = torch.clamp(
                (src_image[:, :, i] - scale * src_mean) * (target_stddev / src_stddev) + scale * target_mean, 0, 1)
        src_mse = torch.mean((src_image - target_image) ** 2)
        modify_mse = torch.mean((corrected_bgr - target_image) ** 2)
        if src_mse < modify_mse:
            corrected_bgr = torch.cat([src_image, alpha_channel], dim=-1)
        else:
            corrected_bgr = torch.cat([corrected_bgr, alpha_channel], dim=-1)
        return corrected_bgr

    def __call__(self, image, num_inference_steps: int | None = None):
        # If caller passes None, defer to env var (HY3D_DELIGHT_STEPS) → fallback 20.
        if num_inference_steps is None:
            num_inference_steps = _delight_steps_env_default()
        image = image.resize((512, 512), Image.Resampling.LANCZOS)

        if image.mode == 'RGBA':
            image_array = np.array(image)
            alpha_channel = image_array[:, :, 3].astype(np.uint8)
            erosion_size = 3
            kernel = np.ones((erosion_size, erosion_size), np.uint8)
            alpha_channel = cv2.erode(alpha_channel, kernel, iterations=1)
            image_array[alpha_channel == 0, :3] = 255
            image_array[:, :, 3] = alpha_channel
            image = Image.fromarray(image_array)
            image_tensor = torch.tensor(np.array(image) / 255.0).to(device=self.device, dtype=torch.float32)
            alpha = image_tensor[:, :, 3:]
            rgb_target = image_tensor[:, :, :3]
        else:
            image_tensor = torch.tensor(np.array(image) / 255.0).to(device=self.device, dtype=torch.float32)
            alpha = torch.ones_like(image_tensor)[:, :, :1]
            rgb_target = image_tensor[:, :, :3]
            image = image.convert('RGB')

        pipe_image = image.convert('RGB') if image.mode != 'RGB' else image

        gen_device = "cpu" if str(self.device) == "mps" else self.device
        image = self.pipeline(
            prompt="",
            image=pipe_image,
            generator=torch.Generator(device=gen_device).manual_seed(42),
            height=512,
            width=512,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=self.cfg_image,
            guidance_scale=self.cfg_text,
            output_type='pil',
        ).images[0]

        image_tensor = torch.tensor(np.array(image)/255.0).to(device=self.device, dtype=torch.float32)
        rgb_src = image_tensor[:,:,:3]
        image = self.recorrect_rgb(rgb_src, rgb_target, alpha)
        image = image[:,:,:3]*image[:,:,3:] + torch.ones_like(image[:,:,:3])*(1.0-image[:,:,3:])
        image = image.nan_to_num(0.0).clamp(0.0, 1.0)
        image = Image.fromarray((image.cpu().numpy()*255).astype(np.uint8))
        return image
