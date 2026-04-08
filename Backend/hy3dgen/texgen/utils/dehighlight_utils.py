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

import cv2
import numpy as np
import torch
from PIL import Image
from hy3dgen.device_utils import get_device_manager
from diffusers import DDIMScheduler, StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler


class Light_Shadow_Remover():
    def __init__(self, config):
        dm = get_device_manager(config.device)
        self.device = str(dm.device)
        torch_dtype = dm.dtype
        
        self.cfg_image = 1.5
        self.cfg_text = 1.0

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            config.light_remover_ckpt_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
        # Temporarily set default device to cpu to avoid MPS conversion error in diffusers
        prev_default_device = torch.get_default_device()
        torch.set_default_device('cpu')
        try:
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
        finally:
            torch.set_default_device(prev_default_device)
        pipeline.set_progress_bar_config(disable=True)

        self.pipeline = pipeline.to(self.device)
        # No need for manual text_encoder offloading if the whole thing is on CPU or CUDA behaves.
        # But if it's MPS, we forced the whole thing to CPU above.
        
    def recorrect_rgb(self, src_image, target_image, alpha_channel, scale=0.95):
        
        def flat_and_mask(bgr, a):
            mask = torch.where(a > 0.5, True, False)
            bgr_flat = bgr.reshape(-1, bgr.shape[-1])
            mask_flat = mask.reshape(-1)
            bgr_flat_masked = bgr_flat[mask_flat, :]
            return bgr_flat_masked
        
        src_flat = flat_and_mask(src_image, alpha_channel)
        target_flat = flat_and_mask(target_image, alpha_channel)
        corrected_bgr = torch.zeros_like(src_image)

        for i in range(3): 
            src_mean, src_stddev = torch.mean(src_flat[:, i]), torch.std(src_flat[:, i])
            target_mean, target_stddev = torch.mean(target_flat[:, i]), torch.std(target_flat[:, i])
            corrected_bgr[:, :, i] = torch.clamp((src_image[:, :, i] - scale * src_mean) * (target_stddev / src_stddev) + scale * target_mean, 0, 1)

        src_mse = torch.mean((src_image - target_image) ** 2)
        modify_mse = torch.mean((corrected_bgr - target_image) ** 2)
        if src_mse < modify_mse:
            corrected_bgr = torch.cat([src_image, alpha_channel], dim=-1)
        else: 
            corrected_bgr = torch.cat([corrected_bgr, alpha_channel], dim=-1)

        return corrected_bgr

    @torch.inference_mode()
    def __call__(self, image):

        image = image.resize((512, 512))

        if image.mode == 'RGBA':
            image_array = np.array(image)
            alpha_channel = image_array[:, :, 3]
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

        with get_device_manager(self.device).autocast():
          image = self.pipeline(
            prompt="",
            image=image,
            generator=get_device_manager(self.device).generator(42),
            height=512,
            width=512,
            num_inference_steps=20,
            image_guidance_scale=self.cfg_image,
            guidance_scale=self.cfg_text,
        ).images[0]

        image_tensor = torch.tensor(np.array(image)/255.0).to(device=self.device, dtype=torch.float32)
        rgb_src = image_tensor[:,:,:3]
        image = self.recorrect_rgb(rgb_src, rgb_target, alpha)
        image = image[:,:,:3]*image[:,:,3:] + torch.ones_like(image[:,:,:3])*(1.0-image[:,:,3:])
        image = Image.fromarray((image.cpu().numpy()*255).astype(np.uint8))

        return image