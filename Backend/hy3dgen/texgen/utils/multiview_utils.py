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

import os
import random

import numpy as np
import torch
from hy3dgen.device_utils import get_device_manager
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler


class Multiview_Diffusion_Net():
    def __init__(self, config) -> None:
        self.device = config.device
        self.view_size = 512
        multiview_ckpt_path = config.multiview_ckpt_path

        current_file_path = os.path.abspath(__file__)
        custom_pipeline_path = os.path.join(os.path.dirname(current_file_path), '..', 'hunyuanpaint')

        _dm = get_device_manager()
        # Texgen models too large for MPS VRAM — fall back to CPU on MPS
        self.device = 'cpu' if _dm.device.type == 'mps' else str(_dm.device)
        self.dtype = torch.float32 if self.device == 'cpu' else _dm.dtype
        self._autocast_dtype = torch.bfloat16 if self.device == 'cpu' else _dm.autocast_dtype

        pipeline = DiffusionPipeline.from_pretrained(
            multiview_ckpt_path,
            custom_pipeline=custom_pipeline_path, 
            torch_dtype=self.dtype,
            use_safetensors=False
        )

        # Temporarily set default device to cpu to avoid MPS conversion error in diffusers
        prev_default_device = torch.get_default_device()
        torch.set_default_device('cpu')
        try:
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config,
                                                                             timestep_spacing='trailing')
        finally:
            torch.set_default_device(prev_default_device)

        pipeline.set_progress_bar_config(disable=True)
        # Ensure all sub-modules are moved to the target device and dtype
        pipeline = pipeline.to(device=self.device, dtype=self.dtype)
        
        # Component-level force to be absolutely certain
        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
            pipeline.vae.to(device=self.device, dtype=self.dtype)
        if hasattr(pipeline, 'unet') and pipeline.unet is not None:
            pipeline.unet.to(device=self.device, dtype=self.dtype)
        if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
            pipeline.text_encoder.to(device=self.device, dtype=self.dtype)

        self.pipeline = pipeline

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PL_GLOBAL_SEED"] = str(seed)

    def __call__(self, input_image, control_images, camera_info):

        self.seed_everything(0)

        input_image = input_image.resize((self.view_size, self.view_size))
        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((self.view_size, self.view_size))
            if control_images[i].mode == 'L':
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode='1')

        kwargs = dict(generator=torch.Generator(device=self.device).manual_seed(0))

        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]

        camera_info_gen = [camera_info]
        camera_info_ref = [[0]]
        kwargs['width'] = self.view_size
        kwargs['height'] = self.view_size
        kwargs['num_in_batch'] = num_view
        kwargs['camera_info_gen'] = camera_info_gen
        kwargs['camera_info_ref'] = camera_info_ref
        kwargs["normal_imgs"] = normal_image
        kwargs["position_imgs"] = position_image

        with torch.inference_mode(), torch.amp.autocast(self.device, dtype=self._autocast_dtype):
          mvd_image = self.pipeline(input_image, num_inference_steps=15, **kwargs).images
        return mvd_image
