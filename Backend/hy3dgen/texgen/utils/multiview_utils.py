import os
import random
import numpy as np
import torch
from hy3dgen.device_utils import get_device_manager
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler


class Multiview_Diffusion_Net():
    def __init__(self, config) -> None:
        self.view_size = 512
        multiview_ckpt_path = config.multiview_ckpt_path

        current_file_path = os.path.abspath(__file__)
        custom_pipeline_path = os.path.join(os.path.dirname(current_file_path), '..', 'hunyuanpaint')

        # Resolve device from config (may be 'mps', 'cuda', 'cpu')
        dm = get_device_manager()
        self.device = str(dm.device)
        self.dtype = dm.dtype

        # Force CPU + float32 when device is MPS — avoids MPS OOM/black output
        if self.device == 'mps':
            print("[MV] Forcing Multiview_Diffusion_Net to CPU/float32 (MPS not supported)")
            self.device = 'cpu'
            self.dtype = torch.float32

        print(f"[MV] Loading Multiview Diffusion Net. Device: {self.device}, Dtype: {self.dtype}")
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
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipeline.scheduler.config, timestep_spacing='trailing')
        finally:
            torch.set_default_device(prev_default_device)

        pipeline.set_progress_bar_config(disable=True)
        pipeline = pipeline.to(device=self.device, dtype=self.dtype)

        # Component-level force (safety net for diffusers auto-placement)
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

        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(0))

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

        import sys, time
        sys.stderr.write(f"\n[MV] Calling pipeline (device={self.pipeline.device}, dtype={self.dtype}, steps=30)...\n")
        sys.stderr.flush()
        t0 = time.time()
        mvd_image = self.pipeline(input_image, num_inference_steps=30, **kwargs).images
        for _vi, _vimg in enumerate(mvd_image):
            _varr = np.array(_vimg)
            _mask = _varr[:,:,:3].sum(axis=2) < 700
            if _mask.sum() > 0:
                _px = _varr[:,:,:3][_mask][:500]
                sys.stderr.write("[MV VIEW %d] R=%.0f G=%.0f B=%.0f ch_std=%.1f%s" % (
                    _vi, _px[:,0].mean(), _px[:,1].mean(), _px[:,2].mean(),
                    _px.std(axis=1).mean(), chr(10)))
        sys.stderr.write(f"[MV] Done in {time.time()-t0:.1f}s\n")
        sys.stderr.flush()
        return mvd_image
