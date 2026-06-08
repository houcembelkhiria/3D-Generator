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
from diffusers import AutoPipelineForText2Image


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)


class HunyuanDiTPipeline:
    def __init__(
        self,
        # v1.2 distilled: ~2x faster than v1.1 with no quality drop (Tencent progressive distillation).
        model_path="Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
        device='cpu',
        dtype=torch.float16
    ):
        torch.set_default_device('cpu')
        self.device = device
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_path,
            torch_dtype=dtype,
            enable_pag=True,
            pag_applied_layers=["blocks.(16|17|18|19)"]
        ).to(device)
        # DPM++ 2M SDE Karras: sharper output at the same step count vs default
        try:
            from diffusers import DPMSolverMultistepScheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                algorithm_type="sde-dpmsolver++",
                use_karras_sigmas=True,
            )
        except Exception:
            pass  # fall back to whatever the model shipped with
        # Prompt template tuned for 3D reconstruction:
        # - centered, single subject, clean white bg → shape-gen sees a clear volume
        # - English keywords work better for diverse prompts than Chinese-only tags
        self.pos_txt = (
            ", single centered object, eye-level view, 3/4 front angle, "
            "clean white background, studio lighting, high detail, accurate proportions, "
            "3D render style, best quality, 白色背景, 正确比例, 3D风格, 最佳质量"
        )
        self.neg_txt = (
            "multiple objects, cluttered, complex scene, busy background, "
            "text, watermark, logo, signature, low quality, blurry, noisy, "
            "deformed, cropped, out of frame, extra limbs, mutated, disfigured, "
            "文本, 特写, 裁剪, 出框, 最差质量, 低质量, JPEG伪影, 重复, 病态, "
            "残缺, 多余的手指, 变异的手, 画得不好的手, 画得不好的脸, 变异, 畸形, 模糊, "
            "糟糕的比例, 多余的肢体, 融合的手指, 手指太多, 长脖子"
        )

    def compile(self):
        # accelarate hunyuan-dit transformer,first inference will cost long time
        torch.set_float32_matmul_precision('high')
        self.pipe.transformer = torch.compile(self.pipe.transformer, fullgraph=True)
        # self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, fullgraph=True)
        generator = torch.Generator(device=self.pipe.device)  # infer once for hot-start
        out_img = self.pipe(
            prompt='美少女战士',
            negative_prompt='模糊',
            num_inference_steps=15,
            pag_scale=1.3,
            width=1024,
            height=1024,
            generator=generator,
            return_dict=False
        )[0][0]

    @torch.no_grad()
    def __call__(self, prompt, seed=0, num_inference_steps=10):
        seed_everything(seed)
        generator = torch.Generator(device=self.device)
        generator = generator.manual_seed(int(seed))
        # Build prompt: put user intent first, then style hints.
        # No truncation — HunyuanDiT tokenizer caps at 77 tokens automatically.
        full_prompt = f"a detailed 3D render of a {prompt.strip()}{self.pos_txt}"
        out_img = self.pipe(
            prompt=full_prompt,
            negative_prompt=self.neg_txt,
            num_inference_steps=num_inference_steps,
            # guidance_scale 6.0 (HunyuanDiT default/recommended for best adherence),
            # pag_scale 2.0 for sharper detail (was 1.3 — too soft)
            guidance_scale=6.0,
            pag_scale=2.0,
            width=1024,
            height=1024,
            generator=generator,
            return_dict=False
        )[0][0]
        return out_img


class SDXLHyperPipeline:
    """Hyper-SDXL 4-step text-to-image pipeline (ByteDance).

    Drop-in replacement for HunyuanDiTPipeline with the same call signature.
    Uses SDXL-base + Hyper-SD 4-step LoRA + TCDScheduler.

    Benchmarks (community, 2024-2026): Hyper-SD >= SDXL-Lightning in quality at
    equal step count. Strong English subject knowledge (LAION-5B training).
    Fits in 32 GB MPS with shape-gen loaded.
    """

    POS_TEMPLATE = (
        ", centered, eye-level view, 3/4 front angle, clean white background, "
        "studio lighting, high detail, accurate proportions, 3D render style, best quality"
    )
    NEG_PROMPT = (
        "blurry, low quality, low resolution, noisy, deformed, multiple objects, "
        "cluttered, complex background, text, watermark, logo, signature, "
        "extra limbs, mutated, disfigured, cropped, out of frame"
    )

    def __init__(
        self,
        model_path="stabilityai/stable-diffusion-xl-base-1.0",
        vae_path="madebyollin/sdxl-vae-fp16-fix",  # fp16-safe VAE (stock SDXL VAE NaNs in fp16)
        lora_repo="ByteDance/Hyper-SD",
        lora_weight="Hyper-SDXL-4steps-lora.safetensors",
        device='cpu',
        dtype=torch.float16,
    ):
        from diffusers import StableDiffusionXLPipeline, TCDScheduler, AutoencoderKL
        from huggingface_hub import hf_hub_download

        torch.set_default_device('cpu')
        self.device = device

        # SDXL base with fp16-fixed VAE (stock SDXL VAE produces NaN in fp16 on MPS)
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
            use_safetensors=True,
            vae=vae,
        )

        # Load Hyper-SD 4-step LoRA as an active adapter (DO NOT fuse — fuse_lora()
        # leaves some submodule tensors at fp32 which breaks MPS matmul dtype check)
        lora_path = hf_hub_download(repo_id=lora_repo, filename=lora_weight)
        self.pipe.load_lora_weights(lora_path)

        # Force the entire pipeline (including freshly-loaded LoRA layers) to the
        # target dtype and device together — avoids MPS "Destination NDArray and
        # Accumulator NDArray cannot have different datatype" in matmul
        self.pipe.to(device=device, dtype=dtype)

        # TCDScheduler — ByteDance-recommended for Hyper-SD few-step sampling
        self.pipe.scheduler = TCDScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)

        # Disable safety checker if present (it isn't on SDXL-base, but defensive)
        if hasattr(self.pipe, "safety_checker"):
            self.pipe.safety_checker = None

    @torch.no_grad()
    def __call__(self, prompt, seed=0, num_inference_steps=4):
        seed_everything(seed)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        full_prompt = f"a detailed 3D render of a {prompt.strip()}{self.POS_TEMPLATE}"
        out_img = self.pipe(
            prompt=full_prompt,
            negative_prompt=self.NEG_PROMPT,
            num_inference_steps=num_inference_steps,
            # Hyper-SD is distilled without CFG; guidance_scale=0 is required.
            guidance_scale=0.0,
            # eta=1.0 for TCDScheduler (stochasticity parameter)
            eta=1.0,
            width=1024,
            height=1024,
            generator=generator,
        ).images[0]
        return out_img

