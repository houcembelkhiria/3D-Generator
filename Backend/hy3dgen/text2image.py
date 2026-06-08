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
import os
import random
from contextlib import contextmanager

import numpy as np
import torch
from diffusers import AutoPipelineForText2Image

logger = logging.getLogger(__name__)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)


@contextmanager
def _temp_default_device(target):
    """Set torch default device for the duration of pipeline construction.

    `torch.set_default_device('cpu')` was previously called unconditionally
    at the top of __init__, leaving global state polluted for any subsequent
    pipeline (and any other code in the process). Wrapping in a context
    manager scopes the change to the loading window only.
    """
    prev = torch.get_default_device()
    torch.set_default_device(target)
    try:
        yield
    finally:
        torch.set_default_device(prev)


def _force_pipeline_to_device(pipe, device, dtype):
    """Cascade .to(device, dtype) onto every nn.Module child of a diffusers pipeline.

    Diffusers' built-in `pipe.to(device)` does not always move every
    sub-module reliably on MPS — most commonly the transformer's
    PatchEmbed.proj (Conv2d) stays on CPU even after the parent .to() call,
    which then crashes on first forward with
        RuntimeError: Input type (MPSFloatType) and weight type (torch.FloatTensor)
        should be the same.
    Walking known component names and calling .to() per child fixes this.
    """
    for attr in (
        "transformer", "unet", "vae",
        "text_encoder", "text_encoder_2",
        "image_encoder", "controlnet",
    ):
        sub = getattr(pipe, attr, None)
        if sub is None:
            continue
        if hasattr(sub, "to"):
            try:
                sub.to(device=device, dtype=dtype)
            except Exception as exc:
                logger.warning("could not move %s to %s: %s", attr, device, exc)


class HunyuanDiTPipeline:
    def __init__(
        self,
        # v1.2 distilled: ~2x faster than v1.1 with no quality drop (Tencent progressive distillation).
        model_path="Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
        device='cpu',
        dtype=torch.float16
    ):
        self.device = device
        with _temp_default_device('cpu'):
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                model_path,
                torch_dtype=dtype,
                enable_pag=True,
                pag_applied_layers=["blocks.(16|17|18|19)"]
            ).to(device)
        # Force-cascade .to() onto every component — diffusers' parent .to()
        # leaves transformer.pos_embed.proj on CPU on this model, which crashes
        # with "Input MPSFloatType / weight torch.FloatTensor" on first forward.
        _force_pipeline_to_device(self.pipe, device, dtype)
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
            ", isolated single object, centered, 3/4 front angle, eye-level view, "
            "pure white background, no shadows, no ground plane, studio lighting, "
            "high detail, sharp edges, accurate proportions, photorealistic 3D render, best quality"
        )
        self.neg_txt = (
            "multiple objects, background clutter, busy scene, "
            "shadows, ground shadow, floor reflection, environment background, "
            "text, watermark, logo, signature, "
            "low quality, blurry, noisy, JPEG artifacts, duplicate, "
            "deformed, mutated, disfigured, extra limbs, extra fingers, mutated hands, "
            "poorly drawn hands, poorly drawn face, bad proportions, fused fingers, long neck, "
            "cropped, out of frame, partial view, 2D, flat, illustration, cartoon, painting, sketch"
        )

    def compile(self):
        # accelarate hunyuan-dit transformer,first inference will cost long time
        torch.set_float32_matmul_precision('high')
        self.pipe.transformer = torch.compile(self.pipe.transformer, fullgraph=True)
        # self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, fullgraph=True)
        gen_device = "cpu" if str(self.pipe.device) == "mps" else self.pipe.device
        generator = torch.Generator(device=gen_device)  # infer once for hot-start
        out_img = self.pipe(
            prompt='sailor moon',
            negative_prompt='blurry',
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
        gen_device = "cpu" if str(self.device) == "mps" else self.device
        generator = torch.Generator(device=gen_device)
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

    LoRA is fused into base UNet weights at first load and the fused pipeline
    is cached under generated/fused_models/hyper-sdxl-<dtype>/. Subsequent
    starts reload from cache and skip the fuse step entirely.
    """

    POS_TEMPLATE = (
        ", isolated single object, centered, 3/4 front angle, eye-level view, "
        "pure white background, no shadows, no ground plane, studio lighting, "
        "high detail, sharp edges, accurate proportions, photorealistic 3D render, best quality"
    )
    NEG_PROMPT = (
        "blurry, low quality, low resolution, noisy, deformed, multiple objects, "
        "cluttered, complex background, text, watermark, logo, signature, "
        "shadows, ground shadow, floor reflection, environment background, "
        "extra limbs, mutated, disfigured, cropped, out of frame, partial view, "
        "2D, flat, illustration, cartoon, painting, sketch"
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

        from app.services.weight_optim import (
            fuse_lora_safe,
            fused_cache_path,
            has_fused_cache,
        )

        self.device = device

        # Cache key is dtype-specific: an fp32 fused cache is not interchangeable
        # with an fp16 one (diffusers stores weights in whatever dtype was used
        # at save time).
        dtype_tag = str(dtype).split(".")[-1]  # float32 / float16 / bfloat16
        cache_name = f"hyper-sdxl-{dtype_tag}"
        cache_path = fused_cache_path(cache_name)

        with _temp_default_device('cpu'):
            if has_fused_cache(cache_name):
                logger.info("[fused-lora] loading from cache → %s", cache_path)
                # Cached pipeline already has LoRA fused and TCDScheduler in its config.
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    str(cache_path),
                    torch_dtype=dtype,
                    use_safetensors=True,
                )
                self.pipe.to(device=device, dtype=dtype)
            else:
                logger.info(
                    "[fused-lora] no cache — fusing Hyper-SD into SDXL UNet "
                    "(one-time, ~5s). Cache target: %s",
                    cache_path,
                )
                # fp16-fixed VAE. Stock SDXL VAE produces NaN in fp16 on MPS.
                vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    variant="fp16" if dtype == torch.float16 else None,
                    use_safetensors=True,
                    vae=vae,
                )

                lora_path = hf_hub_download(repo_id=lora_repo, filename=lora_weight)
                self.pipe.load_lora_weights(lora_path)

                # Pre-set the inference scheduler so it gets written into the
                # saved pipeline's config.
                self.pipe.scheduler = TCDScheduler.from_config(self.pipe.scheduler.config)

                # Fuse LoRA deltas into base weights on CPU, then normalise dtype
                # across the pipeline. Normalisation is the step that was missing
                # in the previous "DO NOT fuse" path — without it, a handful of
                # submodule tensors end up at fp32 after fuse_lora() even when the
                # pipe was loaded at fp16, which triggers MPS matmul dtype errors.
                fused, touched = fuse_lora_safe(
                    self.pipe, target_device="cpu", target_dtype=dtype,
                )
                logger.info(
                    "[fused-lora] fused Hyper-SD LoRA; dtype-normalised %d tensors",
                    touched,
                )

                # Persist so subsequent boots skip all of the above.
                try:
                    self.pipe.save_pretrained(str(cache_path), safe_serialization=True)
                    logger.info("[fused-lora] cached fused pipeline → %s", cache_path)
                except Exception as exc:
                    logger.warning("[fused-lora] could not write cache: %s", exc)

                # Move the live pipeline to its runtime device.
                self.pipe.to(device=device, dtype=dtype)

        # Same MPS placement guard as HunyuanDiT — force-cascade onto every
        # nn.Module child to avoid stranded fp32-on-CPU sub-tensors.
        _force_pipeline_to_device(self.pipe, device, dtype)

        self.pipe.set_progress_bar_config(disable=True)

        # Disable safety checker if present (it isn't on SDXL-base, but defensive)
        if hasattr(self.pipe, "safety_checker"):
            self.pipe.safety_checker = None

    @torch.no_grad()
    def __call__(self, prompt, seed=0, num_inference_steps=4):
        seed_everything(seed)
        gen_device = "cpu" if str(self.device) == "mps" else self.device
        generator = torch.Generator(device=gen_device).manual_seed(int(seed))
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
