"""Weight-tensor-level optimizations for the Hunyuan3D pipeline.

Two manipulations live here:

1. `fuse_lora_safe` — runs diffusers' `fuse_lora()` and then walks the
   whole pipeline forcing every float tensor to a single target dtype.
   This works around the "some submodule tensors remain fp32 after fuse"
   bug that previously blocked us from fusing Hyper-SD into the SDXL UNet
   on MPS. See text2image.py for the call-site.

2. `cast_submodules_to` — recursively cast a root module to a target
   dtype, but skip submodules matching any of `skip_names`. Used by the
   opt-in bf16 cast on texgen DiTs, where VAE decoders must stay fp32 to
   preserve texture colour fidelity (commit 4caf310).

Also holds the on-disk cache path helpers used by the fused-LoRA cache:
first boot fuses + saves, later boots `from_pretrained(cache_path)`.
"""
from __future__ import annotations

import logging
import platform
from pathlib import Path
from typing import Iterable, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Where fused / dtype-manipulated models are persisted so we don't redo the
# fuse work on every process start. Lives under the existing generated/ tree.
FUSED_CACHE_ROOT = Path("generated/fused_models")


def fused_cache_path(name: str) -> Path:
    """Absolute path to a named fused-model cache directory.

    Example: fused_cache_path("hyper-sdxl") -> .../generated/fused_models/hyper-sdxl
    """
    p = FUSED_CACHE_ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def has_fused_cache(name: str) -> bool:
    """True iff a fused pipeline has been persisted under this name."""
    p = fused_cache_path(name)
    return p.is_dir() and any(p.iterdir())


# ---------------------------------------------------------------------------
# LoRA fusion
# ---------------------------------------------------------------------------

def _normalize_float_tensors(root: nn.Module, dtype: torch.dtype) -> int:
    """Force every floating-point parameter/buffer under `root` to `dtype`.

    Returns the number of tensors touched. This is the defensive pass that
    makes `fuse_lora()` safe on MPS — it eliminates any fp32 residue the
    fuse step may have left on otherwise-downcast modules.
    """
    touched = 0
    for m in root.modules():
        for name, p in list(m.named_parameters(recurse=False)):
            if p.is_floating_point() and p.dtype != dtype:
                p.data = p.data.to(dtype=dtype)
                touched += 1
        for name, b in list(m.named_buffers(recurse=False)):
            if b.is_floating_point() and b.dtype != dtype:
                b.data = b.data.to(dtype=dtype)
                touched += 1
    return touched


def fuse_lora_safe(
    pipe,
    *,
    target_device: str | torch.device,
    target_dtype: torch.dtype,
    submodule_names: Iterable[str] = ("unet", "text_encoder", "text_encoder_2", "vae"),
) -> Tuple[int, int]:
    """Fuse any loaded LoRA adapters into base weights and normalise dtype.

    Equivalent in effect to:
        pipe.fuse_lora()
        pipe.unload_lora_weights()
        pipe.to(device=..., dtype=...)
    but with an additional dtype pass over every submodule to work around
    the diffusers bug where `fuse_lora()` occasionally leaves a handful of
    tensors at fp32 on otherwise-downcast modules — which then crashes
    MPS matmul with a dtype mismatch.

    Returns (fused_modules_estimate, dtype_touched) for logging.
    """
    fused = 0
    try:
        pipe.fuse_lora()
        fused = 1  # diffusers doesn't return a count; record only that we ran
    except Exception as exc:  # pragma: no cover — only fires if no adapter loaded
        logger.warning("[weight_optim] fuse_lora() skipped: %s", exc)

    try:
        pipe.unload_lora_weights()
    except Exception as exc:  # pragma: no cover
        logger.warning("[weight_optim] unload_lora_weights() skipped: %s", exc)

    # Walk every listed submodule and force dtype. This is the step that
    # prevents the MPS matmul dtype mismatch.
    touched = 0
    for name in submodule_names:
        sub = getattr(pipe, name, None)
        if isinstance(sub, nn.Module):
            touched += _normalize_float_tensors(sub, target_dtype)

    # Finally place the pipe on-device; .to() with dtype is idempotent after
    # the walk above, but kept for any submodules we didn't enumerate.
    pipe.to(device=target_device, dtype=target_dtype)
    return fused, touched


# ---------------------------------------------------------------------------
# Selective dtype cast (for bf16 texgen)
# ---------------------------------------------------------------------------

def bf16_is_supported(device: str | torch.device) -> bool:
    """Whether casting weights to bfloat16 is known-safe on this host.

    - MPS: yes on modern PyTorch (2.1+). Metal's bf16 matmul path is used.
    - CUDA: fp16 is preferred via DeviceManager; skip bf16 cast (no gain).
    - CPU: only on ARM64 (Apple Silicon has NEON bf16). Skip on x86 since
      bf16 there requires AVX-512 which is not guaranteed.
    """
    dev = device.type if isinstance(device, torch.device) else str(device)
    if dev == "mps":
        return True
    if dev == "cuda":
        return False  # DeviceManager already does fp16; bf16 would regress
    if dev == "cpu":
        return platform.machine() in ("arm64", "aarch64")
    return False


def cast_submodules_to(
    root: nn.Module,
    dtype: torch.dtype,
    *,
    skip_names: Iterable[str] = (),
) -> dict:
    """Cast `root` to `dtype` except immediate children whose name matches skip_names.

    Used to narrow DiT / UNet backbones to bf16 while leaving colour-sensitive
    VAE decoders at fp32. Only immediate-children names are matched — the rule
    "skip=(vae,)" means the top-level `root.vae` stays fp32, not every module
    deep inside named anything related to vae.

    Returns {"cast": [names], "skipped": [names], "bytes_before": ..., "bytes_after": ...}
    for logging.
    """
    skip = {s.lower() for s in skip_names}

    def param_bytes(m: nn.Module) -> int:
        return sum(p.numel() * p.element_size() for p in m.parameters())

    bytes_before = param_bytes(root)
    cast, skipped = [], []

    for name, child in root.named_children():
        if name.lower() in skip:
            skipped.append(name)
            continue
        child.to(dtype=dtype)
        cast.append(name)

    bytes_after = param_bytes(root)

    return {
        "cast": cast,
        "skipped": skipped,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "dtype": str(dtype),
    }


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"
