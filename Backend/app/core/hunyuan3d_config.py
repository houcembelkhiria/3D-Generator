"""
Hunyuan3D model configuration.
Settings for 3D generation models loaded at startup.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


def _get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available') and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


_TRUTHY = frozenset({"1", "true", "yes", "on", "y", "t"})
_FALSY = frozenset({"0", "false", "no", "off", "n", "f", ""})


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean env var. Accepts 1/0, true/false, yes/no, on/off (any case).

    Returns `default` if the var is unset; on any unrecognised value also
    returns `default` (defensive — better than silently flipping behaviour).
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in _TRUTHY:
        return True
    if v in _FALSY:
        return False
    return default


def _env_int_optional(name: str) -> Optional[int]:
    """Read an env var as int, or None if unset / unparseable."""
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


@dataclass
class Hunyuan3DSettings:
    device: str = field(default_factory=lambda: os.environ.get("HY3D_DEVICE", _get_device()))
    cache_path: str = field(default_factory=lambda: os.environ.get("HY3D_CACHE_PATH", "generated/3d_outputs"))

    # Primary model (image-to-3d / text-to-3d)
    model_path: str = field(default_factory=lambda: os.environ.get("HY3D_MODEL_PATH", "tencent/Hunyuan3D-2mini"))
    subfolder: str = field(default_factory=lambda: os.environ.get("HY3D_SUBFOLDER", "hunyuan3d-dit-v2-mini-turbo"))

    # Multi-view model
    enable_mv: bool = field(default_factory=lambda: os.environ.get("HY3D_ENABLE_MV", "true").lower() == "true")
    mv_model_path: str = field(default_factory=lambda: os.environ.get("HY3D_MV_MODEL_PATH", "tencent/Hunyuan3D-2mv"))
    mv_subfolder: str = field(default_factory=lambda: os.environ.get("HY3D_MV_SUBFOLDER", "hunyuan3d-dit-v2-mv-turbo"))

    # Texture model
    enable_tex: bool = field(default_factory=lambda: os.environ.get("HY3D_ENABLE_TEX", "true").lower() == "true")
    tex_model_path: str = field(default_factory=lambda: os.environ.get("HY3D_TEX_MODEL_PATH", "tencent/Hunyuan3D-2"))

    # Text-to-image bridge
    enable_t23d: bool = field(default_factory=lambda: os.environ.get("HY3D_ENABLE_T23D", "true").lower() == "true")
    # T2I model: "hyper_sdxl" (Hyper-SDXL 4-step, default, English-oriented) or "hunyuan" (HunyuanDiT v1.2-Distilled, bilingual fallback)
    t2i_model: str = field(default_factory=lambda: os.environ.get("HY3D_T2I_MODEL", "hyper_sdxl").lower())
    # Text-to-image inference steps. Different models have very different
    # native step counts:
    #   - Hyper-SDXL is distilled for 4-step sampling (running 20 is wasted,
    #     and can hurt quality)
    #   - HunyuanDiT v1.2-Distilled needs ~8-12 steps for clean output;
    #     4 produces noisy garbage that breaks downstream shape-gen.
    # When unset, the service picks a model-aware default at call time.
    # Override via HY3D_T2I_STEPS to force a specific count regardless of model.
    t2i_steps: Optional[int] = field(default_factory=lambda: _env_int_optional("HY3D_T2I_STEPS"))

    # --- T2I model identifiers (change these to swap models without editing code) ---
    t2i_hunyuan_dit_model: str = field(
        default_factory=lambda: os.environ.get(
            "HY3D_T2I_HUNYUAN_MODEL",
            "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
        )
    )
    t2i_sdxl_model: str = field(
        default_factory=lambda: os.environ.get(
            "HY3D_T2I_SDXL_MODEL",
            "stabilityai/stable-diffusion-xl-base-1.0",
        )
    )
    t2i_sdxl_vae: str = field(
        default_factory=lambda: os.environ.get(
            "HY3D_T2I_SDXL_VAE",
            "madebyollin/sdxl-vae-fp16-fix",
        )
    )
    t2i_sdxl_lora_repo: str = field(
        default_factory=lambda: os.environ.get(
            "HY3D_T2I_SDXL_LORA_REPO",
            "ByteDance/Hyper-SD",
        )
    )
    t2i_sdxl_lora_weight: str = field(
        default_factory=lambda: os.environ.get(
            "HY3D_T2I_SDXL_LORA_WEIGHT",
            "Hyper-SDXL-4steps-lora.safetensors",
        )
    )

    # --- LLM model identifiers ---
    llm_model_path: str = field(
        default_factory=lambda: os.environ.get(
            "LLAMA_MODEL_PATH",
            "./models/llama-3-8b-instruct.Q4_K_M.gguf",
        )
    )
    ollama_default_model: str = field(
        default_factory=lambda: os.environ.get(
            "OLLAMA_MODEL",
            "qwen2.5:3b-instruct",
        )
    )

    # Performance
    enable_quantization: bool = field(default_factory=lambda: _env_flag("HY3D_QUANTIZE", False))
    enable_flashvdm: bool = field(default_factory=lambda: _env_flag("HY3D_TURBO", True))
    # Opt-in: cast texgen DiT/UNet backbones (multiview + delight) to bfloat16
    # on MPS / ARM CPU. Leaves VAEs at fp32 to preserve texture colour fidelity
    # (commit 4caf310). CUDA ignores this — DeviceManager already uses fp16.
    bf16_texgen: bool = field(default_factory=lambda: _env_flag("HY3D_BF16_TEXGEN", False))
    mc_algo: str = "mc"
    profile: int = 3
    verbose: int = 1

    # TexGen inference step counts (lower = faster, higher = better quality).
    # These ARE read by the texgen call sites (multiview_utils.py, dehighlight_utils.py).
    # Defaults match the pipeline's hardcoded values prior to this wiring being added.
    delight_steps: int = field(default_factory=lambda: int(os.environ.get("HY3D_DELIGHT_STEPS", "20")))
    multiview_steps: int = field(default_factory=lambda: int(os.environ.get("HY3D_MULTIVIEW_STEPS", "25")))


# Per-model t2i step defaults used when HY3D_T2I_STEPS is unset.
# Centralised here so the service can ask `T2I_STEPS_DEFAULT.get(model_name)`.
T2I_STEPS_DEFAULT = {
    "hyper_sdxl": 4,   # ByteDance's distilled 4-step LoRA
    "hunyuan": 10,     # HunyuanDiT v1.2-Distilled native step count
}
