"""
Hunyuan3D model configuration.
Settings for 3D generation models loaded at startup.
"""
import os
from dataclasses import dataclass, field


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


@dataclass
class Hunyuan3DSettings:
    device: str = field(default_factory=lambda: os.environ.get("HY3D_DEVICE", _get_device()))
    cache_path: str = field(default_factory=lambda: os.environ.get("HY3D_CACHE_PATH", "generated/3d_outputs"))

    # Primary model (image-to-3d / text-to-3d)
    model_path: str = field(default_factory=lambda: os.environ.get("HY3D_MODEL_PATH", "tencent/Hunyuan3D-2mini"))
    subfolder: str = field(default_factory=lambda: os.environ.get("HY3D_SUBFOLDER", "hunyuan3d-dit-v2-mini-turbo"))

    # Multi-view model
    enable_mv: bool = field(default_factory=lambda: os.environ.get("HY3D_ENABLE_MV", "true").lower() == "true")
    mv_model_path: str = "tencent/Hunyuan3D-2mv"
    mv_subfolder: str = field(default_factory=lambda: os.environ.get("HY3D_MV_SUBFOLDER", "hunyuan3d-dit-v2-mv-turbo"))

    # Texture model
    enable_tex: bool = field(default_factory=lambda: os.environ.get("HY3D_ENABLE_TEX", "true").lower() == "true")
    tex_model_path: str = "tencent/Hunyuan3D-2"

    # Text-to-image bridge
    enable_t23d: bool = field(default_factory=lambda: os.environ.get("HY3D_ENABLE_T23D", "true").lower() == "true")

    # Performance
    enable_quantization: bool = field(default_factory=lambda: os.environ.get('HY3D_QUANTIZE', 'false').lower() == 'true')
    enable_flashvdm: bool = field(default_factory=lambda: os.environ.get("HY3D_TURBO", "true").lower() == "true")
    mc_algo: str = "mc"
    profile: int = 3
    verbose: int = 1

    # TexGen inference step counts (lower = faster, higher = better quality)
    delight_steps: int = field(default_factory=lambda: int(os.environ.get("HY3D_DELIGHT_STEPS", "8")))
    multiview_steps: int = field(default_factory=lambda: int(os.environ.get("HY3D_MULTIVIEW_STEPS", "6")))
