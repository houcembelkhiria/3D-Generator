"""
Centralized device management for cross-platform optimal performance.
Works on CUDA GPUs, Apple Silicon (MPS), and CPU-only machines.
"""
import platform
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass
class DeviceManager:
    device: torch.device
    dtype: torch.dtype            # model weight dtype
    autocast_dtype: torch.dtype   # autocast precision
    autocast_enabled: bool        # whether autocast helps
    quantization_engine: str      # qnnpack (ARM) or fbgemm (x86)
    mc_algo: str                  # marching cubes algo

    @staticmethod
    def detect(override: str = None) -> "DeviceManager":
        """Auto-detect the best device and optimal settings."""
        if override:
            device_str = override
        elif torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch.backends, "mps") and hasattr(torch.backends.mps, "is_available") and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"

        device = torch.device(device_str)
        is_arm = platform.machine() in ("arm64", "aarch64")

        if device_str == "cuda":
            return DeviceManager(
                device=device,
                dtype=torch.float16,
                autocast_dtype=torch.float16,
                autocast_enabled=True,
                quantization_engine="fbgemm",
                mc_algo="mc",  # could use dmc on CUDA if available
            )
        elif device_str == "mps":
            return DeviceManager(
                device=device,
                dtype=torch.float32,          # MPS needs float32 weights
                autocast_dtype=torch.float16,  # but ops can use float16
                autocast_enabled=True,
                quantization_engine="qnnpack",
                mc_algo="mc",
            )
        else:
            # CPU
            supports_bf16 = is_arm or _cpu_supports_avx512()
            return DeviceManager(
                device=device,
                dtype=torch.float32,
                autocast_dtype=torch.bfloat16 if supports_bf16 else torch.float32,
                autocast_enabled=supports_bf16,
                quantization_engine="qnnpack" if is_arm else "fbgemm",
                mc_algo="mc",
            )

    @contextmanager
    def autocast(self):
        """Context manager for device-optimal mixed precision."""
        if self.autocast_enabled:
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.autocast_dtype,
                enabled=True,
            ):
                yield
        else:
            yield

    def generator(self, seed: int) -> torch.Generator:
        """Create a seeded generator on the correct device."""
        return torch.Generator(device=self.device).manual_seed(seed)

    def cpu_generator(self, seed: int) -> torch.Generator:
        """Create a CPU generator (for pipelines that require it)."""
        return torch.Generator(device="cpu").manual_seed(seed)

    @property
    def is_gpu(self) -> bool:
        return self.device.type in ("cuda", "mps")

    def setup_globals(self):
        """Set global torch optimizations for this device."""
        torch.set_float32_matmul_precision("high")
        try:
            torch.backends.quantized.engine = self.quantization_engine
        except Exception:
            pass


def _cpu_supports_avx512() -> bool:
    """Check if CPU supports AVX-512 (for bfloat16 on x86)."""
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                return "avx512" in f.read().lower()
    except Exception:
        pass
    return False


# Singleton
_dm: DeviceManager = None


def get_device_manager(override: str = None) -> DeviceManager:
    global _dm
    if _dm is None or override:
        _dm = DeviceManager.detect(override)
    return _dm
