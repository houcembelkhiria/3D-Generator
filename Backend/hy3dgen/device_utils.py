"""
Centralized device management for cross-platform optimal performance.
Works on CUDA GPUs, Apple Silicon (MPS), and CPU-only machines.
"""
import os as _os
# Allow MPS operations that aren't implemented in Metal to fall back to CPU
# instead of hard-crashing. Universally recommended for Mac AI pipelines.
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
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
        # Always set CPU threads — texgen models (delight, multiview) run on CPU
        # even when main device is MPS. Apple Silicon: use 10 of 12 cores.
        import os as _os, multiprocessing as _mp, platform as _platform
        cpu = _mp.cpu_count()
        if _platform.machine() in ("arm64", "aarch64"):
            cpu_threads = min(10, cpu)
        else:
            cpu_threads = max(1, cpu // 2)
        torch.set_num_threads(cpu_threads)
        _os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
        _os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
        if self.device.type == "cuda":
            # TF32: ~3x faster conv/matmul on Ampere+ with <0.1% accuracy loss
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Auto-tune conv algorithms for this hardware/input-size combination
            torch.backends.cudnn.benchmark = True
        if self.device.type == "cpu":
            # On Apple Silicon (M2 Pro = 8 perf + 4 efficiency), use 10 threads
            # so heavy CPU work (multiview/delight) saturates perf cores.
            import os as _os, multiprocessing as _mp, platform as _platform
            cpu = _mp.cpu_count()
            if _platform.machine() in ("arm64", "aarch64"):
                phys = min(10, cpu)  # Apple Silicon: leave 2 cores for OS/IO
            else:
                phys = max(1, cpu // 2)  # x86: avoid hyper-thread contention
            torch.set_num_threads(phys)
            torch.set_num_interop_threads(max(1, min(4, phys // 2)))
            _os.environ.setdefault("OMP_NUM_THREADS", str(phys))
            _os.environ.setdefault("MKL_NUM_THREADS", str(phys))


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
