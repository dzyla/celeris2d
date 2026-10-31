"""The GPU backend, so nothing else names CUDA: NVIDIA (CUDA) and AMD (ROCm: PyTorch's ROCm build answers to torch.cuda
and the device name "cuda", torch.version.hip set) are "cuda"; Intel (torch.xpu, same calls) is "xpu"; "cpu" without
a GPU. CELERIS2D_DEVICE=cuda / xpu / cpu chooses one."""
from __future__ import annotations

import contextlib
import functools
import os

import torch


@functools.cache
def kind() -> str:
    """The device type every job uses: cuda (NVIDIA or AMD), xpu (Intel) or cpu."""
    env = os.environ.get("CELERIS2D_DEVICE", "").strip().lower()
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def available() -> bool:
    return kind() != "cpu"


def _mod(dev=None):
    """torch.cuda or torch.xpu for dev (default: the backend); None for the CPU."""
    t = torch.device(dev).type if dev is not None else kind()
    return getattr(torch, t) if t in ("cuda", "xpu") else None


def is_gpu(dev) -> bool:
    return torch.device(dev).type in ("cuda", "xpu")


def default() -> str:
    """The device of a job that does not choose one: "cuda", "xpu" or "cpu"."""
    return kind()


def device(i: int) -> str:
    return f"{kind()}:{i}" if available() else "cpu"


def count() -> int:
    """GPUs of the backend; 1 (the CPU) when CELERIS2D_DEVICE=cpu asks for the CPU (slow: tests, machines without a
    supported GPU)."""
    if not available():
        return 1 if os.environ.get("CELERIS2D_DEVICE", "").strip().lower() == "cpu" else 0
    return _mod().device_count()


def name(i: int = 0) -> str:
    return _mod().get_device_name(i) if available() else "CPU"


def vendor() -> str:
    if kind() == "xpu":
        return "Intel (XPU)"
    if kind() == "cuda":
        return "AMD (ROCm)" if torch.version.hip else "NVIDIA (CUDA)"
    return "CPU"


def _index(dev) -> torch.device:
    dev = torch.device(dev)
    return dev if dev.index is not None else torch.device(dev.type, _mod(dev).current_device())


def _ram() -> tuple[int, int]:
    """(available, total) bytes of system memory: the "device" memory of the CPU."""
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                info[k] = int(v.split()[0]) * 1024
        return info.get("MemAvailable", info["MemFree"]), info["MemTotal"]
    except (OSError, KeyError, ValueError):          # not Linux: half of the physical memory, if known
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (AttributeError, ValueError, OSError):
            total = 16 << 30
        return total // 2, total


def total_memory(dev) -> int:
    if not is_gpu(dev):
        return _ram()[1]
    return _mod(dev).get_device_properties(_index(dev)).total_memory


def mem_get_info(dev) -> tuple[int, int]:
    """(free, total) bytes of the device (system memory for the CPU)."""
    if not is_gpu(dev):
        return _ram()
    m, dev = _mod(dev), _index(dev)
    try:
        return m.mem_get_info(dev)
    except (AttributeError, RuntimeError):           # a backend without it: what this process has not reserved
        total = total_memory(dev)
        return total - m.memory_reserved(dev), total


def memory_reserved(dev) -> int:
    return _mod(dev).memory_reserved(dev) if is_gpu(dev) else 0


def memory_allocated(dev) -> int:
    return _mod(dev).memory_allocated(dev) if is_gpu(dev) else 0


def max_memory_allocated(dev) -> int:
    return _mod(dev).max_memory_allocated(dev) if is_gpu(dev) else 0


def reset_peak_memory_stats(dev):
    if is_gpu(dev):
        _mod(dev).reset_peak_memory_stats(dev)


def set_memory_fraction(fraction: float, dev):
    f = getattr(_mod(dev), "set_per_process_memory_fraction", None)
    if f is not None:
        f(fraction, _index(dev))


def synchronize(dev=None):
    m = _mod(dev)
    if m is not None and (dev is not None or available()):
        m.synchronize(dev)


def empty_cache():
    if available():
        _mod().empty_cache()


def initialized() -> bool:
    return available() and _mod().is_initialized()


def use(dev):
    """with use(dev): kernels go to dev (the backend's device context; nothing on the CPU)."""
    return _mod(dev).device(dev) if is_gpu(dev) else contextlib.nullcontext()


def set_device(dev):
    if is_gpu(dev):
        _mod(dev).set_device(dev)


def stream(dev):
    """A side stream on dev (None on the CPU); `with on_stream(s)` uses it."""
    return _mod(dev).Stream(dev) if is_gpu(dev) else None


def on_stream(s):
    return _mod(s.device).stream(s) if s is not None else contextlib.nullcontext()


def pin() -> bool:
    """Host buffers pinned for faster copies to the GPU."""
    return available()


OutOfMemoryError = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)
