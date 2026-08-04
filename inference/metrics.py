"""Server-side measurement helpers.

Everything the result card shows is measured here, around the *real* work -- never
estimated. Cold vs warm is reported honestly by the pipeline registry.
"""

from __future__ import annotations

import threading
import time


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Peak VRAM
#
# TensorRT allocates its engine, execution-context workspace, and I/O buffers
# *outside* torch's caching allocator, so ``torch.cuda.max_memory_allocated()``
# misses almost all of it (it reported ~0.02 GB for an 18 GB engine). Instead we
# poll the whole device's free/total memory via ``torch.cuda.mem_get_info`` --
# which reflects ALL allocations (torch, TensorRT, the CUDA context) -- and track
# the peak used between ``reset_peak_vram`` and ``peak_vram_gb`` with a small
# background sampler. On a dedicated serving GPU that used-bytes figure is the
# real VRAM footprint.
# --------------------------------------------------------------------------- #

_sampler: dict = {"thread": None, "stop": False, "peak": 0, "dev": 0}


def _device_used_bytes(dev: int) -> int:
    import torch

    free, total = torch.cuda.mem_get_info(dev)
    return total - free


def _stop_sampler() -> None:
    thread = _sampler["thread"]
    if thread is not None:
        _sampler["stop"] = True
        thread.join(timeout=0.5)
        _sampler["thread"] = None


def reset_peak_vram() -> None:
    """Start tracking peak device VRAM for the next generation."""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        _stop_sampler()
        dev = torch.cuda.current_device()
        _sampler.update(stop=False, dev=dev, peak=_device_used_bytes(dev))

        def _loop() -> None:
            while not _sampler["stop"]:
                try:
                    used = _device_used_bytes(_sampler["dev"])
                    if used > _sampler["peak"]:
                        _sampler["peak"] = used
                except Exception:
                    break
                time.sleep(0.02)  # 20 ms

        thread = threading.Thread(target=_loop, name="vram-sampler", daemon=True)
        _sampler["thread"] = thread
        thread.start()
    except Exception:
        pass


def peak_vram_gb() -> float:
    """Peak device VRAM used since reset, in GB (0.0 off-GPU). Includes TRT + torch."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        torch.cuda.synchronize()
        used = _device_used_bytes(_sampler["dev"])
        if used > _sampler["peak"]:
            _sampler["peak"] = used
        _stop_sampler()
        return round(_sampler["peak"] / (1024**3), 2)
    except Exception:
        return 0.0


def cuda_sync() -> None:
    """Block until queued CUDA work finishes -- required for honest timing."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
