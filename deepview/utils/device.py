"""
Device selection helper
========================
Single source of truth for picking the compute device across DeepView.

Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU.

On Apple Silicon there is no CUDA; the Metal (MPS) backend is used instead so
training/inference still runs on the GPU. ``PYTORCH_ENABLE_MPS_FALLBACK`` is
set so the few ops not yet implemented for MPS transparently fall back to CPU
instead of raising.

Note: MPS does not support float64 (double). Always move float tensors to the
device as float32 (e.g. ``tensor.to(device=device, dtype=torch.float)``).
"""

import os

# Must be set before the first MPS op runs. Lets unsupported ops fall back to
# CPU rather than crashing. setdefault so an explicit user override wins.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch


def _mps_available():
    return (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )


def get_device(cuda_index=None):
    """Return the best available ``torch.device``.

    Priority: CUDA > MPS (Apple Silicon) > CPU.

    Set the env var ``DEEPVIEW_DEVICE`` (e.g. ``cpu``, ``mps``, ``cuda:1``) to
    force a specific device.

    Parameters
    ----------
    cuda_index : int | str | None
        Preferred CUDA device when CUDA is available. Accepts ``0``, ``"0"`` or
        ``"cuda:0"``. Ignored on MPS / CPU.
    """
    forced = os.environ.get("DEEPVIEW_DEVICE")
    if forced:
        return torch.device(forced)

    if torch.cuda.is_available():
        if cuda_index is None:
            return torch.device("cuda")
        cuda_index = str(cuda_index)
        if cuda_index.startswith("cuda"):
            return torch.device(cuda_index)
        return torch.device("cuda:" + cuda_index)

    if _mps_available():
        return torch.device("mps")

    return torch.device("cpu")
