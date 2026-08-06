"""Device selection helpers for CUDA / CPU with graceful fallback."""
from __future__ import annotations


def resolve_device(preference: str = "auto") -> str:
    """Return a torch device string.

    ``auto`` picks CUDA when available, otherwise CPU. Passing ``cuda``
    explicitly still falls back to CPU (with the intent recorded) so that the
    same config can run on a laptop and a GPU server unchanged.
    """
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:  # pragma: no cover
        has_cuda = False

    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        return "cuda" if has_cuda else "cpu"
    return "cuda" if has_cuda else "cpu"


def cuda_device_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:  # pragma: no cover
        return 0
