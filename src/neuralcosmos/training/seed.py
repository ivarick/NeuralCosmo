"""Seeding and determinism.

Plan reference: sections 58, 64, 79.

Section 58 requires results to be reported across multiple seeds, which is only
meaningful if a seed actually determines a run.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

__all__ = ["seed_everything", "worker_init_fn", "describe_environment"]


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and torch.

    ``deterministic`` additionally forces deterministic cuDNN kernels. It is off
    by default because it can cost significant throughput and, for the
    convolutions used here, run-to-run variation is far smaller than the seed
    variation being measured. Turn it on when chasing a reproducibility bug.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Autotuning helps when input shapes are fixed, which they are here.
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a distinct, reproducible stream."""
    try:
        import torch

        base = torch.initial_seed() % (2**32)
    except ImportError:
        base = 0
    np.random.seed((base + worker_id) % (2**32))
    random.seed(base + worker_id)


def describe_environment() -> dict[str, Any]:
    """Software and hardware provenance for the run record (section 64)."""
    import platform
    import sys

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
            )
    except ImportError:
        info["torch"] = None
    return info
