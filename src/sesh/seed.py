from __future__ import annotations

import os
import random

from ._optional import HAS_NUMPY, HAS_TORCH

if HAS_NUMPY:
    import numpy as np

if HAS_TORCH:
    import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Set random seeds across environment engines for reproducibility.

    Always seeds Python's standard ``random`` module. numpy and torch are
    seeded too if they are installed; if either is absent, seeding it is
    silently skipped rather than raising, since neither is a required
    dependency of sesh.

    Args:
        seed: Base seed value used by Python, NumPy, and PyTorch.
        deterministic: If True, forces algorithms to be completely deterministic
            at the cost of minor performance overhead. Only affects PyTorch's
            cuDNN/CUBLAS behaviour, and only when PyTorch is installed.
    """
    # 1. Standard Python random engine (always available)
    random.seed(seed)

    # 2. NumPy random engine, if installed
    if HAS_NUMPY:
        np.random.seed(seed)

    # 3. PyTorch CPU/GPU engine and determinism flags, if installed
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            # Forces cuDNN to use predictable, reproducible algorithms
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # Enforces deterministic behavior inside core PyTorch operations
            torch.use_deterministic_algorithms(True)

            # Ensures multi-threaded operations on CPU/GPU match identically
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        else:
            # Default high-performance behavior
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            torch.use_deterministic_algorithms(False)


if __name__ == "__main__":
    # Test execution block to demonstrate usage
    base_seed = 42

    print(f"--- Initializing reproducible state with seed: {base_seed} ---")
    set_seed(base_seed, deterministic=True)

    if HAS_TORCH:
        test_tensor = torch.randn(2, 3)
        print("\nGenerated Verification Tensor:")
        print(test_tensor)
    else:
        print("\ntorch is not installed; only stdlib `random` was seeded.")

    print("\n--- State successfully initialized ---")
