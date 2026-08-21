from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Set random seeds across environment engines for reproducibility.

    Args:
        seed: Base seed value used by Python, NumPy, and PyTorch.
        deterministic: If True, forces algorithms to be completely deterministic
            at the cost of minor performance overhead.
    """
    # 1. Standard Python and NumPy random engines
    random.seed(seed)
    np.random.seed(seed)

    # 2. Base PyTorch CPU and GPU engine configuration
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 3. Precision CUDA execution locks
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

    # Generate mock tensor data to verify reproducibility layout
    test_tensor = torch.randn(2, 3)
    print("\nGenerated Verification Tensor:")
    print(test_tensor)

    print("\n--- State successfully initialized ---")
