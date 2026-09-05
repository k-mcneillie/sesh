from __future__ import annotations

import random

import pytest

from sesh import seed as seed_module
from sesh.seed import set_seed


def test_set_seed_seeds_stdlib_random_deterministically() -> None:
    set_seed(123)
    first = random.random()

    set_seed(123)
    second = random.random()

    assert first == second


def test_set_seed_works_without_numpy_or_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seed_module, "HAS_NUMPY", False)
    monkeypatch.setattr(seed_module, "HAS_TORCH", False)

    # Should not raise even though numpy/torch are reported as unavailable.
    set_seed(7, deterministic=True)
    set_seed(7, deterministic=False)


@pytest.mark.skipif(not seed_module.HAS_NUMPY, reason="numpy not installed")
def test_set_seed_seeds_numpy_when_installed() -> None:
    import numpy as np

    set_seed(99)
    first = np.random.rand()

    set_seed(99)
    second = np.random.rand()

    assert first == second


@pytest.mark.skipif(not seed_module.HAS_TORCH, reason="torch not installed")
def test_set_seed_seeds_torch_when_installed() -> None:
    import torch

    set_seed(2024)
    first = torch.rand(3)

    set_seed(2024)
    second = torch.rand(3)

    assert torch.equal(first, second)
