"""Single source of truth for sesh's optional runtime dependencies.

sesh's core (session lifecycle, logging, cards) depends only on the
standard library, so it can be used from any Python project. numpy,
torch, and mlflow unlock extra reproducibility/tracking features when
installed, but are never required.
"""

from __future__ import annotations

try:
    import numpy  # noqa: F401

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import mlflow  # noqa: F401

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
