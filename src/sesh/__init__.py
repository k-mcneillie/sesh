from __future__ import annotations

from importlib.metadata import version

from .session import Session

__all__ = ["Session"]

__version__ = version("sesh")
