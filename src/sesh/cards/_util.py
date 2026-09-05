from __future__ import annotations

import re
from pathlib import Path

_SLUG_PATTERN = re.compile(r"[^a-zA-Z0-9]+")


def slugify(name: str) -> str:
    """Convert an arbitrary name into a filesystem-safe slug.

    Args:
        name: The human-readable name to convert.

    Returns:
        A lowercase, hyphen-separated slug. Falls back to ``"unnamed"`` if
        the input contains no alphanumeric characters.
    """
    slug = _SLUG_PATTERN.sub("-", name.strip()).strip("-").lower()
    return slug or "unnamed"


def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    """Resolve a collision-free file path within a directory.

    Mirrors the timestamp-and-counter collision convention already used by
    ``Session`` for output directories: if ``{stem}{suffix}`` already
    exists, an incrementing numeric suffix is appended until a free path is
    found.

    Args:
        directory: Directory the file will be written into.
        stem: Filename stem, without extension.
        suffix: File extension, including the leading dot (e.g. ``".md"``).

    Returns:
        A path guaranteed not to already exist inside ``directory``.
    """
    candidate = directory / f"{stem}{suffix}"
    counter = 0
    while candidate.exists():
        counter += 1
        candidate = directory / f"{stem}_{counter}{suffix}"
    return candidate
