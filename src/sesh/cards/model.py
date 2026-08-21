# model.py
# - construct and save model card when new data is generated.

# =============================
# Import Libraries
# =============================
from __future__ import annotations


from pathlib import Path
from typing import Any


class ModelCard:
    """Human-readable documentation for a trained model."""

    def __init__(
        self,
        *,
        name: str,
        description: str | tuple[str] | list[str],
        architecture: str,
        parameters: dict[str, Any],
        intended_use: list[str],
        limitations: list[str],
        training: dict[str, Any],
    ) -> None:
        """Initialise a model card."""
        self.name = name
        self.description = [description] if isinstance(description, str) else list(description)
        self.architecture = architecture
        self.parameters = parameters
        self.intended_use = intended_use
        self.limitations = limitations
        self.training = training

    def save(self, output_dir: Path) -> Path:
        """Save the model card as Markdown.

        Args:
            output_dir: Directory in which to save the model card.

        Returns:
            Path to the saved model card.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "model_card.md"
        self._save_markdown(path)
        return path

    def _save_markdown(self, path: Path) -> None:
        """Save the human-readable model card."""
        lines = [
            "# Model Card",
            "",
            "## Model",
            "",
            f"- Name: `{self.name}`",
            f"- Architecture: `{self.architecture}`",
            "",
            "## Description",
            "",
            *self.description,
            "",
            "## Parameters",
            "",
        ]

        for name, value in self.parameters.items():
            lines.append(f"- `{name}`: `{value}`")

        lines.extend([
            "",
            "## Intended Use",
            "",
            *[f"- {item}" for item in self.intended_use],
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in self.limitations],
            "",
            "## Training",
            "",
        ])

        for name, value in self.training.items():
            lines.append(f"- `{name}`: `{value}`")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")