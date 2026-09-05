from __future__ import annotations

from pathlib import Path

from sesh.cards._util import slugify, unique_path
from sesh.cards.dataset import DatasetCard
from sesh.cards.model import ModelCard


def _make_model_card(name: str) -> ModelCard:
    return ModelCard(
        name=name,
        description="A test model.",
        architecture="MLP",
        parameters={"layers": 3},
        intended_use=["testing"],
        limitations=["none"],
        training={"epochs": 1},
    )


def _make_dataset_card(name: str) -> DatasetCard:
    return DatasetCard(
        name=name,
        parameters={"rows": 10},
        seed=42,
        device="cpu",
        description="A test dataset.",
    )


def test_slugify_normalises_arbitrary_names() -> None:
    assert slugify("Anomaly Classifier Net!") == "anomaly-classifier-net"
    assert slugify("   ") == "unnamed"


def test_unique_path_appends_counter_on_collision(tmp_path: Path) -> None:
    first = unique_path(tmp_path, "card", ".md")
    first.write_text("first", encoding="utf-8")

    second = unique_path(tmp_path, "card", ".md")

    assert first != second
    assert second.name == "card_1.md"


def test_model_card_multiple_models_do_not_overwrite(tmp_path: Path) -> None:
    path_a = _make_model_card("ModelA").save(tmp_path)
    path_b = _make_model_card("ModelB").save(tmp_path)

    assert path_a != path_b
    assert path_a.exists()
    assert path_b.exists()


def test_model_card_same_name_gets_collision_suffix(tmp_path: Path) -> None:
    path_first = _make_model_card("SameName").save(tmp_path)
    path_second = _make_model_card("SameName").save(tmp_path)

    assert path_first != path_second
    assert path_first.exists()
    assert path_second.exists()
    assert path_second.name.endswith("_1.md")


def test_dataset_card_multiple_datasets_do_not_overwrite(tmp_path: Path) -> None:
    markdown_a, config_a = _make_dataset_card("DatasetA").save(tmp_path)
    markdown_b, config_b = _make_dataset_card("DatasetB").save(tmp_path)

    assert markdown_a != markdown_b
    assert config_a != config_b
    assert markdown_a.exists()
    assert markdown_b.exists()
    assert config_a.exists()
    assert config_b.exists()


def test_dataset_card_same_name_gets_collision_suffix(tmp_path: Path) -> None:
    markdown_first, config_first = _make_dataset_card("SameSet").save(tmp_path)
    markdown_second, config_second = _make_dataset_card("SameSet").save(tmp_path)

    assert markdown_first != markdown_second
    assert config_first != config_second
    assert markdown_second.name.endswith("_1.md")
    assert config_second.name.endswith("_1.json")
