from __future__ import annotations

from pathlib import Path

import pytest

from sesh import Session


def test_no_log_creates_no_output_directory(tmp_path: Path) -> None:
    with Session(name="one-off", output_root=tmp_path, no_log=True) as session:
        assert session.output_dir is None

    assert list(tmp_path.iterdir()) == []


def test_no_log_still_logs_to_console(capsys: pytest.CaptureFixture[str]) -> None:
    # Session's logger sets propagate=False, so caplog (which hooks the root
    # logger) can't see it - assert on the console handler's actual stderr
    # output instead.
    with Session(name="one-off-console", no_log=True) as session:
        session.info("hello from a one-off session")

    assert "hello from a one-off session" in capsys.readouterr().err


def test_no_log_disables_mlflow_even_if_requested(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with Session(
        name="one-off-mlflow",
        output_root=tmp_path,
        no_log=True,
        enable_mlflow=True,
    ) as session:
        assert session._use_mlflow is False

    assert "no_log=True disables MLflow" in capsys.readouterr().err


def test_no_log_path_raises(tmp_path: Path) -> None:
    with (
        Session(name="one-off-path", output_root=tmp_path, no_log=True) as session,
        pytest.raises(RuntimeError),
    ):
        session.path("some_folder")


def test_no_log_log_dataset_card_raises(tmp_path: Path) -> None:
    with (
        Session(name="one-off-dataset", output_root=tmp_path, no_log=True) as session,
        pytest.raises(RuntimeError),
    ):
        session.log_dataset_card(name="d", parameters={}, description="test dataset")


def test_no_log_log_model_card_raises(tmp_path: Path) -> None:
    with (
        Session(name="one-off-model", output_root=tmp_path, no_log=True) as session,
        pytest.raises(RuntimeError),
    ):
        session.log_model_card(
            name="m",
            architecture="mlp",
            parameters={},
            intended_use=[],
            limitations=[],
        )
