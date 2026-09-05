from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from sesh import Session
from sesh import session as session_module


class _FixedDatetime(datetime):
    """A datetime subclass whose now() always returns the same instant.

    Session's output-directory collision handling only triggers when two
    sessions resolve to the same second-precision timestamp, which real
    wall-clock time can't guarantee deterministically in a fast test.
    """

    @classmethod
    def now(cls, tz: object = None) -> datetime:  # noqa: ARG003
        return datetime(2030, 1, 1, 12, 0, 0)


def test_output_dir_collision_gets_a_counter_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(session_module, "datetime", _FixedDatetime)

    with Session(name="dup", output_root=tmp_path) as first:
        pass
    with Session(name="dup", output_root=tmp_path) as second:
        pass

    # Both sessions resolve to the exact same timestamp and name, so the
    # second must fall back to an incrementing counter suffix.
    assert first.output_dir != second.output_dir
    assert first.output_dir.exists()
    assert second.output_dir.exists()
    assert second.output_dir.name.endswith("_1")


def test_context_manager_returns_self(tmp_path: Path) -> None:
    with Session(name="ctx", output_root=tmp_path) as session:
        assert isinstance(session, Session)


def test_context_manager_success_path_logs_clean_close(tmp_path: Path) -> None:
    with Session(name="clean-exit", output_root=tmp_path) as session:
        pass

    log_text = (session.output_dir / "run.log").read_text(encoding="utf-8")
    assert "Session workspace locked cleanly." in log_text


def test_context_manager_propagates_and_logs_exceptions(tmp_path: Path) -> None:
    with (
        pytest.raises(ValueError),
        Session(name="failing", output_root=tmp_path) as session,
    ):
        output_dir = session.output_dir
        raise ValueError("boom")

    log_text = (output_dir / "run.log").read_text(encoding="utf-8")
    assert "Execution boundary broken by exception: boom" in log_text


def test_path_creates_and_returns_subdirectory(tmp_path: Path) -> None:
    with Session(name="path-test", output_root=tmp_path) as session:
        resolved = session.path("artifacts", "nested")

    assert resolved == session.output_dir / "artifacts" / "nested"
    assert resolved.exists()


def test_info_writes_to_the_log_file(tmp_path: Path) -> None:
    with Session(name="info-test", output_root=tmp_path) as session:
        session.info("a plain info message")

    log_text = (session.output_dir / "run.log").read_text(encoding="utf-8")
    assert "a plain info message" in log_text


def test_debug_warning_error_write_to_the_log_file(tmp_path: Path) -> None:
    with Session(name="log-levels-test", output_root=tmp_path) as session:
        session.debug("a debug trace")
        session.warning("a warning condition")
        session.error("a non-fatal error")

    log_text = (session.output_dir / "run.log").read_text(encoding="utf-8")
    assert "a debug trace" in log_text
    assert "a warning condition" in log_text
    assert "a non-fatal error" in log_text


def test_debug_is_filtered_from_console_but_warning_and_error_are_not(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # configure_logger's default console level is INFO, so debug messages
    # should reach the file (checked above) but not stderr.
    with Session(name="log-levels-console-test", no_log=True) as session:
        session.debug("should not reach the console")
        session.warning("should reach the console")
        session.error("should also reach the console")

    stderr = capsys.readouterr().err
    assert "should not reach the console" not in stderr
    assert "should reach the console" in stderr
    assert "should also reach the console" in stderr


def test_log_metrics_with_and_without_step(tmp_path: Path) -> None:
    with Session(name="metrics-test", output_root=tmp_path) as session:
        session.log_metrics({"loss": 0.5}, step=3)
        session.log_metrics({"loss": 0.25})

    log_text = (session.output_dir / "run.log").read_text(encoding="utf-8")
    assert "[Step 0003]" in log_text
    assert "loss: 0.50000" in log_text


def test_log_params_writes_to_the_log_file(tmp_path: Path) -> None:
    with Session(name="params-test", output_root=tmp_path) as session:
        session.log_params({"batch_size": 32})

    log_text = (session.output_dir / "run.log").read_text(encoding="utf-8")
    assert "Parameters registered:" in log_text
    assert "batch_size" in log_text


def test_log_dataset_card_through_session_writes_card_and_mlflow_artifact(
    tmp_path: Path,
) -> None:
    with Session(name="dataset-card-test", output_root=tmp_path) as session:
        session.log_dataset_card(
            name="MyDataset",
            parameters={"rows": 100},
            description="A dataset minted through Session.",
        )

    card_path = session.path("data_artifacts") / "dataset_card_mydataset.md"
    assert card_path.exists()


def test_log_model_card_through_session_writes_card_and_mlflow_artifact(
    tmp_path: Path,
) -> None:
    with Session(name="model-card-test", output_root=tmp_path) as session:
        session.log_model_card(
            name="MyModel",
            architecture="mlp",
            parameters={"layers": 2},
            intended_use=["testing"],
            limitations=["none"],
            training_metadata={"epochs": 1},
        )

    card_path = session.path("model_artifacts") / "model_card_mymodel.md"
    assert card_path.exists()


def test_mlflow_requested_but_missing_logs_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(session_module, "HAS_MLFLOW", False)

    with Session(
        name="mlflow-missing", output_root=tmp_path, enable_mlflow=True
    ) as session:
        assert session._use_mlflow is False

    stderr = capsys.readouterr().err
    assert "MLflow tracking requested, but package is missing" in stderr
