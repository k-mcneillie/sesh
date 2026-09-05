from __future__ import annotations

from pathlib import Path

import pytest

mlflow = pytest.importorskip("mlflow")

from sesh import Session  # noqa: E402


def test_default_mlflow_backend_is_sqlite_scoped_to_session_dir(tmp_path: Path) -> None:
    with Session(name="mlflow-default", output_root=tmp_path) as session:
        tracking_uri = mlflow.get_tracking_uri()

    expected_db = (session.output_dir / "mlflow.db").resolve()
    assert tracking_uri == f"sqlite:///{expected_db}"
    assert expected_db.exists()


def test_explicit_mlflow_tracking_uri_overrides_default(tmp_path: Path) -> None:
    custom_uri = f"sqlite:///{(tmp_path / 'shared.db').resolve()}"

    with Session(
        name="mlflow-custom",
        output_root=tmp_path,
        mlflow_tracking_uri=custom_uri,
    ):
        assert mlflow.get_tracking_uri() == custom_uri


def test_mlflow_disabled_skips_tracking_setup(tmp_path: Path) -> None:
    with Session(name="mlflow-disabled", output_root=tmp_path, enable_mlflow=False):
        pass  # No assertions on global mlflow state; just must not raise.
