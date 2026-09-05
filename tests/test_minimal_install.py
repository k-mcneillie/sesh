"""Smoke tests proving sesh's core works with no optional dependencies.

These run in every environment, but are the specific tests exercised by
the CI "test-minimal" job, which installs only the stdlib-only core
package (no numpy, torch, or mlflow) and runs this file to prove sesh's
central promise: it works with any type of project.
"""

from __future__ import annotations

from pathlib import Path

import sesh
from sesh import Session


def test_import_sesh_succeeds() -> None:
    assert hasattr(sesh, "Session")
    assert sesh.__version__


def test_session_basic_usage_works_without_optional_deps(tmp_path: Path) -> None:
    with Session(name="minimal", output_root=tmp_path, enable_mlflow=False) as session:
        session.info("hello")
        session.log_metrics({"loss": 0.1})
        session.log_params({"batch_size": 32})

    assert session.output_dir.exists()
    assert (session.output_dir / "run.log").exists()
