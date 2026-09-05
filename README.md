# sesh

[![CI](https://github.com/k-mcneillie/sesh/actions/workflows/ci.yml/badge.svg)](https://github.com/k-mcneillie/sesh/actions/workflows/ci.yml)

`sesh` is a lightweight session manager for reproducible experiments: directory
routing, reproducibility seeding, structured logging, optional MLflow
tracking, and model/dataset "card" documentation, all behind a single
`Session` context manager.

`sesh`'s core has **no third-party dependencies** - it works in any Python
3.12+ project. NumPy, PyTorch, and MLflow are optional extras that unlock
extra reproducibility/tracking features when you have them installed, but
none of them are required just to use `sesh`.

## Installation

```bash
# Core only - stdlib-only, works in any project
pip install sesh

# With one optional feature
pip install "sesh[torch]"    # seeds torch's RNG + cuDNN determinism flags
pip install "sesh[numpy]"    # seeds numpy's RNG
pip install "sesh[mlflow]"   # enables MLflow experiment tracking

# Everything
pip install "sesh[all]"

# Contributing to sesh itself (all extras + pytest/ruff/mypy/bandit/pip-audit)
pip install -e ".[dev]"
```

## Quick start

```python
from sesh import Session

with Session(name="my_experiment", seed=42, device="cpu") as session:
    session.log_params({"learning_rate": 1e-3, "batch_size": 32})

    for epoch in range(3):
        session.log_metrics({"loss": 0.5 / (epoch + 1)}, step=epoch)

    session.log_dataset_card(
        name="TrainingSet",
        parameters={"rows": 10_000},
        description="Synthetic training data.",
    )
    session.log_model_card(
        name="BaselineNet",
        architecture="MLP",
        parameters={"hidden_units": 128},
        intended_use=["Baseline comparisons"],
        limitations=["Not validated on production data"],
    )

print(f"Outputs written to: {session.output_dir}")
```

Each `Session` gets its own timestamped output directory
(`outputs/<timestamp>_<name>/`) containing `run.log`, any model/dataset cards
you mint, and (if MLflow is installed and enabled) a self-contained SQLite
MLflow tracking store scoped to that directory.

### One-off sessions

For a quick, disk-free session - console logging only, nothing written to
disk, no MLflow run - pass `no_log=True`:

```python
with Session(name="quick_check", no_log=True) as session:
    session.info("just checking something, no output directory needed")
```

`session.path()`, `log_dataset_card()`, and `log_model_card()` all raise a
clear `RuntimeError` in this mode, since there's no output directory to write
into.

## API overview

- `Session(name, *, seed=42, device="cpu", output_root=Path("outputs"), deterministic_seed=True, enable_mlflow=True, mlflow_tracking_uri=None, no_log=False)`
- `session.path(*parts)` - resolve (and create) a path inside the session's output directory
- `session.info(msg)` / `session.debug(msg)` / `session.warning(msg)` / `session.error(msg)` - the full logging interface, all writing to the same `run.log`
- `session.log_metrics(metrics, step=None)` / `session.log_params(params)`
- `session.log_dataset_card(...)` / `session.log_model_card(...)` - mint Markdown (+ JSON, for datasets) documentation cards; minting the same or a different name multiple times in one session never overwrites a previous card
- Used as a context manager: MLflow runs and log handlers are closed cleanly on exit, including when the `with` block raises

## Development

```bash
just check-all   # lint + format-check + type-check + test
just test        # pytest with coverage
just security    # bandit + pip-audit
```

CI runs the same checks, plus a `test-minimal` job that installs only the
core (no numpy/torch/mlflow) to verify sesh genuinely works without them.
