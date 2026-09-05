# Changelog

## 2.0.0

### Breaking changes

- **Core dependencies are now empty.** `numpy`, `torch`, and `mlflow` moved
  from hard runtime dependencies to optional extras (`sesh[numpy]`,
  `sesh[torch]`, `sesh[mlflow]`, or `sesh[all]`). Existing installs should
  switch to `pip install "sesh[all]"` to keep prior behaviour.
- `DatasetCard.save()` now returns `tuple[Path, Path]` (markdown, config)
  instead of `None`.
- Model/dataset card filenames are now derived from the card's `name`
  (slugified) instead of being fixed (`model_card.md`, `dataset_card.md`,
  `dataset_config.json`). Code that globbed for those exact fixed filenames
  will need to glob for the new pattern instead.

### Fixed

- Minting multiple model or dataset cards in one session no longer silently
  overwrites earlier ones (#13, #9).
- MLflow tracking now defaults to a SQLite-backed store instead of the
  maintenance-mode `file://` backend, which could raise `MlflowException`
  (#5).
- `pyproject.toml`'s pytest/coverage configuration pointed at a nonexistent
  package name (`research_session`); coverage reports were previously empty
  (#8).
- `.github/ISSUE_TEMPLATES/` renamed to the singular `ISSUE_TEMPLATE/` that
  GitHub actually recognises.

### Added

- `Session(..., no_log=True)` for quick, disk-free one-off sessions -
  console logging only, no output directory, no MLflow run (#10).
- `Session(..., mlflow_tracking_uri=...)` to point at a shared/remote MLflow
  server instead of the per-session SQLite default.
- `session.debug()`, `session.warning()`, and `session.error()`, alongside
  the existing `session.info()`, so the full logging interface is available
  without reaching into the private `_logger` attribute.
- CI: a `security-scan` job (Bandit + pip-audit) and a `test-minimal` job
  that verifies the core package works with none of the optional
  dependencies installed (#3).
- `py.typed` marker (PEP 561) and a real public surface for the `cards`
  subpackage (`sesh.cards.DatasetCard`, `sesh.cards.ModelCard`) (#1).

### Removed

- Unused template scaffolding: `docs/`, `references/`, `results/`,
  `notebooks/` (#4).

### Not implemented

- #11 and #12 (adopting `accelerate`) were closed rather than implemented:
  there is no distributed/multi-GPU logic in `sesh` to refactor, and
  `accelerate` is itself a torch-only dependency that would have worked
  against making the core dependency-free. Callers who use `accelerate` in
  their own training loop remain free to do so independently of `Session`.

## 1.0.0

Initial release: `Session` orchestrator, reproducibility seeding, console +
file logging, MLflow tracking, and model/dataset card minting.
