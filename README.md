# sesh
`sesh` is a session manager and assurance framework that includes capabilities such as logging, `mlflow` experiment tracking and model and dataset card minting.

I built this to avoid duplication of code with my repositories and to centralise the functional tools I consistently use. This enables me to work according to my preferences and maintain a SSOT for the tool.

## Project and Data Structure

The directory is structured to isolate source code from large data artifacts:

* `src/sesh`: Entry point script is `session.py` with `seed.py` and `logger.py` called within the `Session` class.
* `tests/`: Under Construction...

## Quick Start and Installation

### 1. Environment Setup

For local:
```bash
pip install -e /.../...[dev]
```

Cloning GitHub repo that uses `sesh`, all templated projects contain the project with their `pyproject.toml`.
