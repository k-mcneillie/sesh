# Default recipe to list all available shortcuts
default:
    @just --list

# Run all code quality gates (Lint, Format, Types, Tests)
check-all: lint format-check type-check test

# Run the pytest suite with code coverage tracking
test:
    pytest

# Run ruff linter and automatically fix safe code violations
lint:
    ruff check . --fix

# Check formatting rules without mutating files
format-check:
    ruff format --check .

# Automatically format all source files using ruff
format:
    ruff format .

# Run static type checking across the tracking library package source
type-check:
    mypy src/

# Run static (Bandit) and dependency (pip-audit) security scans
security:
    bandit -r src -ll
    pip-audit

# Clean up temporary build artifacts, packaging caches, and test artifacts
clean:
    rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build src/*.egg-info

# Build the package distribution archives (wheel and source distribution)
build: clean
    python3 -m pip install --upgrade build
    python3 -m build

# Install the package locally in editable mode for active development testing
dev-install:
    python3 -m pip install -e .[dev]
