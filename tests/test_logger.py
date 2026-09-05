from __future__ import annotations

import logging
from pathlib import Path

from sesh.logger import configure_logger


def test_configure_logger_with_file_attaches_console_and_file_handlers(
    tmp_path: Path,
) -> None:
    log_file = tmp_path / "run.log"

    logger = configure_logger(name="with-file", log_file=log_file)

    assert len(logger.handlers) == 2
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert isinstance(logger.handlers[1], logging.FileHandler)
    assert log_file.parent.exists()


def test_configure_logger_without_file_attaches_only_console_handler() -> None:
    logger = configure_logger(name="console-only", log_file=None)

    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_configure_logger_deduplicates_handlers_on_repeat_calls(
    tmp_path: Path,
) -> None:
    log_file = tmp_path / "run.log"

    configure_logger(name="repeat-config", log_file=log_file)
    logger = configure_logger(name="repeat-config", log_file=log_file)

    # A second call for the same logger name must not accumulate handlers.
    assert len(logger.handlers) == 2


def test_configure_logger_writes_to_the_file(tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"
    logger = configure_logger(name="writes-to-file", log_file=log_file)

    logger.info("hello, file")

    assert "hello, file" in log_file.read_text(encoding="utf-8")
