from __future__ import annotations

import logging
from pathlib import Path

# Default global configurations
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logger(
    name: str,
    log_file: Path,
    *,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> logging.Logger:
    """Configure a logger with independent console and file handlers.

    Args:
        name: Name of the logger.
        log_file: Path to the destination log file.
        console_level: Logging level for the standard output stream.
        file_level: Logging level for the persistent log file.
        log_format: Structured pattern for log messages.
        date_format: Structured pattern for timestamps.

    Returns:
        A configured logging.Logger instance.

    Raises:
        OSError: If the parent directory of log_file cannot be created.
    """
    # Ensure directory shell exists locally
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    
    # Establish the lowest baseline level required to capture all handlers
    logger.setLevel(min(console_level, file_level))
    logger.propagate = False

    # Prevent handler duplication in persistent sessions or notebooks
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # Stream Handler: Keeps live console outputs punchy and clean
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler: Captures granular parameters or debug traces
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    # Define a local execution path for demonstration
    example_log_path = Path("results/logs/pipeline.log")
    
    # Initialize the project logger
    # Terminal will only show INFO and above, file will capture DEBUG traces
    log = configure_logger(
        name="research_pipeline",
        log_file=example_log_path,
        console_level=logging.INFO,
        file_level=logging.DEBUG,
    )
    
    print(f"--- Demonstration Started (Writing logs to {example_log_path}) ---")
    
    # 1. This will appear BOTH in the terminal and in the log file
    log.info("Initializing scientific execution pipeline...")
    
    # 2. This will ONLY appear in the file (terminal filters it out)
    log.debug("Hyperparameters set: learning_rate=0.001, batch_size=32, seed=42")
    
    try:
        # Simulate a processing step
        log.info("Loading dataset matrix...")
        log.debug("Memory allocation for matrix: 450MB")
        
        # Simulate a handled anomaly/warning
        log.warning("Matrix dimensions are non-square. Forcing fallback solver.")
        
        # Simulate an exception block
        raise ValueError("Convergence criteria failed after 1000 iterations.")
        
    except Exception as error:
        # Captures the full error traceback cleanly inside the logs
        log.error(f"Pipeline execution halted: {error}", exc_info=True)

    print("--- Demonstration Finished ---")
