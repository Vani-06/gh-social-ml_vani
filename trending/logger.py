"""Centralized logging setup for the trending ingestion engine.

This module provides a consistent logging configuration across all trending
modules, with support for both file and console output.
"""

import logging
import sys
from typing import Optional

from .config import LOG_LEVEL, LOG_FILE, LOG_FORMAT


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """Set up and configure a logger for trending modules.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Defaults to LOG_LEVEL from config.
        log_file: Path to log file. If None, logs to stdout.
            Defaults to LOG_FILE from config.
        log_format: Log format string.
            Defaults to LOG_FORMAT from config.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Clear existing handlers to allow new configuration to be applied
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    # Set log level
    log_level = level or LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Create formatter
    formatter = logging.Formatter(log_format or LOG_FORMAT)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logger.level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if specified
    file_path = log_file or LOG_FILE
    if file_path:
        try:
            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(logger.level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (IOError, OSError) as exc:
            logger.warning(f"Failed to create log file at {file_path}: {exc}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger or create a new one with default configuration.

    This is a convenience function for modules that don't need custom
    logging configuration.

    Args:
        name: Logger name (typically __name__ of the calling module).

    Returns:
        Logger instance with default configuration.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
