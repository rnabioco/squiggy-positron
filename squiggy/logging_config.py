"""
Centralized logging configuration for Squiggy

This module provides a centralized logging configuration that can be used
across all modules. The logging level can be controlled via the SQUIGGY_LOG_LEVEL
environment variable.

Environment Variables:
    SQUIGGY_LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                       Default: WARNING

Examples:
    >>> from squiggy.logging_config import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing read...")
    >>> logger.warning("No alignment found")
    >>> logger.error("File not found")

    # Set logging level via environment variable before importing:
    >>> import os
    >>> os.environ['SQUIGGY_LOG_LEVEL'] = 'INFO'
    >>> from squiggy.logging_config import get_logger
    >>> logger = get_logger(__name__)
"""

import logging
import os


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the specified name

    Creates a logger with a consistent format and configurable logging level.
    The logging level is controlled by the SQUIGGY_LOG_LEVEL environment variable.

    Args:
        name: Logger name (typically __name__ from the calling module)

    Returns:
        Configured logger instance

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing data...")
        >>> logger.warning("Performance may be slow")
        >>> logger.error("File not found: %s", file_path)
    """
    logger = logging.getLogger(name)

    # Only configure if logger hasn't been configured yet
    # This prevents duplicate handlers when get_logger is called multiple times
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler()

        # Create formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Set level from environment variable (default: WARNING)
        level_name = os.getenv("SQUIGGY_LOG_LEVEL", "WARNING").upper()

        # Validate and set logging level
        try:
            level = getattr(logging, level_name)
            logger.setLevel(level)
        except AttributeError:
            # If invalid level specified, default to WARNING
            logger.setLevel(logging.WARNING)
            logger.warning(
                f"Invalid SQUIGGY_LOG_LEVEL '{level_name}'. Using WARNING instead. "
                f"Valid levels: DEBUG, INFO, WARNING, ERROR, CRITICAL"
            )

    return logger
