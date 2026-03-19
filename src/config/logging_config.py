"""
Centralised logging for demand-forecast-ops"""

import sys
from pathlib import Path

from loguru import logger


def configure_logging(log_level: str = "INFO", log_to_file: bool = True) -> None:
    """
    Configure application-wide logging
    """
    logger.remove()

    logger.add(
        sys.stdout,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler – plain text, persists across terminal sessions
    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logger.add(
            log_dir / "app.log",
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
            rotation="10 MB",  # creates a new file after 10MB
            retention="30 days",  # deletes files older than 30 days
            compression="zip",  # compresses rotated files to save space
        )


def get_logger(name: str) -> logger:
    return logger.bind(name=name)
