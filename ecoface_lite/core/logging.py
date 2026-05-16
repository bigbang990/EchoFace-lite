"""Centralized logging setup.

Why a small wrapper:
- One place to change format, level, and file rotation later.
- Uvicorn and app logs can share the same root logger policy.
"""

import logging
import sys
from typing import Any

from ecoface_lite.core.config import get_settings


def setup_logging() -> None:
    settings = get_settings()
    level = logging.DEBUG if settings.debug else logging.INFO

    log_dir = settings.resolved_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(level)
    stream.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    file_handler = logging.FileHandler(log_dir / "ecoface_lite.log", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root.addHandler(stream)
    root.addHandler(file_handler)

    logging.getLogger("uvicorn").setLevel(level)
    logging.getLogger("uvicorn.error").setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_exception(logger: logging.Logger, msg: str, exc_info: Any = True) -> None:
    logger.exception(msg, exc_info=exc_info)
