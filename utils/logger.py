"""
utils/logger.py — Centralized Logger Factory
=============================================
Module: RabitScal — utils
DRY: Gộp _build_logger() từ ml_model.py, backtest_env.py, main.py

Usage:
    from utils.logger import build_logger
    logger = build_logger("MyModule", log_file="logs/mymodule.log")
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGS_DIR = Path("logs")
DEFAULT_MAX_BYTES    = 10 * 1024 * 1024   # 10 MB
DEFAULT_BACKUP_COUNT = 5
LOG_FORMAT = "[%(asctime)s UTC] - [%(levelname)-8s] - [%(name)s] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def build_logger(
    name:         str,
    log_file:     str | None = None,
    max_bytes:    int        = DEFAULT_MAX_BYTES,
    backup_count: int        = DEFAULT_BACKUP_COUNT,
    level:        int        = logging.DEBUG,
) -> logging.Logger:
    """
    Tạo logger với RotatingFileHandler + StreamHandler (stdout).

    Args:
        name:         Tên logger (module name).
        log_file:     Tên file log trong thư mục logs/. Mặc định: f"logs/{name.lower()}.log"
        max_bytes:    Kích thước tối đa file log trước khi rotate (bytes).
        backup_count: Số file backup giữ lại.
        level:        Log level (mặc định DEBUG — ghi tất cả, filter ở handler).

    Returns:
        logging.Logger instance đã cấu hình.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        # Tránh duplicate handlers khi module bị import nhiều lần
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # File handler — rotating
    log_path = LOGS_DIR / (log_file or f"{name.lower()}.log")
    fh = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)

    # Console handler — stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
