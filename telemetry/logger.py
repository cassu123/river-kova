#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : telemetry/logger.py
Purpose     : Structured file logger for the Kova unit. Writes JSON-formatted
              log records to rotating files in the configured log directory.
              Separate log files for system events, safety events, and tasks.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per log file
_BACKUP_COUNT = 5


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as JSON.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to format.

        Returns
        -------
        str
            JSON-encoded log line.
        """
        data = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


class KovaLogger:
    """
    Structured file logger for a Kova unit.

    Creates three rotating log files:
    - system.log   — general system events
    - safety.log   — safety-critical events (human detection, e-stop, faults)
    - tasks.log    — task lifecycle events

    Attributes
    ----------
    robot_id : str
        Unit identifier included in all log records.
    log_dir : Path
        Directory where log files are written.
    """

    def __init__(
        self,
        robot_id: str,
        log_dir: str = "/var/log/river-kova",
        log_level: str = "INFO",
    ) -> None:
        """
        Initialise the Kova logger.

        Parameters
        ----------
        robot_id : str
            Unit identifier.
        log_dir : str
            Directory for log files.
        log_level : str
            Minimum log level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        """
        self.robot_id = robot_id
        self.log_dir = Path(log_dir) / robot_id
        self._level = getattr(logging, log_level.upper(), logging.INFO)

        self._ensure_log_dir()
        self._system_logger = self._create_logger("system", "system.log")
        self._safety_logger = self._create_logger("safety", "safety.log", level=logging.WARNING)
        self._task_logger = self._create_logger("tasks", "tasks.log")

        log.info("KovaLogger ready (log_dir=%s).", self.log_dir)

    # ── Public logging methods ────────────────────────────────────────────────

    def system(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a system event.

        Parameters
        ----------
        level : str
            Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        message : str
            Log message.
        extra : dict, optional
            Additional structured fields.
        """
        self._log(self._system_logger, level, message, extra)

    def safety(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a safety event.

        Parameters
        ----------
        level : str
            Log level.
        message : str
            Safety event description.
        extra : dict, optional
            Additional structured fields.
        """
        self._log(self._safety_logger, level, message, extra)

    def task(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a task lifecycle event.

        Parameters
        ----------
        level : str
            Log level.
        message : str
            Task event description.
        extra : dict, optional
            Additional structured fields.
        """
        self._log(self._task_logger, level, message, extra)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _log(
        self,
        logger: logging.Logger,
        level: str,
        message: str,
        extra: Optional[Dict[str, Any]],
    ) -> None:
        """Dispatch a log record to the given logger."""
        log_fn = getattr(logger, level.lower(), logger.info)
        if extra:
            message = f"{message} | {json.dumps(extra)}"
        log_fn(message)

    def _ensure_log_dir(self) -> None:
        """Create the log directory if it does not exist."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            log.warning("KovaLogger: cannot create log dir %s — using /tmp.", self.log_dir)
            self.log_dir = Path("/tmp/river-kova") / self.robot_id
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def _create_logger(
        self,
        name: str,
        filename: str,
        level: int = logging.DEBUG,
    ) -> logging.Logger:
        """
        Create a named logger with a rotating file handler.

        Parameters
        ----------
        name : str
            Logger name.
        filename : str
            Log file name within log_dir.
        level : int
            Minimum log level for this logger.

        Returns
        -------
        logging.Logger
        """
        logger = logging.getLogger(f"kova.{self.robot_id}.{name}")
        logger.setLevel(max(level, self._level))
        logger.propagate = False

        handler = logging.handlers.RotatingFileHandler(
            filename=str(self.log_dir / filename),
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        return logger
