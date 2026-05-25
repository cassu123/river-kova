#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : telemetry/alerts.py
Purpose     : Alert manager. Routes safety and system alerts to the River Song
              API, local log, and optionally a local buzzer/LED via the Pico.
              Deduplicates repeated alerts to avoid notification spam.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

from connectivity.api_client import RiverSongAPIClient

log = logging.getLogger(__name__)

_DEDUP_WINDOW_SEC = 30.0  # Suppress duplicate alerts within this window
_ALERT_HISTORY_SIZE = 100


@dataclass
class Alert:
    """A single alert record."""

    level: str
    message: str
    timestamp: float = field(default_factory=time.time)
    sent: bool = False

    def to_dict(self) -> dict:
        """Serialise to a plain dict."""
        return {
            "level": self.level,
            "message": self.message,
            "timestamp": self.timestamp,
            "sent": self.sent,
        }


class AlertManager:
    """
    Alert routing and deduplication manager.

    Routes alerts to:
    1. The Python logger (always)
    2. The River Song API (if connected)

    Deduplicates alerts with the same level+message within a time window
    to prevent notification spam during sustained fault conditions.

    Attributes
    ----------
    robot_id : str
        Unit identifier included in all alerts.
    """

    def __init__(
        self,
        api_client: RiverSongAPIClient,
        robot_id: str,
        dedup_window_sec: float = _DEDUP_WINDOW_SEC,
    ) -> None:
        """
        Initialise the alert manager.

        Parameters
        ----------
        api_client : RiverSongAPIClient
            Used to push alerts to River Song.
        robot_id : str
            Unit identifier.
        dedup_window_sec : float
            Seconds within which duplicate alerts are suppressed.
        """
        self._api = api_client
        self.robot_id = robot_id
        self._dedup_window = dedup_window_sec
        self._recent: Dict[str, float] = {}  # key → last_sent_time
        self._history: Deque[Alert] = deque(maxlen=_ALERT_HISTORY_SIZE)

        log.info("AlertManager ready for unit '%s'.", robot_id)

    def send(self, level: str, message: str) -> bool:
        """
        Send an alert.

        Parameters
        ----------
        level : str
            Severity: 'INFO', 'WARN', 'ERROR', 'CRITICAL'.
        message : str
            Alert message text.

        Returns
        -------
        bool
            True if the alert was sent (not deduplicated).
        """
        dedup_key = f"{level}:{message}"
        now = time.time()

        # Deduplication check
        last_sent = self._recent.get(dedup_key, 0.0)
        if now - last_sent < self._dedup_window:
            log.debug("AlertManager: suppressing duplicate alert '%s'.", dedup_key)
            return False

        self._recent[dedup_key] = now

        alert = Alert(level=level, message=message)
        self._history.append(alert)

        # Log locally
        log_fn = {
            "INFO": log.info,
            "WARN": log.warning,
            "WARNING": log.warning,
            "ERROR": log.error,
            "CRITICAL": log.critical,
        }.get(level.upper(), log.info)
        log_fn("ALERT [%s] %s: %s", self.robot_id, level, message)

        # Push to River Song
        sent = self._api.push_alert(level=level, message=message)
        alert.sent = sent

        return True

    def get_history(self, limit: int = 20) -> list:
        """
        Return recent alert history.

        Parameters
        ----------
        limit : int
            Maximum number of alerts to return.

        Returns
        -------
        list of dict
        """
        alerts = list(self._history)
        return [a.to_dict() for a in alerts[-limit:]]

    def clear_dedup_cache(self) -> None:
        """Clear the deduplication cache (e.g. after a fault is resolved)."""
        self._recent.clear()
        log.debug("AlertManager: dedup cache cleared.")
