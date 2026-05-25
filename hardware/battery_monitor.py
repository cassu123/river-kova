#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : hardware/battery_monitor.py
Purpose     : Continuous battery level monitoring via the Pico bridge ADC.
              Fires callbacks at LOW and CRITICAL thresholds. Tracks charge
              history for telemetry.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable, Deque, Optional, Tuple

from core.constants import BATTERY_CRITICAL, BATTERY_FULL, BATTERY_LOW, BATTERY_WARN
from hardware.pico_bridge import PicoBridge

log = logging.getLogger(__name__)

_POLL_INTERVAL_SEC = 10.0
_HISTORY_SIZE = 60  # Keep last 60 readings (~10 minutes at default poll rate)


class BatteryMonitor:
    """
    Continuous battery level monitor.

    Polls the Pico bridge at a fixed interval and fires callbacks when
    the battery crosses LOW or CRITICAL thresholds.

    Attributes
    ----------
    level : float
        Most recent battery percentage [0.0, 100.0].
    is_low : bool
        True when battery is below the LOW threshold.
    is_critical : bool
        True when battery is below the CRITICAL threshold.
    """

    def __init__(
        self,
        pico_bridge: PicoBridge,
        poll_interval_sec: float = _POLL_INTERVAL_SEC,
        on_low: Optional[Callable[[float], None]] = None,
        on_critical: Optional[Callable[[float], None]] = None,
        on_full: Optional[Callable[[float], None]] = None,
    ) -> None:
        """
        Initialise the battery monitor.

        Parameters
        ----------
        pico_bridge : PicoBridge
            Connected Pico bridge for ADC reads.
        poll_interval_sec : float
            Seconds between battery reads.
        on_low : callable, optional
            Called with battery % when level drops below BATTERY_LOW.
        on_critical : callable, optional
            Called with battery % when level drops below BATTERY_CRITICAL.
        on_full : callable, optional
            Called with battery % when level reaches BATTERY_FULL (charging done).
        """
        self._pico = pico_bridge
        self._poll_interval = poll_interval_sec
        self._on_low = on_low
        self._on_critical = on_critical
        self._on_full = on_full

        self._level: float = 100.0
        self._history: Deque[Tuple[float, float]] = deque(maxlen=_HISTORY_SIZE)
        self._low_fired: bool = False
        self._critical_fired: bool = False
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ── State properties ──────────────────────────────────────────────────────

    @property
    def level(self) -> float:
        """Most recent battery percentage."""
        with self._lock:
            return self._level

    @property
    def is_low(self) -> bool:
        """True when battery is below the LOW threshold."""
        return self.level <= BATTERY_LOW

    @property
    def is_critical(self) -> bool:
        """True when battery is below the CRITICAL threshold."""
        return self.level <= BATTERY_CRITICAL

    @property
    def history(self) -> list:
        """List of (timestamp, percent) tuples from recent readings."""
        with self._lock:
            return list(self._history)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="kova-battery-monitor",
            daemon=True,
        )
        self._thread.start()
        log.info("BatteryMonitor started (poll_interval=%.1fs).", self._poll_interval)

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        log.info("BatteryMonitor stopped.")

    def poll(self) -> float:
        """
        Perform a single battery read and update state.

        Returns
        -------
        float
            Current battery percentage.
        """
        try:
            level = self._pico.read_battery()
            level = max(0.0, min(100.0, level))
        except Exception as exc:
            log.error("BatteryMonitor poll error: %s", exc)
            return self._level

        with self._lock:
            self._level = level
            self._history.append((time.time(), level))

        self._check_thresholds(level)
        return level

    # ── Internal ──────────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """Background thread that polls the battery at the configured interval."""
        while self._running:
            self.poll()
            time.sleep(self._poll_interval)

    def _check_thresholds(self, level: float) -> None:
        """
        Fire threshold callbacks when the battery crosses key levels.

        Parameters
        ----------
        level : float
            Current battery percentage.
        """
        if level <= BATTERY_CRITICAL and not self._critical_fired:
            self._critical_fired = True
            self._low_fired = True
            log.error("Battery CRITICAL: %.1f%%", level)
            if self._on_critical:
                try:
                    self._on_critical(level)
                except Exception as exc:
                    log.error("BatteryMonitor on_critical callback error: %s", exc)

        elif level <= BATTERY_LOW and not self._low_fired:
            self._low_fired = True
            log.warning("Battery LOW: %.1f%%", level)
            if self._on_low:
                try:
                    self._on_low(level)
                except Exception as exc:
                    log.error("BatteryMonitor on_low callback error: %s", exc)

        elif level >= BATTERY_FULL:
            if self._on_full:
                try:
                    self._on_full(level)
                except Exception as exc:
                    log.error("BatteryMonitor on_full callback error: %s", exc)

        # Reset fired flags when battery recovers (e.g. after charging)
        if level > BATTERY_LOW + 5.0:
            self._low_fired = False
        if level > BATTERY_CRITICAL + 5.0:
            self._critical_fired = False
