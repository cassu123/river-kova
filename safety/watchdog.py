#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : safety/watchdog.py
Purpose     : Software watchdog timer. The main control loop must kick the
              watchdog every iteration. If the loop stalls or deadlocks, the
              watchdog fires its timeout callback (typically an e-stop).
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
from typing import Callable, Optional

from core.constants import WATCHDOG_TIMEOUT_SEC

log = logging.getLogger(__name__)


class Watchdog:
    """
    Software watchdog timer for the Kova control loop.

    The watchdog runs in a background daemon thread. The main loop must call
    kick() at least once per timeout_sec interval. If it does not, on_timeout
    is called — which should trigger an e-stop.

    Attributes
    ----------
    timeout_sec : float
        Maximum allowed interval between kicks before the watchdog fires.
    is_running : bool
        True while the watchdog thread is active.
    """

    def __init__(
        self,
        timeout_sec: float = WATCHDOG_TIMEOUT_SEC,
        on_timeout: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Initialise the watchdog.

        Parameters
        ----------
        timeout_sec : float
            Seconds between kicks before the watchdog fires.
        on_timeout : callable, optional
            Called when the watchdog expires. Should trigger an e-stop.
        """
        if timeout_sec <= 0:
            raise ValueError(f"Watchdog timeout must be positive, got {timeout_sec}")

        self.timeout_sec: float = timeout_sec
        self._on_timeout = on_timeout
        self._last_kick: float = time.monotonic()
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        log.debug("Watchdog initialised (timeout=%.1fs)", timeout_sec)

    # ── State ─────────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """True while the watchdog monitor thread is active."""
        return self._running

    @property
    def time_since_last_kick(self) -> float:
        """Seconds elapsed since the last kick."""
        return time.monotonic() - self._last_kick

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Start the watchdog monitor thread.

        Raises
        ------
        RuntimeError
            If the watchdog is already running.
        """
        if self._running:
            raise RuntimeError("Watchdog is already running.")

        self._running = True
        self._last_kick = time.monotonic()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="kova-watchdog",
            daemon=True,
        )
        self._thread.start()
        log.info("Watchdog started (timeout=%.1fs).", self.timeout_sec)

    def stop(self) -> None:
        """
        Stop the watchdog monitor thread gracefully.

        Safe to call if the watchdog is not running.
        """
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        log.info("Watchdog stopped.")

    def kick(self) -> None:
        """
        Reset the watchdog timer — proves the control loop is alive.

        Must be called at least once per timeout_sec from the main loop.
        """
        with self._lock:
            self._last_kick = time.monotonic()

    # ── Monitor loop ──────────────────────────────────────────────────────────

    def _monitor_loop(self) -> None:
        """
        Background thread that checks for timeout.

        Polls at 10× the timeout frequency for responsiveness.
        """
        poll_interval = max(0.1, self.timeout_sec / 10.0)

        while self._running:
            time.sleep(poll_interval)

            with self._lock:
                elapsed = time.monotonic() - self._last_kick

            if elapsed >= self.timeout_sec:
                log.critical(
                    "WATCHDOG TIMEOUT — no kick for %.1fs (limit=%.1fs).",
                    elapsed,
                    self.timeout_sec,
                )
                if self._on_timeout:
                    try:
                        self._on_timeout()
                    except Exception as exc:
                        log.error("Watchdog on_timeout callback error: %s", exc)
                # Reset timer to avoid repeated firing while system recovers
                with self._lock:
                    self._last_kick = time.monotonic()
