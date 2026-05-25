#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : safety/estop.py
Purpose     : Hardware and software emergency-stop controller.
              Provides arm/trigger/reset lifecycle with mandatory recovery delay.
              The EStop is the last line of defence — it must always work.
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

from core.constants import ESTOP_RECOVERY_DELAY_SEC

log = logging.getLogger(__name__)


class EStop:
    """
    Emergency-stop controller for a Kova unit.

    The EStop can be triggered by software (safety callbacks, watchdog) or
    by a physical hardware line via the Pico bridge. Once triggered, the unit
    cannot resume motion until reset() is called after the mandatory recovery
    delay has elapsed.

    Attributes
    ----------
    enabled : bool
        Whether the e-stop system is active. Should always be True in production.
    is_triggered : bool
        True when the e-stop is currently active.
    is_armed : bool
        True when the e-stop is armed and ready to fire.
    """

    def __init__(
        self,
        enabled: bool = True,
        recovery_delay_sec: float = ESTOP_RECOVERY_DELAY_SEC,
        on_trigger: Optional[Callable[[], None]] = None,
        on_reset: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Initialise the EStop controller.

        Parameters
        ----------
        enabled : bool
            Activate the e-stop system. Defaults to True.
        recovery_delay_sec : float
            Mandatory pause (seconds) after trigger before reset is allowed.
        on_trigger : callable, optional
            Callback invoked immediately when the e-stop fires.
        on_reset : callable, optional
            Callback invoked when the e-stop is successfully reset.
        """
        self.enabled: bool = enabled
        self._recovery_delay: float = recovery_delay_sec
        self._on_trigger = on_trigger
        self._on_reset = on_reset

        self._armed: bool = False
        self._triggered: bool = False
        self._trigger_time: Optional[float] = None
        self._lock = threading.Lock()

        log.debug("EStop initialised (enabled=%s, recovery_delay=%.1fs)", enabled, recovery_delay_sec)

    # ── State properties ──────────────────────────────────────────────────────

    @property
    def is_armed(self) -> bool:
        """True when the e-stop is armed and ready to fire."""
        return self._armed

    @property
    def is_triggered(self) -> bool:
        """True when the e-stop is currently active."""
        return self._triggered

    @property
    def recovery_elapsed_sec(self) -> float:
        """Seconds elapsed since the e-stop was triggered (0 if not triggered)."""
        if self._trigger_time is None:
            return 0.0
        return time.monotonic() - self._trigger_time

    @property
    def can_reset(self) -> bool:
        """True when the mandatory recovery delay has elapsed."""
        return self._triggered and self.recovery_elapsed_sec >= self._recovery_delay

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def arm(self) -> None:
        """
        Arm the e-stop, making it ready to fire.

        Raises
        ------
        RuntimeError
            If the e-stop is disabled.
        """
        if not self.enabled:
            log.warning("EStop.arm() called but e-stop is disabled.")
            return
        with self._lock:
            self._armed = True
        log.info("EStop armed.")

    def trigger(self, reason: str = "software trigger") -> None:
        """
        Fire the e-stop immediately.

        Idempotent — calling while already triggered is a no-op.

        Parameters
        ----------
        reason : str
            Human-readable reason for the trigger, written to the log.
        """
        if not self.enabled:
            return
        with self._lock:
            if self._triggered:
                return  # Already triggered
            self._triggered = True
            self._trigger_time = time.monotonic()

        log.critical("E-STOP TRIGGERED — reason: %s", reason)

        if self._on_trigger:
            try:
                self._on_trigger()
            except Exception as exc:
                log.error("EStop on_trigger callback error: %s", exc)

    def reset(self, force: bool = False) -> bool:
        """
        Attempt to reset the e-stop after the recovery delay.

        Parameters
        ----------
        force : bool
            Skip the recovery delay check. Use only in testing.

        Returns
        -------
        bool
            True if reset was successful, False if recovery delay has not elapsed.
        """
        with self._lock:
            if not self._triggered:
                log.debug("EStop.reset() called but e-stop is not triggered.")
                return True

            if not force and not self.can_reset:
                remaining = self._recovery_delay - self.recovery_elapsed_sec
                log.warning(
                    "EStop reset denied — recovery delay not elapsed (%.1fs remaining).",
                    remaining,
                )
                return False

            self._triggered = False
            self._trigger_time = None

        log.info("EStop reset — unit may resume operation.")

        if self._on_reset:
            try:
                self._on_reset()
            except Exception as exc:
                log.error("EStop on_reset callback error: %s", exc)

        return True

    def disarm(self) -> None:
        """
        Disarm the e-stop. Should only be used during controlled shutdown.
        """
        with self._lock:
            self._armed = False
        log.info("EStop disarmed.")
