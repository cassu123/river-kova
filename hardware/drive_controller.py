#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : hardware/drive_controller.py
Purpose     : Differential drive controller. Translates high-level velocity
              commands (linear m/s, angular rad/s) into left/right motor
              speeds sent to the Pico bridge. Enforces speed limits and
              provides an emergency stop that bypasses all ramping.
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
from typing import Callable, Optional, Tuple

from core.constants import (
    MAX_ANGULAR_SPEED,
    MAX_LINEAR_SPEED,
    SAFE_ANGULAR_SPEED,
    SAFE_LINEAR_SPEED,
)
from hardware.pico_bridge import PicoBridge

log = logging.getLogger(__name__)

# Wheel base width in metres — should match the physical robot
_DEFAULT_WHEEL_BASE_M = 0.35


class DriveController:
    """
    High-level differential drive controller.

    Converts (linear_velocity, angular_velocity) commands into left/right
    motor speed fractions and forwards them to the Pico bridge.

    Attributes
    ----------
    is_stopped : bool
        True when the drive is in a stopped or e-stopped state.
    current_linear : float
        Current commanded linear velocity (m/s).
    current_angular : float
        Current commanded angular velocity (rad/s).
    """

    def __init__(
        self,
        pico_bridge: PicoBridge,
        max_speed: float = MAX_LINEAR_SPEED,
        drive_type: str = "differential",
        wheel_base_m: float = _DEFAULT_WHEEL_BASE_M,
        safety_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        """
        Initialise the drive controller.

        Parameters
        ----------
        pico_bridge : PicoBridge
            Connected Pico bridge instance.
        max_speed : float
            Maximum linear speed in m/s.
        drive_type : str
            Drive type identifier (currently only 'differential' supported).
        wheel_base_m : float
            Distance between left and right wheels in metres.
        safety_check : callable, optional
            Called before each move command. If it returns False, the command
            is rejected. Typically wired to the safety level check in main.
        """
        self._pico = pico_bridge
        self._max_speed = max_speed
        self._drive_type = drive_type
        self._wheel_base = wheel_base_m
        self._safety_check = safety_check

        self._linear: float = 0.0
        self._angular: float = 0.0
        self._estopped: bool = False
        self._lock = threading.Lock()

        log.info(
            "DriveController ready (%s, max_speed=%.2f m/s, wheel_base=%.2f m).",
            drive_type, max_speed, wheel_base_m,
        )

    # ── State properties ──────────────────────────────────────────────────────

    @property
    def is_stopped(self) -> bool:
        """True when the drive is stopped or e-stopped."""
        with self._lock:
            return self._estopped or (self._linear == 0.0 and self._angular == 0.0)

    @property
    def current_linear(self) -> float:
        """Current commanded linear velocity (m/s)."""
        with self._lock:
            return self._linear

    @property
    def current_angular(self) -> float:
        """Current commanded angular velocity (rad/s)."""
        with self._lock:
            return self._angular

    # ── Motion commands ───────────────────────────────────────────────────────

    def move(self, linear: float, angular: float) -> bool:
        """
        Command a velocity.

        Parameters
        ----------
        linear : float
            Desired linear velocity in m/s (positive = forward).
        angular : float
            Desired angular velocity in rad/s (positive = counter-clockwise).

        Returns
        -------
        bool
            True if the command was sent, False if blocked by safety or e-stop.
        """
        with self._lock:
            if self._estopped:
                log.warning("DriveController: move rejected — e-stop active.")
                return False

        if self._safety_check and not self._safety_check():
            log.warning("DriveController: move rejected — safety check failed.")
            return False

        # Clamp to limits
        linear = max(-self._max_speed, min(self._max_speed, linear))
        angular = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, angular))

        left, right = self._unicycle_to_differential(linear, angular)

        try:
            self._pico.set_motor_speeds(left, right)
            with self._lock:
                self._linear = linear
                self._angular = angular
            return True
        except Exception as exc:
            log.error("DriveController.move error: %s", exc)
            return False

    def move_safe(self, linear: float, angular: float) -> bool:
        """
        Command a velocity clamped to safe (near-human) speed limits.

        Parameters
        ----------
        linear : float
            Desired linear velocity (will be clamped to SAFE_LINEAR_SPEED).
        angular : float
            Desired angular velocity (will be clamped to SAFE_ANGULAR_SPEED).

        Returns
        -------
        bool
            True if the command was sent.
        """
        safe_linear = max(-SAFE_LINEAR_SPEED, min(SAFE_LINEAR_SPEED, linear))
        safe_angular = max(-SAFE_ANGULAR_SPEED, min(SAFE_ANGULAR_SPEED, angular))
        return self.move(safe_linear, safe_angular)

    def stop(self) -> None:
        """
        Gracefully stop the drive (zero velocity).

        Does not set the e-stop flag — the robot can resume after stop().
        """
        try:
            self._pico.set_motor_speeds(0.0, 0.0)
        except Exception as exc:
            log.error("DriveController.stop error: %s", exc)
        with self._lock:
            self._linear = 0.0
            self._angular = 0.0
        log.debug("DriveController: stopped.")

    def emergency_stop(self) -> None:
        """
        Immediate emergency stop — bypasses all ramping.

        Sets the e-stop flag. The robot will not move again until
        clear_estop() is called.
        """
        try:
            self._pico.emergency_stop_motors()
        except Exception as exc:
            log.error("DriveController.emergency_stop Pico error: %s", exc)
        with self._lock:
            self._estopped = True
            self._linear = 0.0
            self._angular = 0.0
        log.warning("DriveController: EMERGENCY STOP.")

    def clear_estop(self) -> None:
        """
        Clear the e-stop flag, allowing motion to resume.

        Should only be called after the safety condition has been resolved.
        """
        with self._lock:
            self._estopped = False
        log.info("DriveController: e-stop cleared.")

    # ── Kinematics ────────────────────────────────────────────────────────────

    def _unicycle_to_differential(
        self, linear: float, angular: float
    ) -> Tuple[float, float]:
        """
        Convert unicycle (linear, angular) to differential (left, right) speeds.

        Parameters
        ----------
        linear : float
            Linear velocity in m/s.
        angular : float
            Angular velocity in rad/s.

        Returns
        -------
        tuple of float
            (left_fraction, right_fraction) in range [-1.0, 1.0].
        """
        v_left = linear - (angular * self._wheel_base / 2.0)
        v_right = linear + (angular * self._wheel_base / 2.0)

        # Normalise to [-1, 1] relative to max speed
        if self._max_speed > 0:
            v_left /= self._max_speed
            v_right /= self._max_speed

        # Clamp
        v_left = max(-1.0, min(1.0, v_left))
        v_right = max(-1.0, min(1.0, v_right))

        return v_left, v_right
