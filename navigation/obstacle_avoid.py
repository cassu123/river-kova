#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : navigation/obstacle_avoid.py
Purpose     : Reactive obstacle avoidance layer. Sits between the path planner
              and the drive controller. Monitors LiDAR sectors and modifies
              velocity commands to steer around obstacles in real time.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

from core.constants import OBSTACLE_CLEARANCE_MIN, SAFE_LINEAR_SPEED
from hardware.drive_controller import DriveController

log = logging.getLogger(__name__)

# Sector names and their angular centres (radians, 0 = forward)
_SECTORS = {
    "front":       0.0,
    "front_left":  math.radians(45),
    "front_right": math.radians(-45),
    "left":        math.radians(90),
    "right":       math.radians(-90),
    "rear":        math.radians(180),
}


class ObstacleAvoidance:
    """
    Reactive obstacle avoidance using the Vector Field Histogram (VFH) concept.

    Reads LiDAR sector distances and modifies the commanded velocity to steer
    around obstacles. If the front sector is blocked, the robot turns toward
    the clearer side.

    Attributes
    ----------
    is_blocked : bool
        True when the forward path is obstructed.
    """

    def __init__(
        self,
        drive: DriveController,
        clearance_min: float = OBSTACLE_CLEARANCE_MIN,
    ) -> None:
        """
        Initialise the obstacle avoidance module.

        Parameters
        ----------
        drive : DriveController
            Drive controller to send modified velocity commands.
        clearance_min : float
            Minimum obstacle clearance in metres.
        """
        self._drive = drive
        self._clearance_min = clearance_min
        self._scan: Dict[str, float] = {}

        log.info("ObstacleAvoidance ready (clearance_min=%.2fm).", clearance_min)

    @property
    def is_blocked(self) -> bool:
        """True when the forward path is obstructed."""
        front = self._scan.get("front", -1.0)
        return 0.0 < front < self._clearance_min

    # ── Scan update ───────────────────────────────────────────────────────────

    def update_scan(self, scan: Dict[str, float]) -> None:
        """
        Update the internal LiDAR sector scan.

        Parameters
        ----------
        scan : dict
            Sector name → distance in metres.
        """
        self._scan = dict(scan)

    # ── Velocity modification ─────────────────────────────────────────────────

    def apply(self, linear: float, angular: float) -> Tuple[float, float]:
        """
        Modify a velocity command to avoid obstacles.

        Parameters
        ----------
        linear : float
            Desired linear velocity (m/s).
        angular : float
            Desired angular velocity (rad/s).

        Returns
        -------
        tuple of float
            (modified_linear, modified_angular) safe velocity command.
        """
        if not self._scan:
            return linear, angular

        front = self._scan.get("front", -1.0)
        front_left = self._scan.get("front_left", -1.0)
        front_right = self._scan.get("front_right", -1.0)

        # If front is clear, pass through unchanged
        if front < 0 or front >= self._clearance_min * 2:
            return linear, angular

        # Slow down as we approach an obstacle
        if front < self._clearance_min * 4:
            scale = max(0.1, (front - self._clearance_min) / (self._clearance_min * 3))
            linear = linear * scale

        # If very close, stop forward motion
        if front < self._clearance_min:
            linear = min(0.0, linear)  # Allow reverse

        # Steer away from the obstacle
        if linear > 0:
            left_clear = front_left < 0 or front_left > self._clearance_min
            right_clear = front_right < 0 or front_right > self._clearance_min

            if left_clear and not right_clear:
                angular = max(angular, 0.3)   # Turn left
            elif right_clear and not left_clear:
                angular = min(angular, -0.3)  # Turn right
            elif not left_clear and not right_clear:
                linear = 0.0  # Both sides blocked — stop

        return linear, angular
