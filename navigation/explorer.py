#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : navigation/explorer.py
Purpose     : Exploration: how the robot learns a home it has never seen.
              PointDriver is a closed-loop "drive to (x, y)" controller for
              any differential-drive body behind the IOBridge contract.
              Explorer sweeps the reachable area point by point while the
              control loop's passive perception labels rooms into the
              SemanticMap — no pre-programmed routes, no fixed waypoints.

              Body-agnostic: needs only an IOBridge (motors) and a pose
              provider (SimWorld ground truth in simulation; odometry/SLAM
              on real hardware).
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-06-12
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import logging
import math
import time
from typing import Callable, List, Optional, Tuple

log = logging.getLogger(__name__)

_CONTROL_HZ = 20.0
_TURN_IN_PLACE_RAD = 0.4       # Heading error above which we rotate in place
_TURN_SPEED = 0.35             # Motor fraction while rotating in place
_CRUISE_SPEED = 0.7            # Motor fraction while driving
_STEER_GAIN = 1.0
_STEER_LIMIT = 0.3


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class PointDriver:
    """
    Drives a differential-drive body to a target point.

    Rotate-then-arc controller: turn in place until roughly facing the
    target, then drive forward with proportional steering.

    Attributes
    ----------
    tolerance_m : float
        Arrival radius around the target.
    """

    def __init__(
        self,
        bridge,
        pose_provider: Callable[[], Tuple[float, float, float]],
        tolerance_m: float = 0.3,
        safety_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        """
        Parameters
        ----------
        bridge : IOBridge
            Motor interface (set_motor_speeds / emergency_stop_motors).
        pose_provider : callable
            Returns the current (x, y, theta).
        tolerance_m : float
            Distance at which the target counts as reached.
        safety_check : callable, optional
            Returns True while motion is permitted; checked every cycle.
        """
        self._bridge = bridge
        self._pose = pose_provider
        self.tolerance_m = tolerance_m
        self._safety_check = safety_check

    def go_to(self, target_x: float, target_y: float, timeout_sec: float = 60.0) -> bool:
        """
        Drive to (target_x, target_y), blocking until arrival or failure.

        Parameters
        ----------
        target_x, target_y : float
            Target position in metres.
        timeout_sec : float
            Give up after this long.

        Returns
        -------
        bool
            True if the target was reached; False on timeout or safety stop.
        """
        deadline = time.monotonic() + timeout_sec
        interval = 1.0 / _CONTROL_HZ

        try:
            while time.monotonic() < deadline:
                if self._safety_check and not self._safety_check():
                    log.warning("PointDriver: safety check failed — stopping.")
                    return False

                x, y, theta = self._pose()
                dx, dy = target_x - x, target_y - y
                distance = math.hypot(dx, dy)
                if distance <= self.tolerance_m:
                    return True

                heading_error = _wrap_angle(math.atan2(dy, dx) - theta)
                if abs(heading_error) > _TURN_IN_PLACE_RAD:
                    spin = math.copysign(_TURN_SPEED, heading_error)
                    self._bridge.set_motor_speeds(-spin, spin)
                else:
                    steer = max(-_STEER_LIMIT, min(_STEER_LIMIT, _STEER_GAIN * heading_error))
                    self._bridge.set_motor_speeds(_CRUISE_SPEED - steer, _CRUISE_SPEED + steer)

                time.sleep(interval)

            log.warning("PointDriver: timed out short of (%.2f, %.2f).", target_x, target_y)
            return False
        finally:
            self._bridge.set_motor_speeds(0.0, 0.0)


class Explorer:
    """
    Sweeps the home so passive perception can label every room.

    Visits a serpentine grid of reachable points across the given bounds.
    Unreachable points (outside the floor plan, blocked) are skipped — the
    sweep is regenerated from current bounds on every run, so a rearranged
    or extended home is simply re-learned.
    """

    def __init__(
        self,
        driver: PointDriver,
        bounds_provider: Callable[[], Tuple[float, float, float, float]],
        point_filter: Optional[Callable[[float, float], bool]] = None,
        grid_step_m: float = 1.5,
    ) -> None:
        """
        Parameters
        ----------
        driver : PointDriver
            Motion primitive used to reach each sweep point.
        bounds_provider : callable
            Returns (min_x, min_y, max_x, max_y) of the area to explore —
            from SimWorld in simulation, the LiDAR map on real hardware.
        point_filter : callable, optional
            Returns True if (x, y) is worth visiting (e.g. inside the floor
            plan). None = visit every grid point.
        grid_step_m : float
            Spacing between sweep points.
        """
        self._driver = driver
        self._bounds = bounds_provider
        self._filter = point_filter
        self.grid_step_m = grid_step_m

    def sweep_points(self) -> List[Tuple[float, float]]:
        """Serpentine grid of candidate points over the current bounds."""
        min_x, min_y, max_x, max_y = self._bounds()
        step = self.grid_step_m
        points: List[Tuple[float, float]] = []

        ys = []
        y = min_y + step / 2.0
        while y < max_y:
            ys.append(y)
            y += step

        for row, y in enumerate(ys):
            xs = []
            x = min_x + step / 2.0
            while x < max_x:
                xs.append(x)
                x += step
            if row % 2:
                xs.reverse()
            for x in xs:
                if self._filter is None or self._filter(x, y):
                    points.append((round(x, 2), round(y, 2)))
        return points

    def explore(self, abort_check: Optional[Callable[[], bool]] = None) -> bool:
        """
        Run one full sweep of the home.

        Parameters
        ----------
        abort_check : callable, optional
            Returns True when the sweep should stop early.

        Returns
        -------
        bool
            True if the majority of sweep points were reached.
        """
        points = self.sweep_points()
        if not points:
            log.warning("Explorer: no sweep points — bounds empty?")
            return False

        log.info("Explorer: sweeping %d point(s) over the home.", len(points))
        reached = 0
        for index, (x, y) in enumerate(points):
            if abort_check and abort_check():
                log.warning("Explorer: aborted at point %d/%d.", index, len(points))
                return False
            if self._driver.go_to(x, y, timeout_sec=30.0):
                reached += 1
            else:
                log.info("Explorer: point (%.2f, %.2f) unreachable — skipped.", x, y)

        log.info("Explorer: sweep finished — reached %d/%d point(s).", reached, len(points))
        return reached >= len(points) / 2
