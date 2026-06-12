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
        last_check = time.monotonic()
        last_x, last_y, _ = self._pose()

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

                # Stuck against something? Back up briefly and re-approach —
                # the bump-and-retreat every robot vacuum does.
                now = time.monotonic()
                if now - last_check >= 1.2:
                    if math.hypot(x - last_x, y - last_y) < 0.05:
                        log.debug("PointDriver: stalled at (%.2f, %.2f) — backing up.", x, y)
                        self._bridge.set_motor_speeds(-_TURN_SPEED, -_TURN_SPEED)
                        time.sleep(0.4)
                        self._bridge.set_motor_speeds(0.0, 0.0)
                    last_check = now
                    last_x, last_y = x, y

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


class Navigator:
    """
    Map-aware motion: plans on the learned occupancy grid, drives the legs.

    The route comes from what the robot has mapped itself (A* through
    discovered doorways) — never from a preloaded plan. When a leg fails
    (new obstacle, drift), it replans on the freshly updated map.
    """

    def __init__(
        self,
        occupancy_map,
        driver: PointDriver,
        pose_provider: Callable[[], Tuple[float, float, float]],
        waypoint_spacing_m: float = 0.4,
    ) -> None:
        """
        Parameters
        ----------
        occupancy_map : OccupancyMap
            The learned structural map to plan on.
        driver : PointDriver
            Low-level drive-to-point primitive.
        pose_provider : callable
            Returns the current (x, y, theta).
        waypoint_spacing_m : float
            Path decimation — denser keeps the body closer to the planned
            line through narrow doorways.
        """
        self._map = occupancy_map
        self._driver = driver
        self._pose = pose_provider
        self._spacing = waypoint_spacing_m

    def _decimate(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Thin a cell-by-cell path to spaced waypoints, keeping the goal."""
        if not path:
            return []
        waypoints = [path[0]]
        for point in path[1:-1]:
            last = waypoints[-1]
            if math.hypot(point[0] - last[0], point[1] - last[1]) >= self._spacing:
                waypoints.append(point)
        waypoints.append(path[-1])
        return waypoints

    def go_to(self, target_x: float, target_y: float, timeout_sec: float = 90.0) -> bool:
        """
        Route to a target across the learned map.

        Returns
        -------
        bool
            True on arrival; False when no mapped route exists or driving
            fails repeatedly.
        """
        deadline = time.monotonic() + timeout_sec

        for attempt in range(3):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            x, y, _ = self._pose()
            if math.hypot(target_x - x, target_y - y) <= self._driver.tolerance_m:
                return True

            path = self._map.plan_path(x, y, target_x, target_y)
            if path is None:
                log.info("Navigator: no mapped route to (%.2f, %.2f).", target_x, target_y)
                return False

            failed_leg = False
            for wx, wy in self._decimate(path):
                leg_budget = min(20.0, deadline - time.monotonic())
                if leg_budget <= 0 or not self._driver.go_to(wx, wy, timeout_sec=leg_budget):
                    log.info("Navigator: leg to (%.2f, %.2f) failed — replanning (attempt %d).",
                             wx, wy, attempt + 1)
                    failed_leg = True
                    break
            if not failed_leg:
                x, y, _ = self._pose()
                return math.hypot(target_x - x, target_y - y) <= self._driver.tolerance_m * 2

        return False


class FrontierExplorer:
    """
    Vacuum-style discovery: repeatedly drive toward the nearest frontier
    (the edge between mapped floor and unknown space) until none remain.

    No route is pre-programmed — the next target always comes from the
    current state of the robot's own map, so any home, rearranged any way,
    gets explored the same way.
    """

    def __init__(
        self,
        navigator: Navigator,
        occupancy_map,
        pose_provider: Callable[[], Tuple[float, float, float]],
        scan_integrator: Optional[Callable[[], None]] = None,
        min_target_distance_m: float = 0.4,
    ) -> None:
        """
        Parameters
        ----------
        navigator : Navigator
            Map-aware motion used to reach each frontier.
        occupancy_map : OccupancyMap
            The structural map being built.
        pose_provider : callable
            Returns the current (x, y, theta).
        scan_integrator : callable, optional
            Folds one fresh LiDAR scan into the map; called between
            targets (the control loop also integrates passively).
        min_target_distance_m : float
            Ignore frontiers closer than this — they will be absorbed by
            the next scan anyway.
        """
        self._navigator = navigator
        self._map = occupancy_map
        self._pose = pose_provider
        self._integrate = scan_integrator
        self._min_distance = min_target_distance_m

    def explore(
        self,
        abort_check: Optional[Callable[[], bool]] = None,
        max_targets: int = 60,
    ) -> bool:
        """
        Explore until the mapped area has no reachable frontiers left.

        Returns
        -------
        bool
            True when the frontier list is exhausted (home fully mapped as
            far as the robot can reach); False on abort or stall.
        """
        blacklist: List[Tuple[float, float]] = []
        reached_any = False

        for visit in range(max_targets):
            if abort_check and abort_check():
                log.warning("FrontierExplorer: aborted after %d target(s).", visit)
                return False
            if self._integrate:
                self._integrate()

            x, y, _ = self._pose()
            candidates = [
                (math.hypot(fx - x, fy - y), fx, fy)
                for fx, fy in self._map.frontiers(min_clearance_m=0.25)
                if all(math.hypot(fx - bx, fy - by) > 0.4 for bx, by in blacklist)
            ]
            candidates = [c for c in candidates if c[0] >= self._min_distance]
            if not candidates:
                log.info("FrontierExplorer: no frontiers left after %d target(s) — "
                         "%.1f m² mapped.", visit, self._map.explored_area_m2())
                return True

            _, fx, fy = min(candidates)
            if self._navigator.go_to(fx, fy, timeout_sec=60.0):
                reached_any = True
                time.sleep(0.6)    # Let passive perception fold in the new view
            else:
                # Write off the whole pocket, not just this cell
                blacklist.append((fx, fy))

        log.warning("FrontierExplorer: target budget exhausted (%.1f m² mapped).",
                    self._map.explored_area_m2())
        return reached_any


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
