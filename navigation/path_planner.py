#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : navigation/path_planner.py
Purpose     : A* path planner over the occupancy grid produced by RoomMapper.
              Provides waypoint navigation, coverage patterns, and last-
              detection navigation for the task executor.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import heapq
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from core.constants import BASE_STATION_TOLERANCE, PATH_REPLAN_INTERVAL
from navigation.room_mapper import RoomMapper

log = logging.getLogger(__name__)

# Named waypoints — populated from the room map or set manually
_DEFAULT_WAYPOINTS: Dict[str, Tuple[float, float]] = {
    "base": (0.0, 0.0),
    "kitchen": (3.0, 2.0),
    "living_room": (5.0, 1.0),
    "bedroom": (7.0, 3.0),
    "bathroom": (2.0, 5.0),
    "delivery_point": (1.0, 1.0),
    "trash_bin": (4.0, 0.5),
    "trash_collection": (0.5, 0.5),
    "dishwasher": (3.5, 2.5),
    "washer": (2.0, 4.0),
    "dryer": (2.5, 4.0),
    "surface_target": (3.0, 3.0),
    "water_station": (3.2, 2.8),
    "dog_bowl": (1.5, 0.5),
    "dog_food_storage": (1.8, 3.5),
}


class PathPlanner:
    """
    A* path planner over the room occupancy grid.

    Provides:
    - navigate_to_waypoint(name) — move to a named location
    - execute_coverage(pattern)  — boustrophedon / spiral coverage
    - navigate_to_last_detection() — approach the last detected object
    - scan_room()                — rotate in place to scan

    Attributes
    ----------
    waypoints : dict
        Named waypoint registry (name → (x, y)).
    last_detection_pose : tuple or None
        (x, y) of the most recently detected object.
    """

    def __init__(
        self,
        room_mapper: RoomMapper,
        replan_interval: float = PATH_REPLAN_INTERVAL,
    ) -> None:
        """
        Initialise the path planner.

        Parameters
        ----------
        room_mapper : RoomMapper
            Provides the occupancy grid for collision-free planning.
        replan_interval : float
            Seconds between path replanning cycles.
        """
        self._mapper = room_mapper
        self._replan_interval = replan_interval
        self.waypoints: Dict[str, Tuple[float, float]] = dict(_DEFAULT_WAYPOINTS)
        self.last_detection_pose: Optional[Tuple[float, float]] = None
        self._current_pose: Tuple[float, float] = (0.0, 0.0)

        log.info("PathPlanner ready (%d default waypoints).", len(self.waypoints))

    # ── Waypoint navigation ───────────────────────────────────────────────────

    def navigate_to_waypoint(self, waypoint_name: str) -> bool:
        """
        Navigate to a named waypoint.

        Parameters
        ----------
        waypoint_name : str
            Name of the target waypoint.

        Returns
        -------
        bool
            True if the robot reached the waypoint within tolerance.
        """
        target = self.waypoints.get(waypoint_name)
        if target is None:
            log.error("PathPlanner: unknown waypoint '%s'.", waypoint_name)
            return False

        log.info("PathPlanner: navigating to waypoint '%s' %s.", waypoint_name, target)
        return self._navigate_to_pose(target)

    def navigate_to_pose(self, x: float, y: float) -> bool:
        """
        Navigate to an arbitrary (x, y) pose.

        Parameters
        ----------
        x : float
            Target x coordinate in metres.
        y : float
            Target y coordinate in metres.

        Returns
        -------
        bool
            True if the robot reached the target.
        """
        return self._navigate_to_pose((x, y))

    def navigate_to_last_detection(self) -> bool:
        """
        Navigate to the pose of the last detected object.

        Returns
        -------
        bool
            True if a detection pose is available and was reached.
        """
        if self.last_detection_pose is None:
            log.warning("PathPlanner: no detection pose available.")
            return False
        return self._navigate_to_pose(self.last_detection_pose)

    def set_waypoint(self, name: str, x: float, y: float) -> None:
        """
        Register or update a named waypoint.

        Parameters
        ----------
        name : str
            Waypoint name.
        x : float
            X coordinate in metres.
        y : float
            Y coordinate in metres.
        """
        self.waypoints[name] = (x, y)
        log.info("PathPlanner: waypoint '%s' set to (%.2f, %.2f).", name, x, y)

    # ── Coverage navigation ───────────────────────────────────────────────────

    def execute_coverage(
        self,
        pattern: str = "boustrophedon",
        speed_factor: float = 1.0,
    ) -> bool:
        """
        Execute a room coverage navigation pattern.

        Parameters
        ----------
        pattern : str
            Coverage pattern: 'boustrophedon' (lawn-mower) or 'spiral'.
        speed_factor : float
            Speed multiplier [0.1, 1.0].

        Returns
        -------
        bool
            True when coverage is complete.
        """
        log.info("PathPlanner: executing coverage pattern '%s' (speed=%.1f).", pattern, speed_factor)

        grid = self._mapper.get_grid()
        if grid is None:
            log.warning("PathPlanner: no map available — coverage skipped.")
            return False

        if pattern == "boustrophedon":
            waypoints = self._boustrophedon_waypoints(grid)
        elif pattern == "spiral":
            waypoints = self._spiral_waypoints(grid)
        else:
            log.error("PathPlanner: unknown coverage pattern '%s'.", pattern)
            return False

        for wp in waypoints:
            if not self._navigate_to_pose(wp):
                log.warning("PathPlanner: coverage waypoint %s unreachable — skipping.", wp)

        log.info("PathPlanner: coverage complete.")
        return True

    def scan_room(self, target_classes: Optional[List[str]] = None) -> bool:
        """
        Rotate in place to scan the room for objects.

        Parameters
        ----------
        target_classes : list of str, optional
            Object classes to look for during the scan.

        Returns
        -------
        bool
            True when the scan rotation is complete.
        """
        log.info("PathPlanner: scanning room (target_classes=%s).", target_classes)
        # In a real implementation this would command the drive to rotate 360°
        # and trigger the vision pipeline at each angular step.
        time.sleep(2.0)  # Stub delay
        return True

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _navigate_to_pose(self, target: Tuple[float, float]) -> bool:
        """
        Plan and execute a path to the target pose.

        Uses A* on the occupancy grid. In stub mode (no grid), moves directly.

        Parameters
        ----------
        target : tuple of float
            (x, y) target in metres.

        Returns
        -------
        bool
            True when the target is reached within BASE_STATION_TOLERANCE.
        """
        grid = self._mapper.get_grid()

        if grid is not None:
            path = self._astar(self._current_pose, target, grid)
            if path is None:
                log.error("PathPlanner: A* found no path to %s.", target)
                return False
        else:
            path = [target]

        # Simulate traversal (real implementation drives along the path)
        for waypoint in path:
            self._current_pose = waypoint
            time.sleep(0.01)  # Stub

        distance = math.hypot(
            self._current_pose[0] - target[0],
            self._current_pose[1] - target[1],
        )
        reached = distance <= BASE_STATION_TOLERANCE
        if not reached:
            log.warning(
                "PathPlanner: did not reach target %s (distance=%.3fm).", target, distance
            )
        return reached

    def _astar(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        grid,
    ) -> Optional[List[Tuple[float, float]]]:
        """
        A* path search on the occupancy grid.

        Parameters
        ----------
        start : tuple
            Start (x, y) in metres.
        goal : tuple
            Goal (x, y) in metres.
        grid : np.ndarray
            Occupancy grid (0 = free, 1 = occupied).

        Returns
        -------
        list of tuple or None
            Waypoint path in metres, or None if no path found.
        """
        resolution = self._mapper.resolution

        def to_cell(pos):
            return (int(pos[0] / resolution), int(pos[1] / resolution))

        def to_world(cell):
            return (cell[0] * resolution, cell[1] * resolution)

        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        start_cell = to_cell(start)
        goal_cell = to_cell(goal)

        open_heap = [(0.0, start_cell)]
        came_from: Dict[tuple, Optional[tuple]] = {start_cell: None}
        g_score: Dict[tuple, float] = {start_cell: 0.0}

        rows, cols = grid.shape if hasattr(grid, "shape") else (100, 100)

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current == goal_cell:
                # Reconstruct path
                path = []
                node = current
                while node is not None:
                    path.append(to_world(node))
                    node = came_from[node]
                path.reverse()
                return path

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbour = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbour[0] < cols and 0 <= neighbour[1] < rows):
                    continue
                if hasattr(grid, "__getitem__") and grid[neighbour[1]][neighbour[0]] > 0:
                    continue  # Occupied cell

                move_cost = math.hypot(dx, dy)
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbour, float("inf")):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g
                    f = tentative_g + heuristic(neighbour, goal_cell)
                    heapq.heappush(open_heap, (f, neighbour))

        return None  # No path found

    def _boustrophedon_waypoints(self, grid) -> List[Tuple[float, float]]:
        """
        Generate boustrophedon (lawn-mower) coverage waypoints.

        Parameters
        ----------
        grid : np.ndarray
            Occupancy grid.

        Returns
        -------
        list of tuple
            Ordered waypoints in metres.
        """
        resolution = self._mapper.resolution
        rows, cols = (100, 100)
        if hasattr(grid, "shape"):
            rows, cols = grid.shape

        waypoints = []
        for row in range(0, rows, 5):  # Step every 5 cells (~25 cm at 5 cm/cell)
            col_range = range(0, cols) if (row // 5) % 2 == 0 else range(cols - 1, -1, -1)
            for col in col_range:
                if not (hasattr(grid, "__getitem__") and grid[row][col] > 0):
                    waypoints.append((col * resolution, row * resolution))
        return waypoints

    def _spiral_waypoints(self, grid) -> List[Tuple[float, float]]:
        """
        Generate inward spiral coverage waypoints.

        Parameters
        ----------
        grid : np.ndarray
            Occupancy grid.

        Returns
        -------
        list of tuple
            Ordered waypoints in metres.
        """
        resolution = self._mapper.resolution
        rows, cols = (100, 100)
        if hasattr(grid, "shape"):
            rows, cols = grid.shape

        waypoints = []
        top, bottom, left, right = 0, rows - 1, 0, cols - 1
        step = 5

        while top <= bottom and left <= right:
            for col in range(left, right + 1, step):
                waypoints.append((col * resolution, top * resolution))
            top += step
            for row in range(top, bottom + 1, step):
                waypoints.append((right * resolution, row * resolution))
            right -= step
            if top <= bottom:
                for col in range(right, left - 1, -step):
                    waypoints.append((col * resolution, bottom * resolution))
                bottom -= step
            if left <= right:
                for row in range(bottom, top - 1, -step):
                    waypoints.append((left * resolution, row * resolution))
                left += step

        return waypoints
