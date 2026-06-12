#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : navigation/occupancy_map.py
Purpose     : Robot-vacuum-style structural map. The robot builds a 2-D
              occupancy grid of walls and free floor from LiDAR as it
              drives — structure (walls, doorways) is learned once and
              updated slowly, while what's INSIDE rooms is handled by the
              semantic map, because contents change.

              Provides the three structural services the brain needs:
              scan integration, A* path planning over learned free space,
              and frontier detection (the edge between mapped and unmapped
              floor) that drives exploration. Persisted as JSON.

              Body-agnostic: input is pose + range scan, whatever produced
              them — SimWorld rays today, an RPLiDAR later.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-06-12
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import heapq
import json
import logging
import math
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

UNKNOWN = -1
FREE = 0
OCCUPIED = 1

_SQRT2 = math.sqrt(2.0)


def _bresenham(i0: int, j0: int, i1: int, j1: int) -> List[Tuple[int, int]]:
    """Grid cells along the line from (i0, j0) to (i1, j1), inclusive."""
    cells = []
    di, dj = abs(i1 - i0), abs(j1 - j0)
    si = 1 if i0 < i1 else -1
    sj = 1 if j0 < j1 else -1
    err = di - dj
    i, j = i0, j0
    while True:
        cells.append((i, j))
        if i == i1 and j == j1:
            break
        e2 = 2 * err
        if e2 > -dj:
            err -= dj
            i += si
        if e2 < di:
            err += di
            j += sj
    return cells


class OccupancyMap:
    """
    LiDAR-built occupancy grid with planning and frontier queries.

    Thread-safe: the control loop integrates scans while the navigator
    plans and the API reads snapshots.

    Attributes
    ----------
    resolution_m : float
        Cell edge length.
    """

    def __init__(
        self,
        origin_xy: Tuple[float, float] = (-1.0, -1.0),
        width_m: float = 12.0,
        height_m: float = 10.0,
        resolution_m: float = 0.1,
    ) -> None:
        """
        Parameters
        ----------
        origin_xy : tuple
            World position of grid cell (0, 0)'s corner.
        width_m, height_m : float
            Mapped area extent — generous enough to hold any home the
            robot is dropped into.
        resolution_m : float
            Cell size; 0.1 m resolves doorways.
        """
        self.origin_x, self.origin_y = origin_xy
        self.resolution_m = resolution_m
        self._nx = int(round(width_m / resolution_m))
        self._ny = int(round(height_m / resolution_m))
        self._grid = np.full((self._nx, self._ny), UNKNOWN, dtype=np.int8)
        self._lock = threading.RLock()

    # ── Coordinates ───────────────────────────────────────────────────────────

    def _cell(self, x: float, y: float) -> Tuple[int, int]:
        return (
            int((x - self.origin_x) // self.resolution_m),
            int((y - self.origin_y) // self.resolution_m),
        )

    def _world(self, i: int, j: int) -> Tuple[float, float]:
        half = self.resolution_m / 2.0
        return (
            self.origin_x + i * self.resolution_m + half,
            self.origin_y + j * self.resolution_m + half,
        )

    def _in_grid(self, i: int, j: int) -> bool:
        return 0 <= i < self._nx and 0 <= j < self._ny

    # ── Scan integration ──────────────────────────────────────────────────────

    def integrate_scan(
        self,
        x: float,
        y: float,
        theta: float,
        scan: Sequence[Tuple[float, float, bool]],
    ) -> None:
        """
        Fold one LiDAR sweep into the map.

        Cells along each beam are carved FREE; a beam that hit something
        marks its endpoint cell OCCUPIED. Hits overwrite free and vice
        versa, so a removed wall (or an opened door) is re-learned.

        Parameters
        ----------
        x, y, theta : float
            Robot pose when the scan was taken.
        scan : sequence of (bearing, range, hit)
            Beams relative to the robot heading.
        """
        i0, j0 = self._cell(x, y)
        with self._lock:
            for bearing, distance, hit in scan:
                angle = theta + bearing
                end_i, end_j = self._cell(
                    x + distance * math.cos(angle),
                    y + distance * math.sin(angle),
                )
                cells = _bresenham(i0, j0, end_i, end_j)
                for i, j in cells[:-1]:
                    if self._in_grid(i, j):
                        self._grid[i, j] = FREE
                last_i, last_j = cells[-1]
                if self._in_grid(last_i, last_j):
                    self._grid[last_i, last_j] = OCCUPIED if hit else FREE

    # ── Queries ───────────────────────────────────────────────────────────────

    def state_at(self, x: float, y: float) -> int:
        """UNKNOWN / FREE / OCCUPIED at a world position."""
        i, j = self._cell(x, y)
        if not self._in_grid(i, j):
            return UNKNOWN
        with self._lock:
            return int(self._grid[i, j])

    def explored_area_m2(self) -> float:
        """Square metres of floor mapped as free."""
        with self._lock:
            free_cells = int(np.count_nonzero(self._grid == FREE))
        return round(free_cells * self.resolution_m ** 2, 2)

    @staticmethod
    def _dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
        """Grow a boolean mask outward by radius_cells (Chebyshev)."""
        grown = mask.copy()
        for di in range(-radius_cells, radius_cells + 1):
            for dj in range(-radius_cells, radius_cells + 1):
                if di == 0 and dj == 0:
                    continue
                shifted = np.zeros_like(mask)
                src = mask[
                    max(0, -di):mask.shape[0] - max(0, di),
                    max(0, -dj):mask.shape[1] - max(0, dj),
                ]
                shifted[
                    max(0, di):mask.shape[0] - max(0, -di),
                    max(0, dj):mask.shape[1] - max(0, -dj),
                ] = src
                grown |= shifted
        return grown

    def frontiers(self, min_clearance_m: float = 0.0) -> List[Tuple[float, float]]:
        """
        Free cells bordering unknown space — where exploring pays off.

        Parameters
        ----------
        min_clearance_m : float
            Drop frontiers closer than this to a known wall — the robot
            can't be sent there anyway, so they would only waste targets.

        Returns
        -------
        list of (float, float)
            World coordinates of frontier cell centres.
        """
        with self._lock:
            grid = self._grid
            free = grid == FREE
            unknown_neighbour = np.zeros_like(free)
            unknown = grid == UNKNOWN
            unknown_neighbour[1:, :] |= unknown[:-1, :]
            unknown_neighbour[:-1, :] |= unknown[1:, :]
            unknown_neighbour[:, 1:] |= unknown[:, :-1]
            unknown_neighbour[:, :-1] |= unknown[:, 1:]
            mask = free & unknown_neighbour
            if min_clearance_m > 0:
                radius = max(1, int(math.ceil(min_clearance_m / self.resolution_m)))
                mask &= ~self._dilate(grid == OCCUPIED, radius)
            cells = np.argwhere(mask)
        return [self._world(int(i), int(j)) for i, j in cells]

    # ── Path planning ─────────────────────────────────────────────────────────

    def plan_path(
        self,
        start_x: float,
        start_y: float,
        goal_x: float,
        goal_y: float,
        inflate_m: float = 0.2,
    ) -> Optional[List[Tuple[float, float]]]:
        """
        A* over learned free space, walls inflated by the robot's radius.

        Unknown space is treated as blocked — the robot only commits to
        floor it has actually seen, like a vacuum.

        Returns
        -------
        list of (x, y) world waypoints including the goal, or None when no
        route exists through the mapped free space.
        """
        start = self._cell(start_x, start_y)
        goal = self._cell(goal_x, goal_y)
        if not (self._in_grid(*start) and self._in_grid(*goal)):
            return None

        with self._lock:
            blocked = self._grid != FREE
            # Inflate obstacles so paths keep body-radius clearance
            inflate_cells = max(1, int(math.ceil(inflate_m / self.resolution_m)))
            blocked |= self._dilate(self._grid == OCCUPIED, inflate_cells)

        # The robot occupies its own start cell; never let inflation strand it
        si, sj = start
        blocked[max(0, si - 1):si + 2, max(0, sj - 1):sj + 2] = False
        if blocked[goal]:
            return None

        # A* (8-connected)
        open_heap: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
        g_cost: Dict[Tuple[int, int], float] = {start: 0.0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                path = [current]
                while path[-1] in came_from:
                    path.append(came_from[path[-1]])
                path.reverse()
                return [self._world(i, j) for i, j in path]

            ci, cj = current
            for di, dj, step in (
                (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, _SQRT2), (1, -1, _SQRT2), (-1, 1, _SQRT2), (-1, -1, _SQRT2),
            ):
                ni, nj = ci + di, cj + dj
                if not self._in_grid(ni, nj) or blocked[ni, nj]:
                    continue
                tentative = g_cost[current] + step
                if tentative < g_cost.get((ni, nj), float("inf")):
                    g_cost[(ni, nj)] = tentative
                    came_from[(ni, nj)] = current
                    heuristic = math.hypot(goal[0] - ni, goal[1] - nj)
                    heapq.heappush(open_heap, (tentative + heuristic, (ni, nj)))
        return None

    # ── Snapshot & persistence ────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, object]:
        """
        Compact state for the dashboard: '?' unknown, '.' free, '#' wall.

        Rows are indexed by j (world y), characters by i (world x).
        """
        with self._lock:
            grid = self._grid.copy()
        chars = {UNKNOWN: "?", FREE: ".", OCCUPIED: "#"}
        rows = [
            "".join(chars[int(grid[i, j])] for i in range(self._nx))
            for j in range(self._ny)
        ]
        return {
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "resolution_m": self.resolution_m,
            "rows": rows,
            "explored_m2": self.explored_area_m2(),
        }

    def save(self, path: str) -> None:
        """Persist the grid to JSON (atomic best-effort)."""
        snap = self.snapshot()
        payload = {k: snap[k] for k in ("origin_x", "origin_y", "resolution_m", "rows")}
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(target)
        log.info("OccupancyMap: saved %.1f m² mapped to %s.", self.explored_area_m2(), path)

    def load(self, path: str) -> bool:
        """Restore a saved grid; returns False when absent or invalid."""
        target = Path(path)
        if not target.exists():
            return False
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            rows = payload["rows"]
            states = {"?": UNKNOWN, ".": FREE, "#": OCCUPIED}
            grid = np.full((len(rows[0]), len(rows)), UNKNOWN, dtype=np.int8)
            for j, row in enumerate(rows):
                for i, char in enumerate(row):
                    grid[i, j] = states.get(char, UNKNOWN)
            with self._lock:
                self.origin_x = float(payload["origin_x"])
                self.origin_y = float(payload["origin_y"])
                self.resolution_m = float(payload["resolution_m"])
                self._nx, self._ny = grid.shape
                self._grid = grid
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            log.warning("OccupancyMap: could not load %s: %s", path, exc)
            return False
        log.info("OccupancyMap: restored %.1f m² mapped from %s.", self.explored_area_m2(), path)
        return True
