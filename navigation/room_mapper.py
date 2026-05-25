#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : navigation/room_mapper.py
Purpose     : Occupancy grid map builder. Integrates LiDAR scan data to
              maintain a 2D occupancy grid of the environment. Supports
              map save/load for persistent room knowledge.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from core.constants import MAP_RESOLUTION, MAP_UPDATE_INTERVAL

log = logging.getLogger(__name__)

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    log.warning("NumPy not available — RoomMapper running in stub mode.")


class RoomMapper:
    """
    2D occupancy grid map builder.

    Integrates LiDAR scan data using a simple ray-casting update rule.
    The grid is stored as a NumPy array where:
      0   = free
      100 = occupied
      -1  = unknown

    Attributes
    ----------
    resolution : float
        Metres per grid cell.
    grid_size : tuple
        (width, height) of the grid in cells.
    """

    def __init__(
        self,
        resolution: float = MAP_RESOLUTION,
        grid_size: Tuple[int, int] = (400, 400),
        update_interval: float = MAP_UPDATE_INTERVAL,
        map_dir: str = "/var/lib/river-kova/maps",
    ) -> None:
        """
        Initialise the room mapper.

        Parameters
        ----------
        resolution : float
            Metres per grid cell (e.g. 0.05 = 5 cm).
        grid_size : tuple
            (width, height) in cells. Default 400×400 = 20×20 m.
        update_interval : float
            Minimum seconds between grid updates.
        map_dir : str
            Directory for saving/loading persistent maps.
        """
        self.resolution = resolution
        self.grid_size = grid_size
        self._update_interval = update_interval
        self._map_dir = Path(map_dir)
        self._last_update: float = 0.0
        self._lock = threading.Lock()

        if _NUMPY_AVAILABLE:
            self._grid = np.full(
                (grid_size[1], grid_size[0]), fill_value=-1, dtype=np.int8
            )
            # Mark the origin area as free
            cx, cy = grid_size[0] // 2, grid_size[1] // 2
            self._grid[cy - 2:cy + 2, cx - 2:cx + 2] = 0
            self._origin = (cx, cy)  # Grid cell corresponding to (0, 0) world
        else:
            self._grid = None
            self._origin = (grid_size[0] // 2, grid_size[1] // 2)

        log.info(
            "RoomMapper initialised (%dx%d cells, %.2fm/cell, %.1f×%.1fm coverage).",
            grid_size[0], grid_size[1], resolution,
            grid_size[0] * resolution, grid_size[1] * resolution,
        )

    # ── Grid access ───────────────────────────────────────────────────────────

    def get_grid(self):
        """
        Return the current occupancy grid.

        Returns
        -------
        np.ndarray or None
            The occupancy grid, or None if NumPy is unavailable.
        """
        with self._lock:
            return self._grid.copy() if self._grid is not None else None

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates (metres) to grid cell indices.

        Parameters
        ----------
        x : float
            World x in metres.
        y : float
            World y in metres.

        Returns
        -------
        tuple of int
            (col, row) grid cell indices.
        """
        col = int(self._origin[0] + x / self.resolution)
        row = int(self._origin[1] + y / self.resolution)
        col = max(0, min(self.grid_size[0] - 1, col))
        row = max(0, min(self.grid_size[1] - 1, row))
        return col, row

    def cell_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """
        Convert grid cell indices to world coordinates.

        Parameters
        ----------
        col : int
            Grid column.
        row : int
            Grid row.

        Returns
        -------
        tuple of float
            (x, y) in metres.
        """
        x = (col - self._origin[0]) * self.resolution
        y = (row - self._origin[1]) * self.resolution
        return x, y

    # ── Map updates ───────────────────────────────────────────────────────────

    def update_from_lidar(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        scan: Dict[float, float],
    ) -> None:
        """
        Update the occupancy grid from a LiDAR scan.

        Parameters
        ----------
        robot_x : float
            Robot x position in metres.
        robot_y : float
            Robot y position in metres.
        robot_theta : float
            Robot heading in radians.
        scan : dict
            Mapping of angle (radians, relative to robot) → range (metres).
            Range of 0 or negative means no return (ignore).
        """
        if not _NUMPY_AVAILABLE or self._grid is None:
            return

        now = time.monotonic()
        if now - self._last_update < self._update_interval:
            return

        with self._lock:
            robot_col, robot_row = self.world_to_cell(robot_x, robot_y)

            for angle_rel, range_m in scan.items():
                if range_m <= 0:
                    continue

                angle_world = robot_theta + angle_rel
                hit_x = robot_x + range_m * math.cos(angle_world)
                hit_y = robot_y + range_m * math.sin(angle_world)
                hit_col, hit_row = self.world_to_cell(hit_x, hit_y)

                # Mark cells along the ray as free
                self._bresenham_free(robot_col, robot_row, hit_col, hit_row)

                # Mark the hit cell as occupied
                if 0 <= hit_row < self.grid_size[1] and 0 <= hit_col < self.grid_size[0]:
                    self._grid[hit_row, hit_col] = 100

        self._last_update = now

    def _bresenham_free(
        self, x0: int, y0: int, x1: int, y1: int
    ) -> None:
        """
        Mark cells along a Bresenham line as free (ray casting).

        Parameters
        ----------
        x0, y0 : int
            Start cell (robot position).
        x1, y1 : int
            End cell (LiDAR hit point, exclusive).
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if x0 == x1 and y0 == y1:
                break
            if 0 <= y0 < self.grid_size[1] and 0 <= x0 < self.grid_size[0]:
                if self._grid[y0, x0] != 100:  # Don't overwrite occupied cells
                    self._grid[y0, x0] = 0

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, name: str = "default") -> bool:
        """
        Save the current map to disk.

        Parameters
        ----------
        name : str
            Map name (used as filename).

        Returns
        -------
        bool
            True if saved successfully.
        """
        if not _NUMPY_AVAILABLE or self._grid is None:
            return False
        try:
            self._map_dir.mkdir(parents=True, exist_ok=True)
            path = self._map_dir / f"{name}.npy"
            with self._lock:
                np.save(str(path), self._grid)
            log.info("RoomMapper: map saved to %s.", path)
            return True
        except Exception as exc:
            log.error("RoomMapper.save error: %s", exc)
            return False

    def load(self, name: str = "default") -> bool:
        """
        Load a previously saved map from disk.

        Parameters
        ----------
        name : str
            Map name to load.

        Returns
        -------
        bool
            True if loaded successfully.
        """
        if not _NUMPY_AVAILABLE:
            return False
        path = self._map_dir / f"{name}.npy"
        if not path.exists():
            log.warning("RoomMapper: map file not found: %s.", path)
            return False
        try:
            grid = np.load(str(path))
            with self._lock:
                self._grid = grid
            log.info("RoomMapper: map loaded from %s.", path)
            return True
        except Exception as exc:
            log.error("RoomMapper.load error: %s", exc)
            return False
