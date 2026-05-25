#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : safety/collision_avoid.py
Purpose     : Collision detection and avoidance. Monitors force/torque sensors
              and LiDAR proximity data. Triggers an immediate halt when impact
              force exceeds the configured limit or an obstacle enters the
              minimum clearance zone.
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
from typing import Callable, Dict, List, Optional, Tuple

from core.constants import COLLISION_FORCE_LIMIT, OBSTACLE_CLEARANCE_MIN

log = logging.getLogger(__name__)


class CollisionAvoidance:
    """
    Collision detection and avoidance module.

    Monitors:
    - Force/torque sensor readings from the Pico bridge
    - LiDAR scan sectors for obstacle proximity

    When a collision is detected (force > limit) or an obstacle enters the
    minimum clearance zone, the on_collision callback is fired and the
    internal collision flag is set.

    Attributes
    ----------
    collision_detected : bool
        True when an active collision or imminent obstacle is present.
    last_collision_force_n : float
        Force reading (N) from the most recent collision event.
    """

    def __init__(
        self,
        max_force_n: float = COLLISION_FORCE_LIMIT,
        clearance_min_m: float = OBSTACLE_CLEARANCE_MIN,
        on_collision: Optional[Callable[[float], None]] = None,
    ) -> None:
        """
        Initialise the collision avoidance module.

        Parameters
        ----------
        max_force_n : float
            Force threshold in Newtons above which a collision is declared.
        clearance_min_m : float
            Minimum obstacle clearance in metres.
        on_collision : callable, optional
            Called with the collision force (N) when a collision is detected.
        """
        self._max_force = max_force_n
        self._clearance_min = clearance_min_m
        self._on_collision = on_collision

        self._collision_detected: bool = False
        self._last_force: float = 0.0
        self._obstacle_sectors: Dict[str, float] = {}  # sector → distance (m)
        self._lock = threading.Lock()

        log.info(
            "CollisionAvoidance initialised (max_force=%.1fN, clearance=%.2fm).",
            max_force_n, clearance_min_m,
        )

    # ── State properties ──────────────────────────────────────────────────────

    @property
    def collision_detected(self) -> bool:
        """True when an active collision or imminent obstacle is present."""
        with self._lock:
            return self._collision_detected

    @property
    def last_collision_force_n(self) -> float:
        """Force reading (N) from the most recent collision event."""
        with self._lock:
            return self._last_force

    @property
    def nearest_obstacle_m(self) -> float:
        """Distance to the nearest obstacle across all LiDAR sectors (-1 if none)."""
        with self._lock:
            if not self._obstacle_sectors:
                return -1.0
            return min(self._obstacle_sectors.values())

    # ── Sensor update methods ─────────────────────────────────────────────────

    def update_force(self, force_n: float) -> bool:
        """
        Update with a new force/torque sensor reading.

        Parameters
        ----------
        force_n : float
            Measured force in Newtons.

        Returns
        -------
        bool
            True if a collision was detected.
        """
        if force_n > self._max_force:
            with self._lock:
                self._collision_detected = True
                self._last_force = force_n
            log.error("Collision detected: %.1f N (limit=%.1f N).", force_n, self._max_force)
            self._fire_callback(force_n)
            return True
        return False

    def update_lidar(self, scan: Dict[str, float]) -> bool:
        """
        Update with a new LiDAR scan.

        Parameters
        ----------
        scan : dict
            Mapping of sector name (e.g. 'front', 'left') to distance in metres.

        Returns
        -------
        bool
            True if an obstacle is within the minimum clearance zone.
        """
        with self._lock:
            self._obstacle_sectors = dict(scan)

        for sector, distance in scan.items():
            if 0.0 < distance < self._clearance_min:
                log.warning(
                    "Obstacle in clearance zone: sector=%s, distance=%.2fm (min=%.2fm).",
                    sector, distance, self._clearance_min,
                )
                with self._lock:
                    self._collision_detected = True
                self._fire_callback(0.0)
                return True

        # Clear collision flag if all sectors are clear
        with self._lock:
            if self._collision_detected and all(
                d >= self._clearance_min for d in scan.values() if d > 0
            ):
                self._collision_detected = False

        return False

    def clear(self) -> None:
        """
        Manually clear the collision flag after the situation is resolved.

        Should only be called after the robot has been moved to a safe position.
        """
        with self._lock:
            self._collision_detected = False
            self._last_force = 0.0
        log.info("CollisionAvoidance flag cleared.")

    def is_path_clear(self, direction: str = "front") -> bool:
        """
        Check whether a specific direction is clear of obstacles.

        Parameters
        ----------
        direction : str
            Sector name to check (e.g. 'front', 'left', 'right', 'rear').

        Returns
        -------
        bool
            True if the sector distance exceeds the minimum clearance.
        """
        with self._lock:
            distance = self._obstacle_sectors.get(direction, -1.0)
        if distance < 0:
            return True  # No data — assume clear (LiDAR may not cover all sectors)
        return distance >= self._clearance_min

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fire_callback(self, force_n: float) -> None:
        """Invoke the on_collision callback safely."""
        if self._on_collision:
            try:
                self._on_collision(force_n)
            except Exception as exc:
                log.error("CollisionAvoidance on_collision callback error: %s", exc)
