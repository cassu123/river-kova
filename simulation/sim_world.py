#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : simulation/sim_world.py
Purpose     : Simulated household world. Models the robot body (pose, motors,
              battery, gripper), a simple home layout (rooms, waypoints,
              objects), and charging at the dock. Lets the entire brain run
              end-to-end with zero hardware — the SimPicoBridge translates
              normal hardware commands into calls on this world.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-06-12
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_DOCK_RADIUS_M = 0.3          # Within this distance of the dock = charging
_WHEEL_BASE_M = 0.35
_MAX_SPEED_MS = 1.2


@dataclass
class SimObject:
    """An object placed in the simulated home."""

    name: str
    kind: str                 # detection class, e.g. 'cup', 'sock', 'trash_bag'
    x: float
    y: float
    graspable: bool = True
    out_of_place: bool = False


@dataclass
class SimRoom:
    """Axis-aligned rectangular room (x0, y0) → (x1, y1) in metres."""

    name: str
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, x: float, y: float) -> bool:
        """True if the point lies inside this room."""
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


# Default home layout — mirrors the waypoints in navigation/path_planner.py
_DEFAULT_ROOMS: List[SimRoom] = [
    SimRoom("kitchen", 2.0, 1.0, 4.5, 3.0),
    SimRoom("living_room", 4.5, 0.0, 8.0, 2.5),
    SimRoom("bedroom", 6.0, 2.5, 8.0, 4.5),
    SimRoom("bathroom", 1.0, 4.0, 3.0, 6.0),
    SimRoom("hallway", 0.0, 0.0, 2.0, 4.0),
]

_DEFAULT_OBJECTS: List[SimObject] = [
    SimObject("cup-1", "cup", 5.2, 1.2, out_of_place=True),
    SimObject("sock-1", "sock", 6.5, 0.8, out_of_place=True),
    SimObject("trash-bag-1", "trash_bag", 4.0, 0.5),
    SimObject("water-bottle-1", "bottle", 3.2, 2.8),
    SimObject("dog-bowl-1", "dog_bowl", 1.5, 0.5, graspable=False),
    SimObject("food-scoop-1", "scoop", 1.8, 3.5),
]


class SimWorld:
    """
    Simulated household and robot body.

    The robot is a differential-drive base: left/right motor fractions are
    integrated over wall-clock time into a 2-D pose. Battery drains while
    idle/moving and recharges when stopped on the dock. All public methods
    are thread-safe.

    Attributes
    ----------
    pose : tuple of float
        Current (x, y, theta) of the robot.
    battery_pct : float
        Current battery percentage [0, 100].
    is_charging : bool
        True when stopped within dock radius.
    """

    def __init__(
        self,
        start_battery_pct: float = 100.0,
        drain_idle_pct_per_min: float = 0.2,
        drain_moving_pct_per_min: float = 1.5,
        charge_pct_per_min: float = 10.0,
        time_scale: float = 1.0,
        dock_xy: Tuple[float, float] = (0.0, 0.0),
        rooms: Optional[List[SimRoom]] = None,
        objects: Optional[List[SimObject]] = None,
    ) -> None:
        """
        Initialise the simulated world.

        Parameters
        ----------
        start_battery_pct : float
            Battery level at boot.
        drain_idle_pct_per_min : float
            Battery drain while stationary.
        drain_moving_pct_per_min : float
            Battery drain while driving.
        charge_pct_per_min : float
            Charge rate while docked.
        time_scale : float
            Multiplier on simulated time (2.0 = world runs twice as fast).
        dock_xy : tuple
            Charging dock position.
        rooms : list of SimRoom, optional
            Home layout override.
        objects : list of SimObject, optional
            Object placement override.
        """
        self._lock = threading.RLock()

        # Robot body state
        self._x, self._y, self._theta = dock_xy[0], dock_xy[1], 0.0
        self._motor_left: float = 0.0
        self._motor_right: float = 0.0
        self._gripper_pos: float = 1.0      # 1.0 = open
        self._holding: Optional[SimObject] = None
        self._estopped: bool = False

        # Battery model
        self._battery = max(0.0, min(100.0, start_battery_pct))
        self._drain_idle = drain_idle_pct_per_min
        self._drain_moving = drain_moving_pct_per_min
        self._charge_rate = charge_pct_per_min
        self._time_scale = time_scale

        # Environment
        self._dock = dock_xy
        self.rooms: List[SimRoom] = list(rooms) if rooms is not None else list(_DEFAULT_ROOMS)
        self.objects: List[SimObject] = list(objects) if objects is not None else list(_DEFAULT_OBJECTS)

        # Odometry
        self._enc_left: int = 0
        self._enc_right: int = 0
        self._last_update = time.monotonic()

        log.info(
            "SimWorld created (battery=%.0f%%, %d rooms, %d objects).",
            self._battery, len(self.rooms), len(self.objects),
        )

    @classmethod
    def from_config(cls, sim_config) -> "SimWorld":
        """Build a SimWorld from a SimulationConfig dataclass."""
        return cls(
            start_battery_pct=sim_config.start_battery_pct,
            drain_idle_pct_per_min=sim_config.battery_drain_idle_pct_per_min,
            drain_moving_pct_per_min=sim_config.battery_drain_moving_pct_per_min,
            charge_pct_per_min=sim_config.battery_charge_pct_per_min,
            time_scale=sim_config.time_scale,
        )

    # ── Physics integration ───────────────────────────────────────────────────

    def _integrate(self) -> None:
        """Advance the world by the wall-clock time since the last update."""
        now = time.monotonic()
        dt = (now - self._last_update) * self._time_scale
        self._last_update = now
        if dt <= 0:
            return

        # Differential-drive kinematics
        v_left = self._motor_left * _MAX_SPEED_MS
        v_right = self._motor_right * _MAX_SPEED_MS
        v = (v_left + v_right) / 2.0
        w = (v_right - v_left) / _WHEEL_BASE_M

        if not self._estopped and (v != 0.0 or w != 0.0):
            self._theta += w * dt
            self._x += v * math.cos(self._theta) * dt
            self._y += v * math.sin(self._theta) * dt
            # Encoder ticks: ~1000 ticks per metre of wheel travel
            self._enc_left += int(v_left * dt * 1000)
            self._enc_right += int(v_right * dt * 1000)

        # Battery model
        moving = abs(v) > 0.01 or abs(w) > 0.01
        if self.is_charging:
            self._battery = min(100.0, self._battery + self._charge_rate * dt / 60.0)
        elif moving:
            self._battery = max(0.0, self._battery - self._drain_moving * dt / 60.0)
        else:
            self._battery = max(0.0, self._battery - self._drain_idle * dt / 60.0)

    # ── Robot body interface (called by SimPicoBridge) ────────────────────────

    def set_motors(self, left: float, right: float) -> None:
        """Set motor speed fractions [-1.0, 1.0]."""
        with self._lock:
            self._integrate()
            if self._estopped:
                log.debug("SimWorld: motor command ignored — e-stopped.")
                return
            self._motor_left = max(-1.0, min(1.0, left))
            self._motor_right = max(-1.0, min(1.0, right))

    def estop_motors(self) -> None:
        """Immediately stop and latch the e-stop flag."""
        with self._lock:
            self._integrate()
            self._motor_left = 0.0
            self._motor_right = 0.0
            self._estopped = True

    def clear_estop(self) -> None:
        """Clear the simulated e-stop latch."""
        with self._lock:
            self._estopped = False

    def read_battery(self) -> float:
        """Current battery percentage."""
        with self._lock:
            self._integrate()
            return round(self._battery, 2)

    def read_encoders(self) -> Dict[str, int]:
        """Wheel encoder tick counts."""
        with self._lock:
            self._integrate()
            return {"left": self._enc_left, "right": self._enc_right}

    def read_force(self) -> float:
        """Collision force sensor — always 0 N unless a test injects one."""
        return 0.0

    def set_gripper(self, position: float) -> None:
        """
        Set gripper position [0.0 = closed, 1.0 = open].

        Closing within 0.5 m of a graspable object picks it up; opening
        while holding releases it at the robot's current position.
        """
        with self._lock:
            self._integrate()
            closing = position < self._gripper_pos
            self._gripper_pos = max(0.0, min(1.0, position))

            if closing and self._holding is None:
                obj = self.nearest_object(max_distance=0.5, graspable_only=True)
                if obj:
                    self._holding = obj
                    log.info("SimWorld: grasped '%s'.", obj.name)
            elif not closing and self._holding is not None:
                self._holding.x, self._holding.y = self._x, self._y
                log.info("SimWorld: released '%s' at (%.2f, %.2f).", self._holding.name, self._x, self._y)
                self._holding = None

    # ── World queries ─────────────────────────────────────────────────────────

    @property
    def pose(self) -> Tuple[float, float, float]:
        """Current robot pose (x, y, theta)."""
        with self._lock:
            self._integrate()
            return (self._x, self._y, self._theta)

    @property
    def battery_pct(self) -> float:
        """Current battery percentage."""
        return self.read_battery()

    @property
    def is_charging(self) -> bool:
        """True when the robot is stopped within the dock radius."""
        stopped = self._motor_left == 0.0 and self._motor_right == 0.0
        at_dock = math.hypot(self._x - self._dock[0], self._y - self._dock[1]) <= _DOCK_RADIUS_M
        return stopped and at_dock

    @property
    def holding(self) -> Optional[SimObject]:
        """The object currently held by the gripper, if any."""
        with self._lock:
            return self._holding

    def teleport(self, x: float, y: float, theta: float = 0.0) -> None:
        """Instantly move the robot — used by navigation sim and tests."""
        with self._lock:
            self._integrate()
            self._x, self._y, self._theta = x, y, theta

    def room_at(self, x: float, y: float) -> Optional[str]:
        """Name of the room containing the point, or None."""
        for room in self.rooms:
            if room.contains(x, y):
                return room.name
        return None

    def nearest_object(
        self,
        kind: Optional[str] = None,
        max_distance: float = float("inf"),
        graspable_only: bool = False,
    ) -> Optional[SimObject]:
        """
        Find the closest object to the robot.

        Parameters
        ----------
        kind : str, optional
            Restrict to a detection class (e.g. 'cup').
        max_distance : float
            Ignore objects farther than this.
        graspable_only : bool
            Restrict to graspable objects.

        Returns
        -------
        SimObject or None
        """
        best, best_dist = None, max_distance
        for obj in self.objects:
            if obj is self._holding:
                continue
            if kind and obj.kind != kind:
                continue
            if graspable_only and not obj.graspable:
                continue
            dist = math.hypot(obj.x - self._x, obj.y - self._y)
            if dist <= best_dist:
                best, best_dist = obj, dist
        return best

    def out_of_place_objects(self) -> List[SimObject]:
        """Objects flagged as out of place — feeds the autonomy engine."""
        with self._lock:
            return [o for o in self.objects if o.out_of_place and o is not self._holding]

    def snapshot(self) -> Dict[str, Any]:
        """
        Full world-state snapshot for telemetry / the local control API.

        Returns
        -------
        dict
        """
        with self._lock:
            self._integrate()
            return {
                "pose": {"x": round(self._x, 3), "y": round(self._y, 3), "theta": round(self._theta, 3)},
                "room": self.room_at(self._x, self._y),
                "battery_pct": round(self._battery, 2),
                "charging": self.is_charging,
                "estopped": self._estopped,
                "gripper": round(self._gripper_pos, 2),
                "holding": self._holding.name if self._holding else None,
                "objects": [
                    {"name": o.name, "kind": o.kind, "x": o.x, "y": o.y, "out_of_place": o.out_of_place}
                    for o in self.objects
                ],
            }
