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
_ROBOT_RADIUS_M = 0.15        # Body radius for wall collision


@dataclass
class SimWall:
    """A wall segment (x0, y0) → (x1, y1) — blocks motion and LiDAR."""

    x0: float
    y0: float
    x1: float
    y1: float


def _ray_segment_t(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> Optional[float]:
    """Fraction t along ray AB where it crosses segment CD, or None."""
    rx, ry = bx - ax, by - ay
    sx, sy = dx - cx, dy - cy
    denom = rx * sy - ry * sx
    if abs(denom) < 1e-12:
        return None
    t = ((cx - ax) * sy - (cy - ay) * sx) / denom
    u = ((cx - ax) * ry - (cy - ay) * rx) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t
    return None


def _point_segment_distance(
    px: float, py: float,
    ax: float, ay: float, bx: float, by: float,
) -> float:
    """Shortest distance from point P to segment AB."""
    abx, aby = bx - ax, by - ay
    length_sq = abx * abx + aby * aby
    if length_sq < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / length_sq))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


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
    # Movable items
    SimObject("cup-1", "cup", 5.2, 1.2, out_of_place=True),
    SimObject("sock-1", "sock", 6.5, 0.8, out_of_place=True),
    SimObject("trash-bag-1", "trash_bag", 4.0, 0.5),
    SimObject("water-bottle-1", "bottle", 3.2, 2.8),
    SimObject("dog-bowl-1", "dog_bowl", 1.5, 0.5, graspable=False),
    SimObject("food-scoop-1", "scoop", 1.8, 3.5),
    # Fixtures — what room recognition anchors on (a kitchen is wherever
    # the stove and fridge are seen, not a rectangle in a profile)
    SimObject("stove-1", "stove", 2.3, 2.7, graspable=False),
    SimObject("fridge-1", "fridge", 4.2, 2.7, graspable=False),
    SimObject("dishwasher-1", "dishwasher", 3.2, 2.8, graspable=False),
    SimObject("couch-1", "couch", 6.8, 2.2, graspable=False),
    SimObject("tv-1", "tv", 7.7, 1.0, graspable=False),
    SimObject("coffee-table-1", "coffee_table", 6.3, 1.2, graspable=False),
    SimObject("bed-1", "bed", 7.2, 3.9, graspable=False),
    SimObject("dresser-1", "dresser", 6.4, 4.2, graspable=False),
    SimObject("toilet-1", "toilet", 1.4, 5.6, graspable=False),
    SimObject("bathtub-1", "bathtub", 2.6, 5.6, graspable=False),
    SimObject("coat-rack-1", "coat_rack", 0.3, 3.6, graspable=False),
    SimObject("shoe-rack-1", "shoe_rack", 0.4, 0.4, graspable=False),
]


# Walls follow the room boundaries, with doorways connecting:
# hallway↔kitchen, kitchen↔living room, living room↔bedroom, hallway↔bathroom.
_DEFAULT_WALLS: List[SimWall] = [
    SimWall(0.0, 0.0, 0.0, 4.0),       # hallway west
    SimWall(0.0, 0.0, 2.0, 0.0),       # hallway south
    SimWall(0.0, 4.0, 1.1, 4.0),       # hallway north (door to bathroom 1.1–2.0)
    SimWall(2.0, 0.0, 2.0, 1.8),       # hallway east lower (door to kitchen 1.8–2.6)
    SimWall(2.0, 2.6, 2.0, 4.0),       # hallway east upper
    SimWall(2.0, 1.0, 4.5, 1.0),       # kitchen south
    SimWall(2.0, 3.0, 4.5, 3.0),       # kitchen north
    SimWall(4.5, 1.0, 4.5, 1.5),       # kitchen east lower (door to living 1.5–2.3)
    SimWall(4.5, 2.3, 4.5, 3.0),       # kitchen east upper
    SimWall(4.5, 0.0, 4.5, 1.0),       # living west
    SimWall(4.5, 0.0, 8.0, 0.0),       # living south
    SimWall(8.0, 0.0, 8.0, 2.5),       # living east
    SimWall(4.5, 2.5, 6.0, 2.5),       # living north west piece
    SimWall(6.0, 2.5, 6.6, 2.5),       # living north (door to bedroom 6.6–7.4)
    SimWall(7.4, 2.5, 8.0, 2.5),       # living north east piece
    SimWall(6.0, 2.5, 6.0, 4.5),       # bedroom west
    SimWall(6.0, 4.5, 8.0, 4.5),       # bedroom north
    SimWall(8.0, 2.5, 8.0, 4.5),       # bedroom east
    SimWall(1.0, 4.0, 1.0, 6.0),       # bathroom west
    SimWall(1.0, 6.0, 3.0, 6.0),       # bathroom north
    SimWall(3.0, 4.0, 3.0, 6.0),       # bathroom east
    SimWall(2.0, 4.0, 3.0, 4.0),       # bathroom south east piece
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
        dock_xy: Tuple[float, float] = (0.6, 0.6),
        rooms: Optional[List[SimRoom]] = None,
        objects: Optional[List[SimObject]] = None,
        walls: Optional[List[SimWall]] = None,
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
        self.walls: List[SimWall] = list(walls) if walls is not None else list(_DEFAULT_WALLS)

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
            new_x = self._x + v * math.cos(self._theta) * dt
            new_y = self._y + v * math.sin(self._theta) * dt
            # Walls are solid, but contact slides rather than freezes —
            # a body clipping a door frame keeps its parallel motion, like
            # a real robot brushing past. Wheels spin either way, so
            # encoders still tick.
            if not self._wall_blocked(self._x, self._y, new_x, new_y):
                self._x, self._y = new_x, new_y
            elif not self._wall_blocked(self._x, self._y, new_x, self._y):
                self._x = new_x
            elif not self._wall_blocked(self._x, self._y, self._x, new_y):
                self._y = new_y
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

    def _wall_blocked(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """True if moving from (x0,y0) to (x1,y1) would cross or touch a wall."""
        for wall in self.walls:
            if _ray_segment_t(x0, y0, x1, y1, wall.x0, wall.y0, wall.x1, wall.y1) is not None:
                return True
            if _point_segment_distance(x1, y1, wall.x0, wall.y0, wall.x1, wall.y1) < _ROBOT_RADIUS_M:
                return True
        return False

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

    def visible_objects(self, max_distance: float = 3.0) -> List[SimObject]:
        """
        What the robot's camera can see right now — the sim stand-in for the
        vision pipeline's detections.

        Walls block sight: only objects in the robot's current room count,
        and only within range. Outside any room, range alone applies.

        Parameters
        ----------
        max_distance : float
            Camera visibility range in metres.

        Returns
        -------
        list of SimObject
        """
        with self._lock:
            self._integrate()
            robot_room = self.room_at(self._x, self._y)
            visible = []
            for obj in self.objects:
                if obj is self._holding:
                    continue
                if math.hypot(obj.x - self._x, obj.y - self._y) > max_distance:
                    continue
                if robot_room is not None and self.room_at(obj.x, obj.y) != robot_room:
                    continue
                visible.append(obj)
            return visible

    def lidar_scan(
        self,
        num_beams: int = 120,
        max_range: float = 5.0,
    ) -> List[Tuple[float, float, bool]]:
        """
        Simulated 2-D LiDAR sweep — the stand-in for an RPLiDAR.

        Parameters
        ----------
        num_beams : int
            Evenly spaced beams over the full circle.
        max_range : float
            Sensor range in metres.

        Returns
        -------
        list of (float, float, bool)
            (bearing relative to robot heading, measured range, hit) per
            beam; range == max_range and hit == False when nothing is seen.
        """
        with self._lock:
            self._integrate()
            x, y, theta = self._x, self._y, self._theta

        scan: List[Tuple[float, float, bool]] = []
        for index in range(num_beams):
            bearing = -math.pi + (2.0 * math.pi * index) / num_beams
            angle = theta + bearing
            end_x = x + max_range * math.cos(angle)
            end_y = y + max_range * math.sin(angle)
            nearest_t: Optional[float] = None
            for wall in self.walls:
                t = _ray_segment_t(x, y, end_x, end_y, wall.x0, wall.y0, wall.x1, wall.y1)
                if t is not None and (nearest_t is None or t < nearest_t):
                    nearest_t = t
            if nearest_t is None:
                scan.append((bearing, max_range, False))
            else:
                scan.append((bearing, round(nearest_t * max_range, 4), True))
        return scan

    def bounds(self) -> Tuple[float, float, float, float]:
        """Bounding box (min_x, min_y, max_x, max_y) over all rooms."""
        return (
            min(r.x0 for r in self.rooms),
            min(r.y0 for r in self.rooms),
            max(r.x1 for r in self.rooms),
            max(r.y1 for r in self.rooms),
        )

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
                "dock": {"x": self._dock[0], "y": self._dock[1]},
                "walls": [
                    {"x0": w.x0, "y0": w.y0, "x1": w.x1, "y1": w.y1}
                    for w in self.walls
                ],
                "rooms": [
                    {"name": r.name, "x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1}
                    for r in self.rooms
                ],
                "objects": [
                    {"name": o.name, "kind": o.kind, "x": o.x, "y": o.y, "out_of_place": o.out_of_place}
                    for o in self.objects
                ],
            }
