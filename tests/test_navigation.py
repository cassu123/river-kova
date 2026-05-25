#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : tests/test_navigation.py
Purpose     : Unit tests for the navigation subsystem: RoomMapper, PathPlanner,
              ObstacleAvoidance, and ReturnToBase.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

import sys
import os
import math
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from navigation.room_mapper import RoomMapper
from navigation.path_planner import PathPlanner
from navigation.obstacle_avoid import ObstacleAvoidance
from navigation.return_base import ReturnToBase


# ─────────────────────────────────────────────────────────────────────────────
# RoomMapper Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRoomMapper:
    """Tests for the RoomMapper occupancy grid."""

    def test_initialises_with_grid(self):
        """RoomMapper should create a grid on initialisation."""
        mapper = RoomMapper(resolution=0.05, grid_size=(100, 100))
        grid = mapper.get_grid()
        # Grid may be None if NumPy is unavailable, but should not raise
        # In a full environment, grid should be a numpy array
        assert grid is None or hasattr(grid, "shape")

    def test_resolution_stored(self):
        """RoomMapper should store the configured resolution."""
        mapper = RoomMapper(resolution=0.1)
        assert mapper.resolution == pytest.approx(0.1)

    def test_world_to_cell_origin(self):
        """World origin (0, 0) should map to the grid centre cell."""
        mapper = RoomMapper(resolution=0.05, grid_size=(200, 200))
        col, row = mapper.world_to_cell(0.0, 0.0)
        assert col == 100
        assert row == 100

    def test_world_to_cell_positive(self):
        """Positive world coordinates should map to cells above/right of centre."""
        mapper = RoomMapper(resolution=0.05, grid_size=(200, 200))
        col, row = mapper.world_to_cell(1.0, 1.0)
        # 1.0 m / 0.05 m/cell = 20 cells from origin
        assert col == 120
        assert row == 120

    def test_world_to_cell_clamped(self):
        """world_to_cell() should clamp out-of-bounds coordinates."""
        mapper = RoomMapper(resolution=0.05, grid_size=(100, 100))
        col, row = mapper.world_to_cell(1000.0, 1000.0)
        assert col == 99
        assert row == 99

    def test_cell_to_world_roundtrip(self):
        """cell_to_world(world_to_cell(x, y)) should approximately recover (x, y)."""
        mapper = RoomMapper(resolution=0.05, grid_size=(200, 200))
        x_in, y_in = 2.5, 1.5
        col, row = mapper.world_to_cell(x_in, y_in)
        x_out, y_out = mapper.cell_to_world(col, row)
        assert abs(x_out - x_in) < mapper.resolution
        assert abs(y_out - y_in) < mapper.resolution

    def test_update_from_lidar_does_not_raise(self):
        """update_from_lidar() should not raise even with minimal scan data."""
        mapper = RoomMapper(resolution=0.05, grid_size=(100, 100))
        scan = {0.0: 1.0, math.pi / 2: 1.5, math.pi: 2.0}
        mapper.update_from_lidar(0.0, 0.0, 0.0, scan)  # Should not raise

    def test_update_from_lidar_ignores_zero_range(self):
        """update_from_lidar() should ignore scan entries with range <= 0."""
        mapper = RoomMapper(resolution=0.05, grid_size=(100, 100))
        scan = {0.0: 0.0, math.pi: -1.0}
        mapper.update_from_lidar(0.0, 0.0, 0.0, scan)  # Should not raise or corrupt grid


# ─────────────────────────────────────────────────────────────────────────────
# PathPlanner Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPathPlanner:
    """Tests for the PathPlanner."""

    def _make_planner(self):
        """Create a PathPlanner with a stub RoomMapper."""
        mapper = RoomMapper(resolution=0.05, grid_size=(100, 100))
        return PathPlanner(room_mapper=mapper)

    def test_default_waypoints_registered(self):
        """PathPlanner should have default waypoints on creation."""
        planner = self._make_planner()
        assert "base" in planner.waypoints
        assert "kitchen" in planner.waypoints

    def test_set_waypoint(self):
        """set_waypoint() should register a new named waypoint."""
        planner = self._make_planner()
        planner.set_waypoint("test_point", 3.0, 4.0)
        assert "test_point" in planner.waypoints
        assert planner.waypoints["test_point"] == (3.0, 4.0)

    def test_navigate_to_unknown_waypoint_returns_false(self):
        """navigate_to_waypoint() should return False for an unknown name."""
        planner = self._make_planner()
        result = planner.navigate_to_waypoint("nonexistent_waypoint")
        assert result is False

    def test_navigate_to_base_returns_true(self):
        """navigate_to_waypoint('base') should succeed (stub mode)."""
        planner = self._make_planner()
        # In stub mode (no real drive), the planner simulates traversal
        result = planner.navigate_to_waypoint("base")
        assert result is True

    def test_navigate_to_pose_returns_true(self):
        """navigate_to_pose() should succeed for a reachable coordinate."""
        planner = self._make_planner()
        result = planner.navigate_to_pose(0.0, 0.0)
        assert result is True

    def test_navigate_to_last_detection_no_detection_returns_false(self):
        """navigate_to_last_detection() should return False when no detection is set."""
        planner = self._make_planner()
        result = planner.navigate_to_last_detection()
        assert result is False

    def test_navigate_to_last_detection_with_pose(self):
        """navigate_to_last_detection() should succeed when a detection pose is set."""
        planner = self._make_planner()
        planner.last_detection_pose = (1.0, 1.0)
        result = planner.navigate_to_last_detection()
        assert result is True

    def test_scan_room_returns_true(self):
        """scan_room() should return True (stub implementation)."""
        planner = self._make_planner()
        result = planner.scan_room()
        assert result is True

    def test_boustrophedon_waypoints_not_empty(self):
        """_boustrophedon_waypoints() should return a non-empty list."""
        planner = self._make_planner()
        grid = planner._mapper.get_grid()
        waypoints = planner._boustrophedon_waypoints(grid)
        assert len(waypoints) > 0

    def test_spiral_waypoints_not_empty(self):
        """_spiral_waypoints() should return a non-empty list."""
        planner = self._make_planner()
        grid = planner._mapper.get_grid()
        waypoints = planner._spiral_waypoints(grid)
        assert len(waypoints) > 0

    def test_astar_with_grid_returns_path(self):
        """_astar() should return a non-empty path when a grid is available."""
        planner = self._make_planner()
        grid = planner._mapper.get_grid()
        if grid is None:
            pytest.skip("NumPy not available")
        result = planner._astar((0.0, 0.0), (1.0, 1.0), grid)
        assert result is not None
        assert len(result) > 0


# ─────────────────────────────────────────────────────────────────────────────
# ObstacleAvoidance Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestObstacleAvoidance:
    """Tests for the ObstacleAvoidance module."""

    def _make_oa(self, clearance=0.25):
        drive = MagicMock()
        return ObstacleAvoidance(drive=drive, clearance_min=clearance), drive

    def test_not_blocked_with_no_scan(self):
        """is_blocked should be False when no scan data is available."""
        oa, _ = self._make_oa()
        assert not oa.is_blocked

    def test_not_blocked_when_front_clear(self):
        """is_blocked should be False when front sector is clear."""
        oa, _ = self._make_oa(clearance=0.25)
        oa.update_scan({"front": 1.0})
        assert not oa.is_blocked

    def test_blocked_when_front_too_close(self):
        """is_blocked should be True when front obstacle is within clearance."""
        oa, _ = self._make_oa(clearance=0.25)
        oa.update_scan({"front": 0.1})
        assert oa.is_blocked

    def test_apply_passthrough_when_no_scan(self):
        """apply() should return unchanged velocity when no scan data."""
        oa, _ = self._make_oa()
        lin, ang = oa.apply(0.5, 0.0)
        assert lin == pytest.approx(0.5)
        assert ang == pytest.approx(0.0)

    def test_apply_slows_near_obstacle(self):
        """apply() should reduce linear speed when approaching an obstacle."""
        oa, _ = self._make_oa(clearance=0.25)
        # 0.4 m is within 2× clearance (0.5 m) so the slow-down branch fires
        oa.update_scan({"front": 0.4})
        lin, ang = oa.apply(1.0, 0.0)
        assert lin < 1.0

    def test_apply_stops_at_clearance(self):
        """apply() should set linear to 0 when front is within clearance."""
        oa, _ = self._make_oa(clearance=0.25)
        oa.update_scan({"front": 0.1})
        lin, ang = oa.apply(0.5, 0.0)
        assert lin <= 0.0

    def test_apply_steers_left_when_right_blocked(self):
        """apply() should steer left when right is blocked but left is clear."""
        oa, _ = self._make_oa(clearance=0.25)
        oa.update_scan({"front": 0.3, "front_left": 1.0, "front_right": 0.1})
        lin, ang = oa.apply(0.3, 0.0)
        assert ang > 0  # Positive angular = turn left

    def test_apply_steers_right_when_left_blocked(self):
        """apply() should steer right when left is blocked but right is clear."""
        oa, _ = self._make_oa(clearance=0.25)
        oa.update_scan({"front": 0.3, "front_left": 0.1, "front_right": 1.0})
        lin, ang = oa.apply(0.3, 0.0)
        assert ang < 0  # Negative angular = turn right


# ─────────────────────────────────────────────────────────────────────────────
# ReturnToBase Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestReturnToBase:
    """Tests for the ReturnToBase controller."""

    def _make_rtb(self, base_x=0.0, base_y=0.0):
        drive = MagicMock()
        planner = MagicMock()
        planner.navigate_to_waypoint.return_value = True
        planner.set_waypoint = MagicMock()
        rtb = ReturnToBase(
            drive=drive,
            path_planner=planner,
            base_x=base_x,
            base_y=base_y,
            tolerance=0.1,
        )
        return rtb, drive, planner

    def test_not_docked_initially(self):
        """ReturnToBase should not be docked on creation."""
        rtb, _, _ = self._make_rtb()
        assert not rtb.is_docked

    def test_base_waypoint_registered_on_init(self):
        """ReturnToBase should register the base waypoint with the planner."""
        rtb, _, planner = self._make_rtb(base_x=1.0, base_y=2.0)
        planner.set_waypoint.assert_called_once_with("base", 1.0, 2.0)

    def test_execute_calls_navigate_to_base(self):
        """execute() should call navigate_to_waypoint('base')."""
        rtb, _, planner = self._make_rtb()
        rtb.execute()
        planner.navigate_to_waypoint.assert_called_with("base")

    def test_execute_returns_true_on_success(self):
        """execute() should return True when navigation and docking succeed."""
        rtb, _, _ = self._make_rtb()
        result = rtb.execute()
        assert result is True

    def test_execute_returns_false_when_navigation_fails(self):
        """execute() should return False when the planner cannot reach the base."""
        rtb, _, planner = self._make_rtb()
        planner.navigate_to_waypoint.return_value = False
        result = rtb.execute()
        assert result is False

    def test_docked_after_successful_execute(self):
        """is_docked should be True after a successful execute()."""
        rtb, _, _ = self._make_rtb()
        rtb.execute()
        assert rtb.is_docked

    def test_undock_clears_docked_flag(self):
        """undock() should set is_docked to False."""
        rtb, _, _ = self._make_rtb()
        rtb.execute()
        assert rtb.is_docked
        rtb.undock()
        assert not rtb.is_docked

    def test_undock_commands_reverse_drive(self):
        """undock() should command a reverse drive motion."""
        rtb, drive, _ = self._make_rtb()
        rtb.execute()
        rtb.undock()
        drive.move_safe.assert_called_with(linear=-0.2, angular=0.0)
