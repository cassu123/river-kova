#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : tests/test_mapping.py
Purpose     : Tests for structural mapping: sim walls + LiDAR, occupancy
              grid integration, A* planning over learned space, frontier
              detection, and map-aware navigation through doorways.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-06-12
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

import math
import time

import pytest

from navigation.explorer import FrontierExplorer, Navigator, PointDriver
from navigation.occupancy_map import FREE, OCCUPIED, UNKNOWN, OccupancyMap
from simulation.sim_world import SimWorld


class _DirectBridge:
    """Minimal IOBridge stand-in driving a SimWorld directly."""

    def __init__(self, world):
        self._world = world

    def set_motor_speeds(self, left, right):
        self._world.set_motors(left, right)


@pytest.fixture
def world():
    return SimWorld()


# ─────────────────────────────────────────────────────────────────────────────
# WALLS & LIDAR (SimWorld)
# ─────────────────────────────────────────────────────────────────────────────


class TestWalls:
    @pytest.mark.timeout(15)
    def test_wall_blocks_motion(self, world):
        # Drive east in the upper hallway: the x=2 wall (no door there)
        # must stop the robot.
        world.teleport(1.5, 3.5, 0.0)
        world.set_motors(1.0, 1.0)
        time.sleep(2.0)
        x, y, _ = world.pose
        assert x < 2.0
        world.set_motors(0.0, 0.0)

    @pytest.mark.timeout(15)
    def test_doorway_lets_robot_through(self, world):
        # Drive east at the hallway→kitchen door (y between 1.8 and 2.6).
        world.teleport(1.5, 2.2, 0.0)
        world.set_motors(1.0, 1.0)
        time.sleep(2.0)
        x, _, _ = world.pose
        assert x > 2.0
        world.set_motors(0.0, 0.0)

    def test_teleport_ignores_walls(self, world):
        world.teleport(7.0, 3.5)
        assert world.room_at(*world.pose[:2]) == "bedroom"


class TestLidar:
    def test_beam_hits_known_wall(self, world):
        # Facing east at (1, 1): the backwards beam (bearing -pi) hits the
        # hallway west wall at x=0, one metre away.
        world.teleport(1.0, 1.0, 0.0)
        scan = world.lidar_scan(num_beams=120, max_range=5.0)
        bearing, distance, hit = scan[0]          # First beam is bearing -pi
        assert bearing == pytest.approx(-math.pi)
        assert hit
        assert distance == pytest.approx(1.0, abs=0.02)

    def test_all_beams_bounded(self, world):
        world.teleport(3.0, 2.0, 0.5)
        for _, distance, _ in world.lidar_scan():
            assert 0.0 < distance <= 5.0

    def test_walls_in_snapshot(self, world):
        snap = world.snapshot()
        assert len(snap["walls"]) > 10
        assert {"x0", "y0", "x1", "y1"} <= set(snap["walls"][0])


# ─────────────────────────────────────────────────────────────────────────────
# OCCUPANCY MAP
# ─────────────────────────────────────────────────────────────────────────────


class TestOccupancyMap:
    @pytest.fixture
    def occ(self):
        return OccupancyMap(origin_xy=(0.0, 0.0), width_m=5.0, height_m=5.0)

    def test_starts_unknown(self, occ):
        assert occ.state_at(2.0, 2.0) == UNKNOWN
        assert occ.explored_area_m2() == 0.0

    def test_scan_carves_free_and_marks_hit(self, occ):
        # One beam east from (1, 1) hitting at 2 m.
        occ.integrate_scan(1.0, 1.0, 0.0, [(0.0, 2.0, True)])
        assert occ.state_at(2.0, 1.0) == FREE
        assert occ.state_at(3.0, 1.0) == OCCUPIED

    def test_miss_marks_endpoint_free(self, occ):
        occ.integrate_scan(1.0, 1.0, 0.0, [(0.0, 2.0, False)])
        assert occ.state_at(3.0, 1.0) == FREE

    def test_structure_change_relearns(self, occ):
        # A wall is seen, then the wall is removed (door opened) — the next
        # scan passes through and the cell becomes free.
        occ.integrate_scan(1.0, 1.0, 0.0, [(0.0, 2.0, True)])
        assert occ.state_at(3.0, 1.0) == OCCUPIED
        occ.integrate_scan(1.0, 1.0, 0.0, [(0.0, 3.5, True)])
        assert occ.state_at(3.0, 1.0) == FREE

    def test_frontier_appears_at_edge_of_known(self, occ):
        occ.integrate_scan(1.0, 1.0, 0.0, [(0.0, 2.0, False)])
        assert len(occ.frontiers()) > 0

    def test_real_sim_scan_maps_room(self, world):
        occ = OccupancyMap()
        world.teleport(1.0, 2.0, 0.0)
        occ.integrate_scan(1.0, 2.0, 0.0, world.lidar_scan())
        assert occ.state_at(1.0, 2.0) == FREE
        assert occ.state_at(0.0, 2.0) == OCCUPIED      # hallway west wall
        assert occ.explored_area_m2() > 1.0

    def test_save_and_load_roundtrip(self, occ, tmp_path):
        occ.integrate_scan(1.0, 1.0, 0.0, [(0.0, 2.0, True)])
        path = str(tmp_path / "occ.json")
        occ.save(path)
        restored = OccupancyMap()
        assert restored.load(path)
        assert restored.state_at(3.0, 1.0) == OCCUPIED
        assert restored.state_at(2.0, 1.0) == FREE

    def test_snapshot_rows(self, occ):
        snap = occ.snapshot()
        assert len(snap["rows"]) == 50
        assert len(snap["rows"][0]) == 50
        assert set(snap["rows"][0]) <= {"?", ".", "#"}


class TestPathPlanning:
    def _open_floor(self, occ):
        """Mark the whole grid free for synthetic planning tests."""
        occ._grid[:, :] = FREE

    def test_plans_straight_line_on_open_floor(self):
        occ = OccupancyMap(origin_xy=(0.0, 0.0), width_m=5.0, height_m=5.0)
        self._open_floor(occ)
        path = occ.plan_path(0.5, 0.5, 4.0, 4.0)
        assert path is not None
        assert path[-1] == pytest.approx((4.05, 4.05), abs=0.1)

    def test_routes_around_wall_through_gap(self):
        occ = OccupancyMap(origin_xy=(0.0, 0.0), width_m=5.0, height_m=5.0)
        self._open_floor(occ)
        # Vertical wall at x=2.5 with a gap at y in [3.0, 4.0]
        wall_i = int(2.5 / occ.resolution_m)
        occ._grid[wall_i, :int(3.0 / occ.resolution_m)] = OCCUPIED
        occ._grid[wall_i, int(4.0 / occ.resolution_m):] = OCCUPIED
        path = occ.plan_path(1.0, 1.0, 4.0, 1.0)
        assert path is not None
        # The route must detour up through the gap
        assert any(y > 2.9 for _, y in path)

    def test_unknown_space_is_not_traversed(self):
        occ = OccupancyMap(origin_xy=(0.0, 0.0), width_m=5.0, height_m=5.0)
        # Only a thin free corridor is known; goal is in unknown territory
        occ._grid[:, 10] = FREE
        assert occ.plan_path(0.5, 1.05, 4.0, 4.0) is None


# ─────────────────────────────────────────────────────────────────────────────
# MAP-AWARE NAVIGATION (through a real doorway)
# ─────────────────────────────────────────────────────────────────────────────


class TestNavigator:
    @pytest.mark.timeout(90)
    def test_drives_through_doorway_on_learned_map(self, world):
        occ = OccupancyMap()
        # Learn the hallway and bathroom structure from a few scan poses
        for sx, sy in [(1.0, 1.0), (1.0, 2.5), (1.5, 3.5), (1.6, 4.3), (2.0, 5.0)]:
            world.teleport(sx, sy, 0.0)
            occ.integrate_scan(sx, sy, 0.0, world.lidar_scan())

        world.teleport(1.0, 2.5, 0.0)
        driver = PointDriver(bridge=_DirectBridge(world), pose_provider=lambda: world.pose)
        navigator = Navigator(occupancy_map=occ, driver=driver, pose_provider=lambda: world.pose)

        # Hallway → bathroom requires routing through the y=4 doorway
        assert navigator.go_to(2.0, 5.0, timeout_sec=80.0)
        assert world.room_at(*world.pose[:2]) == "bathroom"

    def test_refuses_unmapped_destination(self, world):
        occ = OccupancyMap()      # Nothing learned at all
        driver = PointDriver(bridge=_DirectBridge(world), pose_provider=lambda: world.pose)
        navigator = Navigator(occupancy_map=occ, driver=driver, pose_provider=lambda: world.pose)
        assert not navigator.go_to(7.0, 3.5, timeout_sec=5.0)


# ─────────────────────────────────────────────────────────────────────────────
# FRONTIER EXPLORER (logic — full runs are exercised by live boot)
# ─────────────────────────────────────────────────────────────────────────────


class _FakeNavigator:
    def __init__(self, succeed):
        self._succeed = succeed
        self.calls = 0

    def go_to(self, x, y, timeout_sec=60.0):
        self.calls += 1
        return self._succeed


class TestFrontierExplorer:
    def test_no_frontiers_means_done(self):
        occ = OccupancyMap(origin_xy=(0.0, 0.0), width_m=3.0, height_m=3.0)
        explorer = FrontierExplorer(
            navigator=_FakeNavigator(True),
            occupancy_map=occ,
            pose_provider=lambda: (1.0, 1.0, 0.0),
            min_target_distance_m=0.0,
        )
        assert explorer.explore()

    def test_unreachable_frontiers_get_blacklisted(self):
        occ = OccupancyMap(origin_xy=(0.0, 0.0), width_m=3.0, height_m=3.0)
        occ.integrate_scan(1.0, 1.0, 0.0, [(0.0, 1.0, False)])
        navigator = _FakeNavigator(False)
        explorer = FrontierExplorer(
            navigator=navigator,
            occupancy_map=occ,
            pose_provider=lambda: (1.0, 1.0, 0.0),
            min_target_distance_m=0.0,
        )
        # Every frontier fails → all blacklisted → loop terminates
        assert explorer.explore(max_targets=50)
        assert navigator.calls > 0

    def test_abort_stops_exploration(self):
        occ = OccupancyMap(origin_xy=(0.0, 0.0), width_m=3.0, height_m=3.0)
        occ.integrate_scan(1.0, 1.0, 0.0, [(0.0, 1.0, False)])
        explorer = FrontierExplorer(
            navigator=_FakeNavigator(True),
            occupancy_map=occ,
            pose_provider=lambda: (1.0, 1.0, 0.0),
        )
        assert not explorer.explore(abort_check=lambda: True)
