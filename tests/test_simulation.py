#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : tests/test_simulation.py
Purpose     : Tests for the simulated world, the simulated Pico bridge, and
              the hardware factory — proving the full brain runs against a
              purely virtual robot body.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-06-12
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

import time

import pytest

from hardware.drive_controller import DriveController
from hardware.pico_bridge import PicoBridgeError
from simulation.sim_pico_bridge import SimPicoBridge
from simulation.sim_world import SimObject, SimWorld


@pytest.fixture
def world():
    return SimWorld(start_battery_pct=80.0)


@pytest.fixture
def bridge(world):
    b = SimPicoBridge(world)
    b.connect()
    return b


class TestSimWorld:
    def test_starts_at_dock(self, world):
        x, y, theta = world.pose
        assert (x, y, theta) == (0.6, 0.6, 0.0)

    def test_starts_with_configured_battery(self, world):
        assert world.battery_pct == pytest.approx(80.0, abs=0.5)

    def test_charging_at_dock_when_stopped(self, world):
        assert world.is_charging

    def test_driving_moves_the_robot(self, world):
        world.set_motors(0.5, 0.5)
        time.sleep(0.2)
        x, y, _ = world.pose
        assert x > 0.65
        assert y == pytest.approx(0.6, abs=0.01)

    def test_turning_changes_heading(self, world):
        world.set_motors(-0.5, 0.5)
        time.sleep(0.2)
        _, _, theta = world.pose
        assert theta > 0.01

    def test_estop_stops_motion(self, world):
        world.set_motors(1.0, 1.0)
        world.estop_motors()
        x_before, _, _ = world.pose
        time.sleep(0.15)
        x_after, _, _ = world.pose
        assert x_after == pytest.approx(x_before, abs=0.001)

    def test_motor_commands_ignored_while_estopped(self, world):
        world.estop_motors()
        world.set_motors(1.0, 1.0)
        time.sleep(0.15)
        x, _, _ = world.pose
        assert x == pytest.approx(0.6, abs=0.001)

    def test_battery_charges_at_dock(self):
        w = SimWorld(start_battery_pct=50.0, charge_pct_per_min=60.0, time_scale=60.0)
        time.sleep(0.2)
        assert w.battery_pct > 50.0

    def test_battery_drains_while_moving(self):
        w = SimWorld(start_battery_pct=50.0, drain_moving_pct_per_min=60.0, time_scale=60.0)
        w.set_motors(1.0, 1.0)
        time.sleep(0.2)
        assert w.battery_pct < 50.0

    def test_room_lookup(self, world):
        assert world.room_at(3.0, 2.0) == "kitchen"
        assert world.room_at(50.0, 50.0) is None

    def test_nearest_object_by_kind(self, world):
        obj = world.nearest_object(kind="bottle")
        assert obj is not None
        assert obj.kind == "bottle"

    def test_grasp_picks_up_nearby_object(self, world):
        target = world.nearest_object(kind="cup")
        world.teleport(target.x, target.y)
        world.set_gripper(0.0)
        assert world.holding is target

    def test_release_drops_object_at_current_position(self, world):
        target = world.nearest_object(kind="cup")
        world.teleport(target.x, target.y)
        world.set_gripper(0.0)
        world.teleport(1.0, 1.0)
        world.set_gripper(1.0)
        assert world.holding is None
        assert target.x == pytest.approx(1.0)
        assert target.y == pytest.approx(1.0)

    def test_grasp_ignores_distant_objects(self, world):
        world.teleport(20.0, 20.0)
        world.set_gripper(0.0)
        assert world.holding is None

    def test_out_of_place_objects_reported(self, world):
        kinds = {o.kind for o in world.out_of_place_objects()}
        assert "cup" in kinds
        assert "sock" in kinds

    def test_visible_objects_blocked_by_walls(self, world):
        # From the kitchen the robot sees kitchen fixtures, not the couch
        # one room over — walls block sight.
        world.teleport(3.25, 2.0)
        kinds = {o.kind for o in world.visible_objects()}
        assert "stove" in kinds
        assert "fridge" in kinds
        assert "couch" not in kinds
        assert "bed" not in kinds

    def test_visible_objects_respects_range(self, world):
        world.teleport(3.25, 2.0)
        assert world.visible_objects(max_distance=0.1) == []

    def test_every_room_has_recognition_anchors(self, world):
        from vision.room_classifier import RoomClassifier, UNKNOWN
        classifier = RoomClassifier()
        for room in world.rooms:
            world.teleport((room.x0 + room.x1) / 2, (room.y0 + room.y1) / 2)
            kinds = [o.kind for o in world.visible_objects()]
            label, _ = classifier.classify(kinds)
            assert label == room.name, f"{room.name} recognised as {label}"

    def test_bounds_cover_all_rooms(self, world):
        min_x, min_y, max_x, max_y = world.bounds()
        assert (min_x, min_y) == (0.0, 0.0)
        assert (max_x, max_y) == (8.0, 6.0)

    def test_snapshot_structure(self, world):
        snap = world.snapshot()
        assert {"pose", "room", "battery_pct", "charging", "estopped", "gripper", "holding", "objects", "rooms", "dock"} <= set(snap)
        assert {"name", "x0", "y0", "x1", "y1"} <= set(snap["rooms"][0])


class TestSimPicoBridge:
    def test_ping(self, bridge):
        response = bridge.send_command("ping")
        assert response["status"] == "ok"

    def test_requires_connection(self, world):
        b = SimPicoBridge(world)
        with pytest.raises(PicoBridgeError):
            b.send_command("ping")

    def test_read_battery(self, bridge):
        assert bridge.read_battery() == pytest.approx(80.0, abs=0.5)

    def test_set_motor_speeds_moves_world(self, bridge, world):
        bridge.set_motor_speeds(0.8, 0.8)
        time.sleep(0.15)
        x, _, _ = world.pose
        assert x > 0.03

    def test_emergency_stop(self, bridge, world):
        bridge.set_motor_speeds(1.0, 1.0)
        bridge.emergency_stop_motors()
        x_before, _, _ = world.pose
        time.sleep(0.15)
        x_after, _, _ = world.pose
        assert x_after == pytest.approx(x_before, abs=0.001)

    def test_unknown_command_raises(self, bridge):
        with pytest.raises(PicoBridgeError):
            bridge.send_command("warp_drive")

    def test_read_encoders(self, bridge, world):
        bridge.set_motor_speeds(1.0, 1.0)
        time.sleep(0.15)
        ticks = bridge.read_encoders()
        assert ticks["left"] > 0
        assert ticks["right"] > 0


class TestDriveControllerOnSim:
    """The unmodified DriveController must work against the simulator."""

    def test_move_drives_sim_world(self, bridge, world):
        drive = DriveController(pico_bridge=bridge)
        assert drive.move(0.6, 0.0)
        time.sleep(0.15)
        x, _, _ = world.pose
        assert x > 0.03

    def test_emergency_stop_latches(self, bridge, world):
        drive = DriveController(pico_bridge=bridge)
        drive.move(0.6, 0.0)
        drive.emergency_stop()
        assert not drive.move(0.6, 0.0)


class TestHardwareFactory:
    def test_sim_backend_builds_sim_set(self, tmp_path, monkeypatch):
        import json

        from core.config import Config

        profile = {
            "robot_id": "kova-test",
            "version": "1.0.0",
            "hardware": {"backend": "sim"},
            "connectivity": {"enabled": False, "vpn_required": False},
        }
        profile_path = tmp_path / "profile.json"
        profile_path.write_text(json.dumps(profile))

        monkeypatch.setenv("KOVA_PROFILE", str(profile_path))
        Config.reset()
        try:
            from hardware.factory import build_hardware

            hw = build_hardware(Config())
            assert hw.backend == "sim"
            assert hw.sim_world is not None
            assert hw.bridge.is_connected
            assert hw.drive.move(0.2, 0.0)
        finally:
            Config.reset()

    def test_unknown_backend_raises(self, tmp_path, monkeypatch):
        import json

        from core.config import Config

        profile = {
            "robot_id": "kova-test",
            "version": "1.0.0",
            "hardware": {"backend": "hovercraft"},
        }
        profile_path = tmp_path / "profile.json"
        profile_path.write_text(json.dumps(profile))

        monkeypatch.setenv("KOVA_PROFILE", str(profile_path))
        Config.reset()
        try:
            from hardware.factory import build_hardware

            with pytest.raises(ValueError, match="hovercraft"):
                build_hardware(Config())
        finally:
            Config.reset()
