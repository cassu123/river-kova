#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : tests/test_safety.py
Purpose     : Unit tests for the safety subsystems: EStop, Watchdog,
              FaultManager, HumanDetection, and CollisionAvoidance.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

import sys
import os
import time
import threading
import pytest

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from safety.estop import EStop
from safety.watchdog import Watchdog
from safety.fault_manager import FaultManager, FaultRecord
from safety.collision_avoid import CollisionAvoidance
from core.constants import FaultCode, SafetyLevel


# ─────────────────────────────────────────────────────────────────────────────
# EStop Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEStop:
    """Tests for the EStop controller."""

    def test_initial_state_not_triggered(self):
        """EStop should not be triggered on creation."""
        estop = EStop()
        assert not estop.is_triggered

    def test_initial_state_not_armed(self):
        """EStop should not be armed until arm() is called."""
        estop = EStop()
        assert not estop.is_armed

    def test_arm_sets_armed(self):
        """arm() should set is_armed to True."""
        estop = EStop()
        estop.arm()
        assert estop.is_armed

    def test_trigger_sets_triggered(self):
        """trigger() should set is_triggered to True."""
        estop = EStop()
        estop.arm()
        estop.trigger(reason="test")
        assert estop.is_triggered

    def test_trigger_fires_callback(self):
        """trigger() should invoke the on_trigger callback."""
        fired = []
        estop = EStop(on_trigger=lambda: fired.append(True))
        estop.arm()
        estop.trigger()
        assert len(fired) == 1

    def test_trigger_is_idempotent(self):
        """Calling trigger() twice should only fire the callback once."""
        count = []
        estop = EStop(on_trigger=lambda: count.append(1))
        estop.arm()
        estop.trigger()
        estop.trigger()
        assert len(count) == 1

    def test_reset_before_delay_fails(self):
        """reset() should return False before the recovery delay elapses."""
        estop = EStop(recovery_delay_sec=60.0)
        estop.arm()
        estop.trigger()
        result = estop.reset()
        assert result is False
        assert estop.is_triggered

    def test_reset_force_succeeds(self):
        """reset(force=True) should succeed regardless of delay."""
        estop = EStop(recovery_delay_sec=60.0)
        estop.arm()
        estop.trigger()
        result = estop.reset(force=True)
        assert result is True
        assert not estop.is_triggered

    def test_reset_fires_callback(self):
        """reset() should invoke the on_reset callback."""
        reset_fired = []
        estop = EStop(recovery_delay_sec=0.0, on_reset=lambda: reset_fired.append(True))
        estop.arm()
        estop.trigger()
        time.sleep(0.05)
        estop.reset()
        assert len(reset_fired) == 1

    def test_disabled_estop_does_not_trigger(self):
        """A disabled EStop should not set is_triggered."""
        estop = EStop(enabled=False)
        estop.arm()
        estop.trigger()
        assert not estop.is_triggered

    def test_can_reset_after_delay(self):
        """can_reset should be True after the recovery delay elapses."""
        estop = EStop(recovery_delay_sec=0.05)
        estop.arm()
        estop.trigger()
        time.sleep(0.1)
        assert estop.can_reset


# ─────────────────────────────────────────────────────────────────────────────
# Watchdog Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestWatchdog:
    """Tests for the Watchdog timer."""

    def test_invalid_timeout_raises(self):
        """Watchdog should raise ValueError for non-positive timeout."""
        with pytest.raises(ValueError):
            Watchdog(timeout_sec=0)

    def test_starts_and_stops(self):
        """Watchdog should start and stop cleanly."""
        wd = Watchdog(timeout_sec=5.0)
        wd.start()
        assert wd.is_running
        wd.stop()
        assert not wd.is_running

    def test_double_start_raises(self):
        """Starting an already-running watchdog should raise RuntimeError."""
        wd = Watchdog(timeout_sec=5.0)
        wd.start()
        try:
            with pytest.raises(RuntimeError):
                wd.start()
        finally:
            wd.stop()

    def test_kick_resets_timer(self):
        """kick() should reset the elapsed time."""
        wd = Watchdog(timeout_sec=5.0)
        wd.start()
        time.sleep(0.1)
        wd.kick()
        assert wd.time_since_last_kick < 0.05
        wd.stop()

    def test_timeout_fires_callback(self):
        """Watchdog should fire on_timeout when not kicked within the timeout."""
        fired = threading.Event()
        wd = Watchdog(timeout_sec=0.1, on_timeout=lambda: fired.set())
        wd.start()
        # Do NOT kick — let it time out
        assert fired.wait(timeout=1.0), "Watchdog did not fire within 1 second"
        wd.stop()

    def test_no_timeout_when_kicked(self):
        """Watchdog should NOT fire when kicked regularly."""
        fired = threading.Event()
        wd = Watchdog(timeout_sec=0.2, on_timeout=lambda: fired.set())
        wd.start()
        for _ in range(5):
            wd.kick()
            time.sleep(0.05)
        wd.stop()
        assert not fired.is_set(), "Watchdog fired despite regular kicks"


# ─────────────────────────────────────────────────────────────────────────────
# FaultManager Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFaultManager:
    """Tests for the FaultManager."""

    def test_no_active_faults_initially(self):
        """FaultManager should have no active faults on creation."""
        fm = FaultManager(robot_id="test-unit")
        assert not fm.has_active_faults

    def test_report_creates_fault(self):
        """report() should create an active fault record."""
        fm = FaultManager(robot_id="test-unit")
        fm.report("Test fault", fault_code=FaultCode.UNKNOWN)
        assert fm.has_active_faults
        assert FaultCode.UNKNOWN in fm.active_faults

    def test_report_fires_callback(self):
        """report() should invoke the on_fault callback."""
        received = []
        fm = FaultManager(robot_id="test-unit", on_fault=lambda c, m: received.append((c, m)))
        fm.report("Test fault", fault_code=FaultCode.SENSOR_FAULT)
        assert len(received) >= 1
        assert received[0][0] == FaultCode.SENSOR_FAULT

    def test_repeated_fault_increments_count(self):
        """Reporting the same fault_code twice should increment count."""
        fm = FaultManager(robot_id="test-unit")
        fm.report("First", fault_code=FaultCode.MOTOR_FAULT)
        fm.report("Second", fault_code=FaultCode.MOTOR_FAULT)
        record = fm.active_faults[FaultCode.MOTOR_FAULT]
        assert record.count == 2

    def test_resolve_removes_fault(self):
        """resolve() should remove the fault from active_faults."""
        fm = FaultManager(robot_id="test-unit")
        fm.report("Test", fault_code=FaultCode.COMMS_FAULT)
        result = fm.resolve(FaultCode.COMMS_FAULT)
        assert result is True
        assert not fm.has_active_faults

    def test_resolve_nonexistent_returns_false(self):
        """resolve() on an unknown fault_code should return False."""
        fm = FaultManager(robot_id="test-unit")
        result = fm.resolve("NONEXISTENT_CODE")
        assert result is False

    def test_resolve_all_clears_faults(self):
        """resolve_all() should clear all active faults."""
        fm = FaultManager(robot_id="test-unit")
        fm.report("A", fault_code=FaultCode.MOTOR_FAULT)
        fm.report("B", fault_code=FaultCode.ARM_FAULT)
        count = fm.resolve_all()
        assert count == 2
        assert not fm.has_active_faults

    def test_get_history_returns_records(self):
        """get_history() should return serialisable fault records."""
        fm = FaultManager(robot_id="test-unit")
        fm.report("Test", fault_code=FaultCode.UNKNOWN)
        history = fm.get_history()
        assert len(history) >= 1
        assert "fault_code" in history[0]
        assert "message" in history[0]

    def test_report_without_callback_does_not_error(self, caplog):
        """report() with no on_fault registered must not try to call None."""
        fm = FaultManager(robot_id="test-unit")
        with caplog.at_level("ERROR"):
            fm.report("Test", fault_code=FaultCode.UNKNOWN)
        assert not any("callback error" in r.message for r in caplog.records)

    def test_callback_fires_on_repeat_fault(self):
        """The controller must hear about every report, repeats included."""
        received = []
        fm = FaultManager(robot_id="test-unit", on_fault=lambda c, m: received.append(c))
        fm.report("First", fault_code=FaultCode.MOTOR_FAULT)
        fm.report("Second", fault_code=FaultCode.MOTOR_FAULT)
        assert len(received) == 2


# ─────────────────────────────────────────────────────────────────────────────
# CollisionAvoidance Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCollisionAvoidance:
    """Tests for the CollisionAvoidance module."""

    def test_no_collision_initially(self):
        """CollisionAvoidance should not report a collision on creation."""
        ca = CollisionAvoidance(max_force_n=10.0, clearance_min_m=0.25)
        assert not ca.collision_detected

    def test_force_below_limit_no_collision(self):
        """Force below the limit should not trigger a collision."""
        ca = CollisionAvoidance(max_force_n=10.0)
        result = ca.update_force(5.0)
        assert result is False
        assert not ca.collision_detected

    def test_force_above_limit_triggers_collision(self):
        """Force above the limit should trigger a collision."""
        fired = []
        ca = CollisionAvoidance(max_force_n=10.0, on_collision=lambda f: fired.append(f))
        result = ca.update_force(15.0)
        assert result is True
        assert ca.collision_detected
        assert len(fired) == 1
        assert fired[0] == 15.0

    def test_lidar_clear_no_collision(self):
        """LiDAR scan with all sectors clear should not trigger collision."""
        ca = CollisionAvoidance(clearance_min_m=0.25)
        result = ca.update_lidar({"front": 1.0, "left": 0.8, "right": 0.9})
        assert result is False

    def test_lidar_obstacle_in_clearance_zone(self):
        """LiDAR obstacle within clearance zone should trigger collision."""
        fired = []
        ca = CollisionAvoidance(clearance_min_m=0.25, on_collision=lambda f: fired.append(f))
        result = ca.update_lidar({"front": 0.1, "left": 1.0})
        assert result is True
        assert ca.collision_detected
        assert len(fired) == 1

    def test_clear_resets_collision_flag(self):
        """clear() should reset the collision flag."""
        ca = CollisionAvoidance(max_force_n=10.0)
        ca.update_force(20.0)
        assert ca.collision_detected
        ca.clear()
        assert not ca.collision_detected

    def test_is_path_clear_with_no_data(self):
        """is_path_clear() should return True when no scan data is available."""
        ca = CollisionAvoidance()
        assert ca.is_path_clear("front") is True

    def test_is_path_clear_with_obstacle(self):
        """is_path_clear() should return False when obstacle is within clearance."""
        ca = CollisionAvoidance(clearance_min_m=0.25)
        ca.update_lidar({"front": 0.1})
        assert ca.is_path_clear("front") is False

    def test_nearest_obstacle_returns_minimum(self):
        """nearest_obstacle_m should return the minimum distance across sectors."""
        ca = CollisionAvoidance()
        ca.update_lidar({"front": 1.0, "left": 0.5, "right": 2.0})
        assert ca.nearest_obstacle_m == pytest.approx(0.5)
