#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : tests/test_state_machine.py
Purpose     : Unit tests for KovaCore's thread-safe state machine — the
              compare-and-set transition that stops a finishing task from
              silently overwriting a concurrent safety state (ESTOP/FAULT).
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-06-13
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault(
    "KOVA_PROFILE",
    os.path.join(os.path.dirname(__file__), "..", "units", "kova_sim_profile.json"),
)

from core.constants import RobotState
from core.main import KovaCore


def _bare_core() -> KovaCore:
    """A KovaCore with just the state machine — no boot, no hardware."""
    core = KovaCore.__new__(KovaCore)
    core._state_lock = threading.Lock()
    core._state = RobotState.IDLE
    return core


class TestStateTransitions:
    def test_state_property_round_trip(self):
        core = _bare_core()
        core.state = RobotState.EXECUTING_TASK
        assert core.state == RobotState.EXECUTING_TASK

    def test_transition_succeeds_from_expected_state(self):
        core = _bare_core()
        assert core._transition(RobotState.IDLE, RobotState.EXECUTING_TASK) is True
        assert core.state == RobotState.EXECUTING_TASK

    def test_transition_refuses_from_wrong_state(self):
        """A finishing task must not clobber an e-stop set mid-flight."""
        core = _bare_core()
        core.state = RobotState.ESTOP
        assert core._transition(RobotState.EXECUTING_TASK, RobotState.IDLE) is False
        assert core.state == RobotState.ESTOP

    def test_concurrent_transitions_only_one_wins(self):
        """Exactly one of N racing compare-and-sets may succeed."""
        core = _bare_core()
        wins = []
        barrier = threading.Barrier(8)

        def attempt():
            barrier.wait()
            if core._transition(RobotState.IDLE, RobotState.EXECUTING_TASK):
                wins.append(1)

        threads = [threading.Thread(target=attempt) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(wins) == 1
        assert core.state == RobotState.EXECUTING_TASK
