#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : tests/test_autonomy.py
Purpose     : Tests for the initiative engine — scheduled routines, tidy-up
              observations, battery suppression, and cooldowns.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-06-12
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

import time

from autonomy.initiative_engine import (
    InitiativeEngine,
    RiverSongInitiativeRule,
    ScheduledRoutineRule,
    TaskProposal,
    TidyUpRule,
)

# A Monday, 09:05 local time — inside the 09:00 routine window
MONDAY_905 = datetime(2026, 6, 8, 9, 5)
MONDAY_2300 = datetime(2026, 6, 8, 23, 0)
TUESDAY_905 = datetime(2026, 6, 9, 9, 5)


@pytest.fixture
def task_manager():
    manager = MagicMock()
    manager.submit.return_value = "task-123"
    return manager


class TestScheduledRoutineRule:
    def test_fires_inside_window_on_matching_day(self):
        rule = ScheduledRoutineRule([
            {"chore": "VACUUM", "room": "kitchen", "time": "09:00", "days": ["mon"]},
        ])
        proposals = rule.evaluate(MONDAY_905)
        assert len(proposals) == 1
        assert proposals[0].chore_type == "VACUUM"
        assert proposals[0].room == "kitchen"

    def test_does_not_fire_outside_window(self):
        rule = ScheduledRoutineRule([
            {"chore": "VACUUM", "time": "09:00", "days": ["mon"]},
        ])
        assert rule.evaluate(MONDAY_2300) == []

    def test_does_not_fire_on_wrong_day(self):
        rule = ScheduledRoutineRule([
            {"chore": "VACUUM", "time": "09:00", "days": ["mon"]},
        ])
        assert rule.evaluate(TUESDAY_905) == []

    def test_fires_once_per_day(self):
        rule = ScheduledRoutineRule([
            {"chore": "VACUUM", "time": "09:00", "days": ["mon"]},
        ])
        assert len(rule.evaluate(MONDAY_905)) == 1
        assert rule.evaluate(MONDAY_905) == []

    def test_no_days_means_every_day(self):
        rule = ScheduledRoutineRule([
            {"chore": "FEED_DOGS", "time": "09:00"},
        ])
        assert len(rule.evaluate(TUESDAY_905)) == 1

    def test_invalid_time_skipped(self):
        rule = ScheduledRoutineRule([
            {"chore": "VACUUM", "time": "not-a-time"},
        ])
        assert rule.evaluate(MONDAY_905) == []

    def test_fired_state_survives_restart(self, tmp_path):
        """A rebuilt rule (reboot) must not fire the same routine twice in a day."""
        state_path = str(tmp_path / "routine_state.json")
        routines = [{"chore": "FEED_DOGS", "time": "09:00", "days": ["mon"]}]

        rule = ScheduledRoutineRule(routines, state_path=state_path)
        assert len(rule.evaluate(MONDAY_905)) == 1

        rebooted = ScheduledRoutineRule(routines, state_path=state_path)
        assert rebooted.evaluate(MONDAY_905) == []

    def test_corrupt_state_file_starts_fresh(self, tmp_path):
        """An unreadable state file must not prevent routines from firing."""
        state_path = tmp_path / "routine_state.json"
        state_path.write_text("{not json")
        rule = ScheduledRoutineRule(
            [{"chore": "VACUUM", "time": "09:00", "days": ["mon"]}],
            state_path=str(state_path),
        )
        assert len(rule.evaluate(MONDAY_905)) == 1


class TestTidyUpRule:
    def test_proposes_organize_when_items_seen(self):
        rule = TidyUpRule(observation_provider=lambda: [
            {"kind": "sock", "room": "living_room"},
            {"kind": "cup", "room": "living_room"},
        ])
        proposals = rule.evaluate(MONDAY_905)
        assert len(proposals) == 1
        assert proposals[0].chore_type == "ORGANIZE"
        assert proposals[0].room == "living_room"

    def test_silent_when_nothing_seen(self):
        rule = TidyUpRule(observation_provider=lambda: [])
        assert rule.evaluate(MONDAY_905) == []

    def test_provider_error_is_contained(self):
        def explode():
            raise RuntimeError("camera offline")
        rule = TidyUpRule(observation_provider=explode)
        assert rule.evaluate(MONDAY_905) == []


_CATALOG = [
    {"chore_type": "VACUUM", "description": "Vacuum a room."},
    {"chore_type": "FEED_DOGS", "description": "Feed the dogs."},
]


def _drain(rule, provider_calls, timeout=2.0):
    """Run evaluate() until the async worker delivers a result.

    The first call kicks off the background request and returns []. We wait
    for the worker to finish, then evaluate() again to pick up the result.
    """
    first = rule.evaluate(MONDAY_905)
    deadline = time.monotonic() + timeout
    while rule._in_flight and time.monotonic() < deadline:
        time.sleep(0.005)
    second = rule.evaluate(MONDAY_905)
    return first, second


class TestRiverSongInitiativeRule:
    def _rule(self, provider, **kw):
        return RiverSongInitiativeRule(
            proposal_provider=provider,
            chore_catalog=_CATALOG,
            context_provider=kw.pop("context_provider", lambda: "ctx"),
            **kw,
        )

    def test_first_call_is_nonblocking_then_delivers(self):
        calls = []

        def provider(ctx, state):
            calls.append((ctx, state))
            return [{"chore_type": "FEED_DOGS", "room": None,
                     "priority": 99, "reason": "dogs are hungry"}]

        rule = self._rule(provider,
                          context_provider=lambda: "kids home, dogs unfed",
                          state_provider=lambda: {"battery_pct": 80})
        first, second = _drain(rule, calls)

        assert first == []                       # never blocks the control loop
        assert len(second) == 1
        assert second[0].chore_type == "FEED_DOGS"
        assert second[0].priority == 6           # clamped from 99 to the ≤6 ceiling
        assert second[0].reason.startswith("River Song:")
        assert calls[0][0] == "kids home, dogs unfed"
        assert calls[0][1] == {"battery_pct": 80}

    def test_drops_chores_outside_catalogue(self):
        rule = self._rule(lambda c, s: [{"chore_type": "LAUNCH_ROCKET", "priority": 5}])
        _, second = _drain(rule, [])
        assert second == []

    def test_empty_proposals_fine(self):
        rule = self._rule(lambda c, s: [])
        _, second = _drain(rule, [])
        assert second == []

    def test_provider_error_is_contained(self):
        def boom(c, s):
            raise RuntimeError("server 503")
        rule = self._rule(boom)
        _, second = _drain(rule, [])
        assert second == []

    def test_context_provider_error_does_not_crash(self):
        def explode():
            raise RuntimeError("notes file locked")
        seen = []
        rule = self._rule(lambda c, s: seen.append(c) or [], context_provider=explode)
        _, second = _drain(rule, [])
        assert second == []
        assert seen == [""]                      # falls back to empty context

    def test_self_throttles_requests(self):
        calls = []
        rule = self._rule(lambda c, s: calls.append(1) or [], request_interval_sec=999.0)
        _drain(rule, calls)
        # A second evaluate well inside the interval must not fire a new request
        rule.evaluate(MONDAY_905)
        assert len(calls) == 1

    def test_no_request_while_one_in_flight(self):
        import threading
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def slow(ctx, state):
            calls.append(1)
            entered.set()
            release.wait(1.0)
            return []

        rule = self._rule(slow)
        rule.evaluate(MONDAY_905)                # kicks off the (blocked) worker
        assert entered.wait(1.0)                 # worker is now inside the request
        rule.evaluate(MONDAY_905)                # in flight → must not start another
        release.set()
        assert len(calls) == 1


class TestInitiativeEngine:
    def test_submits_rule_proposals(self, task_manager):
        engine = InitiativeEngine(task_manager)
        engine.add_rule(ScheduledRoutineRule([
            {"chore": "VACUUM", "room": "kitchen", "time": "09:00", "days": ["mon"]},
        ]))
        submitted = engine.tick(now=MONDAY_905)
        assert submitted == ["task-123"]
        task_manager.submit.assert_called_once()
        assert task_manager.submit.call_args.kwargs["chore_type"] == "VACUUM"

    def test_suppressed_when_battery_low(self, task_manager):
        engine = InitiativeEngine(task_manager, min_battery_pct=30.0, battery_provider=lambda: 10.0)
        engine.add_rule(ScheduledRoutineRule([
            {"chore": "VACUUM", "time": "09:00", "days": ["mon"]},
        ]))
        assert engine.tick(now=MONDAY_905) == []
        task_manager.submit.assert_not_called()

    def test_allowed_when_battery_ok(self, task_manager):
        engine = InitiativeEngine(task_manager, min_battery_pct=30.0, battery_provider=lambda: 80.0)
        engine.add_rule(ScheduledRoutineRule([
            {"chore": "VACUUM", "time": "09:00", "days": ["mon"]},
        ]))
        assert engine.tick(now=MONDAY_905) == ["task-123"]

    def test_rule_cooldown_enforced(self, task_manager):
        rule = TidyUpRule(observation_provider=lambda: [{"kind": "sock"}])
        engine = InitiativeEngine(task_manager)
        engine.add_rule(rule)
        assert engine.tick(now=MONDAY_905) == ["task-123"]
        # Immediately again — cooldown (1 h) blocks a second proposal
        assert engine.tick(now=MONDAY_905) == []

    def test_rule_error_does_not_kill_engine(self, task_manager):
        bad_rule = MagicMock()
        bad_rule.name = "bad"
        bad_rule.cooldown_sec = 0.0
        bad_rule.evaluate.side_effect = RuntimeError("boom")
        engine = InitiativeEngine(task_manager)
        engine.add_rule(bad_rule)
        engine.add_rule(ScheduledRoutineRule([
            {"chore": "VACUUM", "time": "09:00", "days": ["mon"]},
        ]))
        assert engine.tick(now=MONDAY_905) == ["task-123"]

    def test_rejected_submission_not_counted(self, task_manager):
        task_manager.submit.return_value = None     # e.g. capability gate
        engine = InitiativeEngine(task_manager)
        engine.add_rule(ScheduledRoutineRule([
            {"chore": "MOP", "time": "09:00", "days": ["mon"]},
        ]))
        assert engine.tick(now=MONDAY_905) == []
