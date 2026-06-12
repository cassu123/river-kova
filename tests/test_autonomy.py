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

from autonomy.initiative_engine import (
    InitiativeEngine,
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
