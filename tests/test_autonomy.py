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

from types import SimpleNamespace

from autonomy.initiative_engine import (
    InitiativeEngine,
    LLMInitiativeRule,
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


def _fake_client(response_text=None, raises=None, stop_reason="end_turn"):
    """A stand-in Anthropic client returning a canned structured response."""
    client = MagicMock()
    if raises is not None:
        client.messages.create.side_effect = raises
    else:
        client.messages.create.return_value = SimpleNamespace(
            stop_reason=stop_reason,
            content=[SimpleNamespace(type="text", text=response_text or "")],
        )
    return client


_CATALOG = [
    {"chore_type": "VACUUM", "description": "Vacuum a room."},
    {"chore_type": "FEED_DOGS", "description": "Feed the dogs."},
]


class TestLLMInitiativeRule:
    def test_parses_and_clamps_proposals(self):
        client = _fake_client(
            '{"proposals": [{"chore_type": "FEED_DOGS", "room": null, '
            '"priority": 99, "reason": "kids said the dogs are hungry"}]}'
        )
        rule = LLMInitiativeRule(
            chore_catalog=_CATALOG,
            context_provider=lambda: "The kids are home and the dogs haven't eaten.",
            client=client,
        )
        proposals = rule.evaluate(MONDAY_905)
        assert len(proposals) == 1
        assert proposals[0].chore_type == "FEED_DOGS"
        assert proposals[0].priority == 6          # clamped from 99 to the ≤6 ceiling
        assert proposals[0].reason.startswith("LLM:")

    def test_drops_chores_outside_catalogue(self):
        client = _fake_client(
            '{"proposals": [{"chore_type": "LAUNCH_ROCKET", "room": null, '
            '"priority": 5, "reason": "hallucinated"}]}'
        )
        rule = LLMInitiativeRule(_CATALOG, lambda: "ctx", client=client)
        assert rule.evaluate(MONDAY_905) == []

    def test_empty_proposal_list_is_fine(self):
        rule = LLMInitiativeRule(_CATALOG, lambda: "all calm", client=_fake_client('{"proposals": []}'))
        assert rule.evaluate(MONDAY_905) == []

    def test_unparseable_response_is_contained(self):
        rule = LLMInitiativeRule(_CATALOG, lambda: "ctx", client=_fake_client("not json"))
        assert rule.evaluate(MONDAY_905) == []

    def test_api_error_is_contained(self):
        rule = LLMInitiativeRule(_CATALOG, lambda: "ctx", client=_fake_client(raises=RuntimeError("503")))
        assert rule.evaluate(MONDAY_905) == []

    def test_refusal_yields_no_proposals(self):
        client = _fake_client('{"proposals": [{"chore_type": "VACUUM"}]}', stop_reason="refusal")
        rule = LLMInitiativeRule(_CATALOG, lambda: "ctx", client=client)
        assert rule.evaluate(MONDAY_905) == []

    def test_disabled_without_sdk_or_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        # No injected client → must build one; with no key it disables silently.
        rule = LLMInitiativeRule(_CATALOG, lambda: "ctx")
        assert rule.evaluate(MONDAY_905) == []

    def test_context_provider_error_does_not_crash(self):
        def explode():
            raise RuntimeError("notes file locked")
        rule = LLMInitiativeRule(_CATALOG, explode, client=_fake_client('{"proposals": []}'))
        assert rule.evaluate(MONDAY_905) == []

    def test_schema_restricts_enum_to_catalogue(self):
        client = _fake_client('{"proposals": []}')
        rule = LLMInitiativeRule(_CATALOG, lambda: "ctx", client=client)
        rule.evaluate(MONDAY_905)
        kwargs = client.messages.create.call_args.kwargs
        schema = kwargs["output_config"]["format"]["schema"]
        enum = schema["properties"]["proposals"]["items"]["properties"]["chore_type"]["enum"]
        assert set(enum) == {"VACUUM", "FEED_DOGS"}
        assert kwargs["model"] == "claude-opus-4-8"


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
