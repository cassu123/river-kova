#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : tests/test_capabilities.py
Purpose     : Tests for the universal capability model — chores declare what
              they need, profiles declare what the body has, and the task
              manager gates submission. Also covers the new GET_WATER /
              FEED_DOGS chores and voice routing.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-06-12
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from unittest.mock import MagicMock

import pytest

from tasks.chore_library import ChoreLibrary
from tasks.task_manager import TaskManager
from tasks.task_queue import TaskQueue


@pytest.fixture
def library():
    return ChoreLibrary()


def make_manager(capabilities):
    api_client = MagicMock()
    api_client.interpret_command.return_value = None
    return TaskManager(
        task_queue=TaskQueue(max_size=10),
        chore_library=ChoreLibrary(),
        api_client=api_client,
        robot_id="kova-test",
        capabilities=capabilities,
    )


class TestNewChores:
    def test_get_water_registered(self, library):
        chore = library.get("GET_WATER")
        assert chore is not None
        assert chore["name"] == "Get Water"
        assert len(chore["steps"]) > 0

    def test_feed_dogs_registered(self, library):
        chore = library.get("FEED_DOGS")
        assert chore is not None
        assert any(s["action"] == "arm_pour_motion" for s in chore["steps"])

    def test_all_chores_declare_capabilities(self, library):
        for chore_type in library.list_chores():
            chore = library.get(chore_type)
            assert "required_capabilities" in chore, chore_type
            assert isinstance(chore["required_capabilities"], list)


class TestCapabilityGating:
    def test_none_capabilities_allows_everything(self):
        manager = make_manager(capabilities=None)
        assert manager.submit("FETCH") is not None

    def test_full_capabilities_allows_chore(self):
        manager = make_manager(capabilities={"fetch", "arm_manipulation"})
        assert manager.submit("FETCH") is not None

    def test_missing_capability_rejects_chore(self):
        manager = make_manager(capabilities={"vacuum"})       # No arm!
        assert manager.submit("FETCH") is None

    def test_armless_body_can_still_vacuum(self):
        manager = make_manager(capabilities={"vacuum"})
        assert manager.submit("VACUUM") is not None

    def test_get_water_requires_capability(self):
        manager = make_manager(capabilities={"arm_manipulation"})
        assert manager.submit("GET_WATER") is None
        manager2 = make_manager(capabilities={"arm_manipulation", "get_water"})
        assert manager2.submit("GET_WATER") is not None


class TestVoiceRouting:
    def test_get_water_voice(self):
        manager = make_manager(capabilities=None)
        task_id = manager.submit_from_voice("River, have Kova get me some water")
        assert task_id is not None
        task = manager.get_next_task()
        assert task["chore_type"] == "GET_WATER"

    def test_feed_dogs_voice(self):
        manager = make_manager(capabilities=None)
        task_id = manager.submit_from_voice("River, have Kova feed the dogs")
        assert task_id is not None
        task = manager.get_next_task()
        assert task["chore_type"] == "FEED_DOGS"

    def test_plain_fetch_still_routes_to_fetch(self):
        manager = make_manager(capabilities=None)
        manager.submit_from_voice("River, have Kova fetch my phone from the bedroom")
        task = manager.get_next_task()
        assert task["chore_type"] == "FETCH"
