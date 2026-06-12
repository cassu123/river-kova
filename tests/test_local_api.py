#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : tests/test_local_api.py
Purpose     : Tests for the self-hosted local control API using a fake core.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-06-12
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from unittest.mock import MagicMock

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from api.local_server import create_app
from core.constants import RobotState, SafetyLevel
from tasks.chore_library import ChoreLibrary


@pytest.fixture
def fake_core():
    core = MagicMock()
    core.robot_id = "kova-test"
    core.state = RobotState.IDLE
    core.safety_level = SafetyLevel.NOMINAL
    core.battery_monitor.level = 87.5
    core.chore_library = ChoreLibrary()
    core.task_manager.active_task = None
    core.task_manager.queue_size = 0
    core.task_manager.submit.return_value = "task-42"
    core.task_manager.submit_from_voice.return_value = "task-43"
    core.sim_world = None
    return core


@pytest.fixture
def client(fake_core):
    return TestClient(create_app(fake_core))


class TestLocalAPI:
    def test_dashboard_served_at_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "River Kova" in response.text

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["robot_id"] == "kova-test"

    def test_status(self, client):
        body = client.get("/status").json()
        assert body["state"] == "IDLE"
        assert body["safety_level"] == "NOMINAL"
        assert body["battery_pct"] == 87.5

    def test_chores_lists_new_chores(self, client):
        body = client.get("/chores").json()
        names = " ".join(body.keys())
        assert "GET_WATER" in names
        assert "FEED_DOGS" in names

    def test_submit_task(self, client, fake_core):
        response = client.post("/tasks", json={"chore_type": "vacuum", "room": "kitchen"})
        assert response.status_code == 200
        assert response.json()["task_id"] == "task-42"
        assert fake_core.task_manager.submit.call_args.kwargs["chore_type"] == "VACUUM"

    def test_submit_rejected_task_is_400(self, client, fake_core):
        fake_core.task_manager.submit.return_value = None
        response = client.post("/tasks", json={"chore_type": "FLY_TO_MOON"})
        assert response.status_code == 400

    def test_voice_command(self, client):
        response = client.post("/voice", json={"command": "feed the dogs"})
        assert response.status_code == 200
        assert response.json()["task_id"] == "task-43"

    def test_voice_no_match_is_400(self, client, fake_core):
        fake_core.task_manager.submit_from_voice.return_value = None
        response = client.post("/voice", json={"command": "sing me a song"})
        assert response.status_code == 400

    def test_estop(self, client, fake_core):
        response = client.post("/estop")
        assert response.status_code == 200
        fake_core._emergency_halt.assert_called_once()
