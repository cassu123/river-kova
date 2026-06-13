#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : tests/test_api_client.py
Purpose     : Tests for the River Song API client — offline mode and the
              non-blocking background sender (heartbeats must never stall
              the control loop, even with an unreachable server).
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-06-12
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

import time

import pytest

from connectivity.api_client import RiverSongAPIClient

# An address that refuses connections instantly
DEAD_URL = "http://127.0.0.1:9"


@pytest.fixture
def offline_client():
    return RiverSongAPIClient(
        base_url=DEAD_URL, api_key="", robot_id="kova-test", enabled=False,
    )


@pytest.fixture
def online_client():
    client = RiverSongAPIClient(
        base_url=DEAD_URL, api_key="", robot_id="kova-test", enabled=True, timeout=1,
    )
    yield client
    client.stop()


class TestOfflineMode:
    def test_register_is_noop(self, offline_client):
        assert offline_client.register_unit() is True

    def test_deregister_is_noop(self, offline_client):
        assert offline_client.deregister_unit() is True

    def test_heartbeat_is_noop(self, offline_client):
        assert offline_client.heartbeat("IDLE", "NOMINAL", 100.0) is True

    def test_poll_tasks_returns_empty(self, offline_client):
        assert offline_client.poll_tasks() == []

    def test_telemetry_and_alerts_are_noops(self, offline_client):
        assert offline_client.push_telemetry({"cpu": 10}) is True
        assert offline_client.push_alert("INFO", "hello") is True

    def test_request_initiative_returns_empty_offline(self, offline_client):
        assert offline_client.request_initiative("ctx", {}, []) == []

    def test_interpret_command_returns_none_offline(self, offline_client):
        assert offline_client.interpret_command("clean the den") is None


class TestServerBrain:
    def test_request_initiative_parses_proposals(self, online_client):
        proposals = [{"chore_type": "VACUUM", "room": "kitchen", "priority": 4, "reason": "x"}]
        online_client._post = lambda *a, **k: {"proposals": proposals}
        assert online_client.request_initiative("ctx", {"battery_pct": 90}, []) == proposals

    def test_request_initiative_empty_on_error(self, online_client):
        from connectivity.api_client import RiverSongAPIError

        def boom(*a, **k):
            raise RiverSongAPIError("unreachable")
        online_client._post = boom
        assert online_client.request_initiative("ctx", {}, []) == []

    def test_interpret_command_parses_intent(self, online_client):
        online_client._post = lambda *a, **k: {"chore_type": "MOP", "room": "bathroom"}
        assert online_client.interpret_command("wet-clean the loo") == {
            "chore_type": "MOP", "room": "bathroom",
        }

    def test_interpret_command_none_when_no_match(self, online_client):
        online_client._post = lambda *a, **k: {}
        assert online_client.interpret_command("gibberish") is None

    def test_interpret_command_none_on_error(self, online_client):
        from connectivity.api_client import RiverSongAPIError

        def boom(*a, **k):
            raise RiverSongAPIError("unreachable")
        online_client._post = boom
        assert online_client.interpret_command("anything") is None


class TestNonBlockingSender:
    def test_heartbeat_returns_immediately_with_dead_server(self, online_client):
        start = time.monotonic()
        for _ in range(20):
            online_client.heartbeat("IDLE", "NOMINAL", 100.0)
        elapsed = time.monotonic() - start
        # 20 heartbeats against a dead server must enqueue instantly —
        # well under the 5 s watchdog window.
        assert elapsed < 1.0

    def test_task_status_report_is_non_blocking(self, online_client):
        start = time.monotonic()
        for _ in range(20):
            online_client.report_task_status("t-1", "COMPLETED", "done")
        assert time.monotonic() - start < 1.0

    def test_queue_overflow_drops_not_blocks(self, online_client):
        start = time.monotonic()
        for _ in range(500):
            online_client.push_telemetry({"n": 1})
        assert time.monotonic() - start < 2.0

    def test_stop_terminates_worker(self, online_client):
        online_client.heartbeat("IDLE", "NOMINAL", 100.0)
        online_client.stop()
        assert not online_client._worker.is_alive()
