#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : tests/test_tasks.py
Purpose     : Unit tests for the task subsystem: ChoreLibrary, TaskQueue,
              TaskManager, and TaskExecutor.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

import sys
import os
import time
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tasks.chore_library import ChoreLibrary
from tasks.task_queue import TaskQueue, TaskQueueFullError
from tasks.task_manager import TaskManager
from core.constants import ChoreType, TaskStatus


# ─────────────────────────────────────────────────────────────────────────────
# ChoreLibrary Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestChoreLibrary:
    """Tests for the ChoreLibrary."""

    def test_loads_builtin_chores(self):
        """ChoreLibrary should load all built-in chore definitions."""
        lib = ChoreLibrary()
        assert len(lib) > 0

    def test_get_vacuum_chore(self):
        """get() should return the VACUUM chore definition."""
        lib = ChoreLibrary()
        chore = lib.get(ChoreType.VACUUM)
        assert chore is not None
        assert chore["chore_type"] == ChoreType.VACUUM
        assert "steps" in chore
        assert len(chore["steps"]) > 0

    def test_get_unknown_chore_returns_none(self):
        """get() should return None for an unknown chore type."""
        lib = ChoreLibrary()
        result = lib.get("NONEXISTENT_CHORE")
        assert result is None

    def test_all_builtin_chores_have_steps(self):
        """Every built-in chore should have at least one step."""
        lib = ChoreLibrary()
        for chore_type in lib.list_chores():
            chore = lib.get(chore_type)
            assert len(chore["steps"]) > 0, f"Chore '{chore_type}' has no steps"

    def test_all_steps_have_required_fields(self):
        """Every step should have 'action', 'params', and 'timeout_sec'."""
        lib = ChoreLibrary()
        for chore_type in lib.list_chores():
            for step in lib.get_steps(chore_type):
                assert "action" in step, f"Step in '{chore_type}' missing 'action'"
                assert "params" in step, f"Step in '{chore_type}' missing 'params'"
                assert "timeout_sec" in step, f"Step in '{chore_type}' missing 'timeout_sec'"

    def test_register_custom_chore(self):
        """register() should add a custom chore to the library."""
        lib = ChoreLibrary()
        lib.register("CUSTOM_TEST", {
            "name": "Custom Test Chore",
            "steps": [{"action": "navigate_to_base", "params": {}, "timeout_sec": 30, "retry": 0}],
        })
        assert lib.get("CUSTOM_TEST") is not None
        assert lib.get("CUSTOM_TEST")["name"] == "Custom Test Chore"

    def test_register_missing_name_raises(self):
        """register() should raise ValueError if 'name' is missing."""
        lib = ChoreLibrary()
        with pytest.raises(ValueError, match="name"):
            lib.register("BAD_CHORE", {"steps": []})

    def test_register_missing_steps_raises(self):
        """register() should raise ValueError if 'steps' is missing."""
        lib = ChoreLibrary()
        with pytest.raises(ValueError, match="steps"):
            lib.register("BAD_CHORE", {"name": "Bad"})

    def test_list_chores_returns_all_types(self):
        """list_chores() should include all ChoreType values."""
        lib = ChoreLibrary()
        chore_list = lib.list_chores()
        for ct in [ChoreType.VACUUM, ChoreType.FETCH, ChoreType.ORGANIZE]:
            assert ct in chore_list

    def test_get_steps_returns_list(self):
        """get_steps() should return a list of step dicts."""
        lib = ChoreLibrary()
        steps = lib.get_steps(ChoreType.FETCH)
        assert isinstance(steps, list)
        assert len(steps) > 0

    def test_get_steps_unknown_returns_empty(self):
        """get_steps() for an unknown chore should return an empty list."""
        lib = ChoreLibrary()
        steps = lib.get_steps("DOES_NOT_EXIST")
        assert steps == []


# ─────────────────────────────────────────────────────────────────────────────
# TaskQueue Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTaskQueue:
    """Tests for the TaskQueue."""

    def _make_task(self, name="test", priority=5):
        return {"name": name, "priority": priority}

    def test_empty_on_creation(self):
        """Queue should be empty on creation."""
        q = TaskQueue()
        assert q.is_empty
        assert q.size == 0

    def test_enqueue_increases_size(self):
        """enqueue() should increase the queue size."""
        q = TaskQueue()
        q.enqueue(self._make_task())
        assert q.size == 1

    def test_enqueue_assigns_id(self):
        """enqueue() should assign a unique ID if not present."""
        q = TaskQueue()
        task = self._make_task()
        task_id = q.enqueue(task)
        assert task_id is not None
        assert task.get("id") == task_id

    def test_dequeue_returns_task(self):
        """dequeue() should return the enqueued task."""
        q = TaskQueue()
        q.enqueue(self._make_task("my_task"))
        task = q.dequeue()
        assert task is not None
        assert task["name"] == "my_task"

    def test_dequeue_empty_returns_none(self):
        """dequeue() on an empty queue should return None."""
        q = TaskQueue()
        assert q.dequeue() is None

    def test_priority_ordering(self):
        """Higher priority tasks should be dequeued first."""
        q = TaskQueue()
        q.enqueue(self._make_task("low", priority=1))
        q.enqueue(self._make_task("high", priority=10))
        q.enqueue(self._make_task("mid", priority=5))
        first = q.dequeue()
        assert first["name"] == "high"

    def test_fifo_within_same_priority(self):
        """Tasks with equal priority should be dequeued in insertion order."""
        q = TaskQueue()
        q.enqueue(self._make_task("first", priority=5))
        time.sleep(0.01)
        q.enqueue(self._make_task("second", priority=5))
        assert q.dequeue()["name"] == "first"
        assert q.dequeue()["name"] == "second"

    def test_queue_full_raises(self):
        """Enqueuing beyond max_size should raise TaskQueueFullError."""
        q = TaskQueue(max_size=2)
        q.enqueue(self._make_task())
        q.enqueue(self._make_task())
        with pytest.raises(TaskQueueFullError):
            q.enqueue(self._make_task())

    def test_remove_by_id(self):
        """remove() should remove a task by ID."""
        q = TaskQueue()
        task_id = q.enqueue(self._make_task("removable"))
        result = q.remove(task_id)
        assert result is True
        assert q.is_empty

    def test_remove_nonexistent_returns_false(self):
        """remove() with an unknown ID should return False."""
        q = TaskQueue()
        result = q.remove("nonexistent-id")
        assert result is False

    def test_drain_clears_queue(self):
        """drain() should remove and return all tasks."""
        q = TaskQueue()
        q.enqueue(self._make_task("a"))
        q.enqueue(self._make_task("b"))
        tasks = q.drain()
        assert len(tasks) == 2
        assert q.is_empty

    def test_peek_does_not_remove(self):
        """peek() should return the next task without removing it."""
        q = TaskQueue()
        q.enqueue(self._make_task("peek_me", priority=9))
        peeked = q.peek()
        assert peeked is not None
        assert peeked["name"] == "peek_me"
        assert q.size == 1  # Still in queue

    def test_dequeued_task_status_is_running(self):
        """Dequeued task should have status RUNNING."""
        q = TaskQueue()
        q.enqueue(self._make_task())
        task = q.dequeue()
        assert task["status"] == TaskStatus.RUNNING


# ─────────────────────────────────────────────────────────────────────────────
# TaskManager Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTaskManager:
    """Tests for the TaskManager."""

    def _make_manager(self):
        """Create a TaskManager with mocked dependencies."""
        queue = TaskQueue()
        library = ChoreLibrary()
        api_client = MagicMock()
        api_client.report_task_status.return_value = True
        return TaskManager(
            task_queue=queue,
            chore_library=library,
            api_client=api_client,
            robot_id="test-unit",
        ), api_client

    def test_submit_valid_chore(self):
        """submit() should return a task ID for a valid chore type."""
        manager, _ = self._make_manager()
        task_id = manager.submit(ChoreType.VACUUM, room="kitchen")
        assert task_id is not None

    def test_submit_invalid_chore_returns_none(self):
        """submit() should return None for an unknown chore type."""
        manager, _ = self._make_manager()
        result = manager.submit("INVALID_CHORE")
        assert result is None

    def test_submit_priority_clamped(self):
        """submit() should clamp priority to [1, 10]."""
        manager, _ = self._make_manager()
        task_id = manager.submit(ChoreType.VACUUM, priority=999)
        assert task_id is not None
        task = manager._queue.peek()
        assert task["priority"] == 10

    def test_get_next_task_returns_task(self):
        """get_next_task() should return the highest-priority queued task."""
        manager, _ = self._make_manager()
        manager.submit(ChoreType.VACUUM, priority=5)
        task = manager.get_next_task()
        assert task is not None
        assert task["chore_type"] == ChoreType.VACUUM

    def test_get_next_task_empty_returns_none(self):
        """get_next_task() on an empty queue should return None."""
        manager, _ = self._make_manager()
        assert manager.get_next_task() is None

    def test_mark_completed_reports_to_api(self):
        """mark_completed() should call api_client.report_task_status."""
        manager, api = self._make_manager()
        manager.submit(ChoreType.VACUUM)
        task = manager.get_next_task()
        manager.mark_completed(task["id"])
        api.report_task_status.assert_called_once_with(
            task["id"], TaskStatus.COMPLETED, "Task completed successfully."
        )

    def test_mark_failed_requeues_within_retry_limit(self):
        """mark_failed() should re-queue the task if under the retry limit."""
        manager, _ = self._make_manager()
        manager.submit(ChoreType.VACUUM)
        task = manager.get_next_task()
        manager.mark_failed(task["id"], reason="test failure")
        # Task should be back in the queue
        assert manager.queue_size == 1

    def test_submit_from_voice_vacuum(self):
        """submit_from_voice() should parse a vacuum command."""
        manager, _ = self._make_manager()
        task_id = manager.submit_from_voice("River, have Kova vacuum the living room")
        assert task_id is not None
        task = manager._queue.peek()
        assert task["chore_type"] == "VACUUM"
        assert task["room"] == "living_room"

    def test_submit_from_voice_fetch(self):
        """submit_from_voice() should parse a fetch command."""
        manager, _ = self._make_manager()
        task_id = manager.submit_from_voice("River, have Kova fetch my phone from the bedroom")
        assert task_id is not None
        task = manager._queue.peek()
        assert task["chore_type"] == "FETCH"

    def test_submit_from_voice_unknown_returns_none(self):
        """submit_from_voice() should return None for unrecognised commands."""
        manager, _ = self._make_manager()
        result = manager.submit_from_voice("River, do something weird")
        assert result is None

    def test_cancel_task_removes_from_queue(self):
        """cancel_task() should remove the task from the queue."""
        manager, _ = self._make_manager()
        task_id = manager.submit(ChoreType.VACUUM)
        result = manager.cancel_task(task_id)
        assert result is True
        assert manager.queue_size == 0
