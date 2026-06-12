#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : tasks/task_manager.py
Purpose     : Task lifecycle manager. Accepts tasks from River Song voice
              commands and the API, validates them against the chore library,
              enqueues them, and tracks their status. Provides the main
              controller with the next task to execute.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from connectivity.api_client import RiverSongAPIClient
from core.constants import TASK_RETRY_LIMIT, TaskStatus
from tasks.chore_library import ChoreLibrary
from tasks.task_queue import TaskQueue, TaskQueueFullError

log = logging.getLogger(__name__)


class TaskManager:
    """
    Task lifecycle manager for a Kova unit.

    Responsibilities:
    - Accept task requests from River Song API and voice commands
    - Validate tasks against the chore library
    - Enqueue tasks with priority
    - Track task status and retry counts
    - Report status back to River Song

    Attributes
    ----------
    robot_id : str
        Unit identifier.
    active_task : dict or None
        The task currently being executed.
    """

    def __init__(
        self,
        task_queue: TaskQueue,
        chore_library: ChoreLibrary,
        api_client: RiverSongAPIClient,
        robot_id: str,
        capabilities: Optional[set] = None,
    ) -> None:
        """
        Initialise the task manager.

        Parameters
        ----------
        task_queue : TaskQueue
            The priority queue to enqueue tasks into.
        chore_library : ChoreLibrary
            Registry of valid chore definitions.
        api_client : RiverSongAPIClient
            Used to report task status back to River Song.
        robot_id : str
            Unique unit identifier.
        capabilities : set, optional
            Capability flags this robot body declares. Chores whose
            'required_capabilities' aren't covered are rejected at submit
            time. None means accept everything (backward compatible).
        """
        self.robot_id = robot_id
        self._queue = task_queue
        self._library = chore_library
        self._api = api_client
        self._capabilities = capabilities

        self._active_task: Optional[Dict[str, Any]] = None
        self._task_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        log.info("TaskManager ready for unit '%s'.", robot_id)

    # ── State ─────────────────────────────────────────────────────────────────

    @property
    def active_task(self) -> Optional[Dict[str, Any]]:
        """The task currently being executed, or None."""
        with self._lock:
            return self._active_task

    @property
    def queue_size(self) -> int:
        """Number of tasks waiting in the queue."""
        return self._queue.size

    # ── Task submission ───────────────────────────────────────────────────────

    def submit(
        self,
        chore_type: str,
        room: Optional[str] = None,
        priority: int = 5,
        params: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Submit a new chore task.

        Validates the chore type, builds the task descriptor, and enqueues it.

        Parameters
        ----------
        chore_type : str
            ChoreType value (e.g. 'VACUUM', 'FETCH').
        room : str, optional
            Target room name (e.g. 'kitchen', 'living_room').
        priority : int
            Task priority 1–10 (higher = sooner). Default 5.
        params : dict, optional
            Additional parameters merged into the task.
        task_id : str, optional
            Caller-supplied ID (e.g. from River Song). Auto-generated if None.

        Returns
        -------
        str or None
            The task ID if enqueued successfully, None on failure.
        """
        chore = self._library.get(chore_type)
        if chore is None:
            log.error("TaskManager.submit: unknown chore type '%s'.", chore_type)
            return None

        # Capability gate — this body must declare everything the chore needs
        if self._capabilities is not None:
            missing = [
                cap for cap in chore.get("required_capabilities", [])
                if cap not in self._capabilities
            ]
            if missing:
                log.error(
                    "TaskManager.submit: chore '%s' rejected — unit '%s' lacks "
                    "capabilities %s.",
                    chore_type, self.robot_id, missing,
                )
                return None

        task: Dict[str, Any] = {
            "id": task_id or str(uuid.uuid4()),
            "chore_type": chore_type,
            "name": chore["name"],
            "description": chore.get("description", ""),
            "steps": list(chore["steps"]),
            "room": room,
            "priority": max(1, min(10, priority)),
            "status": TaskStatus.QUEUED,
            "retry_count": 0,
            "submitted_at": time.time(),
            "params": params or {},
        }

        try:
            self._queue.enqueue(task)
            log.info(
                "Task submitted: id=%s chore='%s' room='%s' priority=%d.",
                task["id"], chore_type, room, priority,
            )
            return task["id"]
        except TaskQueueFullError as exc:
            log.error("TaskManager.submit: queue full — %s", exc)
            return None

    def submit_from_voice(self, command: str) -> Optional[str]:
        """
        Parse a River Song voice command and submit the corresponding task.

        Supported patterns:
          "River, have Kova clean the kitchen"
          "River, have Kova fetch my phone from the bedroom"
          "River, have Kova vacuum the living room"

        Parameters
        ----------
        command : str
            Raw voice command string.

        Returns
        -------
        str or None
            Task ID if a matching chore was found and submitted.
        """
        command_lower = command.lower()
        log.info("TaskManager: parsing voice command: '%s'", command)

        # Simple keyword mapping — order matters: more specific phrases
        # (water, feed, dog) must match before generic verbs (get, bring).
        chore_keywords = {
            "water": "GET_WATER",
            "feed": "FEED_DOGS",
            "dog": "FEED_DOGS",
            "vacuum": "VACUUM",
            "mop": "MOP",
            "clean": "VACUUM",
            "fetch": "FETCH",
            "get": "FETCH",
            "bring": "FETCH",
            "organize": "ORGANIZE",
            "organise": "ORGANIZE",
            "tidy": "ORGANIZE",
            "wipe": "WIPE_SURFACE",
            "trash": "TAKE_OUT_TRASH",
            "rubbish": "TAKE_OUT_TRASH",
            "dishwasher": "LOAD_DISHWASHER",
            "dishes": "LOAD_DISHWASHER",
            "laundry": "LAUNDRY_TRANSFER",
        }

        room_keywords = [
            "kitchen", "living room", "bedroom", "bathroom",
            "hallway", "dining room", "office", "garage",
        ]

        chore_type = None
        for keyword, ctype in chore_keywords.items():
            if keyword in command_lower:
                chore_type = ctype
                break

        if chore_type is None:
            log.warning("TaskManager: no chore matched for voice command '%s'.", command)
            return None

        room = None
        for r in room_keywords:
            if r in command_lower:
                room = r.replace(" ", "_")
                break

        return self.submit(chore_type=chore_type, room=room, priority=7)

    # ── Task dispatch ─────────────────────────────────────────────────────────

    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """
        Dequeue and return the next task for execution.

        Returns
        -------
        dict or None
            The next task, or None if the queue is empty.
        """
        task = self._queue.dequeue()
        if task:
            with self._lock:
                self._active_task = task
            log.info("TaskManager: dispatching task id=%s '%s'.", task["id"], task["name"])
        return task

    # ── Status reporting ──────────────────────────────────────────────────────

    def mark_completed(self, task_id: str) -> None:
        """
        Mark a task as completed and report to River Song.

        Parameters
        ----------
        task_id : str
            Task identifier.
        """
        self._update_status(task_id, TaskStatus.COMPLETED, "Task completed successfully.")

    def mark_failed(self, task_id: str, reason: str = "") -> None:
        """
        Mark a task as failed. Retries if under the retry limit.

        Parameters
        ----------
        task_id : str
            Task identifier.
        reason : str
            Failure reason for logging and reporting.
        """
        with self._lock:
            task = self._active_task
            if task and task.get("id") == task_id:
                retry_count = task.get("retry_count", 0)
                if retry_count < TASK_RETRY_LIMIT:
                    task["retry_count"] = retry_count + 1
                    task["status"] = TaskStatus.QUEUED
                    log.warning(
                        "Task id=%s failed (attempt %d/%d) — re-queuing. Reason: %s",
                        task_id, retry_count + 1, TASK_RETRY_LIMIT, reason,
                    )
                    self._queue.enqueue(task)
                    self._active_task = None
                    return

        self._update_status(task_id, TaskStatus.FAILED, reason)

    def mark_aborted(self, task_id: str, reason: str = "aborted") -> None:
        """
        Mark a task as aborted (e.g. due to e-stop or battery critical).

        Parameters
        ----------
        task_id : str
            Task identifier.
        reason : str
            Abort reason.
        """
        self._update_status(task_id, TaskStatus.ABORTED, reason)

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a queued task by ID.

        Parameters
        ----------
        task_id : str
            Task to cancel.

        Returns
        -------
        bool
            True if the task was found and removed.
        """
        removed = self._queue.remove(task_id)
        if removed:
            self._api.report_task_status(task_id, TaskStatus.ABORTED, "Cancelled by operator.")
        return removed

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _update_status(self, task_id: str, status: str, message: str) -> None:
        """Update active task status, move to history, and report to API."""
        with self._lock:
            if self._active_task and self._active_task.get("id") == task_id:
                self._active_task["status"] = status
                self._active_task["completed_at"] = time.time()
                self._task_history.append(dict(self._active_task))
                self._active_task = None

        log.info("Task id=%s status → %s: %s", task_id, status, message)
        self._api.report_task_status(task_id, status, message)
