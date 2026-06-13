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

import copy
import logging
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from connectivity.api_client import RiverSongAPIClient
from core.constants import TASK_RETRY_LIMIT, TaskStatus
from tasks.chore_library import ChoreLibrary
from tasks.task_queue import TaskQueue, TaskQueueFullError

log = logging.getLogger(__name__)

# Voice command → chore mapping. Patterns are tried top to bottom and the
# first match wins, so multi-word phrases ("unload the dishwasher", "get me
# some water") must come before the generic verbs that would shadow them
# ("get" → FETCH). All matching is on word boundaries: "getting" never
# matches "get".
_VOICE_CHORE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(pattern), chore) for pattern, chore in [
        (r"\b(?:get|bring|fetch|grab)\b.*\bwater\b", "GET_WATER"),
        (r"\bwater\s+bottle\b", "GET_WATER"),
        (r"\bfeed\b.*\b(?:dog|dogs|puppy|puppies)\b", "FEED_DOGS"),
        (r"\b(?:dog|dogs)\b.*\b(?:food|dinner|breakfast)\b", "FEED_DOGS"),
        (r"\bfeed\b", "FEED_DOGS"),
        (r"\bunload\b.*\b(?:dishwasher|dishes)\b", "UNLOAD_DISHWASHER"),
        (r"\b(?:dishwasher|dishes|dish)\b", "LOAD_DISHWASHER"),
        (r"\blaundry\b|\bwasher\b.*\bdryer\b", "LAUNDRY_TRANSFER"),
        (r"\b(?:trash|rubbish|garbage)\b", "TAKE_OUT_TRASH"),
        (r"\bvacuum\b", "VACUUM"),
        (r"\bmop\b", "MOP"),
        (r"\bwipe\b", "WIPE_SURFACE"),
        (r"\b(?:organize|organise|tidy)\b", "ORGANIZE"),
        (r"\bexplore\b|\blearn\b.*\b(?:house|home|map|layout)\b", "EXPLORE"),
        (r"\bclean\b", "VACUUM"),
        (r"\bwater\b", "GET_WATER"),
        (r"\b(?:fetch|get|bring|grab)\b", "FETCH"),
    ]
]

_VOICE_ROOM_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + name.replace("_", r"\s+") + r"\b"), name) for name in [
        "kitchen", "living_room", "bedroom", "bathroom",
        "hallway", "dining_room", "office", "garage",
    ]
]


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

        chore_type = None
        for pattern, ctype in _VOICE_CHORE_PATTERNS:
            if pattern.search(command_lower):
                chore_type = ctype
                break

        room = None
        for pattern, name in _VOICE_ROOM_PATTERNS:
            if pattern.search(command_lower):
                room = name
                break

        # Local keyword parsing handles the common phrases for free. Anything
        # it can't match escalates to the River Song server's language model —
        # the robot never reaches an online model itself, and an offline /
        # self-hosted unit simply gets None back and reports no match.
        if chore_type is None:
            interpreted = self._api.interpret_command(command)
            if interpreted:
                chore_type = interpreted.get("chore_type")
                room = interpreted.get("room") or room
                log.info("TaskManager: River Song interpreted '%s' as %s.", command, chore_type)

        if chore_type is None:
            log.warning("TaskManager: no chore matched for voice command '%s'.", command)
            return None

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

    def requeue(self, task: Dict[str, Any]) -> bool:
        """
        Put a dequeued-but-not-started task back in the queue.

        Used when dispatch is cancelled after dequeue (e.g. a safety event
        landed between the controller's idle check and the actual start).

        Parameters
        ----------
        task : dict
            The task descriptor previously returned by get_next_task().

        Returns
        -------
        bool
            True if the task went back in the queue.
        """
        with self._lock:
            if self._active_task is task:
                self._active_task = None
        task["status"] = TaskStatus.QUEUED
        try:
            self._queue.enqueue(task)
            return True
        except TaskQueueFullError:
            log.error("TaskManager.requeue: queue full — task id=%s dropped.", task.get("id"))
            return False

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
                    # Re-queue a fresh copy: the executor still holds a
                    # reference to the original dict, so sharing it would
                    # let late mutations corrupt the queued attempt.
                    retry = copy.deepcopy(task)
                    retry["retry_count"] = retry_count + 1
                    retry["status"] = TaskStatus.QUEUED
                    try:
                        self._queue.enqueue(retry)
                    except TaskQueueFullError:
                        log.error(
                            "Task id=%s retry dropped — queue full.", task_id,
                        )
                    else:
                        log.warning(
                            "Task id=%s failed (attempt %d/%d) — re-queuing. Reason: %s",
                            task_id, retry_count + 1, TASK_RETRY_LIMIT, reason,
                        )
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
