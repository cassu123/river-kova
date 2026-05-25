#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : tasks/task_queue.py
Purpose     : Thread-safe priority task queue. Tasks are ordered by priority
              (higher = sooner) then by insertion time (FIFO within same
              priority). Supports pause, resume, and drain operations.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.constants import TASK_QUEUE_MAX_SIZE, TaskStatus

log = logging.getLogger(__name__)


@dataclass(order=True)
class _QueueEntry:
    """Internal heap entry — ordered by (-priority, insertion_time)."""

    sort_key: tuple = field(compare=True)
    task: Dict[str, Any] = field(compare=False)


class TaskQueueFullError(Exception):
    """Raised when a task is enqueued but the queue is at capacity."""


class TaskQueue:
    """
    Thread-safe priority task queue.

    Tasks are dicts with at minimum:
      - id       : str  — unique task identifier
      - name     : str  — human-readable name
      - priority : int  — higher value = higher priority (default 5)
      - status   : str  — TaskStatus value

    Attributes
    ----------
    size : int
        Current number of tasks in the queue.
    is_empty : bool
        True when the queue has no pending tasks.
    """

    def __init__(self, max_size: int = TASK_QUEUE_MAX_SIZE) -> None:
        """
        Initialise the task queue.

        Parameters
        ----------
        max_size : int
            Maximum number of tasks the queue will hold.
        """
        self._max_size = max_size
        self._heap: List[_QueueEntry] = []
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    # ── State properties ──────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Current number of tasks in the queue."""
        with self._lock:
            return len(self._heap)

    @property
    def is_empty(self) -> bool:
        """True when the queue has no pending tasks."""
        return self.size == 0

    # ── Queue operations ──────────────────────────────────────────────────────

    def enqueue(self, task: Dict[str, Any]) -> str:
        """
        Add a task to the queue.

        Parameters
        ----------
        task : dict
            Task descriptor. A unique 'id' is assigned if not present.

        Returns
        -------
        str
            The task's unique identifier.

        Raises
        ------
        TaskQueueFullError
            If the queue is at capacity.
        """
        with self._not_empty:
            if len(self._heap) >= self._max_size:
                raise TaskQueueFullError(
                    f"Task queue is full ({self._max_size} tasks)."
                )

            if "id" not in task:
                task["id"] = str(uuid.uuid4())
            task.setdefault("priority", 5)
            task.setdefault("status", TaskStatus.QUEUED)
            task.setdefault("enqueued_at", time.time())

            # Heap key: (-priority, time) so highest priority + earliest time wins
            entry = _QueueEntry(
                sort_key=(-task["priority"], task["enqueued_at"]),
                task=task,
            )
            heapq.heappush(self._heap, entry)
            self._not_empty.notify()

        log.debug(
            "Task enqueued: id=%s name='%s' priority=%d (queue size=%d).",
            task["id"], task.get("name", "?"), task["priority"], len(self._heap),
        )
        return task["id"]

    def dequeue(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Remove and return the highest-priority task.

        Parameters
        ----------
        timeout : float, optional
            Seconds to wait for a task if the queue is empty.
            None = return immediately (non-blocking).

        Returns
        -------
        dict or None
            The next task, or None if the queue is empty (or timeout elapsed).
        """
        with self._not_empty:
            if not self._heap:
                if timeout is None:
                    return None
                self._not_empty.wait(timeout=timeout)
                if not self._heap:
                    return None

            entry = heapq.heappop(self._heap)
            task = entry.task
            task["status"] = TaskStatus.RUNNING

        log.debug("Task dequeued: id=%s name='%s'.", task["id"], task.get("name", "?"))
        return task

    def peek(self) -> Optional[Dict[str, Any]]:
        """
        Return the next task without removing it.

        Returns
        -------
        dict or None
        """
        with self._lock:
            if not self._heap:
                return None
            return self._heap[0].task

    def remove(self, task_id: str) -> bool:
        """
        Remove a specific task by ID (e.g. cancelled by River Song).

        Parameters
        ----------
        task_id : str
            Task identifier to remove.

        Returns
        -------
        bool
            True if the task was found and removed.
        """
        with self._lock:
            for i, entry in enumerate(self._heap):
                if entry.task.get("id") == task_id:
                    self._heap.pop(i)
                    heapq.heapify(self._heap)
                    log.info("Task removed from queue: id=%s.", task_id)
                    return True
        log.warning("TaskQueue.remove: task id '%s' not found.", task_id)
        return False

    def drain(self) -> List[Dict[str, Any]]:
        """
        Remove and return all tasks (e.g. on shutdown or e-stop).

        Returns
        -------
        list of dict
            All tasks that were in the queue.
        """
        with self._lock:
            tasks = [entry.task for entry in self._heap]
            self._heap.clear()
        log.info("TaskQueue drained (%d tasks removed).", len(tasks))
        return tasks

    def list_tasks(self) -> List[Dict[str, Any]]:
        """
        Return a snapshot of all queued tasks (does not remove them).

        Returns
        -------
        list of dict
            Tasks ordered by priority.
        """
        with self._lock:
            return [entry.task for entry in sorted(self._heap)]
