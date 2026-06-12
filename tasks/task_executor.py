#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : tasks/task_executor.py
Purpose     : Step-by-step chore executor. Iterates through a task's step
              list, dispatches each action to the appropriate subsystem, and
              handles timeouts, retries, and abort signals.
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
from typing import Any, Callable, Dict, Optional

from core.constants import TASK_STEP_TIMEOUT_SEC, TaskStatus
from hardware.arm_controller import ArmController
from hardware.drive_controller import DriveController
from hardware.gripper_manager import GripperManager
from navigation.path_planner import PathPlanner
from tasks.task_manager import TaskManager
from vision.object_detect import ObjectDetector

log = logging.getLogger(__name__)


class TaskExecutionError(Exception):
    """Raised when a task step fails and cannot be retried."""


class TaskExecutor:
    """
    Executes chore tasks step by step.

    Each step's 'action' string is dispatched to a registered handler.
    Handlers are methods on this class prefixed with '_action_'.
    New actions can be added by subclassing or registering handlers.

    Attributes
    ----------
    is_executing : bool
        True while a task is actively running.
    current_task_id : str or None
        ID of the task currently being executed.
    """

    def __init__(
        self,
        task_manager: TaskManager,
        drive: DriveController,
        arm: ArmController,
        gripper: GripperManager,
        navigation: PathPlanner,
        vision: ObjectDetector,
        safety_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        """
        Initialise the task executor.

        Parameters
        ----------
        task_manager : TaskManager
            Used to report step/task status.
        drive : DriveController
            Drive controller for navigation steps.
        arm : ArmController
            Arm controller for manipulation steps.
        gripper : GripperManager
            Gripper controller.
        navigation : PathPlanner
            Path planner for waypoint navigation.
        vision : ObjectDetector
            Object detector for detection steps.
        safety_check : callable, optional
            Returns True when motion is permitted. Checked before each step.
        """
        self._task_manager = task_manager
        self._drive = drive
        self._arm = arm
        self._gripper = gripper
        self._navigation = navigation
        self._vision = vision
        self._safety_check = safety_check

        self._executing: bool = False
        self._abort_flag: bool = False
        self._current_task_id: Optional[str] = None
        self._lock = threading.Lock()

    # ── State ─────────────────────────────────────────────────────────────────

    @property
    def is_executing(self) -> bool:
        """True while a task is actively running."""
        with self._lock:
            return self._executing

    @property
    def current_task_id(self) -> Optional[str]:
        """ID of the task currently being executed."""
        with self._lock:
            return self._current_task_id

    # ── Execution ─────────────────────────────────────────────────────────────

    def execute(self, task: Dict[str, Any]) -> bool:
        """
        Execute a task by running its steps in sequence.

        Parameters
        ----------
        task : dict
            Task descriptor from the task manager.

        Returns
        -------
        bool
            True if all steps completed successfully.
        """
        task_id = task.get("id", "unknown")
        task_name = task.get("name", "unknown")
        steps = task.get("steps", [])

        with self._lock:
            self._executing = True
            self._abort_flag = False
            self._current_task_id = task_id

        log.info("TaskExecutor: starting task id=%s '%s' (%d steps).", task_id, task_name, len(steps))

        try:
            for step_index, step in enumerate(steps):
                if self._abort_flag:
                    log.warning("TaskExecutor: task id=%s aborted at step %d.", task_id, step_index)
                    self._task_manager.mark_aborted(task_id, "Abort requested.")
                    return False

                if self._safety_check and not self._safety_check():
                    log.warning(
                        "TaskExecutor: safety check failed at step %d — pausing.", step_index
                    )
                    self._wait_for_safe(timeout=30.0)
                    if not self._safety_check():
                        self._task_manager.mark_aborted(task_id, "Safety check failed.")
                        return False

                success = self._execute_step(step, step_index, task)
                if not success:
                    self._task_manager.mark_failed(task_id, f"Step {step_index} failed.")
                    return False

            self._task_manager.mark_completed(task_id)
            log.info("TaskExecutor: task id=%s '%s' COMPLETED.", task_id, task_name)
            return True

        except Exception as exc:
            log.error("TaskExecutor: unhandled error in task id=%s: %s", task_id, exc, exc_info=True)
            self._task_manager.mark_failed(task_id, str(exc))
            return False

        finally:
            with self._lock:
                self._executing = False
                self._current_task_id = None

    def abort_current(self) -> None:
        """
        Signal the executor to abort the current task at the next step boundary.

        Thread-safe — can be called from any thread.
        """
        with self._lock:
            self._abort_flag = True
        log.warning("TaskExecutor: abort requested.")

    # ── Step dispatch ─────────────────────────────────────────────────────────

    def _execute_step(
        self,
        step: Dict[str, Any],
        step_index: int,
        task: Dict[str, Any],
    ) -> bool:
        """
        Execute a single task step with timeout and retry handling.

        Parameters
        ----------
        step : dict
            Step descriptor.
        step_index : int
            Index of this step in the task's step list.
        task : dict
            Parent task descriptor (for context).

        Returns
        -------
        bool
            True if the step succeeded.
        """
        action = step.get("action", "")
        params = step.get("params", {})
        timeout = step.get("timeout_sec", TASK_STEP_TIMEOUT_SEC)
        max_retries = step.get("retry", 0)

        handler_name = f"_action_{action}"
        handler = getattr(self, handler_name, None)

        if handler is None:
            log.error("TaskExecutor: no handler for action '%s' (step %d).", action, step_index)
            return False

        for attempt in range(max_retries + 1):
            if attempt > 0:
                log.info("TaskExecutor: retrying step %d '%s' (attempt %d).", step_index, action, attempt + 1)
                time.sleep(2.0)

            try:
                log.debug("TaskExecutor: step %d '%s' params=%s.", step_index, action, params)
                result = self._run_with_timeout(handler, params, timeout)
                if result:
                    return True
            except Exception as exc:
                log.warning("TaskExecutor: step %d '%s' error: %s", step_index, action, exc)

        log.error("TaskExecutor: step %d '%s' failed after %d attempts.", step_index, action, max_retries + 1)
        return False

    def _run_with_timeout(
        self,
        handler: Callable,
        params: Dict[str, Any],
        timeout: float,
    ) -> bool:
        """
        Run a step handler in a thread with a timeout.

        Parameters
        ----------
        handler : callable
            Step handler method.
        params : dict
            Parameters to pass to the handler.
        timeout : float
            Maximum seconds to wait.

        Returns
        -------
        bool
            True if the handler returned True within the timeout.
        """
        result_container = [False]
        exception_container = [None]

        def target():
            try:
                result_container[0] = handler(params)
            except Exception as exc:
                exception_container[0] = exc

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            log.error("TaskExecutor: step timed out after %.1fs.", timeout)
            return False

        if exception_container[0]:
            raise exception_container[0]

        return result_container[0]

    # ── Action handlers ───────────────────────────────────────────────────────

    def _action_arm_pose(self, params: Dict[str, Any]) -> bool:
        """Move the arm to a named pose."""
        pose = params.get("pose", "home")
        return self._arm.move_to_named_pose(pose)

    def _action_gripper_open(self, params: Dict[str, Any]) -> bool:
        """Open the gripper."""
        return self._gripper.open()

    def _action_gripper_close(self, params: Dict[str, Any]) -> bool:
        """Close the gripper."""
        return self._gripper.close()

    def _action_arm_grasp(self, params: Dict[str, Any]) -> bool:
        """Grasp an object with the gripper."""
        width_mm = params.get("width_mm", 40.0)
        force_n = params.get("force_n", 8.0)
        return self._gripper.grasp(target_width_mm=width_mm, force_n=force_n)

    def _action_navigate_to_waypoint(self, params: Dict[str, Any]) -> bool:
        """Navigate to a named waypoint."""
        waypoint = params.get("waypoint", "base")
        return self._navigation.navigate_to_waypoint(waypoint)

    def _action_navigate_to_base(self, params: Dict[str, Any]) -> bool:
        """Navigate back to the base station."""
        return self._navigation.navigate_to_waypoint("base")

    def _action_navigate_coverage(self, params: Dict[str, Any]) -> bool:
        """Execute a coverage navigation pattern."""
        pattern = params.get("pattern", "boustrophedon")
        speed_factor = params.get("speed_factor", 1.0)
        return self._navigation.execute_coverage(pattern=pattern, speed_factor=speed_factor)

    def _action_navigate_to_object(self, params: Dict[str, Any]) -> bool:
        """Navigate to the last detected object."""
        return self._navigation.navigate_to_last_detection()

    def _action_detect_object(self, params: Dict[str, Any]) -> bool:
        """Run object detection and store the result for subsequent steps."""
        target_class = params.get("target_class")
        detections = self._vision.detect(target_class=target_class)
        return len(detections) > 0

    def _action_scan_room(self, params: Dict[str, Any]) -> bool:
        """Rotate in place to scan the room for objects."""
        target_classes = params.get("target_classes", [])
        return self._navigation.scan_room(target_classes=target_classes)

    def _action_arm_wipe_motion(self, params: Dict[str, Any]) -> bool:
        """Execute a wiping motion with the arm."""
        passes = params.get("passes", 3)
        width_m = params.get("width_m", 0.4)
        for _ in range(passes):
            if not self._arm.move_to_named_pose("wipe_left"):
                return False
            if not self._arm.move_to_named_pose("wipe_right"):
                return False
        return True

    def _action_arm_pour_motion(self, params: Dict[str, Any]) -> bool:
        """Tilt the held container to pour its contents (e.g. dog food scoop)."""
        if not self._arm.move_to_named_pose("pour_ready"):
            return False
        if not self._arm.move_to_named_pose("pour_tilt"):
            return False
        return self._arm.move_to_named_pose("pour_ready")

    def _action_organize_loop(self, params: Dict[str, Any]) -> bool:
        """Stub for the organize loop — implemented by a higher-level planner."""
        log.info("TaskExecutor: organize_loop (stub) — max_items=%s.", params.get("max_items"))
        return True

    def _action_load_dishwasher_loop(self, params: Dict[str, Any]) -> bool:
        """Stub for the load dishwasher loop."""
        log.info("TaskExecutor: load_dishwasher_loop (stub).")
        return True

    def _action_unload_dishwasher_loop(self, params: Dict[str, Any]) -> bool:
        """Stub for the unload dishwasher loop."""
        log.info("TaskExecutor: unload_dishwasher_loop (stub).")
        return True

    def _action_transfer_laundry(self, params: Dict[str, Any]) -> bool:
        """Stub for the laundry transfer action."""
        log.info("TaskExecutor: transfer_laundry (stub).")
        return True

    def _action_open_appliance_door(self, params: Dict[str, Any]) -> bool:
        """Open an appliance door (washer, dryer, dishwasher)."""
        appliance = params.get("appliance", "unknown")
        log.info("TaskExecutor: open_appliance_door '%s' (stub).", appliance)
        return True

    def _action_close_appliance_door(self, params: Dict[str, Any]) -> bool:
        """Close an appliance door."""
        appliance = params.get("appliance", "unknown")
        log.info("TaskExecutor: close_appliance_door '%s' (stub).", appliance)
        return True

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _wait_for_safe(self, timeout: float = 30.0) -> None:
        """
        Block until the safety check passes or timeout elapses.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._safety_check and self._safety_check():
                return
            time.sleep(0.5)
        log.warning("TaskExecutor: timed out waiting for safe condition.")
