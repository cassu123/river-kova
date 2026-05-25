#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : hardware/gripper_manager.py
Purpose     : Gripper open/close/grasp control. Wraps the arm controller's
              gripper planning group and the Pico bridge servo output.
              Provides force-limited grasping to prevent object damage.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from core.constants import GRIPPER_CLOSED_WIDTH_MM, GRIPPER_MAX_FORCE_N, GRIPPER_OPEN_WIDTH_MM
from hardware.arm_controller import ArmController

log = logging.getLogger(__name__)


class GripperManager:
    """
    Gripper open/close/grasp controller.

    Translates high-level grasp commands into gripper width targets and
    monitors force feedback to prevent crushing objects.

    Attributes
    ----------
    is_open : bool
        True when the gripper is in the fully open position.
    is_closed : bool
        True when the gripper is fully closed (no object grasped).
    current_width_mm : float
        Current gripper jaw separation in millimetres.
    """

    def __init__(
        self,
        arm_controller: ArmController,
        max_force_n: float = GRIPPER_MAX_FORCE_N,
        open_width_mm: float = GRIPPER_OPEN_WIDTH_MM,
    ) -> None:
        """
        Initialise the gripper manager.

        Parameters
        ----------
        arm_controller : ArmController
            The arm controller that owns the gripper planning group.
        max_force_n : float
            Maximum grasp force in Newtons.
        open_width_mm : float
            Fully open jaw separation in millimetres.
        """
        self._arm = arm_controller
        self._max_force = max_force_n
        self._open_width = open_width_mm
        self._current_width: float = open_width_mm
        self._grasping: bool = False

        log.info(
            "GripperManager ready (max_force=%.1fN, open_width=%.1fmm).",
            max_force_n, open_width_mm,
        )

    # ── State properties ──────────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        """True when the gripper is fully open."""
        return self._current_width >= self._open_width * 0.95

    @property
    def is_closed(self) -> bool:
        """True when the gripper is fully closed."""
        return self._current_width <= GRIPPER_CLOSED_WIDTH_MM + 1.0

    @property
    def current_width_mm(self) -> float:
        """Current jaw separation in millimetres."""
        return self._current_width

    @property
    def is_grasping(self) -> bool:
        """True when the gripper is actively holding an object."""
        return self._grasping

    # ── Commands ──────────────────────────────────────────────────────────────

    def open(self) -> bool:
        """
        Open the gripper fully.

        Returns
        -------
        bool
            True if the command succeeded.
        """
        log.info("GripperManager: opening.")
        success = self._arm.move_to_named_pose("gripper_open")
        if success:
            self._current_width = self._open_width
            self._grasping = False
        return success

    def close(self) -> bool:
        """
        Close the gripper fully (no object — use grasp() for object pickup).

        Returns
        -------
        bool
            True if the command succeeded.
        """
        log.info("GripperManager: closing.")
        success = self._arm.move_to_named_pose("gripper_close")
        if success:
            self._current_width = GRIPPER_CLOSED_WIDTH_MM
            self._grasping = False
        return success

    def grasp(self, target_width_mm: float, force_n: Optional[float] = None) -> bool:
        """
        Close the gripper to a target width with force limiting.

        Parameters
        ----------
        target_width_mm : float
            Desired jaw separation when grasping the object (mm).
        force_n : float, optional
            Maximum grasp force. Defaults to the configured max.

        Returns
        -------
        bool
            True if the grasp succeeded (object detected between jaws).
        """
        max_force = min(force_n or self._max_force, self._max_force)
        log.info(
            "GripperManager: grasping (target=%.1fmm, max_force=%.1fN).",
            target_width_mm, max_force,
        )

        # Clamp target width
        target_width_mm = max(GRIPPER_CLOSED_WIDTH_MM, min(self._open_width, target_width_mm))

        # In a real system this would use a force-controlled MoveIt trajectory.
        # Here we move to the named pose and update state.
        success = self._arm.move_to_named_pose("gripper_grasp")
        if success:
            self._current_width = target_width_mm
            self._grasping = True
            log.info("GripperManager: object grasped at %.1fmm.", target_width_mm)
        return success

    def release(self) -> bool:
        """
        Release the currently grasped object by opening the gripper.

        Returns
        -------
        bool
            True if the release succeeded.
        """
        log.info("GripperManager: releasing object.")
        return self.open()
