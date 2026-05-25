#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : hardware/arm_controller.py
Purpose     : ROS2 MoveIt arm controller interface. Wraps MoveIt motion
              planning and execution for the 6-DOF manipulator arm.
              Provides high-level pose/joint commands with safety guards.
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
from typing import Callable, Dict, List, Optional, Tuple

from core.constants import ARM_DOF, ARM_MAX_PAYLOAD_KG

log = logging.getLogger(__name__)

try:
    import rclpy
    from moveit.planning import MoveItPy
    _MOVEIT_AVAILABLE = True
except ImportError:
    _MOVEIT_AVAILABLE = False
    log.warning("MoveIt not available — ArmController running in stub mode.")


class ArmControllerError(Exception):
    """Raised when an arm motion command fails."""


class ArmController:
    """
    High-level interface to the ROS2 MoveIt arm controller.

    Wraps MoveItPy to provide named pose moves, Cartesian moves, and
    joint-space moves. All commands are guarded by a safety check callback
    and a payload limit.

    Attributes
    ----------
    arm_type : str
        Arm configuration identifier (e.g. '6-dof').
    is_halted : bool
        True when the arm is in a halted/e-stopped state.
    """

    def __init__(
        self,
        arm_type: str = "6-dof",
        controller: str = "moveit",
        max_payload_kg: float = ARM_MAX_PAYLOAD_KG,
        safety_check: Optional[Callable[[], bool]] = None,
        planning_group: str = "arm",
    ) -> None:
        """
        Initialise the arm controller.

        Parameters
        ----------
        arm_type : str
            Arm configuration identifier.
        controller : str
            Motion planning backend ('moveit').
        max_payload_kg : float
            Maximum safe payload in kilograms.
        safety_check : callable, optional
            Returns True when motion is permitted.
        planning_group : str
            MoveIt planning group name.
        """
        self.arm_type = arm_type
        self._controller = controller
        self._max_payload = max_payload_kg
        self._safety_check = safety_check
        self._planning_group = planning_group

        self._halted: bool = False
        self._lock = threading.Lock()
        self._moveit: Optional[object] = None

        if _MOVEIT_AVAILABLE:
            try:
                self._moveit = MoveItPy(node_name="kova_arm_moveit")
                self._arm_group = self._moveit.get_planning_component(planning_group)
                log.info("ArmController: MoveIt connected (group='%s').", planning_group)
            except Exception as exc:
                log.error("ArmController: MoveIt init failed: %s", exc)
        else:
            log.warning("ArmController: stub mode — no real arm motion.")

        log.info(
            "ArmController ready (%s, max_payload=%.1f kg).",
            arm_type, max_payload_kg,
        )

    # ── State ─────────────────────────────────────────────────────────────────

    @property
    def is_halted(self) -> bool:
        """True when the arm is halted and will not accept motion commands."""
        with self._lock:
            return self._halted

    # ── Motion commands ───────────────────────────────────────────────────────

    def move_to_named_pose(self, pose_name: str) -> bool:
        """
        Move the arm to a predefined named pose (e.g. 'home', 'ready', 'stow').

        Parameters
        ----------
        pose_name : str
            Name of the pose as defined in the SRDF.

        Returns
        -------
        bool
            True if the motion completed successfully.
        """
        if not self._check_safe():
            return False

        log.info("ArmController: moving to named pose '%s'.", pose_name)

        if not _MOVEIT_AVAILABLE or self._moveit is None:
            log.debug("ArmController STUB: move_to_named_pose('%s').", pose_name)
            return True

        try:
            self._arm_group.set_start_state_to_current_state()
            self._arm_group.set_goal_state(configuration_name=pose_name)
            plan_result = self._arm_group.plan()
            if not plan_result:
                log.error("ArmController: planning failed for pose '%s'.", pose_name)
                return False
            self._moveit.execute(plan_result, controllers=[])
            log.info("ArmController: reached named pose '%s'.", pose_name)
            return True
        except Exception as exc:
            log.error("ArmController.move_to_named_pose error: %s", exc)
            return False

    def move_to_pose(
        self,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float],
    ) -> bool:
        """
        Move the end-effector to a Cartesian pose.

        Parameters
        ----------
        position : tuple of float
            (x, y, z) in metres, in the robot base frame.
        orientation : tuple of float
            (qx, qy, qz, qw) quaternion.

        Returns
        -------
        bool
            True if the motion completed successfully.
        """
        if not self._check_safe():
            return False

        log.info("ArmController: moving to pose %s.", position)

        if not _MOVEIT_AVAILABLE or self._moveit is None:
            log.debug("ArmController STUB: move_to_pose(%s).", position)
            return True

        try:
            from geometry_msgs.msg import Pose
            target_pose = Pose()
            target_pose.position.x, target_pose.position.y, target_pose.position.z = position
            target_pose.orientation.x, target_pose.orientation.y = orientation[0], orientation[1]
            target_pose.orientation.z, target_pose.orientation.w = orientation[2], orientation[3]

            self._arm_group.set_start_state_to_current_state()
            self._arm_group.set_goal_state(pose_stamped_msg=target_pose, pose_link="end_effector")
            plan_result = self._arm_group.plan()
            if not plan_result:
                log.error("ArmController: planning failed for Cartesian pose.")
                return False
            self._moveit.execute(plan_result, controllers=[])
            return True
        except Exception as exc:
            log.error("ArmController.move_to_pose error: %s", exc)
            return False

    def move_joints(self, joint_positions: List[float]) -> bool:
        """
        Move the arm to specific joint positions.

        Parameters
        ----------
        joint_positions : list of float
            Target joint angles in radians. Must have ARM_DOF elements.

        Returns
        -------
        bool
            True if the motion completed successfully.

        Raises
        ------
        ValueError
            If the number of joint positions does not match ARM_DOF.
        """
        if len(joint_positions) != ARM_DOF:
            raise ValueError(
                f"Expected {ARM_DOF} joint positions, got {len(joint_positions)}."
            )
        if not self._check_safe():
            return False

        log.info("ArmController: moving to joint positions %s.", joint_positions)

        if not _MOVEIT_AVAILABLE or self._moveit is None:
            log.debug("ArmController STUB: move_joints(%s).", joint_positions)
            return True

        try:
            robot_state = self._moveit.get_robot_state()
            robot_state.set_joint_group_positions(self._planning_group, joint_positions)
            self._arm_group.set_start_state_to_current_state()
            self._arm_group.set_goal_state(robot_state=robot_state)
            plan_result = self._arm_group.plan()
            if not plan_result:
                log.error("ArmController: joint planning failed.")
                return False
            self._moveit.execute(plan_result, controllers=[])
            return True
        except Exception as exc:
            log.error("ArmController.move_joints error: %s", exc)
            return False

    def halt(self) -> None:
        """
        Immediately halt all arm motion.

        Sets the halted flag — no further commands will be accepted until
        clear_halt() is called.
        """
        with self._lock:
            self._halted = True
        log.warning("ArmController: HALTED.")
        if _MOVEIT_AVAILABLE and self._moveit:
            try:
                self._moveit.stop()
            except Exception as exc:
                log.error("ArmController halt MoveIt stop error: %s", exc)

    def clear_halt(self) -> None:
        """Clear the halt flag, allowing motion to resume."""
        with self._lock:
            self._halted = False
        log.info("ArmController: halt cleared.")

    def go_home(self) -> bool:
        """
        Move the arm to the 'home' stow position.

        Returns
        -------
        bool
            True if successful.
        """
        return self.move_to_named_pose("home")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_safe(self) -> bool:
        """
        Check whether motion is currently permitted.

        Returns
        -------
        bool
            False if halted or safety check fails.
        """
        with self._lock:
            if self._halted:
                log.warning("ArmController: command rejected — arm is halted.")
                return False
        if self._safety_check and not self._safety_check():
            log.warning("ArmController: command rejected — safety check failed.")
            return False
        return True
