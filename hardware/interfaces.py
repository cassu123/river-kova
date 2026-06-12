#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : hardware/interfaces.py
Purpose     : Hardware Abstraction Layer (HAL) contracts. Defines the abstract
              interfaces every robot body must satisfy. The brain (safety,
              navigation, tasks, autonomy) only ever talks to these
              interfaces — never to a concrete robot.

              PORTING A NEW ROBOT BODY
              ------------------------
              1. Implement an I/O bridge satisfying IOBridge (or subclass
                 PicoBridge if the body uses the same JSON-serial protocol).
              2. Add a backend value in core/constants.HardwareBackend.
              3. Add a branch in hardware/factory.build_hardware().
              4. Write a unit profile JSON declaring the body's capabilities.
              Nothing else in the system changes.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-06-12
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class IOBridge(ABC):
    """
    Low-level I/O contract: motors, battery ADC, encoders, force, gripper.

    Concrete implementations: hardware.pico_bridge.PicoBridge (serial),
    simulation.sim_pico_bridge.SimPicoBridge (simulated).
    """

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """True when the bridge is live."""

    @abstractmethod
    def connect(self) -> None:
        """Open the link to the body. Raises on failure."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the link."""

    @abstractmethod
    def set_motor_speeds(self, left: float, right: float) -> None:
        """Set drive motor fractions in [-1.0, 1.0]."""

    @abstractmethod
    def emergency_stop_motors(self) -> None:
        """Immediate motor cut — must bypass all ramping."""

    @abstractmethod
    def read_battery(self) -> float:
        """Battery percentage [0.0, 100.0]."""

    @abstractmethod
    def read_encoders(self) -> Dict[str, int]:
        """Wheel encoder tick counts {'left': int, 'right': int}."""

    @abstractmethod
    def read_force_sensor(self) -> float:
        """Collision/contact force in Newtons."""

    @abstractmethod
    def set_gripper(self, position: float) -> None:
        """Gripper position [0.0 = closed, 1.0 = open]."""


class Manipulator(ABC):
    """
    Arm contract. Bodies without an arm simply don't declare the
    'arm_manipulation' capability and never receive arm steps.

    Concrete implementation: hardware.arm_controller.ArmController.
    """

    @property
    @abstractmethod
    def is_halted(self) -> bool:
        """True when the arm refuses motion commands."""

    @abstractmethod
    def move_to_named_pose(self, pose_name: str) -> bool:
        """Move to a named pose ('home', 'ready', 'carry', 'stow', ...)."""

    @abstractmethod
    def move_to_pose(
        self,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float],
    ) -> bool:
        """Move the end-effector to a Cartesian pose."""

    @abstractmethod
    def halt(self) -> None:
        """Immediately stop all arm motion."""


class FrameSource(ABC):
    """
    Camera contract — RGB and optional depth frames for the vision pipeline.

    Concrete implementation: hardware.camera_manager.CameraManager.
    """

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """True when the camera is streaming."""

    @abstractmethod
    def open(self) -> None:
        """Start streaming. Raises on failure."""

    @abstractmethod
    def close(self) -> None:
        """Stop streaming and release the device."""

    @abstractmethod
    def capture(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Capture (rgb_frame, depth_frame); depth may be None."""
