#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : hardware/factory.py
Purpose     : Hardware backend factory. Reads `hardware.backend` from the unit
              profile and assembles the matching set of drivers — real serial
              hardware, full simulation, or any future robot body adapter.
              This is the single point where a physical platform is chosen;
              everything above it is hardware-agnostic.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-06-12
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from core.constants import HardwareBackend
from hardware.arm_controller import ArmController
from hardware.camera_manager import CameraManager
from hardware.drive_controller import DriveController
from hardware.gripper_manager import GripperManager
from hardware.pico_bridge import PicoBridge

log = logging.getLogger(__name__)


@dataclass
class HardwareSet:
    """The assembled hardware drivers for one robot body."""

    backend: str
    bridge: PicoBridge
    drive: DriveController
    arm: ArmController
    gripper: GripperManager
    camera: CameraManager
    sim_world: Optional[object] = None   # SimWorld when backend == 'sim'


def build_hardware(config) -> HardwareSet:
    """
    Assemble the hardware driver set for the backend named in the profile.

    Parameters
    ----------
    config : core.config.Config
        The loaded unit configuration.

    Returns
    -------
    HardwareSet
        Connected bridge plus drive/arm/gripper/camera drivers.

    Raises
    ------
    ValueError
        If the profile names an unknown backend.
    """
    backend = config.hardware.backend.lower()
    sim_world = None

    if backend == HardwareBackend.SIM:
        # Import here so simulation code is never loaded on a real robot
        from simulation.sim_pico_bridge import SimPicoBridge
        from simulation.sim_world import SimWorld

        sim_world = SimWorld.from_config(config.simulation)
        bridge: PicoBridge = SimPicoBridge(sim_world)
        log.info("Hardware factory: SIMULATED body selected.")

    elif backend == HardwareBackend.PICO:
        bridge = PicoBridge(port=config.hardware.pico_serial_port)
        log.info("Hardware factory: Pico serial body selected (%s).", config.hardware.pico_serial_port)

    else:
        raise ValueError(
            f"Unknown hardware backend '{backend}'. "
            f"Valid backends: {[b.value for b in HardwareBackend]}. "
            "To port a new robot body, see hardware/interfaces.py."
        )

    bridge.connect()

    drive = DriveController(
        pico_bridge=bridge,
        max_speed=config.hardware.drive_max_speed,
        drive_type=config.hardware.drive_type,
    )

    arm = ArmController(
        arm_type=config.hardware.arm_type,
        controller=config.hardware.arm_controller,
        max_payload_kg=config.hardware.arm_max_payload_kg,
    )

    gripper = GripperManager(arm_controller=arm)

    camera = CameraManager(
        model=config.hardware.camera_model,
        fps=config.vision.camera_fps,
        width=config.vision.camera_width,
        height=config.vision.camera_height,
    )
    camera.open()

    return HardwareSet(
        backend=backend,
        bridge=bridge,
        drive=drive,
        arm=arm,
        gripper=gripper,
        camera=camera,
        sim_world=sim_world,
    )
