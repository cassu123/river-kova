#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : simulation/sim_pico_bridge.py
Purpose     : Drop-in replacement for hardware/pico_bridge.PicoBridge that
              talks to a SimWorld instead of a serial port. Every module that
              uses the Pico bridge (drive, battery, gripper, force sensing)
              works unchanged against the simulator.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-06-12
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from hardware.pico_bridge import PicoBridge, PicoBridgeError
from simulation.sim_world import SimWorld

log = logging.getLogger(__name__)


class SimPicoBridge(PicoBridge):
    """
    Simulated Pico bridge backed by a SimWorld.

    Implements the same JSON command protocol as the real bridge, so all
    higher-level modules (DriveController, BatteryMonitor, ...) are
    completely unaware they are running in simulation.
    """

    def __init__(self, world: SimWorld) -> None:
        """
        Initialise the simulated bridge.

        Parameters
        ----------
        world : SimWorld
            The simulated world this bridge controls.
        """
        super().__init__(port="sim://world", baud_rate=0)
        self._world = world

    # ── Connection lifecycle ──────────────────────────────────────────────────

    def connect(self) -> None:
        """Mark the bridge connected — no serial port involved."""
        self._connected = True
        log.info("SimPicoBridge connected to simulated world.")

    def disconnect(self) -> None:
        """Mark the bridge disconnected."""
        self._connected = False
        log.info("SimPicoBridge disconnected.")

    # ── Command interface ─────────────────────────────────────────────────────

    def send_command(self, cmd: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Dispatch a Pico protocol command to the simulated world.

        Parameters
        ----------
        cmd : str
            Command name (same set as the real Pico firmware).
        params : dict, optional
            Command parameters.

        Returns
        -------
        dict
            {"status": "ok", "data": {...}} response.

        Raises
        ------
        PicoBridgeError
            If the bridge is not connected or the command is unknown.
        """
        if not self._connected:
            raise PicoBridgeError("SimPicoBridge is not connected.")

        params = params or {}

        if cmd == "ping":
            return {"status": "ok", "data": {"firmware": "sim-1.0"}}
        if cmd == "set_motor":
            self._world.set_motors(params.get("left", 0.0), params.get("right", 0.0))
            return {"status": "ok", "data": {}}
        if cmd == "estop_motors":
            self._world.estop_motors()
            return {"status": "ok", "data": {}}
        if cmd == "read_battery":
            return {"status": "ok", "data": {"percent": self._world.read_battery()}}
        if cmd == "read_encoders":
            return {"status": "ok", "data": self._world.read_encoders()}
        if cmd == "read_force":
            return {"status": "ok", "data": {"force_n": self._world.read_force()}}
        if cmd == "set_gripper":
            self._world.set_gripper(params.get("position", 1.0))
            return {"status": "ok", "data": {}}

        raise PicoBridgeError(f"SimPicoBridge: unknown command '{cmd}'.")
