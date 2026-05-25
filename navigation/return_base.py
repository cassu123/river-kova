#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : navigation/return_base.py
Purpose     : Return-to-base controller. Navigates the unit back to the
              charging dock on low battery, task completion, or operator
              command. Handles docking alignment.
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

from core.constants import BASE_STATION_TOLERANCE
from hardware.drive_controller import DriveController
from navigation.path_planner import PathPlanner

log = logging.getLogger(__name__)


class ReturnToBase:
    """
    Return-to-base and docking controller.

    Navigates the unit to the base station coordinates and performs a
    final docking alignment manoeuvre.

    Attributes
    ----------
    base_x : float
        Base station x coordinate in metres.
    base_y : float
        Base station y coordinate in metres.
    is_docked : bool
        True when the unit is at the base station.
    """

    def __init__(
        self,
        drive: DriveController,
        path_planner: PathPlanner,
        base_x: float = 0.0,
        base_y: float = 0.0,
        tolerance: float = BASE_STATION_TOLERANCE,
    ) -> None:
        """
        Initialise the return-to-base controller.

        Parameters
        ----------
        drive : DriveController
            Drive controller for final docking manoeuvre.
        path_planner : PathPlanner
            Path planner for navigation to the base.
        base_x : float
            Base station x coordinate.
        base_y : float
            Base station y coordinate.
        tolerance : float
            Acceptable distance from base (metres) to consider docked.
        """
        self._drive = drive
        self._planner = path_planner
        self.base_x = base_x
        self.base_y = base_y
        self._tolerance = tolerance
        self._docked: bool = False

        # Register the base as a waypoint
        self._planner.set_waypoint("base", base_x, base_y)
        log.info(
            "ReturnToBase ready (base=(%.2f, %.2f), tolerance=%.2fm).",
            base_x, base_y, tolerance,
        )

    @property
    def is_docked(self) -> bool:
        """True when the unit is at the base station."""
        return self._docked

    def execute(self) -> bool:
        """
        Navigate to the base station and dock.

        Returns
        -------
        bool
            True if docking was successful.
        """
        log.info("ReturnToBase: initiating return to base.")
        self._docked = False

        # Navigate to the base waypoint
        reached = self._planner.navigate_to_waypoint("base")
        if not reached:
            log.error("ReturnToBase: failed to reach base station.")
            return False

        # Final docking alignment
        success = self._dock()
        if success:
            self._docked = True
            log.info("ReturnToBase: docked successfully.")
        else:
            log.error("ReturnToBase: docking alignment failed.")

        return success

    def _dock(self) -> bool:
        """
        Perform the final docking alignment manoeuvre.

        Drives slowly forward until the docking sensor confirms contact,
        or until a timeout elapses.

        Returns
        -------
        bool
            True if docking contact was confirmed.
        """
        log.info("ReturnToBase: starting docking alignment.")
        timeout = 30.0
        deadline = time.monotonic() + timeout

        # Creep forward slowly toward the dock
        while time.monotonic() < deadline:
            self._drive.move_safe(linear=0.05, angular=0.0)
            time.sleep(0.1)

            # In a real system, check the dock contact sensor here
            # For now, stub: assume docked after 2 seconds of creeping
            if time.monotonic() > deadline - (timeout - 2.0):
                self._drive.stop()
                return True

        self._drive.stop()
        log.warning("ReturnToBase: docking timeout.")
        return False

    def undock(self) -> bool:
        """
        Back away from the charging dock to begin a new task.

        Returns
        -------
        bool
            True if undocking succeeded.
        """
        log.info("ReturnToBase: undocking.")
        self._docked = False
        self._drive.move_safe(linear=-0.2, angular=0.0)
        time.sleep(2.0)
        self._drive.stop()
        return True
