#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : tasks/chore_library.py
Purpose     : Predefined chore routine library. Each chore is a named sequence
              of steps that the task executor can run. The library is
              extensible — new chores can be added without modifying core code.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.constants import ChoreType

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# STEP SCHEMA
# Each step is a dict:
#   {
#     "action": str,          # e.g. "navigate", "arm_pose", "gripper_open"
#     "params": dict,         # action-specific parameters
#     "timeout_sec": float,   # max time for this step
#     "retry": int,           # retry count on failure (0 = no retry)
#   }
# ─────────────────────────────────────────────────────────────────────────────


def _step(
    action: str,
    params: Optional[Dict[str, Any]] = None,
    timeout_sec: float = 30.0,
    retry: int = 0,
) -> Dict[str, Any]:
    """
    Helper to build a chore step dict.

    Parameters
    ----------
    action : str
        Step action identifier.
    params : dict, optional
        Action parameters.
    timeout_sec : float
        Maximum time allowed for this step.
    retry : int
        Number of retries on failure.

    Returns
    -------
    dict
        Step descriptor.
    """
    return {
        "action": action,
        "params": params or {},
        "timeout_sec": timeout_sec,
        "retry": retry,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHORE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

_CHORE_DEFINITIONS: Dict[str, Dict[str, Any]] = {

    ChoreType.VACUUM: {
        "name": "Vacuum Room",
        "description": "Navigate the room in a coverage pattern and vacuum the floor.",
        "chore_type": ChoreType.VACUUM,
        "estimated_duration_sec": 900,
        "steps": [
            _step("arm_pose", {"pose": "stow"}, timeout_sec=10),
            _step("navigate_coverage", {"pattern": "boustrophedon"}, timeout_sec=900, retry=1),
            _step("navigate_to_base", {}, timeout_sec=120),
        ],
    },

    ChoreType.MOP: {
        "name": "Mop Floor",
        "description": "Navigate the room in a coverage pattern and mop the floor.",
        "chore_type": ChoreType.MOP,
        "estimated_duration_sec": 1200,
        "steps": [
            _step("arm_pose", {"pose": "stow"}, timeout_sec=10),
            _step("navigate_coverage", {"pattern": "boustrophedon", "speed_factor": 0.6}, timeout_sec=1200, retry=1),
            _step("navigate_to_base", {}, timeout_sec=120),
        ],
    },

    ChoreType.FETCH: {
        "name": "Fetch Object",
        "description": "Locate, navigate to, grasp, and deliver a specified object.",
        "chore_type": ChoreType.FETCH,
        "estimated_duration_sec": 300,
        "steps": [
            _step("detect_object", {"target_class": None}, timeout_sec=30, retry=2),
            _step("navigate_to_object", {}, timeout_sec=60, retry=1),
            _step("arm_pose", {"pose": "ready"}, timeout_sec=10),
            _step("gripper_open", {}, timeout_sec=5),
            _step("arm_grasp", {"force_n": 8.0}, timeout_sec=15, retry=1),
            _step("arm_pose", {"pose": "carry"}, timeout_sec=10),
            _step("navigate_to_waypoint", {"waypoint": "delivery_point"}, timeout_sec=120),
            _step("arm_pose", {"pose": "place"}, timeout_sec=10),
            _step("gripper_open", {}, timeout_sec=5),
            _step("arm_pose", {"pose": "stow"}, timeout_sec=10),
            _step("navigate_to_base", {}, timeout_sec=120),
        ],
    },

    ChoreType.ORGANIZE: {
        "name": "Organize Items",
        "description": "Pick up out-of-place items and return them to designated locations.",
        "chore_type": ChoreType.ORGANIZE,
        "estimated_duration_sec": 600,
        "steps": [
            _step("scan_room", {}, timeout_sec=30),
            _step("organize_loop", {"max_items": 10}, timeout_sec=540, retry=0),
            _step("arm_pose", {"pose": "stow"}, timeout_sec=10),
            _step("navigate_to_base", {}, timeout_sec=120),
        ],
    },

    ChoreType.WIPE_SURFACE: {
        "name": "Wipe Surface",
        "description": "Navigate to a surface and wipe it with the arm-mounted tool.",
        "chore_type": ChoreType.WIPE_SURFACE,
        "estimated_duration_sec": 180,
        "steps": [
            _step("navigate_to_waypoint", {"waypoint": "surface_target"}, timeout_sec=60),
            _step("arm_pose", {"pose": "wipe_ready"}, timeout_sec=10),
            _step("arm_wipe_motion", {"passes": 3, "width_m": 0.4}, timeout_sec=90, retry=1),
            _step("arm_pose", {"pose": "stow"}, timeout_sec=10),
            _step("navigate_to_base", {}, timeout_sec=60),
        ],
    },

    ChoreType.TAKE_OUT_TRASH: {
        "name": "Take Out Trash",
        "description": "Navigate to the bin, grasp the bag, and carry it to the collection point.",
        "chore_type": ChoreType.TAKE_OUT_TRASH,
        "estimated_duration_sec": 300,
        "steps": [
            _step("navigate_to_waypoint", {"waypoint": "trash_bin"}, timeout_sec=60),
            _step("arm_pose", {"pose": "ready"}, timeout_sec=10),
            _step("gripper_open", {}, timeout_sec=5),
            _step("arm_grasp", {"target": "trash_bag", "force_n": 15.0}, timeout_sec=20, retry=2),
            _step("arm_pose", {"pose": "carry"}, timeout_sec=10),
            _step("navigate_to_waypoint", {"waypoint": "trash_collection"}, timeout_sec=120),
            _step("arm_pose", {"pose": "place"}, timeout_sec=10),
            _step("gripper_open", {}, timeout_sec=5),
            _step("arm_pose", {"pose": "stow"}, timeout_sec=10),
            _step("navigate_to_base", {}, timeout_sec=120),
        ],
    },

    ChoreType.LOAD_DISHWASHER: {
        "name": "Load Dishwasher",
        "description": "Collect dirty dishes and load them into the dishwasher.",
        "chore_type": ChoreType.LOAD_DISHWASHER,
        "estimated_duration_sec": 600,
        "steps": [
            _step("scan_room", {"target_classes": ["cup", "plate", "bowl", "utensil"]}, timeout_sec=30),
            _step("load_dishwasher_loop", {"max_items": 12}, timeout_sec=540),
            _step("arm_pose", {"pose": "stow"}, timeout_sec=10),
            _step("navigate_to_base", {}, timeout_sec=120),
        ],
    },

    ChoreType.UNLOAD_DISHWASHER: {
        "name": "Unload Dishwasher",
        "description": "Remove clean dishes from the dishwasher and place them in cupboards.",
        "chore_type": ChoreType.UNLOAD_DISHWASHER,
        "estimated_duration_sec": 600,
        "steps": [
            _step("navigate_to_waypoint", {"waypoint": "dishwasher"}, timeout_sec=60),
            _step("unload_dishwasher_loop", {"max_items": 12}, timeout_sec=540),
            _step("arm_pose", {"pose": "stow"}, timeout_sec=10),
            _step("navigate_to_base", {}, timeout_sec=120),
        ],
    },

    ChoreType.LAUNDRY_TRANSFER: {
        "name": "Laundry Transfer",
        "description": "Transfer laundry from washer to dryer.",
        "chore_type": ChoreType.LAUNDRY_TRANSFER,
        "estimated_duration_sec": 300,
        "steps": [
            _step("navigate_to_waypoint", {"waypoint": "washer"}, timeout_sec=60),
            _step("open_appliance_door", {"appliance": "washer"}, timeout_sec=15, retry=2),
            _step("transfer_laundry", {"source": "washer", "destination": "dryer"}, timeout_sec=180),
            _step("close_appliance_door", {"appliance": "washer"}, timeout_sec=10),
            _step("close_appliance_door", {"appliance": "dryer"}, timeout_sec=10),
            _step("arm_pose", {"pose": "stow"}, timeout_sec=10),
            _step("navigate_to_base", {}, timeout_sec=120),
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CHORE LIBRARY CLASS
# ─────────────────────────────────────────────────────────────────────────────


class ChoreLibrary:
    """
    Registry of predefined chore routines.

    Chores are looked up by ChoreType or by name. Custom chores can be
    registered at runtime without modifying this file.

    Attributes
    ----------
    chore_count : int
        Number of registered chores.
    """

    def __init__(self) -> None:
        """Initialise the library with all built-in chore definitions."""
        self._chores: Dict[str, Dict[str, Any]] = dict(_CHORE_DEFINITIONS)
        log.info("ChoreLibrary loaded with %d chores.", len(self._chores))

    def __len__(self) -> int:
        """Return the number of registered chores."""
        return len(self._chores)

    @property
    def chore_count(self) -> int:
        """Number of registered chores."""
        return len(self._chores)

    def get(self, chore_type: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a chore definition by type.

        Parameters
        ----------
        chore_type : str
            ChoreType value or custom chore name.

        Returns
        -------
        dict or None
            Chore definition, or None if not found.
        """
        chore = self._chores.get(chore_type)
        if chore is None:
            log.warning("ChoreLibrary: unknown chore type '%s'.", chore_type)
        return chore

    def register(self, chore_type: str, definition: Dict[str, Any]) -> None:
        """
        Register a custom chore definition.

        Parameters
        ----------
        chore_type : str
            Unique identifier for the chore.
        definition : dict
            Chore definition following the standard schema.

        Raises
        ------
        ValueError
            If required fields are missing from the definition.
        """
        required = ["name", "steps"]
        for field in required:
            if field not in definition:
                raise ValueError(f"Chore definition missing required field: '{field}'")
        self._chores[chore_type] = definition
        log.info("ChoreLibrary: registered custom chore '%s'.", chore_type)

    def list_chores(self) -> List[str]:
        """
        Return a list of all registered chore type identifiers.

        Returns
        -------
        list of str
        """
        return list(self._chores.keys())

    def get_steps(self, chore_type: str) -> List[Dict[str, Any]]:
        """
        Return the step list for a chore.

        Parameters
        ----------
        chore_type : str
            ChoreType value.

        Returns
        -------
        list of dict
            Step descriptors, or empty list if chore not found.
        """
        chore = self.get(chore_type)
        return chore.get("steps", []) if chore else []
