#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : safety/fault_manager.py
Purpose     : Centralised fault registry. All subsystems report faults here.
              Tracks fault history, escalates repeated faults, and notifies
              the River Song API and alert manager.
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
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from core.constants import FaultCode

log = logging.getLogger(__name__)

# Maximum faults retained in memory
_MAX_FAULT_HISTORY = 200


@dataclass
class FaultRecord:
    """A single fault event."""

    fault_code: str
    message: str
    timestamp: float = field(default_factory=time.time)
    count: int = 1
    resolved: bool = False

    def to_dict(self) -> dict:
        """Serialise to a plain dict for API payloads."""
        return {
            "fault_code": self.fault_code,
            "message": self.message,
            "timestamp": self.timestamp,
            "count": self.count,
            "resolved": self.resolved,
        }


class FaultManager:
    """
    Centralised fault registry for a Kova unit.

    Subsystems call report() to register a fault. The manager deduplicates
    repeated faults, maintains a history ring buffer, and invokes the
    on_fault callback so the main controller can react.

    Attributes
    ----------
    robot_id : str
        Unit identifier, included in all fault reports.
    active_faults : dict
        Currently unresolved faults keyed by fault_code.
    """

    def __init__(
        self,
        robot_id: str,
        on_fault: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """
        Initialise the fault manager.

        Parameters
        ----------
        robot_id : str
            Unique identifier for this unit.
        on_fault : callable, optional
            Called with (fault_code, message) when a new fault is registered.
        """
        self.robot_id: str = robot_id
        self._on_fault = on_fault
        self._active: Dict[str, FaultRecord] = {}
        self._history: List[FaultRecord] = []
        self._lock = threading.Lock()

        log.debug("FaultManager initialised for unit '%s'.", robot_id)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def active_faults(self) -> Dict[str, FaultRecord]:
        """Currently unresolved faults keyed by fault_code."""
        with self._lock:
            return dict(self._active)

    @property
    def has_active_faults(self) -> bool:
        """True if any unresolved faults exist."""
        with self._lock:
            return bool(self._active)

    def report(
        self,
        message: str,
        fault_code: str = FaultCode.UNKNOWN,
    ) -> FaultRecord:
        """
        Register a fault.

        If the same fault_code is already active, its count is incremented.
        Otherwise a new FaultRecord is created and the on_fault callback fires.

        Parameters
        ----------
        message : str
            Human-readable description of the fault.
        fault_code : str
            FaultCode enum value. Defaults to UNKNOWN.

        Returns
        -------
        FaultRecord
            The created or updated fault record.
        """
        with self._lock:
            if fault_code in self._active:
                record = self._active[fault_code]
                record.count += 1
                record.timestamp = time.time()
                record.message = message
                log.warning(
                    "Fault [%s] repeated (count=%d): %s",
                    fault_code, record.count, message,
                )
            else:
                record = FaultRecord(fault_code=fault_code, message=message)
                self._active[fault_code] = record
                self._history.append(record)
                if len(self._history) > _MAX_FAULT_HISTORY:
                    self._history.pop(0)
                log.error("New fault [%s]: %s", fault_code, message)

        if self._on_fault and fault_code not in self._active or True:
            # Always fire callback so the main controller can react
            try:
                self._on_fault(fault_code, message)
            except Exception as exc:
                log.error("FaultManager on_fault callback error: %s", exc)

        return record

    def resolve(self, fault_code: str) -> bool:
        """
        Mark a fault as resolved and remove it from the active set.

        Parameters
        ----------
        fault_code : str
            The fault code to resolve.

        Returns
        -------
        bool
            True if the fault was found and resolved, False if not found.
        """
        with self._lock:
            record = self._active.pop(fault_code, None)
            if record:
                record.resolved = True
                log.info("Fault [%s] resolved.", fault_code)
                return True
        log.debug("FaultManager.resolve() — fault_code '%s' not found.", fault_code)
        return False

    def resolve_all(self) -> int:
        """
        Resolve all active faults.

        Returns
        -------
        int
            Number of faults resolved.
        """
        with self._lock:
            count = len(self._active)
            for record in self._active.values():
                record.resolved = True
            self._active.clear()
        log.info("All %d active faults resolved.", count)
        return count

    def get_history(self, limit: int = 50) -> List[dict]:
        """
        Return the most recent fault records as serialisable dicts.

        Parameters
        ----------
        limit : int
            Maximum number of records to return.

        Returns
        -------
        list of dict
        """
        with self._lock:
            return [r.to_dict() for r in self._history[-limit:]]
