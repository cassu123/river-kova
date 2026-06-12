#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : autonomy/initiative_engine.py
Purpose     : Initiative engine — the robot's self-directed brain. Instead of
              only waiting to be told, the engine periodically evaluates a set
              of rules ("is it 9am Saturday? vacuum the kitchen", "is there a
              sock on the living-room floor? tidy up") and submits chores on
              its own.

              The rule set is deliberately pluggable: today the rules are
              simple schedules and observations; later, a River Song LLM rule
              can propose tasks from natural-language context without
              changing anything here.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-06-12
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)

_DAY_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
_ROUTINE_GRACE_MIN = 30        # Fire a routine up to 30 min after its time
_DEFAULT_COOLDOWN_SEC = 3600.0


@dataclass
class TaskProposal:
    """A chore the initiative engine wants to run."""

    chore_type: str
    room: Optional[str] = None
    priority: int = 4              # Below voice commands (7) by default
    reason: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


class InitiativeRule(ABC):
    """
    Base class for initiative rules.

    Each rule inspects the world (time, observations, robot state) and may
    propose chores. The engine enforces a per-rule cooldown so a rule cannot
    flood the queue.
    """

    name: str = "rule"
    cooldown_sec: float = _DEFAULT_COOLDOWN_SEC

    @abstractmethod
    def evaluate(self, now: datetime) -> List[TaskProposal]:
        """
        Evaluate the rule.

        Parameters
        ----------
        now : datetime
            Current local time (injectable for tests).

        Returns
        -------
        list of TaskProposal
            Chores this rule wants to run right now (often empty).
        """


class ScheduledRoutineRule(InitiativeRule):
    """
    Fires chores from the profile's routine schedule.

    Routine entry format (profile `autonomy.routines`):
        {"chore": "VACUUM", "room": "kitchen", "time": "09:00",
         "days": ["mon", "wed", "fri"], "priority": 4}

    Omitting "days" means every day.
    """

    name = "scheduled_routine"
    cooldown_sec = 0.0             # Per-routine dedup handles repeats

    def __init__(self, routines: List[Dict[str, Any]]) -> None:
        """
        Parameters
        ----------
        routines : list of dict
            Routine entries from the unit profile.
        """
        self._routines = routines
        self._last_fired_date: Dict[int, str] = {}    # routine index → ISO date
        log.info("ScheduledRoutineRule loaded %d routine(s).", len(routines))

    def evaluate(self, now: datetime) -> List[TaskProposal]:
        """Propose any routine whose scheduled time window is open today."""
        proposals: List[TaskProposal] = []
        today = now.date().isoformat()
        day_name = _DAY_NAMES[now.weekday()]

        for index, routine in enumerate(self._routines):
            days = [d.lower()[:3] for d in routine.get("days", _DAY_NAMES)]
            if day_name not in days:
                continue
            if self._last_fired_date.get(index) == today:
                continue

            try:
                hour, minute = (int(p) for p in routine.get("time", "09:00").split(":"))
            except ValueError:
                log.warning("Routine %d has invalid time '%s' — skipped.", index, routine.get("time"))
                continue

            scheduled_min = hour * 60 + minute
            now_min = now.hour * 60 + now.minute
            if scheduled_min <= now_min <= scheduled_min + _ROUTINE_GRACE_MIN:
                self._last_fired_date[index] = today
                proposals.append(TaskProposal(
                    chore_type=routine.get("chore", "VACUUM"),
                    room=routine.get("room"),
                    priority=int(routine.get("priority", 4)),
                    reason=f"Scheduled routine ({routine.get('time')} {day_name}).",
                ))
        return proposals


class TidyUpRule(InitiativeRule):
    """
    Proposes an ORGANIZE chore when out-of-place objects are observed.

    The observation provider abstracts where sightings come from — the
    vision pipeline on a real robot, the SimWorld in simulation, or River
    Song shared context later.
    """

    name = "tidy_up"
    cooldown_sec = 3600.0          # At most one tidy-up proposal per hour

    def __init__(
        self,
        observation_provider: Callable[[], List[Dict[str, Any]]],
        min_items: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        observation_provider : callable
            Returns a list of out-of-place observations, each a dict with at
            least {"kind": str} and optionally {"room": str}.
        min_items : int
            Minimum sightings before proposing a tidy-up.
        """
        self._observe = observation_provider
        self._min_items = min_items

    def evaluate(self, now: datetime) -> List[TaskProposal]:
        """Propose ORGANIZE when enough out-of-place items are seen."""
        try:
            sightings = self._observe() or []
        except Exception as exc:
            log.warning("TidyUpRule observation provider error: %s", exc)
            return []

        if len(sightings) < self._min_items:
            return []

        room = sightings[0].get("room")
        kinds = sorted({s.get("kind", "item") for s in sightings})
        return [TaskProposal(
            chore_type="ORGANIZE",
            room=room,
            priority=3,
            reason=f"Observed {len(sightings)} out-of-place item(s): {', '.join(kinds)}.",
        )]


class ExploreRule(InitiativeRule):
    """
    Proposes an EXPLORE chore while the home is still unfamiliar.

    The robot has no preloaded floor plan — rooms are recognised from what
    the camera sees and accumulated into the semantic map. Until enough
    rooms are known, this rule sends the robot out to learn the layout; if
    the home is later rearranged and labels fade, it fires again.
    """

    name = "explore"
    cooldown_sec = 600.0           # Re-attempt an unfinished sweep every 10 min

    def __init__(
        self,
        known_room_count_provider: Callable[[], int],
        min_known_rooms: int = 3,
    ) -> None:
        """
        Parameters
        ----------
        known_room_count_provider : callable
            Returns how many distinct rooms the semantic map has recognised.
        min_known_rooms : int
            Stop proposing exploration once this many rooms are known.
        """
        self._known_rooms = known_room_count_provider
        self._min_rooms = min_known_rooms

    def evaluate(self, now: datetime) -> List[TaskProposal]:
        """Propose EXPLORE while too few rooms have been recognised."""
        try:
            known = self._known_rooms()
        except Exception as exc:
            log.warning("ExploreRule provider error: %s", exc)
            return []

        if known >= self._min_rooms:
            return []

        return [TaskProposal(
            chore_type="EXPLORE",
            priority=2,                # Below tidy-ups and all direct commands
            reason=f"Home layout unfamiliar — {known}/{self._min_rooms} rooms recognised.",
        )]


class InitiativeEngine:
    """
    Evaluates initiative rules and submits the resulting chores.

    The engine only acts when the robot can actually take on work: the
    caller gates ticks on IDLE state and nominal safety, and the engine
    itself suppresses proposals when the battery is below the floor.

    Attributes
    ----------
    rules : list of InitiativeRule
        Active rule set — extend at runtime with add_rule().
    """

    def __init__(
        self,
        task_manager,
        min_battery_pct: float = 30.0,
        battery_provider: Optional[Callable[[], float]] = None,
    ) -> None:
        """
        Parameters
        ----------
        task_manager : TaskManager
            Receives submitted proposals.
        min_battery_pct : float
            No initiative below this battery level.
        battery_provider : callable, optional
            Returns the current battery percentage. None = always allowed.
        """
        self._task_manager = task_manager
        self._min_battery = min_battery_pct
        self._battery = battery_provider
        self.rules: List[InitiativeRule] = []
        self._last_fired: Dict[str, float] = {}

        log.info("InitiativeEngine ready (min_battery=%.0f%%).", min_battery_pct)

    def add_rule(self, rule: InitiativeRule) -> None:
        """Register an initiative rule."""
        self.rules.append(rule)
        log.info("InitiativeEngine: rule '%s' added.", rule.name)

    def tick(self, now: Optional[datetime] = None) -> List[str]:
        """
        Evaluate all rules once and submit any proposals.

        Parameters
        ----------
        now : datetime, optional
            Injectable clock for tests. Defaults to local time.

        Returns
        -------
        list of str
            Task IDs submitted this tick.
        """
        now = now or datetime.now()

        if self._battery is not None:
            level = self._battery()
            if level < self._min_battery:
                log.debug("InitiativeEngine: suppressed (battery %.0f%% < %.0f%%).", level, self._min_battery)
                return []

        submitted: List[str] = []
        for rule in self.rules:
            last = self._last_fired.get(rule.name)
            if (
                rule.cooldown_sec > 0
                and last is not None
                and (time.monotonic() - last) < rule.cooldown_sec
            ):
                continue

            try:
                proposals = rule.evaluate(now)
            except Exception as exc:
                log.error("InitiativeEngine: rule '%s' error: %s", rule.name, exc)
                continue

            for proposal in proposals:
                task_id = self._task_manager.submit(
                    chore_type=proposal.chore_type,
                    room=proposal.room,
                    priority=proposal.priority,
                    params={**proposal.params, "initiative_reason": proposal.reason},
                )
                if task_id:
                    self._last_fired[rule.name] = time.monotonic()
                    submitted.append(task_id)
                    log.info(
                        "InitiativeEngine: submitted %s (%s) — %s",
                        proposal.chore_type, task_id, proposal.reason,
                    )
        return submitted
