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

import json
import logging
import os
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

    def __init__(
        self,
        routines: List[Dict[str, Any]],
        state_path: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        routines : list of dict
            Routine entries from the unit profile.
        state_path : str, optional
            JSON file for persisting last-fired dates. Without it, a reboot
            forgets what already ran today and routines can fire twice.
        """
        self._routines = routines
        self._state_path = state_path
        self._last_fired_date: Dict[int, str] = self._load_state()
        log.info("ScheduledRoutineRule loaded %d routine(s).", len(routines))

    def _load_state(self) -> Dict[int, str]:
        """Restore last-fired dates from disk; empty on any problem."""
        if not self._state_path:
            return {}
        try:
            with open(self._state_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            return {int(index): date for index, date in raw.items()}
        except FileNotFoundError:
            return {}
        except Exception as exc:
            log.warning("ScheduledRoutineRule: state load failed (%s) — starting fresh.", exc)
            return {}

    def _save_state(self) -> None:
        """Persist last-fired dates; failure only costs reboot dedup."""
        if not self._state_path:
            return
        try:
            with open(self._state_path, "w", encoding="utf-8") as fh:
                json.dump({str(k): v for k, v in self._last_fired_date.items()}, fh)
        except Exception as exc:
            log.warning("ScheduledRoutineRule: state save failed: %s", exc)

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
                self._save_state()
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


class LLMInitiativeRule(InitiativeRule):
    """
    Proposes chores by asking a Claude model to reason over natural-language
    household context plus the robot's current state.

    This is the slot the README promises: "a future River Song LLM rule slots
    in without core changes." It implements the same ``InitiativeRule``
    contract as the hand-written rules, so the engine treats it identically —
    cooldown, battery suppression, capability gating on submit.

    The model is constrained two ways so a hallucination can never drive the
    robot: structured outputs restrict ``chore_type`` to the exact catalogue
    passed in, and every proposal still passes through ``TaskManager.submit``,
    which independently rejects unknown chores and ones the body can't perform.

    Degrades to a silent no-op — never raising, never blocking — when the
    ``anthropic`` SDK isn't installed, no API key is set, or the call fails.
    The robot stays fully functional offline; this rule simply proposes
    nothing until the brain is reachable again.
    """

    name = "llm_initiative"
    cooldown_sec = 1800.0          # LLM calls cost tokens — at most twice an hour

    def __init__(
        self,
        chore_catalog: List[Dict[str, Any]],
        context_provider: Callable[[], str],
        state_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        model: str = "claude-opus-4-8",
        cooldown_sec: float = 1800.0,
        max_proposals: int = 3,
        client: Optional[Any] = None,
    ) -> None:
        """
        Parameters
        ----------
        chore_catalog : list of dict
            The chores the robot can actually run, each
            ``{"chore_type": str, "description": str}``. The model may only
            propose from this set (enforced by the output schema).
        context_provider : callable
            Returns the current natural-language household context (e.g. the
            contents of a notes file River Song keeps updated). Empty string
            is fine — the model then reasons from state alone.
        state_provider : callable, optional
            Returns a dict of structured state to hand the model (battery,
            known rooms, out-of-place sightings, time of day).
        model : str
            Claude model id. Defaults to the most capable Opus; set a smaller
            model in the profile for a cost-sensitive fleet.
        cooldown_sec : float
            Minimum seconds between calls.
        max_proposals : int
            Cap on chores proposed per evaluation.
        client : Anthropic, optional
            Injected client (tests). When None, one is built lazily from the
            environment on first use.
        """
        self._catalog = chore_catalog
        self._context = context_provider
        self._state = state_provider
        self._model = model
        self.cooldown_sec = cooldown_sec
        self._max_proposals = max_proposals
        self._client = client
        self._client_ready = client is not None
        self._disabled_reason: Optional[str] = None
        self._allowed = {c["chore_type"] for c in chore_catalog}

    def _ensure_client(self) -> bool:
        """Build the Anthropic client on first use; disable on any problem."""
        if self._client_ready:
            return True
        if self._disabled_reason is not None:
            return False
        try:
            import anthropic
        except ImportError:
            self._disabled_reason = "anthropic SDK not installed"
            log.warning("LLMInitiativeRule disabled — %s.", self._disabled_reason)
            return False
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            self._disabled_reason = "no ANTHROPIC_API_KEY in environment"
            log.warning("LLMInitiativeRule disabled — %s.", self._disabled_reason)
            return False
        try:
            self._client = anthropic.Anthropic()
        except Exception as exc:
            self._disabled_reason = f"client init failed: {exc}"
            log.warning("LLMInitiativeRule disabled — %s.", self._disabled_reason)
            return False
        self._client_ready = True
        return True

    def _system_prompt(self) -> str:
        """Stable instruction + chore catalogue — cached across calls."""
        lines = [
            "You are the initiative planner for River Kova, a household chore "
            "robot. Given the home's current context and the robot's state, "
            "decide which chores (if any) the robot should start now.",
            "",
            "Rules:",
            "- Only propose chores from the catalogue below.",
            "- Propose nothing if nothing is clearly worth doing — an empty "
            "list is the right answer when the home needs no attention.",
            "- Never propose work that a direct human command would override; "
            "keep priorities at or below 6 (direct commands are 7).",
            "- Prefer fewer, well-justified chores over a long list.",
            "",
            "Chore catalogue:",
        ]
        for chore in self._catalog:
            lines.append(f"- {chore['chore_type']}: {chore.get('description', '')}")
        return "\n".join(lines)

    def _user_message(self, now: datetime) -> str:
        """Per-call context — the volatile half of the prompt."""
        try:
            context = self._context() or ""
        except Exception as exc:
            log.warning("LLMInitiativeRule context provider error: %s", exc)
            context = ""

        state: Dict[str, Any] = {}
        if self._state is not None:
            try:
                state = self._state() or {}
            except Exception as exc:
                log.warning("LLMInitiativeRule state provider error: %s", exc)

        return (
            f"Current local time: {now.isoformat()}\n"
            f"Robot state: {json.dumps(state, default=str)}\n"
            f"Household context:\n{context.strip() or '(none provided)'}"
        )

    def _output_schema(self) -> Dict[str, Any]:
        """Structured-output schema — constrains chore_type to the catalogue."""
        return {
            "type": "object",
            "properties": {
                "proposals": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "chore_type": {"type": "string", "enum": sorted(self._allowed)},
                            "room": {"type": ["string", "null"]},
                            "priority": {"type": "integer"},
                            "reason": {"type": "string"},
                        },
                        "required": ["chore_type", "room", "priority", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["proposals"],
            "additionalProperties": False,
        }

    def evaluate(self, now: datetime) -> List[TaskProposal]:
        """Ask the model for chore proposals; return [] on any failure."""
        if not self._allowed or not self._ensure_client():
            return []

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                thinking={"type": "adaptive"},
                output_config={"effort": "low", "format": {
                    "type": "json_schema",
                    "schema": self._output_schema(),
                }},
                system=[{
                    "type": "text",
                    "text": self._system_prompt(),
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": self._user_message(now)}],
            )
        except Exception as exc:
            log.warning("LLMInitiativeRule: model call failed: %s", exc)
            return []

        if getattr(response, "stop_reason", None) == "refusal":
            log.info("LLMInitiativeRule: model declined to propose.")
            return []

        text = next(
            (b.text for b in response.content if getattr(b, "type", None) == "text"),
            "",
        )
        try:
            payload = json.loads(text)
        except (ValueError, TypeError) as exc:
            log.warning("LLMInitiativeRule: unparseable response: %s", exc)
            return []

        proposals: List[TaskProposal] = []
        for item in payload.get("proposals", [])[: self._max_proposals]:
            chore_type = item.get("chore_type")
            if chore_type not in self._allowed:
                # Schema should prevent this, but never trust it.
                continue
            proposals.append(TaskProposal(
                chore_type=chore_type,
                room=item.get("room"),
                priority=max(1, min(6, int(item.get("priority", 4)))),
                reason=f"LLM: {item.get('reason', 'proposed from household context')}",
            ))
        return proposals


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
