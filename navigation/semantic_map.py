#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : navigation/semantic_map.py
Purpose     : The robot's learned memory of the home. As it drives, every
              observation (position + visible objects) is classified and
              accumulated into grid cells, so "the kitchen" is wherever the
              robot has repeatedly SEEN kitchen things — never a hardcoded
              rectangle. Evidence decays on re-observation, so when the
              furniture moves the map follows.

              Body-agnostic: observations come from SimWorld in simulation
              and the vision pipeline on real hardware. Persisted as JSON so
              the home survives reboots.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-06-12
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from vision.room_classifier import RoomClassifier, UNKNOWN

log = logging.getLogger(__name__)

# Older evidence is multiplied by this on every new observation of the same
# cell — high enough to be stable, low enough that a moved couch re-labels
# its old spot within a handful of sightings.
_EVIDENCE_DECAY = 0.8

# A cell needs at least this much accumulated evidence to report a label.
_MIN_CELL_EVIDENCE = 0.5


class SemanticMap:
    """
    Grid of learned room labels built from live observations.

    Thread-safe: the control loop observes while the API and task system
    query concurrently.

    Attributes
    ----------
    cell_size_m : float
        Edge length of one grid cell in metres.
    """

    def __init__(
        self,
        cell_size_m: float = 1.0,
        classifier: Optional[RoomClassifier] = None,
    ) -> None:
        """
        Parameters
        ----------
        cell_size_m : float
            Grid resolution — 1 m cells suit room-scale labelling.
        classifier : RoomClassifier, optional
            Injectable for tests; defaults to the standard signatures.
        """
        self.cell_size_m = cell_size_m
        self._classifier = classifier or RoomClassifier()
        self._lock = threading.RLock()
        # (i, j) → {room label: accumulated evidence}
        self._cells: Dict[Tuple[int, int], Dict[str, float]] = {}
        # Every cell the robot has observed from, labelled or not.
        self._visited: set = set()

    # ── Observation ───────────────────────────────────────────────────────────

    def _key(self, x: float, y: float) -> Tuple[int, int]:
        return (int(x // self.cell_size_m), int(y // self.cell_size_m))

    def observe(self, x: float, y: float, object_kinds: Iterable[str]) -> str:
        """
        Record one observation from position (x, y).

        Parameters
        ----------
        x, y : float
            Robot position when the observation was made.
        object_kinds : iterable of str
            Object class names currently visible.

        Returns
        -------
        str
            The label now assigned to this cell (may be "unknown").
        """
        label, confidence = self._classifier.classify(object_kinds)
        key = self._key(x, y)

        with self._lock:
            self._visited.add(key)
            scores = self._cells.setdefault(key, {})
            for existing in scores:
                scores[existing] *= _EVIDENCE_DECAY
            if label != UNKNOWN:
                scores[label] = scores.get(label, 0.0) + confidence
            return self._cell_label(scores)[0]

    @staticmethod
    def _cell_label(scores: Dict[str, float]) -> Tuple[str, float]:
        """Best label for one cell's evidence, or unknown if too weak."""
        if not scores:
            return (UNKNOWN, 0.0)
        label = max(scores, key=scores.get)
        evidence = scores[label]
        if evidence < _MIN_CELL_EVIDENCE:
            return (UNKNOWN, 0.0)
        confidence = evidence / sum(scores.values())
        return (label, round(confidence, 3))

    # ── Queries ───────────────────────────────────────────────────────────────

    def room_at(self, x: float, y: float) -> Tuple[str, float]:
        """
        Learned label for the cell containing (x, y).

        Returns
        -------
        tuple of (str, float)
            (room label, confidence) — ("unknown", 0.0) if never observed.
        """
        with self._lock:
            return self._cell_label(self._cells.get(self._key(x, y), {}))

    def known_rooms(self) -> Dict[str, Dict[str, float]]:
        """
        All learned room labels with their centroids.

        Returns
        -------
        dict
            label → {"x": centroid x, "y": centroid y, "cells": count,
                     "confidence": mean confidence}.
        """
        with self._lock:
            grouped: Dict[str, List[Tuple[int, int, float]]] = {}
            for (i, j), scores in self._cells.items():
                label, confidence = self._cell_label(scores)
                if label != UNKNOWN:
                    grouped.setdefault(label, []).append((i, j, confidence))

        half = self.cell_size_m / 2.0
        result: Dict[str, Dict[str, float]] = {}
        for label, cells in grouped.items():
            count = len(cells)
            result[label] = {
                "x": round(sum(i for i, _, _ in cells) / count * self.cell_size_m + half, 2),
                "y": round(sum(j for _, j, _ in cells) / count * self.cell_size_m + half, 2),
                "cells": count,
                "confidence": round(sum(c for _, _, c in cells) / count, 3),
            }
        return result

    def locate(self, room_label: str) -> Optional[Tuple[float, float]]:
        """
        Where a room was learned to be — for "vacuum the kitchen".

        Parameters
        ----------
        room_label : str
            Learned room label (e.g. "kitchen").

        Returns
        -------
        tuple of (float, float) or None
            Centroid of the labelled region, or None if never recognised.
        """
        entry = self.known_rooms().get(room_label)
        return (entry["x"], entry["y"]) if entry else None

    @property
    def visited_cell_count(self) -> int:
        """Number of distinct cells the robot has observed from."""
        with self._lock:
            return len(self._visited)

    def snapshot(self) -> Dict[str, object]:
        """Compact state for the local API / dashboard."""
        return {
            "cell_size_m": self.cell_size_m,
            "visited_cells": self.visited_cell_count,
            "rooms": self.known_rooms(),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist learned cells to a JSON file (atomic best-effort)."""
        with self._lock:
            payload = {
                "cell_size_m": self.cell_size_m,
                "cells": [
                    {"i": i, "j": j, "scores": scores}
                    for (i, j), scores in self._cells.items()
                ],
                "visited": sorted(self._visited),
            }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(target)
        log.info("SemanticMap: saved %d cell(s) to %s.", len(payload["cells"]), path)

    def load(self, path: str) -> bool:
        """
        Restore learned cells from a JSON file.

        Returns
        -------
        bool
            True if a map was loaded, False if the file is absent/invalid.
        """
        target = Path(path)
        if not target.exists():
            return False
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            with self._lock:
                self.cell_size_m = float(payload.get("cell_size_m", self.cell_size_m))
                self._cells = {
                    (int(c["i"]), int(c["j"])): {str(k): float(v) for k, v in c["scores"].items()}
                    for c in payload.get("cells", [])
                }
                self._visited = {tuple(v) for v in payload.get("visited", [])}
                self._visited.update(self._cells.keys())
        except (ValueError, KeyError, TypeError) as exc:
            log.warning("SemanticMap: could not load %s: %s", path, exc)
            return False
        log.info("SemanticMap: loaded %d cell(s) from %s.", len(self._cells), path)
        return True
