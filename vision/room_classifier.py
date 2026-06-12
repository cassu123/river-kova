#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : vision/room_classifier.py
Purpose     : Recognises what room the robot is in from the objects it can
              see — a kitchen is wherever the stove and fridge are, not a
              rectangle in a config file. Homes change: furniture moves,
              rooms get repurposed. Nothing here depends on a floor plan.

              Body-agnostic: the input is just a list of detected object
              class names. In simulation they come from SimWorld; on real
              hardware the same call is fed by the YOLO detector.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-06-12
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Tuple

log = logging.getLogger(__name__)

# Evidence weights: how strongly seeing one object of this kind suggests a
# room type. Fixtures (stove, toilet, bed) are strong anchors; movable items
# (cup, sock) are deliberately weightless — they wander between rooms.
ROOM_SIGNATURES: Dict[str, Dict[str, float]] = {
    "kitchen": {
        "stove": 3.0, "fridge": 3.0, "oven": 3.0, "dishwasher": 2.5,
        "microwave": 2.0, "kitchen_counter": 2.0, "sink": 1.0, "dog_bowl": 0.5,
    },
    "living_room": {
        "couch": 3.0, "sofa": 3.0, "tv": 2.5, "coffee_table": 2.0,
        "armchair": 2.0, "bookshelf": 1.0,
    },
    "bedroom": {
        "bed": 3.0, "wardrobe": 2.5, "dresser": 2.0, "nightstand": 1.5,
    },
    "bathroom": {
        "toilet": 3.0, "bathtub": 3.0, "shower": 3.0, "sink": 1.0,
        "towel_rack": 1.5,
    },
    "laundry_room": {
        "washer": 3.0, "dryer": 3.0,
    },
    "hallway": {
        "coat_rack": 1.5, "shoe_rack": 1.5, "front_door": 2.0,
    },
}

# Minimum evidence score before committing to a label — a lone sink could be
# a kitchen or a bathroom, so it stays "unknown" until an anchor appears.
_MIN_EVIDENCE = 2.0

UNKNOWN = "unknown"


class RoomClassifier:
    """
    Infers a room type from visible object class names.

    Attributes
    ----------
    signatures : dict
        Room type → {object kind: evidence weight}. Extensible at runtime
        so new homes can teach new anchors (e.g. "piano" → music_room).
    """

    def __init__(self, signatures: Dict[str, Dict[str, float]] = None) -> None:
        """
        Parameters
        ----------
        signatures : dict, optional
            Override the built-in room signatures.
        """
        self.signatures = signatures if signatures is not None else dict(ROOM_SIGNATURES)

    def classify(self, object_kinds: Iterable[str]) -> Tuple[str, float]:
        """
        Classify a set of visible objects into a room type.

        Parameters
        ----------
        object_kinds : iterable of str
            Detected object class names (duplicates are ignored — one stove
            is as convincing as three).

        Returns
        -------
        tuple of (str, float)
            (room_type, confidence in [0, 1]); ("unknown", 0.0) when the
            evidence is too weak or ambiguous.
        """
        kinds = set(object_kinds)
        if not kinds:
            return (UNKNOWN, 0.0)

        scores: Dict[str, float] = {}
        for room_type, weights in self.signatures.items():
            score = sum(weight for kind, weight in weights.items() if kind in kinds)
            if score > 0:
                scores[room_type] = score

        if not scores:
            return (UNKNOWN, 0.0)

        best_room = max(scores, key=scores.get)
        best_score = scores[best_room]
        if best_score < _MIN_EVIDENCE:
            return (UNKNOWN, 0.0)

        confidence = best_score / sum(scores.values())
        return (best_room, round(confidence, 3))

    def add_signature(self, room_type: str, weights: Dict[str, float]) -> None:
        """
        Register or extend evidence weights for a room type.

        Parameters
        ----------
        room_type : str
            Room label (new or existing).
        weights : dict
            {object kind: evidence weight} merged over any existing entry.
        """
        self.signatures.setdefault(room_type, {}).update(weights)
        log.info("RoomClassifier: signature for '%s' updated (%d kinds).",
                 room_type, len(self.signatures[room_type]))
