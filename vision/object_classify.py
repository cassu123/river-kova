#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : vision/object_classify.py
Purpose     : Fine-grained object classification. Takes a cropped region from
              the detector and classifies it into a more specific category
              (e.g. 'cup' → 'coffee_mug', 'water_glass'). Used to determine
              the correct grasp strategy.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from core.constants import CLASSIFICATION_CONFIDENCE_THRESHOLD

log = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

# Grasp strategy mapping: fine-grained class → (grasp_type, width_mm, force_n)
_GRASP_STRATEGIES: Dict[str, Tuple[str, float, float]] = {
    "coffee_mug":    ("pinch",    60.0, 6.0),
    "water_glass":   ("pinch",    55.0, 5.0),
    "plate":         ("flat",     80.0, 8.0),
    "bowl":          ("scoop",    70.0, 7.0),
    "bottle":        ("wrap",     50.0, 10.0),
    "can":           ("wrap",     45.0, 10.0),
    "book":          ("flat",     75.0, 8.0),
    "remote":        ("pinch",    40.0, 5.0),
    "phone":         ("pinch",    35.0, 4.0),
    "trash_bag":     ("wrap",     80.0, 15.0),
    "default":       ("wrap",     50.0, 8.0),
}


class ClassificationResult:
    """Result of a fine-grained classification."""

    def __init__(
        self,
        class_name: str,
        confidence: float,
        grasp_type: str = "wrap",
        grasp_width_mm: float = 50.0,
        grasp_force_n: float = 8.0,
    ) -> None:
        """
        Initialise a classification result.

        Parameters
        ----------
        class_name : str
            Fine-grained class label.
        confidence : float
            Classification confidence [0.0, 1.0].
        grasp_type : str
            Recommended grasp type.
        grasp_width_mm : float
            Recommended gripper width for grasping.
        grasp_force_n : float
            Recommended grasp force in Newtons.
        """
        self.class_name = class_name
        self.confidence = confidence
        self.grasp_type = grasp_type
        self.grasp_width_mm = grasp_width_mm
        self.grasp_force_n = grasp_force_n

    def to_dict(self) -> dict:
        """Serialise to a plain dict."""
        return {
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "grasp_type": self.grasp_type,
            "grasp_width_mm": self.grasp_width_mm,
            "grasp_force_n": self.grasp_force_n,
        }


class ObjectClassifier:
    """
    Fine-grained object classifier.

    Takes a cropped image region (from the detector) and classifies it
    into a specific sub-category with an associated grasp strategy.

    Attributes
    ----------
    confidence_threshold : float
        Minimum confidence to accept a classification.
    """

    def __init__(
        self,
        confidence_threshold: float = CLASSIFICATION_CONFIDENCE_THRESHOLD,
        model_path: str = "/opt/kova/models/classifier.onnx",
    ) -> None:
        """
        Initialise the classifier.

        Parameters
        ----------
        confidence_threshold : float
            Minimum confidence to accept a result.
        model_path : str
            Path to the ONNX classification model.
        """
        self.confidence_threshold = confidence_threshold
        self._model = None

        if _CV2_AVAILABLE:
            self._model = self._load_model(model_path)

    def _load_model(self, model_path: str):
        """
        Load the ONNX classification model.

        Parameters
        ----------
        model_path : str
            Path to the .onnx file.

        Returns
        -------
        cv2.dnn.Net or None
        """
        try:
            net = cv2.dnn.readNetFromONNX(model_path)
            log.info("ObjectClassifier: model loaded from %s.", model_path)
            return net
        except Exception as exc:
            log.warning("ObjectClassifier: model load failed (%s) — stub mode.", exc)
            return None

    def classify(
        self,
        frame,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        coarse_class: str = "unknown",
    ) -> ClassificationResult:
        """
        Classify an object in a frame region.

        Parameters
        ----------
        frame : np.ndarray
            Full BGR frame.
        bbox : tuple, optional
            (x, y, w, h) bounding box to crop. If None, uses the full frame.
        coarse_class : str
            Coarse class from the detector (used as fallback).

        Returns
        -------
        ClassificationResult
            Fine-grained classification with grasp strategy.
        """
        if not _CV2_AVAILABLE or self._model is None:
            return self._default_result(coarse_class)

        try:
            if bbox is not None:
                x, y, w, h = bbox
                crop = frame[max(0, y):y + h, max(0, x):x + w]
                if crop.size == 0:
                    return self._default_result(coarse_class)
            else:
                crop = frame

            blob = cv2.dnn.blobFromImage(
                crop, 1 / 255.0, (224, 224), swapRB=True, crop=True
            )
            self._model.setInput(blob)
            output = self._model.forward()
            class_id = int(np.argmax(output))
            confidence = float(output[0][class_id])

            if confidence < self.confidence_threshold:
                return self._default_result(coarse_class)

            # Map class_id to a fine-grained label (model-specific)
            class_name = self._id_to_label(class_id, coarse_class)
            grasp = _GRASP_STRATEGIES.get(class_name, _GRASP_STRATEGIES["default"])

            return ClassificationResult(
                class_name=class_name,
                confidence=confidence,
                grasp_type=grasp[0],
                grasp_width_mm=grasp[1],
                grasp_force_n=grasp[2],
            )

        except Exception as exc:
            log.error("ObjectClassifier.classify error: %s", exc)
            return self._default_result(coarse_class)

    def _default_result(self, coarse_class: str) -> ClassificationResult:
        """Return a default result based on the coarse class."""
        grasp = _GRASP_STRATEGIES.get(coarse_class, _GRASP_STRATEGIES["default"])
        return ClassificationResult(
            class_name=coarse_class,
            confidence=0.0,
            grasp_type=grasp[0],
            grasp_width_mm=grasp[1],
            grasp_force_n=grasp[2],
        )

    def _id_to_label(self, class_id: int, fallback: str) -> str:
        """
        Map a model output class ID to a label string.

        Parameters
        ----------
        class_id : int
            Model output index.
        fallback : str
            Label to use if the ID is out of range.

        Returns
        -------
        str
        """
        labels = list(_GRASP_STRATEGIES.keys())
        if 0 <= class_id < len(labels) - 1:  # Exclude 'default'
            return labels[class_id]
        return fallback
