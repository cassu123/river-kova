#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : vision/object_detect.py
Purpose     : Object detection using OpenCV DNN (YOLO) or a configurable
              backend. Detects household objects in camera frames and returns
              bounding boxes, class labels, and confidence scores.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.constants import DETECTION_CONFIDENCE_THRESHOLD

log = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    log.warning("OpenCV not available — ObjectDetector running in stub mode.")

# COCO class names relevant to household chores
_HOUSEHOLD_CLASSES = [
    "person", "cup", "bottle", "bowl", "plate", "fork", "knife", "spoon",
    "chair", "couch", "bed", "dining table", "toilet", "tv", "laptop",
    "remote", "cell phone", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush", "backpack", "handbag",
    "suitcase", "umbrella", "shoe", "trash bag",
]


class Detection:
    """A single object detection result."""

    def __init__(
        self,
        class_name: str,
        confidence: float,
        bbox: tuple,  # (x, y, w, h) in pixels
        depth_m: float = -1.0,
    ) -> None:
        """
        Initialise a detection result.

        Parameters
        ----------
        class_name : str
            Detected object class label.
        confidence : float
            Detection confidence [0.0, 1.0].
        bbox : tuple
            Bounding box (x, y, width, height) in pixels.
        depth_m : float
            Estimated depth in metres (-1 if unavailable).
        """
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        self.depth_m = depth_m

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox,
            "depth_m": self.depth_m,
        }

    def __repr__(self) -> str:
        return (
            f"Detection(class='{self.class_name}', "
            f"conf={self.confidence:.2f}, bbox={self.bbox}, depth={self.depth_m:.2f}m)"
        )


class ObjectDetector:
    """
    Household object detector using OpenCV DNN (YOLO).

    Falls back to stub mode if OpenCV or model weights are unavailable.

    Attributes
    ----------
    confidence_threshold : float
        Minimum confidence to report a detection.
    last_detections : list of Detection
        Results from the most recent detect() call.
    """

    def __init__(
        self,
        confidence_threshold: float = DETECTION_CONFIDENCE_THRESHOLD,
        model_weights: str = "/opt/kova/models/yolov4-tiny.weights",
        model_config: str = "/opt/kova/models/yolov4-tiny.cfg",
        nms_threshold: float = 0.4,
    ) -> None:
        """
        Initialise the object detector.

        Parameters
        ----------
        confidence_threshold : float
            Minimum confidence score to include a detection.
        model_weights : str
            Path to YOLO weights file.
        model_config : str
            Path to YOLO config file.
        nms_threshold : float
            Non-maximum suppression threshold.
        """
        self.confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold
        self.last_detections: List[Detection] = []
        self._net = None

        if _CV2_AVAILABLE:
            self._net = self._load_model(model_weights, model_config)

    def _load_model(self, weights: str, config: str):
        """
        Load the YOLO DNN model.

        Parameters
        ----------
        weights : str
            Path to weights file.
        config : str
            Path to config file.

        Returns
        -------
        cv2.dnn.Net or None
        """
        try:
            net = cv2.dnn.readNet(weights, config)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            log.info("ObjectDetector: YOLO model loaded.")
            return net
        except Exception as exc:
            log.warning("ObjectDetector: model load failed (%s) — stub mode.", exc)
            return None

    def detect(
        self,
        frame=None,
        depth_frame=None,
        target_class: Optional[str] = None,
    ) -> List[Detection]:
        """
        Run object detection on a frame.

        Parameters
        ----------
        frame : np.ndarray, optional
            BGR image. If None, returns last detections.
        depth_frame : np.ndarray, optional
            Aligned depth image in mm for distance estimation.
        target_class : str, optional
            If set, only return detections of this class.

        Returns
        -------
        list of Detection
            Filtered detection results.
        """
        if frame is None:
            return self.last_detections

        if not _CV2_AVAILABLE or self._net is None:
            # Stub: return empty list
            self.last_detections = []
            return []

        try:
            detections = self._run_yolo(frame, depth_frame)
            if target_class:
                detections = [d for d in detections if d.class_name == target_class]
            self.last_detections = detections
            return detections
        except Exception as exc:
            log.error("ObjectDetector.detect error: %s", exc, exc_info=True)
            return []

    def _run_yolo(self, frame, depth_frame=None) -> List[Detection]:
        """
        Run YOLO inference on a frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image.
        depth_frame : np.ndarray, optional
            Depth image in mm.

        Returns
        -------
        list of Detection
        """
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self._net.setInput(blob)

        output_layers = self._net.getUnconnectedOutLayersNames()
        outputs = self._net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence < self.confidence_threshold:
                    continue
                cx, cy, bw, bh = detection[:4]
                x = int((cx - bw / 2) * w)
                y = int((cy - bh / 2) * h)
                boxes.append([x, y, int(bw * w), int(bh * h)])
                confidences.append(confidence)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self._nms_threshold)

        results = []
        for i in (indices.flatten() if len(indices) > 0 else []):
            box = tuple(boxes[i])
            class_name = (
                _HOUSEHOLD_CLASSES[class_ids[i]]
                if class_ids[i] < len(_HOUSEHOLD_CLASSES)
                else f"class_{class_ids[i]}"
            )
            depth_m = -1.0
            if depth_frame is not None:
                cx = boxes[i][0] + boxes[i][2] // 2
                cy = boxes[i][1] + boxes[i][3] // 2
                cx = max(0, min(cx, depth_frame.shape[1] - 1))
                cy = max(0, min(cy, depth_frame.shape[0] - 1))
                raw_depth = float(depth_frame[cy, cx])
                if raw_depth > 0:
                    depth_m = raw_depth / 1000.0

            results.append(Detection(
                class_name=class_name,
                confidence=confidences[i],
                bbox=box,
                depth_m=depth_m,
            ))

        return results
