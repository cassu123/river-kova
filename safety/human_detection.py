#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : safety/human_detection.py
Purpose     : Real-time human proximity detection using MediaPipe Pose and
              depth camera data. Enforces three concentric safety zones:
              WARN → SLOW → ESTOP. Immediate stop if a human enters the
              1-metre e-stop radius.
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
from typing import Callable, List, Optional, Tuple

from core.constants import (
    HUMAN_PROXIMITY_ESTOP,
    HUMAN_PROXIMITY_SLOW,
    HUMAN_PROXIMITY_WARN,
    MEDIAPIPE_MODEL_COMPLEXITY,
)

log = logging.getLogger(__name__)

# Lazy imports — MediaPipe and OpenCV are optional at import time
# so the module can be loaded in environments without them (e.g. CI).
try:
    import cv2
    import mediapipe as mp
    import numpy as np
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False
    log.warning("MediaPipe / OpenCV not available — HumanDetection running in stub mode.")


class HumanDetection:
    """
    Human proximity detection and safety zone enforcement.

    Uses MediaPipe Pose to detect human skeletons in the camera frame and
    the depth channel (if available) to estimate distance. Falls back to
    bounding-box area heuristics when depth is unavailable.

    Three zones (configurable via kova_profile.json):
    - WARN  zone : slow alert, continue at reduced speed
    - SLOW  zone : reduce speed to safe_speed
    - ESTOP zone : immediate full stop (default 1.0 m)

    Attributes
    ----------
    human_in_estop_zone : bool
        True when a human is within the e-stop radius.
    human_in_slow_zone : bool
        True when a human is within the slow-down radius.
    human_in_warn_zone : bool
        True when a human is within the warning radius.
    nearest_human_distance_m : float
        Distance to the nearest detected human in metres (-1 if none).
    """

    def __init__(
        self,
        stop_radius_m: float = HUMAN_PROXIMITY_ESTOP,
        slow_radius_m: float = HUMAN_PROXIMITY_SLOW,
        warn_radius_m: float = HUMAN_PROXIMITY_WARN,
        on_human_detected: Optional[Callable[[float], None]] = None,
        model_complexity: int = MEDIAPIPE_MODEL_COMPLEXITY,
    ) -> None:
        """
        Initialise the human detection module.

        Parameters
        ----------
        stop_radius_m : float
            Distance (metres) at which an immediate e-stop is triggered.
        slow_radius_m : float
            Distance at which the robot slows to safe speed.
        warn_radius_m : float
            Distance at which a warning is issued.
        on_human_detected : callable, optional
            Called with the distance (m) when a human enters the e-stop zone.
        model_complexity : int
            MediaPipe model complexity (0=lite, 1=full, 2=heavy).
        """
        self._stop_radius = stop_radius_m
        self._slow_radius = slow_radius_m
        self._warn_radius = warn_radius_m
        self._on_human_detected = on_human_detected

        self._nearest_distance: float = -1.0
        self._lock = threading.Lock()

        # MediaPipe pose detector
        self._pose = None
        if _MEDIAPIPE_AVAILABLE:
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                enable_segmentation=False,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
            )
            log.info(
                "HumanDetection initialised (stop=%.1fm, slow=%.1fm, warn=%.1fm).",
                stop_radius_m, slow_radius_m, warn_radius_m,
            )
        else:
            log.warning("HumanDetection running in stub mode — no real detection.")

    # ── Zone state properties ─────────────────────────────────────────────────

    @property
    def nearest_human_distance_m(self) -> float:
        """Distance to the nearest detected human in metres (-1 if none)."""
        with self._lock:
            return self._nearest_distance

    @property
    def human_in_estop_zone(self) -> bool:
        """True when a human is within the e-stop radius."""
        d = self.nearest_human_distance_m
        return 0.0 <= d <= self._stop_radius

    @property
    def human_in_slow_zone(self) -> bool:
        """True when a human is within the slow-down radius (but outside e-stop)."""
        d = self.nearest_human_distance_m
        return self._stop_radius < d <= self._slow_radius

    @property
    def human_in_warn_zone(self) -> bool:
        """True when a human is within the warning radius (but outside slow zone)."""
        d = self.nearest_human_distance_m
        return self._slow_radius < d <= self._warn_radius

    # ── Frame processing ──────────────────────────────────────────────────────

    def process_frame(
        self,
        rgb_frame,
        depth_frame=None,
    ) -> float:
        """
        Process a single camera frame and update proximity state.

        Parameters
        ----------
        rgb_frame : np.ndarray
            BGR or RGB image from the camera (H×W×3).
        depth_frame : np.ndarray, optional
            Aligned depth image in millimetres (H×W). Used for accurate
            distance estimation when available.

        Returns
        -------
        float
            Distance to the nearest detected human in metres, or -1 if none.
        """
        if not _MEDIAPIPE_AVAILABLE or self._pose is None:
            return -1.0

        try:
            import numpy as np
            frame_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
            results = self._pose.process(frame_rgb)

            if not results.pose_landmarks:
                self._set_distance(-1.0)
                return -1.0

            distance_m = self._estimate_distance(
                results.pose_landmarks,
                rgb_frame.shape,
                depth_frame,
            )
            self._set_distance(distance_m)

            if self.human_in_estop_zone and self._on_human_detected:
                try:
                    self._on_human_detected(distance_m)
                except Exception as exc:
                    log.error("on_human_detected callback error: %s", exc)

            return distance_m

        except Exception as exc:
            log.error("HumanDetection.process_frame error: %s", exc, exc_info=True)
            return -1.0

    def _estimate_distance(
        self,
        landmarks,
        frame_shape: Tuple[int, int, int],
        depth_frame=None,
    ) -> float:
        """
        Estimate the distance to the detected human.

        Uses depth data when available; falls back to bounding-box heuristic.

        Parameters
        ----------
        landmarks : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
            Detected pose landmarks.
        frame_shape : tuple
            (height, width, channels) of the RGB frame.
        depth_frame : np.ndarray, optional
            Depth image in millimetres.

        Returns
        -------
        float
            Estimated distance in metres.
        """
        import numpy as np
        import mediapipe as mp

        h, w = frame_shape[:2]
        lm = landmarks.landmark

        # Use torso centre (midpoint of shoulders) as the reference point
        left_shoulder = lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        cx = int(((left_shoulder.x + right_shoulder.x) / 2) * w)
        cy = int(((left_shoulder.y + right_shoulder.y) / 2) * h)
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))

        if depth_frame is not None:
            # Sample a small patch around the torso centre for robustness
            patch_size = 5
            y1 = max(0, cy - patch_size)
            y2 = min(h, cy + patch_size)
            x1 = max(0, cx - patch_size)
            x2 = min(w, cx + patch_size)
            patch = depth_frame[y1:y2, x1:x2]
            valid = patch[patch > 0]
            if valid.size > 0:
                return float(np.median(valid)) / 1000.0  # mm → m

        # Fallback: bounding-box height heuristic
        # Assumes average human torso height ~0.5 m maps to known pixel heights
        ys = [lm[i].y for i in range(len(lm)) if lm[i].visibility > 0.5]
        if len(ys) < 2:
            return 2.0  # Conservative default
        bbox_height_px = (max(ys) - min(ys)) * h
        if bbox_height_px < 1:
            return 2.0
        # Empirical constant: ~500 px at 1 m for 720p
        estimated_m = 500.0 / bbox_height_px
        return float(np.clip(estimated_m, 0.1, 10.0))

    def _set_distance(self, distance_m: float) -> None:
        """Thread-safe update of the nearest human distance."""
        with self._lock:
            self._nearest_distance = distance_m

    def shutdown(self) -> None:
        """Release MediaPipe resources."""
        if self._pose:
            self._pose.close()
            log.debug("HumanDetection MediaPipe pose closed.")
