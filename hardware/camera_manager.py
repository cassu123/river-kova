#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : hardware/camera_manager.py
Purpose     : Camera hardware abstraction. Supports Intel RealSense depth
              cameras and standard USB/CSI cameras via OpenCV. Provides
              thread-safe frame capture for the vision pipeline.
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
from typing import Optional, Tuple

import numpy as np

from core.constants import CAMERA_FPS, CAMERA_HEIGHT, CAMERA_WIDTH

log = logging.getLogger(__name__)

try:
    import pyrealsense2 as rs
    _REALSENSE_AVAILABLE = True
except ImportError:
    _REALSENSE_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    log.warning("OpenCV not available — CameraManager running in stub mode.")


class CameraManagerError(Exception):
    """Raised when the camera cannot be opened or a frame cannot be captured."""


class CameraManager:
    """
    Camera hardware abstraction layer.

    Supports:
    - Intel RealSense D435 (RGB + depth)
    - Standard USB/CSI cameras via OpenCV

    Provides thread-safe access to the latest RGB and depth frames.

    Attributes
    ----------
    model : str
        Camera model identifier.
    is_open : bool
        True when the camera is streaming.
    """

    def __init__(
        self,
        model: str = "intel-realsense-d435",
        fps: int = CAMERA_FPS,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        device_index: int = 0,
    ) -> None:
        """
        Initialise the camera manager.

        Parameters
        ----------
        model : str
            Camera model string. 'intel-realsense-d435' uses the RealSense SDK;
            anything else falls back to OpenCV.
        fps : int
            Target frame rate.
        width : int
            Frame width in pixels.
        height : int
            Frame height in pixels.
        device_index : int
            OpenCV device index (ignored for RealSense).
        """
        self.model = model
        self._fps = fps
        self._width = width
        self._height = height
        self._device_index = device_index

        self._rgb_frame: Optional[np.ndarray] = None
        self._depth_frame: Optional[np.ndarray] = None
        self._frame_timestamp: float = 0.0
        self._lock = threading.Lock()
        self._open: bool = False

        # Backend handles
        self._rs_pipeline = None
        self._cv_cap = None

    # ── State ─────────────────────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        """True when the camera is streaming."""
        return self._open

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        """
        Open the camera and start streaming.

        Raises
        ------
        CameraManagerError
            If the camera cannot be opened.
        """
        if self._open:
            return

        if "realsense" in self.model.lower() and _REALSENSE_AVAILABLE:
            self._open_realsense()
        elif _CV2_AVAILABLE:
            self._open_opencv()
        else:
            log.warning("CameraManager: no camera backend available — stub mode.")
            self._open = True
            return

        log.info("CameraManager: %s opened (%dx%d @ %dfps).", self.model, self._width, self._height, self._fps)

    def close(self) -> None:
        """Stop streaming and release camera resources."""
        if self._rs_pipeline:
            try:
                self._rs_pipeline.stop()
            except Exception:
                pass
            self._rs_pipeline = None

        if self._cv_cap:
            try:
                self._cv_cap.release()
            except Exception:
                pass
            self._cv_cap = None

        self._open = False
        log.info("CameraManager: closed.")

    def stop(self) -> None:
        """Alias for close() — used by the shutdown sequence."""
        self.close()

    # ── Frame capture ─────────────────────────────────────────────────────────

    def capture(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture the latest RGB and depth frames.

        Returns
        -------
        tuple
            (rgb_frame, depth_frame) as numpy arrays.
            depth_frame is None for cameras without depth support.
            Both are None if the camera is not open or capture fails.
        """
        if not self._open:
            return None, None

        if self._rs_pipeline:
            return self._capture_realsense()
        elif self._cv_cap:
            return self._capture_opencv()
        else:
            # Stub: return a blank frame
            blank = np.zeros((self._height, self._width, 3), dtype=np.uint8)
            return blank, None

    def get_latest_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Return the most recently captured frames without triggering a new capture.

        Returns
        -------
        tuple
            (rgb_frame, depth_frame)
        """
        with self._lock:
            return self._rgb_frame, self._depth_frame

    # ── Backend implementations ───────────────────────────────────────────────

    def _open_realsense(self) -> None:
        """Open an Intel RealSense camera using the RealSense SDK."""
        try:
            self._rs_pipeline = rs.pipeline()
            rs_config = rs.config()
            rs_config.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)
            rs_config.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, self._fps)
            self._rs_pipeline.start(rs_config)
            self._open = True
        except Exception as exc:
            raise CameraManagerError(f"RealSense open failed: {exc}") from exc

    def _open_opencv(self) -> None:
        """Open a standard camera using OpenCV."""
        try:
            self._cv_cap = cv2.VideoCapture(self._device_index)
            self._cv_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cv_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cv_cap.set(cv2.CAP_PROP_FPS, self._fps)
            if not self._cv_cap.isOpened():
                raise CameraManagerError(f"OpenCV cannot open camera index {self._device_index}.")
            self._open = True
        except CameraManagerError:
            raise
        except Exception as exc:
            raise CameraManagerError(f"OpenCV camera open failed: {exc}") from exc

    def _capture_realsense(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture a frame pair from the RealSense pipeline."""
        try:
            frames = self._rs_pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            rgb = np.asanyarray(color_frame.get_data()) if color_frame else None
            depth = np.asanyarray(depth_frame.get_data()) if depth_frame else None
            with self._lock:
                self._rgb_frame = rgb
                self._depth_frame = depth
                self._frame_timestamp = time.monotonic()
            return rgb, depth
        except Exception as exc:
            log.error("RealSense capture error: %s", exc)
            return None, None

    def _capture_opencv(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture a frame from the OpenCV camera."""
        try:
            ret, frame = self._cv_cap.read()
            if not ret:
                log.warning("CameraManager: OpenCV frame read failed.")
                return None, None
            with self._lock:
                self._rgb_frame = frame
                self._depth_frame = None
                self._frame_timestamp = time.monotonic()
            return frame, None
        except Exception as exc:
            log.error("OpenCV capture error: %s", exc)
            return None, None
