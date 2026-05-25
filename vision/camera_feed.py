#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : vision/camera_feed.py
Purpose     : Continuous camera frame provider. Runs a background capture
              thread and exposes the latest RGB and depth frames to the
              vision pipeline and stream server.
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

from hardware.camera_manager import CameraManager

log = logging.getLogger(__name__)


class CameraFeed:
    """
    Continuous camera frame provider.

    Runs a background thread that captures frames from the CameraManager
    and stores the latest pair for consumers (vision pipeline, stream server).

    Attributes
    ----------
    is_running : bool
        True while the capture thread is active.
    frame_count : int
        Total frames captured since start.
    """

    def __init__(
        self,
        camera_manager: CameraManager,
        target_fps: int = 30,
    ) -> None:
        """
        Initialise the camera feed.

        Parameters
        ----------
        camera_manager : CameraManager
            Opened camera manager instance.
        target_fps : int
            Target capture rate. Actual rate depends on camera hardware.
        """
        self._camera = camera_manager
        self._target_fps = target_fps
        self._interval = 1.0 / max(1, target_fps)

        self._rgb_frame = None
        self._depth_frame = None
        self._frame_count: int = 0
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """True while the capture thread is active."""
        return self._running

    @property
    def frame_count(self) -> int:
        """Total frames captured since start."""
        with self._lock:
            return self._frame_count

    def start(self) -> None:
        """Start the background capture thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="kova-camera-feed",
            daemon=True,
        )
        self._thread.start()
        log.info("CameraFeed started (target=%dfps).", self._target_fps)

    def stop(self) -> None:
        """Stop the background capture thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        log.info("CameraFeed stopped.")

    def get_frames(self) -> Tuple[Optional[object], Optional[object]]:
        """
        Return the most recently captured frame pair.

        Returns
        -------
        tuple
            (rgb_frame, depth_frame) — either may be None.
        """
        with self._lock:
            return self._rgb_frame, self._depth_frame

    def get_rgb(self):
        """
        Return the most recent RGB frame.

        Returns
        -------
        np.ndarray or None
        """
        with self._lock:
            return self._rgb_frame

    def _capture_loop(self) -> None:
        """Background thread: capture frames at the target rate."""
        while self._running:
            t0 = time.monotonic()
            try:
                rgb, depth = self._camera.capture()
                with self._lock:
                    self._rgb_frame = rgb
                    self._depth_frame = depth
                    self._frame_count += 1
            except Exception as exc:
                log.error("CameraFeed capture error: %s", exc)

            elapsed = time.monotonic() - t0
            sleep_time = max(0.0, self._interval - elapsed)
            time.sleep(sleep_time)
