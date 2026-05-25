#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : vision/stream_server.py
Purpose     : MJPEG live camera stream server. Serves the camera feed over
              HTTP so the River Song dashboard can display live video.
              Runs as a background thread on a configurable port.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import io
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from core.constants import STREAM_PORT
from vision.camera_feed import CameraFeed

log = logging.getLogger(__name__)

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


class StreamServer:
    """
    MJPEG HTTP stream server for the live camera feed.

    Serves a multipart/x-mixed-replace MJPEG stream at:
      http://<unit-ip>:<port>/stream

    Attributes
    ----------
    port : int
        HTTP port the server listens on.
    is_running : bool
        True while the server thread is active.
    """

    def __init__(
        self,
        camera_feed: CameraFeed,
        port: int = STREAM_PORT,
        jpeg_quality: int = 70,
    ) -> None:
        """
        Initialise the stream server.

        Parameters
        ----------
        camera_feed : CameraFeed
            Source of live frames.
        port : int
            HTTP port to listen on.
        jpeg_quality : int
            JPEG compression quality [1, 100].
        """
        self._feed = camera_feed
        self.port = port
        self._quality = jpeg_quality
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False

    @property
    def is_running(self) -> bool:
        """True while the server thread is active."""
        return self._running

    def start(self) -> None:
        """Start the MJPEG stream server in a background thread."""
        if self._running:
            return

        if not _CV2_AVAILABLE:
            log.warning("StreamServer: OpenCV not available — stream disabled.")
            return

        # Start the camera feed if not already running
        if not self._feed.is_running:
            self._feed.start()

        feed_ref = self._feed
        quality = self._quality

        class MJPEGHandler(BaseHTTPRequestHandler):
            """HTTP request handler for MJPEG streaming."""

            def log_message(self, format, *args):
                """Suppress default HTTP access log."""
                pass

            def do_GET(self):
                """Handle GET requests."""
                if self.path == "/stream":
                    self._stream()
                elif self.path == "/snapshot":
                    self._snapshot()
                elif self.path == "/health":
                    self._health()
                else:
                    self.send_error(404, "Not Found")

            def _stream(self):
                """Stream MJPEG frames."""
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    while True:
                        frame = feed_ref.get_rgb()
                        if frame is None:
                            time.sleep(0.05)
                            continue
                        ret, jpeg = cv2.imencode(
                            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
                        )
                        if not ret:
                            continue
                        data = jpeg.tobytes()
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
                        time.sleep(1.0 / 15)  # 15 fps stream
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _snapshot(self):
                """Return a single JPEG snapshot."""
                frame = feed_ref.get_rgb()
                if frame is None:
                    self.send_error(503, "No frame available")
                    return
                ret, jpeg = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
                if not ret:
                    self.send_error(500, "Encode failed")
                    return
                data = jpeg.tobytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _health(self):
                """Return a simple health check response."""
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"OK")

        try:
            self._server = HTTPServer(("0.0.0.0", self.port), MJPEGHandler)
            self._running = True
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name="kova-stream-server",
                daemon=True,
            )
            self._thread.start()
            log.info(
                "StreamServer started on port %d (stream: http://0.0.0.0:%d/stream).",
                self.port, self.port,
            )
        except OSError as exc:
            log.error("StreamServer failed to start on port %d: %s", self.port, exc)

    def stop(self) -> None:
        """Stop the stream server."""
        self._running = False
        if self._server:
            self._server.shutdown()
        self._feed.stop()
        log.info("StreamServer stopped.")
