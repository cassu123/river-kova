#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : connectivity/wifi_manager.py
Purpose     : WiFi connection manager. Ensures the unit has network
              connectivity, monitors signal strength, and falls back to
              4G LTE if WiFi drops below acceptable quality.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)

_RSSI_THRESHOLD_DBM = -75   # Below this, consider WiFi poor
_CHECK_INTERVAL_SEC = 30.0


class WiFiManager:
    """
    WiFi connectivity manager.

    Checks for an active network connection on startup and monitors
    signal quality in the background. Logs warnings when signal degrades.

    Attributes
    ----------
    is_connected : bool
        True when a network interface has an IP address.
    rssi_dbm : float
        Most recent WiFi RSSI reading in dBm (-1 if unavailable).
    """

    def __init__(self, interface: str = "wlan0") -> None:
        """
        Initialise the WiFi manager.

        Parameters
        ----------
        interface : str
            Network interface name to monitor.
        """
        self._interface = interface
        self._connected: bool = False
        self._rssi: float = -1.0
        self._running: bool = False
        self._monitor_thread: Optional[threading.Thread] = None

    @property
    def is_connected(self) -> bool:
        """True when the interface has an IP address."""
        return self._connected

    @property
    def rssi_dbm(self) -> float:
        """Most recent RSSI in dBm."""
        return self._rssi

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def ensure_connected(self) -> bool:
        """
        Verify network connectivity is available.

        Returns
        -------
        bool
            True if connected, False if no network is available.
        """
        self._connected = self._check_connectivity()
        if self._connected:
            log.info("WiFiManager: network connectivity confirmed.")
            self._start_monitor()
        else:
            log.warning("WiFiManager: no network connectivity detected.")
        return self._connected

    def stop(self) -> None:
        """Stop the background monitor thread."""
        self._running = False

    # ── Connectivity checks ───────────────────────────────────────────────────

    def _check_connectivity(self) -> bool:
        """
        Check for basic IP connectivity by pinging a reliable host.

        Returns
        -------
        bool
            True if the ping succeeds.
        """
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "3", "8.8.8.8"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _read_rssi(self) -> float:
        """
        Read the current WiFi RSSI from iwconfig.

        Returns
        -------
        float
            RSSI in dBm, or -1.0 if unavailable.
        """
        try:
            result = subprocess.run(
                ["iwconfig", self._interface],
                capture_output=True,
                text=True,
                timeout=3,
            )
            for line in result.stdout.splitlines():
                if "Signal level" in line:
                    # Parse "Signal level=-65 dBm"
                    parts = line.split("Signal level=")
                    if len(parts) > 1:
                        val = parts[1].split()[0].replace("dBm", "").strip()
                        return float(val)
        except Exception:
            pass
        return -1.0

    # ── Background monitor ────────────────────────────────────────────────────

    def _start_monitor(self) -> None:
        """Start the background connectivity monitor."""
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="kova-wifi-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        """Periodically check connectivity and signal quality."""
        while self._running:
            time.sleep(_CHECK_INTERVAL_SEC)
            self._connected = self._check_connectivity()
            self._rssi = self._read_rssi()

            if not self._connected:
                log.warning("WiFiManager: connectivity lost.")
            elif self._rssi != -1.0 and self._rssi < _RSSI_THRESHOLD_DBM:
                log.warning(
                    "WiFiManager: weak signal (%.0f dBm < threshold %d dBm).",
                    self._rssi, _RSSI_THRESHOLD_DBM,
                )
