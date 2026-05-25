#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : connectivity/vpn.py
Purpose     : WireGuard VPN lifecycle manager. Brings the VPN tunnel up on
              boot and monitors it for drops, reconnecting automatically.
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
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_RECONNECT_INTERVAL_SEC = 30.0
_HEALTH_CHECK_INTERVAL_SEC = 60.0


class VPNError(Exception):
    """Raised when the VPN cannot be brought up."""


class VPNManager:
    """
    WireGuard VPN lifecycle manager.

    Brings the wg-quick tunnel up on connect(), monitors it in a background
    thread, and reconnects automatically if the tunnel drops.

    Attributes
    ----------
    is_connected : bool
        True when the WireGuard interface is active.
    config_path : str
        Path to the WireGuard configuration file.
    """

    def __init__(
        self,
        config_path: str = "/etc/wireguard/wg0.conf",
        interface: str = "wg0",
    ) -> None:
        """
        Initialise the VPN manager.

        Parameters
        ----------
        config_path : str
            Path to the WireGuard .conf file.
        interface : str
            WireGuard interface name (derived from config filename by default).
        """
        self.config_path = config_path
        self._interface = interface or Path(config_path).stem
        self._connected: bool = False
        self._running: bool = False
        self._monitor_thread: Optional[threading.Thread] = None

    @property
    def is_connected(self) -> bool:
        """True when the WireGuard interface is active."""
        return self._connected

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Bring up the WireGuard tunnel.

        Raises
        ------
        VPNError
            If wg-quick fails to bring the interface up.
        """
        if self._connected:
            log.debug("VPN already connected.")
            return

        log.info("Bringing up WireGuard VPN (%s)...", self._interface)
        try:
            subprocess.run(
                ["wg-quick", "up", self.config_path],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self._connected = True
            log.info("WireGuard VPN up on interface '%s'.", self._interface)
            self._start_monitor()
        except subprocess.CalledProcessError as exc:
            raise VPNError(
                f"wg-quick up failed: {exc.stderr.strip()}"
            ) from exc
        except FileNotFoundError:
            raise VPNError("wg-quick not found — is WireGuard installed?")

    def disconnect(self) -> None:
        """Bring down the WireGuard tunnel."""
        self._running = False
        if not self._connected:
            return
        try:
            subprocess.run(
                ["wg-quick", "down", self.config_path],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self._connected = False
            log.info("WireGuard VPN down.")
        except subprocess.CalledProcessError as exc:
            log.error("wg-quick down failed: %s", exc.stderr.strip())

    def stop(self) -> None:
        """Alias for disconnect() — used by the shutdown sequence."""
        self.disconnect()

    # ── Health monitoring ─────────────────────────────────────────────────────

    def _start_monitor(self) -> None:
        """Start the background VPN health monitor thread."""
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="kova-vpn-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        """Periodically check VPN health and reconnect if needed."""
        while self._running:
            time.sleep(_HEALTH_CHECK_INTERVAL_SEC)
            if not self._running:
                break
            if not self._check_interface_up():
                log.warning("VPN interface '%s' is down — reconnecting...", self._interface)
                self._connected = False
                self._reconnect()

    def _check_interface_up(self) -> bool:
        """
        Check whether the WireGuard interface is currently active.

        Returns
        -------
        bool
            True if the interface exists and has a peer.
        """
        try:
            result = subprocess.run(
                ["wg", "show", self._interface],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0 and "peer" in result.stdout.lower()
        except Exception:
            return False

    def _reconnect(self) -> None:
        """Attempt to reconnect the VPN with backoff."""
        for attempt in range(1, 4):
            log.info("VPN reconnect attempt %d/3...", attempt)
            try:
                self.connect()
                return
            except VPNError as exc:
                log.error("VPN reconnect attempt %d failed: %s", attempt, exc)
                time.sleep(_RECONNECT_INTERVAL_SEC * attempt)
        log.critical("VPN reconnect failed after 3 attempts.")
