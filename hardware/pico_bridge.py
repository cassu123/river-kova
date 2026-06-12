#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : hardware/pico_bridge.py
Purpose     : Serial communication bridge to the Raspberry Pi Pico.
              The Pico handles low-level I/O: motor PWM, encoder reads,
              force sensors, battery ADC, and GPIO. All higher-level modules
              communicate with the Pico through this bridge.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, Optional

from core.constants import PICO_BAUD_RATE, PICO_SERIAL_TIMEOUT

log = logging.getLogger(__name__)

try:
    import serial
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False
    log.warning("pyserial not available — PicoBridge running in stub mode.")


class PicoBridgeError(Exception):
    """Raised when the Pico bridge encounters a communication error."""


class PicoBridge:
    """
    Serial bridge to the Raspberry Pi Pico microcontroller.

    Sends JSON command frames and receives JSON telemetry frames over UART.
    All public methods are thread-safe.

    Protocol
    --------
    Commands  : {"cmd": "<name>", "params": {...}}\\n
    Responses : {"status": "ok"|"err", "data": {...}}\\n

    Attributes
    ----------
    port : str
        Serial port path (e.g. '/dev/ttyACM0').
    baud_rate : int
        UART baud rate.
    is_connected : bool
        True when the serial port is open and the Pico has acknowledged.
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baud_rate: int = PICO_BAUD_RATE,
        timeout: float = PICO_SERIAL_TIMEOUT,
    ) -> None:
        """
        Initialise the Pico bridge.

        Parameters
        ----------
        port : str
            Serial device path.
        baud_rate : int
            UART baud rate (must match Pico firmware).
        timeout : float
            Read timeout in seconds.
        """
        self.port = port
        self.baud_rate = baud_rate
        self._timeout = timeout
        self._serial: Optional[Any] = None
        self._lock = threading.Lock()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """True when the serial port is open."""
        return self._connected

    # ── Connection lifecycle ──────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Open the serial port and verify Pico is responsive.

        Raises
        ------
        PicoBridgeError
            If the port cannot be opened or the Pico does not respond.
        """
        if not _SERIAL_AVAILABLE:
            log.warning("PicoBridge.connect() — stub mode, no real serial.")
            self._connected = True
            return

        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self._timeout,
            )
            time.sleep(0.5)  # Allow Pico to reset after DTR toggle
            self._serial.reset_input_buffer()
            self._connected = True
            log.info("PicoBridge connected on %s @ %d baud.", self.port, self.baud_rate)
            self._ping()
        except serial.SerialException as exc:
            raise PicoBridgeError(f"Cannot open serial port {self.port}: {exc}") from exc

    def disconnect(self) -> None:
        """Close the serial port."""
        with self._lock:
            if self._serial and self._serial.is_open:
                self._serial.close()
            self._connected = False
        log.info("PicoBridge disconnected.")

    def stop(self) -> None:
        """Alias for disconnect — used by the shutdown sequence."""
        self.disconnect()

    # ── Command interface ─────────────────────────────────────────────────────

    def send_command(self, cmd: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send a command to the Pico and return the response.

        Parameters
        ----------
        cmd : str
            Command name (e.g. 'set_motor', 'read_battery').
        params : dict, optional
            Command parameters.

        Returns
        -------
        dict
            Parsed JSON response from the Pico.

        Raises
        ------
        PicoBridgeError
            On communication failure or error response.
        """
        if not self._connected:
            raise PicoBridgeError("PicoBridge is not connected.")

        payload = json.dumps({"cmd": cmd, "params": params or {}}) + "\n"

        if not _SERIAL_AVAILABLE:
            # Stub mode — return a synthetic OK response with sane defaults
            # (full battery, zero force) so safety logic doesn't trip on
            # phantom readings.
            log.debug("PicoBridge STUB send: %s", payload.strip())
            return {
                "status": "ok",
                "data": {"percent": 100.0, "force_n": 0.0, "left": 0, "right": 0},
            }

        with self._lock:
            try:
                self._serial.write(payload.encode("utf-8"))
                raw = self._serial.readline()
                if not raw:
                    raise PicoBridgeError(f"No response from Pico for command '{cmd}'.")
                response = json.loads(raw.decode("utf-8").strip())
                if response.get("status") == "err":
                    raise PicoBridgeError(
                        f"Pico error for '{cmd}': {response.get('data', {})}"
                    )
                return response
            except (serial.SerialException, json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise PicoBridgeError(f"PicoBridge communication error: {exc}") from exc

    # ── Convenience wrappers ──────────────────────────────────────────────────

    def set_motor_speeds(self, left: float, right: float) -> None:
        """
        Set differential drive motor speeds.

        Parameters
        ----------
        left : float
            Left motor speed in range [-1.0, 1.0].
        right : float
            Right motor speed in range [-1.0, 1.0].
        """
        left = max(-1.0, min(1.0, left))
        right = max(-1.0, min(1.0, right))
        self.send_command("set_motor", {"left": left, "right": right})

    def emergency_stop_motors(self) -> None:
        """Send an immediate motor stop command — bypasses speed ramping."""
        self.send_command("estop_motors", {})
        log.warning("PicoBridge: emergency motor stop sent.")

    def read_battery(self) -> float:
        """
        Read the battery voltage from the Pico ADC.

        Returns
        -------
        float
            Battery percentage [0.0, 100.0].
        """
        response = self.send_command("read_battery")
        return float(response.get("data", {}).get("percent", 0.0))

    def read_encoders(self) -> Dict[str, int]:
        """
        Read wheel encoder tick counts.

        Returns
        -------
        dict
            {'left': int, 'right': int} encoder tick counts.
        """
        response = self.send_command("read_encoders")
        return response.get("data", {"left": 0, "right": 0})

    def read_force_sensor(self) -> float:
        """
        Read the force/torque sensor value.

        Returns
        -------
        float
            Force in Newtons.
        """
        response = self.send_command("read_force")
        return float(response.get("data", {}).get("force_n", 0.0))

    def set_gripper(self, position: float) -> None:
        """
        Set gripper position.

        Parameters
        ----------
        position : float
            Gripper position [0.0 = closed, 1.0 = fully open].
        """
        position = max(0.0, min(1.0, position))
        self.send_command("set_gripper", {"position": position})

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ping(self) -> None:
        """
        Send a ping to verify the Pico is responsive.

        Raises
        ------
        PicoBridgeError
            If the Pico does not respond to the ping.
        """
        response = self.send_command("ping")
        if response.get("status") != "ok":
            raise PicoBridgeError("Pico ping failed — firmware may not be running.")
        log.debug("PicoBridge ping OK.")
