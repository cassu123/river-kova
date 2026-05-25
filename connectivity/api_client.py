#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : connectivity/api_client.py
Purpose     : River Song API client. All communication with the River Song
              ecosystem goes through this module. Handles authentication,
              retries, heartbeats, task polling, and telemetry pushes.
              API routes are mounted under /api/kova/ on the River Song server.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.constants import DEFAULT_API_TIMEOUT, RIVER_SONG_API_BASE

log = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_FACTOR = 0.5


class RiverSongAPIError(Exception):
    """Raised when the River Song API returns an error or is unreachable."""


class RiverSongAPIClient:
    """
    HTTP client for the River Song AI ecosystem API.

    All Kova-specific endpoints are under /api/kova/.
    Provides automatic retry with exponential backoff and bearer token auth.

    Attributes
    ----------
    base_url : str
        Root URL of the River Song API server.
    robot_id : str
        Unit identifier sent with every request.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        robot_id: str,
        timeout: int = DEFAULT_API_TIMEOUT,
    ) -> None:
        """
        Initialise the API client.

        Parameters
        ----------
        base_url : str
            Root URL (e.g. 'https://api.riversongai.com').
        api_key : str
            Bearer token for authentication.
        robot_id : str
            Unique unit identifier.
        timeout : int
            Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.robot_id = robot_id
        self._api_key = api_key
        self._timeout = timeout
        self._session = self._build_session()

        log.info("RiverSongAPIClient initialised (base=%s, unit=%s).", base_url, robot_id)

    # ── Session setup ─────────────────────────────────────────────────────────

    def _build_session(self) -> requests.Session:
        """
        Build a requests Session with retry logic and auth headers.

        Returns
        -------
        requests.Session
        """
        session = requests.Session()
        retry = Retry(
            total=_MAX_RETRIES,
            backoff_factor=_BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "PATCH"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Kova-Unit": self.robot_id,
        })
        return session

    # ── Unit registration ─────────────────────────────────────────────────────

    def register_unit(self) -> bool:
        """
        Register this unit with the River Song server on boot.

        Returns
        -------
        bool
            True if registration succeeded.
        """
        try:
            response = self._post(
                "/api/kova/units/register",
                {"robot_id": self.robot_id, "timestamp": time.time()},
            )
            log.info("Unit '%s' registered with River Song.", self.robot_id)
            return True
        except RiverSongAPIError as exc:
            log.warning("Unit registration failed (non-fatal): %s", exc)
            return False

    def deregister_unit(self) -> bool:
        """
        Deregister this unit on shutdown.

        Returns
        -------
        bool
            True if deregistration succeeded.
        """
        try:
            self._post(
                "/api/kova/units/deregister",
                {"robot_id": self.robot_id, "timestamp": time.time()},
            )
            log.info("Unit '%s' deregistered.", self.robot_id)
            return True
        except RiverSongAPIError as exc:
            log.warning("Unit deregistration failed: %s", exc)
            return False

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    def heartbeat(
        self,
        state: str,
        safety_level: str,
        battery_pct: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send a periodic heartbeat to River Song.

        Parameters
        ----------
        state : str
            Current RobotState value.
        safety_level : str
            Current SafetyLevel value.
        battery_pct : float
            Battery percentage.
        extra : dict, optional
            Additional key-value pairs to include.

        Returns
        -------
        bool
            True if the heartbeat was acknowledged.
        """
        payload = {
            "robot_id": self.robot_id,
            "state": state,
            "safety_level": safety_level,
            "battery_pct": battery_pct,
            "timestamp": time.time(),
        }
        if extra:
            payload.update(extra)

        try:
            self._post("/api/kova/heartbeat", payload)
            return True
        except RiverSongAPIError as exc:
            log.warning("Heartbeat failed: %s", exc)
            return False

    # ── Task polling ──────────────────────────────────────────────────────────

    def poll_tasks(self) -> List[Dict[str, Any]]:
        """
        Poll River Song for pending tasks assigned to this unit.

        Returns
        -------
        list of dict
            Task descriptors. Empty list if none pending or on error.
        """
        try:
            data = self._get(f"/api/kova/units/{self.robot_id}/tasks")
            return data.get("tasks", [])
        except RiverSongAPIError as exc:
            log.warning("Task poll failed: %s", exc)
            return []

    def report_task_status(
        self,
        task_id: str,
        status: str,
        message: str = "",
    ) -> bool:
        """
        Report the status of a task back to River Song.

        Parameters
        ----------
        task_id : str
            Unique task identifier.
        status : str
            TaskStatus value (e.g. 'COMPLETED', 'FAILED').
        message : str
            Optional human-readable status message.

        Returns
        -------
        bool
            True if the report was acknowledged.
        """
        try:
            self._post(
                f"/api/kova/tasks/{task_id}/status",
                {
                    "robot_id": self.robot_id,
                    "status": status,
                    "message": message,
                    "timestamp": time.time(),
                },
            )
            return True
        except RiverSongAPIError as exc:
            log.warning("Task status report failed: %s", exc)
            return False

    # ── Telemetry push ────────────────────────────────────────────────────────

    def push_telemetry(self, metrics: Dict[str, Any]) -> bool:
        """
        Push a telemetry snapshot to River Song.

        Parameters
        ----------
        metrics : dict
            Key-value metric pairs.

        Returns
        -------
        bool
            True if accepted.
        """
        payload = {
            "robot_id": self.robot_id,
            "timestamp": time.time(),
            "metrics": metrics,
        }
        try:
            self._post("/api/kova/telemetry", payload)
            return True
        except RiverSongAPIError as exc:
            log.debug("Telemetry push failed: %s", exc)
            return False

    # ── Alert push ────────────────────────────────────────────────────────────

    def push_alert(self, level: str, message: str) -> bool:
        """
        Push an alert to River Song for display in the dashboard.

        Parameters
        ----------
        level : str
            Severity: 'INFO', 'WARN', 'ERROR', 'CRITICAL'.
        message : str
            Alert message text.

        Returns
        -------
        bool
            True if accepted.
        """
        try:
            self._post(
                "/api/kova/alerts",
                {
                    "robot_id": self.robot_id,
                    "level": level,
                    "message": message,
                    "timestamp": time.time(),
                },
            )
            return True
        except RiverSongAPIError as exc:
            log.warning("Alert push failed: %s", exc)
            return False

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _get(self, path: str) -> Dict[str, Any]:
        """
        Perform a GET request.

        Parameters
        ----------
        path : str
            API path (appended to base_url).

        Returns
        -------
        dict
            Parsed JSON response body.

        Raises
        ------
        RiverSongAPIError
            On HTTP error or network failure.
        """
        url = self.base_url + path
        try:
            resp = self._session.get(url, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise RiverSongAPIError(f"GET {path} failed: {exc}") from exc

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a POST request.

        Parameters
        ----------
        path : str
            API path.
        payload : dict
            JSON body.

        Returns
        -------
        dict
            Parsed JSON response body.

        Raises
        ------
        RiverSongAPIError
            On HTTP error or network failure.
        """
        url = self.base_url + path
        try:
            resp = self._session.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json() if resp.content else {}
        except requests.RequestException as exc:
            raise RiverSongAPIError(f"POST {path} failed: {exc}") from exc
