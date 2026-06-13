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
import queue
import threading
import time
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.constants import DEFAULT_API_TIMEOUT, RIVER_SONG_API_BASE

log = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_FACTOR = 0.5
_ASYNC_QUEUE_SIZE = 200        # Pending fire-and-forget messages
_ASYNC_TIMEOUT_SEC = 5         # Short timeout for background sends


class RiverSongAPIError(Exception):
    """Raised when the River Song API returns an error or is unreachable."""


class RiverSongAPIClient:
    """
    HTTP client for the River Song AI ecosystem API.

    All Kova-specific endpoints are under /api/kova/.
    Provides automatic retry with exponential backoff and bearer token auth.

    Periodic messages (heartbeat, telemetry, alerts, task status) are sent
    from a background worker thread so they can NEVER block the main control
    loop — a slow or unreachable server must not starve the watchdog.

    When `enabled` is False the client runs fully offline: every method
    becomes a local no-op and the robot is self-hosted only. This is the mode
    to use until the River Song server side exists.

    Attributes
    ----------
    base_url : str
        Root URL of the River Song API server.
    robot_id : str
        Unit identifier sent with every request.
    enabled : bool
        False = offline / self-hosted mode, no network calls at all.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        robot_id: str,
        timeout: int = DEFAULT_API_TIMEOUT,
        enabled: bool = True,
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
            Request timeout in seconds (synchronous calls only).
        enabled : bool
            False disables all network traffic (offline / self-hosted mode).
        """
        self.base_url = base_url.rstrip("/")
        self.robot_id = robot_id
        self.enabled = enabled
        self._api_key = api_key
        self._timeout = timeout
        self._session = self._build_session()
        # Async sends never retry — the next heartbeat/telemetry push
        # supersedes a lost one, and retries would delay shutdown.
        self._async_session = self._build_session(retries=0)

        # Background sender — fire-and-forget queue drained by a worker thread
        self._send_queue: "queue.Queue[tuple]" = queue.Queue(maxsize=_ASYNC_QUEUE_SIZE)
        self._worker_running = False
        self._worker: Optional[threading.Thread] = None
        if self.enabled:
            self._start_worker()
            log.info("RiverSongAPIClient initialised (base=%s, unit=%s).", base_url, robot_id)
        else:
            log.info("RiverSongAPIClient in OFFLINE mode — unit '%s' is self-hosted only.", robot_id)

    # ── Background sender ─────────────────────────────────────────────────────

    def _start_worker(self) -> None:
        """Start the background send worker thread."""
        self._worker_running = True
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="kova-api-sender",
            daemon=True,
        )
        self._worker.start()

    def _worker_loop(self) -> None:
        """Drain the send queue, posting each message with a short timeout."""
        while self._worker_running:
            try:
                path, payload = self._send_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._post(path, payload, timeout=_ASYNC_TIMEOUT_SEC, session=self._async_session)
            except RiverSongAPIError as exc:
                log.debug("Async send to %s failed (dropped): %s", path, exc)

    def _enqueue(self, path: str, payload: Dict[str, Any]) -> bool:
        """
        Queue a message for background delivery. Never blocks.

        Drops the oldest pending message if the queue is full — losing a
        stale heartbeat is preferable to stalling the control loop.
        """
        if not self.enabled:
            return True
        try:
            self._send_queue.put_nowait((path, payload))
        except queue.Full:
            try:
                self._send_queue.get_nowait()
                self._send_queue.put_nowait((path, payload))
            except (queue.Empty, queue.Full):
                pass
        return True

    def stop(self) -> None:
        """Stop the background sender thread."""
        self._worker_running = False
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=3.0)

    # ── Session setup ─────────────────────────────────────────────────────────

    def _build_session(self, retries: int = _MAX_RETRIES) -> requests.Session:
        """
        Build a requests Session with retry logic and auth headers.

        Parameters
        ----------
        retries : int
            Total retry count (0 disables retries — used for async sends).

        Returns
        -------
        requests.Session
        """
        session = requests.Session()
        retry = Retry(
            total=retries,
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
        if not self.enabled:
            return True
        try:
            response = self._post(
                "/api/kova/units/register",
                {"robot_id": self.robot_id, "timestamp": time.time()},
                timeout=_ASYNC_TIMEOUT_SEC,
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
        if not self.enabled:
            return True
        try:
            self._post(
                "/api/kova/units/deregister",
                {"robot_id": self.robot_id, "timestamp": time.time()},
                timeout=_ASYNC_TIMEOUT_SEC,
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

        # Non-blocking: queued for the background sender so the control loop
        # (and therefore the watchdog) can never stall on a slow network.
        return self._enqueue("/api/kova/heartbeat", payload)

    # ── Task polling ──────────────────────────────────────────────────────────

    def poll_tasks(self) -> List[Dict[str, Any]]:
        """
        Poll River Song for pending tasks assigned to this unit.

        Returns
        -------
        list of dict
            Task descriptors. Empty list if none pending or on error.
        """
        if not self.enabled:
            return []
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
        return self._enqueue(
            f"/api/kova/tasks/{task_id}/status",
            {
                "robot_id": self.robot_id,
                "status": status,
                "message": message,
                "timestamp": time.time(),
            },
        )

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
        return self._enqueue("/api/kova/telemetry", payload)

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
        return self._enqueue(
            "/api/kova/alerts",
            {
                "robot_id": self.robot_id,
                "level": level,
                "message": message,
                "timestamp": time.time(),
            },
        )

    # ── LLM brain (server-side) ───────────────────────────────────────────────

    def request_initiative(
        self,
        context: str,
        state: Dict[str, Any],
        catalogue: List[Dict[str, Any]],
        timeout: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ask the River Song server's planner which chores to start now.

        The LLM lives on the server, not the robot: one credential, one bill,
        and server-side caching/batching across the whole fleet — the unit
        never calls an online model directly. Synchronous request/response.

        IMPORTANT: this blocks on the network, so it MUST be called off the
        control-loop thread (the initiative rule runs it on a worker).

        Parameters
        ----------
        context : str
            Natural-language household context.
        state : dict
            Structured robot state (battery, known rooms, sightings).
        catalogue : list of dict
            The chores this body can actually run — so the server only
            proposes things the unit supports.
        timeout : int, optional
            Per-request timeout override.

        Returns
        -------
        list of dict
            Proposal dicts, or [] when offline or on any error.
        """
        if not self.enabled:
            return []
        try:
            data = self._post(
                "/api/kova/initiative",
                {
                    "robot_id": self.robot_id,
                    "context": context,
                    "state": state,
                    "catalogue": catalogue,
                    "timestamp": time.time(),
                },
                timeout=timeout or self._timeout,
            )
            return data.get("proposals", [])
        except RiverSongAPIError as exc:
            log.warning("Initiative request failed (non-fatal): %s", exc)
            return []

    def interpret_command(self, command: str) -> Optional[Dict[str, Any]]:
        """
        Ask the server to interpret a command the local parser couldn't match.

        The robot handles common phrases locally for free; only the leftovers
        escalate to the server's language model. Returns the parsed intent or
        None — the robot never reaches an online model itself.

        Parameters
        ----------
        command : str
            The raw natural-language command.

        Returns
        -------
        dict or None
            ``{"chore_type": str, "room": str|None}`` on success, else None
            (no match, offline, or error).
        """
        if not self.enabled:
            return None
        try:
            data = self._post(
                "/api/kova/interpret",
                {"robot_id": self.robot_id, "command": command, "timestamp": time.time()},
            )
            chore_type = data.get("chore_type")
            if not chore_type:
                return None
            return {"chore_type": chore_type, "room": data.get("room")}
        except RiverSongAPIError as exc:
            log.warning("Command interpretation failed (non-fatal): %s", exc)
            return None

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

    def _post(
        self,
        path: str,
        payload: Dict[str, Any],
        timeout: Optional[int] = None,
        session: Optional[requests.Session] = None,
    ) -> Dict[str, Any]:
        """
        Perform a POST request.

        Parameters
        ----------
        path : str
            API path.
        payload : dict
            JSON body.
        timeout : int, optional
            Per-request timeout override (used by the background sender).
        session : requests.Session, optional
            Session override (the background sender uses a no-retry session).

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
            resp = (session or self._session).post(url, json=payload, timeout=timeout or self._timeout)
            resp.raise_for_status()
            return resp.json() if resp.content else {}
        except requests.RequestException as exc:
            raise RiverSongAPIError(f"POST {path} failed: {exc}") from exc
