#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : api/local_server.py
Purpose     : Local control API served from the robot itself. Lets you check
              status, submit chores (typed or voice-style), and trigger /
              clear the e-stop from any browser or script on the network —
              no River Song server required. This is the self-hosted control
              surface; River Song will later call these same operations
              remotely.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-06-12
License     : Proprietary — River Song AI (riversongai.com)
================================================================================

Endpoints
---------
GET  /                  — browser dashboard (dashboard.html)
GET  /health            — liveness probe
GET  /status            — robot state, safety level, battery, active task
GET  /chores            — available chore types and their requirements
GET  /tasks             — queued + active tasks
POST /tasks             — submit a chore  {"chore_type": "VACUUM", "room": "kitchen", "priority": 5}
POST /voice             — natural command {"command": "have kova feed the dogs"}
POST /estop             — software emergency stop
POST /estop/clear       — clear the e-stop after recovery delay
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_DASHBOARD_PATH = Path(__file__).parent / "dashboard.html"

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    log.warning("FastAPI/uvicorn not available — local control API disabled.")


if _FASTAPI_AVAILABLE:

    class TaskRequest(BaseModel):
        """Body for POST /tasks."""

        chore_type: str
        room: Optional[str] = None
        priority: int = 5

    class VoiceRequest(BaseModel):
        """Body for POST /voice."""

        command: str


def create_app(core) -> "FastAPI":
    """
    Build the FastAPI application bound to a KovaCore instance.

    Parameters
    ----------
    core : KovaCore
        The running robot core (provides task manager, safety, telemetry).

    Returns
    -------
    FastAPI
    """
    app = FastAPI(title="River Kova — Local Control", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        if not _DASHBOARD_PATH.exists():
            raise HTTPException(status_code=404, detail="dashboard.html not found.")
        return _DASHBOARD_PATH.read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        return {"status": "ok", "robot_id": core.robot_id}

    @app.get("/status")
    def status():
        payload = {
            "robot_id": core.robot_id,
            "state": core.state.value,
            "safety_level": core.safety_level.value,
            "battery_pct": core.battery_monitor.level if core.battery_monitor else None,
            "active_task": core.task_manager.active_task if core.task_manager else None,
            "queue_size": core.task_manager.queue_size if core.task_manager else 0,
        }
        if getattr(core, "sim_world", None):
            payload["sim"] = core.sim_world.snapshot()
        semantic_map = getattr(core, "semantic_map", None)
        if semantic_map is not None:
            recognition = semantic_map.snapshot()
            if getattr(core, "sim_world", None):
                x, y, _ = core.sim_world.pose
                label, confidence = semantic_map.room_at(x, y)
                recognition["current_room"] = label
                recognition["current_confidence"] = confidence
            payload["recognition"] = recognition
        return payload

    @app.get("/chores")
    def chores():
        library = core.chore_library
        result = {}
        for chore_type in library.list_chores():
            chore = library.get(chore_type)
            result[str(chore_type)] = {
                "name": chore.get("name"),
                "description": chore.get("description", ""),
                "required_capabilities": chore.get("required_capabilities", []),
                "estimated_duration_sec": chore.get("estimated_duration_sec"),
            }
        return result

    @app.get("/tasks")
    def tasks():
        return {
            "active": core.task_manager.active_task,
            "queued": core.task_manager.queue_size,
        }

    @app.post("/tasks")
    def submit_task(request: TaskRequest):
        task_id = core.task_manager.submit(
            chore_type=request.chore_type.upper(),
            room=request.room,
            priority=request.priority,
        )
        if task_id is None:
            raise HTTPException(
                status_code=400,
                detail=f"Chore '{request.chore_type}' rejected — unknown type, "
                       "missing capability, or queue full.",
            )
        return {"task_id": task_id, "status": "queued"}

    @app.post("/voice")
    def voice(request: VoiceRequest):
        task_id = core.task_manager.submit_from_voice(request.command)
        if task_id is None:
            raise HTTPException(status_code=400, detail="No chore matched that command.")
        return {"task_id": task_id, "status": "queued"}

    @app.post("/estop")
    def estop():
        core._emergency_halt(reason="Local API e-stop")
        return {"status": "estopped"}

    @app.post("/estop/clear")
    def estop_clear():
        if core.estop and not core.estop.reset():
            raise HTTPException(status_code=409, detail="E-stop recovery delay not elapsed.")
        if core.drive:
            core.drive.clear_estop()
        if core.arm:
            core.arm.clear_halt()
        from core.constants import RobotState, SafetyLevel
        core.safety_level = SafetyLevel.NOMINAL
        core.state = RobotState.IDLE
        return {"status": "cleared"}

    return app


class LocalControlServer:
    """
    Runs the local control API in a background thread.

    Degrades gracefully: if FastAPI/uvicorn aren't installed, start() logs a
    warning and the robot runs headless.
    """

    def __init__(self, core, host: str = "0.0.0.0", port: int = 8000) -> None:
        """
        Parameters
        ----------
        core : KovaCore
            The robot core to expose.
        host : str
            Bind address.
        port : int
            TCP port for the API.
        """
        self._core = core
        self._host = host
        self._port = port
        self._server: Optional[object] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """
        Start serving in a daemon thread.

        Returns
        -------
        bool
            True if the server started, False if FastAPI is unavailable.
        """
        if not _FASTAPI_AVAILABLE:
            log.warning("LocalControlServer: FastAPI not installed — API disabled.")
            return False

        app = create_app(self._core)
        config = uvicorn.Config(app, host=self._host, port=self._port, log_level="warning")
        self._server = uvicorn.Server(config)

        self._thread = threading.Thread(
            target=self._server.run,
            name="kova-local-api",
            daemon=True,
        )
        self._thread.start()
        log.info("LocalControlServer listening on %s:%d.", self._host, self._port)
        return True

    def stop(self) -> None:
        """Signal the server to exit."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        log.info("LocalControlServer stopped.")
