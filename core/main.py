#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : core/main.py
Purpose     : Primary entry point. Bootstraps all subsystems in the correct
              safety-first order, starts the ROS2 spin loop, and handles
              graceful shutdown. Nothing runs until safety systems are live.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================

Boot order (non-negotiable):
  1. Logging
  2. Config validation
  3. E-Stop hardware line
  4. Watchdog
  5. Fault manager
  6. Human detection
  7. Collision avoidance
  8. Hardware drivers (Pico bridge → drive → arm → gripper → camera → battery)
  9. Connectivity (VPN → WiFi → API client)
 10. Vision pipeline
 11. Navigation (room mapper → path planner → obstacle avoidance → return-to-base)
 12. Telemetry (logger → collector → alerts)
 13. Task system (chore library → queue → manager → executor)
 14. River Song API server (FastAPI)
 15. Main control loop
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# ── ROS2 ─────────────────────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    logging.warning("rclpy not found — running in simulation/stub mode.")

# ── River Kova core ───────────────────────────────────────────────────────────
from core.config import config
from core.constants import (
    BATTERY_FULL,
    MAIN_NODE_NAME,
    RobotState,
    SafetyLevel,
    SYSTEM_NAME,
    SYSTEM_VERSION,
)

# ── Safety (must import before anything else) ─────────────────────────────────
from safety.estop import EStop
from safety.watchdog import Watchdog
from safety.fault_manager import FaultManager
from safety.human_detection import HumanDetection
from safety.collision_avoid import CollisionAvoidance

# ── Hardware ──────────────────────────────────────────────────────────────────
from hardware.factory import build_hardware
from hardware.pico_bridge import PicoBridge
from hardware.drive_controller import DriveController
from hardware.arm_controller import ArmController
from hardware.gripper_manager import GripperManager
from hardware.camera_manager import CameraManager
from hardware.battery_monitor import BatteryMonitor

# ── Connectivity ──────────────────────────────────────────────────────────────
from connectivity.vpn import VPNManager
from connectivity.wifi_manager import WiFiManager
from connectivity.api_client import RiverSongAPIClient

# ── Vision ────────────────────────────────────────────────────────────────────
from vision.camera_feed import CameraFeed
from vision.object_detect import ObjectDetector
from vision.object_classify import ObjectClassifier
from vision.stream_server import StreamServer

# ── Navigation ────────────────────────────────────────────────────────────────
from navigation.room_mapper import RoomMapper
from navigation.path_planner import PathPlanner
from navigation.obstacle_avoid import ObstacleAvoidance
from navigation.return_base import ReturnToBase
from navigation.semantic_map import SemanticMap
from navigation.occupancy_map import OccupancyMap
from navigation.explorer import PointDriver, Navigator, FrontierExplorer

# ── Telemetry ─────────────────────────────────────────────────────────────────
from telemetry.logger import KovaLogger
from telemetry.collector import TelemetryCollector
from telemetry.alerts import AlertManager

# ── Tasks ─────────────────────────────────────────────────────────────────────
from tasks.chore_library import ChoreLibrary
from tasks.task_queue import TaskQueue
from tasks.task_manager import TaskManager
from tasks.task_executor import TaskExecutor

# ── Autonomy & local control ──────────────────────────────────────────────────
from autonomy.initiative_engine import (
    InitiativeEngine,
    ScheduledRoutineRule,
    TidyUpRule,
    ExploreRule,
    RiverSongInitiativeRule,
)
from api.local_server import LocalControlServer

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING BOOTSTRAP  (before anything else touches the logger)
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, config.telemetry.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    force=True,  # Import-time warnings auto-configure the root logger first
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# KOVA CORE NODE
# ─────────────────────────────────────────────────────────────────────────────


class KovaCore(Node if ROS2_AVAILABLE else object):
    """
    Central orchestrator for a single Kova unit.

    Owns all subsystem references, enforces the safety-first boot order,
    runs the main control loop, and coordinates graceful shutdown.

    Attributes
    ----------
    robot_id : str
        Unique identifier for this unit (from kova_profile.json).
    state : RobotState
        Current operational state of the unit.
    safety_level : SafetyLevel
        Current safety alert level — drives motion permission.
    """

    def __init__(self) -> None:
        """Initialise the KovaCore node and all subsystems."""
        if ROS2_AVAILABLE:
            super().__init__(MAIN_NODE_NAME)

        self.robot_id: str = config.robot_id
        # State is read and written from the control loop, the task executor
        # worker, safety callbacks, and the local API thread — all access
        # goes through the locked property below.
        self._state_lock = threading.Lock()
        self._state: RobotState = RobotState.BOOTING
        self.safety_level: SafetyLevel = SafetyLevel.NOMINAL
        self._shutdown_event = threading.Event()

        log.info("=" * 72)
        log.info("%s v%s  —  Unit: %s", SYSTEM_NAME, SYSTEM_VERSION, self.robot_id)
        log.info("=" * 72)

        # Subsystem references (populated during _boot)
        self.estop: Optional[EStop] = None
        self.watchdog: Optional[Watchdog] = None
        self.fault_manager: Optional[FaultManager] = None
        self.human_detection: Optional[HumanDetection] = None
        self.collision_avoidance: Optional[CollisionAvoidance] = None

        self.pico_bridge: Optional[PicoBridge] = None
        self.drive: Optional[DriveController] = None
        self.arm: Optional[ArmController] = None
        self.gripper: Optional[GripperManager] = None
        self.camera_manager: Optional[CameraManager] = None
        self.battery_monitor: Optional[BatteryMonitor] = None

        self.vpn: Optional[VPNManager] = None
        self.wifi: Optional[WiFiManager] = None
        self.api_client: Optional[RiverSongAPIClient] = None

        self.camera_feed: Optional[CameraFeed] = None
        self.object_detector: Optional[ObjectDetector] = None
        self.object_classifier: Optional[ObjectClassifier] = None
        self.stream_server: Optional[StreamServer] = None

        self.room_mapper: Optional[RoomMapper] = None
        self.path_planner: Optional[PathPlanner] = None
        self.obstacle_avoidance: Optional[ObstacleAvoidance] = None
        self.return_to_base: Optional[ReturnToBase] = None
        self.semantic_map: Optional[SemanticMap] = None
        self.occupancy_map: Optional[OccupancyMap] = None
        self.navigator: Optional[Navigator] = None
        self.explorer: Optional[FrontierExplorer] = None

        self.kova_logger: Optional[KovaLogger] = None
        self.telemetry_collector: Optional[TelemetryCollector] = None
        self.alert_manager: Optional[AlertManager] = None

        self.chore_library: Optional[ChoreLibrary] = None
        self.task_queue: Optional[TaskQueue] = None
        self.task_manager: Optional[TaskManager] = None
        self.task_executor: Optional[TaskExecutor] = None

        self.sim_world = None                  # Populated when backend == 'sim'
        self.initiative_engine: Optional[InitiativeEngine] = None
        self.local_api: Optional[LocalControlServer] = None
        self._last_heartbeat: float = 0.0
        self._last_initiative_tick: float = 0.0
        self._last_perception_tick: float = 0.0
        self._executor_thread: Optional[threading.Thread] = None

        self._boot()

    # ── State machine ─────────────────────────────────────────────────────────

    @property
    def state(self) -> RobotState:
        """Current operational state (thread-safe)."""
        with self._state_lock:
            return self._state

    @state.setter
    def state(self, new_state: RobotState) -> None:
        with self._state_lock:
            self._state = new_state

    def _transition(self, expected: RobotState, new_state: RobotState) -> bool:
        """
        Atomically move from `expected` to `new_state`.

        Returns False (without changing anything) if another thread moved
        the state first — e.g. a safety callback set ESTOP while a task
        was finishing. The caller must not assume the transition happened.
        """
        with self._state_lock:
            if self._state != expected:
                return False
            self._state = new_state
            return True

    # ── Boot sequence ─────────────────────────────────────────────────────────

    def _boot(self) -> None:
        """
        Execute the safety-first boot sequence.

        Raises
        ------
        RuntimeError
            If any safety-critical subsystem fails to initialise.
        """
        log.info("Starting boot sequence...")

        try:
            self._init_safety_systems()
            self._init_hardware()
            self._init_connectivity()
            self._init_vision()
            self._init_navigation()
            self._init_telemetry()
            self._init_tasks()
            self._init_autonomy()
            self._init_local_api()
            self._register_signal_handlers()
        except Exception as exc:
            log.critical("Boot failed: %s — triggering e-stop.", exc, exc_info=True)
            self._emergency_halt(reason=str(exc))
            raise RuntimeError(f"Boot sequence failed: {exc}") from exc

        self.state = RobotState.IDLE
        log.info("Boot complete. Unit %s is IDLE and ready.", self.robot_id)

    def _init_safety_systems(self) -> None:
        """
        Initialise all safety subsystems.

        Safety systems MUST be live before any hardware is powered.
        Order: EStop → Watchdog → FaultManager → HumanDetection → CollisionAvoidance.
        """
        log.info("[BOOT 1/9] Initialising safety systems...")

        self.estop = EStop(enabled=config.safety.estop_enabled)
        self.estop.arm()
        log.info("  ✓ EStop armed")

        # Created here but started at the top of run() — boot steps can
        # legitimately take longer than the watchdog window (e.g. slow
        # network), and the control loop is what the watchdog supervises.
        self.watchdog = Watchdog(
            timeout_sec=config.safety.watchdog_timeout_sec,
            on_timeout=self._on_watchdog_timeout,
        )
        log.info("  ✓ Watchdog armed (timeout=%.1fs, starts with control loop)", config.safety.watchdog_timeout_sec)

        self.fault_manager = FaultManager(
            robot_id=self.robot_id,
            on_fault=self._on_fault,
        )
        log.info("  ✓ FaultManager ready")

        self.human_detection = HumanDetection(
            stop_radius_m=config.safety.human_detection_radius,
            on_human_detected=self._on_human_detected,
        )
        log.info(
            "  ✓ HumanDetection active (stop radius=%.1fm)",
            config.safety.human_detection_radius,
        )

        self.collision_avoidance = CollisionAvoidance(
            max_force_n=config.safety.max_collision_force,
            on_collision=self._on_collision,
        )
        log.info("  ✓ CollisionAvoidance active")

    def _init_hardware(self) -> None:
        """
        Initialise hardware drivers via the backend factory.

        The factory assembles the driver set for whatever robot body the
        profile names (real serial hardware, full simulation, or a future
        platform adapter) — everything above this point is body-agnostic.
        """
        log.info("[BOOT 2/9] Initialising hardware (backend='%s')...", config.hardware.backend)

        hw = build_hardware(config)
        self.pico_bridge = hw.bridge
        self.drive = hw.drive
        self.arm = hw.arm
        self.gripper = hw.gripper
        self.camera_manager = hw.camera
        self.sim_world = hw.sim_world
        log.info("  ✓ Hardware set assembled (%s backend)", hw.backend)

        self.battery_monitor = BatteryMonitor(
            pico_bridge=self.pico_bridge,
            on_low=self._on_battery_low,
            on_critical=self._on_battery_critical,
        )
        self.battery_monitor.start()
        log.info("  ✓ BatteryMonitor started")

    def _init_connectivity(self) -> None:
        """
        Bring up network connectivity.

        When `connectivity.enabled` is false in the profile, the unit runs
        fully self-hosted: no WiFi check, no VPN, and the API client is an
        offline no-op. VPN is established before any API calls otherwise.
        """
        log.info("[BOOT 3/9] Initialising connectivity...")

        if not config.connectivity.enabled:
            log.info("  — Connectivity disabled in profile: self-hosted / offline mode")
        else:
            self.wifi = WiFiManager()
            self.wifi.ensure_connected()
            log.info("  ✓ WiFi connected")

            if config.connectivity.vpn_required:
                self.vpn = VPNManager(config_path=config.connectivity.vpn_config_path)
                self.vpn.connect()
                log.info("  ✓ WireGuard VPN up")
            else:
                log.info("  — VPN disabled in profile")

        self.api_client = RiverSongAPIClient(
            base_url=config.connectivity.river_song_api_url,
            api_key=config.connectivity.api_key,
            timeout=config.connectivity.api_timeout_sec,
            robot_id=self.robot_id,
            enabled=config.connectivity.enabled,
        )
        self.api_client.register_unit()
        log.info("  ✓ River Song API client ready (%s)",
                 "online" if config.connectivity.enabled else "offline")

    def _init_vision(self) -> None:
        """Initialise the vision pipeline."""
        log.info("[BOOT 4/9] Initialising vision pipeline...")

        self.camera_feed = CameraFeed(camera_manager=self.camera_manager)
        log.info("  ✓ CameraFeed ready")

        self.object_detector = ObjectDetector(
            confidence_threshold=config.vision.detection_confidence,
        )
        log.info("  ✓ ObjectDetector ready")

        self.object_classifier = ObjectClassifier(
            confidence_threshold=config.vision.classification_confidence,
        )
        log.info("  ✓ ObjectClassifier ready")

        self.stream_server = StreamServer(
            camera_feed=self.camera_feed,
            port=config.connectivity.stream_port,
        )
        self.stream_server.start()
        log.info("  ✓ StreamServer started on port %d", config.connectivity.stream_port)

    def _init_navigation(self) -> None:
        """Initialise navigation and mapping subsystems."""
        log.info("[BOOT 5/9] Initialising navigation...")

        self.room_mapper = RoomMapper(
            resolution=config.navigation.map_resolution,
            update_interval=config.navigation.map_update_interval,
        )
        log.info("  ✓ RoomMapper ready")

        self.path_planner = PathPlanner(room_mapper=self.room_mapper)
        log.info("  ✓ PathPlanner ready")

        self.obstacle_avoidance = ObstacleAvoidance(
            drive=self.drive,
            clearance_min=0.25,
        )
        log.info("  ✓ ObstacleAvoidance ready")

        self.return_to_base = ReturnToBase(
            drive=self.drive,
            path_planner=self.path_planner,
            base_x=config.navigation.base_station_x,
            base_y=config.navigation.base_station_y,
            tolerance=config.navigation.base_station_tolerance,
        )
        log.info("  ✓ ReturnToBase ready")

        # Learned room recognition — the robot's own map of the home, built
        # from what it sees rather than a preloaded floor plan.
        self.semantic_map = SemanticMap()
        map_path = str(Path(config.telemetry.log_dir) / "semantic_map.json")
        if self.semantic_map.load(map_path):
            log.info("  ✓ SemanticMap restored (%d room(s) known)",
                     len(self.semantic_map.known_rooms()))
        elif Path(map_path).exists():
            log.warning("  ✗ SemanticMap file unreadable — relearning the home from scratch")
        else:
            log.info("  ✓ SemanticMap ready (home not yet learned)")
        self._semantic_map_path = map_path

        # Structural map — vacuum-style: walls and doorways learned from
        # LiDAR, persisted because structure rarely changes. Contents are
        # the semantic map's job, because contents change all the time.
        self.occupancy_map = OccupancyMap()
        occ_path = str(Path(config.telemetry.log_dir) / "occupancy_map.json")
        if self.occupancy_map.load(occ_path):
            log.info("  ✓ OccupancyMap restored (%.1f m² mapped)",
                     self.occupancy_map.explored_area_m2())
        elif Path(occ_path).exists():
            log.warning("  ✗ OccupancyMap file unreadable — remapping structure from scratch")
        else:
            log.info("  ✓ OccupancyMap ready (structure not yet mapped)")
        self._occupancy_map_path = occ_path

        # Map-aware motion + frontier exploration: in simulation, pose and
        # LiDAR come from the SimWorld; a real body plugs in odometry/SLAM
        # and an RPLiDAR here instead.
        if self.sim_world is not None:
            driver = PointDriver(
                bridge=self.pico_bridge,
                pose_provider=lambda: self.sim_world.pose,
                safety_check=self._is_safe_to_move,
            )
            self.navigator = Navigator(
                occupancy_map=self.occupancy_map,
                driver=driver,
                pose_provider=lambda: self.sim_world.pose,
            )
            self.explorer = FrontierExplorer(
                navigator=self.navigator,
                occupancy_map=self.occupancy_map,
                pose_provider=lambda: self.sim_world.pose,
                scan_integrator=self._integrate_lidar,
            )
            log.info("  ✓ Navigator + FrontierExplorer ready (sim sensors)")

    def _init_telemetry(self) -> None:
        """Initialise logging, metrics collection, and alerting."""
        log.info("[BOOT 6/9] Initialising telemetry...")

        self.kova_logger = KovaLogger(
            robot_id=self.robot_id,
            log_dir=config.telemetry.log_dir,
            log_level=config.telemetry.log_level,
        )
        log.info("  ✓ KovaLogger ready")

        self.telemetry_collector = TelemetryCollector(
            robot_id=self.robot_id,
            influxdb_url=config.telemetry.influxdb_url,
            token=config.telemetry.influxdb_token,
            org=config.telemetry.influxdb_org,
            bucket=config.telemetry.influxdb_bucket,
            push_interval=config.connectivity.telemetry_push_interval_sec,
        )
        self.telemetry_collector.start()
        log.info("  ✓ TelemetryCollector started")

        self.alert_manager = AlertManager(
            api_client=self.api_client,
            robot_id=self.robot_id,
        )
        log.info("  ✓ AlertManager ready")

    def _init_tasks(self) -> None:
        """Initialise the task execution pipeline."""
        log.info("[BOOT 7/9] Initialising task system...")

        self.chore_library = ChoreLibrary()
        log.info("  ✓ ChoreLibrary loaded (%d chores)", len(self.chore_library))

        self.task_queue = TaskQueue(max_size=50)
        log.info("  ✓ TaskQueue ready")

        self.task_manager = TaskManager(
            task_queue=self.task_queue,
            chore_library=self.chore_library,
            api_client=self.api_client,
            robot_id=self.robot_id,
            capabilities=config.capability_set,
        )
        log.info("  ✓ TaskManager ready")

        self.task_executor = TaskExecutor(
            task_manager=self.task_manager,
            drive=self.drive,
            arm=self.arm,
            gripper=self.gripper,
            navigation=self.path_planner,
            vision=self.object_detector,
            safety_check=self._is_safe_to_move,
            explorer=self.explorer,
        )
        log.info("  ✓ TaskExecutor ready")

    def _init_autonomy(self) -> None:
        """
        Initialise the initiative engine (self-directed task generation).

        Disabled unless the profile's `autonomy.enabled` flag is true.
        """
        if not config.autonomy.enabled:
            log.info("[BOOT 8/9] Autonomy disabled in profile — command-driven only.")
            return

        log.info("[BOOT 8/9] Initialising autonomy...")

        self.initiative_engine = InitiativeEngine(
            task_manager=self.task_manager,
            min_battery_pct=config.autonomy.min_battery_pct,
            battery_provider=lambda: self.battery_monitor.level if self.battery_monitor else 100.0,
        )

        if config.autonomy.routines:
            # Persist last-fired dates so a reboot after 07:30 doesn't feed
            # the dogs a second time.
            self.initiative_engine.add_rule(ScheduledRoutineRule(
                config.autonomy.routines,
                state_path=str(Path(config.telemetry.log_dir) / "routine_state.json"),
            ))

        # Out-of-place observations: SimWorld in simulation; on a real body
        # this provider will be backed by the vision pipeline.
        if self.sim_world is not None:
            self.initiative_engine.add_rule(TidyUpRule(
                observation_provider=lambda: [
                    {"kind": o.kind, "room": self.sim_world.room_at(o.x, o.y)}
                    for o in self.sim_world.out_of_place_objects()
                ],
            ))

        # Explore while the home is unfamiliar — recognition over routes.
        if self.explorer is not None and self.semantic_map is not None:
            self.initiative_engine.add_rule(ExploreRule(
                known_room_count_provider=lambda: len(self.semantic_map.known_rooms()),
                min_known_rooms=4,
            ))

        # The River Song LLM brain — proposes chores from natural-language
        # household context. The reasoning runs on the River Song server (one
        # credential and one bill for the fleet, no online model call from the
        # robot), so it needs connectivity. The call is made off the control
        # loop by the rule's worker thread.
        if config.autonomy.llm_enabled:
            if not config.connectivity.enabled:
                log.info("  — LLM initiative skipped: needs the River Song server "
                         "(connectivity.enabled is false)")
            else:
                catalogue = [
                    {
                        "chore_type": str(ct),
                        "description": self.chore_library.get(ct).get("description", ""),
                    }
                    for ct in self.chore_library.list_chores()
                ]
                self.initiative_engine.add_rule(RiverSongInitiativeRule(
                    proposal_provider=lambda ctx, state: self.api_client.request_initiative(
                        ctx, state, catalogue,
                    ),
                    chore_catalog=catalogue,
                    context_provider=self._household_context,
                    state_provider=self._llm_state_snapshot,
                    request_interval_sec=config.autonomy.llm_cooldown_sec,
                ))

        log.info("  ✓ InitiativeEngine ready (%d rule(s))", len(self.initiative_engine.rules))

    def _household_context(self) -> str:
        """Read the natural-language household context file, if configured."""
        path = config.autonomy.llm_context_path
        if not path:
            return ""
        try:
            return Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
        except Exception as exc:
            log.warning("Household context read failed: %s", exc)
            return ""

    def _llm_state_snapshot(self) -> dict:
        """Structured state handed to the LLM initiative rule."""
        snapshot: dict = {
            "battery_pct": self.battery_monitor.level if self.battery_monitor else None,
            "state": self.state.value,
        }
        if self.semantic_map is not None:
            snapshot["known_rooms"] = self.semantic_map.known_rooms()
        if self.sim_world is not None:
            snapshot["out_of_place_items"] = [
                {"kind": o.kind, "room": self.sim_world.room_at(o.x, o.y)}
                for o in self.sim_world.out_of_place_objects()
            ]
        return snapshot

    def _init_local_api(self) -> None:
        """Start the self-hosted control API."""
        log.info("[BOOT 9/9] Starting local control API...")
        self.local_api = LocalControlServer(
            core=self,
            host=config.connectivity.api_host,
            port=config.connectivity.fastapi_port,
        )
        if self.local_api.start():
            log.info("  ✓ Local control API on port %d", config.connectivity.fastapi_port)
        else:
            log.info("  — Local control API unavailable (FastAPI not installed)")

    # ── Main control loop ─────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Start the main control loop.

        Runs until a shutdown signal is received or a fatal fault occurs.
        The watchdog is kicked every iteration to prove liveness.
        """
        log.info("Entering main control loop.")
        if self.watchdog:
            self.watchdog.start()
        loop_hz = 10  # 10 Hz control loop
        interval = 1.0 / loop_hz

        while not self._shutdown_event.is_set():
            loop_start = time.monotonic()

            try:
                self._tick()
            except Exception as exc:
                log.error("Control loop error: %s", exc, exc_info=True)
                if self.fault_manager:
                    self.fault_manager.report(str(exc))

            # Kick watchdog to prove we're alive
            if self.watchdog:
                self.watchdog.kick()

            elapsed = time.monotonic() - loop_start
            sleep_time = max(0.0, interval - elapsed)
            time.sleep(sleep_time)

        log.info("Main control loop exited.")

    def _tick(self) -> None:
        """
        Single control loop iteration.

        Checks safety state, polls battery, dispatches pending tasks,
        and pushes a heartbeat to River Song.
        """
        # 1. Safety gate — nothing moves if safety is not nominal
        self._update_safety_level()

        # 2. Battery check
        if self.battery_monitor:
            self.battery_monitor.poll()

            # Charged back up at the dock? The unit is available again.
            if (
                self.state == RobotState.CHARGING
                and self.battery_monitor.level >= BATTERY_FULL
                and self._transition(RobotState.CHARGING, RobotState.IDLE)
            ):
                log.info(
                    "Battery recharged to %.1f%% — unit available again.",
                    self.battery_monitor.level,
                )

        # 2b. Passive room recognition — wherever the robot is, whatever it
        # is doing, what it currently sees feeds the learned home map.
        now = time.monotonic()
        if self.semantic_map and now - self._last_perception_tick >= 0.5:
            self._last_perception_tick = now
            self._observe_surroundings()

        # 3. Initiative — let the robot propose its own work when idle
        if (
            self.initiative_engine
            and self.state == RobotState.IDLE
            and self.safety_level == SafetyLevel.NOMINAL
        ):
            now = time.monotonic()
            if now - self._last_initiative_tick >= config.autonomy.tick_interval_sec:
                self._last_initiative_tick = now
                self.initiative_engine.tick()

        # 4. Task dispatch (only when safe and idle)
        if (
            self.state == RobotState.IDLE
            and self.safety_level == SafetyLevel.NOMINAL
            and self.task_manager
        ):
            next_task = self.task_manager.get_next_task()
            if next_task:
                self._dispatch_task(next_task)

        # 5. Heartbeat to River Song — throttled to the push interval and
        # queued to a background sender, so it can never stall this loop.
        now = time.monotonic()
        if self.api_client and now - self._last_heartbeat >= config.connectivity.telemetry_push_interval_sec:
            self._last_heartbeat = now
            self.api_client.heartbeat(
                state=self.state.value,
                safety_level=self.safety_level.value,
                battery_pct=self.battery_monitor.level if self.battery_monitor else -1,
            )

    def _observe_surroundings(self) -> None:
        """
        Feed one visual observation into the semantic map and one LiDAR
        sweep into the structural map.

        In simulation the SimWorld reports what the sensors would see; on
        a real body these same calls are fed by the object detector and
        the LiDAR driver.
        """
        if self.sim_world is not None:
            x, y, _ = self.sim_world.pose
            kinds = [o.kind for o in self.sim_world.visible_objects()]
            self.semantic_map.observe(x, y, kinds)
            self._integrate_lidar()

    def _integrate_lidar(self) -> None:
        """Fold one LiDAR sweep into the occupancy map."""
        if self.sim_world is None or self.occupancy_map is None:
            return
        x, y, theta = self.sim_world.pose
        self.occupancy_map.integrate_scan(x, y, theta, self.sim_world.lidar_scan())

    def _dispatch_task(self, task: dict) -> None:
        """
        Hand a task to the executor on a worker thread.

        Execution must NOT block the control loop: the loop keeps ticking
        (safety checks, watchdog kicks, heartbeats) while the chore runs.
        The loop's IDLE-state gate prevents double dispatch.

        Parameters
        ----------
        task : dict
            Task descriptor from the task manager.
        """
        if not self._transition(RobotState.IDLE, RobotState.EXECUTING_TASK):
            # A safety callback changed state between the tick's gate check
            # and here — put the task back rather than run it unsafely.
            log.warning(
                "Dispatch of '%s' cancelled — state is now %s; re-queuing.",
                task.get("name", "unknown"), self.state.value,
            )
            self.task_manager.requeue(task)
            return

        log.info("Dispatching task: %s", task.get("name", "unknown"))
        self._executor_thread = threading.Thread(
            target=self._run_task,
            args=(task,),
            name="kova-task-executor",
            daemon=True,
        )
        self._executor_thread.start()

    def _run_task(self, task: dict) -> None:
        """Worker-thread body for a single task execution."""
        try:
            self.task_executor.execute(task)
        except Exception as exc:
            log.error("Task execution error: %s", exc, exc_info=True)
            if self.fault_manager:
                self.fault_manager.report(f"Task failed: {exc}")
        finally:
            # Atomic: only go back to IDLE if no safety callback moved the
            # state (ESTOP/FAULT/RETURNING_TO_BASE) while the task ran.
            self._transition(RobotState.EXECUTING_TASK, RobotState.IDLE)

    # ── Safety helpers ────────────────────────────────────────────────────────

    def _update_safety_level(self) -> None:
        """
        Poll safety subsystems and update the unit's safety level.

        The most severe active condition wins.
        """
        if not self.human_detection or not self.collision_avoidance:
            return

        if self.estop and self.estop.is_triggered:
            self.safety_level = SafetyLevel.ESTOP
            return

        if self.human_detection.human_in_estop_zone:
            self.safety_level = SafetyLevel.ESTOP
            return

        if self.human_detection.human_in_slow_zone:
            self.safety_level = SafetyLevel.SLOW
            return

        if self.human_detection.human_in_warn_zone:
            self.safety_level = SafetyLevel.WARN
            return

        self.safety_level = SafetyLevel.NOMINAL

    def _is_safe_to_move(self) -> bool:
        """
        Return True only when the safety level permits motion.

        Used as a callback injected into the task executor and drive controller.
        """
        return self.safety_level in (SafetyLevel.NOMINAL, SafetyLevel.WARN)

    # ── Safety event callbacks ────────────────────────────────────────────────

    def _on_human_detected(self, distance_m: float) -> None:
        """
        Callback fired by HumanDetection when a person enters the safety radius.

        Parameters
        ----------
        distance_m : float
            Measured distance to the nearest detected human, in metres.
        """
        log.warning("HUMAN DETECTED at %.2fm — issuing immediate stop.", distance_m)
        self.safety_level = SafetyLevel.ESTOP
        if self.drive:
            self.drive.emergency_stop()
        if self.arm:
            self.arm.halt()
        if self.alert_manager:
            self.alert_manager.send(
                level="CRITICAL",
                message=f"Human detected at {distance_m:.2f}m — unit stopped.",
            )

    def _on_collision(self, force_n: float) -> None:
        """
        Callback fired by CollisionAvoidance when impact force exceeds limit.

        Parameters
        ----------
        force_n : float
            Measured collision force in Newtons.
        """
        log.error("COLLISION detected (%.1f N) — halting.", force_n)
        self.safety_level = SafetyLevel.ESTOP
        if self.drive:
            self.drive.emergency_stop()
        if self.arm:
            self.arm.halt()
        if self.fault_manager:
            self.fault_manager.report(f"Collision: {force_n:.1f} N")

    def _on_watchdog_timeout(self) -> None:
        """
        Callback fired when the watchdog timer expires without a kick.

        Triggers an e-stop and reports a fault — the system is unresponsive.
        """
        log.critical("WATCHDOG TIMEOUT — system unresponsive. Triggering e-stop.")
        self._emergency_halt(reason="Watchdog timeout")

    def _on_fault(self, fault_code: str, message: str) -> None:
        """
        Callback fired by FaultManager when a fault is registered.

        Parameters
        ----------
        fault_code : str
            FaultCode enum value as string.
        message : str
            Human-readable fault description.
        """
        log.error("FAULT [%s]: %s", fault_code, message)
        self.state = RobotState.FAULT
        if self.alert_manager:
            self.alert_manager.send(level="ERROR", message=f"[{fault_code}] {message}")

    def _on_battery_low(self, level: float) -> None:
        """
        Callback fired when battery drops below the low threshold.

        Parameters
        ----------
        level : float
            Current battery percentage.
        """
        log.warning("Battery LOW: %.1f%% — will return to base after current step.", level)
        if self.alert_manager:
            self.alert_manager.send(level="WARN", message=f"Battery low: {level:.1f}%")

    def _on_battery_critical(self, level: float) -> None:
        """
        Callback fired when battery drops below the critical threshold.

        Parameters
        ----------
        level : float
            Current battery percentage.
        """
        if self.state in (RobotState.RETURNING_TO_BASE, RobotState.CHARGING):
            return
        log.error("Battery CRITICAL: %.1f%% — returning to base immediately.", level)
        self.state = RobotState.RETURNING_TO_BASE
        if self.task_executor:
            self.task_executor.abort_current()
        if self.return_to_base:
            # Navigation takes many seconds and this callback fires from the
            # control loop / battery poll thread — it must not block them
            # (a stalled control loop trips the watchdog).
            threading.Thread(
                target=self._return_to_base_worker,
                name="kova-return-to-base",
                daemon=True,
            ).start()

    def _return_to_base_worker(self) -> None:
        """Worker-thread body for the critical-battery return-to-base run."""
        try:
            docked = self.return_to_base.execute()
        except Exception as exc:
            log.error("Return-to-base error: %s", exc, exc_info=True)
            docked = False

        if docked:
            self._transition(RobotState.RETURNING_TO_BASE, RobotState.CHARGING)
            log.info("Docked on critical battery — charging.")
        else:
            if self.fault_manager:
                self.fault_manager.report("Return to base failed on critical battery")

    # ── Emergency halt ────────────────────────────────────────────────────────

    def _emergency_halt(self, reason: str = "unspecified") -> None:
        """
        Immediately stop all motion and set the unit to ESTOP state.

        Parameters
        ----------
        reason : str
            Human-readable reason for the halt, logged and reported.
        """
        log.critical("EMERGENCY HALT — reason: %s", reason)
        self.state = RobotState.ESTOP
        self.safety_level = SafetyLevel.ESTOP

        if self.drive:
            try:
                self.drive.emergency_stop()
            except Exception:
                pass

        if self.arm:
            try:
                self.arm.halt()
            except Exception:
                pass

        if self.estop:
            try:
                self.estop.trigger()
            except Exception:
                pass

    # ── Signal handling & shutdown ────────────────────────────────────────────

    def _register_signal_handlers(self) -> None:
        """Register OS signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        log.debug("Signal handlers registered (SIGINT, SIGTERM).")

    def _handle_signal(self, signum: int, frame) -> None:
        """
        Handle OS shutdown signals.

        Parameters
        ----------
        signum : int
            Signal number received.
        frame : frame
            Current stack frame (unused).
        """
        log.info("Signal %d received — initiating graceful shutdown.", signum)
        self.shutdown()

    def shutdown(self) -> None:
        """
        Gracefully shut down all subsystems in reverse boot order.

        Safe to call multiple times — idempotent.
        """
        if self._shutdown_event.is_set():
            return

        log.info("Shutting down %s unit %s...", SYSTEM_NAME, self.robot_id)
        self._shutdown_event.set()
        self.state = RobotState.SHUTDOWN

        # The control loop has exited — stop its watchdog before the (slow)
        # teardown joins below, or it will fire a spurious e-stop mid-shutdown.
        if self.watchdog:
            try:
                self.watchdog.stop()
            except Exception:
                pass

        # Abort any in-flight chore and give its worker thread a moment to
        # reach the abort point — don't kill it mid-motion with hardware in
        # an unknown state.
        if self.task_executor:
            self.task_executor.abort_current()
        if self._executor_thread and self._executor_thread.is_alive():
            self._executor_thread.join(timeout=5.0)
        self._emergency_halt(reason="Graceful shutdown")

        # Persist the learned home maps so recognition and structure
        # survive reboots
        if self.semantic_map:
            try:
                self.semantic_map.save(self._semantic_map_path)
            except Exception as exc:
                log.warning("SemanticMap save failed: %s", exc)
        if self.occupancy_map:
            try:
                self.occupancy_map.save(self._occupancy_map_path)
            except Exception as exc:
                log.warning("OccupancyMap save failed: %s", exc)

        # Tear down in reverse order
        for subsystem_name, subsystem in [
            ("LocalControlServer", self.local_api),
            ("TaskExecutor", self.task_executor),
            ("TaskManager", self.task_manager),
            ("StreamServer", self.stream_server),
            ("TelemetryCollector", self.telemetry_collector),
            ("BatteryMonitor", self.battery_monitor),
            ("CameraManager", self.camera_manager),
            ("PicoBridge", self.pico_bridge),
            ("VPN", self.vpn),
        ]:
            if subsystem and hasattr(subsystem, "stop"):
                try:
                    subsystem.stop()
                    log.debug("  ✓ %s stopped", subsystem_name)
                except Exception as exc:
                    log.warning("  ✗ %s stop error: %s", subsystem_name, exc)

        if self.api_client:
            try:
                self.api_client.deregister_unit()
                self.api_client.stop()
            except Exception:
                pass

        log.info("Shutdown complete. Goodbye.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


def main(args=None) -> int:
    """
    Main entry point for the River Kova control system.

    Parameters
    ----------
    args : list, optional
        Command-line arguments (passed to rclpy.init if ROS2 is available).

    Returns
    -------
    int
        Exit code — 0 for clean exit, 1 for fatal error.
    """
    if ROS2_AVAILABLE:
        rclpy.init(args=args)

    kova: Optional[KovaCore] = None
    exit_code = 0

    try:
        kova = KovaCore()

        if ROS2_AVAILABLE:
            executor = MultiThreadedExecutor()
            executor.add_node(kova)
            spin_thread = threading.Thread(target=executor.spin, daemon=True)
            spin_thread.start()

        kova.run()

    except RuntimeError as exc:
        log.critical("Fatal startup error: %s", exc)
        exit_code = 1
    except Exception as exc:
        log.critical("Unhandled exception: %s", exc, exc_info=True)
        exit_code = 1
    finally:
        if kova:
            kova.shutdown()
        if ROS2_AVAILABLE:
            try:
                rclpy.shutdown()
            except Exception:
                pass

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
