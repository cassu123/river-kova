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

# ── Telemetry ─────────────────────────────────────────────────────────────────
from telemetry.logger import KovaLogger
from telemetry.collector import TelemetryCollector
from telemetry.alerts import AlertManager

# ── Tasks ─────────────────────────────────────────────────────────────────────
from tasks.chore_library import ChoreLibrary
from tasks.task_queue import TaskQueue
from tasks.task_manager import TaskManager
from tasks.task_executor import TaskExecutor

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING BOOTSTRAP  (before anything else touches the logger)
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, config.telemetry.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
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
        self.state: RobotState = RobotState.BOOTING
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

        self.kova_logger: Optional[KovaLogger] = None
        self.telemetry_collector: Optional[TelemetryCollector] = None
        self.alert_manager: Optional[AlertManager] = None

        self.chore_library: Optional[ChoreLibrary] = None
        self.task_queue: Optional[TaskQueue] = None
        self.task_manager: Optional[TaskManager] = None
        self.task_executor: Optional[TaskExecutor] = None

        self._boot()

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
        log.info("[BOOT 1/8] Initialising safety systems...")

        self.estop = EStop(enabled=config.safety.estop_enabled)
        self.estop.arm()
        log.info("  ✓ EStop armed")

        self.watchdog = Watchdog(
            timeout_sec=config.safety.watchdog_timeout_sec,
            on_timeout=self._on_watchdog_timeout,
        )
        self.watchdog.start()
        log.info("  ✓ Watchdog started (timeout=%.1fs)", config.safety.watchdog_timeout_sec)

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
        Initialise hardware drivers in dependency order.

        Pico bridge must be live before drive/arm/gripper can be commanded.
        """
        log.info("[BOOT 2/8] Initialising hardware drivers...")

        self.pico_bridge = PicoBridge(
            port=config.hardware.pico_serial_port,
            baud_rate=115200,
        )
        self.pico_bridge.connect()
        log.info("  ✓ PicoBridge connected on %s", config.hardware.pico_serial_port)

        self.drive = DriveController(
            pico_bridge=self.pico_bridge,
            max_speed=config.hardware.drive_max_speed,
            drive_type=config.hardware.drive_type,
        )
        log.info("  ✓ DriveController ready (%s)", config.hardware.drive_type)

        self.arm = ArmController(
            arm_type=config.hardware.arm_type,
            controller=config.hardware.arm_controller,
            max_payload_kg=config.hardware.arm_max_payload_kg,
        )
        log.info("  ✓ ArmController ready (%s / %s)", config.hardware.arm_type, config.hardware.arm_controller)

        self.gripper = GripperManager(arm_controller=self.arm)
        log.info("  ✓ GripperManager ready")

        self.camera_manager = CameraManager(
            model=config.hardware.camera_model,
            fps=config.vision.camera_fps,
            width=config.vision.camera_width,
            height=config.vision.camera_height,
        )
        self.camera_manager.open()
        log.info("  ✓ CameraManager open (%s)", config.hardware.camera_model)

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

        VPN is established before any API calls are attempted.
        """
        log.info("[BOOT 3/8] Initialising connectivity...")

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
        )
        self.api_client.register_unit()
        log.info("  ✓ River Song API client registered")

    def _init_vision(self) -> None:
        """Initialise the vision pipeline."""
        log.info("[BOOT 4/8] Initialising vision pipeline...")

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
        log.info("[BOOT 5/8] Initialising navigation...")

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

    def _init_telemetry(self) -> None:
        """Initialise logging, metrics collection, and alerting."""
        log.info("[BOOT 6/8] Initialising telemetry...")

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
        log.info("[BOOT 7/8] Initialising task system...")

        self.chore_library = ChoreLibrary()
        log.info("  ✓ ChoreLibrary loaded (%d chores)", len(self.chore_library))

        self.task_queue = TaskQueue(max_size=50)
        log.info("  ✓ TaskQueue ready")

        self.task_manager = TaskManager(
            task_queue=self.task_queue,
            chore_library=self.chore_library,
            api_client=self.api_client,
            robot_id=self.robot_id,
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
        )
        log.info("  ✓ TaskExecutor ready")

    # ── Main control loop ─────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Start the main control loop.

        Runs until a shutdown signal is received or a fatal fault occurs.
        The watchdog is kicked every iteration to prove liveness.
        """
        log.info("[BOOT 8/8] Entering main control loop.")
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

        # 3. Task dispatch (only when safe and idle)
        if (
            self.state == RobotState.IDLE
            and self.safety_level == SafetyLevel.NOMINAL
            and self.task_manager
        ):
            next_task = self.task_manager.get_next_task()
            if next_task:
                self._dispatch_task(next_task)

        # 4. Heartbeat to River Song
        if self.api_client:
            self.api_client.heartbeat(
                state=self.state.value,
                safety_level=self.safety_level.value,
                battery_pct=self.battery_monitor.level if self.battery_monitor else -1,
            )

    def _dispatch_task(self, task: dict) -> None:
        """
        Hand a task to the executor and update robot state.

        Parameters
        ----------
        task : dict
            Task descriptor from the task manager.
        """
        log.info("Dispatching task: %s", task.get("name", "unknown"))
        self.state = RobotState.EXECUTING_TASK
        try:
            self.task_executor.execute(task)
        except Exception as exc:
            log.error("Task execution error: %s", exc, exc_info=True)
            if self.fault_manager:
                self.fault_manager.report(f"Task failed: {exc}")
        finally:
            self.state = RobotState.IDLE

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
        log.error("Battery CRITICAL: %.1f%% — returning to base immediately.", level)
        self.state = RobotState.RETURNING_TO_BASE
        if self.task_executor:
            self.task_executor.abort_current()
        if self.return_to_base:
            self.return_to_base.execute()

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

        # Stop motion first
        self._emergency_halt(reason="Graceful shutdown")

        # Tear down in reverse order
        for subsystem_name, subsystem in [
            ("TaskExecutor", self.task_executor),
            ("TaskManager", self.task_manager),
            ("StreamServer", self.stream_server),
            ("TelemetryCollector", self.telemetry_collector),
            ("BatteryMonitor", self.battery_monitor),
            ("CameraManager", self.camera_manager),
            ("PicoBridge", self.pico_bridge),
            ("VPN", self.vpn),
            ("Watchdog", self.watchdog),
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
