#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : core/constants.py
Purpose     : Global constants, enumerations, and fixed parameters used
              system-wide. No logic lives here — only named values.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from enum import Enum, auto

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM IDENTITY
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_NAME: str = "River Kova"
SYSTEM_VERSION: str = "1.0.0"
RIVER_SONG_ECOSYSTEM: str = "River Song AI"
RIVER_SONG_URL: str = "https://riversongai.com"

# ─────────────────────────────────────────────────────────────────────────────
# ROS2 NODE NAMES
# ─────────────────────────────────────────────────────────────────────────────

MAIN_NODE_NAME: str = "kova_main"
ARM_NODE_NAME: str = "kova_arm_controller"
DRIVE_NODE_NAME: str = "kova_drive_controller"
GRIPPER_NODE_NAME: str = "kova_gripper_manager"
CAMERA_NODE_NAME: str = "kova_camera_manager"
BATTERY_NODE_NAME: str = "kova_battery_monitor"
SAFETY_NODE_NAME: str = "kova_safety_manager"
NAVIGATION_NODE_NAME: str = "kova_navigation"
VISION_NODE_NAME: str = "kova_vision"
TELEMETRY_NODE_NAME: str = "kova_telemetry"

# ─────────────────────────────────────────────────────────────────────────────
# SAFETY THRESHOLDS  (all distances in metres, forces in Newtons)
# ─────────────────────────────────────────────────────────────────────────────

HUMAN_PROXIMITY_ESTOP: float = 1.0    # Immediate full stop
HUMAN_PROXIMITY_SLOW: float = 1.5     # Reduce speed to 20 %
HUMAN_PROXIMITY_WARN: float = 2.5     # Emit warning, continue at reduced speed
COLLISION_FORCE_LIMIT: float = 10.0   # Newtons — triggers fault
OBSTACLE_CLEARANCE_MIN: float = 0.25  # Minimum gap to navigate through
WATCHDOG_TIMEOUT_SEC: float = 5.0     # Seconds before watchdog fires
ESTOP_RECOVERY_DELAY_SEC: float = 3.0 # Mandatory pause after e-stop clears

# ─────────────────────────────────────────────────────────────────────────────
# BATTERY
# ─────────────────────────────────────────────────────────────────────────────

BATTERY_CRITICAL: float = 5.0    # % — immediate return to base
BATTERY_LOW: float = 15.0        # % — finish current step then return
BATTERY_WARN: float = 25.0       # % — alert River Song, continue
BATTERY_FULL: float = 95.0       # % — stop charging

# ─────────────────────────────────────────────────────────────────────────────
# DRIVE / MOTION
# ─────────────────────────────────────────────────────────────────────────────

MAX_LINEAR_SPEED: float = 1.2    # m/s
MAX_ANGULAR_SPEED: float = 1.0   # rad/s
SAFE_LINEAR_SPEED: float = 0.4   # m/s — used near humans / obstacles
SAFE_ANGULAR_SPEED: float = 0.3  # rad/s

# ─────────────────────────────────────────────────────────────────────────────
# ARM / GRIPPER
# ─────────────────────────────────────────────────────────────────────────────

ARM_MAX_PAYLOAD_KG: float = 2.5
ARM_DOF: int = 6
GRIPPER_MAX_FORCE_N: float = 20.0
GRIPPER_OPEN_WIDTH_MM: float = 85.0
GRIPPER_CLOSED_WIDTH_MM: float = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# COMMUNICATION
# ─────────────────────────────────────────────────────────────────────────────

PICO_BAUD_RATE: int = 115200
PICO_SERIAL_TIMEOUT: float = 1.0
DEFAULT_API_TIMEOUT: int = 30       # seconds
RIVER_SONG_API_BASE: str = "/api/kova"
TELEMETRY_PUSH_INTERVAL: float = 5.0  # seconds
STREAM_PORT: int = 8080
FASTAPI_PORT: int = 8000

# ─────────────────────────────────────────────────────────────────────────────
# VISION
# ─────────────────────────────────────────────────────────────────────────────

CAMERA_FPS: int = 30
CAMERA_WIDTH: int = 1280
CAMERA_HEIGHT: int = 720
DETECTION_CONFIDENCE_THRESHOLD: float = 0.65
CLASSIFICATION_CONFIDENCE_THRESHOLD: float = 0.70
MEDIAPIPE_MODEL_COMPLEXITY: int = 1

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION / MAPPING
# ─────────────────────────────────────────────────────────────────────────────

MAP_RESOLUTION: float = 0.05       # metres per cell
MAP_UPDATE_INTERVAL: float = 0.5   # seconds
PATH_REPLAN_INTERVAL: float = 1.0  # seconds
BASE_STATION_TOLERANCE: float = 0.1  # metres — "close enough" to dock

# ─────────────────────────────────────────────────────────────────────────────
# TASK SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

MAX_CONCURRENT_TASKS: int = 1       # per unit — sequential by default
TASK_QUEUE_MAX_SIZE: int = 50
TASK_RETRY_LIMIT: int = 3
TASK_STEP_TIMEOUT_SEC: float = 60.0

# ─────────────────────────────────────────────────────────────────────────────
# ENUMERATIONS
# ─────────────────────────────────────────────────────────────────────────────


class TaskStatus(str, Enum):
    """Lifecycle states for a chore task."""
    IDLE = "IDLE"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


class SafetyLevel(str, Enum):
    """System-wide safety alert levels."""
    NOMINAL = "NOMINAL"
    WARN = "WARN"
    SLOW = "SLOW"
    ESTOP = "ESTOP"
    FAULT = "FAULT"


class RobotState(str, Enum):
    """Top-level operational states of a Kova unit."""
    BOOTING = "BOOTING"
    IDLE = "IDLE"
    NAVIGATING = "NAVIGATING"
    EXECUTING_TASK = "EXECUTING_TASK"
    RETURNING_TO_BASE = "RETURNING_TO_BASE"
    CHARGING = "CHARGING"
    ESTOP = "ESTOP"
    FAULT = "FAULT"
    SHUTDOWN = "SHUTDOWN"


class ChoreType(str, Enum):
    """Predefined chore categories understood by the task library."""
    VACUUM = "VACUUM"
    MOP = "MOP"
    FETCH = "FETCH"
    ORGANIZE = "ORGANIZE"
    WIPE_SURFACE = "WIPE_SURFACE"
    TAKE_OUT_TRASH = "TAKE_OUT_TRASH"
    LOAD_DISHWASHER = "LOAD_DISHWASHER"
    UNLOAD_DISHWASHER = "UNLOAD_DISHWASHER"
    LAUNDRY_TRANSFER = "LAUNDRY_TRANSFER"
    GET_WATER = "GET_WATER"
    FEED_DOGS = "FEED_DOGS"
    CUSTOM = "CUSTOM"


class HardwareBackend(str, Enum):
    """Hardware driver backend selected in the unit profile.

    The brain is hardware-agnostic: every robot body is reached through a
    backend adapter. New robot platforms add a new backend value plus an
    adapter module — nothing else in the system changes.
    """
    PICO = "pico"          # Raspberry Pi Pico serial bridge (reference build)
    SIM = "sim"            # Fully simulated body — no hardware required


class FaultCode(str, Enum):
    """Fault codes reported to the fault manager and River Song."""
    NONE = "NONE"
    HUMAN_PROXIMITY = "HUMAN_PROXIMITY"
    COLLISION = "COLLISION"
    BATTERY_CRITICAL = "BATTERY_CRITICAL"
    MOTOR_FAULT = "MOTOR_FAULT"
    ARM_FAULT = "ARM_FAULT"
    SENSOR_FAULT = "SENSOR_FAULT"
    COMMS_FAULT = "COMMS_FAULT"
    WATCHDOG_TIMEOUT = "WATCHDOG_TIMEOUT"
    UNKNOWN = "UNKNOWN"


class ConnectivityMode(str, Enum):
    """Active network transport."""
    WIFI = "WIFI"
    LTE_4G = "LTE_4G"
    VPN_WIREGUARD = "VPN_WIREGUARD"
    OFFLINE = "OFFLINE"
