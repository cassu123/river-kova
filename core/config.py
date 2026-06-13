#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : core/config.py
Purpose     : Configuration loader, validator, and runtime accessor.
              Reads the unit profile JSON and environment variables, exposes
              a single validated Config singleton used across all modules.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT PROFILE PATH — override via KOVA_PROFILE env var
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_PROFILE = Path(__file__).resolve().parent.parent / "units" / "kova_profile.json"


# ─────────────────────────────────────────────────────────────────────────────
# TYPED SUB-CONFIGS
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SafetyConfig:
    """Safety-related thresholds loaded from the unit profile."""

    human_detection_radius: float = 2.5
    estop_enabled: bool = True
    max_collision_force: float = 10.0
    watchdog_timeout_sec: float = 5.0
    estop_recovery_delay_sec: float = 3.0


@dataclass
class HardwareConfig:
    """Hardware capability descriptors for this unit."""

    backend: str = "pico"
    arm_type: str = "6-dof"
    arm_controller: str = "moveit"
    arm_max_payload_kg: float = 2.5
    drive_type: str = "differential"
    drive_max_speed: float = 1.2
    camera_model: str = "intel-realsense-d435"
    lidar_model: str = "rplidar-a1"
    pico_serial_port: str = "/dev/ttyACM0"


@dataclass
class ConnectivityConfig:
    """Network and API connectivity settings."""

    enabled: bool = True
    river_song_api_url: str = "https://api.riversongai.com"
    api_key: str = ""
    vpn_required: bool = True
    vpn_config_path: str = "/etc/wireguard/wg0.conf"
    api_timeout_sec: int = 30
    telemetry_push_interval_sec: float = 5.0
    stream_port: int = 8080
    fastapi_port: int = 8000
    # Bind address for the local control API. The API is unauthenticated,
    # so anyone who can reach the socket can drive the robot — set
    # "127.0.0.1" to restrict control to the unit itself.
    api_host: str = "0.0.0.0"


@dataclass
class SimulationConfig:
    """Simulated-world parameters (used when hardware.backend == 'sim')."""

    start_battery_pct: float = 100.0
    battery_drain_idle_pct_per_min: float = 0.2
    battery_drain_moving_pct_per_min: float = 1.5
    battery_charge_pct_per_min: float = 10.0
    time_scale: float = 1.0


@dataclass
class AutonomyConfig:
    """Initiative engine settings — self-directed task generation."""

    enabled: bool = False
    tick_interval_sec: float = 5.0
    min_battery_pct: float = 30.0
    routines: list = field(default_factory=list)


@dataclass
class NavigationConfig:
    """Navigation and mapping parameters."""

    map_resolution: float = 0.05
    map_update_interval: float = 0.5
    path_replan_interval: float = 1.0
    base_station_x: float = 0.0
    base_station_y: float = 0.0
    base_station_tolerance: float = 0.1


@dataclass
class VisionConfig:
    """Vision pipeline settings."""

    camera_fps: int = 30
    camera_width: int = 1280
    camera_height: int = 720
    detection_confidence: float = 0.65
    classification_confidence: float = 0.70
    mediapipe_model_complexity: int = 1
    enable_depth: bool = True


@dataclass
class TelemetryConfig:
    """InfluxDB / telemetry sink settings."""

    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str = ""
    influxdb_org: str = "river-kova"
    influxdb_bucket: str = "kova-telemetry"
    log_level: str = "INFO"
    log_dir: str = "/var/log/river-kova"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONFIG CLASS
# ─────────────────────────────────────────────────────────────────────────────


class Config:
    """
    Singleton configuration manager for a Kova unit.

    Loads the unit profile JSON, merges environment variable overrides,
    validates required fields, and exposes typed sub-configs.

    Usage
    -----
    from core.config import config
    url = config.connectivity.river_song_api_url
    """

    _instance: Optional[Config] = None

    def __new__(cls, profile_path: Optional[Path] = None) -> "Config":
        """Enforce singleton pattern."""
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instance = instance
        return cls._instance

    def __init__(self, profile_path: Optional[Path] = None) -> None:
        """
        Initialize configuration from profile JSON and environment.

        Parameters
        ----------
        profile_path : Path, optional
            Override the default profile path. Useful for testing.
        """
        if self._initialized:
            return

        self._profile_path: Path = Path(
            os.environ.get("KOVA_PROFILE", str(profile_path or _DEFAULT_PROFILE))
        )
        self._raw: Dict[str, Any] = self._load_profile()
        self._apply_env_overrides()
        self._validate()

        # Typed sub-configs
        self.safety: SafetyConfig = self._build_safety()
        self.hardware: HardwareConfig = self._build_hardware()
        self.connectivity: ConnectivityConfig = self._build_connectivity()
        self.navigation: NavigationConfig = self._build_navigation()
        self.vision: VisionConfig = self._build_vision()
        self.telemetry: TelemetryConfig = self._build_telemetry()
        self.simulation: SimulationConfig = self._build_simulation()
        self.autonomy: AutonomyConfig = self._build_autonomy()

        self._initialized = True
        logger.info("Config loaded for unit '%s' (v%s)", self.robot_id, self.version)

    # ── Profile loading ───────────────────────────────────────────────────────

    def _load_profile(self) -> Dict[str, Any]:
        """
        Read and parse the unit profile JSON.

        Returns
        -------
        dict
            Parsed profile data.

        Raises
        ------
        FileNotFoundError
            If the profile file does not exist.
        ValueError
            If the file is not valid JSON.
        """
        if not self._profile_path.exists():
            raise FileNotFoundError(
                f"Kova profile not found: {self._profile_path}. "
                "Set KOVA_PROFILE env var or place kova_profile.json in units/."
            )
        try:
            with self._profile_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            logger.debug("Profile loaded from %s", self._profile_path)
            return data
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in profile {self._profile_path}: {exc}") from exc

    def _apply_env_overrides(self) -> None:
        """
        Merge environment variable overrides into the raw config.

        Supported env vars
        ------------------
        KOVA_ROBOT_ID          — override robot_id
        KOVA_API_URL           — override connectivity.api_endpoint
        KOVA_API_KEY           — set API key (never stored in JSON)
        KOVA_INFLUX_TOKEN      — InfluxDB auth token
        KOVA_LOG_LEVEL         — logging level
        KOVA_VPN_REQUIRED      — "true" / "false"
        """
        overrides = {
            "KOVA_ROBOT_ID": ("robot_id",),
            "KOVA_API_URL": ("connectivity", "api_endpoint"),
        }
        for env_var, key_path in overrides.items():
            value = os.environ.get(env_var)
            if value:
                target = self._raw
                for key in key_path[:-1]:
                    target = target.setdefault(key, {})
                target[key_path[-1]] = value
                logger.debug("Config override from env: %s", env_var)

        # Sensitive values — injected directly, not stored in raw dict
        self._api_key = os.environ.get("KOVA_API_KEY", "")
        self._influx_token = os.environ.get("KOVA_INFLUX_TOKEN", "")

    def _validate(self) -> None:
        """
        Assert required top-level fields are present.

        Raises
        ------
        ValueError
            If a required field is missing.
        """
        required = ["robot_id", "version"]
        for field_name in required:
            if not self._raw.get(field_name):
                raise ValueError(f"Required config field missing: '{field_name}'")

    # ── Sub-config builders ───────────────────────────────────────────────────

    def _build_safety(self) -> SafetyConfig:
        """Build SafetyConfig from raw profile data."""
        s = self._raw.get("safety", {})
        return SafetyConfig(
            human_detection_radius=float(s.get("human_detection_radius", 2.5)),
            estop_enabled=bool(s.get("estop_enabled", True)),
            max_collision_force=float(s.get("max_collision_force", 10.0)),
            watchdog_timeout_sec=float(s.get("watchdog_timeout_sec", 5.0)),
            estop_recovery_delay_sec=float(s.get("estop_recovery_delay_sec", 3.0)),
        )

    def _build_hardware(self) -> HardwareConfig:
        """Build HardwareConfig from raw profile data."""
        h = self._raw.get("hardware", {})
        arm = h.get("arm", {})
        drive = h.get("drive", {})
        sensors = h.get("sensors", {})
        return HardwareConfig(
            backend=h.get("backend", "pico"),
            arm_type=arm.get("type", "6-dof"),
            arm_controller=arm.get("controller", "moveit"),
            arm_max_payload_kg=float(arm.get("max_payload", 2.5)),
            drive_type=drive.get("type", "differential"),
            drive_max_speed=float(drive.get("max_speed", 1.2)),
            camera_model=sensors.get("camera", "intel-realsense-d435"),
            lidar_model=sensors.get("lidar", "rplidar-a1"),
            pico_serial_port=h.get("pico_serial_port", "/dev/ttyACM0"),
        )

    def _build_connectivity(self) -> ConnectivityConfig:
        """Build ConnectivityConfig from raw profile data."""
        c = self._raw.get("connectivity", {})
        return ConnectivityConfig(
            enabled=bool(c.get("enabled", True)),
            river_song_api_url=c.get("api_endpoint", "https://api.riversongai.com"),
            api_key=self._api_key,
            vpn_required=bool(c.get("vpn_required", True)),
            vpn_config_path=c.get("vpn_config_path", "/etc/wireguard/wg0.conf"),
            api_timeout_sec=int(c.get("api_timeout_sec", 30)),
            telemetry_push_interval_sec=float(c.get("telemetry_push_interval_sec", 5.0)),
            stream_port=int(c.get("stream_port", 8080)),
            fastapi_port=int(c.get("fastapi_port", 8000)),
            api_host=str(c.get("api_host", "0.0.0.0")),
        )

    def _build_navigation(self) -> NavigationConfig:
        """Build NavigationConfig from raw profile data."""
        n = self._raw.get("navigation", {})
        base = n.get("base_station", {})
        return NavigationConfig(
            map_resolution=float(n.get("map_resolution", 0.05)),
            map_update_interval=float(n.get("map_update_interval", 0.5)),
            path_replan_interval=float(n.get("path_replan_interval", 1.0)),
            base_station_x=float(base.get("x", 0.0)),
            base_station_y=float(base.get("y", 0.0)),
            base_station_tolerance=float(base.get("tolerance", 0.1)),
        )

    def _build_vision(self) -> VisionConfig:
        """Build VisionConfig from raw profile data."""
        v = self._raw.get("vision", {})
        return VisionConfig(
            camera_fps=int(v.get("fps", 30)),
            camera_width=int(v.get("width", 1280)),
            camera_height=int(v.get("height", 720)),
            detection_confidence=float(v.get("detection_confidence", 0.65)),
            classification_confidence=float(v.get("classification_confidence", 0.70)),
            mediapipe_model_complexity=int(v.get("mediapipe_model_complexity", 1)),
            enable_depth=bool(v.get("enable_depth", True)),
        )

    def _build_telemetry(self) -> TelemetryConfig:
        """Build TelemetryConfig from raw profile data."""
        t = self._raw.get("telemetry", {})
        return TelemetryConfig(
            influxdb_url=t.get("influxdb_url", "http://localhost:8086"),
            influxdb_token=self._influx_token or t.get("influxdb_token", ""),
            influxdb_org=t.get("influxdb_org", "river-kova"),
            influxdb_bucket=t.get("influxdb_bucket", "kova-telemetry"),
            log_level=t.get("log_level", "INFO"),
            log_dir=t.get("log_dir", "/var/log/river-kova"),
        )

    def _build_simulation(self) -> SimulationConfig:
        """Build SimulationConfig from raw profile data."""
        s = self._raw.get("simulation", {})
        return SimulationConfig(
            start_battery_pct=float(s.get("start_battery_pct", 100.0)),
            battery_drain_idle_pct_per_min=float(s.get("battery_drain_idle_pct_per_min", 0.2)),
            battery_drain_moving_pct_per_min=float(s.get("battery_drain_moving_pct_per_min", 1.5)),
            battery_charge_pct_per_min=float(s.get("battery_charge_pct_per_min", 10.0)),
            time_scale=float(s.get("time_scale", 1.0)),
        )

    def _build_autonomy(self) -> AutonomyConfig:
        """Build AutonomyConfig from raw profile data."""
        a = self._raw.get("autonomy", {})
        return AutonomyConfig(
            enabled=bool(a.get("enabled", False)),
            tick_interval_sec=float(a.get("tick_interval_sec", 5.0)),
            min_battery_pct=float(a.get("min_battery_pct", 30.0)),
            routines=list(a.get("routines", [])),
        )

    # ── Convenience accessors ─────────────────────────────────────────────────

    @property
    def capability_set(self) -> Optional[set]:
        """
        Set of enabled capability flags from the profile, or None when the
        profile declares no 'capabilities' section (None = allow everything).
        """
        caps = self._raw.get("capabilities")
        if caps is None:
            return None
        return {name for name, enabled in caps.items() if enabled}

    @property
    def robot_id(self) -> str:
        """Unique identifier for this Kova unit (e.g. 'kova-01')."""
        return self._raw.get("robot_id", "unknown")

    @property
    def version(self) -> str:
        """Firmware / software version string."""
        return self._raw.get("version", "0.0.0")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Dot-notation accessor for arbitrary raw config values.

        Parameters
        ----------
        key : str
            Dot-separated path, e.g. 'hardware.arm.max_payload'.
        default : Any
            Returned if the path does not exist.
        """
        parts = key.split(".")
        value: Any = self._raw
        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default

    @classmethod
    def reset(cls) -> None:
        """
        Destroy the singleton — used in tests to reload config with a
        different profile path.
        """
        cls._instance = None


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

config = Config()
