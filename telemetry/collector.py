#!/usr/bin/env python3
"""
================================================================================
Project     : River Kova — Household Chore Robot Control System
File        : telemetry/collector.py
Purpose     : Metrics collector and InfluxDB publisher. Gathers system metrics
              (battery, CPU, memory, task counts, safety events) and pushes
              them to InfluxDB at a configurable interval.
Author      : [Author Placeholder]
Version     : 1.0.0
Date        : 2026-05-25
License     : Proprietary — River Song AI (riversongai.com)
================================================================================
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

try:
    from influxdb_client import InfluxDBClient, WriteOptions
    from influxdb_client.client.write_api import SYNCHRONOUS
    _INFLUX_AVAILABLE = True
except ImportError:
    _INFLUX_AVAILABLE = False
    log.warning("influxdb-client not available — TelemetryCollector in stub mode.")

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


class TelemetryCollector:
    """
    System metrics collector and InfluxDB publisher.

    Collects metrics from registered providers and pushes them to InfluxDB
    at a fixed interval. Metrics are also available in-memory for the
    River Song API heartbeat.

    Attributes
    ----------
    robot_id : str
        Unit identifier used as an InfluxDB tag.
    latest_metrics : dict
        Most recently collected metrics snapshot.
    """

    def __init__(
        self,
        robot_id: str,
        influxdb_url: str = "http://localhost:8086",
        token: str = "",
        org: str = "river-kova",
        bucket: str = "kova-telemetry",
        push_interval: float = 5.0,
    ) -> None:
        """
        Initialise the telemetry collector.

        Parameters
        ----------
        robot_id : str
            Unit identifier.
        influxdb_url : str
            InfluxDB server URL.
        token : str
            InfluxDB authentication token.
        org : str
            InfluxDB organisation.
        bucket : str
            InfluxDB bucket name.
        push_interval : float
            Seconds between metric pushes.
        """
        self.robot_id = robot_id
        self._url = influxdb_url
        self._token = token
        self._org = org
        self._bucket = bucket
        self._push_interval = push_interval

        self.latest_metrics: Dict[str, Any] = {}
        self._providers: Dict[str, Any] = {}
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._client = None
        self._write_api = None

        if _INFLUX_AVAILABLE and token:
            try:
                self._client = InfluxDBClient(url=influxdb_url, token=token, org=org)
                self._write_api = self._client.write_api(write_options=SYNCHRONOUS)
                log.info("TelemetryCollector: InfluxDB connected (%s).", influxdb_url)
            except Exception as exc:
                log.warning("TelemetryCollector: InfluxDB connection failed: %s", exc)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background metrics collection thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._collect_loop,
            name="kova-telemetry",
            daemon=True,
        )
        self._thread.start()
        log.info("TelemetryCollector started (interval=%.1fs).", self._push_interval)

    def stop(self) -> None:
        """Stop the collection thread and close the InfluxDB connection."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
        log.info("TelemetryCollector stopped.")

    # ── Provider registration ─────────────────────────────────────────────────

    def register_provider(self, name: str, provider: Any) -> None:
        """
        Register a metric provider object.

        The provider must have a get_metrics() method that returns a dict.

        Parameters
        ----------
        name : str
            Provider name (used as a metric prefix).
        provider : object
            Object with a get_metrics() -> dict method.
        """
        self._providers[name] = provider
        log.debug("TelemetryCollector: registered provider '%s'.", name)

    # ── Collection loop ───────────────────────────────────────────────────────

    def _collect_loop(self) -> None:
        """Background thread: collect and push metrics at the configured interval."""
        while self._running:
            try:
                metrics = self._collect()
                self.latest_metrics = metrics
                self._push(metrics)
            except Exception as exc:
                log.error("TelemetryCollector collect error: %s", exc)
            time.sleep(self._push_interval)

    def _collect(self) -> Dict[str, Any]:
        """
        Gather metrics from all sources.

        Returns
        -------
        dict
            Flat metrics dict with string keys and numeric values.
        """
        metrics: Dict[str, Any] = {
            "timestamp": time.time(),
            "robot_id": self.robot_id,
        }

        # System metrics
        if _PSUTIL_AVAILABLE:
            metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
            metrics["memory_percent"] = psutil.virtual_memory().percent
            metrics["disk_percent"] = psutil.disk_usage("/").percent
            temps = psutil.sensors_temperatures() if hasattr(psutil, "sensors_temperatures") else {}
            if "cpu_thermal" in temps:
                metrics["cpu_temp_c"] = temps["cpu_thermal"][0].current

        # Provider metrics
        for name, provider in self._providers.items():
            try:
                provider_metrics = provider.get_metrics()
                for k, v in provider_metrics.items():
                    metrics[f"{name}_{k}"] = v
            except Exception as exc:
                log.debug("TelemetryCollector: provider '%s' error: %s", name, exc)

        return metrics

    def _push(self, metrics: Dict[str, Any]) -> None:
        """
        Push metrics to InfluxDB.

        Parameters
        ----------
        metrics : dict
            Metrics to write.
        """
        if not self._write_api:
            return

        try:
            from influxdb_client import Point
            point = Point("kova_metrics").tag("robot_id", self.robot_id)
            for key, value in metrics.items():
                if key in ("timestamp", "robot_id"):
                    continue
                if isinstance(value, (int, float)):
                    point = point.field(key, value)
            self._write_api.write(bucket=self._bucket, org=self._org, record=point)
        except Exception as exc:
            log.debug("TelemetryCollector: InfluxDB write error: %s", exc)
