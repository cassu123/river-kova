# River Kova

**River Kova** is the household chore robot control system for the [River Song AI](https://riversongai.com) personal ecosystem. It controls a fleet of autonomous robots that clean, fetch, organise, and navigate the home — all orchestrated by River Song voice commands and scheduling.

---

## Overview

River Kova runs on a **Raspberry Pi 5** as the main compute unit, with a **Raspberry Pi Pico** handling low-level I/O. The system is built on **ROS2 Humble**, uses **MoveIt2** for arm motion planning, and communicates with the River Song cloud over **WireGuard VPN + 4G LTE / WiFi**.

Human safety is the absolute top priority. The unit performs an immediate full stop if a person is detected within 1 metre, and no hardware initialises until all safety systems are live.

---

## Architecture

```
river-kova/
├── core/               # Entry point, config, constants
├── hardware/           # Pico bridge, drive, arm, gripper, camera, battery
├── safety/             # EStop, watchdog, fault manager, human detection, collision avoidance
├── navigation/         # Room mapper (occupancy grid), A* path planner, obstacle avoidance, return-to-base
├── vision/             # Camera feed, YOLO object detection, classifier, MJPEG stream server
├── telemetry/          # Structured logger, InfluxDB collector, alert manager
├── connectivity/       # WiFi manager, WireGuard VPN, River Song API client
├── tasks/              # Chore library, priority task queue, task manager, task executor
├── units/              # Per-unit JSON profile (kova_profile.json)
└── tests/              # pytest test suites
```

---

## Compute Stack

| Component | Technology |
|---|---|
| Main compute | Raspberry Pi 5 |
| Low-level I/O | Raspberry Pi Pico (serial bridge) |
| OS | Ubuntu 22.04 |
| Language | Python 3.11+ |
| Robot middleware | ROS2 Humble |
| Arm motion planning | MoveIt2 |
| Computer vision | OpenCV 4 + MediaPipe + YOLO |
| Depth camera | Intel RealSense D435 |
| LiDAR | RPLiDAR A1 |
| API server | FastAPI + Uvicorn |
| Metrics | InfluxDB |
| VPN | WireGuard |
| Connectivity | WiFi + 4G LTE |

---

## Safety Systems

Safety initialises **before any hardware** on every boot. The hierarchy is:

1. **EStop** — hardware + software emergency stop with mandatory recovery delay
2. **Watchdog** — fires if the main control loop stalls for > 5 seconds
3. **FaultManager** — centralised fault registry with retry tracking and River Song reporting
4. **HumanDetection** — MediaPipe Pose + depth camera, three concentric zones:
   - **WARN** zone (2.5 m) — alert, continue at reduced speed
   - **SLOW** zone (1.5 m) — reduce to safe speed
   - **ESTOP** zone (1.0 m) — **immediate full stop**
5. **CollisionAvoidance** — force/torque sensor + LiDAR proximity monitoring

---

## Chore Library

Built-in chores (all extensible at runtime):

| Chore | Description |
|---|---|
| `VACUUM` | Boustrophedon coverage pattern |
| `MOP` | Coverage pattern at reduced speed |
| `FETCH` | Detect → navigate → grasp → deliver |
| `ORGANIZE` | Scan room, pick up out-of-place items |
| `WIPE_SURFACE` | Navigate to surface, arm wipe motion |
| `TAKE_OUT_TRASH` | Grasp bag, carry to collection point |
| `LOAD_DISHWASHER` | Collect dishes, load into dishwasher |
| `UNLOAD_DISHWASHER` | Remove clean dishes, place in cupboards |
| `LAUNDRY_TRANSFER` | Transfer washer → dryer |
| `CUSTOM` | User-defined via `ChoreLibrary.register()` |

---

## River Song Integration

All Kova API routes are mounted under `/api/kova/` on the River Song server.

| Endpoint | Purpose |
|---|---|
| `POST /api/kova/units/register` | Unit registration on boot |
| `POST /api/kova/heartbeat` | Periodic state + battery heartbeat |
| `GET  /api/kova/units/{id}/tasks` | Poll for pending tasks |
| `POST /api/kova/tasks/{id}/status` | Report task completion / failure |
| `POST /api/kova/telemetry` | Push metrics snapshot |
| `POST /api/kova/alerts` | Push safety / system alerts |
| `POST /api/kova/units/deregister` | Clean shutdown deregistration |

**Voice command example:**
> "River, have Kova clean the kitchen"

The `TaskManager.submit_from_voice()` method parses the command, maps it to a chore type and room, and enqueues it with priority 7.

---

## Live Camera Stream

The `StreamServer` serves an MJPEG stream at:

```
http://<unit-ip>:8080/stream      # Live MJPEG feed
http://<unit-ip>:8080/snapshot    # Single JPEG frame
http://<unit-ip>:8080/health      # Health check
```

---

## Getting Started

### 1. Prerequisites

```bash
# ROS2 Humble
sudo apt install ros-humble-desktop ros-humble-moveit

# Python dependencies
pip install -r requirements.txt

# WireGuard
sudo apt install wireguard
```

### 2. Configure the unit profile

Edit `units/kova_profile.json` for your hardware:

```json
{
  "robot_id": "kova-01",
  "hardware": {
    "pico_serial_port": "/dev/ttyACM0",
    "sensors": { "camera": "intel-realsense-d435" }
  },
  "connectivity": {
    "api_endpoint": "https://api.riversongai.com"
  }
}
```

### 3. Set environment variables

```bash
export KOVA_API_KEY="your-river-song-api-key"
export KOVA_INFLUX_TOKEN="your-influxdb-token"
export KOVA_PROFILE="/path/to/kova_profile.json"   # optional override
```

### 4. Source ROS2 and run

```bash
source /opt/ros/humble/setup.bash
python3 core/main.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Fleet Management

Multiple Kova units can be deployed simultaneously. Each unit has its own `kova_profile.json` with a unique `robot_id`. The `fleet` section of the profile assigns rooms:

```json
"fleet": {
  "fleet_id": "home-01",
  "assigned_rooms": ["kitchen", "living_room", "hallway"]
}
```

River Song coordinates task assignment across the fleet from the dashboard.

---

## Project Info

- **Ecosystem:** River Song AI — [riversongai.com](https://riversongai.com)
- **Version:** 1.0.0
- **License:** Proprietary
