# River Kova — Architecture & Universal Robot Guide

River Kova is a **universal household-robot brain**. The goal: buy or build any
robot body later — a wheeled one-armed helper (Neo/Ballie/Astro style), a
converted "dumb" vacuum, a custom build — plug it into this program, and the
same brain runs it. Self-hosted on the robot, optionally managed by the
River Song AI server (riversongai.com).

---

## The Layer Stack

```
┌────────────────────────────────────────────────────────────────┐
│  River Song server (riversongai.com)          [FUTURE / TABLED]│
│  setup • remote control • voice routing • fleet dashboard      │
└──────────────────────────┬─────────────────────────────────────┘
                           │ /api/kova/* (offline-tolerant)
┌──────────────────────────┴─────────────────────────────────────┐
│  LOCAL CONTROL API (api/local_server.py)                       │
│  Self-hosted REST: status, submit chores, voice, e-stop        │
├────────────────────────────────────────────────────────────────┤
│  AUTONOMY (autonomy/initiative_engine.py)                      │
│  "What needs doing?" — schedules, observations, pluggable rules│
├────────────────────────────────────────────────────────────────┤
│  TASKS (tasks/)                                                │
│  Chore library • priority queue • capability gate • executor  │
├────────────────────────────────────────────────────────────────┤
│  NAVIGATION + VISION (navigation/, vision/)                    │
│  Mapping, A*, coverage • detection, classification, streaming  │
├────────────────────────────────────────────────────────────────┤
│  SAFETY (safety/) — boots FIRST, vetoes everything             │
│  E-stop • watchdog • faults • human detection • collision      │
├────────────────────────────────────────────────────────────────┤
│  HARDWARE ABSTRACTION (hardware/interfaces.py + factory.py)    │
│  IOBridge • Manipulator • FrameSource contracts                │
├──────────────┬──────────────────────┬──────────────────────────┤
│  Pico serial │  SIMULATED BODY      │  your next robot here    │
│  (reference) │  (simulation/)       │  (new adapter)           │
└──────────────┴──────────────────────┴──────────────────────────┘
```

Everything above the hardware abstraction line never knows which body it is
driving. That is the universality guarantee.

---

## Porting the Brain to a New Robot Body

When you buy/build a robot, you write **one adapter and one profile**:

1. **Implement the I/O bridge** — satisfy the `IOBridge` contract in
   `hardware/interfaces.py` (motors, battery, encoders, force, gripper).
   If the body speaks the same JSON-over-serial protocol as the reference
   build, just point `PicoBridge` at its port and you're done.
   `simulation/sim_pico_bridge.py` is a complete worked example (~100 lines).
2. **Register a backend** — add a value to `core/constants.HardwareBackend`
   and a branch in `hardware/factory.build_hardware()`.
3. **Write the unit profile** — copy `units/kova_profile.json`, set
   `hardware.backend`, and declare honest `capabilities` (see below).
4. Run `pytest tests/` — the brain-level tests already cover your adapter
   through the same interfaces.

Bodies without an arm, without a mop, without anything — fine. The capability
gate keeps impossible chores away from them.

## The Capability Model

Chores declare what they need; profiles declare what the body has; the
`TaskManager` refuses mismatches at submit time.

```jsonc
// chore (tasks/chore_library.py)
"required_capabilities": ["feed_dogs", "arm_manipulation"]

// profile (units/*.json)
"capabilities": { "arm_manipulation": true, "feed_dogs": true, "mop": false }
```

A voice command like "mop the kitchen" sent to an armless vacuum body simply
returns a polite rejection instead of a faceplant.

---

## Simulation Mode — Develop With Zero Hardware

The simulated body (`simulation/`) is a complete stand-in: differential-drive
physics, battery that drains and recharges at the dock, a 5-room home with
graspable objects, and a gripper that actually picks things up.

```bash
pip install -r requirements.txt
KOVA_PROFILE=units/kova_sim_profile.json python3 -m core.main
```

What you'll see: full safety-first boot, the initiative engine noticing the
out-of-place cup and sock and dispatching ORGANIZE on its own, heartbeat-safe
task execution, and a clean Ctrl-C shutdown.

While it runs, the local control API is live:

```bash
curl localhost:8000/status
curl -X POST localhost:8000/voice -H 'content-type: application/json' \
     -d '{"command": "have kova feed the dogs"}'
curl -X POST localhost:8000/estop
```

`tests/test_boot_sim.py` runs this exact scenario in CI.

---

## Autonomy — "Determine What Needs to Be Done"

`autonomy/initiative_engine.py` is the seed of self-directed behaviour. It is
rule-based and deliberately pluggable:

| Today | Next | Later |
|---|---|---|
| `ScheduledRoutineRule` — profile routines ("feed dogs 07:30 daily") | Vision-backed `TidyUpRule` on real cameras | River Song LLM rule: proposes chores from household context, conversations, patterns |
| `TidyUpRule` — sees out-of-place objects, queues ORGANIZE | Battery/charging strategy rules | Multi-unit fleet arbitration via River Song |
| `ExploreRule` — home unfamiliar, queues EXPLORE to learn the layout | Frontier-based exploration on a live LiDAR map | Continuous re-mapping as the home changes |

Initiative is suppressed when battery is low, when the robot isn't IDLE, or
when safety isn't NOMINAL. Initiative tasks carry lower priority than direct
human commands, so "get me water" always jumps the queue.

## Room Recognition — Learned, Not Configured

The brain never relies on a preloaded floor plan. `vision/room_classifier.py`
infers room types from visible object classes (stove+fridge → kitchen);
`navigation/semantic_map.py` accumulates those observations into a persistent
grid whose labels decay and re-learn as the home changes. The control loop
feeds it passively every 0.5 s from whatever the body is doing. In simulation
the observations come from `SimWorld.visible_objects()` (walls block sight);
on real hardware the identical `observe()` call is fed by the YOLO detector.
`navigation/explorer.py` provides the closed-loop drive-to-point primitive and
the discovery sweep, written against the `IOBridge` contract only.

## Safety Invariants (non-negotiable, body-agnostic)

1. Safety subsystems boot **before** any hardware driver.
2. Human within 1 m → immediate full stop, whatever the body.
3. The watchdog supervises the control loop; chores run on a worker thread so
   long tasks can never starve it.
4. Network I/O to River Song is fire-and-forget from a background sender —
   an unreachable server can never block the brain.
5. E-stop recovery requires an explicit clear plus a mandatory delay.

---

## River Song Server Integration (tabled until the server exists)

The robot is **server-optional by design**: with `connectivity.enabled: false`
it is fully self-hosted forever. When riversongai.com is ready, flip the flag
and the already-written client (`connectivity/api_client.py`) starts
registering, heartbeating, and accepting remote tasks — no robot-side changes.

Contract the server must implement (all under `/api/kova/`):

| Endpoint | Direction | Purpose |
|---|---|---|
| `POST /units/register` / `/units/deregister` | robot → server | lifecycle |
| `POST /heartbeat` | robot → server | state, safety, battery every 5 s |
| `GET  /units/{id}/tasks` | robot polls | remote task assignment |
| `POST /tasks/{id}/status` | robot → server | completion/failure reports |
| `POST /telemetry`, `POST /alerts` | robot → server | metrics & alerts |

---

## Repository Map

| Path | Role |
|---|---|
| `core/` | entry point, config singleton, constants/enums |
| `safety/` | e-stop, watchdog, faults, human detection, collision |
| `hardware/` | HAL contracts, backend factory, reference Pico drivers |
| `simulation/` | simulated body + world (no hardware required) |
| `navigation/`, `vision/` | mapping/planning, detection/streaming |
| `tasks/` | chore library, queue, manager (capability gate), executor |
| `autonomy/` | initiative engine + rules |
| `api/` | self-hosted local control REST API |
| `connectivity/` | WiFi/VPN, offline-tolerant River Song client |
| `telemetry/` | logging, metrics, alerting |
| `units/` | per-unit profiles (`kova_profile.json`, `kova_sim_profile.json`) |
| `tests/` | 175 tests incl. end-to-end simulated boot |
