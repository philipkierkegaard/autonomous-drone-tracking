"""Offline reproductions of deployment audit findings; never opens hardware.

Loads selected definitions through AST to avoid model loading, MAVSDK, GUI,
or camera initialization. Assertions characterize existing defects, not safety.
Run from the repository root with .venv/bin/python scripts/audit_deployment_logic.py.
"""
from __future__ import annotations

import ast
import asyncio
import math
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)


ns = dict(globals())
load_definitions("control/spatial.py", {"unproject_camera_to_horizon"}, ns)
load_definitions("control/pid_controller.py", {"KinematicVisualServoController"}, ns)
load_definitions("perception/pipeline.py", {"KalmanBoxTracker", "DroneTracker", "compute_iou_matrix"}, ns)
load_definitions("run_tracker.py", {"VehicleState", "PerceptionState", "AutonomousTrackerNode"}, ns)
C = ns["KinematicVisualServoController"]
Node = ns["AutonomousTrackerNode"]


def check(name, condition, details):
    assert condition, name
    print(f"REPRODUCED: {name}: {details}")


c = C(search_mode="legacy_coast")
c.prev_cmd_vx = 3.0
coast = [c.compute_cmd({"status": "SEARCHING"}, dt=0.02)[0] for _ in range(40)]
check("search coast does not progressively decelerate", np.allclose(coast, 2.97), f"40 outputs = {coast[0]:.2f} m/s")
for _ in range(70):
    scan = c.compute_cmd({"status": "SEARCHING"}, dt=0.02)
check("persistent sensor failure triggers yaw scan", scan[3] == 15.0, scan.tolist())

# Verification of remediated Option A immediate brake
c_remediated = C(search_mode="brake")
c_remediated.prev_cmd_vx = 3.0
brake_out = c_remediated.compute_cmd({"status": "SEARCHING"}, dt=0.02)
check("remediated search mode immediately commands zero-velocity brake", np.allclose(brake_out, 0.0), brake_out.tolist())

c = C(use_bbox_size=True, coast_mode="follow")
for _ in range(60):
    cmd = c.compute_cmd({"status": "COASTING", "error_x": 0.4, "error_y": 0, "error_range": 0.5}, dt=0.02)
check("coasting can accelerate toward unseen target", cmd[0] > 0 and cmd[1] > 0, cmd.tolist())

# Verification of remediated coasting governor
c_coast_rem = C(use_bbox_size=True, coast_mode="decay")
cmd_rem = c_coast_rem.compute_cmd({"status": "COASTING", "error_x": 0.4, "error_y": 0, "error_range": 0.5}, dt=0.02)
check("remediated coast mode inhibits forward acceleration", cmd_rem[0] == 0.0, cmd_rem.tolist())

c = C()
bad = c.compute_cmd({"status": "LOCKED", "error_x": float("nan"), "error_y": 0, "distance": 3.5})
check("remediated controller safely clamps nonfinite telemetry to zero", np.all(np.isfinite(bad)) and np.allclose(bad, 0.0), bad.tolist())

n = Node.__new__(Node)
n.args = SimpleNamespace(min_alt=2.0, max_alt=30.0, alt_cushion=1.0, max_climb=2.5, max_desc=1.5)
n.vehicle_state = ns["VehicleState"]()
v, state = n.enforce_altitude_limits(0.0, n.get_current_altitude())
check("remediated altitude logic prevents uncommanded climb on ground", v == 0.0 and state == "ON_GROUND", (v, state))
n.vehicle_state.altitude_rel_m = 10.0
n.vehicle_state.distance_sensor_m = 0.5
check("range sensor accepted without health or age", n.get_current_altitude() == 0.5, n.get_current_altitude())
n.vehicle_state.in_air = True
v, state = n.enforce_altitude_limits(1.0, float("nan"))
check("remediated altitude logic clamps descent to zero on invalid altitude", v == 0.0 and state == "INVALID_ALT", (v, state))
n.vehicle_state.is_taking_off = True
n.vehicle_state.in_air = False
v, state = n.enforce_altitude_limits(0.0, 0.0)
check("takeoff transition inhibits autonomous vertical commands", v == 0.0 and state == "TAKEOFF_ACTIVE", (v, state))
n.vehicle_state.is_taking_off = False

k = ns["KalmanBoxTracker"](np.array([0., 0., 20., 20.]))
k.x[4, 0] = 60.0
initial = float(k.x[0, 0])
for _ in range(15):
    k.predict()
displacement = float(k.x[0, 0]) - initial
check("Kalman time assumes 30 FPS", np.isclose(displacement, 30.0), f"15 calls displace {displacement}px even if capture runs at 15 FPS")

t = ns["DroneTracker"]()
t.trackers = [ns["KalmanBoxTracker"](np.array([0., 0., 20., 20.]))]
t.trackers[0].hits = 2
tracks = t.update(np.empty((0, 4)), np.empty(0))
check("new primary lock may select an already coasting track", t.primary_track_id is not None and tracks[0]["is_coasting"], t.primary_track_id)


async def reproduce_runtime_watchdog():
    n = Node.__new__(Node)
    n.args = SimpleNamespace(ctrl_rate=500, watchdog_timeout=0.4, max_accel_xy=4., max_accel_z=3., max_accel_yaw=180., dry_run=False)
    n.vehicle_state = ns["VehicleState"]()
    n.perception_state = ns["PerceptionState"]()
    n.controller = C(search_mode="legacy_coast")
    n.controller.prev_cmd_vx = 3.0
    n.drone = None
    n.running = True
    n.get_current_altitude = lambda: 5.0
    n.enforce_altitude_limits = lambda v, alt: (v, "OK")
    task = asyncio.create_task(n.control_loop())
    await asyncio.sleep(0.06)
    n.running = False
    await task
    check("actual runtime watchdog emits forward motion", n.perception_state.is_stale and n.latest_cmd_safe[0] > 0.0, n.latest_cmd_safe.tolist())


asyncio.run(reproduce_runtime_watchdog())
print("All 10 defect characterizations reproduced. This is not a flight acceptance test.")
