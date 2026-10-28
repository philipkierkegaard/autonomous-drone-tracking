#!/usr/bin/env python3
"""
Automated Parameter Tuning for Classical Kinematic Visual Servoing Controller
Under Realistic In-Flight Observation Dynamics.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception.

Optimizes 7 key visual servoing parameters against the realistic observation pipeline:
    - 30 Hz camera sampling rate
    - 5% random frame dropouts
    - Synthetic YOLO bounding box noise & false positives (Jetson empirical mode)
    - 8D Kalman filter tracking state estimation (DroneTracker)
    - SO(3) gyroscopic ego-motion compensation (ego_shifts)
    - Coasting / searching recovery dynamics (momentum_cruise)

Target objective:
    1. 1.0s Basket Dwell Retention ([2.5m, 4.5m] within camera FOV)
    2. Image plane tracking error: (e_x^2 + e_y^2) with smooth unclipped off-screen projection
    3. Standoff distance error: (d - 3.5m)^2
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import multiprocessing as mp
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats.qmc import LatinHypercube, scale

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.simulation import FastPixhawkQuadSim
from control.pid_controller import KinematicVisualServoController
from control.trajectory import StochasticTargetTrajectory
from control.dataset_generator import PursuitDatasetGenerator, EncounterType, PursuitScenario, PursuitDataset
from control.synthetic_sensor import SyntheticYOLOSensor
from perception.pipeline import DroneTracker
from control.spatial import compute_ego_motion_compensation


def create_tuning_suite(
    suite_path: Path,
    base_seed: int = 60_000,
    duration: float = 18.0,
    scenarios_per_group: Tuple[int, int, int] = (4, 6, 6)
) -> List[Dict[str, Any]]:
    """
    Creates a dedicated, independent 80-scenario tuning dataset with balanced distribution
    across Cruising (20), Evasive (30), and Hyper-Evasive (30) regimes across all 5 encounter geometries.
    """
    if suite_path.exists():
        print(f"[TUNING] Loading existing tuning suite from: {suite_path}")
        with open(suite_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["scenarios"]

    c_cnt, e_cnt, h_cnt = scenarios_per_group
    total_expected = 5 * c_cnt + 5 * e_cnt + 5 * h_cnt
    print(f"[TUNING] Generating balanced {total_expected}-scenario realistic tuning dataset (base_seed={base_seed})...")
    gen = PursuitDatasetGenerator()
    scenarios = []
    
    distribution = [
        ("cruising", EncounterType.TAIL_CHASE, c_cnt),
        ("cruising", EncounterType.CROSSING_RIGHT, c_cnt),
        ("cruising", EncounterType.CROSSING_LEFT, c_cnt),
        ("cruising", EncounterType.HEAD_ON, c_cnt),
        ("cruising", EncounterType.OBLIQUE_AWAY, c_cnt),
        ("evasive", EncounterType.TAIL_CHASE, e_cnt),
        ("evasive", EncounterType.CROSSING_RIGHT, e_cnt),
        ("evasive", EncounterType.CROSSING_LEFT, e_cnt),
        ("evasive", EncounterType.HEAD_ON, e_cnt),
        ("evasive", EncounterType.OBLIQUE_AWAY, e_cnt),
        ("hyper_evasive", EncounterType.TAIL_CHASE, h_cnt),
        ("hyper_evasive", EncounterType.CROSSING_RIGHT, h_cnt),
        ("hyper_evasive", EncounterType.CROSSING_LEFT, h_cnt),
        ("hyper_evasive", EncounterType.HEAD_ON, h_cnt),
        ("hyper_evasive", EncounterType.OBLIQUE_AWAY, h_cnt),
    ]

    scen_idx = 1
    speeds_cycle = [0.0, 4.0, 5.0, 6.5]  # 25% hover scramble, 50% cruise matching, 25% fast ingress
    for prof, enc, count in distribution:
        for c in range(count):
            seed = base_seed + scen_idx * 37
            scen_id = f"tune_{scen_idx:03d}_{prof}_{enc.value}_s{seed}"
            init_speed = speeds_cycle[(scen_idx - 1) % len(speeds_cycle)]
            sc = gen.generate_scenario(
                scenario_id=scen_id,
                profile=prof,
                encounter_type=enc,
                seed=seed,
                duration=duration,
                chaser_alt_m=3.0
            )
            sc_dict = sc.to_dict()
            sc_dict["initial_chaser_speed"] = float(init_speed)
            scenarios.append(sc_dict)
            scen_idx += 1

    suite_path.parent.mkdir(parents=True, exist_ok=True)
    with open(suite_path, "w", encoding="utf-8") as f:
        json.dump({"name": suite_path.stem, "num_scenarios": len(scenarios), "scenarios": scenarios}, f, indent=2)

    print(f"[TUNING] Successfully saved {len(scenarios)} tuning scenarios to: {suite_path}")
    return scenarios


def simulate_scenario(
    scenario: Dict[str, Any],
    params: np.ndarray,
    dt: float = 0.02,
    lambda_image: float = 15.0,
    camera_fps: float = 30.0,
    frame_drop_rate: float = 0.05
) -> Dict[str, float]:
    """
    Simulates a flight scenario under given candidate PID parameters driven by the
    realistic in-flight observation pipeline:
      - 30 Hz camera ticks with empirical YOLO noise & 5% frame drops
      - 8D Kalman filter tracking state estimation
      - SO(3) gyroscopic ego-motion compensation
      - Momentum cruise search recovery dynamics
    """
    (
        kp_fwd, ki_fwd, kd_fwd,
        max_decel, centering_damp, min_centering,
        kp_lat, ki_lat, kd_lat,
        kp_z, ki_z, kd_z,
        kp_yaw, ki_yaw, kd_yaw
    ) = params

    c_init = scenario.get("initial_chaser_pos", [0.0, 0.0, -2.5])
    c_yaw = float(scenario.get("initial_chaser_yaw", 0.0))
    # Diverse operational initial velocities: read from scenario or fallback to deterministic cycle
    fallback_speeds = [0.0, 4.0, 5.0, 6.5]
    seed_idx = int(scenario.get("seed", 0)) % len(fallback_speeds)
    c_speed = float(scenario.get("initial_chaser_speed", fallback_speeds[seed_idx]))
    c_vel = np.array([np.cos(c_yaw) * c_speed, np.sin(c_yaw) * c_speed, 0.0], dtype=np.float64)

    # 1. Quadrotor simulator matching 15 m/s physical flight envelope
    sim = FastPixhawkQuadSim(
        dt=dt,
        max_vel_xy=15.0,
        max_accel_xy=6.0,
        max_vel_up=3.0,
        max_vel_down=2.0
    )
    sim.reset(initial_pos=np.array(c_init, dtype=np.float64), initial_yaw=c_yaw, initial_vel=c_vel)

    # 2. Candidate Controller with Momentum Cruise recovery
    ctrl = KinematicVisualServoController(
        camera_uptilt_deg=15.0,
        hfov_deg=62.0,
        vfov_deg=38.0,
        desired_bbox_size=110.0,
        desired_standoff_dist=3.5,
        scale_mode="width",
        use_bbox_size=True,
        enable_lateral_strafe=True,
        kp_forward=float(kp_fwd),
        ki_forward=float(ki_fwd),
        kd_forward=float(kd_fwd),
        centering_damping=float(centering_damp),
        min_centering_factor=float(min_centering),
        kp_lat=float(kp_lat),
        ki_lat=float(ki_lat),
        kd_lat=float(kd_lat),
        kp_z=float(kp_z),
        ki_z=float(ki_z),
        kd_z=float(kd_z),
        kp_yaw=float(kp_yaw),
        ki_yaw=float(ki_yaw),
        kd_yaw=float(kd_yaw),
        max_lat_vel=4.0,
        max_lat_accel=6.0,
        max_accel=6.0,
        max_decel=float(max_decel),
        max_accel_yaw=240.0,
        min_limits=np.array([-0.5, -4.0, -3.0, -120.0]),
        max_limits=np.array([15.0,  4.0,  2.0,  120.0]),
        search_mode="momentum_cruise",
    )

    # 3. Target Trajectory
    target = StochasticTargetTrajectory(
        duration=scenario["duration"] + 5.0,
        profile=scenario["profile"],
        initial_pos=scenario["initial_target_pos"],
        initial_heading_rad=scenario["initial_target_heading_rad"],
        seed=int(scenario["seed"])
    )

    # 4. Realistic Perception Pipeline
    seed = int(scenario["seed"])
    rng = np.random.default_rng(seed)
    synthetic_sensor = SyntheticYOLOSensor(
        img_w=1280,
        img_h=720,
        env_mode="nominal",
        seed=seed
    )
    drone_tracker = DroneTracker(
        dt=dt,
        max_lost_frames=15,
        min_hits_to_confirm=2
    )

    camera_dt = 1.0 / camera_fps
    time_since_camera = 0.0

    prev_chaser_pitch = sim.pitch
    prev_chaser_roll = sim.roll
    prev_chaser_yaw = sim.yaw

    total_steps = int(round(scenario["duration"] / dt))
    
    standoff_costs = []
    image_costs = []
    in_views = []
    dists = []
    continuous_dwell = 0.0
    max_continuous_dwell = 0.0
    proximity_penalty = 0.0
    collision = False
    ground_strike = False

    for step in range(total_steps):
        t = step * dt
        tgt_pos = target.get_position(t)

        # Ground-truth camera projection
        true_telem = sim.get_camera_telemetry(
            tgt_pos,
            hfov_deg=62.0,
            vfov_deg=38.0,
            target_w_m=0.35,
            target_h_m=0.20,
            img_w=1280,
            img_h=720,
            desired_target_size=110.0
        )
        
        dist = float(true_telem["distance"])
        dists.append(dist)

        # Unclipped image plane projection for continuous optimization gradient
        u_ex, u_ey, in_view, _, _, _ = sim.project_to_camera(
            tgt_pos,
            hfov_deg=62.0,
            vfov_deg=38.0,
            target_w_m=0.35,
            target_h_m=0.20,
            img_w=1280,
            img_h=720,
            return_unclipped=True
        )
        in_views.append(bool(in_view))

        # Standoff error: (d - 3.5)^2
        err_standoff = (dist - 3.5) ** 2
        standoff_costs.append(err_standoff)

        # Saturating robust visual loss (Geman-McClure / Cauchy formulation):
        # Prevents unclipped off-screen projection errors from quadratically blowing up
        # and dominating the engagement/dwell objectives
        e_sq = float(u_ex ** 2 + u_ey ** 2)
        err_image = float(e_sq / (1.0 + e_sq / 4.0))
        image_costs.append(err_image)

        # Strict continuous dwell tracking: strictly resets to 0.0 upon any engagement departure
        # (directly mirroring the benchmark evaluation)
        if in_view and (2.5 <= dist <= 4.5):
            continuous_dwell += dt
            max_continuous_dwell = max(max_continuous_dwell, continuous_dwell)
        else:
            continuous_dwell = 0.0

        # Safety & Proximity barrier (aligned with benchmark 0.5m collision threshold)
        if dist < 0.5:
            collision = True
        elif dist < 1.8:
            # Soft quadratic repulsive barrier discouraging uncomfortably close approaches
            prox_ratio = (1.8 - dist) / (1.8 - 0.5)
            proximity_penalty += float((prox_ratio ** 2) * 30.0 * dt)

        if -sim.pos[2] < 0.8:
            ground_strike = True

        # Attitude deltas for SO(3) gyroscopic ego-motion compensation
        d_pitch = sim.pitch - prev_chaser_pitch
        d_roll = sim.roll - prev_chaser_roll
        d_yaw = float((sim.yaw - prev_chaser_yaw + np.pi) % (2.0 * np.pi) - np.pi)
        prev_chaser_pitch = sim.pitch
        prev_chaser_roll = sim.roll
        prev_chaser_yaw = sim.yaw

        # Compute ego shifts for active Kalman trackers
        ego_shifts = {}
        for tr in drone_tracker.trackers:
            st = tr.get_state()
            tcx = (st[0] + st[2]) / 2.0
            tcy = (st[1] + st[3]) / 2.0
            tw = max(1.0, st[2] - st[0])
            th = max(1.0, st[3] - st[1])
            shift = compute_ego_motion_compensation(
                cx=tcx, cy=tcy, w=tw, h=th,
                delta_pitch_rad=d_pitch,
                delta_roll_rad=d_roll,
                delta_yaw_rad=d_yaw,
                vx_body=0.0, vy_body=0.0, vz_body=0.0,
                dt=dt,
                img_width=1280.0,
                img_height=720.0
            )
            ego_shifts[tr.id] = shift

        # -------------------------------------------------------------
        # Realistic Perception Processing (30Hz Camera + 50Hz Kalman)
        # -------------------------------------------------------------
        time_since_camera += dt
        is_camera_tick = (time_since_camera >= camera_dt - 1e-5)
        if is_camera_tick:
            time_since_camera -= camera_dt
            norm_cx = (true_telem["error_x"] + 1.0) / 2.0
            norm_cy = (true_telem["error_y"] + 1.0) / 2.0
            norm_w = true_telem["bbox_w"] / 1280.0
            norm_h = true_telem["bbox_h"] / 720.0
            in_fov = bool(true_telem["in_view"])

            det_bbox, conf, fps_boxes = synthetic_sensor.step(
                norm_cx, norm_cy, norm_w, norm_h, in_fov=in_fov
            )
            if frame_drop_rate > 0.0 and rng.uniform(0.0, 1.0) < frame_drop_rate:
                det_bbox = None

            detections = []
            confs = []
            if det_bbox is not None:
                detections.append(det_bbox)
                confs.append(conf)
            for fp in fps_boxes:
                detections.append(fp)
                confs.append(0.36)

            det_arr = np.array(detections, dtype=np.float32) if detections else np.empty((0, 4), dtype=np.float32)
            conf_arr = np.array(confs, dtype=np.float32) if confs else np.empty((0,), dtype=np.float32)
            drone_tracker.update(det_arr, conf_arr, ego_shifts=ego_shifts, dt_step=dt)
        else:
            for trk in drone_tracker.trackers:
                shift = ego_shifts.get(trk.id, None)
                trk.predict(ego_shift=shift, dt_step=dt)

        # Extract realistic controller telemetry (what the drone actually receives)
        primary = None
        if drone_tracker.primary_track_id is not None:
            primary = next((t for t in drone_tracker.trackers if t.id == drone_tracker.primary_track_id), None)

        if primary is not None and primary.hits >= drone_tracker.min_hits_to_confirm and primary.time_since_update <= 15:
            pred_box = primary.get_state()
            tcx = (pred_box[0] + pred_box[2]) / 2.0
            tcy = (pred_box[1] + pred_box[3]) / 2.0
            bw = max(1.0, pred_box[2] - pred_box[0])
            bh = max(1.0, pred_box[3] - pred_box[1])
            curr_sz = bw

            err_x = (tcx - 1280.0 / 2.0) / (1280.0 / 2.0)
            err_y = (tcy - 720.0 / 2.0) / (720.0 / 2.0)
            range_err = (110.0 - curr_sz) / 110.0

            ctrl_telem = {
                "status": "COASTING" if primary.time_since_update > 0 else "LOCKED",
                "error_x": float(np.clip(err_x, -1.0, 1.0)),
                "error_y": float(np.clip(err_y, -1.0, 1.0)),
                "error_range": float(np.clip(range_err, -1.0, 1.0)),
                "target_size": float(curr_sz),
                "desired_size": 110.0,
                "bbox_w": float(bw),
                "bbox_h": float(bh),
                "w_norm": float(bw / 1280.0),
                "scale_mode": "width",
            }
        else:
            ctrl_telem = {
                "status": "SEARCHING",
                "error_x": 0.0,
                "error_y": 0.0,
                "error_range": 0.0,
                "target_size": 0.0,
                "desired_size": 110.0,
                "bbox_w": 0.0,
                "bbox_h": 0.0,
                "w_norm": 0.0,
                "scale_mode": "width",
            }

        # Strictly sanitize telemetry: no controller may ever receive ground-truth metric distance
        ctrl_telem.pop("distance", None)

        # Compute command and step simulator
        c_y, s_y = np.cos(sim.yaw), np.sin(sim.yaw)
        v_body = np.array([c_y * sim.vel[0] + s_y * sim.vel[1], -s_y * sim.vel[0] + c_y * sim.vel[1], sim.vel[2]], dtype=np.float64)
        cmd = ctrl.compute_cmd(
            ctrl_telem,
            drone_pitch=sim.pitch,
            drone_roll=sim.roll,
            dt=dt,
            vehicle_vel=v_body,
            vehicle_yaw_rate=sim.yaw_rate,
            current_alt_m=-sim.pos[2]
        )
        sim.step(cmd)

    mean_standoff = float(np.mean(standoff_costs))
    mean_image = float(np.mean(image_costs))
    
    # Strict continuous dwell optimization: heavily reward >= 1.0s dwell
    dwell_penalty = max(0.0, 1.0 - max_continuous_dwell) * 30.0
    dwell_reward = -15.0 * min(3.0, max_continuous_dwell)

    base_cost = mean_standoff + lambda_image * mean_image + dwell_penalty + dwell_reward + proximity_penalty

    # Collision & Ground Strike Overwhelming Barrier Penalties
    # Strictly prevents unsafe parameter sets from trading collisions for tracking score
    safety_penalty = 0.0
    if collision:
        safety_penalty += 10000.0
    if ground_strike:
        safety_penalty += 2000.0

    total_cost = base_cost + safety_penalty

    return {
        "cost": total_cost,
        "mean_standoff_err_m2": mean_standoff,
        "mean_image_err": mean_image,
        "in_view_pct": float(np.mean(in_views) * 100.0),
        "min_dist": float(min(dists)),
        "final_dist": float(dists[-1]),
        "max_dwell_s": float(max_continuous_dwell),
        "collision": collision,
        "ground_strike": ground_strike
    }


def _eval_worker(args: Tuple[Dict[str, Any], np.ndarray, float]) -> float:
    sc, params, lambda_image = args
    res = simulate_scenario(sc, params, lambda_image=lambda_image)
    return res["cost"]


# Global pool holder for multi-process scenario evaluation
_GLOBAL_POOL = None


def evaluate_candidate(
    params: np.ndarray,
    scenarios: List[Dict[str, Any]],
    lambda_image: float = 15.0
) -> float:
    """Evaluates candidate parameters across the tuning scenarios and returns mean cost."""
    global _GLOBAL_POOL
    if _GLOBAL_POOL is not None:
        work_items = [(sc, params, lambda_image) for sc in scenarios]
        costs = _GLOBAL_POOL.map(_eval_worker, work_items)
    else:
        costs = [simulate_scenario(sc, params, lambda_image=lambda_image)["cost"] for sc in scenarios]
    return float(np.mean(costs))


PARAM_KEYS = [
    "kp_forward", "ki_forward", "kd_forward", "max_decel",
    "centering_damping", "min_centering_factor",
    "kp_lat", "ki_lat", "kd_lat",
    "kp_z", "ki_z", "kd_z",
    "kp_yaw", "ki_yaw", "kd_yaw"
]

DEFAULT_BOUNDS = [
    (0.2, 3.5),     # 0: kp_fwd (forward proportional gain)
    (0.0, 0.80),    # 1: ki_fwd (forward integral gain)
    (0.05, 0.60),   # 2: kd_fwd (forward derivative closing velocity damping)
    (1.5, 4.5),     # 3: max_decel (m/s² braking limit to prevent pitch-up blinding)
    (0.10, 1.20),   # 4: centering_damp (cornering deceleration factor per unit |e_x|)
    (0.20, 0.75),   # 5: min_centering_factor (speed retention floor during sharp turns)
    (0.5, 4.5),     # 6: kp_lat (lateral proportional gain)
    (0.0, 0.80),    # 7: ki_lat (lateral integral gain)
    (0.05, 1.2),    # 8: kd_lat (lateral derivative gain)
    (2.0, 9.0),     # 9: kp_z (vertical proportional gain)
    (0.0, 1.50),    # 10: ki_z (vertical integral gain)
    (0.2, 2.5),     # 11: kd_z (vertical derivative gain)
    (120.0, 320.0), # 12: kp_yaw (yaw proportional gain)
    (0.0, 35.0),    # 13: ki_yaw (yaw integral gain)
    (5.0, 75.0),    # 14: kd_yaw (yaw derivative gain)
]

BASELINE_PARAMS = np.array([
    1.2,    # kp_fwd
    0.35,   # ki_fwd
    0.20,   # kd_fwd
    3.0,    # max_decel
    0.70,   # centering_damp
    0.35,   # min_centering_factor
    2.0,    # kp_lat
    0.15,   # ki_lat
    0.40,   # kd_lat
    6.0,    # kp_z
    0.20,   # ki_z
    1.0,    # kd_z
    200.0,  # kp_yaw
    5.0,    # ki_yaw
    35.0,   # kd_yaw
], dtype=np.float64)


def run_tuning_optimization(
    scenarios: List[Dict[str, Any]],
    output_weights_path: Path,
    init_weights_path: Optional[Path] = None,
    lambda_image: float = 15.0,
    workers: int = 10,
    popsize: int = 2,
    maxiter: int = 8
) -> Dict[str, Any]:
    """Runs global differential evolution followed by L-BFGS-B polishing under realistic observations and attitude dynamics."""
    global _GLOBAL_POOL
    if workers > 1:
        ctx = mp.get_context("fork")
        _GLOBAL_POOL = ctx.Pool(workers)
        print(f"[TUNING] Initialized parallel worker pool with {workers} cores.")
    
    bounds = DEFAULT_BOUNDS
    lower_b = np.array([b[0] for b in bounds])
    upper_b = np.array([b[1] for b in bounds])
    baseline_params = BASELINE_PARAMS.copy()
    param_keys = PARAM_KEYS

    # Check for existing tuned weights to seed population
    prev_tuned_params = None
    p_init = None
    if init_weights_path is not None and Path(init_weights_path).exists():
        p_init = Path(init_weights_path)
    elif output_weights_path.exists():
        p_init = output_weights_path

    if p_init is not None and p_init.exists():
        try:
            with open(p_init, "r", encoding="utf-8") as f:
                d_init = json.load(f)
            t_params = d_init.get("tuned_parameters", d_init)
            prev_tuned_params = np.array([float(t_params[k]) for k in param_keys], dtype=np.float64)
            prev_tuned_params = np.clip(prev_tuned_params, lower_b, upper_b)
            print(f"[TUNING] Loaded previous tuned gains from: {p_init}")
        except Exception as e:
            print(f"[WARN] Failed to load previous tuned weights from {p_init}: {e}")

    print(f"\n[EVAL] Evaluating baseline parameters on realistic observation suite ({len(scenarios)} scenarios):")
    print(f"       Baseline Gains: {baseline_params.round(3)}...")
    t0 = time.time()
    baseline_cost = evaluate_candidate(baseline_params, scenarios, lambda_image=lambda_image)
    print(f"       Baseline Mean Cost: {baseline_cost:.4f} (evaluated in {time.time()-t0:.2f}s)")

    prev_cost = None
    if prev_tuned_params is not None:
        t0 = time.time()
        prev_cost = evaluate_candidate(prev_tuned_params, scenarios, lambda_image=lambda_image)
        print(f"       Previous Tuned Mean Cost: {prev_cost:.4f} (evaluated in {time.time()-t0:.2f}s)")

    # Construct initial population array
    total_pop = max(len(bounds) * popsize, 30)
    rng = np.random.default_rng(42)
    init_pop = rng.uniform(lower_b, upper_b, size=(total_pop, len(bounds)))
    init_pop[0] = baseline_params
    if prev_tuned_params is not None:
        init_pop[1] = prev_tuned_params
        for idx in range(2, min(16, total_pop)):
            noise_scale = 0.05 * (idx - 1)
            perturbed = prev_tuned_params * (1.0 + rng.normal(0.0, noise_scale, size=len(bounds)))
            init_pop[idx] = np.clip(perturbed, lower_b, upper_b)
    else:
        for idx in range(1, min(10, total_pop)):
            noise_scale = 0.05 * idx
            perturbed = baseline_params * (1.0 + rng.normal(0.0, noise_scale, size=len(bounds)))
            init_pop[idx] = np.clip(perturbed, lower_b, upper_b)

    print(f"\n[OPTIM] Launching Differential Evolution Global Optimization (pop_size={total_pop}, maxiter={maxiter})...")
    t_start = time.time()
    
    eval_counter = [0]
    best_seen = [min(baseline_cost, prev_cost if prev_cost is not None else float("inf"))]

    def callback_fn(xk, convergence=None):
        cost = evaluate_candidate(xk, scenarios, lambda_image=lambda_image)
        if cost < best_seen[0]:
            best_seen[0] = cost
        eval_counter[0] += 1
        print(f"  DE Gen {eval_counter[0]:2d} | Best Cost: {best_seen[0]:.4f} | Current Gen Candidate: {cost:.4f}", flush=True)

    de_res = differential_evolution(
        func=evaluate_candidate,
        bounds=bounds,
        args=(scenarios, lambda_image),
        strategy="best1bin",
        maxiter=maxiter,
        init=init_pop,
        tol=0.005,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        workers=1, # Parallelism handled directly across scenarios via _GLOBAL_POOL
        updating="immediate",
        polish=False,
        callback=callback_fn,
        disp=False
    )

    print(f"\n[OPTIM] Global optimization complete in {time.time()-t_start:.1f}s. Cost: {de_res.fun:.4f}", flush=True)
    print(f"        Intermediate candidate: {de_res.x.round(3)}", flush=True)

    # Local Bounded Refinement (L-BFGS-B with strict box bounds)
    print("\n[OPTIM] Polishing with bounded local search (L-BFGS-B)...", flush=True)
    lower_b = np.array([b[0] for b in bounds])
    upper_b = np.array([b[1] for b in bounds])
    x0_de = np.clip(de_res.x, lower_b, upper_b)

    local_res = minimize(
        fun=evaluate_candidate,
        x0=x0_de,
        args=(scenarios, lambda_image),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 30, "disp": False, "eps": 0.02}
    )

    x_cand = np.clip(local_res.x, lower_b, upper_b)
    cost_cand = float(local_res.fun)
    de_cost = float(de_res.fun)

    # Verification guard: only accept candidate if it strictly improves upon global DE
    if cost_cand < de_cost:
        print(f"[OPTIM] Local polish strictly improved cost: {de_cost:.4f} -> {cost_cand:.4f}", flush=True)
        opt_params = x_cand
        opt_cost = cost_cand
    else:
        print(f"[OPTIM] Local polish did not improve upon DE ({cost_cand:.4f} >= {de_cost:.4f}). Retaining global DE result.", flush=True)
        opt_params = x0_de
        opt_cost = de_cost

    elapsed = time.time() - t_start
    ref_cost = prev_cost if prev_cost is not None else baseline_cost
    pct_gain = (ref_cost - opt_cost) / ref_cost * 100.0

    print(f"[OPTIM] Optimization finished in {elapsed:.1f}s.")
    print(f"        Tuned Parameters:   {opt_params.round(3)}")
    print(f"        Final Tuned Cost:   {opt_cost:.4f} (Prior Best: {ref_cost:.4f}, Improvement: {pct_gain:+.1f}%)")

    tuned_dict = {k: float(opt_params[i]) for i, k in enumerate(param_keys)}
    baseline_dict = {k: float(baseline_params[i]) for i, k in enumerate(param_keys)}

    results = {
        "status": "SUCCESS",
        "optimization_time_s": round(elapsed, 1),
        "lambda_image": lambda_image,
        "num_scenarios": len(scenarios),
        "attitude_dynamics": {
            "tau_att_s": 0.035,
            "max_tilt_rate_deg_s": 250.0
        },
        "baseline_cost": round(baseline_cost, 4),
        "previous_tuned_cost": round(prev_cost, 4) if prev_cost is not None else None,
        "tuned_cost": round(opt_cost, 4),
        "cost_reduction_pct": round(pct_gain, 2),
        "baseline_parameters": baseline_dict,
        "tuned_parameters": tuned_dict,
    }
    if prev_tuned_params is not None:
        results["previous_tuned_parameters"] = {k: float(prev_tuned_params[i]) for i, k in enumerate(param_keys)}

    output_weights_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_weights_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVE] Exported retuned PID gains to: {output_weights_path}")

    if _GLOBAL_POOL is not None:
        _GLOBAL_POOL.close()
        _GLOBAL_POOL.join()
        _GLOBAL_POOL = None

    return results


def run_random_search_optimization(
    scenarios: List[Dict[str, Any]],
    output_weights_path: Path,
    n_global: int = 300,
    n_local: int = 300,
    elite_k: int = 3,
    lambda_image: float = 15.0,
    workers: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Two-Stage Coarse-to-Fine Latin Hypercube Random Search (Bergstra & Bengio, 2012 / Solis & Wets, 1981):
      - Stage 1: Global Latin Hypercube Sampling (LHS) across the 15D parameter envelope.
      - Stage 2: Local exploitation sampling centered around the top elite candidate basins.
    """
    global _GLOBAL_POOL
    if workers > 1:
        ctx = mp.get_context("fork")
        _GLOBAL_POOL = ctx.Pool(workers)
        print(f"[RANDOM-SEARCH] Initialized parallel worker pool with {workers} cores.")

    bounds = DEFAULT_BOUNDS
    lower_b = np.array([b[0] for b in bounds])
    upper_b = np.array([b[1] for b in bounds])
    ranges = upper_b - lower_b
    baseline_params = BASELINE_PARAMS.copy()
    param_keys = PARAM_KEYS

    print(f"\n[EVAL] Evaluating baseline parameters on realistic observation suite ({len(scenarios)} scenarios):")
    t0 = time.time()
    baseline_cost = evaluate_candidate(baseline_params, scenarios, lambda_image=lambda_image)
    print(f"       Baseline Mean Cost: {baseline_cost:.4f} (evaluated in {time.time()-t0:.2f}s)")

    best_cost = baseline_cost
    best_params = baseline_params.copy()
    best_trial_overall = 0
    history = []

    t_start = time.time()

    # -------------------------------------------------------------
    # STAGE 1: Global Latin Hypercube Sampling (LHS)
    # -------------------------------------------------------------
    print(f"\n[STAGE 1/2: GLOBAL LHS] Generating {n_global} stratified samples across 15D parameter space (seed={seed})...")
    lhs_sampler = LatinHypercube(d=len(bounds), seed=seed)
    unit_samples = lhs_sampler.random(n=n_global)
    global_candidates = scale(unit_samples, lower_b, upper_b)

    stage1_results = []
    for i in range(n_global):
        cand = global_candidates[i]
        cost = evaluate_candidate(cand, scenarios, lambda_image=lambda_image)
        stage1_results.append((cost, cand.copy(), i + 1))

        if cost < best_cost:
            improvement = (best_cost - cost) / best_cost * 100.0 if best_cost < 1e5 else 0.0
            best_cost = cost
            best_params = cand.copy()
            best_trial_overall = i + 1
            print(f"  [★ STAGE 1 NEW BEST] Trial {i+1:3d}/{n_global} | Cost: {best_cost:.4f} | Improvement: {improvement:+.1f}% | Lat kp={cand[6]:.2f}, kd={cand[8]:.2f}", flush=True)
        elif (i + 1) % 50 == 0 or (i + 1) == n_global:
            elapsed_now = time.time() - t_start
            rate = (i + 1) / elapsed_now
            print(f"  Stage 1 Trial {i+1:3d}/{n_global} | Best So Far: {best_cost:.4f} | Rate: {rate:.2f} trials/s", flush=True)

        history.append({
            "stage": 1,
            "trial": i + 1,
            "cost": float(cost),
            "best_cost_so_far": float(best_cost)
        })

    stage1_elapsed = time.time() - t_start
    stage1_best_cost = best_cost
    print(f"[STAGE 1 COMPLETE] {n_global} LHS trials in {stage1_elapsed:.1f}s. Stage 1 Best Cost: {stage1_best_cost:.4f}")

    # Extract top elite candidates from Stage 1
    stage1_results.sort(key=lambda x: x[0])
    elite_pool = [x[1] for x in stage1_results[:elite_k]]
    print(f"[ELITE POOL] Selected top {elite_k} candidate basins for Stage 2 refinement (Costs: {[round(x[0], 2) for x in stage1_results[:elite_k]]}).")

    # -------------------------------------------------------------
    # STAGE 2: Local Exploitation (Random Perturbations around Elites)
    # -------------------------------------------------------------
    print(f"\n[STAGE 2/2: LOCAL EXPLOITATION] Generating {n_local} localized perturbations around top {elite_k} elite basins...")
    rng_local = np.random.default_rng(seed + 100)
    stage2_t0 = time.time()

    for j in range(n_local):
        overall_idx = n_global + j + 1
        # Pick an elite candidate to perturb (weighted toward rank 1)
        elite_idx = int(rng_local.choice(len(elite_pool), p=[0.55, 0.30, 0.15] if len(elite_pool) >= 3 else None))
        elite_center = elite_pool[elite_idx]

        # Multi-scale perturbation: 50% fine (10% range) and 50% medium (20% range)
        scale_factor = 0.10 if (j % 2 == 0) else 0.20
        delta = rng_local.normal(0.0, scale_factor * ranges, size=len(bounds))
        cand = np.clip(elite_center + delta, lower_b, upper_b)

        cost = evaluate_candidate(cand, scenarios, lambda_image=lambda_image)

        if cost < best_cost:
            improvement = (best_cost - cost) / best_cost * 100.0 if best_cost < 1e5 else 0.0
            best_cost = cost
            best_params = cand.copy()
            best_trial_overall = overall_idx
            print(f"  [★ STAGE 2 NEW BEST] Trial {j+1:3d}/{n_local} (Overall #{overall_idx}) | Cost: {best_cost:.4f} | Improvement: {improvement:+.1f}% | Lat kp={cand[6]:.2f}, kd={cand[8]:.2f}", flush=True)
        elif (j + 1) % 50 == 0 or (j + 1) == n_local:
            elapsed_now = time.time() - stage2_t0
            rate = (j + 1) / elapsed_now
            print(f"  Stage 2 Trial {j+1:3d}/{n_local} | Best So Far: {best_cost:.4f} | Rate: {rate:.2f} trials/s", flush=True)

        history.append({
            "stage": 2,
            "trial": overall_idx,
            "cost": float(cost),
            "best_cost_so_far": float(best_cost)
        })

    total_elapsed = time.time() - t_start
    total_trials = n_global + n_local
    pct_gain = (baseline_cost - best_cost) / baseline_cost * 100.0

    print(f"\n[RANDOM-SEARCH COMPLETE] {total_trials} total trials in {total_elapsed:.1f}s ({total_elapsed/total_trials:.2f}s/trial).")
    print(f"                         Stage 1 Best: {stage1_best_cost:.4f} ──► Stage 2 Final Best: {best_cost:.4f}")
    print(f"                         Baseline Cost: {baseline_cost:.4f} | Total Improvement: {pct_gain:+.1f}%")
    print(f"                         Winning Trial: #{best_trial_overall}")
    print(f"                         Best Parameters: {best_params.round(3)}")

    tuned_dict = {k: float(best_params[i]) for i, k in enumerate(param_keys)}
    baseline_dict = {k: float(baseline_params[i]) for i, k in enumerate(param_keys)}

    results = {
        "status": "SUCCESS",
        "method": "TWO_STAGE_LATIN_HYPERCUBE_RANDOM_SEARCH",
        "seed": seed,
        "n_global": n_global,
        "n_local": n_local,
        "total_trials": total_trials,
        "best_trial_overall": best_trial_overall,
        "stage1_best_cost": round(stage1_best_cost, 4),
        "final_tuned_cost": round(best_cost, 4),
        "optimization_time_s": round(total_elapsed, 1),
        "lambda_image": lambda_image,
        "num_scenarios": len(scenarios),
        "attitude_dynamics": {
            "tau_att_s": 0.035,
            "max_tilt_rate_deg_s": 250.0
        },
        "baseline_cost": round(baseline_cost, 4),
        "cost_reduction_pct": round(pct_gain, 2),
        "baseline_parameters": baseline_dict,
        "tuned_parameters": tuned_dict,
        "history": history
    }

    output_weights_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_weights_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVE] Exported Two-Stage Random Search PID gains to: {output_weights_path}")

    if _GLOBAL_POOL is not None:
        _GLOBAL_POOL.close()
        _GLOBAL_POOL.join()
        _GLOBAL_POOL = None

    return results


def validate_on_held_out(
    val_suite_path: Path,
    candidates: Union[Dict[str, np.ndarray], np.ndarray],
    tuned_params: Optional[np.ndarray] = None,
    prev_params: Optional[np.ndarray] = None,
    lambda_image: float = 15.0
):
    """Directly compares candidate controllers on the independent 30-scenario validation suite."""
    if not val_suite_path.exists():
        print(f"[WARN] Validation suite not found at {val_suite_path}")
        return

    print(f"\n[EVAL-VALIDATION] Testing on held-out test suite: {val_suite_path.name} (30 scenarios)...")
    with open(val_suite_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    scenarios = val_data["scenarios"]

    # Adapt backward-compatible positional call if dictionary was not passed directly
    if isinstance(candidates, dict):
        cand_dict = candidates
    else:
        cand_dict = {"BASELINE PID": candidates}
        if tuned_params is not None:
            cand_dict["RETUNED PID"] = tuned_params
        if prev_params is not None:
            cand_dict["PREV TUNED"] = prev_params

    results_map = {}
    for name, params in cand_dict.items():
        if params is None:
            continue
        res_list = [simulate_scenario(s, params, lambda_image=lambda_image) for s in scenarios]
        dwell_s = [r["max_dwell_s"] for r in res_list]
        results_map[name] = {
            "fov": float(np.mean([r["in_view_pct"] for r in res_list])),
            "standoff_rmse": float(np.mean([np.sqrt(r["mean_standoff_err_m2"]) for r in res_list])),
            "image_err": float(np.mean([r["mean_image_err"] for r in res_list])),
            "dwell": float(np.mean(dwell_s)),
            "catch_1s_pct": float(np.mean([d >= 1.0 for d in dwell_s]) * 100.0),
            "dwell_3s_pct": float(np.mean([d >= 3.0 for d in dwell_s]) * 100.0),
            "final_dist": float(np.mean([r["final_dist"] for r in res_list]))
        }

    names = list(results_map.keys())
    col_w = max(20, max(len(n) for n in names) + 2)
    header = f"{'METRIC':<30} | " + " | ".join(f"{n:^{col_w}}" for n in names)
    sep = "=" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    metrics = [
        ("Camera In-View (% FOV)", "fov", "{:16.1f}%"),
        ("1.0s Basket Catch Rate (%)", "catch_1s_pct", "{:16.1f}%"),
        ("3.0s Continuous Dwell Rate", "dwell_3s_pct", "{:16.1f}%"),
        ("Mean Max Catch Dwell (s)", "dwell", "{:16.2f}s"),
        ("RMSE Standoff Dist (m)", "standoff_rmse", "{:16.2f}m"),
        ("Mean Image Plane Error (e^2)", "image_err", "{:16.4f} "),
        ("Mean Final Distance (m)", "final_dist", "{:16.2f}m")
    ]

    for label, key, fmt in metrics:
        row = f"{label:<30} | " + " | ".join(f"{fmt.format(results_map[n][key]):^{col_w}}" for n in names)
        print(row)
    print(sep + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune Kinematic Controller Under Realistic Observations & Attitude Dynamics")
    parser.add_argument("--method", type=str, default="de", choices=["de", "random"], help="Optimization algorithm: 'de' (Differential Evolution) or 'random' (Two-Stage Latin Hypercube Random Search)")
    parser.add_argument("--suite", type=str, default="outputs/benchmarks/datasets/pid_realistic_tuning_suite_80.json")
    parser.add_argument("--val-suite", type=str, default="outputs/benchmarks/datasets/validation_suite_30.json")
    parser.add_argument("--out", type=str, default=None, help="Output weights JSON path (defaults based on method)")
    parser.add_argument("--init-weights", type=str, default="control/weights/tuned_pid_gains_pre_attitude_dynamics.json")
    parser.add_argument("--lambda-image", type=float, default=15.0, help="Weight for image plane error vs standoff error")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel CPU workers")
    parser.add_argument("--popsize", type=int, default=2, help="Differential evolution population size factor (e.g. 2 -> 30 candidates)")
    parser.add_argument("--maxiter", type=int, default=8, help="Maximum differential evolution generations")
    parser.add_argument("--n-global", type=int, default=300, help="Stage 1 Latin Hypercube global trials (default: 300)")
    parser.add_argument("--n-local", type=int, default=300, help="Stage 2 localized perturbation trials (default: 300)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    suite_scenarios = create_tuning_suite(Path(args.suite))

    if args.method == "random":
        out_path = Path(args.out) if args.out else Path("control/weights/tuned_pid_gains_random_search.json")
        res = run_random_search_optimization(
            scenarios=suite_scenarios,
            output_weights_path=out_path,
            n_global=args.n_global,
            n_local=args.n_local,
            lambda_image=args.lambda_image,
            workers=args.workers,
            seed=args.seed
        )
    else:
        out_path = Path(args.out) if args.out else Path("control/weights/tuned_pid_gains.json")
        res = run_tuning_optimization(
            scenarios=suite_scenarios,
            output_weights_path=out_path,
            init_weights_path=Path(args.init_weights) if args.init_weights else None,
            lambda_image=args.lambda_image,
            workers=args.workers,
            popsize=args.popsize,
            maxiter=args.maxiter
        )

    # Prepare candidate vectors for validation comparison
    baseline_vec = np.array([res["baseline_parameters"][k] for k in PARAM_KEYS])
    tuned_vec = np.array([res["tuned_parameters"][k] for k in PARAM_KEYS])

    comparison_dict = {
        "BASELINE PID": baseline_vec,
    }

    if args.method == "random":
        comparison_dict["2-STAGE RANDOM"] = tuned_vec
        # Also load DE-tuned weights if available for side-by-side comparison
        de_weights_path = Path("control/weights/tuned_pid_gains.json")
        if de_weights_path.exists():
            try:
                with open(de_weights_path, "r", encoding="utf-8") as f:
                    de_data = json.load(f)
                de_params = de_data.get("tuned_parameters", de_data)
                comparison_dict["DE RETUNED"] = np.array([float(de_params[k]) for k in PARAM_KEYS])
            except Exception as e:
                print(f"[WARN] Could not load DE weights for comparison: {e}")
    else:
        comparison_dict["DE RETUNED"] = tuned_vec
        if "previous_tuned_parameters" in res:
            comparison_dict["PREV TUNED"] = np.array([res["previous_tuned_parameters"][k] for k in PARAM_KEYS])

    validate_on_held_out(
        val_suite_path=Path(args.val_suite),
        candidates=comparison_dict,
        lambda_image=args.lambda_image
    )
