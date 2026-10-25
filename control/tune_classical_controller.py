#!/usr/bin/env python3
"""
Automated Parameter Tuning for Classical Kinematic Visual Servoing Controller.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception.

Optimizes 7 key kinematic visual servoing parameters against the user-specified dual objective:
    1. Distance to desired standoff: (d - 3.5m)^2
    2. Image plane tracking error: (e_x^2 + e_y^2) with smooth unclipped off-screen projection
    3. Cornering deceleration & speed retention floor: min_centering_factor + centering_damping
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.optimize import differential_evolution, minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.simulation import FastPixhawkQuadSim
from control.pid_controller import KinematicVisualServoController
from control.trajectory import StochasticTargetTrajectory
from control.dataset_generator import PursuitDatasetGenerator, EncounterType, PursuitScenario, PursuitDataset


def create_tuning_suite(
    suite_path: Path,
    base_seed: int = 60_000,
    duration: float = 18.0
) -> List[Dict[str, Any]]:
    """
    Creates a dedicated, independent 36-scenario tuning dataset with balanced distribution
    across Cruising, Evasive, and Hyper-Evasive regimes and all 4 encounter geometries.
    """
    if suite_path.exists():
        print(f"[TUNING] Loading existing tuning suite from: {suite_path}")
        with open(suite_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["scenarios"]

    print(f"[TUNING] Generating balanced 36-scenario tuning dataset (base_seed={base_seed})...")
    gen = PursuitDatasetGenerator()
    scenarios = []
    
    # 12 Cruising, 12 Evasive, 12 Hyper-Evasive across 4 encounter types
    distribution = [
        ("cruising", EncounterType.TAIL_CHASE, 3),
        ("cruising", EncounterType.CROSSING_RIGHT, 3),
        ("cruising", EncounterType.CROSSING_LEFT, 3),
        ("cruising", EncounterType.OBLIQUE_AWAY, 3),
        ("evasive", EncounterType.TAIL_CHASE, 3),
        ("evasive", EncounterType.CROSSING_RIGHT, 3),
        ("evasive", EncounterType.CROSSING_LEFT, 3),
        ("evasive", EncounterType.OBLIQUE_AWAY, 3),
        ("hyper_evasive", EncounterType.TAIL_CHASE, 3),
        ("hyper_evasive", EncounterType.CROSSING_RIGHT, 3),
        ("hyper_evasive", EncounterType.CROSSING_LEFT, 3),
        ("hyper_evasive", EncounterType.OBLIQUE_AWAY, 3),
    ]

    scen_idx = 1
    for prof, enc, count in distribution:
        for c in range(count):
            seed = base_seed + scen_idx * 31
            scen_id = f"tune_{scen_idx:03d}_{prof}_{enc.value}_s{seed}"
            sc = gen.generate_scenario(
                scenario_id=scen_id,
                profile=prof,
                encounter_type=enc,
                seed=seed,
                duration=duration
            )
            scenarios.append(sc.to_dict())
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
    lambda_image: float = 15.0
) -> Dict[str, float]:
    """
    Simulates a flight scenario under given candidate PID parameters and computes
    the composite cost: Standoff Error + lambda * Image Plane Tracking Error.
    """
    kp_fwd, centering_damp, min_centering, kp_lat, kd_lat, kp_yaw, kd_yaw = params

    c_init = scenario.get("initial_chaser_pos", [0.0, 0.0, -2.5])
    c_yaw = float(scenario.get("initial_chaser_yaw", 0.0))
    c_vel = np.array([np.cos(c_yaw) * 4.0, np.sin(c_yaw) * 4.0, 0.0], dtype=np.float64)

    # 1. Simulator with identical 15 m/s physical flight envelope
    sim = FastPixhawkQuadSim(
        dt=dt,
        max_vel_xy=15.0,
        max_accel_xy=6.0,
        max_vel_up=3.0,
        max_vel_down=2.0
    )
    sim.reset(initial_pos=np.array(c_init, dtype=np.float64), initial_yaw=c_yaw, initial_vel=c_vel)

    # 2. Candidate Controller (Normal PID across all 4 axes with zero integral terms)
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
        ki_forward=0.0,
        kd_forward=0.15,
        centering_damping=float(centering_damp),
        min_centering_factor=float(min_centering),
        kp_lat=float(kp_lat),
        ki_lat=0.0,
        kd_lat=float(kd_lat),
        kp_z=6.0,
        ki_z=0.0,
        kd_z=1.0,
        kp_yaw=float(kp_yaw),
        ki_yaw=0.0,
        kd_yaw=float(kd_yaw),
        max_lat_vel=4.0,
        max_lat_accel=6.0,
        max_accel=6.0,
        max_decel=6.0,
        max_accel_yaw=240.0,
        min_limits=np.array([-0.5, -4.0, -3.0, -120.0]),
        max_limits=np.array([15.0,  4.0,  2.0,  120.0]),
    )

    # 3. Target Trajectory
    target = StochasticTargetTrajectory(
        duration=scenario["duration"] + 5.0,
        profile=scenario["profile"],
        initial_pos=scenario["initial_target_pos"],
        initial_heading_rad=scenario["initial_target_heading_rad"],
        seed=int(scenario["seed"])
    )

    total_steps = int(round(scenario["duration"] / dt))
    
    standoff_costs = []
    image_costs = []
    in_views = []
    dists = []
    dwell_counter = 0.0
    max_dwell = 0.0
    collision = False
    ground_strike = False

    for step in range(total_steps):
        t = step * dt
        tgt_pos = target.get_position(t)

        # Telemetry with standard projection for controller input
        telem = sim.get_camera_telemetry(
            tgt_pos,
            hfov_deg=62.0,
            vfov_deg=38.0,
            target_w_m=0.35,
            target_h_m=0.20,
            img_w=1280,
            img_h=720,
            desired_target_size=110.0
        )
        
        # Ground-truth metric distance
        dist = float(telem["distance"])
        dists.append(dist)
        
        # Standoff error: (d - 3.5)^2
        err_standoff = (dist - 3.5) ** 2
        standoff_costs.append(err_standoff)

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

        # Image plane tracking error: e_x^2 + e_y^2
        # Smooth unclipped error penalizes losing sightline proportionally to angle
        err_image = float(u_ex ** 2 + u_ey ** 2)
        image_costs.append(err_image)

        # Continuous basket dwell tracking
        if in_view and (2.5 <= dist <= 4.5):
            dwell_counter += dt
            max_dwell = max(max_dwell, dwell_counter)
        else:
            dwell_counter = max(0.0, dwell_counter - 0.5 * dt)

        # Safety checks
        if dist < 1.2:
            collision = True
        if -sim.pos[2] < 0.8:
            ground_strike = True

        # Compute command and integrate quadrotor dynamics
        cmd = ctrl.compute_cmd(telem, drone_pitch=sim.pitch, drone_roll=sim.roll, dt=dt)
        sim.step(cmd)

    mean_standoff = float(np.mean(standoff_costs))
    mean_image = float(np.mean(image_costs))
    
    # Composite objective
    base_cost = mean_standoff + lambda_image * mean_image

    # Safety penalties
    safety_penalty = 0.0
    if collision:
        safety_penalty += 150.0
    if ground_strike:
        safety_penalty += 100.0

    total_cost = base_cost + safety_penalty

    return {
        "cost": total_cost,
        "mean_standoff_err_m2": mean_standoff,
        "mean_image_err": mean_image,
        "in_view_pct": float(np.mean(in_views) * 100.0),
        "min_dist": float(min(dists)),
        "final_dist": float(dists[-1]),
        "max_dwell_s": float(max_dwell),
        "collision": collision,
        "ground_strike": ground_strike
    }


def evaluate_candidate(params: np.ndarray, scenarios: List[Dict[str, Any]], lambda_image: float = 15.0) -> float:
    """Evaluates candidate parameters across the tuning scenarios and returns mean cost."""
    costs = []
    for sc in scenarios:
        res = simulate_scenario(sc, params, lambda_image=lambda_image)
        costs.append(res["cost"])
    return float(np.mean(costs))


def run_tuning_optimization(
    scenarios: List[Dict[str, Any]],
    output_weights_path: Path,
    lambda_image: float = 15.0,
    workers: int = 8
) -> Dict[str, Any]:
    """Runs global differential evolution followed by Nelder-Mead polishing."""
    
    # 7 Tunable Parameters: [kp_fwd, centering_damp, min_centering_factor, kp_lat, kd_lat, kp_yaw, kd_yaw]
    bounds = [
        (0.6, 2.2),     # kp_fwd (default: 1.2)
        (0.15, 1.20),   # centering_damp (cornering deceleration factor per unit |e_x|)
        (0.20, 0.65),   # min_centering_factor (speed retention floor during sharp turns)
        (1.0, 4.5),     # kp_lat (default: 2.5)
        (0.1, 1.2),     # kd_lat (default: 0.35)
        (120.0, 320.0), # kp_yaw (default: 180.0)
        (15.0, 65.0),   # kd_yaw (default: 20.0)
    ]

    baseline_params = np.array([1.2, 0.25, 0.35, 2.5, 0.35, 180.0, 20.0], dtype=np.float64)
    print(f"\n[EVAL] Evaluating baseline parameters on tuning suite ({len(scenarios)} scenarios):")
    print(f"       Gains: {baseline_params.round(3)}...")
    t0 = time.time()
    baseline_cost = evaluate_candidate(baseline_params, scenarios, lambda_image=lambda_image)
    print(f"       Baseline Mean Cost: {baseline_cost:.4f} (evaluated in {time.time()-t0:.2f}s)")

    # Evaluate previous uncalibrated tuned gains (with min_centering_factor=0.85) for reference
    prev_uncalibrated = np.array([0.863, 0.186, 0.850, 1.865, 0.736, 222.97, 47.84], dtype=np.float64)
    prev_cost = evaluate_candidate(prev_uncalibrated, scenarios, lambda_image=lambda_image)
    print(f"       Prior Tuned Gains (min_centering=0.85): Cost = {prev_cost:.4f}")

    print(f"\n[OPTIM] Launching Differential Evolution Global Optimization (population=6, maxiter=12, workers={workers})...")
    t_start = time.time()
    
    eval_counter = [0]
    best_seen = [min(baseline_cost, prev_cost)]

    def callback_fn(xk, convergence=None):
        cost = evaluate_candidate(xk, scenarios, lambda_image=lambda_image)
        if cost < best_seen[0]:
            best_seen[0] = cost
        eval_counter[0] += 1
        print(f"  DE Gen {eval_counter[0]:2d} | Best Cost: {best_seen[0]:.4f} | Current Gen Best: {cost:.4f}", flush=True)

    de_res = differential_evolution(
        func=evaluate_candidate,
        bounds=bounds,
        args=(scenarios, lambda_image),
        strategy="best1bin",
        maxiter=12,
        popsize=6,
        tol=0.01,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        workers=workers,
        updating="deferred",
        polish=False,
        callback=callback_fn,
        disp=False
    )

    print(f"\n[OPTIM] Global optimization complete in {time.time()-t_start:.1f}s. Cost: {de_res.fun:.4f}", flush=True)
    print(f"        Intermediate candidate: {de_res.x.round(3)}", flush=True)

    # Local Simplex Polish
    print("\n[OPTIM] Polishing with Nelder-Mead simplex search...", flush=True)
    nm_res = minimize(
        fun=evaluate_candidate,
        x0=de_res.x,
        args=(scenarios, lambda_image),
        method="Nelder-Mead",
        options={"maxiter": 40, "disp": False}
    )

    opt_params = nm_res.x
    opt_cost = nm_res.fun
    elapsed = time.time() - t_start

    ref_cost = min(baseline_cost, prev_cost)
    pct_gain = (ref_cost - opt_cost) / ref_cost * 100.0

    print(f"[OPTIM] Optimization finished in {elapsed:.1f}s.")
    print(f"        Tuned Parameters:   {opt_params.round(3)}")
    print(f"        Final Tuned Cost:   {opt_cost:.4f} (Baseline: {baseline_cost:.4f}, Prev Tuned: {prev_cost:.4f}, Improvement: {pct_gain:+.1f}%)")

    tuned_dict = {
        "kp_forward": float(opt_params[0]),
        "ki_forward": 0.0,
        "kd_forward": 0.15,
        "centering_damping": float(opt_params[1]),
        "min_centering_factor": float(opt_params[2]),
        "kp_lat": float(opt_params[3]),
        "ki_lat": 0.0,
        "kd_lat": float(opt_params[4]),
        "kp_z": 6.0,
        "ki_z": 0.0,
        "kd_z": 1.0,
        "kp_yaw": float(opt_params[5]),
        "ki_yaw": 0.0,
        "kd_yaw": float(opt_params[6]),
    }
    baseline_dict = {
        "kp_forward": float(baseline_params[0]),
        "ki_forward": 0.0,
        "kd_forward": 0.15,
        "centering_damping": float(baseline_params[1]),
        "min_centering_factor": float(baseline_params[2]),
        "kp_lat": float(baseline_params[3]),
        "ki_lat": 0.0,
        "kd_lat": float(baseline_params[4]),
        "kp_z": 6.0,
        "ki_z": 0.0,
        "kd_z": 1.0,
        "kp_yaw": float(baseline_params[5]),
        "ki_yaw": 0.0,
        "kd_yaw": float(baseline_params[6]),
    }

    results = {
        "status": "SUCCESS",
        "optimization_time_s": round(elapsed, 1),
        "lambda_image": lambda_image,
        "num_scenarios": len(scenarios),
        "baseline_cost": round(baseline_cost, 4),
        "previous_tuned_cost": round(prev_cost, 4),
        "tuned_cost": round(opt_cost, 4),
        "cost_reduction_pct": round(pct_gain, 2),
        "baseline_parameters": baseline_dict,
        "tuned_parameters": tuned_dict,
    }

    output_weights_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_weights_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVE] Exported tuned PID gains to: {output_weights_path}")

    return results


def validate_on_held_out(
    val_suite_path: Path,
    baseline_params: np.ndarray,
    tuned_params: np.ndarray,
    lambda_image: float = 15.0
):
    """Directly compares baseline vs tuned PID on the independent 30-scenario validation suite."""
    if not val_suite_path.exists():
        print(f"[WARN] Validation suite not found at {val_suite_path}")
        return

    print(f"\n[EVAL-VALIDATION] Testing on held-out test suite: {val_suite_path.name} (30 scenarios)...")
    with open(val_suite_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    scenarios = val_data["scenarios"]

    b_res = [simulate_scenario(s, baseline_params, lambda_image=lambda_image) for s in scenarios]
    t_res = [simulate_scenario(s, tuned_params, lambda_image=lambda_image) for s in scenarios]

    print("\n" + "="*80)
    print(f"{'METRIC':<30} | {'BASELINE PID':<18} | {'TUNED PID':<18} | {'DELTA':<10}")
    print("="*80)
    
    b_fov = np.mean([r["in_view_pct"] for r in b_res])
    t_fov = np.mean([r["in_view_pct"] for r in t_res])
    print(f"{'Camera In-View (% FOV)':<30} | {b_fov:15.1f}% | {t_fov:15.1f}% | {t_fov - b_fov:+5.1f}%")

    b_standoff = np.mean([np.sqrt(r["mean_standoff_err_m2"]) for r in b_res])
    t_standoff = np.mean([np.sqrt(r["mean_standoff_err_m2"]) for r in t_res])
    print(f"{'RMSE Standoff Dist (m)':<30} | {b_standoff:15.2f}m | {t_standoff:15.2f}m | {t_standoff - b_standoff:+5.2f}m")

    b_image = np.mean([r["mean_image_err"] for r in b_res])
    t_image = np.mean([r["mean_image_err"] for r in t_res])
    print(f"{'Mean Image Plane Error (e^2)':<30} | {b_image:15.4f}  | {t_image:15.4f}  | {t_image - b_image:+5.4f}")

    b_dwell = np.mean([r["max_dwell_s"] for r in b_res])
    t_dwell = np.mean([r["max_dwell_s"] for r in t_res])
    print(f"{'Mean Max Catch Dwell (s)':<30} | {b_dwell:15.2f}s | {t_dwell:15.2f}s | {t_dwell - b_dwell:+5.2f}s")

    b_final = np.mean([r["final_dist"] for r in b_res])
    t_final = np.mean([r["final_dist"] for r in t_res])
    print(f"{'Mean Final Distance (m)':<30} | {b_final:15.2f}m | {t_final:15.2f}m | {t_final - b_final:+5.2f}m")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune Kinematic Controller for Standoff and Image Tracking")
    parser.add_argument("--suite", type=str, default="outputs/benchmarks/datasets/pid_tuning_suite_36.json")
    parser.add_argument("--val-suite", type=str, default="outputs/benchmarks/datasets/validation_suite_30.json")
    parser.add_argument("--out", type=str, default="control/weights/tuned_pid_gains.json")
    parser.add_argument("--lambda-image", type=float, default=15.0, help="Weight for image plane error vs standoff error")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel CPU workers")
    args = parser.parse_args()

    suite_scenarios = create_tuning_suite(Path(args.suite))
    res = run_tuning_optimization(
        scenarios=suite_scenarios,
        output_weights_path=Path(args.out),
        lambda_image=args.lambda_image,
        workers=args.workers
    )

    param_keys = ["kp_forward", "centering_damping", "min_centering_factor", "kp_lat", "kd_lat", "kp_yaw", "kd_yaw"]
    baseline_vec = np.array([res["baseline_parameters"][k] for k in param_keys])
    tuned_vec = np.array([res["tuned_parameters"][k] for k in param_keys])

    validate_on_held_out(
        val_suite_path=Path(args.val_suite),
        baseline_params=baseline_vec,
        tuned_params=tuned_vec,
        lambda_image=args.lambda_image
    )
