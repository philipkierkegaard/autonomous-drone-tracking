#!/usr/bin/env python3
"""
Autonomous Drone Pursuit Benchmarking Suite.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Evaluates controller performance across:
1. Target Flight Profiles ('cruising', 'evasive', 'hyper_evasive')
2. Frame Drop Rates (0%, 5%, 10%, 20%)
3. Perception Filter Modes (8D Kalman Filter vs. Raw Zero-Order Hold)
4. Comprehensive Aerospace Guidance Metrics:
   - Time-to-Engagement Lock (sustained 5-7m standoff for >= 1.5s in FOV)
   - Interception / Engagement Success Rate (%)
   - FOV Lock Retention Rate (%)
   - Cumulative Firing Window Duration (s and %)
   - Safety Margin / Collision Breach Rate (min distance < 2.5m)
   - Flight Path Length (meters) & Control Effort (smoothness)
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from control.simulation import FastPixhawkQuadSim
from control.pid_controller import KinematicVisualServoController
from control.trajectory import StochasticTargetTrajectory, PROFILES
from perception.pipeline import KalmanBoxTracker
from control.dataset_generator import PursuitScenario, PursuitDataset


def evaluate_single_run(
    profile: str = "evasive",
    seed: int = 42,
    use_kalman: bool = True,
    frame_drop_rate: float = 0.0,
    duration: float = 25.0,
    dt: float = 0.02,
    standoff_min: float = 5.0,
    standoff_max: float = 7.0,
    dwell_time_required: float = 1.5,
    max_lost_frames: int = 15,
    img_w: int = 640,
    img_h: int = 480,
    max_decel: float = 2.8,
    scenario: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Executes a single closed-loop pursuit simulation run and computes performance metrics.
    Can run either a parameterized scenario from PursuitDataset or standalone parameters.
    """
    if scenario is not None:
        target_oracle = scenario.create_target_oracle()
        sim = FastPixhawkQuadSim(dt=dt)
        sim.reset(initial_pos=scenario.initial_chaser_pos, initial_yaw=scenario.initial_chaser_yaw)
        profile = scenario.profile
        seed = scenario.seed
        duration = scenario.duration
        encounter_type = scenario.encounter_type
        scenario_id = scenario.scenario_id
    else:
        target_oracle = StochasticTargetTrajectory(
            duration=duration + 5.0,
            profile=profile,
            initial_pos=np.array([12.0, 0.0, -2.0], dtype=np.float64),
            initial_heading_rad=0.0,
            seed=seed
        )
        sim = FastPixhawkQuadSim(dt=dt)
        sim.reset(initial_pos=[0.0, 0.0, -2.0], initial_yaw=0.0)
        encounter_type = "tail_chase"
        scenario_id = f"{profile}_{seed}"

    rng = np.random.default_rng(seed)
    total_steps = int(round(duration / dt))

    # 1. Initialize Drone Controller
    controller = KinematicVisualServoController(
        camera_uptilt_deg=15.0,
        hfov_deg=60.0,
        vfov_deg=45.0,
        desired_bbox_size=35.0,
        max_decel=max_decel,
    )

    # 2. Perception Tracker Setup (8D Kalman Filter vs Raw)
    tracker: Optional[KalmanBoxTracker] = None
    last_valid_raw_telem: Optional[Dict[str, Any]] = None
    raw_time_since_valid = 0

    # 3. Telemetry & Metric Accumulators
    continuous_lock_dwell = 0.0
    time_to_first_lock: Optional[float] = None
    lock_achieved = False

    in_view_count = 0
    cumulative_firing_time = 0.0
    min_dist = float("inf")
    total_path_length = 0.0
    control_effort = 0.0
    prev_chaser_pos = sim.pos.copy()

    for step in range(total_steps):
        t = step * dt
        target_world = target_oracle.get_position(t)

        # Ground-truth camera projection
        true_telem = sim.get_camera_telemetry(
            target_world,
            hfov_deg=60.0,
            vfov_deg=45.0,
            target_w_m=0.35,
            target_h_m=0.20,
            img_w=img_w,
            img_h=img_h,
            desired_target_size=35.0
        )

        # Simulate camera frame drop / detector miss
        is_dropped = (rng.uniform(0.0, 1.0) < frame_drop_rate)
        measured_bbox = true_telem["bbox"] if (true_telem["in_view"] and not is_dropped) else None

        # -------------------------------------------------------------
        # Perception Processing (Kalman Filter vs. Raw Zero-Order Hold)
        # -------------------------------------------------------------
        if use_kalman:
            if tracker is not None:
                tracker.predict()

            if measured_bbox is not None:
                if tracker is None:
                    tracker = KalmanBoxTracker(np.array(measured_bbox, dtype=np.float32), dt=dt)
                else:
                    tracker.update(np.array(measured_bbox, dtype=np.float32))

            if tracker is not None and tracker.time_since_update <= max_lost_frames:
                pred_box = tracker.get_state()
                tcx = (pred_box[0] + pred_box[2]) / 2.0
                tcy = (pred_box[1] + pred_box[3]) / 2.0
                bw = max(1.0, pred_box[2] - pred_box[0])
                bh = max(1.0, pred_box[3] - pred_box[1])
                curr_sz = max(bw, bh)

                err_x = (tcx - img_w / 2.0) / (img_w / 2.0)
                err_y = (tcy - img_h / 2.0) / (img_h / 2.0)
                range_err = (35.0 - curr_sz) / 35.0

                ctrl_telem = {
                    "status": "COASTING" if tracker.time_since_update > 0 else "LOCKED",
                    "error_x": float(np.clip(err_x, -1.0, 1.0)),
                    "error_y": float(np.clip(err_y, -1.0, 1.0)),
                    "error_range": float(np.clip(range_err, -1.0, 1.0)),
                    "target_size": float(curr_sz),
                    "desired_size": 35.0,
                    "distance": true_telem["distance"]
                }
            else:
                ctrl_telem = {
                    "status": "SEARCHING",
                    "error_x": 0.0,
                    "error_y": 0.0,
                    "error_range": 0.0,
                    "target_size": 0.0,
                    "desired_size": 35.0,
                    "distance": true_telem["distance"]
                }
        else:
            # Raw Mode (Zero-Order Hold / Stale lock on drops)
            if measured_bbox is not None:
                last_valid_raw_telem = true_telem.copy()
                raw_time_since_valid = 0
            else:
                raw_time_since_valid += 1

            if last_valid_raw_telem is not None and raw_time_since_valid <= max_lost_frames:
                ctrl_telem = last_valid_raw_telem.copy()
                ctrl_telem["status"] = "COASTING" if raw_time_since_valid > 0 else "LOCKED"
            else:
                ctrl_telem = {
                    "status": "SEARCHING",
                    "error_x": 0.0,
                    "error_y": 0.0,
                    "error_range": 0.0,
                    "target_size": 0.0,
                    "desired_size": 35.0,
                    "distance": true_telem["distance"]
                }

        # -------------------------------------------------------------
        # Flight Control Step
        # -------------------------------------------------------------
        cmd = controller.compute_cmd(ctrl_telem, drone_pitch=sim.pitch, dt=dt)
        sim.step(cmd)

        # -------------------------------------------------------------
        # Metrics Evaluation
        # -------------------------------------------------------------
        dist = float(np.linalg.norm(target_world - sim.pos))
        min_dist = min(min_dist, dist)

        step_dist = float(np.linalg.norm(sim.pos - prev_chaser_pos))
        total_path_length += step_dist
        prev_chaser_pos = sim.pos.copy()

        control_effort += float(np.sum(cmd ** 2) * dt)

        in_view_gt = bool(true_telem["in_view"])
        if in_view_gt:
            in_view_count += 1

        # Check engagement envelope condition: 5.0m <= distance <= 7.0m AND visible in FOV
        in_envelope = (standoff_min <= dist <= standoff_max) and in_view_gt
        if in_envelope:
            cumulative_firing_time += dt
            continuous_lock_dwell += dt
            if continuous_lock_dwell >= dwell_time_required and time_to_first_lock is None:
                time_to_first_lock = t - dwell_time_required
                lock_achieved = True
        else:
            continuous_lock_dwell = 0.0

    return {
        "scenario_id": scenario_id,
        "encounter_type": encounter_type,
        "profile": profile,
        "seed": seed,
        "use_kalman": use_kalman,
        "frame_drop_rate": frame_drop_rate,
        "success": lock_achieved,
        "time_to_lock": time_to_first_lock,
        "fov_lock_pct": (in_view_count / total_steps) * 100.0,
        "firing_window_s": cumulative_firing_time,
        "firing_window_pct": (cumulative_firing_time / duration) * 100.0,
        "min_distance": min_dist,
        "collision_breach": bool(min_dist < 2.5),
        "path_length": total_path_length,
        "control_effort": control_effort
    }


def run_benchmark_matrix(
    profiles: List[str] = ["cruising", "evasive", "hyper_evasive"],
    drop_rates: List[float] = [0.0, 0.05, 0.10, 0.20],
    num_seeds: int = 20,
    test_ablation: bool = True,
    duration: float = 25.0,
    max_decel: float = 2.8,
    output_dir: Path = PROJECT_ROOT / "outputs" / "benchmarks" / "control"
) -> Dict[str, Any]:
    """
    Executes the full Monte Carlo benchmarking matrix and exports structured reports & plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    kalman_modes = [True, False] if test_ablation else [True]
    total_runs = len(profiles) * len(drop_rates) * len(kalman_modes) * num_seeds

    print(f"\n{'='*76}")
    print(f"  RUNNING AUTONOMOUS PURSUIT BENCHMARK MATRIX ({total_runs} RUNS)")
    print(f"  Profiles:    {profiles}")
    print(f"  Drop Rates:  {[f'{int(d*100)}%' for d in drop_rates]}")
    print(f"  Ablation:    {'Kalman vs Raw' if test_ablation else 'Kalman Only'}")
    print(f"  Seeds/Cond:  {num_seeds} runs")
    print(f"  Duration:    {duration:.1f}s at 50 Hz")
    print(f"{'='*76}\n")

    raw_results = []
    completed = 0

    for profile in profiles:
        for drop_rate in drop_rates:
            for use_kalman in kalman_modes:
                mode_str = "Kalman" if use_kalman else "Raw"
                for s_idx in range(1, num_seeds + 1):
                    res = evaluate_single_run(
                        profile=profile,
                        seed=s_idx * 101,  # deterministic varied seeds
                        use_kalman=use_kalman,
                        frame_drop_rate=drop_rate,
                        duration=duration,
                        max_decel=max_decel
                    )
                    raw_results.append(res)
                    completed += 1

                print(f"  [{completed:3d}/{total_runs}] Profile: {profile:13s} | Drops: {int(drop_rate*100):2d}% | Mode: {mode_str:6s} | Complete")

    elapsed_time = time.perf_counter() - t_start
    print(f"\n[BENCHMARK] All {total_runs} simulation runs finished in {elapsed_time:.1f}s ({total_runs/elapsed_time:.1f} runs/sec)!")

    # -------------------------------------------------------------
    # Aggregate Metrics Calculation
    # -------------------------------------------------------------
    summary_table = []
    
    for profile in profiles:
        for drop_rate in drop_rates:
            for use_kalman in kalman_modes:
                subset = [
                    r for r in raw_results
                    if r["profile"] == profile and r["frame_drop_rate"] == drop_rate and r["use_kalman"] == use_kalman
                ]
                
                success_runs = [r for r in subset if r["success"]]
                times_to_lock = [r["time_to_lock"] for r in success_runs if r["time_to_lock"] is not None]

                summary_entry = {
                    "profile": profile,
                    "drop_rate": drop_rate,
                    "use_kalman": use_kalman,
                    "mode": "Kalman" if use_kalman else "Raw",
                    "num_runs": len(subset),
                    "success_rate_pct": float(100.0 * len(success_runs) / len(subset)),
                    "mean_time_to_lock": float(np.mean(times_to_lock)) if times_to_lock else None,
                    "std_time_to_lock": float(np.std(times_to_lock)) if times_to_lock else None,
                    "mean_fov_lock_pct": float(np.mean([r["fov_lock_pct"] for r in subset])),
                    "mean_firing_window_pct": float(np.mean([r["firing_window_pct"] for r in subset])),
                    "mean_min_distance": float(np.mean([r["min_distance"] for r in subset])),
                    "collision_breach_pct": float(100.0 * sum(1 for r in subset if r["collision_breach"]) / len(subset)),
                    "mean_path_length": float(np.mean([r["path_length"] for r in subset])),
                    "mean_control_effort": float(np.mean([r["control_effort"] for r in subset])),
                }
                summary_table.append(summary_entry)

    # -------------------------------------------------------------
    # Save JSON Report
    # -------------------------------------------------------------
    report_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_runs": total_runs,
            "duration_s": duration,
            "profiles": profiles,
            "drop_rates": drop_rates,
            "seeds_per_condition": num_seeds
        },
        "summary": summary_table,
        "raw_results": raw_results
    }

    json_path = output_dir / "pid_baseline_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"[REPORT] Structured benchmark data saved to: {json_path}")

    # -------------------------------------------------------------
    # Print Console Summary Table
    # -------------------------------------------------------------
    print("\n" + "=" * 105)
    print(f"{'Profile':14s} | {'Drops':5s} | {'Mode':6s} | {'Success%':8s} | {'Time-to-Lock':13s} | {'FOV Lock%':9s} | {'Firing Win%':11s} | {'Min Dist':8s} | {'Breaches':8s}")
    print("-" * 105)
    for row in summary_table:
        time_str = f"{row['mean_time_to_lock']:.2f}s" if row['mean_time_to_lock'] is not None else "TIMEOUT"
        print(f"{row['profile']:14s} | {int(row['drop_rate']*100):4d}% | {row['mode']:6s} | {row['success_rate_pct']:7.1f}% | {time_str:13s} | {row['mean_fov_lock_pct']:8.1f}% | {row['mean_firing_window_pct']:10.1f}% | {row['mean_min_distance']:7.2f}m | {row['collision_breach_pct']:7.1f}%")
    print("=" * 105 + "\n")

    # -------------------------------------------------------------
    # Generate Scientific Comparison Plots
    # -------------------------------------------------------------
    plot_path = output_dir / "pid_benchmark_analysis.png"
    _generate_benchmark_plots(summary_table, profiles, drop_rates, plot_path)
    print(f"[PLOTS] Benchmark performance plots saved to: {plot_path}")

    return report_data


def _generate_benchmark_plots(summary: List[Dict[str, Any]], profiles: List[str], drop_rates: List[float], out_path: Path):
    """
    Renders clean, publication-ready multi-panel performance curves.
    """
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150, facecolor="#ffffff")

    drop_pcts = [int(d * 100) for d in drop_rates]
    colors = {"cruising": "#2ca02c", "evasive": "#1f77b4", "hyper_evasive": "#d62728"}

    # 1. FOV Lock Retention vs Drop Rate (Kalman vs Raw)
    ax1 = axes[0, 0]
    ax1.set_facecolor("#fafafa")
    ax1.grid(True, linestyle="--", alpha=0.5, color="#c0c0c0")
    for prof in profiles:
        k_fov = [row["mean_fov_lock_pct"] for row in summary if row["profile"] == prof and row["use_kalman"]]
        r_fov = [row["mean_fov_lock_pct"] for row in summary if row["profile"] == prof and not row["use_kalman"]]
        if k_fov:
            ax1.plot(drop_pcts, k_fov, marker="o", lw=2.2, color=colors.get(prof, "#333"), label=f"{prof.title()} (Kalman)")
        if r_fov:
            ax1.plot(drop_pcts, r_fov, marker="s", ls="--", lw=1.6, color=colors.get(prof, "#333"), alpha=0.6, label=f"{prof.title()} (Raw)")
    ax1.set_xlabel("Frame Drop Rate (%)", fontsize=10)
    ax1.set_ylabel("FOV Lock Retention (%)", fontsize=10)
    ax1.set_title("Visual Lock Retention vs. Frame Drops (Kalman Ablation)", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, loc="lower left", framealpha=0.9)

    # 2. Success Rate vs Drop Rate
    ax2 = axes[0, 1]
    ax2.set_facecolor("#fafafa")
    ax2.grid(True, linestyle="--", alpha=0.5, color="#c0c0c0")
    for prof in profiles:
        k_succ = [row["success_rate_pct"] for row in summary if row["profile"] == prof and row["use_kalman"]]
        ax2.plot(drop_pcts, k_succ, marker="o", lw=2.2, color=colors.get(prof, "#333"), label=prof.title())
    ax2.set_xlabel("Frame Drop Rate (%)", fontsize=10)
    ax2.set_ylabel("Engagement Success Rate (%)", fontsize=10)
    ax2.set_title("Engagement Success Rate vs. Frame Drops", fontsize=11, fontweight="bold")
    ax2.set_ylim(-5, 105)
    ax2.legend(fontsize=8.5, loc="lower left", framealpha=0.9)

    # 3. Firing Window % vs Profile
    ax3 = axes[1, 0]
    ax3.set_facecolor("#fafafa")
    ax3.grid(True, linestyle="--", alpha=0.5, color="#c0c0c0")
    for prof in profiles:
        k_win = [row["mean_firing_window_pct"] for row in summary if row["profile"] == prof and row["use_kalman"]]
        ax3.plot(drop_pcts, k_win, marker="o", lw=2.2, color=colors.get(prof, "#333"), label=prof.title())
    ax3.set_xlabel("Frame Drop Rate (%)", fontsize=10)
    ax3.set_ylabel("Net Firing Window (% of flight)", fontsize=10)
    ax3.set_title("Effective Firing Solution Window (5m - 7m)", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8.5, loc="upper right", framealpha=0.9)

    # 4. Time-to-Lock vs Profile (at 0% and 10% drops)
    ax4 = axes[1, 1]
    ax4.set_facecolor("#fafafa")
    ax4.grid(True, linestyle="--", alpha=0.5, color="#c0c0c0")
    bar_width = 0.35
    x_indices = np.arange(len(profiles))

    times_0 = [row["mean_time_to_lock"] or 25.0 for row in summary if row["drop_rate"] == 0.0 and row["use_kalman"]]
    times_10 = [row["mean_time_to_lock"] or 25.0 for row in summary if row["drop_rate"] == 0.10 and row["use_kalman"]]

    ax4.bar(x_indices - bar_width/2, times_0, width=bar_width, color="#1f77b4", label="0% Drops", alpha=0.85)
    ax4.bar(x_indices + bar_width/2, times_10, width=bar_width, color="#ff7f0e", label="10% Drops", alpha=0.85)
    ax4.set_xticks(x_indices)
    ax4.set_xticklabels([p.title() for p in profiles], fontsize=10)
    ax4.set_ylabel("Mean Time-to-Lock (s)", fontsize=10)
    ax4.set_title("Time-to-Lock Across Flight Profiles", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=8.5, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def run_dataset_benchmark(
    dataset_path: Union[str, Path],
    use_kalman: bool = True,
    frame_drop_rate: float = 0.0,
    max_decel: float = 2.8,
    output_dir: Path = PROJECT_ROOT / "outputs" / "benchmarks" / "control"
) -> Dict[str, Any]:
    """
    Evaluates the controller across a standardized PursuitDataset manifest and prints grouped metrics.
    """
    dataset = PursuitDataset.load_json(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    mode_str = "Kalman" if use_kalman else "Raw"
    print(f"\n{'='*95}")
    print(f"  BENCHMARKING ON DATASET: {dataset.name} ({len(dataset)} SCENARIOS)")
    print(f"  Perception Mode:  {mode_str} | Frame Drop Rate: {int(frame_drop_rate*100)}%")
    print(f"  Max Deceleration: {max_decel:.1f} m/s^2")
    print(f"{'='*95}\n")

    raw_results = []
    for idx, scenario in enumerate(dataset.scenarios, 1):
        res = evaluate_single_run(
            scenario=scenario,
            use_kalman=use_kalman,
            frame_drop_rate=frame_drop_rate,
            max_decel=max_decel
        )
        raw_results.append(res)
        if idx % 20 == 0 or idx == len(dataset):
            print(f"  [{idx:3d}/{len(dataset)}] Scenario: {scenario.scenario_id:<38} | Complete")

    elapsed = time.perf_counter() - t_start
    print(f"\n[BENCHMARK] Evaluated {len(dataset)} scenarios in {elapsed:.1f}s ({len(dataset)/elapsed:.1f} runs/sec)!")

    # Group metrics by Encounter Type and Profile
    encounters = sorted(list(set(r["encounter_type"] for r in raw_results)))
    profiles = sorted(list(set(r["profile"] for r in raw_results)))

    summary_rows = []
    print("\n" + "=" * 95)
    print(f"{'Encounter Type':16s} | {'Profile':14s} | {'Success%':8s} | {'Time-to-Lock':13s} | {'FOV Lock%':9s} | {'Firing Win%':11s} | {'Min Dist':8s} | {'Breaches':8s}")
    print("-" * 95)

    for enc in encounters:
        for prof in profiles:
            subset = [r for r in raw_results if r["encounter_type"] == enc and r["profile"] == prof]
            if not subset:
                continue
            succ = np.mean([1 if r["success"] else 0 for r in subset]) * 100
            ttls = [r["time_to_lock"] for r in subset if r["time_to_lock"] is not None]
            ttl_mean = np.mean(ttls) if ttls else None
            fov = np.mean([r["fov_lock_pct"] for r in subset])
            fwin = np.mean([r["firing_window_pct"] for r in subset])
            dmin = np.mean([r["min_distance"] for r in subset])
            breach = np.mean([1 if r["collision_breach"] else 0 for r in subset]) * 100

            ttl_str = f"{ttl_mean:.2f}s" if ttl_mean is not None else "TIMEOUT"
            print(f"{enc:16s} | {prof:14s} | {succ:7.1f}% | {ttl_str:13s} | {fov:8.1f}% | {fwin:10.1f}% | {dmin:7.2f}m | {breach:7.1f}%")

            summary_rows.append({
                "encounter_type": enc,
                "profile": prof,
                "success_rate_pct": succ,
                "mean_time_to_lock": ttl_mean,
                "mean_fov_lock_pct": fov,
                "mean_firing_window_pct": fwin,
                "mean_min_distance": dmin,
                "collision_breach_pct": breach
            })

    print("=" * 95 + "\n")

    out_json = output_dir / f"benchmark_results_{dataset.name}_{mode_str.lower()}_d{int(frame_drop_rate*100)}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset.name, "summary": summary_rows, "results": raw_results}, f, indent=2)
    print(f"[REPORT] Dataset benchmark report saved to: {out_json}")
    return {"summary": summary_rows, "results": raw_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Drone Pursuit Benchmark Suite")
    parser.add_argument("--dataset", type=str, default=None, help="Path to PursuitDataset JSON manifest to benchmark")
    parser.add_argument("--seeds", type=int, default=15, help="Number of Monte Carlo seeds per condition (default: 15)")
    parser.add_argument("--duration", type=float, default=25.0, help="Flight duration in seconds (default: 25.0)")
    parser.add_argument("--max-decel", type=float, default=2.8, help="Controller forward deceleration limit in m/s^2 (default: 2.8)")
    parser.add_argument("--no-ablation", action="store_true", help="Skip Kalman vs Raw ablation comparison")
    parser.add_argument("--drop-rate", type=float, default=0.0, help="Frame drop rate when running dataset (default: 0.0)")
    args = parser.parse_args()

    if args.dataset:
        run_dataset_benchmark(
            dataset_path=args.dataset,
            use_kalman=True,
            frame_drop_rate=args.drop_rate,
            max_decel=args.max_decel
        )
    else:
        run_benchmark_matrix(
            num_seeds=args.seeds,
            test_ablation=(not args.no_ablation),
            duration=args.duration,
            max_decel=args.max_decel
        )
