#!/usr/bin/env python3
"""
Constant-Velocity Target Pursuit & Distance Holding Evaluation.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Evaluates the tuned Kinematic Visual Servo PID Controller with Optical Looming Feedforward:
- Target moves along +X at a constant velocity of 5.0 m/s.
- Chaser starts behind the target.
- Tests both:
  1. Catch-up from hover (v0 = 0 m/s, d0 = 7.0 m -> catch up and hold 3.5 m standoff).
  2. In-flight station-keeping (v0 = 5.0 m/s, d0 = 3.5 m -> hold 3.5 m standoff with zero lag).
- Generates comprehensive time-series diagnostics and bright broadcast MP4 video.
"""

import sys
import os
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.control.benchmark_controller import (
    create_pid_controller,
    evaluate_single_run,
)
from control.rendering.render_pid_chase_bright import render_pid_bright_video


class ConstantVelocityTargetOracle:
    """Target oracle maintaining an exact constant forward velocity vector."""
    def __init__(
        self,
        initial_pos: np.ndarray = np.array([7.0, 0.0, -2.5], dtype=np.float64),
        speed: float = 5.0,
        heading_rad: float = 0.0,
    ):
        self.initial_pos = np.array(initial_pos, dtype=np.float64)
        self.speed = float(speed)
        self.heading = float(heading_rad)
        self.vel = np.array(
            [self.speed * np.cos(self.heading), self.speed * np.sin(self.heading), 0.0],
            dtype=np.float64
        )
        self.acc = np.zeros(3, dtype=np.float64)

    def get_position(self, t: float) -> np.ndarray:
        return self.initial_pos + self.vel * t

    def get_velocity(self, t: float) -> np.ndarray:
        return self.vel.copy()

    def get_acceleration(self, t: float) -> np.ndarray:
        return self.acc.copy()

    def get_state(self, t: float):
        return self.get_position(t), self.get_velocity(t), self.get_acceleration(t)


class ConstantVelocityScenario:
    """PursuitScenario-compatible container for constant-velocity target flight."""
    def __init__(
        self,
        scenario_id: str,
        initial_distance: float = 7.0,
        target_speed: float = 5.0,
        initial_chaser_speed: float = 0.0,
        duration: float = 25.0,
        initial_alt: float = -2.5,
    ):
        self.scenario_id = scenario_id
        self.profile = f"const_vel_{target_speed:.1f}ms"
        self.encounter_type = "tail_chase"
        self.duration = float(duration)
        self.seed = 42
        self.initial_distance = float(initial_distance)
        self.initial_azimuth_deg = 0.0
        self.initial_elevation_deg = 0.0
        self.initial_target_pos = [float(initial_distance), 0.0, float(initial_alt)]
        self.initial_target_heading_rad = 0.0
        self.initial_chaser_pos = [0.0, 0.0, float(initial_alt)]
        self.initial_chaser_yaw = 0.0
        self.target_speed = float(target_speed)
        self.initial_chaser_speed = float(initial_chaser_speed)

    def create_target_oracle(self) -> ConstantVelocityTargetOracle:
        return ConstantVelocityTargetOracle(
            initial_pos=np.array(self.initial_target_pos, dtype=np.float64),
            speed=self.target_speed,
            heading_rad=self.initial_target_heading_rad
        )


def evaluate_chase(
    scenario: ConstantVelocityScenario,
    noise_mode: str = "empirical",
    frame_drop_rate: float = 0.05,
    standoff_nominal: float = 3.5,
) -> Dict[str, Any]:
    """Runs closed-loop pursuit simulation for the constant velocity scenario."""
    pid = create_pid_controller(standoff_nominal=standoff_nominal)
    pid.reset()

    res = evaluate_single_run(
        scenario=scenario,
        controller=pid,
        controller_type="pid",
        noise_mode=noise_mode,
        frame_drop_rate=frame_drop_rate,
        standoff_nominal=standoff_nominal,
        standoff_min=2.5,
        standoff_max=4.5,
        dwell_time_required=1.0,
        use_kalman=True,
        return_history=True,
        enable_ego_compensation=True,
        collision_dist_m=0.5,
    )
    res["controller_instance"] = pid
    return res


def compute_steady_state_metrics(
    history: Dict[str, Any],
    t_settle_start: float = 8.0,
    standoff_nominal: float = 3.5,
    target_speed: float = 5.0,
) -> Dict[str, float]:
    """Computes precision steady-state station-keeping and velocity matching metrics."""
    times = np.array(history["times"])
    c_pos = np.array(history["chaser_pos"])
    t_pos = np.array(history["target_pos"])
    c_vel = np.array(history["chaser_vel"])

    dists = np.linalg.norm(c_pos - t_pos, axis=1)
    chaser_speeds = np.linalg.norm(c_vel[:, :2], axis=1)

    # Steady state slice
    ss_mask = times >= t_settle_start
    if not np.any(ss_mask):
        ss_mask = np.ones_like(times, dtype=bool)

    ss_times = times[ss_mask]
    ss_dists = dists[ss_mask]
    ss_speeds = chaser_speeds[ss_mask]

    mean_dist = float(np.mean(ss_dists))
    std_dist = float(np.std(ss_dists))
    dist_error = float(abs(mean_dist - standoff_nominal))
    max_dist_error = float(np.max(np.abs(ss_dists - standoff_nominal)))

    mean_speed = float(np.mean(ss_speeds))
    std_speed = float(np.std(ss_speeds))
    speed_error = float(abs(mean_speed - target_speed))

    # In basket percentage in steady state
    in_basket_ss = np.sum((ss_dists >= 2.5) & (ss_dists <= 4.5)) / len(ss_dists) * 100.0

    return {
        "ss_mean_dist_m": mean_dist,
        "ss_std_dist_m": std_dist,
        "ss_dist_error_m": dist_error,
        "ss_max_dist_error_m": max_dist_error,
        "ss_mean_speed_ms": mean_speed,
        "ss_std_speed_ms": std_speed,
        "ss_speed_error_ms": speed_error,
        "ss_basket_retention_pct": float(in_basket_ss),
        "total_min_dist_m": float(np.min(dists)),
        "total_max_dist_m": float(np.max(dists)),
    }


def plot_diagnostics(
    res_hover: Dict[str, Any],
    res_cruise: Dict[str, Any],
    output_path: Path,
    target_speed: float = 5.0,
    standoff_nominal: float = 3.5,
):
    """Generates broadcast-grade publication diagnostic plots for both test cases."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), dpi=150)
    plt.subplots_adjust(wspace=0.22, hspace=0.38, left=0.07, right=0.96, top=0.93, bottom=0.06)

    cases = [
        ("Case 1: Catch-Up from Hover (v0 = 0 m/s, d0 = 7.0 m)", res_hover, 0, 8.0),
        ("Case 2: In-Flight Rendezvous (v0 = 5 m/s, d0 = 3.5 m)", res_cruise, 1, 2.0),
    ]

    for title, res, col, ss_t in cases:
        h = res["history"]
        t = np.array(h["times"])
        c_pos = np.array(h["chaser_pos"])
        t_pos = np.array(h["target_pos"])
        c_vel = np.array(h["chaser_vel"])
        c_cmd = np.array(h["chaser_cmd"])
        telems = h["chaser_telem"]

        dist = np.linalg.norm(c_pos - t_pos, axis=1)
        speed = np.linalg.norm(c_vel[:, :2], axis=1)
        cmd_vx = c_cmd[:, 0]
        bbox_w = np.array([float(tel.get("bbox_w", 0.0)) for tel in telems])

        # -------------------------------------------------------------
        # Row 0: Distance vs Standoff Basket
        # -------------------------------------------------------------
        ax = axes[0, col]
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8, color="#0F172A")
        ax.axhspan(2.5, 4.5, color="#10B981", alpha=0.15, label="Capture Basket [2.5m - 4.5m]")
        ax.axhline(standoff_nominal, color="#059669", ls="--", lw=1.8, label=f"Standoff Setpoint ({standoff_nominal}m)")
        ax.axhline(0.5, color="#EF4444", ls=":", lw=1.5, label="Collision Breach (0.5m)")
        ax.plot(t, dist, color="#2563EB", lw=2.2, label="Euclidean Distance d(t)")
        ax.axvline(ss_t, color="#64748B", ls="-.", lw=1.2, alpha=0.7, label=f"Steady-State Window (>= {ss_t}s)")

        ax.set_ylabel("Distance (m)", fontsize=10, fontweight="bold")
        ax.set_ylim(0.0, max(8.5, np.max(dist) + 0.5))
        ax.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")
        ax.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

        # -------------------------------------------------------------
        # Row 1: Velocity Tracking (Drone vs Target 5 m/s)
        # -------------------------------------------------------------
        ax = axes[1, col]
        ax.axhline(target_speed, color="#DC2626", ls="--", lw=2.0, label=f"Target Speed ({target_speed:.1f} m/s)")
        ax.plot(t, speed, color="#2563EB", lw=2.0, label="Drone Velocity v_drone(t)")
        ax.plot(t, cmd_vx, color="#9333EA", ls=":", lw=1.5, alpha=0.8, label="Commanded Surge v_cmd(t)")
        ax.axvline(ss_t, color="#64748B", ls="-.", lw=1.2, alpha=0.7)

        ax.set_ylabel("Speed (m/s)", fontsize=10, fontweight="bold")
        ax.set_ylim(-0.5, 9.0)
        ax.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")
        ax.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

        # -------------------------------------------------------------
        # Row 2: Vision Perception (BBox Width vs 110 px)
        # -------------------------------------------------------------
        ax = axes[2, col]
        ax.axhline(110.0, color="#059669", ls="--", lw=1.8, label="Desired BBox Width (110 px)")
        ax.plot(t, bbox_w, color="#D97706", lw=2.0, label="Tracked BBox Width w(t)")
        ax.axvline(ss_t, color="#64748B", ls="-.", lw=1.2, alpha=0.7)

        ax.set_ylabel("Target Width (px)", fontsize=10, fontweight="bold")
        ax.set_ylim(40.0, 180.0)
        ax.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")
        ax.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

        # -------------------------------------------------------------
        # Row 3: Tracking Error (d(t) - d*)
        # -------------------------------------------------------------
        ax = axes[3, col]
        dist_err = dist - standoff_nominal
        ax.axhline(0.0, color="#059669", ls="-", lw=1.5, alpha=0.7)
        ax.plot(t, dist_err, color="#0284C7", lw=2.0, label="Distance Error: e_d = d - d*")
        ax.fill_between(t, -1.0, 1.0, color="#10B981", alpha=0.10, label="Allowed Error ±1.0m")
        ax.axvline(ss_t, color="#64748B", ls="-.", lw=1.2, alpha=0.7)

        ax.set_xlabel("Time (s)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Error e_d (m)", fontsize=10, fontweight="bold")
        ax.set_ylim(-2.0, max(4.0, np.max(dist_err) + 0.5))
        ax.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")
        ax.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

    fig.suptitle(
        "Kinematic Visual Servo PID Controller — Constant 5.0 m/s Target Pursuit & Distance Holding",
        fontsize=14,
        fontweight="bold",
        color="#0F172A",
        y=0.97
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved comprehensive diagnostics to: {output_path}")


def main():
    print("=" * 80)
    print("EVALUATING TUNED PID CONTROLLER: CONSTANT 5.0 M/S TARGET PURSUIT")
    print("=" * 80)

    # Test 1: Catch-up from hover
    scen_hover = ConstantVelocityScenario(
        scenario_id="const_vel_5ms_hover_start",
        initial_distance=7.0,
        target_speed=5.0,
        initial_chaser_speed=0.0,
        duration=25.0
    )
    print("\n--- Running Case 1: Chaser starts at hover (v0 = 0 m/s, d0 = 7.0 m) ---")
    res_hover = evaluate_chase(scen_hover, noise_mode="empirical", frame_drop_rate=0.05)
    ss_hover = compute_steady_state_metrics(res_hover["history"], t_settle_start=8.0)

    print(f"  1s Catch: {res_hover['success_1s']} | 3s Catch: {res_hover['success_3s']}")
    print(f"  Max Continuous Dwell: {res_hover['max_continuous_dwell_s']:.2f} s")
    print(f"  Steady-State Mean Distance: {ss_hover['ss_mean_dist_m']:.3f} m (Std: {ss_hover['ss_std_dist_m']:.3f} m)")
    print(f"  Steady-State Distance Error: {ss_hover['ss_dist_error_m']:.3f} m")
    print(f"  Steady-State Mean Speed: {ss_hover['ss_mean_speed_ms']:.3f} m/s (Target: 5.0 m/s, Error: {ss_hover['ss_speed_error_ms']:.3f} m/s)")
    print(f"  Basket Retention (t >= 8s): {ss_hover['ss_basket_retention_pct']:.1f}%")

    # Test 2: In-flight rendezvous / cruise start
    scen_cruise = ConstantVelocityScenario(
        scenario_id="const_vel_5ms_cruise_start",
        initial_distance=3.5,
        target_speed=5.0,
        initial_chaser_speed=5.0,
        duration=25.0
    )
    print("\n--- Running Case 2: In-flight rendezvous (v0 = 5.0 m/s, d0 = 3.5 m) ---")
    res_cruise = evaluate_chase(scen_cruise, noise_mode="empirical", frame_drop_rate=0.05)
    ss_cruise = compute_steady_state_metrics(res_cruise["history"], t_settle_start=2.0)

    print(f"  1s Catch: {res_cruise['success_1s']} | 3s Catch: {res_cruise['success_3s']}")
    print(f"  Max Continuous Dwell: {res_cruise['max_continuous_dwell_s']:.2f} s")
    print(f"  Steady-State Mean Distance: {ss_cruise['ss_mean_dist_m']:.3f} m (Std: {ss_cruise['ss_std_dist_m']:.3f} m)")
    print(f"  Steady-State Distance Error: {ss_cruise['ss_dist_error_m']:.3f} m")
    print(f"  Steady-State Mean Speed: {ss_cruise['ss_mean_speed_ms']:.3f} m/s (Target: 5.0 m/s, Error: {ss_cruise['ss_speed_error_ms']:.3f} m/s)")
    print(f"  Basket Retention (t >= 2s): {ss_cruise['ss_basket_retention_pct']:.1f}%")

    # Save diagnostic figure
    plot_path = PROJECT_ROOT / "outputs" / "control" / "constant_velocity_5ms_evaluation.png"
    plot_diagnostics(res_hover, res_cruise, plot_path)

    # Render bright video for Case 1 (catch-up & hold)
    video_path = PROJECT_ROOT / "outputs" / "control" / "constant_velocity_5ms_chase_bright.mp4"
    print(f"\n[VIDEO] Rendering bright-mode MP4 video of constant 5 m/s chase to: {video_path}")
    render_pid_bright_video(
        scenario=scen_hover,
        pid_controller=res_hover["controller_instance"],
        output_path=video_path,
        fps=25
    )

    # Save JSON summary report
    summary_path = PROJECT_ROOT / "outputs" / "control" / "constant_velocity_5ms_metrics.json"
    summary_data = {
        "target_speed_ms": 5.0,
        "standoff_nominal_m": 3.5,
        "basket_min_m": 2.5,
        "basket_max_m": 4.5,
        "case_1_hover_start": {
            "initial_distance_m": 7.0,
            "initial_chaser_speed_ms": 0.0,
            "success_1s": bool(res_hover["success_1s"]),
            "success_3s": bool(res_hover["success_3s"]),
            "max_continuous_dwell_s": float(res_hover["max_continuous_dwell_s"]),
            "fov_lock_pct": float(res_hover["fov_lock_pct"]),
            "steady_state_metrics": ss_hover
        },
        "case_2_cruise_start": {
            "initial_distance_m": 3.5,
            "initial_chaser_speed_ms": 5.0,
            "success_1s": bool(res_cruise["success_1s"]),
            "success_3s": bool(res_cruise["success_3s"]),
            "max_continuous_dwell_s": float(res_cruise["max_continuous_dwell_s"]),
            "fov_lock_pct": float(res_cruise["fov_lock_pct"]),
            "steady_state_metrics": ss_cruise
        }
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[METRICS] Saved JSON report to: {summary_path}")

    # Also copy plot and video to artifact directory for presentation
    artifact_dir = Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/5967d853-7a7e-4cdd-8792-4a77e6d704a2")
    if artifact_dir.exists():
        import shutil
        shutil.copy2(plot_path, artifact_dir / "constant_velocity_5ms_evaluation.png")
        shutil.copy2(video_path, artifact_dir / "constant_velocity_5ms_chase_bright.mp4")
        print(f"[ARTIFACT] Copied plots and video to: {artifact_dir}")


if __name__ == "__main__":
    main()
