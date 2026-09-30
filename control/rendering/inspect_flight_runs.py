#!/usr/bin/env python3
"""
Flight Inspection & Evaluation Suite for Autonomous Drone Tracking.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Simulates and evaluates the trained Recurrent PPO policy across representative
encounter scenarios (Evasive, Break-Turn & Dive, Aerobatic Figure-8) and compares
its flight characteristics directly against the classical PID baseline.

Generates:
1. Multi-panel 3D & 2D trajectory plots with ground projections and sightlines.
2. Real-time telemetry strip charts (standoff distance, body velocities, camera errors).
3. Summary aerospace metrics table (FOV lock %, standoff error, safety envelope, control jerk).
4. High-resolution artifact figures for direct visual inspection.
"""

import sys
import os
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

from control.rendering.create_chase_movie import (
    simulate_pursuit,
    get_target_trajectory,
)
from control.predictive_controller import RecurrentVisualServoController
from control.pid_controller import KinematicVisualServoController

ARTIFACT_DIR = Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/53eb0b3e-b75c-4830-a9d4-c69e644ccb6a")


def compute_flight_metrics(sim_data: dict, nominal_standoff: float = 6.0) -> dict:
    """Computes aerospace tracking performance metrics from flight simulation telemetry."""
    times = sim_data["times"]
    dists = sim_data["distance"]
    in_views = sim_data["in_view"]
    c_vel = sim_data["chaser_vel"]
    cmds = sim_data["cmd"]
    dt = sim_data["dt"]

    total_time = times[-1] - times[0] if len(times) > 1 else 0.01
    fov_time = np.sum(in_views) * dt
    fov_percent = (fov_time / total_time) * 100.0

    mean_dist = float(np.mean(dists))
    std_dist = float(np.std(dists))
    min_dist = float(np.min(dists))
    max_dist = float(np.max(dists))
    rmse_dist = float(np.sqrt(np.mean((dists - nominal_standoff) ** 2)))

    # In-corridor retention [5.0m, 7.0m]
    in_corridor = (dists >= 5.0) & (dists <= 7.0) & in_views
    corridor_time = np.sum(in_corridor) * dt
    corridor_percent = (corridor_time / total_time) * 100.0

    # Chaser speeds
    speeds = np.linalg.norm(c_vel, axis=1)
    max_speed = float(np.max(speeds))
    mean_speed = float(np.mean(speeds))

    # Control jerk / smoothness: finite differences of commanded acceleration
    cmd_accels = np.diff(cmds, axis=0) / dt
    cmd_jerks = np.diff(cmd_accels, axis=0) / dt
    jerk_rms = float(np.sqrt(np.mean(cmd_jerks ** 2)))

    return {
        "duration_s": total_time,
        "fov_retention_pct": fov_percent,
        "corridor_retention_pct": corridor_percent,
        "mean_distance_m": mean_dist,
        "std_distance_m": std_dist,
        "min_distance_m": min_dist,
        "max_distance_m": max_dist,
        "rmse_distance_m": rmse_dist,
        "mean_speed_m_s": mean_speed,
        "max_speed_m_s": max_speed,
        "control_jerk_rms": jerk_rms,
        "collision_breach": bool(min_dist < 2.5),
    }


def plot_single_run_inspection(
    rl_sim: dict,
    pid_sim: dict,
    scenario_title: str,
    output_path: Path,
):
    """
    Renders an in-depth 4-panel aerospace inspection dashboard for a single flight scenario.
    Compares the learned Recurrent PPO policy vs. Classical PID baseline.
    """
    times = rl_sim["times"]
    t_tgt = rl_sim["target_pos"]
    c_rl = rl_sim["chaser_pos"]
    c_pid = pid_sim["chaser_pos"]

    fig = plt.figure(figsize=(18, 12), dpi=140)
    fig.patch.set_facecolor("#0f141c")
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.3, 1.0, 1.0], hspace=0.32, wspace=0.22)

    # --------------------------------------------------------------------------
    # Panel 1: 3D Spatial Pursuit Arena
    # --------------------------------------------------------------------------
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax3d.set_facecolor("#0f141c")
    ax3d.xaxis.set_pane_color((0.08, 0.11, 0.16, 0.8))
    ax3d.yaxis.set_pane_color((0.08, 0.11, 0.16, 0.8))
    ax3d.zaxis.set_pane_color((0.08, 0.11, 0.16, 0.8))

    # NED to display: X=East, Y=North, Z=Altitude (positive up)
    tgt_e, tgt_n, tgt_alt = t_tgt[:, 1], t_tgt[:, 0], -t_tgt[:, 2]
    rl_e, rl_n, rl_alt = c_rl[:, 1], c_rl[:, 0], -c_rl[:, 2]
    pid_e, pid_n, pid_alt = c_pid[:, 1], c_pid[:, 0], -c_pid[:, 2]

    # Target path
    ax3d.plot(tgt_e, tgt_n, tgt_alt, color="#ff3366", lw=2.2, label="Target Trajectory", alpha=0.95)
    # Ground shadows
    ax3d.plot(tgt_e, tgt_n, np.zeros_like(tgt_alt), color="#ff3366", lw=1.0, ls="--", alpha=0.35)

    # RL Chaser path
    ax3d.plot(rl_e, rl_n, rl_alt, color="#00ffcc", lw=2.5, label="Recurrent PPO (Learned)", alpha=0.95)
    ax3d.plot(rl_e, rl_n, np.zeros_like(rl_alt), color="#00ffcc", lw=1.0, ls="--", alpha=0.35)

    # PID Chaser path
    ax3d.plot(pid_e, pid_n, pid_alt, color="#ffaa00", lw=1.8, ls=":", label="Classical PID Baseline", alpha=0.75)

    # Start and End Markers
    ax3d.scatter([tgt_e[0]], [tgt_n[0]], [tgt_alt[0]], color="#ff3366", s=50, marker="o")
    ax3d.scatter([rl_e[0]], [rl_n[0]], [rl_alt[0]], color="#00ffcc", s=50, marker="^")
    ax3d.scatter([rl_e[-1]], [rl_n[-1]], [rl_alt[-1]], color="#00ffcc", s=80, marker="*")

    ax3d.set_xlabel("East (m)", color="#8fa3bf", labelpad=8)
    ax3d.set_ylabel("North (m)", color="#8fa3bf", labelpad=8)
    ax3d.set_zlabel("Altitude AGL (m)", color="#8fa3bf", labelpad=8)
    ax3d.tick_params(colors="#8fa3bf", labelsize=8)
    ax3d.set_title("3D Flight Space (NED → ENU Projection)", color="#e0e8f5", fontsize=11, fontweight="bold")
    ax3d.legend(loc="upper left", facecolor="#182232", edgecolor="#223348", labelcolor="#e0e8f5", fontsize=8.5)

    # --------------------------------------------------------------------------
    # Panel 2: Top-Down 2D Pursuit View (North vs East)
    # --------------------------------------------------------------------------
    ax2d = fig.add_subplot(gs[0, 1])
    ax2d.set_facecolor("#141b26")
    ax2d.grid(True, color="#223044", ls="--", lw=0.6, alpha=0.7)

    ax2d.plot(tgt_e, tgt_n, color="#ff3366", lw=2.2, label="Target Path", zorder=3)
    ax2d.plot(rl_e, rl_n, color="#00ffcc", lw=2.4, label="Recurrent PPO", zorder=4)
    ax2d.plot(pid_e, pid_n, color="#ffaa00", lw=1.6, ls=":", label="PID Baseline", zorder=2)

    # Draw periodic sightline lasers for RL
    sample_indices = np.linspace(0, len(times) - 1, 12, dtype=int)
    for idx in sample_indices:
        ax2d.plot([rl_e[idx], tgt_e[idx]], [rl_n[idx], tgt_n[idx]], color="#00ffcc", lw=0.8, ls="--", alpha=0.45)
        # Heading arrow for RL
        yaw = rl_sim["chaser_yaw"][idx]
        ax2d.arrow(rl_e[idx], rl_n[idx], 1.8 * np.sin(yaw), 1.8 * np.cos(yaw),
                   head_width=0.8, head_length=1.0, fc="#00ffcc", ec="#00ffcc", alpha=0.8)

    ax2d.scatter([tgt_e[0]], [tgt_n[0]], color="#ff3366", s=60, marker="o", label="Target Start")
    ax2d.scatter([rl_e[0]], [rl_n[0]], color="#00ffcc", s=60, marker="^", label="Chaser Start")
    ax2d.set_xlabel("East (m)", color="#8fa3bf", fontsize=10)
    ax2d.set_ylabel("North (m)", color="#8fa3bf", fontsize=10)
    ax2d.tick_params(colors="#8fa3bf", labelsize=8)
    ax2d.set_title("Top-Down 2D Pursuit Geometry (with LOS Sightlines)", color="#e0e8f5", fontsize=11, fontweight="bold")
    ax2d.legend(loc="best", facecolor="#182232", edgecolor="#223348", labelcolor="#e0e8f5", fontsize=8.5)
    ax2d.set_aspect("equal", adjustable="datalim")

    # --------------------------------------------------------------------------
    # Panel 3: Standoff Distance & Engagement Envelope
    # --------------------------------------------------------------------------
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_dist.set_facecolor("#141b26")
    ax_dist.grid(True, color="#223044", ls="--", lw=0.6, alpha=0.7)

    # Zones
    ax_dist.axhspan(5.0, 7.0, color="#00ffcc", alpha=0.12, label="Nominal Envelope [5-7m]")
    ax_dist.axhline(6.0, color="#00ffcc", ls="--", lw=1.2, alpha=0.7, label="Target Standoff (6.0m)")
    ax_dist.axhline(2.5, color="#ff3366", ls="-.", lw=1.5, alpha=0.8, label="Safety Threshold (2.5m)")

    ax_dist.plot(times, rl_sim["distance"], color="#00ffcc", lw=2.2, label="Recurrent PPO")
    ax_dist.plot(times, pid_sim["distance"], color="#ffaa00", lw=1.5, ls=":", label="Classical PID")

    ax_dist.set_xlim(times[0], times[-1])
    ax_dist.set_ylim(0, max(15.0, np.max(rl_sim["distance"]) * 1.15))
    ax_dist.set_xlabel("Time (s)", color="#8fa3bf", fontsize=9)
    ax_dist.set_ylabel("Range / Standoff (m)", color="#8fa3bf", fontsize=9)
    ax_dist.tick_params(colors="#8fa3bf", labelsize=8)
    ax_dist.set_title("Metric Standoff Distance Profile", color="#e0e8f5", fontsize=10, fontweight="bold")
    ax_dist.legend(loc="upper right", facecolor="#182232", edgecolor="#223348", labelcolor="#e0e8f5", fontsize=8)

    # --------------------------------------------------------------------------
    # Panel 4: Drone Body Velocities (Forward, Lateral, Vertical)
    # --------------------------------------------------------------------------
    ax_vel = fig.add_subplot(gs[1, 1])
    ax_vel.set_facecolor("#141b26")
    ax_vel.grid(True, color="#223044", ls="--", lw=0.6, alpha=0.7)

    # Convert chaser world velocities to body frame
    yaws = rl_sim["chaser_yaw"]
    c, s = np.cos(yaws), np.sin(yaws)
    vx_b = c * rl_sim["chaser_vel"][:, 0] + s * rl_sim["chaser_vel"][:, 1]
    vy_b = -s * rl_sim["chaser_vel"][:, 0] + c * rl_sim["chaser_vel"][:, 1]
    vz_b = rl_sim["chaser_vel"][:, 2]

    ax_vel.plot(times, vx_b, color="#00ffcc", lw=1.8, label="Body $v_x$ (Forward)")
    ax_vel.plot(times, vy_b, color="#ffaa00", lw=1.6, label="Body $v_y$ (Lateral Strafe)")
    ax_vel.plot(times, -vz_b, color="#3399ff", lw=1.4, ls="--", label="Climb Rate (-$v_z$)")

    ax_vel.axhline(18.0, color="#8fa3bf", ls=":", lw=0.8, alpha=0.5, label="Max $v_x$ (18 m/s)")
    ax_vel.set_xlim(times[0], times[-1])
    ax_vel.set_xlabel("Time (s)", color="#8fa3bf", fontsize=9)
    ax_vel.set_ylabel("Velocity (m/s)", color="#8fa3bf", fontsize=9)
    ax_vel.tick_params(colors="#8fa3bf", labelsize=8)
    ax_vel.set_title("Recurrent PPO Quadrotor Body-Frame Dynamics", color="#e0e8f5", fontsize=10, fontweight="bold")
    ax_vel.legend(loc="upper right", facecolor="#182232", edgecolor="#223348", labelcolor="#e0e8f5", fontsize=8)

    # --------------------------------------------------------------------------
    # Panel 5: Horizon-Stabilized Visual Sightline Tracking Errors
    # --------------------------------------------------------------------------
    ax_err = fig.add_subplot(gs[2, 0])
    ax_err.set_facecolor("#141b26")
    ax_err.grid(True, color="#223044", ls="--", lw=0.6, alpha=0.7)

    ax_err.plot(times, rl_sim["err_x"], color="#00ffcc", lw=1.8, label="Azimuth Error $e_{x,h}$")
    ax_err.plot(times, rl_sim["err_y"], color="#ff3366", lw=1.6, label="Elevation Error $e_{y,h}$")
    ax_err.axhspan(-0.25, 0.25, color="#ffffff", alpha=0.06, label="Tight Center Lock [±0.25]")
    ax_err.axhline(0.0, color="#8fa3bf", ls="--", lw=0.8, alpha=0.5)

    ax_err.set_xlim(times[0], times[-1])
    ax_err.set_ylim(-1.05, 1.05)
    ax_err.set_xlabel("Time (s)", color="#8fa3bf", fontsize=9)
    ax_err.set_ylabel("Normalized Error [-1, 1]", color="#8fa3bf", fontsize=9)
    ax_err.tick_params(colors="#8fa3bf", labelsize=8)
    ax_err.set_title("On-Board Camera Tracking Error (Attitude Decoupled)", color="#e0e8f5", fontsize=10, fontweight="bold")
    ax_err.legend(loc="upper right", facecolor="#182232", edgecolor="#223348", labelcolor="#e0e8f5", fontsize=8)

    # --------------------------------------------------------------------------
    # Panel 6: Yaw Rate & Attitude Deflection
    # --------------------------------------------------------------------------
    ax_yaw = fig.add_subplot(gs[2, 1])
    ax_yaw.set_facecolor("#141b26")
    ax_yaw.grid(True, color="#223044", ls="--", lw=0.6, alpha=0.7)

    yaw_rate_deg = np.rad2deg(np.gradient(rl_sim["chaser_yaw"], times))
    pitch_deg = np.rad2deg(rl_sim["chaser_pitch"])
    roll_deg = np.rad2deg(rl_sim["chaser_roll"])

    ax_yaw.plot(times, yaw_rate_deg, color="#ffaa00", lw=1.6, label=r"Yaw Rate $\dot{\psi}$ (deg/s)")
    ax_yaw.plot(times, pitch_deg, color="#00ffcc", lw=1.4, ls="--", label=r"Body Pitch $\theta$ (deg)")
    ax_yaw.plot(times, roll_deg, color="#ff3366", lw=1.2, ls=":", label=r"Body Roll $\phi$ (deg)")

    ax_yaw.set_xlim(times[0], times[-1])
    ax_yaw.set_xlabel("Time (s)", color="#8fa3bf", fontsize=9)
    ax_yaw.set_ylabel("Rate / Angle (deg, deg/s)", color="#8fa3bf", fontsize=9)
    ax_yaw.tick_params(colors="#8fa3bf", labelsize=8)
    ax_yaw.set_title("Chaser Airframe Attitude & Turn Rate Dynamics", color="#e0e8f5", fontsize=10, fontweight="bold")
    ax_yaw.legend(loc="upper right", facecolor="#182232", edgecolor="#223348", labelcolor="#e0e8f5", fontsize=8)

    # Figure Super-Title
    fig.suptitle(f"Autonomous Drone Tracking — Flight Dynamics Inspection: {scenario_title}",
                 color="#ffffff", fontsize=15, fontweight="bold", y=0.98)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Generated flight inspection figure: {output_path}")


def plot_scenario_overview_comparison(
    results: List[Tuple[str, dict, dict, dict, dict]],
    output_path: Path,
):
    """
    Renders a unified side-by-side comparison across 3 distinct test scenarios:
    1. Evasive Maneuver
    2. Tactical Break-Turn & Dive
    3. Aerobatic Figure-8 Loop
    """
    num_scenarios = len(results)
    fig, axes = plt.subplots(num_scenarios, 3, figsize=(18, 5 * num_scenarios), dpi=140)
    fig.patch.set_facecolor("#0f141c")

    for row_idx, (name, rl_sim, pid_sim, rl_met, pid_met) in enumerate(results):
        times = rl_sim["times"]
        t_tgt = rl_sim["target_pos"]
        c_rl = rl_sim["chaser_pos"]
        c_pid = pid_sim["chaser_pos"]

        # Column 1: Top-Down 2D Trajectory
        ax1 = axes[row_idx, 0]
        ax1.set_facecolor("#141b26")
        ax1.grid(True, color="#223044", ls="--", lw=0.6, alpha=0.7)

        ax1.plot(t_tgt[:, 1], t_tgt[:, 0], color="#ff3366", lw=2.2, label="Target")
        ax1.plot(c_rl[:, 1], c_rl[:, 0], color="#00ffcc", lw=2.2, label="Recurrent PPO")
        ax1.plot(c_pid[:, 1], c_pid[:, 0], color="#ffaa00", lw=1.5, ls=":", label="PID Baseline")
        ax1.scatter([c_rl[0, 1]], [c_rl[0, 0]], color="#00ffcc", marker="^", s=40)
        ax1.set_xlabel("East (m)", color="#8fa3bf", fontsize=8.5)
        ax1.set_ylabel("North (m)", color="#8fa3bf", fontsize=8.5)
        ax1.tick_params(colors="#8fa3bf", labelsize=8)
        ax1.set_title(f"{name}: Overhead Pursuit Map", color="#e0e8f5", fontsize=10, fontweight="bold")
        ax1.legend(loc="best", facecolor="#182232", edgecolor="#223348", labelcolor="#e0e8f5", fontsize=7.5)
        ax1.set_aspect("equal", adjustable="datalim")

        # Column 2: Standoff Distance Profile
        ax2 = axes[row_idx, 1]
        ax2.set_facecolor("#141b26")
        ax2.grid(True, color="#223044", ls="--", lw=0.6, alpha=0.7)

        ax2.axhspan(5.0, 7.0, color="#00ffcc", alpha=0.10)
        ax2.axhline(6.0, color="#00ffcc", ls="--", lw=1.0, alpha=0.7, label="Nominal 6.0m")
        ax2.axhline(2.5, color="#ff3366", ls="-.", lw=1.2, alpha=0.7, label="Safety Limit 2.5m")

        ax2.plot(times, rl_sim["distance"], color="#00ffcc", lw=2.0,
                 label=f"RL (Mean: {rl_met['mean_distance_m']:.1f}m, Min: {rl_met['min_distance_m']:.1f}m)")
        ax2.plot(times, pid_sim["distance"], color="#ffaa00", lw=1.4, ls=":",
                 label=f"PID (Mean: {pid_met['mean_distance_m']:.1f}m, Min: {pid_met['min_distance_m']:.1f}m)")

        ax2.set_xlim(times[0], times[-1])
        ax2.set_ylim(0, max(14.0, np.max(rl_sim["distance"]) * 1.1))
        ax2.set_xlabel("Time (s)", color="#8fa3bf", fontsize=8.5)
        ax2.set_ylabel("Distance (m)", color="#8fa3bf", fontsize=8.5)
        ax2.tick_params(colors="#8fa3bf", labelsize=8)
        ax2.set_title(f"{name}: Standoff Distance Comparison", color="#e0e8f5", fontsize=10, fontweight="bold")
        ax2.legend(loc="upper right", facecolor="#182232", edgecolor="#223348", labelcolor="#e0e8f5", fontsize=7.5)

        # Column 3: Azimuth Sightline Error & FOV Lock
        ax3 = axes[row_idx, 2]
        ax3.set_facecolor("#141b26")
        ax3.grid(True, color="#223044", ls="--", lw=0.6, alpha=0.7)

        ax3.plot(times, rl_sim["err_x"], color="#00ffcc", lw=1.8,
                 label=f"RL Error $e_{{x,h}}$ (FOV: {rl_met['fov_retention_pct']:.1f}%)")
        ax3.plot(times, pid_sim["err_x"], color="#ffaa00", lw=1.4, ls=":",
                 label=f"PID Error $e_{{x,h}}$ (FOV: {pid_met['fov_retention_pct']:.1f}%)")
        ax3.axhspan(-0.25, 0.25, color="#ffffff", alpha=0.06)
        ax3.axhline(0.0, color="#8fa3bf", ls="--", lw=0.8, alpha=0.4)

        ax3.set_xlim(times[0], times[-1])
        ax3.set_ylim(-1.05, 1.05)
        ax3.set_xlabel("Time (s)", color="#8fa3bf", fontsize=8.5)
        ax3.set_ylabel("Normalized Azimuth Error", color="#8fa3bf", fontsize=8.5)
        ax3.tick_params(colors="#8fa3bf", labelsize=8)
        ax3.set_title(f"{name}: Camera Sightline Lock", color="#e0e8f5", fontsize=10, fontweight="bold")
        ax3.legend(loc="upper right", facecolor="#182232", edgecolor="#223348", labelcolor="#e0e8f5", fontsize=7.5)

    plt.subplots_adjust(hspace=0.35, wspace=0.22, top=0.95, bottom=0.05, left=0.06, right=0.96)
    fig.suptitle("Comparative Evaluation Across Benchmark Tactical Flight Encounters",
                 color="#ffffff", fontsize=15, fontweight="bold", y=0.985)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Generated scenario overview comparison: {output_path}")


def run_flight_inspections():
    """Executes the full evaluation pipeline and renders figures."""
    model_path = PROJECT_ROOT / "control" / "weights" / "recurrent_ppo" / "best_model" / "best_model.zip"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "control" / "weights" / "recurrent_ppo" / "recurrent_ppo_drone_final.zip"

    print(f"\n{'='*78}")
    print(f"  AUTONOMOUS DRONE FLIGHT INSPECTION SUITE")
    print(f"  Policy: {model_path.name}")
    print(f"{'='*78}\n")

    eval_dir = PROJECT_ROOT / "outputs" / "evaluation" / "flight_inspection"
    eval_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        ("evasive", "Stochastic Evasive Target", 25.0, 42),
        ("break_turn", "Tactical Break-Turn & Dive", 20.0, 10),
        ("figure8", "3D Aerobatic Figure-8 Loop", 22.0, 7),
    ]

    all_results = []
    summary_table = []

    for prof, title, dur, seed in scenarios:
        print(f"[SIMULATING] Scenario: {title} (Profile: {prof}, Duration: {dur}s, Seed: {seed})...")
        traj_fn, _ = get_target_trajectory(prof, duration=dur, seed=seed)

        # 1. Simulate Recurrent PPO
        rl_sim = simulate_pursuit(
            trajectory_fn=traj_fn,
            duration=dur,
            dt=0.02,
            controller_type="recurrent_ppo",
            model_path=model_path,
        )
        rl_metrics = compute_flight_metrics(rl_sim)

        # 2. Simulate Classical PID Baseline
        pid_sim = simulate_pursuit(
            trajectory_fn=traj_fn,
            duration=dur,
            dt=0.02,
            controller_type="pid",
        )
        pid_metrics = compute_flight_metrics(pid_sim)

        all_results.append((title, rl_sim, pid_sim, rl_metrics, pid_metrics))

        # Render single scenario in-depth dashboard
        single_fig_path = eval_dir / f"inspection_{prof}.png"
        plot_single_run_inspection(rl_sim, pid_sim, title, single_fig_path)

        # Copy to artifact directory for instant viewing
        if ARTIFACT_DIR.exists():
            shutil.copy(single_fig_path, ARTIFACT_DIR / f"inspection_{prof}.png")

        summary_table.append({
            "scenario": title,
            "rl_fov_pct": rl_metrics["fov_retention_pct"],
            "pid_fov_pct": pid_metrics["fov_retention_pct"],
            "rl_corridor_pct": rl_metrics["corridor_retention_pct"],
            "pid_corridor_pct": pid_metrics["corridor_retention_pct"],
            "rl_mean_dist": rl_metrics["mean_distance_m"],
            "pid_mean_dist": pid_metrics["mean_distance_m"],
            "rl_min_dist": rl_metrics["min_distance_m"],
            "pid_min_dist": pid_metrics["min_distance_m"],
            "rl_jerk": rl_metrics["control_jerk_rms"],
            "pid_jerk": pid_metrics["control_jerk_rms"],
        })

    # Render unified multi-scenario comparative figure
    overview_path = eval_dir / "scenario_overview_comparison.png"
    plot_scenario_overview_comparison(all_results, overview_path)
    if ARTIFACT_DIR.exists():
        shutil.copy(overview_path, ARTIFACT_DIR / "scenario_overview_comparison.png")

    # Print summary performance table
    print("\n" + "="*95)
    print(f"{'SCENARIO':<30} | {'FOV LOCK %':<15} | {'STANDOFF (m)':<15} | {'MIN DIST (m)':<13} | {'JERK (m/s³)':<10}")
    print(f"{'':<30} | {'RL vs PID':<15} | {'RL vs PID':<15} | {'RL vs PID':<13} | {'RL vs PID':<10}")
    print("="*95)
    for r in summary_table:
        print(f"{r['scenario']:<30} | "
              f"{r['rl_fov_pct']:4.1f}% vs {r['pid_fov_pct']:4.1f}% | "
              f"{r['rl_mean_dist']:4.1f}m vs {r['pid_mean_dist']:4.1f}m | "
              f"{r['rl_min_dist']:4.1f}m vs {r['pid_min_dist']:4.1f}m | "
              f"{r['rl_jerk']:4.1f} vs {r['pid_jerk']:4.1f}")
    print("="*95 + "\n")

    return all_results, summary_table


if __name__ == "__main__":
    run_flight_inspections()
