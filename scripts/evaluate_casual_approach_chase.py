#!/usr/bin/env python3
"""
Casual Target Wandering & Approaching Pursuit Evaluation.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Evaluates autonomous standoff distance regulation when the target drone moves casually
(mean speed ~ 1.0 - 1.5 m/s) with 3D meandering and periods where the target actively
turns and moves *towards* the chaser:
1. Single-run time-series diagnostics (Distance, Velocities, Trajectory, Bounding Box).
2. 50-seed Monte Carlo suite comparing:
   - Tuned Kinematic PID Baseline (Standard reverse limit: -0.5 m/s)
   - Tuned Kinematic PID with Active Reverse (Expanded reverse limit: -2.0 m/s)
   - Residual RL Champion (50k checkpoint)
3. High-definition broadcast video rendering showing casual wander & retreat holding.
4. Comprehensive markdown analysis report.
"""

import sys
import os
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
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
from control.predictive_controller import ResidualVisualServoController
from control.rendering.render_pid_chase_bright import render_pid_bright_video


class CasualWanderApproachTargetOracle:
    """
    Target oracle that wanders casually around in 3D with periodic approaches toward the chaser.
    - Speed magnitude: ~0.8 - 1.5 m/s (casual transit, gentle patrol).
    - Longitudinal (X): Sweeps back and forth relative to chaser, with periods of active closure (dx/dt < 0).
    - Lateral (Y): Sweeps across a wide +- 2.5m corridor.
    - Vertical (Z): Gentle altitude variations between -2.1m and -2.8m.
    - Continuous C^2 analytical functions guaranteeing exact state vectors.
    """
    def __init__(
        self,
        initial_pos: np.ndarray = np.array([6.0, 0.0, -2.5], dtype=np.float64),
        seed: int = 42,
    ):
        self.initial_pos = np.array(initial_pos, dtype=np.float64)
        rng = np.random.default_rng(seed)

        # Longitudinal X harmonics: target begins at x=6m, advances forward, then turns towards chaser
        self.w_x1 = 0.35 * rng.uniform(0.9, 1.1)
        self.w_x2 = 0.70 * rng.uniform(0.9, 1.1)
        self.phi_x1 = rng.uniform(0.0, 2.0 * np.pi)
        self.phi_x2 = rng.uniform(0.0, 2.0 * np.pi)
        self.A_x1 = 1.8
        self.A_x2 = 0.6
        self.x0_offset = self.A_x1 * np.sin(self.phi_x1) + self.A_x2 * np.sin(self.phi_x2)

        # Lateral Y harmonics: sweeping wander across +- 2.5m
        self.w_y = 0.30 * rng.uniform(0.9, 1.1)
        self.phi_y = rng.uniform(0.0, 2.0 * np.pi)
        self.A_y = 2.5
        self.y0_offset = self.A_y * np.sin(self.phi_y)

        # Vertical Z harmonics: gentle breathing between -2.2m and -2.8m
        self.w_z = 0.45 * rng.uniform(0.9, 1.1)
        self.phi_z = rng.uniform(0.0, 2.0 * np.pi)
        self.A_z = 0.35
        self.z0_offset = self.A_z * np.sin(self.phi_z)

    def get_position(self, t: float) -> np.ndarray:
        dx = (self.A_x1 * np.sin(self.w_x1 * t + self.phi_x1) +
              self.A_x2 * np.sin(self.w_x2 * t + self.phi_x2) - self.x0_offset)
        dy = (self.A_y * np.sin(self.w_y * t + self.phi_y) - self.y0_offset)
        dz = (self.A_z * np.sin(self.w_z * t + self.phi_z) - self.z0_offset)
        return self.initial_pos + np.array([dx, dy, dz], dtype=np.float64)

    def get_velocity(self, t: float) -> np.ndarray:
        vx = (self.A_x1 * self.w_x1 * np.cos(self.w_x1 * t + self.phi_x1) +
              self.A_x2 * self.w_x2 * np.cos(self.w_x2 * t + self.phi_x2))
        vy = (self.A_y * self.w_y * np.cos(self.w_y * t + self.phi_y))
        vz = (self.A_z * self.w_z * np.cos(self.w_z * t + self.phi_z))
        return np.array([vx, vy, vz], dtype=np.float64)

    def get_acceleration(self, t: float) -> np.ndarray:
        ax = -(self.A_x1 * (self.w_x1 ** 2) * np.sin(self.w_x1 * t + self.phi_x1) +
               self.A_x2 * (self.w_x2 ** 2) * np.sin(self.w_x2 * t + self.phi_x2))
        ay = -(self.A_y * (self.w_y ** 2) * np.sin(self.w_y * t + self.phi_y))
        az = -(self.A_z * (self.w_z ** 2) * np.sin(self.w_z * t + self.phi_z))
        return np.array([ax, ay, az], dtype=np.float64)

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.get_position(t), self.get_velocity(t), self.get_acceleration(t)


class CasualApproachScenario:
    """PursuitScenario-compatible container for casual wander and approaching target."""
    def __init__(
        self,
        scenario_id: str,
        initial_distance: float = 6.0,
        initial_chaser_speed: float = 0.0,
        duration: float = 25.0,
        initial_alt: float = -2.5,
        seed: int = 42,
    ):
        self.scenario_id = scenario_id
        self.profile = "casual_wander_approach"
        self.encounter_type = "tail_chase"
        self.duration = float(duration)
        self.seed = int(seed)
        self.initial_distance = float(initial_distance)
        self.initial_azimuth_deg = 0.0
        self.initial_elevation_deg = 0.0
        self.initial_target_pos = [float(initial_distance), 0.0, float(initial_alt)]
        self.initial_target_heading_rad = 0.0
        self.initial_chaser_pos = [0.0, 0.0, float(initial_alt)]
        self.initial_chaser_yaw = 0.0
        self.target_speed = 1.2
        self.initial_chaser_speed = float(initial_chaser_speed)

    def create_target_oracle(self):
        return CasualWanderApproachTargetOracle(
            initial_pos=np.array(self.initial_target_pos, dtype=np.float64),
            seed=self.seed
        )


def compute_approach_metrics(
    history: Dict[str, Any],
    t_settle_start: float = 8.0,
    standoff_nominal: float = 3.5,
) -> Dict[str, float]:
    """Computes precision metrics for casual pursuit with closing target motion."""
    times = np.array(history["times"])
    c_pos = np.array(history["chaser_pos"])
    t_pos = np.array(history["target_pos"])
    c_vel = np.array(history["chaser_vel"])

    dists = np.linalg.norm(c_pos - t_pos, axis=1)
    chaser_speeds = np.linalg.norm(c_vel[:, :2], axis=1)

    # Steady-state window (t >= t_settle_start)
    ss_mask = times >= t_settle_start
    if not np.any(ss_mask):
        ss_mask = np.ones_like(times, dtype=bool)

    ss_dists = dists[ss_mask]
    ss_speeds = chaser_speeds[ss_mask]

    mean_dist = float(np.mean(ss_dists))
    std_dist = float(np.std(ss_dists))
    mae_dist = float(np.mean(np.abs(ss_dists - standoff_nominal)))
    max_dev = float(np.max(np.abs(ss_dists - standoff_nominal)))
    min_dist_overall = float(np.min(dists))

    # Standoff corridor retention in steady state [2.5, 4.5]m
    basket_retention_pct = float(np.sum((ss_dists >= 2.5) & (ss_dists <= 4.5)) / len(ss_dists) * 100.0)

    # Velocity ripple (std of forward velocity)
    vx_ss = c_vel[ss_mask, 0]
    vel_ripple_rms = float(np.std(vx_ss))

    return {
        "ss_mean_dist_m": mean_dist,
        "ss_std_dist_m": std_dist,
        "ss_mae_dist_m": mae_dist,
        "ss_max_deviation_m": max_dev,
        "ss_vel_ripple_rms": vel_ripple_rms,
        "ss_basket_retention_pct": basket_retention_pct,
        "min_dist_overall_m": min_dist_overall,
        "max_dist_overall_m": float(np.max(dists)),
    }


def plot_casual_approach_diagnostics(
    res_pid_std: Dict[str, Any],
    res_pid_rev: Dict[str, Any],
    res_rl: Optional[Dict[str, Any]],
    output_path: Path,
    standoff_nominal: float = 3.5,
):
    """Generates 4-panel multi-controller diagnostic comparison plots."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), dpi=150)
    plt.subplots_adjust(wspace=0.22, hspace=0.38, left=0.07, right=0.96, top=0.93, bottom=0.06)

    h_std = res_pid_std["history"]
    h_rev = res_pid_rev["history"]
    t = np.array(h_std["times"])

    c_pos_std = np.array(h_std["chaser_pos"])
    t_pos = np.array(h_std["target_pos"])
    c_vel_std = np.array(h_std["chaser_vel"])
    dists_std = np.linalg.norm(c_pos_std - t_pos, axis=1)

    c_pos_rev = np.array(h_rev["chaser_pos"])
    c_vel_rev = np.array(h_rev["chaser_vel"])
    dists_rev = np.linalg.norm(c_pos_rev - t_pos, axis=1)

    # Column 0: Standard PID (Reverse limit -0.5 m/s)
    # 1. Standoff Distance
    ax1 = axes[0, 0]
    ax1.axhspan(2.5, 4.5, color="#10B981", alpha=0.15, label="Firing Basket [2.5, 4.5] m")
    ax1.axhline(standoff_nominal, color="#059669", linestyle="--", linewidth=1.8, label=f"Nominal {standoff_nominal} m")
    ax1.axhline(0.5, color="#DC2626", linestyle=":", linewidth=1.5, label="Collision Threshold (0.5m)")
    ax1.plot(t, dists_std, color="#2563EB", linewidth=2.2, label="Distance (Std Reverse -0.5m/s)")
    ax1.set_title("Standard PID Baseline (Reverse Limit: -0.5 m/s)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Distance (m)", fontsize=10, fontweight="bold")
    ax1.set_ylim(0.0, 7.5)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

    # 2. Forward Velocity
    ax2 = axes[1, 0]
    ax2.plot(t, c_vel_std[:, 0], color="#2563EB", linewidth=2.0, label="Chaser Surge vx")
    ax2.axhline(0.0, color="#64748B", linestyle="-", linewidth=0.8, alpha=0.7)
    ax2.axhline(-0.5, color="#DC2626", linestyle="--", linewidth=1.2, label="Reverse Cap (-0.5 m/s)")
    ax2.set_ylabel("Surge Vel vx (m/s)", fontsize=10, fontweight="bold")
    ax2.set_ylim(-1.5, 4.5)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

    # 3. 2D Flight Trajectory (X vs Y)
    ax3 = axes[2, 0]
    ax3.plot(t_pos[:, 0], t_pos[:, 1], color="#DC2626", linewidth=2.5, label="Target Path (Approaching)")
    ax3.plot(c_pos_std[:, 0], c_pos_std[:, 1], color="#2563EB", linewidth=2.0, linestyle="--", label="Chaser Path")
    ax3.scatter([t_pos[0, 0]], [t_pos[0, 1]], color="#DC2626", marker="o", s=60, label="Tgt Start")
    ax3.scatter([c_pos_std[0, 0]], [c_pos_std[0, 1]], color="#2563EB", marker="s", s=60, label="Chaser Start")
    ax3.set_ylabel("East / Y (m)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("North / X (m)", fontsize=10, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

    # 4. Bounding Box Width
    ax4 = axes[3, 0]
    telem_std = h_std.get("chaser_telem", [])
    bw_std = [item.get("bbox_w", 0.0) for item in telem_std]
    ax4.axhline(110.0, color="#059669", linestyle="--", linewidth=1.8, label="Desired w* (110 px)")
    ax4.plot(t, bw_std, color="#7C3AED", linewidth=1.8, label="Estimated BBox Width")
    ax4.set_ylabel("BBox Width (px)", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Time (s)", fontsize=10, fontweight="bold")
    ax4.set_ylim(0.0, 220.0)
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="lower right", fontsize=8.0, framealpha=0.9)

    # Column 1: Active Reverse PID (Reverse limit -2.0 m/s) with Residual RL overlay
    # 1. Standoff Distance Comparison
    ax5 = axes[0, 1]
    ax5.axhspan(2.5, 4.5, color="#10B981", alpha=0.15, label="Firing Basket [2.5, 4.5] m")
    ax5.axhline(standoff_nominal, color="#059669", linestyle="--", linewidth=1.8, label=f"Nominal {standoff_nominal} m")
    ax5.axhline(0.5, color="#DC2626", linestyle=":", linewidth=1.5, label="Collision Threshold (0.5m)")
    ax5.plot(t, dists_rev, color="#059669", linewidth=2.2, label="Distance (Active Reverse -2.0m/s)")
    if res_rl is not None:
        h_rl = res_rl["history"]
        c_pos_rl = np.array(h_rl["chaser_pos"])
        dists_rl = np.linalg.norm(c_pos_rl - t_pos, axis=1)
        ax5.plot(t, dists_rl, color="#D97706", linewidth=2.0, linestyle="-.", label="Distance (Residual RL 50k)")
    ax5.set_title("Active Reverse PID & Residual RL Comparison", fontsize=11, fontweight="bold")
    ax5.set_ylabel("Distance (m)", fontsize=10, fontweight="bold")
    ax5.set_ylim(0.0, 7.5)
    ax5.grid(True, linestyle=":", alpha=0.6)
    ax5.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

    # 2. Forward Velocity
    ax6 = axes[1, 1]
    ax6.plot(t, c_vel_rev[:, 0], color="#059669", linewidth=2.0, label="Chaser Surge vx (Active Rev)")
    if res_rl is not None:
        c_vel_rl = np.array(res_rl["history"]["chaser_vel"])
        ax6.plot(t, c_vel_rl[:, 0], color="#D97706", linewidth=1.8, linestyle="-.", label="Chaser Surge vx (RL)")
    ax6.axhline(0.0, color="#64748B", linestyle="-", linewidth=0.8, alpha=0.7)
    ax6.axhline(-2.0, color="#059669", linestyle="--", linewidth=1.2, label="Active Reverse Cap (-2.0 m/s)")
    ax6.set_ylabel("Surge Vel vx (m/s)", fontsize=10, fontweight="bold")
    ax6.set_ylim(-2.5, 4.5)
    ax6.grid(True, linestyle=":", alpha=0.6)
    ax6.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

    # 3. 2D Flight Trajectory
    ax7 = axes[2, 1]
    ax7.plot(t_pos[:, 0], t_pos[:, 1], color="#DC2626", linewidth=2.5, label="Target Path")
    ax7.plot(c_pos_rev[:, 0], c_pos_rev[:, 1], color="#059669", linewidth=2.0, linestyle="--", label="Chaser (Active Rev)")
    if res_rl is not None:
        ax7.plot(c_pos_rl[:, 0], c_pos_rl[:, 1], color="#D97706", linewidth=1.8, linestyle=":", label="Chaser (Residual RL)")
    ax7.set_ylabel("East / Y (m)", fontsize=10, fontweight="bold")
    ax7.set_xlabel("North / X (m)", fontsize=10, fontweight="bold")
    ax7.grid(True, linestyle=":", alpha=0.6)
    ax7.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

    # 4. Bounding Box Width
    ax8 = axes[3, 1]
    telem_rev = h_rev.get("chaser_telem", [])
    bw_rev = [item.get("bbox_w", 0.0) for item in telem_rev]
    ax8.axhline(110.0, color="#059669", linestyle="--", linewidth=1.8, label="Desired w* (110 px)")
    ax8.plot(t, bw_rev, color="#059669", linewidth=1.8, label="BBox (Active Rev)")
    if res_rl is not None:
        telem_rl = res_rl["history"].get("chaser_telem", [])
        bw_rl = [item.get("bbox_w", 0.0) for item in telem_rl]
        ax8.plot(t, bw_rl, color="#D97706", linewidth=1.5, linestyle="-.", label="BBox (RL)")
    ax8.set_ylabel("BBox Width (px)", fontsize=10, fontweight="bold")
    ax8.set_xlabel("Time (s)", fontsize=10, fontweight="bold")
    ax8.set_ylim(0.0, 220.0)
    ax8.grid(True, linestyle=":", alpha=0.6)
    ax8.legend(loc="lower right", fontsize=8.0, framealpha=0.9)

    fig.suptitle(
        "Autonomous Chaser Distance-Holding Performance: Casual Wandering & Approaching Target\n"
        "Handling Target Heading Reversals via Optical Looming Deceleration & Active Reverse Cushion",
        fontsize=14, fontweight="bold", color="#0F172A", y=0.98
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved casual approach diagnostic figure to: {output_path}")


def run_evaluation(num_seeds: int = 50, render_video: bool = True) -> Dict[str, Any]:
    print("=" * 78)
    print("  AUTONOMOUS DRONE PURSUIT: CASUAL WANDER & APPROACH EVALUATION")
    print("=" * 78)

    # 1. Single Run (Seed 42) for all 3 controllers
    scen_s42 = CasualApproachScenario(scenario_id="casual_approach_s42", seed=42)

    # A. Standard PID (-0.5 m/s rev limit)
    pid_std = create_pid_controller(standoff_nominal=3.5)
    res_pid_std = evaluate_single_run(
        scenario=scen_s42,
        controller=pid_std,
        controller_type="pid",
        noise_mode="empirical",
        frame_drop_rate=0.05,
        standoff_nominal=3.5,
        use_kalman=True,
        return_history=True,
        enable_ego_compensation=True
    )
    m_std_s42 = compute_approach_metrics(res_pid_std["history"])

    # B. Active Reverse PID (-2.0 m/s rev limit)
    pid_rev = create_pid_controller(standoff_nominal=3.5)
    pid_rev.min_limits[0] = -2.0
    res_pid_rev = evaluate_single_run(
        scenario=scen_s42,
        controller=pid_rev,
        controller_type="pid",
        noise_mode="empirical",
        frame_drop_rate=0.05,
        standoff_nominal=3.5,
        use_kalman=True,
        return_history=True,
        enable_ego_compensation=True
    )
    m_rev_s42 = compute_approach_metrics(res_pid_rev["history"])

    # C. Residual RL Champion (50k)
    p_champ = PROJECT_ROOT / "outputs/models/residual_ppo_v1/checkpoints/residual_ppo_drone_50000_steps.zip"
    rl_controller = None
    res_rl = None
    m_rl_s42 = None
    if p_champ.exists():
        rl_controller = ResidualVisualServoController(
            model_path=str(p_champ),
            pid_gains_path="control/weights/tuned_pid_gains.json",
            standoff_nominal=3.5
        )
        res_rl = evaluate_single_run(
            scenario=scen_s42,
            controller=rl_controller,
            controller_type="residual",
            noise_mode="empirical",
            frame_drop_rate=0.05,
            standoff_nominal=3.5,
            use_kalman=True,
            return_history=True,
            enable_ego_compensation=True
        )
        m_rl_s42 = compute_approach_metrics(res_rl["history"])

    # 2. Save Diagnostic Figure
    plot_path = PROJECT_ROOT / "outputs/benchmarks/control/casual_approach_distance_holding.png"
    plot_casual_approach_diagnostics(
        res_pid_std=res_pid_std,
        res_pid_rev=res_pid_rev,
        res_rl=res_rl,
        output_path=plot_path
    )

    artifact_dir = Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/7dc8da15-984b-4499-8ca6-5e32efd45198")
    if artifact_dir.exists():
        import shutil
        shutil.copy2(plot_path, artifact_dir / "casual_approach_distance_holding.png")

    # 3. 50-Seed Monte Carlo Suite across all 3 controllers
    print(f"\n[MONTE CARLO] Evaluating {num_seeds} randomized casual wander & approach trajectories...")
    mc_std, mc_rev, mc_rl = [], [], []

    for seed in range(200, 200 + num_seeds):
        scen_mc = CasualApproachScenario(f"casual_approach_s{seed}", seed=seed)

        # A. PID Standard
        p_s = create_pid_controller(standoff_nominal=3.5)
        r_s = evaluate_single_run(
            scenario=scen_mc,
            controller=p_s,
            controller_type="pid",
            noise_mode="empirical",
            frame_drop_rate=0.05,
            standoff_nominal=3.5,
            use_kalman=True,
            return_history=True,
            enable_ego_compensation=True
        )
        m_s = compute_approach_metrics(r_s["history"])
        m_s["success_1s"] = r_s["success_1s"]
        m_s["success_3s"] = r_s["success_3s"]
        m_s["dwell_time"] = r_s["max_continuous_dwell_s"]
        m_s["firing_time"] = r_s["firing_window_s"]
        m_s["fov_retention"] = r_s["fov_lock_pct"]
        m_s["collision"] = r_s["collision_breach"]
        mc_std.append(m_s)

        # B. PID Active Reverse
        p_r = create_pid_controller(standoff_nominal=3.5)
        p_r.min_limits[0] = -2.0
        r_r = evaluate_single_run(
            scenario=scen_mc,
            controller=p_r,
            controller_type="pid",
            noise_mode="empirical",
            frame_drop_rate=0.05,
            standoff_nominal=3.5,
            use_kalman=True,
            return_history=True,
            enable_ego_compensation=True
        )
        m_r = compute_approach_metrics(r_r["history"])
        m_r["success_1s"] = r_r["success_1s"]
        m_r["success_3s"] = r_r["success_3s"]
        m_r["dwell_time"] = r_r["max_continuous_dwell_s"]
        m_r["firing_time"] = r_r["firing_window_s"]
        m_r["fov_retention"] = r_r["fov_lock_pct"]
        m_r["collision"] = r_r["collision_breach"]
        mc_rev.append(m_r)

        # C. Residual RL
        if rl_controller is not None:
            rl_controller.reset()
            r_l = evaluate_single_run(
                scenario=scen_mc,
                controller=rl_controller,
                controller_type="residual",
                noise_mode="empirical",
                frame_drop_rate=0.05,
                standoff_nominal=3.5,
                use_kalman=True,
                return_history=True,
                enable_ego_compensation=True
            )
            m_l = compute_approach_metrics(r_l["history"])
            m_l["success_1s"] = r_l["success_1s"]
            m_l["success_3s"] = r_l["success_3s"]
            m_l["dwell_time"] = r_l["max_continuous_dwell_s"]
            m_l["firing_time"] = r_l["firing_window_s"]
            m_l["fov_retention"] = r_l["fov_lock_pct"]
            m_l["collision"] = r_l["collision_breach"]
            mc_rl.append(m_l)

    def aggregate_mc(results_list: List[Dict[str, Any]]) -> Dict[str, float]:
        return {
            "success_1s_pct": float(np.mean([r["success_1s"] for r in results_list]) * 100.0),
            "success_3s_pct": float(np.mean([r["success_3s"] for r in results_list]) * 100.0),
            "dwell_time_mean_s": float(np.mean([r["dwell_time"] for r in results_list])),
            "firing_time_mean_s": float(np.mean([r["firing_time"] for r in results_list])),
            "fov_retention_pct": float(np.mean([r["fov_retention"] for r in results_list])),
            "collision_rate_pct": float(np.mean([r["collision"] for r in results_list]) * 100.0),
            "ss_mean_dist_m": float(np.mean([r["ss_mean_dist_m"] for r in results_list])),
            "ss_std_dist_m": float(np.mean([r["ss_std_dist_m"] for r in results_list])),
            "ss_mae_dist_m": float(np.mean([r["ss_mae_dist_m"] for r in results_list])),
            "ss_max_deviation_m": float(np.mean([r["ss_max_deviation_m"] for r in results_list])),
            "ss_basket_retention_pct": float(np.mean([r["ss_basket_retention_pct"] for r in results_list])),
            "min_dist_overall_m": float(np.mean([r["min_dist_overall_m"] for r in results_list])),
        }

    agg_std = aggregate_mc(mc_std)
    agg_rev = aggregate_mc(mc_rev)
    agg_rl = aggregate_mc(mc_rl) if mc_rl else None

    # 4. Render Video
    video_path_str = None
    if render_video:
        vid_out = PROJECT_ROOT / "outputs/benchmarks/videos/casual_approach_chase_bright.mp4"
        vid_out.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[VIDEO] Rendering broadcast video of casual approach chase to: {vid_out}")
        pid_render = create_pid_controller(standoff_nominal=3.5)
        pid_render.min_limits[0] = -2.0
        render_pid_bright_video(
            scenario=scen_s42,
            pid_controller=pid_render,
            output_path=vid_out,
            fps=25
        )
        video_path_str = str(vid_out)
        if artifact_dir.exists():
            import shutil
            shutil.copy2(vid_out, artifact_dir / "casual_approach_chase_bright.mp4")

    # 5. Print Results
    print("\n" + "=" * 78)
    print("  EXACT CASUAL WANDER & APPROACH DISTANCE-HOLDING RESULTS (50 SEEDS)")
    print("=" * 78)
    print(f"  Metric                             | PID Standard (-0.5) | PID Active Rev (-2.0) | Residual RL (50k)   ")
    print(f"  -----------------------------------|---------------------|-----------------------|---------------------")
    print(f"  1s Catch Rate (Basket Entry)       | {agg_std['success_1s_pct']:>19.2f}% | {agg_rev['success_1s_pct']:>21.2f}% | {agg_rl['success_1s_pct'] if agg_rl else 0.0:>19.2f}%")
    print(f"  3s Sustained Lock Rate             | {agg_std['success_3s_pct']:>19.2f}% | {agg_rev['success_3s_pct']:>21.2f}% | {agg_rl['success_3s_pct'] if agg_rl else 0.0:>19.2f}%")
    print(f"  Max Continuous Dwell Time (s)      | {agg_std['dwell_time_mean_s']:>19.2f}s | {agg_rev['dwell_time_mean_s']:>21.2f}s | {agg_rl['dwell_time_mean_s'] if agg_rl else 0.0:>19.2f}s")
    print(f"  Cumulative Firing Window (s)       | {agg_std['firing_time_mean_s']:>19.2f}s | {agg_rev['firing_time_mean_s']:>21.2f}s | {agg_rl['firing_time_mean_s'] if agg_rl else 0.0:>19.2f}s")
    print(f"  Camera FOV Retention Rate          | {agg_std['fov_retention_pct']:>19.2f}% | {agg_rev['fov_retention_pct']:>21.2f}% | {agg_rl['fov_retention_pct'] if agg_rl else 0.0:>19.2f}%")
    print(f"  Steady-State Mean Distance         | {agg_std['ss_mean_dist_m']:>19.3f}m | {agg_rev['ss_mean_dist_m']:>21.3f}m | {agg_rl['ss_mean_dist_m'] if agg_rl else 0.0:>19.3f}m")
    print(f"  Steady-State Distance MAE          | {agg_std['ss_mae_dist_m']:>19.3f}m | {agg_rev['ss_mae_dist_m']:>21.3f}m | {agg_rl['ss_mae_dist_m'] if agg_rl else 0.0:>19.3f}m")
    print(f"  Steady-State Distance Std          | {agg_std['ss_std_dist_m']:>19.3f}m | {agg_rev['ss_std_dist_m']:>21.3f}m | {agg_rl['ss_std_dist_m'] if agg_rl else 0.0:>19.3f}m")
    print(f"  SS Basket Retention ([2.5, 4.5]m)  | {agg_std['ss_basket_retention_pct']:>19.2f}% | {agg_rev['ss_basket_retention_pct']:>21.2f}% | {agg_rl['ss_basket_retention_pct'] if agg_rl else 0.0:>19.2f}%")
    print(f"  Mean Minimum Distance Overall      | {agg_std['min_dist_overall_m']:>19.3f}m | {agg_rev['min_dist_overall_m']:>21.3f}m | {agg_rl['min_dist_overall_m'] if agg_rl else 0.0:>19.3f}m")
    print(f"  Collision Breach Rate (< 0.5m)     | {agg_std['collision_rate_pct']:>19.2f}% | {agg_rev['collision_rate_pct']:>21.2f}% | {agg_rl['collision_rate_pct'] if agg_rl else 0.0:>19.2f}%")

    out_json = PROJECT_ROOT / "outputs/benchmarks/control/casual_approach_evaluation_results.json"
    data_to_save = {
        "single_run_s42": {
            "std": m_std_s42,
            "rev": m_rev_s42,
            "rl": m_rl_s42
        },
        "monte_carlo_std": agg_std,
        "monte_carlo_rev": agg_rev,
        "monte_carlo_rl": agg_rl,
        "plot_path": str(plot_path),
        "video_path": video_path_str
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2)
    print(f"\n[DATA] Saved full casual approach evaluation data to: {out_json}")

    return data_to_save


if __name__ == "__main__":
    run_evaluation(num_seeds=50, render_video=True)
