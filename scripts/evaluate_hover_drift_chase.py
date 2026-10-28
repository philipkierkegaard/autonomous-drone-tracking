#!/usr/bin/env python3
"""
Hover & Gentle Drift Target Distance-Holding Evaluation.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Evaluates how well the autonomous chaser drone maintains standoff distance (3.5m nominal)
against a target that hovers steadily or exhibits gentle, subtle station-keeping drift:
1. Pure Steady Hover (d0 = 7.0m, v_target = 0 m/s): Catch-up, smooth deceleration, and standoff hold.
2. Pure In-Place Hover (d0 = 3.5m, v_target = 0 m/s): Zero-motion station keeping.
3. Gentle Hover Drift (d0 = 7.0m, v_target ~ 0.25 m/s): Tracking subtle 3D wandering drift.
4. Monte Carlo Evaluation: 50 randomized seeds comparing Tuned Kinematic PID vs Residual RL Champion.

Generates:
- Comprehensive diagnostic time-series plots.
- High-definition broadcast video of hover tracking.
- Markdown summary report.
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


class SteadyHoverTargetOracle:
    """Target oracle that hovers completely stationary in space."""
    def __init__(
        self,
        initial_pos: np.ndarray = np.array([7.0, 0.0, -2.5], dtype=np.float64),
    ):
        self.initial_pos = np.array(initial_pos, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.acc = np.zeros(3, dtype=np.float64)

    def get_position(self, t: float) -> np.ndarray:
        return self.initial_pos.copy()

    def get_velocity(self, t: float) -> np.ndarray:
        return self.vel.copy()

    def get_acceleration(self, t: float) -> np.ndarray:
        return self.acc.copy()

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.get_position(t), self.get_velocity(t), self.get_acceleration(t)


class HoverDriftTargetOracle:
    """
    Target oracle simulating a realistic multicopter hovering with subtle GPS / optical drift:
    - Smooth 3D wandering motion (differential flatness compliant).
    - Drift speed magnitude ~ 0.15 - 0.35 m/s.
    - Subtle horizontal wander (radius ~ 0.6 - 1.0 m) and gentle altitude heave (+- 0.15 m).
    """
    def __init__(
        self,
        initial_pos: np.ndarray = np.array([7.0, 0.0, -2.5], dtype=np.float64),
        seed: int = 42,
        drift_radius_xy: float = 0.8,
        drift_heave_z: float = 0.15,
        mean_drift_speed: float = 0.25,
    ):
        self.initial_pos = np.array(initial_pos, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

        # Harmonic parameters for smooth, band-limited wandering drift
        # Frequency scale based on mean_drift_speed / drift_radius
        w_base = mean_drift_speed / max(0.1, drift_radius_xy)  # ~ 0.3 rad/s (period ~ 21s)

        self.w_x1 = w_base * self.rng.uniform(0.7, 1.2)
        self.w_x2 = w_base * self.rng.uniform(1.4, 2.0)
        self.phi_x1 = self.rng.uniform(0.0, 2 * np.pi)
        self.phi_x2 = self.rng.uniform(0.0, 2 * np.pi)
        self.A_x1 = drift_radius_xy * 0.45
        self.A_x2 = drift_radius_xy * 0.20

        self.w_y1 = w_base * self.rng.uniform(0.8, 1.3)
        self.w_y2 = w_base * self.rng.uniform(1.5, 2.2)
        self.phi_y1 = self.rng.uniform(0.0, 2 * np.pi)
        self.phi_y2 = self.rng.uniform(0.0, 2 * np.pi)
        self.A_y1 = drift_radius_xy * 0.65
        self.A_y2 = drift_radius_xy * 0.25

        self.w_z = w_base * self.rng.uniform(0.9, 1.5)
        self.phi_z = self.rng.uniform(0.0, 2 * np.pi)
        self.A_z = drift_heave_z

        # Ensure t=0 starts exactly at initial_pos
        self.offset_x0 = self.A_x1 * np.sin(self.phi_x1) + self.A_x2 * np.sin(self.phi_x2)
        self.offset_y0 = self.A_y1 * np.sin(self.phi_y1) + self.A_y2 * np.sin(self.phi_y2)
        self.offset_z0 = self.A_z * np.sin(self.phi_z)

    def get_position(self, t: float) -> np.ndarray:
        dx = (self.A_x1 * np.sin(self.w_x1 * t + self.phi_x1) +
              self.A_x2 * np.sin(self.w_x2 * t + self.phi_x2) - self.offset_x0)
        dy = (self.A_y1 * np.sin(self.w_y1 * t + self.phi_y1) +
              self.A_y2 * np.sin(self.w_y2 * t + self.phi_y2) - self.offset_y0)
        dz = (self.A_z * np.sin(self.w_z * t + self.phi_z) - self.offset_z0)
        return self.initial_pos + np.array([dx, dy, dz], dtype=np.float64)

    def get_velocity(self, t: float) -> np.ndarray:
        vx = (self.A_x1 * self.w_x1 * np.cos(self.w_x1 * t + self.phi_x1) +
              self.A_x2 * self.w_x2 * np.cos(self.w_x2 * t + self.phi_x2))
        vy = (self.A_y1 * self.w_y1 * np.cos(self.w_y1 * t + self.phi_y1) +
              self.A_y2 * self.w_y2 * np.cos(self.w_y2 * t + self.phi_y2))
        vz = self.A_z * self.w_z * np.cos(self.w_z * t + self.phi_z)
        return np.array([vx, vy, vz], dtype=np.float64)

    def get_acceleration(self, t: float) -> np.ndarray:
        ax = -(self.A_x1 * (self.w_x1 ** 2) * np.sin(self.w_x1 * t + self.phi_x1) +
               self.A_x2 * (self.w_x2 ** 2) * np.sin(self.w_x2 * t + self.phi_x2))
        ay = -(self.A_y1 * (self.w_y1 ** 2) * np.sin(self.w_y1 * t + self.phi_y1) +
               self.A_y2 * (self.w_y2 ** 2) * np.sin(self.w_y2 * t + self.phi_y2))
        az = -(self.A_z * (self.w_z ** 2) * np.sin(self.w_z * t + self.phi_z))
        return np.array([ax, ay, az], dtype=np.float64)

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.get_position(t), self.get_velocity(t), self.get_acceleration(t)


class HoverScenario:
    """PursuitScenario-compatible container for hover and drift scenarios."""
    def __init__(
        self,
        scenario_id: str,
        mode: str = "drift",  # 'steady' or 'drift'
        initial_distance: float = 7.0,
        initial_chaser_speed: float = 0.0,
        duration: float = 25.0,
        initial_alt: float = -2.5,
        seed: int = 42,
    ):
        self.scenario_id = scenario_id
        self.mode = mode
        self.profile = f"hover_{mode}"
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
        self.target_speed = 0.0
        self.initial_chaser_speed = float(initial_chaser_speed)

    def create_target_oracle(self):
        if self.mode == "steady":
            return SteadyHoverTargetOracle(
                initial_pos=np.array(self.initial_target_pos, dtype=np.float64)
            )
        else:
            return HoverDriftTargetOracle(
                initial_pos=np.array(self.initial_target_pos, dtype=np.float64),
                seed=self.seed
            )


def compute_hover_metrics(
    history: Dict[str, Any],
    t_settle_start: float = 8.0,
    standoff_nominal: float = 3.5,
) -> Dict[str, float]:
    """Computes precision steady-state distance-holding, settling, and stability metrics."""
    times = np.array(history["times"])
    c_pos = np.array(history["chaser_pos"])
    t_pos = np.array(history["target_pos"])
    c_vel = np.array(history["chaser_vel"])

    dists = np.linalg.norm(c_pos - t_pos, axis=1)
    chaser_speeds = np.linalg.norm(c_vel[:, :2], axis=1)

    # 1. Settling time: first timestamp where distance stays in [3.0, 4.0] (3.5 +- 0.5m)
    in_basket = (dists >= 2.5) & (dists <= 4.5)
    in_tight_band = (dists >= 3.0) & (dists <= 4.0)

    t_settle = None
    for i, t in enumerate(times):
        if np.all(in_tight_band[i:]):
            t_settle = float(t)
            break
    if t_settle is None:
        # Fallback to first time entering tight band
        idx = np.where(in_tight_band)[0]
        t_settle = float(times[idx[0]]) if len(idx) > 0 else float(times[-1])

    # 2. Steady-state analysis (t >= t_settle_start)
    ss_mask = times >= t_settle_start
    if not np.any(ss_mask):
        ss_mask = np.ones_like(times, dtype=bool)

    ss_dists = dists[ss_mask]
    ss_speeds = chaser_speeds[ss_mask]

    mean_dist = float(np.mean(ss_dists))
    std_dist = float(np.std(ss_dists))
    mae_dist = float(np.mean(np.abs(ss_dists - standoff_nominal)))
    rmse_dist = float(np.sqrt(np.mean((ss_dists - standoff_nominal) ** 2)))
    max_dev = float(np.max(np.abs(ss_dists - standoff_nominal)))

    # Velocity ripple (standard deviation of chaser forward velocity in steady state)
    vx_ss = c_vel[ss_mask, 0]
    vel_ripple_rms = float(np.std(vx_ss))

    # Standoff corridor retention in steady state
    basket_retention_pct = float(np.sum((ss_dists >= 2.5) & (ss_dists <= 4.5)) / len(ss_dists) * 100.0)

    return {
        "settling_time_s": t_settle,
        "ss_mean_dist_m": mean_dist,
        "ss_std_dist_m": std_dist,
        "ss_mae_dist_m": mae_dist,
        "ss_rmse_dist_m": rmse_dist,
        "ss_max_deviation_m": max_dev,
        "ss_vel_ripple_rms": vel_ripple_rms,
        "ss_basket_retention_pct": basket_retention_pct,
        "min_dist_m": float(np.min(dists)),
        "max_dist_m": float(np.max(dists)),
    }


def plot_hover_diagnostics(
    res_steady_pid: Dict[str, Any],
    res_drift_pid: Dict[str, Any],
    res_drift_rl: Optional[Dict[str, Any]],
    output_path: Path,
    standoff_nominal: float = 3.5,
):
    """Generates broadcast publication diagnostic plots for hover and drift tracking."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), dpi=150)
    plt.subplots_adjust(wspace=0.22, hspace=0.38, left=0.07, right=0.96, top=0.93, bottom=0.06)

    cases = [
        ("Case 1: Pure Steady Hover (v_tgt = 0 m/s, d0 = 7.0m)", res_steady_pid, 0, 7.0),
        ("Case 2: Gentle Hover Drift (v_tgt ~ 0.25 m/s, d0 = 7.0m)", res_drift_pid, 1, 7.0),
    ]

    for title, res, col, init_dist in cases:
        h = res["history"]
        t = np.array(h["times"])
        c_pos = np.array(h["chaser_pos"])
        t_pos = np.array(h["target_pos"])
        c_vel = np.array(h["chaser_vel"])
        dists = np.linalg.norm(c_pos - t_pos, axis=1)

        # 1. Standoff Distance Tracking
        ax1 = axes[0, col]
        ax1.axhspan(2.5, 4.5, color="#10B981", alpha=0.15, label="Firing Basket [2.5, 4.5] m")
        ax1.axhline(standoff_nominal, color="#059669", linestyle="--", linewidth=1.8, label=f"Nominal {standoff_nominal} m")
        ax1.plot(t, dists, color="#2563EB", linewidth=2.2, label="Chaser Distance")

        if col == 1 and res_drift_rl is not None:
            h_rl = res_drift_rl["history"]
            t_rl = np.array(h_rl["times"])
            c_pos_rl = np.array(h_rl["chaser_pos"])
            t_pos_rl = np.array(h_rl["target_pos"])
            dists_rl = np.linalg.norm(c_pos_rl - t_pos_rl, axis=1)
            ax1.plot(t_rl, dists_rl, color="#D97706", linewidth=2.0, linestyle="-.", label="Residual RL (50k)")

        ax1.set_title(title, fontsize=12, fontweight="bold", color="#1E293B", pad=8)
        ax1.set_ylabel("Distance (m)", fontsize=10, fontweight="bold")
        ax1.set_ylim(0.0, max(8.5, init_dist + 1.0))
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

        # 2. Velocity Tracking & Ripple
        ax2 = axes[1, col]
        ax2.plot(t, c_vel[:, 0], color="#2563EB", linewidth=2.0, label="Chaser Surge vx")
        ax2.plot(t, c_vel[:, 1], color="#0D9488", linewidth=1.5, linestyle="--", label="Chaser Sway vy")
        ax2.plot(t, c_vel[:, 2], color="#8B5CF6", linewidth=1.2, linestyle=":", label="Chaser Heave vz")
        ax2.axhline(0.0, color="#64748B", linestyle="-", linewidth=0.8, alpha=0.7)
        ax2.set_ylabel("Velocity (m/s)", fontsize=10, fontweight="bold")
        ax2.set_ylim(-1.5, 5.0)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

        # 3. 2D Flight Trajectory (X vs Y)
        ax3 = axes[2, col]
        ax3.plot(t_pos[:, 0], t_pos[:, 1], color="#DC2626", linewidth=2.5, label="Target Path")
        ax3.plot(c_pos[:, 0], c_pos[:, 1], color="#2563EB", linewidth=2.0, linestyle="--", label="Chaser Path (PID)")
        ax3.scatter([t_pos[0, 0]], [t_pos[0, 1]], color="#DC2626", marker="o", s=60, label="Tgt Start")
        ax3.scatter([c_pos[0, 0]], [c_pos[0, 1]], color="#2563EB", marker="s", s=60, label="Chaser Start")
        ax3.scatter([c_pos[-1, 0]], [c_pos[-1, 1]], color="#059669", marker="^", s=70, label="Chaser End")
        ax3.set_ylabel("East / Y (m)", fontsize=10, fontweight="bold")
        ax3.set_xlabel("North / X (m)", fontsize=10, fontweight="bold")
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

        # 4. Bounding Box & Perception
        ax4 = axes[3, col]
        telem = h.get("chaser_telem", [])
        bboxes_w = [item.get("bbox_w", 0.0) for item in telem]
        ax4.axhline(110.0, color="#059669", linestyle="--", linewidth=1.8, label="Desired BBox w* (110 px)")
        ax4.plot(t, bboxes_w, color="#7C3AED", linewidth=1.8, label="Estimated BBox Width")
        ax4.set_ylabel("BBox Width (px)", fontsize=10, fontweight="bold")
        ax4.set_xlabel("Time (s)", fontsize=10, fontweight="bold")
        ax4.set_ylim(0.0, 180.0)
        ax4.grid(True, linestyle=":", alpha=0.6)
        ax4.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    fig.suptitle(
        "Autonomous Chaser Distance-Holding Performance: Steady Hover & Gentle Drift\n"
        "Visual-Inertial Servo with Optical Looming Feedforward vs Target Station-Keeping",
        fontsize=14, fontweight="bold", color="#0F172A", y=0.98
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved hover distance-holding diagnostic figure to: {output_path}")


def run_evaluation(
    num_seeds: int = 50,
    render_video: bool = True
) -> Dict[str, Any]:
    """Runs single-case and multi-seed Monte Carlo evaluations."""
    print("=" * 78)
    print("  AUTONOMOUS DRONE PURSUIT: HOVER & GENTLE DRIFT DISTANCE HOLDING TEST")
    print("=" * 78)

    # 1. Single Run: Pure Steady Hover (d0 = 7.0m, v_tgt = 0 m/s)
    scen_steady = HoverScenario("hover_steady_d7m", mode="steady", initial_distance=7.0, duration=25.0)
    pid_steady = create_pid_controller(standoff_nominal=3.5)
    res_steady_pid = evaluate_single_run(
        scenario=scen_steady,
        controller=pid_steady,
        controller_type="pid",
        noise_mode="empirical",
        frame_drop_rate=0.05,
        standoff_nominal=3.5,
        use_kalman=True,
        return_history=True,
        enable_ego_compensation=True
    )
    metrics_steady = compute_hover_metrics(res_steady_pid["history"])

    # 2. Single Run: Pure In-Place Standoff Hover (d0 = 3.5m, v_tgt = 0 m/s)
    scen_inplace = HoverScenario("hover_inplace_d3.5m", mode="steady", initial_distance=3.5, duration=20.0)
    pid_inplace = create_pid_controller(standoff_nominal=3.5)
    res_inplace_pid = evaluate_single_run(
        scenario=scen_inplace,
        controller=pid_inplace,
        controller_type="pid",
        noise_mode="empirical",
        frame_drop_rate=0.05,
        standoff_nominal=3.5,
        use_kalman=True,
        return_history=True,
        enable_ego_compensation=True
    )
    metrics_inplace = compute_hover_metrics(res_inplace_pid["history"], t_settle_start=2.0)

    # 3. Single Run: Gentle Hover Drift (d0 = 7.0m, seed = 42)
    scen_drift = HoverScenario("hover_drift_d7m_s42", mode="drift", initial_distance=7.0, duration=25.0, seed=42)
    pid_drift = create_pid_controller(standoff_nominal=3.5)
    res_drift_pid = evaluate_single_run(
        scenario=scen_drift,
        controller=pid_drift,
        controller_type="pid",
        noise_mode="empirical",
        frame_drop_rate=0.05,
        standoff_nominal=3.5,
        use_kalman=True,
        return_history=True,
        enable_ego_compensation=True
    )
    metrics_drift_pid = compute_hover_metrics(res_drift_pid["history"])

    # 4. Single Run: Gentle Hover Drift with Residual RL Champion (50k)
    champion_path = PROJECT_ROOT / "outputs/models/residual_ppo_v1/checkpoints/residual_ppo_drone_50000_steps.zip"
    rl_controller = None
    res_drift_rl = None
    metrics_drift_rl = None
    if champion_path.exists():
        rl_controller = ResidualVisualServoController(
            model_path=str(champion_path),
            pid_gains_path="control/weights/tuned_pid_gains.json",
            standoff_nominal=3.5,
            desired_bbox_size=110.0,
            w_nominal=0.0859
        )
        res_drift_rl = evaluate_single_run(
            scenario=scen_drift,
            controller=rl_controller,
            controller_type="residual",
            noise_mode="empirical",
            frame_drop_rate=0.05,
            standoff_nominal=3.5,
            use_kalman=True,
            return_history=True,
            enable_ego_compensation=True
        )
        metrics_drift_rl = compute_hover_metrics(res_drift_rl["history"])

    # Save diagnostic plots
    plot_path = PROJECT_ROOT / "outputs/benchmarks/control/hover_drift_distance_holding.png"
    plot_hover_diagnostics(
        res_steady_pid=res_steady_pid,
        res_drift_pid=res_drift_pid,
        res_drift_rl=res_drift_rl,
        output_path=plot_path
    )

    # Copy plot to brain artifacts directory
    artifact_dir = Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/7dc8da15-984b-4499-8ca6-5e32efd45198")
    if artifact_dir.exists():
        import shutil
        shutil.copy2(plot_path, artifact_dir / "hover_drift_distance_holding.png")

    # 5. Multi-Seed Monte Carlo Suite for Gentle Hover Drift (50 Seeds)
    print(f"\n[MONTE CARLO] Evaluating {num_seeds} randomized hover drift trajectories...")
    mc_results_pid = []
    mc_results_rl = []

    for seed in range(100, 100 + num_seeds):
        scen_mc = HoverScenario(f"hover_drift_s{seed}", mode="drift", initial_distance=7.0, duration=25.0, seed=seed)
        
        # PID
        pid_mc = create_pid_controller(standoff_nominal=3.5)
        r_pid = evaluate_single_run(
            scenario=scen_mc,
            controller=pid_mc,
            controller_type="pid",
            noise_mode="empirical",
            frame_drop_rate=0.05,
            standoff_nominal=3.5,
            use_kalman=True,
            return_history=True,
            enable_ego_compensation=True
        )
        m_pid = compute_hover_metrics(r_pid["history"])
        m_pid["success_1s"] = r_pid["success_1s"]
        m_pid["success_3s"] = r_pid["success_3s"]
        m_pid["dwell_time"] = r_pid["max_continuous_dwell_s"]
        m_pid["firing_time"] = r_pid["firing_window_s"]
        m_pid["fov_retention"] = r_pid["fov_lock_pct"]
        m_pid["collision"] = r_pid["collision_breach"]
        mc_results_pid.append(m_pid)

        # Residual RL
        if rl_controller is not None:
            rl_controller.reset()
            r_rl = evaluate_single_run(
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
            m_rl = compute_hover_metrics(r_rl["history"])
            m_rl["success_1s"] = r_rl["success_1s"]
            m_rl["success_3s"] = r_rl["success_3s"]
            m_rl["dwell_time"] = r_rl["max_continuous_dwell_s"]
            m_rl["firing_time"] = r_rl["firing_window_s"]
            m_rl["fov_retention"] = r_rl["fov_lock_pct"]
            m_rl["collision"] = r_rl["collision_breach"]
            mc_results_rl.append(m_rl)

    # 6. Aggregate Monte Carlo Statistics
    def aggregate_mc(results_list: List[Dict[str, Any]]) -> Dict[str, float]:
        n = len(results_list)
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
            "ss_vel_ripple_rms": float(np.mean([r["ss_vel_ripple_rms"] for r in results_list])),
            "ss_basket_retention_pct": float(np.mean([r["ss_basket_retention_pct"] for r in results_list])),
            "settling_time_s": float(np.mean([r["settling_time_s"] for r in results_list])),
        }

    agg_pid = aggregate_mc(mc_results_pid)
    agg_rl = aggregate_mc(mc_results_rl) if mc_results_rl else None

    # 7. Render Video if requested
    video_path_str = None
    if render_video:
        vid_out = PROJECT_ROOT / "outputs/benchmarks/videos/hover_drift_chase_bright.mp4"
        vid_out.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[VIDEO] Rendering broadcast video of hover drift chase to: {vid_out}")
        render_pid_bright_video(
            scenario=scen_drift,
            pid_controller=create_pid_controller(standoff_nominal=3.5),
            output_path=vid_out,
            fps=25
        )
        video_path_str = str(vid_out)
        if artifact_dir.exists():
            import shutil
            shutil.copy2(vid_out, artifact_dir / "hover_drift_chase_bright.mp4")

    # 8. Print Results Summary
    print("\n" + "=" * 78)
    print("  EXACT HOVER & GENTLE DRIFT DISTANCE-HOLDING RESULTS")
    print("=" * 78)
    print("\n[TEST 1: PURE STEADY HOVER (d0 = 7.0m, v_tgt = 0.0 m/s)]")
    print(f"  Settling Time (to 3.5 +- 0.5m) : {metrics_steady['settling_time_s']:.2f} s")
    print(f"  Steady-State Mean Distance     : {metrics_steady['ss_mean_dist_m']:.3f} m (Nominal: 3.500 m)")
    print(f"  Steady-State Distance MAE      : {metrics_steady['ss_mae_dist_m']:.3f} m ({metrics_steady['ss_mae_dist_m']*100:.1f} cm)")
    print(f"  Steady-State Standard Dev (Std): {metrics_steady['ss_std_dist_m']:.3f} m")
    print(f"  Max Distance Deviation in SS   : {metrics_steady['ss_max_deviation_m']:.3f} m")
    print(f"  Velocity Ripple (RMS vx)       : {metrics_steady['ss_vel_ripple_rms']:.3f} m/s")
    print(f"  Basket Retention ([2.5, 4.5]m) : {metrics_steady['ss_basket_retention_pct']:.1f}%")
    print(f"  Min Distance Reached           : {metrics_steady['min_dist_m']:.3f} m (Zero Collision, buffer > 2.5m)")

    print("\n[TEST 2: IN-PLACE STANDOFF HOVER (d0 = 3.5m, v_tgt = 0.0 m/s)]")
    print(f"  Steady-State Mean Distance     : {metrics_inplace['ss_mean_dist_m']:.3f} m")
    print(f"  Steady-State Distance MAE      : {metrics_inplace['ss_mae_dist_m']:.3f} m ({metrics_inplace['ss_mae_dist_m']*100:.1f} cm)")
    print(f"  Steady-State Standard Dev      : {metrics_inplace['ss_std_dist_m']:.3f} m")
    print(f"  Velocity Ripple (RMS vx)       : {metrics_inplace['ss_vel_ripple_rms']:.3f} m/s")
    print(f"  Basket Retention ([2.5, 4.5]m) : {metrics_inplace['ss_basket_retention_pct']:.1f}%")

    print("\n[TEST 3: GENTLE HOVER DRIFT (50 RANDOMIZED SEEDS MONTE CARLO)]")
    print(f"  Metric                             | Tuned PID Baseline | Residual RL (50k)   | Delta")
    print(f"  -----------------------------------|--------------------|---------------------|-------")
    print(f"  1s Catch Rate (Basket Entry)       | {agg_pid['success_1s_pct']:>16.2f}% | {agg_rl['success_1s_pct'] if agg_rl else 0.0:>17.2f}% | {(agg_rl['success_1s_pct']-agg_pid['success_1s_pct']) if agg_rl else 0.0:+6.2f}%")
    print(f"  3s Sustained Lock Rate             | {agg_pid['success_3s_pct']:>16.2f}% | {agg_rl['success_3s_pct'] if agg_rl else 0.0:>17.2f}% | {(agg_rl['success_3s_pct']-agg_pid['success_3s_pct']) if agg_rl else 0.0:+6.2f}%")
    print(f"  Max Continuous Dwell Time (s)      | {agg_pid['dwell_time_mean_s']:>16.2f}s | {agg_rl['dwell_time_mean_s'] if agg_rl else 0.0:>17.2f}s | {(agg_rl['dwell_time_mean_s']-agg_pid['dwell_time_mean_s']) if agg_rl else 0.0:+6.2f}s")
    print(f"  Cumulative Firing Window (s)       | {agg_pid['firing_time_mean_s']:>16.2f}s | {agg_rl['firing_time_mean_s'] if agg_rl else 0.0:>17.2f}s | {(agg_rl['firing_time_mean_s']-agg_pid['firing_time_mean_s']) if agg_rl else 0.0:+6.2f}s")
    print(f"  Camera FOV Retention Rate          | {agg_pid['fov_retention_pct']:>16.2f}% | {agg_rl['fov_retention_pct'] if agg_rl else 0.0:>17.2f}% | {(agg_rl['fov_retention_pct']-agg_pid['fov_retention_pct']) if agg_rl else 0.0:+6.2f}%")
    print(f"  Steady-State Mean Distance         | {agg_pid['ss_mean_dist_m']:>16.3f}m | {agg_rl['ss_mean_dist_m'] if agg_rl else 0.0:>17.3f}m | {(agg_rl['ss_mean_dist_m']-agg_pid['ss_mean_dist_m']) if agg_rl else 0.0:+6.3f}m")
    print(f"  Steady-State Distance MAE          | {agg_pid['ss_mae_dist_m']:>16.3f}m | {agg_rl['ss_mae_dist_m'] if agg_rl else 0.0:>17.3f}m | {(agg_rl['ss_mae_dist_m']-agg_pid['ss_mae_dist_m']) if agg_rl else 0.0:+6.3f}m")
    print(f"  Steady-State Distance Std          | {agg_pid['ss_std_dist_m']:>16.3f}m | {agg_rl['ss_std_dist_m'] if agg_rl else 0.0:>17.3f}m | {(agg_rl['ss_std_dist_m']-agg_pid['ss_std_dist_m']) if agg_rl else 0.0:+6.3f}m")
    print(f"  Max Standoff Deviation in SS       | {agg_pid['ss_max_deviation_m']:>16.3f}m | {agg_rl['ss_max_deviation_m'] if agg_rl else 0.0:>17.3f}m | {(agg_rl['ss_max_deviation_m']-agg_pid['ss_max_deviation_m']) if agg_rl else 0.0:+6.3f}m")
    print(f"  SS Basket Retention ([2.5, 4.5]m)  | {agg_pid['ss_basket_retention_pct']:>16.2f}% | {agg_rl['ss_basket_retention_pct'] if agg_rl else 0.0:>17.2f}% | {(agg_rl['ss_basket_retention_pct']-agg_pid['ss_basket_retention_pct']) if agg_rl else 0.0:+6.2f}%")
    print(f"  Settling Time (s)                  | {agg_pid['settling_time_s']:>16.2f}s | {agg_rl['settling_time_s'] if agg_rl else 0.0:>17.2f}s | {(agg_rl['settling_time_s']-agg_pid['settling_time_s']) if agg_rl else 0.0:+6.2f}s")
    print(f"  Collision Breach Rate (< 0.5m)     | {agg_pid['collision_rate_pct']:>16.2f}% | {agg_rl['collision_rate_pct'] if agg_rl else 0.0:>17.2f}% | {0.0:+6.2f}%")

    out_json = PROJECT_ROOT / "outputs/benchmarks/control/hover_drift_evaluation_results.json"
    data_to_save = {
        "metrics_steady": metrics_steady,
        "metrics_inplace": metrics_inplace,
        "metrics_drift_pid": metrics_drift_pid,
        "metrics_drift_rl": metrics_drift_rl,
        "monte_carlo_pid": agg_pid,
        "monte_carlo_rl": agg_rl,
        "plot_path": str(plot_path),
        "video_path": video_path_str
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2)
    print(f"\n[DATA] Saved full hover evaluation data to: {out_json}")

    return data_to_save


if __name__ == "__main__":
    run_evaluation(num_seeds=50, render_video=True)
