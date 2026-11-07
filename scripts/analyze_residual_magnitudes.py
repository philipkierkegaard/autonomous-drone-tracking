"""
Residual Policy Magnitude & Saturation Analysis Script
Compares residual action distributions, magnitudes, and boundary saturation frequencies
between RRL v9 (wide35: [3.5, 3.5, 3.5, 35.0]) and RRL v6 (thesis: [3.0, 2.0, 1.0, 30.0])
across all 600 scenarios of the paired benchmark test suite.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import multiprocessing as mp
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.predictive_controller import (
    ResidualVisualServoController,
    apply_command_envelope_and_slew
)
from control.dataset_generator import PursuitDataset, PursuitScenario
from benchmarks.control.benchmark_controller import evaluate_single_run


class InstrumentedResidualController(ResidualVisualServoController):
    """Residual controller instrumented to record every internal command step."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_logs = []

    def reset(self):
        super().reset()
        self.step_logs = []

    def compute_cmd(
        self,
        telemetry: Dict[str, Any],
        drone_pitch: float = 0.0,
        drone_roll: float = 0.0,
        dt: float = 0.02,
        vehicle_vel: Any = None,
        vehicle_yaw_rate: Any = None,
        current_alt_m: Any = None
    ) -> np.ndarray:
        if self.episode_start[0] and vehicle_vel is not None and len(vehicle_vel) >= 3:
            self.prev_cmd[0] = float(vehicle_vel[0])
            self.prev_cmd[1] = float(vehicle_vel[1])
            self.prev_cmd[2] = float(vehicle_vel[2])
            self.prev_cmd[3] = float(vehicle_yaw_rate if vehicle_yaw_rate is not None else 0.0)

        # 1. Base PID setpoint
        pid_cmd = self.pid.compute_cmd(
            telemetry,
            drone_pitch=drone_pitch,
            drone_roll=drone_roll,
            dt=dt,
            vehicle_vel=vehicle_vel,
            vehicle_yaw_rate=vehicle_yaw_rate if vehicle_yaw_rate is not None else 0.0,
            current_alt_m=current_alt_m if current_alt_m is not None else 2.5
        )

        # 2. Learned Residual
        if self.model is not None:
            obs = self.build_obs(
                telemetry=telemetry,
                drone_pitch=drone_pitch,
                drone_roll=drone_roll,
                dt=dt,
                vehicle_vel=vehicle_vel,
                vehicle_yaw_rate=vehicle_yaw_rate,
                current_alt_m=current_alt_m
            )
            action, self.hidden_state = self.model.predict(
                obs,
                state=self.hidden_state,
                episode_start=self.episode_start,
                deterministic=self.deterministic
            )
            a = np.clip(action, -1.0, 1.0)
            residual = a * self.residual_scales
        else:
            action = np.zeros(4, dtype=np.float64)
            a = np.zeros(4, dtype=np.float64)
            residual = np.zeros(4, dtype=np.float64)

        self.episode_start = np.array([False], dtype=bool)
        target_cmd = pid_cmd + residual

        # 3. Flight command envelope & slew rate limiter
        cmd = apply_command_envelope_and_slew(
            target_cmd=target_cmd,
            prev_cmd=self.prev_cmd,
            dt=dt,
            max_rev_vel=self.max_rev_vel,
            max_fwd_vel=self.max_fwd_vel,
            max_lat_vel=self.max_lat_vel,
            max_vel_up=self.max_vel_up,
            max_vel_down=self.max_vel_down,
            max_yaw_rate=self.max_yaw_rate,
            max_accel_xy=self.max_accel_xy,
            max_decel_xy=float(getattr(self.pid, "max_decel", 2.45)),
            max_lat_accel=float(getattr(self.pid, "max_lat_accel", 6.0)),
            max_accel_z=self.max_accel_z,
            max_accel_yaw=self.max_accel_yaw,
            current_alt_m=current_alt_m
        )
        self.prev_cmd = cmd.copy()

        self.step_logs.append({
            "pid_cmd": pid_cmd.copy(),
            "raw_action": action.copy(),
            "action": a.copy(),
            "residual": residual.copy(),
            "target_cmd": target_cmd.copy(),
            "cmd": cmd.copy(),
            "status": str(telemetry.get("status", "SEARCHING")),
        })
        return cmd


def _worker_eval_scenario(args: Tuple[int, Dict[str, Any], str, str, List[float], float]) -> Dict[str, Any]:
    scenario_idx, scenario_dict, model_path_str, pid_path_str, residual_scales, tau_yaw = args
    scenario = PursuitScenario.from_dict(scenario_dict)

    ctrl = InstrumentedResidualController(
        model_path=model_path_str,
        pid_gains_path=pid_path_str,
        residual_scales=residual_scales,
        standoff_nominal=3.5,
        desired_bbox_size=110.0,
    )

    res = evaluate_single_run(
        scenario=scenario,
        controller=ctrl,
        controller_type="residual",
        frame_drop_rate=0.05,
        noise_mode="empirical",
        tau_yaw=tau_yaw,
        return_history=True,
    )

    logs = ctrl.step_logs
    history = res.get("history", {})
    distances = history.get("distance", np.zeros(len(logs), dtype=np.float32))
    in_views = history.get("in_view", np.zeros(len(logs), dtype=bool))

    # Convert to compact arrays
    N = len(logs)
    actions = np.array([l["action"] for l in logs], dtype=np.float32)       # (N, 4)
    residuals = np.array([l["residual"] for l in logs], dtype=np.float32)   # (N, 4)
    pid_cmds = np.array([l["pid_cmd"] for l in logs], dtype=np.float32)     # (N, 4)
    target_cmds = np.array([l["target_cmd"] for l in logs], dtype=np.float32)
    final_cmds = np.array([l["cmd"] for l in logs], dtype=np.float32)
    statuses = [l["status"] for l in logs]

    # Classify phase per step
    # Approach: distance > 5.0
    # Dwell: 2.5 <= distance <= 4.5 and in_view
    # Other: intermediate
    is_approach = distances > 5.0
    is_dwell = (distances >= 2.5) & (distances <= 4.5) & in_views
    is_coasting = np.array([s != "LOCKED" for s in statuses], dtype=bool)

    # Compute scenario summary stats to return
    # Histograms for action values in [-1, 1], 50 bins
    bins = np.linspace(-1.0, 1.0, 51)
    hist_actions = [np.histogram(actions[:, i], bins=bins)[0].tolist() for i in range(4)]

    # Envelope clipping occurrences
    env_clipped = np.abs(target_cmds - final_cmds) > 1e-3

    return {
        "scenario_id": scenario.scenario_id,
        "profile": scenario.profile,
        "encounter_type": scenario.encounter_type,
        "success_2s": bool(res.get("success_2s", False)),
        "max_continuous_dwell_s": float(res.get("max_continuous_dwell_s", 0.0)),
        "num_steps": N,
        # Sums for overall means & rms
        "sum_action": np.sum(actions, axis=0).tolist(),
        "sum_abs_action": np.sum(np.abs(actions), axis=0).tolist(),
        "sum_sq_action": np.sum(actions ** 2, axis=0).tolist(),
        "max_abs_action": np.max(np.abs(actions), axis=0).tolist(),

        "sum_res": np.sum(residuals, axis=0).tolist(),
        "sum_abs_res": np.sum(np.abs(residuals), axis=0).tolist(),
        "sum_sq_res": np.sum(residuals ** 2, axis=0).tolist(),
        "max_abs_res": np.max(np.abs(residuals), axis=0).tolist(),

        "sum_abs_pid": np.sum(np.abs(pid_cmds), axis=0).tolist(),
        "sum_sq_pid": np.sum(pid_cmds ** 2, axis=0).tolist(),

        # Saturation counts
        "count_sat_80": np.sum(np.abs(actions) >= 0.80, axis=0).tolist(),
        "count_sat_90": np.sum(np.abs(actions) >= 0.90, axis=0).tolist(),
        "count_sat_95": np.sum(np.abs(actions) >= 0.95, axis=0).tolist(),
        "count_sat_99": np.sum(np.abs(actions) >= 0.99, axis=0).tolist(),
        "count_sat_pos_95": np.sum(actions >= 0.95, axis=0).tolist(),
        "count_sat_neg_95": np.sum(actions <= -0.95, axis=0).tolist(),
        "count_any_sat_95": int(np.sum(np.any(np.abs(actions) >= 0.95, axis=1))),
        "count_env_clipped": np.sum(env_clipped, axis=0).tolist(),

        # Phase breakdowns
        "steps_approach": int(np.sum(is_approach)),
        "sum_abs_res_approach": np.sum(np.abs(residuals[is_approach]), axis=0).tolist() if np.any(is_approach) else [0.0]*4,
        "sat_95_approach": np.sum(np.abs(actions[is_approach]) >= 0.95, axis=0).tolist() if np.any(is_approach) else [0]*4,

        "steps_dwell": int(np.sum(is_dwell)),
        "sum_abs_res_dwell": np.sum(np.abs(residuals[is_dwell]), axis=0).tolist() if np.any(is_dwell) else [0.0]*4,
        "sat_95_dwell": np.sum(np.abs(actions[is_dwell]) >= 0.95, axis=0).tolist() if np.any(is_dwell) else [0]*4,

        "steps_coasting": int(np.sum(is_coasting)),
        "sum_abs_res_coasting": np.sum(np.abs(residuals[is_coasting]), axis=0).tolist() if np.any(is_coasting) else [0.0]*4,
        "sat_95_coasting": np.sum(np.abs(actions[is_coasting]) >= 0.95, axis=0).tolist() if np.any(is_coasting) else [0]*4,

        "hist_actions": hist_actions,
    }


def aggregate_model_results(raw_results: List[Dict[str, Any]], residual_scales: List[float]) -> Dict[str, Any]:
    """Aggregates scenario results into overall, profile, and phase level analytics."""
    axis_names = ["Surge (vx)", "Sway (vy)", "Heave (vz)", "Yaw Rate (yaw)"]
    units = ["m/s", "m/s", "m/s", "deg/s"]

    total_steps = sum(r["num_steps"] for r in raw_results)
    profiles = sorted(list(set(r["profile"] for r in raw_results)))

    # Global accumulation
    sum_action = np.zeros(4, dtype=np.float64)
    sum_abs_action = np.zeros(4, dtype=np.float64)
    sum_sq_action = np.zeros(4, dtype=np.float64)
    max_abs_action = np.zeros(4, dtype=np.float64)

    sum_res = np.zeros(4, dtype=np.float64)
    sum_abs_res = np.zeros(4, dtype=np.float64)
    sum_sq_res = np.zeros(4, dtype=np.float64)
    max_abs_res = np.zeros(4, dtype=np.float64)

    sum_abs_pid = np.zeros(4, dtype=np.float64)
    sum_sq_pid = np.zeros(4, dtype=np.float64)

    count_sat_80 = np.zeros(4, dtype=np.int64)
    count_sat_90 = np.zeros(4, dtype=np.int64)
    count_sat_95 = np.zeros(4, dtype=np.int64)
    count_sat_99 = np.zeros(4, dtype=np.int64)
    count_sat_pos_95 = np.zeros(4, dtype=np.int64)
    count_sat_neg_95 = np.zeros(4, dtype=np.int64)
    count_any_sat_95 = 0
    count_env_clipped = np.zeros(4, dtype=np.int64)

    total_approach_steps = 0
    sum_abs_res_approach = np.zeros(4, dtype=np.float64)
    sat_95_approach = np.zeros(4, dtype=np.int64)

    total_dwell_steps = 0
    sum_abs_res_dwell = np.zeros(4, dtype=np.float64)
    sat_95_dwell = np.zeros(4, dtype=np.int64)

    total_coasting_steps = 0
    sum_abs_res_coasting = np.zeros(4, dtype=np.float64)
    sat_95_coasting = np.zeros(4, dtype=np.int64)

    hist_action_sum = np.zeros((4, 50), dtype=np.int64)

    # Per-profile accumulation
    profile_data = {p: {
        "steps": 0,
        "sum_abs_res": np.zeros(4, dtype=np.float64),
        "sum_abs_action": np.zeros(4, dtype=np.float64),
        "count_sat_95": np.zeros(4, dtype=np.int64),
        "count_sat_99": np.zeros(4, dtype=np.int64),
        "sum_abs_pid": np.zeros(4, dtype=np.float64),
        "success_2s": 0,
        "count": 0,
    } for p in profiles}

    for r in raw_results:
        sum_action += np.array(r["sum_action"])
        sum_abs_action += np.array(r["sum_abs_action"])
        sum_sq_action += np.array(r["sum_sq_action"])
        max_abs_action = np.maximum(max_abs_action, np.array(r["max_abs_action"]))

        sum_res += np.array(r["sum_res"])
        sum_abs_res += np.array(r["sum_abs_res"])
        sum_sq_res += np.array(r["sum_sq_res"])
        max_abs_res = np.maximum(max_abs_res, np.array(r["max_abs_res"]))

        sum_abs_pid += np.array(r["sum_abs_pid"])
        sum_sq_pid += np.array(r["sum_sq_pid"])

        count_sat_80 += np.array(r["count_sat_80"])
        count_sat_90 += np.array(r["count_sat_90"])
        count_sat_95 += np.array(r["count_sat_95"])
        count_sat_99 += np.array(r["count_sat_99"])
        count_sat_pos_95 += np.array(r["count_sat_pos_95"])
        count_sat_neg_95 += np.array(r["count_sat_neg_95"])
        count_any_sat_95 += r["count_any_sat_95"]
        count_env_clipped += np.array(r["count_env_clipped"])

        total_approach_steps += r["steps_approach"]
        sum_abs_res_approach += np.array(r["sum_abs_res_approach"])
        sat_95_approach += np.array(r["sat_95_approach"])

        total_dwell_steps += r["steps_dwell"]
        sum_abs_res_dwell += np.array(r["sum_abs_res_dwell"])
        sat_95_dwell += np.array(r["sat_95_dwell"])

        total_coasting_steps += r["steps_coasting"]
        sum_abs_res_coasting += np.array(r["sum_abs_res_coasting"])
        sat_95_coasting += np.array(r["sat_95_coasting"])

        for i in range(4):
            hist_action_sum[i] += np.array(r["hist_actions"][i])

        # Profile level
        p = r["profile"]
        profile_data[p]["steps"] += r["num_steps"]
        profile_data[p]["sum_abs_res"] += np.array(r["sum_abs_res"])
        profile_data[p]["sum_abs_action"] += np.array(r["sum_abs_action"])
        profile_data[p]["count_sat_95"] += np.array(r["count_sat_95"])
        profile_data[p]["count_sat_99"] += np.array(r["count_sat_99"])
        profile_data[p]["sum_abs_pid"] += np.array(r["sum_abs_pid"])
        if r["success_2s"]:
            profile_data[p]["success_2s"] += 1
        profile_data[p]["count"] += 1

    # Format axis stats
    axes_stats = []
    for i in range(4):
        mean_signed = float(sum_res[i] / total_steps)
        mean_abs = float(sum_abs_res[i] / total_steps)
        rms = float(np.sqrt(sum_sq_res[i] / total_steps))
        peak = float(max_abs_res[i])
        scale = float(residual_scales[i])

        mean_abs_a = float(sum_abs_action[i] / total_steps)
        pct_authority = float(mean_abs_a * 100.0)

        mean_pid = float(sum_abs_pid[i] / total_steps)
        rms_pid = float(np.sqrt(sum_sq_pid[i] / total_steps))

        ratio_to_pid = float(mean_abs / (mean_pid + 1e-6))
        residual_share_pct = float(mean_abs / (mean_pid + mean_abs + 1e-6) * 100.0)

        sat_80_pct = float(count_sat_80[i] / total_steps * 100.0)
        sat_90_pct = float(count_sat_90[i] / total_steps * 100.0)
        sat_95_pct = float(count_sat_95[i] / total_steps * 100.0)
        sat_99_pct = float(count_sat_99[i] / total_steps * 100.0)
        sat_pos_95_pct = float(count_sat_pos_95[i] / total_steps * 100.0)
        sat_neg_95_pct = float(count_sat_neg_95[i] / total_steps * 100.0)

        clip_pct = float(count_env_clipped[i] / total_steps * 100.0)

        axes_stats.append({
            "axis": axis_names[i],
            "unit": units[i],
            "scale_limit": scale,
            "mean_signed_res": mean_signed,
            "mean_abs_res": mean_abs,
            "rms_res": rms,
            "max_abs_res": peak,
            "pct_authority_used": pct_authority,
            "mean_abs_pid": mean_pid,
            "rms_pid": rms_pid,
            "ratio_res_to_pid": ratio_to_pid,
            "residual_share_pct": residual_share_pct,
            "sat_ge_80_pct": sat_80_pct,
            "sat_ge_90_pct": sat_90_pct,
            "sat_ge_95_pct": sat_95_pct,
            "sat_ge_99_pct": sat_99_pct,
            "sat_pos_95_pct": sat_pos_95_pct,
            "sat_neg_95_pct": sat_neg_95_pct,
            "envelope_clipped_pct": clip_pct,
        })

    # Profile breakdowns
    profile_summary = {}
    for p, pd in profile_data.items():
        n_p = pd["steps"]
        profile_summary[p] = {
            "num_scenarios": pd["count"],
            "catch_rate_2s_pct": float(pd["success_2s"] / pd["count"] * 100.0),
            "axes": [
                {
                    "axis": axis_names[i],
                    "mean_abs_res": float(pd["sum_abs_res"][i] / n_p),
                    "pct_authority": float(pd["sum_abs_action"][i] / n_p * 100.0),
                    "mean_abs_pid": float(pd["sum_abs_pid"][i] / n_p),
                    "sat_ge_95_pct": float(pd["count_sat_95"][i] / n_p * 100.0),
                    "sat_ge_99_pct": float(pd["count_sat_99"][i] / n_p * 100.0),
                } for i in range(4)
            ]
        }

    # Phase breakdowns
    phase_summary = {
        "approach": {
            "total_steps": int(total_approach_steps),
            "pct_of_flight": float(total_approach_steps / total_steps * 100.0),
            "axes": [
                {
                    "axis": axis_names[i],
                    "mean_abs_res": float(sum_abs_res_approach[i] / max(1, total_approach_steps)),
                    "sat_ge_95_pct": float(sat_95_approach[i] / max(1, total_approach_steps) * 100.0)
                } for i in range(4)
            ]
        },
        "dwell": {
            "total_steps": int(total_dwell_steps),
            "pct_of_flight": float(total_dwell_steps / total_steps * 100.0),
            "axes": [
                {
                    "axis": axis_names[i],
                    "mean_abs_res": float(sum_abs_res_dwell[i] / max(1, total_dwell_steps)),
                    "sat_ge_95_pct": float(sat_95_dwell[i] / max(1, total_dwell_steps) * 100.0)
                } for i in range(4)
            ]
        },
        "coasting": {
            "total_steps": int(total_coasting_steps),
            "pct_of_flight": float(total_coasting_steps / total_steps * 100.0),
            "axes": [
                {
                    "axis": axis_names[i],
                    "mean_abs_res": float(sum_abs_res_coasting[i] / max(1, total_coasting_steps)),
                    "sat_ge_95_pct": float(sat_95_coasting[i] / max(1, total_coasting_steps) * 100.0)
                } for i in range(4)
            ]
        }
    }

    return {
        "num_scenarios": len(raw_results),
        "total_timesteps": total_steps,
        "residual_scales": residual_scales,
        "overall_catch_rate_2s": float(sum(r["success_2s"] for r in raw_results) / len(raw_results) * 100.0),
        "any_axis_sat_95_pct": float(count_any_sat_95 / total_steps * 100.0),
        "axes_stats": axes_stats,
        "profile_summary": profile_summary,
        "phase_summary": phase_summary,
        "hist_actions": hist_action_sum.tolist(),
        "hist_bins": np.linspace(-1.0, 1.0, 51).tolist(),
    }


def plot_distributions_and_profiles(
    stats_v9: Dict[str, Any],
    stats_v6: Dict[str, Any],
    out_dir: Path
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Action Histograms Comparison Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    bins = np.array(stats_v9["hist_bins"])
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    axis_labels = [
        ("Surge (vx)", "m/s", stats_v9["residual_scales"][0], stats_v6["residual_scales"][0]),
        ("Sway (vy)", "m/s", stats_v9["residual_scales"][1], stats_v6["residual_scales"][1]),
        ("Heave (vz)", "m/s", stats_v9["residual_scales"][2], stats_v6["residual_scales"][2]),
        ("Yaw Rate (yaw)", "deg/s", stats_v9["residual_scales"][3], stats_v6["residual_scales"][3]),
    ]

    for i, (name, unit, s_v9, s_v6) in enumerate(axis_labels):
        ax = axes[i // 2, i % 2]
        h_v9 = np.array(stats_v9["hist_actions"][i]) / stats_v9["total_timesteps"] * 100.0
        h_v6 = np.array(stats_v6["hist_actions"][i]) / stats_v6["total_timesteps"] * 100.0

        ax.plot(bin_centers, h_v9, label=f"v9 wide35 (limit = \u00b1{s_v9} {unit})", color="#2563eb", lw=2.2)
        ax.fill_between(bin_centers, h_v9, color="#2563eb", alpha=0.18)

        ax.plot(bin_centers, h_v6, label=f"v6 thesis (limit = \u00b1{s_v6} {unit})", color="#dc2626", lw=2.0, linestyle="--")
        ax.fill_between(bin_centers, h_v6, color="#dc2626", alpha=0.12)

        ax.axvline(x=-0.95, color="gray", linestyle=":", alpha=0.7, label="Saturation (\u00b195%)" if i==0 else "")
        ax.axvline(x=0.95, color="gray", linestyle=":", alpha=0.7)

        ax.set_title(f"Normalized Action Distribution: {name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Normalized Action $a_i \in [-1, 1]$", fontsize=10)
        ax.set_ylabel("Frequency (%)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_xlim(-1.05, 1.05)

    plt.suptitle("RRL Residual Action Distributions: v9 (wide35) vs v6 (thesis baseline)\nEvaluated on 600 Scenarios (750,000 timesteps)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    hist_path = out_dir / "residual_magnitude_distributions.png"
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved action distribution plot to {hist_path}")

    # 2. Magnitude & Saturation by Profile Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    profiles = ["cruising", "evasive", "hyper_evasive"]
    x = np.arange(len(profiles))
    width = 0.2

    # Left: Mean Absolute Residual Yaw Rate across profiles
    ax_yaw = axes[0]
    yaw_v9 = [stats_v9["profile_summary"][p]["axes"][3]["mean_abs_res"] for p in profiles]
    yaw_v6 = [stats_v6["profile_summary"][p]["axes"][3]["mean_abs_res"] for p in profiles]
    yaw_pid_v9 = [stats_v9["profile_summary"][p]["axes"][3]["mean_abs_pid"] for p in profiles]

    ax_yaw.bar(x - width, yaw_pid_v9, width, label="Baseline PID Command", color="#94a3b8")
    ax_yaw.bar(x, yaw_v6, width, label="v6 Thesis Residual (scale = \u00b130\u00b0/s)", color="#ef4444")
    ax_yaw.bar(x + width, yaw_v9, width, label="v9 Wide Residual (scale = \u00b135\u00b0/s)", color="#3b82f6")

    ax_yaw.set_title("Mean Absolute Yaw Command by Target Profile", fontsize=12, fontweight="bold")
    ax_yaw.set_ylabel("Command Magnitude (deg/s)", fontsize=11)
    ax_yaw.set_xticks(x)
    ax_yaw.set_xticklabels([p.replace("_", " ").title() for p in profiles], fontsize=11)
    ax_yaw.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax_yaw.legend(fontsize=10)

    # Right: Sway Residual Authority Used across profiles
    ax_sway = axes[1]
    sway_auth_v9 = [stats_v9["profile_summary"][p]["axes"][1]["pct_authority"] for p in profiles]
    sway_auth_v6 = [stats_v6["profile_summary"][p]["axes"][1]["pct_authority"] for p in profiles]

    ax_sway.bar(x - width/2, sway_auth_v6, width, label="v6 Thesis Sway (limit = \u00b12.0 m/s)", color="#ef4444")
    ax_sway.bar(x + width/2, sway_auth_v9, width, label="v9 Wide Sway (limit = \u00b13.5 m/s)", color="#3b82f6")

    ax_sway.set_title("Sway Residual Authority Utilization (%)", fontsize=12, fontweight="bold")
    ax_sway.set_ylabel("Mean % of Authority Used", fontsize=11)
    ax_sway.set_xticks(x)
    ax_sway.set_xticklabels([p.replace("_", " ").title() for p in profiles], fontsize=11)
    ax_sway.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax_sway.legend(fontsize=10)

    plt.suptitle("Residual Policy Behavior Across Target Evasion Profiles", fontsize=14, fontweight="bold")
    plt.tight_layout()
    prof_path = out_dir / "residual_saturation_by_profile.png"
    plt.savefig(prof_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved profile comparison plot to {prof_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze residual magnitude and saturation on benchmark suite.")
    parser.add_argument("--dataset", type=str, default="outputs/benchmarks/datasets/ci_10x600/ci_benchmark_600_fold_01.json")
    parser.add_argument("--v9-model", type=str, default="outputs/models/residual_ppo_v9_wide35/best_model/best_model.zip")
    parser.add_argument("--v6-model", type=str, default="outputs/models/residual_ppo_v6_hard_2s_dwell/best_model/best_model.zip")
    parser.add_argument("--pid-gains", type=str, default="control/weights/tuned_pid_gains_pre_attitude_dynamics.json")
    parser.add_argument("--tau-yaw", type=float, default=0.150)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-scenarios", type=int, default=None, help="Optional scenario limit for quick testing")
    args = parser.parse_args()

    dataset_path = PROJECT_ROOT / args.dataset
    print(f"\n{'='*85}")
    print(f"  RESIDUAL MAGNITUDE & SATURATION ANALYSIS")
    print(f"  Dataset: {dataset_path}")
    print(f"  Workers: {args.workers}")
    print(f"{'='*85}\n")

    dataset = PursuitDataset.load_json(dataset_path)
    scenarios_dicts = [s.to_dict() for s in dataset.scenarios]
    if args.max_scenarios is not None:
        scenarios_dicts = scenarios_dicts[:args.max_scenarios]

    num_scens = len(scenarios_dicts)
    print(f"Evaluating {num_scens} scenarios...")

    # 1. Evaluate v9 wide35
    print("\n>>> Evaluating RRL v9 wide35 (scales: [3.5, 3.5, 3.5, 35.0])...")
    v9_tasks = [
        (i, s_dict, str(PROJECT_ROOT / args.v9_model), str(PROJECT_ROOT / args.pid_gains), [3.5, 3.5, 3.5, 35.0], args.tau_yaw)
        for i, s_dict in enumerate(scenarios_dicts)
    ]
    t0 = time.perf_counter()
    with mp.Pool(processes=args.workers) as pool:
        raw_v9 = pool.map(_worker_eval_scenario, v9_tasks)
    t1 = time.perf_counter()
    print(f"Completed v9 wide35 in {t1 - t0:.2f}s ({((t1 - t0)/num_scens)*1000:.1f}ms per scenario)")

    stats_v9 = aggregate_model_results(raw_v9, [3.5, 3.5, 3.5, 35.0])

    # 2. Evaluate v6 thesis
    print("\n>>> Evaluating RRL v6 thesis (scales: [3.0, 2.0, 1.0, 30.0])...")
    v6_tasks = [
        (i, s_dict, str(PROJECT_ROOT / args.v6_model), str(PROJECT_ROOT / args.pid_gains), [3.0, 2.0, 1.0, 30.0], args.tau_yaw)
        for i, s_dict in enumerate(scenarios_dicts)
    ]
    t2 = time.perf_counter()
    with mp.Pool(processes=args.workers) as pool:
        raw_v6 = pool.map(_worker_eval_scenario, v6_tasks)
    t3 = time.perf_counter()
    print(f"Completed v6 thesis in {t3 - t2:.2f}s ({((t3 - t2)/num_scens)*1000:.1f}ms per scenario)")

    stats_v6 = aggregate_model_results(raw_v6, [3.0, 2.0, 1.0, 30.0])

    # Save JSON summary
    out_dir = PROJECT_ROOT / "outputs" / "benchmarks" / "control"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "residual_magnitude_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"v9_wide35": stats_v9, "v6_thesis": stats_v6}, f, indent=2)
    print(f"\n[REPORT] Saved analysis data to {json_path}")

    # Generate plots
    plot_distributions_and_profiles(stats_v9, stats_v6, out_dir)

    # Print summary tables to console
    print(f"\n{'='*105}")
    print(f"  RRL v9 (wide35) RESIDUAL METRICS & SATURATION (Overall: {stats_v9['total_timesteps']:,} Steps)")
    print(f"{'='*105}")
    print(f"{'Axis':18s} | {'Limit':9s} | {'Mean |Res|':10s} | {'RMS Res':9s} | {'Max Res':9s} | {'% Auth':7s} | {'Mean |PID|':10s} | {'Res Share':9s} | {'Sat >= 95%':10s} | {'Sat >= 99%':10s}")
    print(f"{'-'*105}")
    for a in stats_v9["axes_stats"]:
        unit = a["unit"]
        print(f"{a['axis']:18s} | \u00b1{a['scale_limit']:4.1f} {unit:5s} | {a['mean_abs_res']:7.3f} {unit:2s} | {a['rms_res']:6.3f} {unit:2s} | {a['max_abs_res']:6.3f} {unit:2s} | {a['pct_authority_used']:5.1f}% | {a['mean_abs_pid']:7.3f} {unit:2s} | {a['residual_share_pct']:7.1f}%  | {a['sat_ge_95_pct']:8.2f}%  | {a['sat_ge_99_pct']:8.2f}%")

    print(f"\n{'='*105}")
    print(f"  RRL v6 (thesis) RESIDUAL METRICS & SATURATION (Overall: {stats_v6['total_timesteps']:,} Steps)")
    print(f"{'='*105}")
    print(f"{'Axis':18s} | {'Limit':9s} | {'Mean |Res|':10s} | {'RMS Res':9s} | {'Max Res':9s} | {'% Auth':7s} | {'Mean |PID|':10s} | {'Res Share':9s} | {'Sat >= 95%':10s} | {'Sat >= 99%':10s}")
    print(f"{'-'*105}")
    for a in stats_v6["axes_stats"]:
        unit = a["unit"]
        print(f"{a['axis']:18s} | \u00b1{a['scale_limit']:4.1f} {unit:5s} | {a['mean_abs_res']:7.3f} {unit:2s} | {a['rms_res']:6.3f} {unit:2s} | {a['max_abs_res']:6.3f} {unit:2s} | {a['pct_authority_used']:5.1f}% | {a['mean_abs_pid']:7.3f} {unit:2s} | {a['residual_share_pct']:7.1f}%  | {a['sat_ge_95_pct']:8.2f}%  | {a['sat_ge_99_pct']:8.2f}%")


if __name__ == "__main__":
    main()

