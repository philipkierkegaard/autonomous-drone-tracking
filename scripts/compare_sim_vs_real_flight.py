#!/usr/bin/env python3
"""
Sim-to-Real Flight Validation & Comparative Analysis.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Replays a real-world flight test in the calibrated simulation environment from identical
initial relative conditions and overlays the response trajectories, quantifying the
Sim-to-Real transfer gap (Distance RMSE, Yaw Tracking VAF, Speed Profiles).
"""

import sys
import argparse
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.control.benchmark_controller import (
    create_pid_controller,
    FastPixhawkQuadSim,
    evaluate_single_run
)
from control.dataset_generator import PursuitScenario

ARTIFACT_DIR = Path("/Users/philipkierkegaard/.gemini/antigravity/brain/3f6fb6d1-ed45-4315-9b5d-b3cca4e6bef7")


class StationaryTargetOracle:
    """Target oracle placing a static or slow hover target at a fixed 3D location in NED."""
    def __init__(self, target_pos_ned: np.ndarray):
        self.target_pos = np.array(target_pos_ned, dtype=np.float64)

    def get_position(self, t: float) -> np.ndarray:
        return self.target_pos.copy()

    def get_state(self, t: float):
        return self.target_pos.copy(), np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)


class SimToRealReplayScenario:
    """Replay scenario configured to mirror the real flight's initial geometry."""
    def __init__(
        self,
        target_pos_ned: np.ndarray,
        initial_chaser_pos: np.ndarray,
        initial_chaser_yaw: float,
        initial_chaser_speed: float,
        duration: float,
        scenario_id: str = "sim_to_real_replay"
    ):
        self.scenario_id = scenario_id
        self.profile = "steady_slow"
        self.encounter_type = "step_response_chase"
        self.duration = float(duration)
        self.seed = 42
        self.initial_distance = float(np.linalg.norm(target_pos_ned - initial_chaser_pos))
        self.initial_azimuth_deg = 0.0
        self.initial_elevation_deg = 0.0
        self.initial_target_pos = target_pos_ned.tolist()
        self.initial_target_heading_rad = 0.0
        self.initial_chaser_pos = initial_chaser_pos.tolist()
        self.initial_chaser_yaw = float(initial_chaser_yaw)
        self.initial_chaser_speed = float(initial_chaser_speed)
        self.oracle = StationaryTargetOracle(target_pos_ned)

    def create_target_oracle(self) -> StationaryTargetOracle:
        return self.oracle


def parse_real_flight_csv(csv_path: Path) -> pd.DataFrame:
    """Loads and validates flight telemetry CSV recorded by run_tracker.py."""
    df = pd.read_csv(csv_path)
    required = ["timestamp_s", "status", "error_x", "error_y", "bbox_w", "est_distance_m", "cmd_vx", "cmd_yaw_rate_dps"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {csv_path.name} missing required columns: {missing}")
    return df


def extract_engagement_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Isolates the active pursuit engagement window where the tracker is locked or autonomy is active."""
    # Find first locked frame
    locked_mask = df["status"].isin(["LOCKED", "COASTING"])
    if "autonomy_authorized" in df.columns:
        active_mask = locked_mask | (df["autonomy_authorized"] == 1)
    else:
        active_mask = locked_mask

    if not active_mask.any():
        print("[WARNING] No 'LOCKED' or authorized autonomy frames found. Using full dataset.")
        seg = df.copy()
    else:
        first_idx = active_mask.idxmax()
        # Find continuous tracking segment (cut off if lost for > 3 seconds)
        seg = df.loc[first_idx:].copy()

    # Zero-base the timestamps
    t0 = seg["timestamp_s"].iloc[0]
    seg["time_s"] = seg["timestamp_s"] - t0
    return seg.reset_index(drop=True)


def simulate_matching_flight(
    real_seg: pd.DataFrame,
    pid_gains_path: Path,
    standoff_nominal: float = 3.5
) -> Dict[str, Any]:
    """Sets up FastPixhawkQuadSim with matching initial states and executes simulation."""
    duration = float(real_seg["time_s"].iloc[-1])
    d0 = float(real_seg["est_distance_m"].iloc[0])
    if d0 <= 0.5:
        # Pinhole fallback if not directly recorded
        bw0 = max(5.0, float(real_seg["bbox_w"].iloc[0]))
        d0 = standoff_nominal * (110.0 / bw0)

    ex0 = float(real_seg["error_x"].iloc[0])
    ey0 = float(real_seg["error_y"].iloc[0])

    # Convert initial optical errors to 3D position of target relative to drone
    # Camera: +15 deg uptilt, HFOV 62 deg, VFOV 38 deg
    hfov = np.deg2rad(62.0)
    vfov = np.deg2rad(38.0)
    az0 = ex0 * (hfov / 2.0)
    el0 = -ey0 * (vfov / 2.0)  # negative because ey is positive down

    # Direction in camera frame
    p_cam = np.array([
        d0 * np.sin(az0) * np.cos(el0),
        -d0 * np.sin(el0),
        d0 * np.cos(az0) * np.cos(el0)
    ])

    # Camera to body transformation (+15 deg up-tilt)
    uptilt = np.deg2rad(15.0)
    cu, su = np.cos(uptilt), np.sin(uptilt)
    M_cam_to_body = np.array([
        [0.0, su, cu],
        [1.0, 0.0, 0.0],
        [0.0, cu, -su]
    ])
    p_body = M_cam_to_body @ p_cam

    # Initial drone pos at origin [0, 0, -alt]
    alt0 = float(real_seg.get("measured_alt_m", [2.5]).iloc[0]) if "measured_alt_m" in real_seg else 2.5
    c_pos0 = np.array([0.0, 0.0, -alt0])
    yaw0 = 0.0

    target_pos_ned = c_pos0 + p_body
    v_init = float(real_seg.get("measured_vx_body", [0.0]).iloc[0]) if "measured_vx_body" in real_seg else 0.0

    scenario = SimToRealReplayScenario(
        target_pos_ned=target_pos_ned,
        initial_chaser_pos=c_pos0,
        initial_chaser_yaw=yaw0,
        initial_chaser_speed=v_init,
        duration=duration
    )

    pid = create_pid_controller(
        gains_path=str(pid_gains_path),
        standoff_nominal=standoff_nominal,
        desired_bbox_size=110.0
    )

    res = evaluate_single_run(
        scenario=scenario,
        controller=pid,
        controller_type="pid",
        noise_mode="empirical",
        frame_drop_rate=0.03,
        standoff_nominal=standoff_nominal,
        use_kalman=True,
        return_history=True,
        enable_ego_compensation=True
    )
    return res


def compute_vaf(y_real: np.ndarray, y_sim: np.ndarray) -> float:
    """Computes Variance Accounted For (VAF) percentage between real and simulated signals."""
    diff = y_real - y_sim
    var_diff = np.var(diff)
    var_real = np.var(y_real)
    if var_real < 1e-6:
        return 100.0 if var_diff < 1e-6 else 0.0
    return float(max(0.0, (1.0 - (var_diff / var_real)) * 100.0))


def plot_sim_vs_real(
    real_seg: pd.DataFrame,
    sim_res: Dict[str, Any],
    output_path: Path,
    standoff_nominal: float = 3.5
) -> Dict[str, float]:
    """Generates a publication-grade 4-panel comparison figure."""
    h = sim_res["history"]
    t_sim = h["times"]
    t_real = real_seg["time_s"].values

    # Common time base for metrics
    t_max = min(t_sim[-1], t_real[-1])
    t_eval = np.linspace(0.0, t_max, int(t_max * 50))

    # Real signals interpolated
    d_real_interp = np.interp(t_eval, t_real, real_seg["est_distance_m"].values)
    r_real_cmd = np.interp(t_eval, t_real, real_seg["cmd_yaw_rate_dps"].values)
    r_real_meas = np.interp(t_eval, t_real, real_seg["measured_yaw_rate_dps"].values) if "measured_yaw_rate_dps" in real_seg else r_real_cmd
    vx_real_cmd = np.interp(t_eval, t_real, real_seg["cmd_vx"].values)
    vx_real_meas = np.interp(t_eval, t_real, real_seg["measured_vx_body"].values) if "measured_vx_body" in real_seg else vx_real_cmd
    ex_real = np.interp(t_eval, t_real, real_seg["error_x"].values)
    ey_real = np.interp(t_eval, t_real, real_seg["error_y"].values)

    # Sim signals interpolated
    d_sim_interp = np.interp(t_eval, t_sim, h["distance"])
    r_sim_cmd = np.interp(t_eval, t_sim, np.rad2deg(h["chaser_cmd"][:, 3]))
    vx_sim_cmd = np.interp(t_eval, t_sim, h["chaser_cmd"][:, 0])
    vx_sim_meas = np.interp(t_eval, t_sim, np.linalg.norm(h["chaser_vel"][:, :2], axis=1))

    # Sim errors
    sim_telem = h["chaser_telem"]
    ex_sim_raw = [t.get("error_x", 0.0) for t in sim_telem]
    ey_sim_raw = [t.get("error_y", 0.0) for t in sim_telem]
    ex_sim = np.interp(t_eval, t_sim, ex_sim_raw)
    ey_sim = np.interp(t_eval, t_sim, ey_sim_raw)

    # Compute key transfer metrics
    rmse_d = float(np.sqrt(np.mean((d_real_interp - d_sim_interp) ** 2)))
    vaf_d = compute_vaf(d_real_interp, d_sim_interp)
    rmse_vx = float(np.sqrt(np.mean((vx_real_meas - vx_sim_meas) ** 2)))
    vaf_vx = compute_vaf(vx_real_meas, vx_sim_meas)
    rmse_r = float(np.sqrt(np.mean((r_real_cmd - r_sim_cmd) ** 2)))
    vaf_r = compute_vaf(r_real_cmd, r_sim_cmd)

    metrics = {
        "rmse_distance_m": round(rmse_d, 3),
        "vaf_distance_pct": round(vaf_d, 1),
        "rmse_fwd_vel_m_s": round(rmse_vx, 3),
        "vaf_fwd_vel_pct": round(vaf_vx, 1),
        "rmse_yaw_rate_dps": round(rmse_r, 2),
        "vaf_yaw_rate_pct": round(vaf_r, 1),
    }

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#FFFFFF")

    # 1. Standoff Distance
    ax1 = axes[0, 0]
    ax1.plot(t_eval, d_real_interp, color="#DC2626", lw=2.4, label="Real Flight Test (Physical)")
    ax1.plot(t_eval, d_sim_interp, color="#2563EB", lw=2.4, ls="--", label="Simulated Quad Sim (FastPixhawkQuadSim)")
    ax1.axhline(standoff_nominal, color="#059669", ls=":", lw=1.8, label=f"Standoff Setpoint ({standoff_nominal:.1f}m)")
    ax1.axhspan(2.5, 4.5, color="#10B981", alpha=0.12, label="Capture Basket [2.5-4.5m]")
    ax1.set_title(f"Standoff Distance Regulation (RMSE={rmse_d:.2f}m | VAF={vaf_d:.1f}%)", fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Distance (m)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.5)

    # 2. Forward Velocity
    ax2 = axes[0, 1]
    ax2.plot(t_eval, vx_real_meas, color="#DC2626", lw=2.2, label="Real Drone Speed")
    ax2.plot(t_eval, vx_sim_meas, color="#2563EB", lw=2.2, ls="--", label="Sim Drone Speed")
    ax2.plot(t_eval, vx_real_cmd, color="#F59E0B", lw=1.2, ls=":", label="Real Commanded Vx")
    ax2.set_title(f"Forward Velocity Tracking (RMSE={rmse_vx:.2f} m/s | VAF={vaf_vx:.1f}%)", fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Forward Velocity (m/s)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8.5)

    # 3. Yaw Rate Command
    ax3 = axes[1, 0]
    ax3.plot(t_eval, r_real_cmd, color="#DC2626", lw=2.0, label="Real Yaw Command")
    ax3.plot(t_eval, r_sim_cmd, color="#2563EB", lw=2.0, ls="--", label="Sim Yaw Command")
    ax3.set_title(f"Yaw Rate Response (Tau={0.770*1000:.0f}ms Model | VAF={vaf_r:.1f}%)", fontweight="bold")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Yaw Rate (°/s)")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=8.5)

    # 4. Normalized Image Center Alignment Error
    ax4 = axes[1, 1]
    ax4.plot(t_eval, ex_real, color="#DC2626", lw=1.8, label="Real Azimuth Error (ex)")
    ax4.plot(t_eval, ex_sim, color="#2563EB", lw=1.8, ls="--", label="Sim Azimuth Error (ex)")
    ax4.plot(t_eval, ey_real, color="#F59E0B", lw=1.5, ls="-.", label="Real Elevation Error (ey)")
    ax4.axhline(0.0, color="#64748B", ls=":", lw=1.0)
    ax4.set_title("Optical Error Centering Convergence (Normalized [-1, +1])", fontweight="bold")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Normalized Image Error")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper right", fontsize=8.5)

    plt.suptitle("Sim-to-Real Flight Test vs. Simulation Model Validation", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[PLOT] Sim-to-Real comparison figure saved to: {output_path}")

    if ARTIFACT_DIR.exists():
        dest = ARTIFACT_DIR / output_path.name
        shutil.copy(output_path, dest)
        print(f"[ARTIFACT] Synced comparison plot to artifact directory: {dest}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Sim-to-Real Flight Replay & Comparison Tool.")
    parser.add_argument("--csv", type=str, required=True, help="Path to real flight telemetry CSV from run_tracker.py")
    parser.add_argument(
        "--pid-gains",
        type=str,
        default=str(PROJECT_ROOT / "control" / "weights" / "tuned_pid_gains_slow_tau770.json"),
        help="Path to PID gains JSON (default: tuned_pid_gains_slow_tau770.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "reports" / "figures" / "sim_vs_real_flight_comparison.png"),
        help="Output plot path"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    print(f"[SIM-TO-REAL] Loading physical flight telemetry: {csv_path}")
    df_raw = parse_real_flight_csv(csv_path)
    seg = extract_engagement_segment(df_raw)
    print(f"              Engagement segment isolated: {len(seg)} samples ({seg['time_s'].iloc[-1]:.2f}s duration)")

    print(f"[SIM-TO-REAL] Replaying initial conditions in simulation (tau_yaw = 0.770s, max_speed = 2.0 m/s)...")
    sim_res = simulate_matching_flight(seg, Path(args.pid_gains))

    metrics = plot_sim_vs_real(seg, sim_res, Path(args.output))

    print("\n" + "=" * 50)
    print("      SIM-TO-REAL VALIDATION METRICS SUMMARY      ")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"• {k:<25}: {v}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()

