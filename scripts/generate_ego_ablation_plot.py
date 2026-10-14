#!/usr/bin/env python3
"""
Ego-Motion Ablation Analysis & Publication Plotter

Analyzes dual-tracker telemetry logs comparing:
1. Vanilla Kalman Filter (Uncompensated, prone to attitude-induced velocity spikes)
2. SO(3) Attitude Decoupled Kalman Filter (Ego-Motion Compensated)

Computes quantitative RMSE metrics, peak error reduction, and generates
a publication-grade 3-panel figure for Chapter 4/5 of the Bachelor Thesis.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless rendering for Jetson & automated runs
import matplotlib.pyplot as plt


def generate_ego_ablation_report(csv_path: str, save_plot: bool = True) -> Dict[str, Any]:
    csv_file = Path(csv_path).resolve()
    if not csv_file.exists():
        raise FileNotFoundError(f"Ablation CSV file not found: {csv_file}")

    print(f"\n[ANALYSIS] Loading ego-motion ablation dataset: {csv_file.name}")

    # Load CSV using numpy/standard library to avoid heavy dependencies if needed
    data = []
    headers = []
    with open(csv_file, "r") as f:
        for idx, line in enumerate(f):
            parts = [p.strip() for p in line.strip().split(",")]
            if idx == 0:
                headers = parts
            else:
                if len(parts) == len(headers):
                    data.append(parts)

    if len(data) < 5:
        print("[WARNING] Insufficient data points in CSV (< 5 frames). Skipping analysis.")
        return {}

    header_map = {h: i for i, h in enumerate(headers)}

    def get_col(name: str, default_val: float = 0.0) -> np.ndarray:
        if name not in header_map:
            return np.zeros(len(data), dtype=np.float64)
        col_idx = header_map[name]
        vals = []
        for row in data:
            val_str = row[col_idx]
            try:
                vals.append(float(val_str) if val_str != "" else default_val)
            except ValueError:
                vals.append(default_val)
        return np.array(vals, dtype=np.float64)

    t = get_col("timestamp")
    dt = get_col("dt", default_val=1.0 / 30.0)
    roll = get_col("roll_deg")
    pitch = get_col("pitch_deg")
    yaw = get_col("yaw_deg")
    comp_vmag = get_col("comp_vmag")
    uncomp_vmag = get_col("uncomp_vmag")
    comp_cx = get_col("comp_cx")
    comp_cy = get_col("comp_cy")
    uncomp_cx = get_col("uncomp_cx")
    uncomp_cy = get_col("uncomp_cy")
    yolo_det = get_col("yolo_detected")

    vx_body = get_col("vx_body")
    vy_body = get_col("vy_body")
    vz_body = get_col("vz_body")
    v_trans = np.hypot(vx_body, vy_body)

    # Full 6-DOF filter columns (if available)
    has_full_6dof = ("full_vmag" in header_map) and (np.max(get_col("full_vmag")) > 0.0 or np.sum(get_col("full_cx") > 0) > 5)
    full_vmag = get_col("full_vmag") if has_full_6dof else None
    full_cx = get_col("full_cx") if has_full_6dof else None
    full_cy = get_col("full_cy") if has_full_6dof else None

    # Filter to frames where the target was actively locked / coasted
    valid_mask = (comp_vmag > 0.0) | (uncomp_vmag > 0.0) | (yolo_det > 0.5)
    if np.sum(valid_mask) < 5:
        valid_mask = np.ones(len(t), dtype=bool)

    # Compute unwrapped yaw and relative heading change
    yaw_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(yaw)))
    rel_yaw = yaw_unwrapped - (yaw_unwrapped[0] if len(yaw_unwrapped) > 0 else 0.0)

    # Compute angular rates (deg/s) using timestamp coordinates
    if len(t) > 1 and (t[-1] - t[0]) > 1e-4:
        roll_rate = np.abs(np.gradient(roll, t))
        pitch_rate = np.abs(np.gradient(pitch, t))
        yaw_rate = np.abs(np.gradient(yaw_unwrapped, t))
    else:
        roll_rate = np.zeros_like(roll)
        pitch_rate = np.zeros_like(pitch)
        yaw_rate = np.zeros_like(yaw)
    total_rot_rate = roll_rate + pitch_rate + yaw_rate

    # Identify decoupled motion segments
    rot_active = total_rot_rate > 5.0
    trans_active = v_trans > 0.25  # actively walking / flying > 0.25 m/s

    maneuver_mask = valid_mask & (rot_active | trans_active)
    rot_only_mask = valid_mask & (rot_active & ~trans_active)
    trans_only_mask = valid_mask & (trans_active & ~rot_active)
    comb_mask = valid_mask & (rot_active & trans_active)

    # Compute RMSE
    rmse_uncomp_all = np.sqrt(np.mean(uncomp_vmag[valid_mask] ** 2))
    rmse_comp_all = np.sqrt(np.mean(comp_vmag[valid_mask] ** 2))

    rmse_uncomp_dyn = np.sqrt(np.mean(uncomp_vmag[maneuver_mask] ** 2)) if np.any(maneuver_mask) else rmse_uncomp_all
    rmse_comp_dyn = np.sqrt(np.mean(comp_vmag[maneuver_mask] ** 2)) if np.any(maneuver_mask) else rmse_comp_all

    peak_uncomp = np.max(uncomp_vmag[valid_mask])
    peak_comp = np.max(comp_vmag[valid_mask])

    reduction_all = (1.0 - (rmse_comp_all / max(1e-4, rmse_uncomp_all))) * 100.0
    reduction_dyn = (1.0 - (rmse_comp_dyn / max(1e-4, rmse_uncomp_dyn))) * 100.0

    stats = {
        "duration_sec": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "total_frames": len(t),
        "tracked_frames": int(np.sum(valid_mask)),
        "rmse_uncompensated_all": float(rmse_uncomp_all),
        "rmse_compensated_all": float(rmse_comp_all),
        "error_reduction_all_pct": float(reduction_all),
        "rmse_uncompensated_dynamic": float(rmse_uncomp_dyn),
        "rmse_compensated_dynamic": float(rmse_comp_dyn),
        "error_reduction_dynamic_pct": float(reduction_dyn),
        "peak_uncompensated_spike": float(peak_uncomp),
        "peak_compensated_spike": float(peak_comp),
        "peak_spike_reduction_pct": float((1.0 - (peak_comp / max(1e-4, peak_uncomp))) * 100.0),
        "has_full_6dof": bool(has_full_6dof),
    }

    if has_full_6dof and full_vmag is not None:
        rmse_full_all = np.sqrt(np.mean(full_vmag[valid_mask] ** 2))
        rmse_full_dyn = np.sqrt(np.mean(full_vmag[maneuver_mask] ** 2)) if np.any(maneuver_mask) else rmse_full_all
        peak_full = np.max(full_vmag[valid_mask])
        reduction_full_all = (1.0 - (rmse_full_all / max(1e-4, rmse_uncomp_all))) * 100.0
        reduction_full_dyn = (1.0 - (rmse_full_dyn / max(1e-4, rmse_uncomp_dyn))) * 100.0

        stats["rmse_full_all"] = float(rmse_full_all)
        stats["error_reduction_full_all_pct"] = float(reduction_full_all)
        stats["rmse_full_dynamic"] = float(rmse_full_dyn)
        stats["error_reduction_full_dynamic_pct"] = float(reduction_full_dyn)
        stats["peak_full_spike"] = float(peak_full)
        stats["peak_full_spike_reduction_pct"] = float((1.0 - (peak_full / max(1e-4, peak_uncomp))) * 100.0)

    if np.any(rot_only_mask):
        u_rot = np.sqrt(np.mean(uncomp_vmag[rot_only_mask] ** 2))
        c_rot = np.sqrt(np.mean(comp_vmag[rot_only_mask] ** 2))
        stats["has_rot_only"] = True
        stats["rmse_uncomp_rot"] = float(u_rot)
        stats["rmse_comp_rot"] = float(c_rot)
        stats["rot_reduction_pct"] = float((1.0 - (c_rot / max(1e-4, u_rot))) * 100.0)

    if np.any(trans_only_mask):
        u_trans = np.sqrt(np.mean(uncomp_vmag[trans_only_mask] ** 2))
        c_trans = np.sqrt(np.mean(comp_vmag[trans_only_mask] ** 2))
        stats["has_trans_only"] = True
        stats["rmse_uncomp_trans"] = float(u_trans)
        stats["rmse_comp_trans"] = float(c_trans)
        stats["trans_reduction_pct"] = float((1.0 - (c_trans / max(1e-4, u_trans))) * 100.0)

    # Print Clean Formatted Summary Report
    if stats.get("has_full_6dof", False):
        print("=" * 86)
        print("     EGO-MOTION ABLATION: 3-FILTER ARCHITECTURE BENCHMARK")
        print("=" * 86)
        print(f"Total Duration:        {stats['duration_sec']:.2f} s ({stats['total_frames']} frames)")
        print(f"Target Tracked Ratio:  {stats['tracked_frames']/max(1, stats['total_frames'])*100:.1f}%\n")
        print(f"{'Condition':<24} | {'Vanilla Kalman':<16} | {'Pure SO(3) (Prop)':<19} | {'Full 6-DOF (Rot+Trans)'}")
        print("-" * 86)
        print(f"{'Overall Velocity RMSE':<24} | {stats['rmse_uncompensated_all']:>11.2f} px/s  | {stats['rmse_compensated_all']:>9.2f} px/s ({stats['error_reduction_all_pct']:>+5.1f}%) | {stats['rmse_full_all']:>11.2f} px/s ({stats['error_reduction_full_all_pct']:>+5.1f}%)")
        if stats.get("has_rot_only", False):
            print(f"{'  ↳ Pure Rotation (SO3)':<24} | {stats['rmse_uncomp_rot']:>11.2f} px/s  | {stats['rmse_comp_rot']:>9.2f} px/s ({stats['rot_reduction_pct']:>+5.1f}%) |                -")
        if stats.get("has_trans_only", False):
            print(f"{'  ↳ Pure Trans (R3)':<24} | {stats['rmse_uncomp_trans']:>11.2f} px/s  | {stats['rmse_comp_trans']:>9.2f} px/s ({stats['trans_reduction_pct']:>+5.1f}%) |                -")
        print(f"{'Maneuver Window RMSE':<24} | {stats['rmse_uncompensated_dynamic']:>11.2f} px/s  | {stats['rmse_compensated_dynamic']:>9.2f} px/s ({stats['error_reduction_dynamic_pct']:>+5.1f}%) | {stats['rmse_full_dynamic']:>11.2f} px/s ({stats['error_reduction_full_dynamic_pct']:>+5.1f}%)")
        print(f"{'Peak Velocity Spike':<24} | {stats['peak_uncompensated_spike']:>11.2f} px/s  | {stats['peak_compensated_spike']:>9.2f} px/s ({stats['peak_spike_reduction_pct']:>+5.1f}%) | {stats['peak_full_spike']:>11.2f} px/s ({stats['peak_full_spike_reduction_pct']:>+5.1f}%)")
        print("=" * 86)
    else:
        print("=" * 68)
        print("     EGO-MOTION ABLATION: ROTATION & TRANSLATION DECOUPLING")
        print("=" * 68)
        print(f"Total Duration:        {stats['duration_sec']:.2f} s ({stats['total_frames']} frames)")
        print(f"Target Tracked Ratio:  {stats['tracked_frames']/max(1, stats['total_frames'])*100:.1f}%\n")
        print(f"{'Condition':<26} | {'Vanilla Kalman':<16} | {'Compensated':<14} | {'Error Drop'}")
        print("-" * 68)
        print(f"{'Overall Velocity RMSE':<26} | {stats['rmse_uncompensated_all']:>11.2f} px/s  | {stats['rmse_compensated_all']:>9.2f} px/s | {stats['error_reduction_all_pct']:>8.1f}%")
        if stats.get("has_rot_only", False):
            print(f"{'  ↳ Pure Rotation (SO3)':<26} | {stats['rmse_uncomp_rot']:>11.2f} px/s  | {stats['rmse_comp_rot']:>9.2f} px/s | {stats['rot_reduction_pct']:>8.1f}%")
        if stats.get("has_trans_only", False):
            print(f"{'  ↳ Pure Translation (R3)':<26} | {stats['rmse_uncomp_trans']:>11.2f} px/s  | {stats['rmse_comp_trans']:>9.2f} px/s | {stats['trans_reduction_pct']:>8.1f}%")
        print(f"{'Combined Maneuver RMSE':<26} | {stats['rmse_uncompensated_dynamic']:>11.2f} px/s  | {stats['rmse_compensated_dynamic']:>9.2f} px/s | {stats['error_reduction_dynamic_pct']:>8.1f}%")
        print(f"{'Peak Velocity Spike':<26} | {stats['peak_uncompensated_spike']:>11.2f} px/s  | {stats['peak_compensated_spike']:>9.2f} px/s | {stats['peak_spike_reduction_pct']:>8.1f}%")
        print("=" * 68)

    if save_plot:
        plot_path = csv_file.with_suffix(".png")
        _plot_ablation_figure(t, roll, pitch, rel_yaw, v_trans, comp_cx, comp_cy, uncomp_cx, uncomp_cy, full_cx, full_cy, comp_vmag, uncomp_vmag, full_vmag, valid_mask, rot_only_mask, trans_only_mask, stats, plot_path)
        print(f"[SUCCESS] Publication comparison plot saved to: {plot_path}")
        stats["plot_path"] = str(plot_path)

    return stats


def _plot_ablation_figure(
    t, roll, pitch, rel_yaw, v_trans, comp_cx, comp_cy, uncomp_cx, uncomp_cy, full_cx, full_cy,
    comp_vmag, uncomp_vmag, full_vmag, valid_mask, rot_only_mask, trans_only_mask, stats, out_path: Path
):
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8

    fig, axes = plt.subplots(3, 1, figsize=(11, 9.0), sharex=True, dpi=300)
    plt.subplots_adjust(hspace=0.28)

    has_full = stats.get("has_full_6dof", False) and full_vmag is not None

    # 1. Top Panel: Chase Drone Kinematics (Attitude & Translational Speed)
    ax0 = axes[0]
    l1 = ax0.plot(t, roll, label=r"Chase Roll ($\phi$)", color="#e67e22", linewidth=1.6)
    l2 = ax0.plot(t, pitch, label=r"Chase Pitch ($\theta$)", color="#27ae60", linewidth=1.6)
    l3 = ax0.plot(t, rel_yaw, label=r"Relative Yaw ($\Delta\psi$)", color="#2980b9", linewidth=1.8, linestyle="--")
    ax0.axhline(0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
    ax0.set_ylabel("Attitude [deg]", fontsize=10, fontweight="bold")
    title_text = "Ego-Motion Decoupling: 3-Filter Parallel Benchmark (v_true = 0)" if has_full else "Ego-Motion Decoupling Validation: Rotation vs. Translation (v_true = 0)"
    ax0.set_title(title_text, fontsize=13, fontweight="bold", pad=10)
    ax0.grid(True, linestyle=":", alpha=0.6)

    # Dynamic y-limits based on attitude range
    y_min_data = min(-30.0, float(np.min(roll)), float(np.min(pitch)), float(np.min(rel_yaw)))
    y_max_data = max(35.0, float(np.max(roll)), float(np.max(pitch)), float(np.max(rel_yaw)))
    y_pad = (y_max_data - y_min_data) * 0.1
    y_plot_min = y_min_data - y_pad
    y_plot_max = y_max_data + y_pad
    ax0.set_ylim(y_plot_min, y_plot_max)

    # Secondary y-axis for translation speed if active
    lines = l1 + l2 + l3
    if np.max(v_trans) > 0.15:
        ax0_r = ax0.twinx()
        l4 = ax0_r.plot(t, v_trans, label="Translational Speed ($v_{xy}$)", color="#8e44ad", linestyle="-.", linewidth=1.6)
        ax0_r.set_ylabel("Speed [m/s]", fontsize=10, fontweight="bold", color="#8e44ad")
        ax0_r.tick_params(axis="y", labelcolor="#8e44ad")
        lines += l4

    labels = [l.get_label() for l in lines]
    ax0.legend(lines, labels, loc="upper right", framealpha=0.9, fontsize=9)

    # 2. Middle Panel: Estimated Target Image-Plane Center Position
    ax1 = axes[1]
    ax1.plot(t, uncomp_cx, label="Vanilla Kalman Center X", color="#e74c3c", linestyle=":", linewidth=1.4)
    ax1.plot(t, comp_cx, label="Pure SO(3) Proposed Center X", color="#2980b9", linewidth=1.8)
    if has_full and full_cx is not None:
        ax1.plot(t, full_cx, label="Full 6-DOF (Rot+Trans) Center X", color="#8e44ad", linestyle="--", linewidth=1.5)
    ax1.set_ylabel("Image X [px]", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # 3. Bottom Panel: Estimated Target Velocity Magnitude (The Benchmark)
    ax2 = axes[2]
    ax2.plot(t, uncomp_vmag, label=f"Vanilla Kalman (RMSE: {stats['rmse_uncompensated_all']:.1f} px/s)", color="#e74c3c", linewidth=1.6, alpha=0.75)
    ax2.plot(t, comp_vmag, label=f"Pure SO(3) Proposed (RMSE: {stats['rmse_compensated_all']:.1f} px/s, {stats['error_reduction_all_pct']:+.0f}%)", color="#2980b9", linewidth=2.0)
    if has_full and full_vmag is not None:
        ax2.plot(t, full_vmag, label=f"Full 6-DOF Rot+Trans (RMSE: {stats['rmse_full_all']:.1f} px/s, {stats['error_reduction_full_all_pct']:+.0f}%)", color="#8e44ad", linewidth=1.8, linestyle="--")
    ax2.axhline(0, color="#2ecc71", linestyle="-", linewidth=1.2, alpha=0.8, label="Ground Truth Baseline (0.0 px/s)")
    ax2.set_xlabel("Time [seconds]", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Target Velocity [px/s]", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # Annotation box with detailed breakdown
    summary_parts = [
        f"Pure SO(3) Error Drop: {stats['error_reduction_all_pct']:+.1f}% (Peak Cut: {stats['peak_spike_reduction_pct']:+.1f}%)"
    ]
    if has_full:
        summary_parts.append(f"Full 6-DOF Error Drop: {stats['error_reduction_full_all_pct']:+.1f}% (Peak Cut: {stats['peak_full_spike_reduction_pct']:+.1f}%)")
    if stats.get("has_rot_only", False):
        summary_parts.append(f"Pure Rotation Window: {stats['rot_reduction_pct']:+.1f}%")
    if stats.get("has_trans_only", False):
        summary_parts.append(f"Pure Translation Window: {stats['trans_reduction_pct']:+.1f}%")

    summary_text = "\n".join(summary_parts)
    ax2.text(0.02, 0.92, summary_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#bdc3c7", alpha=0.9))

    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and plot ego-motion ablation telemetry dataset.")
    parser.add_argument("csv", nargs="?", default=None, help="Path to ablation CSV file. If omitted, uses latest file in outputs/ablation/.")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate PNG plot, output metrics only.")
    args = parser.parse_args()

    target_csv = args.csv
    if target_csv is None:
        ablation_dir = Path("outputs/ablation")
        if ablation_dir.exists():
            csv_candidates = sorted(list(ablation_dir.glob("ego_motion_ablation_*.csv")))
            if csv_candidates:
                target_csv = str(csv_candidates[-1])

    if target_csv is None:
        print("[ERROR] No ablation CSV specified and no files found in outputs/ablation/.")
        sys.exit(1)

    generate_ego_ablation_report(target_csv, save_plot=(not args.no_plot))
