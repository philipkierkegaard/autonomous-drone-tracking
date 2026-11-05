#!/usr/bin/env python3
"""
Publication-Grade Visual Showcase of Steady 1-2 m/s Target Trajectories.
Visualizes:
1. 2D Top-Down Spatial Map (with drone FOV cone & standoff target zone)
2. 3D Isometric Flight Paths
3. Velocity Magnitude Profile vs Time (1-2 m/s nominal band)
4. Acceleration Profile vs Time (<= 1.0 m/s^2 smooth ceiling)
5. Comparison vs Evasive Interceptor Trajectories (Speed & Accel Contrast)
"""

import sys
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.trajectory import StochasticTargetTrajectory

ARTIFACT_DIR = Path("/Users/philipkierkegaard/.gemini/antigravity/brain/3f6fb6d1-ed45-4315-9b5d-b3cca4e6bef7")


def main():
    json_path = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "pid_steady_tuning_suite_56.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    scenarios = data["scenarios"]

    # Select representative scenarios across distinct encounter geometries
    encounter_colors = {
        "tail_chase": "#38bdf8",       # Sky Blue
        "crossing_right": "#4ade80",   # Emerald Green
        "crossing_left": "#a855f7",    # Purple
        "head_on": "#f43f5e",          # Coral Rose
        "oblique_away": "#fbbf24",     # Warm Amber
        "oblique_inward": "#f97316",   # Orange
        "omnidirectional": "#2dd4bf"   # Teal
    }

    encounter_names = {
        "tail_chase": "Tail Chase",
        "crossing_right": "Crossing Right",
        "crossing_left": "Crossing Left",
        "head_on": "Head-On Passing",
        "oblique_away": "Oblique Away",
        "oblique_inward": "Oblique Inward",
        "omnidirectional": "Omnidirectional"
    }

    # Pick 1 sample per encounter type for clear visualization
    selected_scenarios = {}
    for sc in scenarios:
        enc = sc["encounter_type"]
        if enc not in selected_scenarios:
            selected_scenarios[enc] = sc

    # Style: Modern Dark Aerospace Theme
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 12), dpi=200, facecolor="#090d14")
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.3, 1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.30)

    # -------------------------------------------------------------
    # Panel 1: Top-Down 2D Spatial Trajectories (X vs Y)
    # -------------------------------------------------------------
    ax_map = fig.add_subplot(gs[:, 0], facecolor="#0b111e")
    ax_map.set_title("TOP-DOWN 2D SPATIAL PATHS (NED: +X North, +Y East)", fontsize=13, fontweight="bold", color="#f1f5f9", pad=12)

    # Camera Field of View (FOV) cone from Chaser at origin (0, 0)
    # 62 deg HFOV = +/- 31 deg
    fov_range = 28.0
    half_angle = np.deg2rad(31.0)
    cone_pts = np.array([
        [0.0, 0.0],
        [fov_range * np.cos(half_angle), fov_range * np.sin(half_angle)],
        [fov_range * np.cos(half_angle), -fov_range * np.sin(half_angle)],
    ])
    cone_poly = Polygon(cone_pts[:, [1, 0]], closed=True, facecolor="#38bdf8", alpha=0.07, edgecolor="#38bdf8", linestyle="--", linewidth=1.0)
    ax_map.add_patch(cone_poly)

    # Standoff zone ring (3.5m radius)
    standoff_circle = Circle((0, 0), 3.5, facecolor="none", edgecolor="#fbbf24", linestyle=":", linewidth=1.2, alpha=0.6, label="Desired Standoff (3.5m)")
    ax_map.add_patch(standoff_circle)

    # Range rings
    for r in [5.0, 10.0, 15.0, 20.0]:
        ax_map.add_patch(Circle((0, 0), r, facecolor="none", edgecolor="#334155", linestyle=":", linewidth=0.7, alpha=0.4))
        ax_map.text(0.3, r, f"{r:.0f}m", color="#64748b", fontsize=8, alpha=0.7)

    # Chaser Drone Marker at Origin
    ax_map.plot(0, 0, marker="^", color="#ffffff", markersize=10, markeredgecolor="#38bdf8", markeredgewidth=1.5, zorder=10, label="Pursuit Drone (Hover/Origin)")

    time_vec = np.linspace(0.0, 18.0, 450)

    # Storage for time series
    ts_data = {}

    for enc, sc in selected_scenarios.items():
        color = encounter_colors[enc]
        label = encounter_names[enc]
        oracle = StochasticTargetTrajectory(
            duration=sc["duration"] + 2.0,
            profile=sc["profile"],
            initial_pos=np.array(sc["initial_target_pos"]),
            initial_heading_rad=sc["initial_target_heading_rad"],
            seed=sc["seed"]
        )

        pos_list, vel_list, acc_list = [], [], []
        for t in time_vec:
            p, v, a = oracle.get_state(t)
            pos_list.append(p)
            vel_list.append(v)
            acc_list.append(a)

        pos_arr = np.array(pos_list)
        vel_arr = np.array(vel_list)
        acc_arr = np.array(acc_list)

        speeds = np.linalg.norm(vel_arr, axis=1)
        accels = np.linalg.norm(acc_arr, axis=1)

        ts_data[enc] = {
            "pos": pos_arr,
            "speeds": speeds,
            "accels": accels,
            "color": color,
            "label": label
        }

        # Plot X (North) on vertical, Y (East) on horizontal, or East (X-axis) vs North (Y-axis)
        # In multirotor standard top-down plot: East (+Y) on x-axis, North (+X) on y-axis
        ax_map.plot(pos_arr[:, 1], pos_arr[:, 0], color=color, linewidth=2.0, alpha=0.9, label=label)
        ax_map.plot(pos_arr[0, 1], pos_arr[0, 0], marker="o", color=color, markersize=5)
        ax_map.plot(pos_arr[-1, 1], pos_arr[-1, 0], marker="s", color=color, markersize=5)

    ax_map.set_xlabel("East (Y) [meters]", fontsize=11, color="#94a3b8")
    ax_map.set_ylabel("North (X) [meters]", fontsize=11, color="#94a3b8")
    ax_map.set_xlim(-16, 16)
    ax_map.set_ylim(-6, 26)
    ax_map.grid(True, linestyle=":", color="#1e293b", alpha=0.7)
    ax_map.legend(loc="upper left", fontsize=8.5, framealpha=0.6, facecolor="#090d14", edgecolor="#334155")

    # -------------------------------------------------------------
    # Panel 2: Speed vs Time |v(t)|
    # -------------------------------------------------------------
    ax_spd = fig.add_subplot(gs[0, 1], facecolor="#0b111e")
    ax_spd.set_title("VELOCITY MAGNITUDE vs TIME", fontsize=11, fontweight="bold", color="#f1f5f9", pad=8)
    ax_spd.axhspan(1.0, 2.0, color="#38bdf8", alpha=0.12, label="Nominal Band (1.0 - 2.0 m/s)")
    ax_spd.axhline(1.5, color="#38bdf8", linestyle="--", linewidth=1.2, alpha=0.7, label="Mean Speed (1.5 m/s)")

    for enc, d in ts_data.items():
        ax_spd.plot(time_vec, d["speeds"], color=d["color"], linewidth=1.8, alpha=0.85, label=d["label"])

    ax_spd.set_xlabel("Time [s]", fontsize=10, color="#94a3b8")
    ax_spd.set_ylabel("Speed [m/s]", fontsize=10, color="#94a3b8")
    ax_spd.set_ylim(0.5, 2.5)
    ax_spd.grid(True, linestyle=":", color="#1e293b", alpha=0.7)
    ax_spd.legend(loc="upper right", fontsize=7.5, framealpha=0.5, facecolor="#090d14")

    # -------------------------------------------------------------
    # Panel 3: Acceleration vs Time |a(t)|
    # -------------------------------------------------------------
    ax_acc = fig.add_subplot(gs[0, 2], facecolor="#0b111e")
    ax_acc.set_title("PHYSICAL ACCELERATION vs TIME", fontsize=11, fontweight="bold", color="#f1f5f9", pad=8)
    ax_acc.axhline(1.0, color="#f43f5e", linestyle="--", linewidth=1.4, alpha=0.85, label="Max Accel Ceiling (1.0 m/s²)")
    ax_acc.axhspan(0.0, 1.0, color="#4ade80", alpha=0.08, label="Gentle Maneuvering Zone")

    for enc, d in ts_data.items():
        ax_acc.plot(time_vec, d["accels"], color=d["color"], linewidth=1.8, alpha=0.85)

    ax_acc.set_xlabel("Time [s]", fontsize=10, color="#94a3b8")
    ax_acc.set_ylabel("Acceleration [m/s²]", fontsize=10, color="#94a3b8")
    ax_acc.set_ylim(0.0, 1.35)
    ax_acc.grid(True, linestyle=":", color="#1e293b", alpha=0.7)
    ax_acc.legend(loc="upper right", fontsize=7.5, framealpha=0.5, facecolor="#090d14")

    # -------------------------------------------------------------
    # Panel 4: Altitude Profile vs Time (NED -Z)
    # -------------------------------------------------------------
    ax_alt = fig.add_subplot(gs[1, 1], facecolor="#0b111e")
    ax_alt.set_title("ALTITUDE PROFILE (AGL)", fontsize=11, fontweight="bold", color="#f1f5f9", pad=8)
    ax_alt.axhline(1.5, color="#f43f5e", linestyle=":", linewidth=1.2, alpha=0.7, label="Floor Safety Cushion (1.5m)")

    for enc, d in ts_data.items():
        # NED Z: altitude AGL is -Z
        agl = -d["pos"][:, 2]
        ax_alt.plot(time_vec, agl, color=d["color"], linewidth=1.8, alpha=0.85)

    ax_alt.set_xlabel("Time [s]", fontsize=10, color="#94a3b8")
    ax_alt.set_ylabel("Altitude AGL [m]", fontsize=10, color="#94a3b8")
    ax_alt.set_ylim(1.0, 6.5)
    ax_alt.grid(True, linestyle=":", color="#1e293b", alpha=0.7)
    ax_alt.legend(loc="upper right", fontsize=7.5, framealpha=0.5, facecolor="#090d14")

    # -------------------------------------------------------------
    # Panel 5: Contrast: Steady vs Evasive vs Hyper-Evasive Dynamics
    # -------------------------------------------------------------
    ax_comp = fig.add_subplot(gs[1, 2], facecolor="#0b111e")
    ax_comp.set_title("SPEED REGIME COMPARISON", fontsize=11, fontweight="bold", color="#f1f5f9", pad=8)

    # Generate samples of the 3 profiles for tail chase
    profiles_to_compare = [
        ("steady_slow", "#38bdf8", "Steady (1.5 m/s)"),
        ("cruising", "#4ade80", "Cruising (3.0 m/s)"),
        ("evasive", "#fbbf24", "Evasive (4.5 m/s)"),
        ("hyper_evasive", "#f43f5e", "Hyper-Evasive (7.0 m/s)")
    ]

    for prof, pcolor, plabel in profiles_to_compare:
        orac = StochasticTargetTrajectory(duration=20.0, profile=prof, seed=123)
        p_speeds = [np.linalg.norm(orac.get_velocity(t)) for t in time_vec]
        ax_comp.plot(time_vec, p_speeds, color=pcolor, linewidth=2.0, alpha=0.9, label=plabel)

    ax_comp.set_xlabel("Time [s]", fontsize=10, color="#94a3b8")
    ax_comp.set_ylabel("Target Speed [m/s]", fontsize=10, color="#94a3b8")
    ax_comp.set_ylim(0.0, 9.5)
    ax_comp.grid(True, linestyle=":", color="#1e293b", alpha=0.7)
    ax_comp.legend(loc="upper left", fontsize=7.5, framealpha=0.5, facecolor="#090d14")

    # Supertitle & Metadata
    fig.suptitle("STEADY 1–2 m/s TARGET FLIGHT TRAJECTORIES (PID TUNING SUITE)", fontsize=15, fontweight="bold", color="#ffffff", y=0.98)

    # Save to outputs
    out_dir = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "steady_trajectories_overview.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"[SUCCESS] Saved steady trajectory plot to: {out_png}")

    # Copy to artifact directory for presentation
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    art_png = ARTIFACT_DIR / "steady_trajectories_overview.png"
    shutil.copy(out_png, art_png)
    print(f"[SUCCESS] Copied to artifact directory: {art_png}")


if __name__ == "__main__":
    main()

