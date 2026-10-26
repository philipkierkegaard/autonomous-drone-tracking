#!/usr/bin/env python3
"""
Simple, Elegant Top-Down 2D Projection of All 30 Validation Trajectories.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception.
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.trajectory import StochasticTargetTrajectory

def main():
    json_path = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "validation_suite_30.json"
    if not json_path.exists():
        json_path = PROJECT_ROOT / "outputs" / "benchmarks" / "control" / "datasets" / "validation_suite_30.json"

    with open(json_path, "r") as f:
        data = json.load(f)

    scenarios = data["scenarios"]

    # Modern Dark Aerospace Theme
    fig, ax = plt.subplots(figsize=(11, 11), dpi=250, facecolor="#090d14")
    ax.set_facecolor("#090d14")

    # Cohesive color palette by flight profile
    profile_colors = {
        "cruising": "#38bdf8",       # Sky Blue (Smooth long arcs)
        "evasive": "#fbbf24",        # Warm Amber (Agile turns)
        "hyper_evasive": "#f43f5e"   # Coral Red (High-g weaves & breaks)
    }
    profile_labels = {
        "cruising": "Cruising (9 scenarios)",
        "evasive": "Evasive (14 scenarios)",
        "hyper_evasive": "Hyper-Evasive (7 scenarios)"
    }

    # 1. Subtle Camera Visual Cone from Chaser at (0, 0) pointing North (+X)
    # 62 deg HFOV = +/- 31 deg
    fov_range = 25.0
    half_angle = np.deg2rad(31.0)
    cone_pts = np.array([
        [0.0, 0.0],
        [fov_range * np.sin(half_angle), fov_range * np.cos(half_angle)],
        [-fov_range * np.sin(half_angle), fov_range * np.cos(half_angle)],
    ])
    cone_poly = Polygon(cone_pts, closed=True, facecolor="#38bdf8", alpha=0.06, edgecolor="#38bdf8", linestyle=":", linewidth=0.8)
    ax.add_patch(cone_poly)

    # Subtle range circles for spatial scale (50m, 100m)
    for r in [50, 100]:
        circle = Circle((0, 0), r, color="#334155", fill=False, linestyle=":", linewidth=0.7, alpha=0.4)
        ax.add_patch(circle)
        ax.text(r * np.cos(np.deg2rad(45)), r * np.sin(np.deg2rad(45)), f"{r}m", color="#64748b", fontsize=8, ha="left", va="bottom", alpha=0.6)

    # 2. Plot All 30 Trajectories
    plotted_labels = set()
    all_east, all_north = [], []

    for sc in scenarios:
        prof = sc["profile"]
        color = profile_colors.get(prof, "#94a3b8")
        label = profile_labels.get(prof, prof) if prof not in plotted_labels else None
        if label:
            plotted_labels.add(prof)

        traj = StochasticTargetTrajectory(
            duration=sc["duration"],
            profile=prof,
            initial_pos=sc["initial_target_pos"],
            initial_heading_rad=sc["initial_target_heading_rad"],
            seed=sc["seed"]
        )

        ts = np.linspace(0, sc["duration"], int(sc["duration"] / 0.04))
        pts = np.array([traj.get_position(t) for t in ts])
        
        # Horizontal = East (pts[:, 1]), Vertical = North (pts[:, 0])
        north = pts[:, 0]
        east = pts[:, 1]
        all_north.append(north)
        all_east.append(east)

        # Draw smooth trajectory line
        ax.plot(east, north, color=color, linewidth=1.35, alpha=0.68, label=label)

        # Mark Start Position (small subtle dot)
        ax.plot(east[0], north[0], marker="o", markersize=3.0, color=color, alpha=0.8)

        # Mark end position with subtle cross
        ax.plot(east[-1], north[-1], marker="x", markersize=3.5, color=color, alpha=0.6, markeredgewidth=1.0)

        # Add single small directional arrow at 60% path length
        idx_arrow = int(len(east) * 0.60)
        d_east = east[idx_arrow + 2] - east[idx_arrow - 2]
        d_north = north[idx_arrow + 2] - north[idx_arrow - 2]
        norm = np.hypot(d_east, d_north)
        if norm > 1e-4:
            ax.annotate(
                "",
                xy=(east[idx_arrow] + (d_east / norm) * 1.5, north[idx_arrow] + (d_north / norm) * 1.5),
                xytext=(east[idx_arrow], north[idx_arrow]),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.25,head_length=0.35",
                    color=color,
                    lw=1.2,
                    alpha=0.75
                )
            )

    # 3. Mark Chaser Origin (0, 0)
    ax.plot(0, 0, marker="P", markersize=9, color="#ffffff", markeredgecolor="#38bdf8", markeredgewidth=1.5, label="Chaser Origin (0, 0)")
    ax.annotate(
        "Chaser Origin (0, 0)\nHeading North ↑",
        xy=(0, 0), xytext=(0, -18),
        textcoords="offset points", ha="center", va="top",
        fontsize=8.5, fontweight="bold", color="#f8fafc",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", edgecolor="#475569", alpha=0.9, lw=0.8)
    )

    # Aspect Ratio & Grid
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("East Position $y$ (meters)", fontsize=11, fontweight="bold", color="#cbd5e1", labelpad=8)
    ax.set_ylabel("North Position $x$ (meters)", fontsize=11, fontweight="bold", color="#cbd5e1", labelpad=8)
    ax.set_title("Validation Suite: 30 Target Trajectories (Top-Down Projection)", fontsize=13, fontweight="bold", color="#f8fafc", pad=14)

    ax.grid(True, linestyle="--", linewidth=0.5, color="#1e293b", alpha=0.7)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#1e293b")

    # Compass Indicator in upper right corner
    ax.text(0.94, 0.96, "North (+x)", transform=ax.transAxes, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#94a3b8")
    ax.text(0.94, 0.92, "↑", transform=ax.transAxes, ha="center", va="center", fontsize=14, fontweight="bold", color="#38bdf8")
    ax.text(0.94, 0.88, "East (+y) →", transform=ax.transAxes, ha="center", va="center", fontsize=7.5, color="#64748b")

    # Clean Legend in upper left
    legend = ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="#0f172a",
        edgecolor="#334155",
        fontsize=9,
        labelcolor="#e2e8f0",
        framealpha=0.92
    )

    # Metadata Box in lower left
    meta_text = "Dataset: validation_suite_30.json\nDuration: 25.0s | Initial Dist: 5.1m – 9.6m\nEncounters: Tail chase, Crossing, Head-on, Oblique"
    ax.text(
        0.02, 0.02, meta_text,
        transform=ax.transAxes,
        fontsize=8,
        color="#94a3b8",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f172a", edgecolor="#334155", alpha=0.9, lw=0.8)
    )

    plt.tight_layout()

    out_file = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "validation_suite_30_topdown_2d.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close()

    brain_dir = Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/5967d853-7a7e-4cdd-8792-4a77e6d704a2")
    if brain_dir.exists():
        import shutil
        dest = brain_dir / "validation_suite_30_topdown_2d.png"
        shutil.copy(out_file, dest)

    print(f"[SUCCESS] Exported top-down 2D plot to: {out_file}")

if __name__ == "__main__":
    main()
