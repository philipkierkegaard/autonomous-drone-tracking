#!/usr/bin/env python3
"""
Publication-ready Top-Down 2D Projection for Report.
Plots 3 representative trajectories per profile (9 total) on a clean white background.
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

    # Select 3 representative scenarios per profile: 1 Left, 1 Center, 1 Right
    # This guarantees a perfectly balanced spatial distribution without left/right bias.
    # Cruising:
    #   - Left: val_04 (crossing left, end East = -33.6m)
    #   - Center: val_01 (tail chase, forward North, East = -22.5m)
    #   - Right: val_03 (crossing right, end East = +69.8m)
    # Evasive:
    #   - Left: val_15 (crossing left, end East = -38.3m)
    #   - Center: val_11 (tail chase, straight ahead North, East in [-8.5, +10.5m], end = -2.3m)
    #   - Right: val_14 (crossing right, end East = +50.0m)
    # Hyper-evasive:
    #   - Left: val_26 (crossing left, end East = -15.1m, min East = -25.8m)
    #   - Center: val_27 (head-on passing, oscillates symmetrically East between -35.8m and +37.9m)
    #   - Right: val_25 (crossing right, end East = +55.7m, max East = +72.2m)
    selected_ids = [
        # Cruising (1 Left, 1 Center, 1 Right)
        "val_04_cruising_crossing_left_s10003",
        "val_01_cruising_tail_chase_s10000",
        "val_03_cruising_crossing_right_s10002",
        # Evasive (1 Left, 1 Center, 1 Right)
        "val_15_evasive_crossing_left_s10014",
        "val_11_evasive_tail_chase_s10010",
        "val_14_evasive_crossing_right_s10013",
        # Hyper-evasive (1 Left, 1 Center, 1 Right)
        "val_26_hyper_evasive_crossing_left_s10025",
        "val_27_hyper_evasive_head_on_s10026",
        "val_25_hyper_evasive_crossing_right_s10024",
    ]

    selected_scenarios = [s for s in scenarios if s["scenario_id"] in selected_ids]

    # Academic publication styling (Clean White)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "mathtext.fontset": "stix",
    })

    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")

    # Standard Academic Publication Palette (Classic Matplotlib / ColorBrewer tab10)
    profile_colors = {
        "cruising": "#3B5BA5",       # Standard Steel / Navy Blue (C0)
        "evasive": "#E68613",        # Standard Muted Orange (C1)
        "hyper_evasive": "#C44E52"   # Standard Crimson / Brick Red (C3)
    }
    profile_names = {
        "cruising": "Cruising",
        "evasive": "Evasive",
        "hyper_evasive": "Hyper-Evasive"
    }

    # Plot trajectories
    plotted_profiles = set()

    for sc in selected_scenarios:
        prof = sc["profile"]
        color = profile_colors[prof]
        label = profile_names[prof] if prof not in plotted_profiles else None
        if label:
            plotted_profiles.add(prof)

        traj = StochasticTargetTrajectory(
            duration=sc["duration"],
            profile=prof,
            initial_pos=sc["initial_target_pos"],
            initial_heading_rad=sc["initial_target_heading_rad"],
            seed=sc["seed"]
        )

        ts = np.linspace(0, sc["duration"], int(sc["duration"] / 0.04))
        pts = np.array([traj.get_position(t) for t in ts])

        # Top-down projection: Horizontal = East (y), Vertical = North (x)
        north = pts[:, 0]
        east = pts[:, 1]

        # Clean trajectory line without arrows
        ax.plot(east, north, color=color, linewidth=2.0, alpha=0.92, label=label)

        # Start position marker (filled circle with white edge)
        ax.plot(east[0], north[0], marker="o", markersize=5.0, color=color, markeredgecolor="white", markeredgewidth=0.8, zorder=4)

        # End position marker (subtle filled square with white edge)
        ax.plot(east[-1], north[-1], marker="s", markersize=4.5, color=color, markeredgecolor="white", markeredgewidth=0.8, zorder=4)

    # Chaser initial position (0, 0)
    ax.plot(0, 0, marker="P", markersize=9, color="#000000", markeredgecolor="white", markeredgewidth=0.8, zorder=5, label="Chaser Origin (0, 0)")

    # Grid & Spines
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.6, color="#e5e7eb", alpha=0.85)
    for spine in ax.spines.values():
        spine.set_color("#9ca3af")
        spine.set_linewidth(1.0)

    ax.tick_params(colors="#1f2937", width=0.9, length=4.5, direction="out")
    ax.set_xlabel("East Position [m]", fontweight="bold", color="#111827", labelpad=8)
    ax.set_ylabel("North Position [m]", fontweight="bold", color="#111827", labelpad=8)

    # Clean Legend in upper-left
    custom_legend = [
        Line2D([0], [0], color=profile_colors["cruising"], lw=2.0, label="Cruising"),
        Line2D([0], [0], color=profile_colors["evasive"], lw=2.0, label="Evasive"),
        Line2D([0], [0], color=profile_colors["hyper_evasive"], lw=2.0, label="Hyper-Evasive"),
        Line2D([0], [0], marker="P", color="w", markerfacecolor="#000000", markeredgecolor="white", markersize=8, label="Chaser Origin (0, 0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4b5563", markeredgecolor="white", markersize=6, label="Start ($t = 0\\,\\mathrm{s}$)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#4b5563", markeredgecolor="white", markersize=5.5, label="End ($t = 25\\,\\mathrm{s}$)"),
    ]

    ax.legend(
        handles=custom_legend,
        loc="upper right",
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#d1d5db",
        framealpha=0.95,
        borderpad=0.6,
        labelspacing=0.5
    )



    plt.tight_layout()

    out_file = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "validation_suite_report_topdown.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    brain_dir = Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/5967d853-7a7e-4cdd-8792-4a77e6d704a2")
    if brain_dir.exists():
        import shutil
        dest = brain_dir / "validation_suite_report_topdown.png"
        shutil.copy(out_file, dest)

    print(f"[SUCCESS] Exported report plot to: {out_file}")

if __name__ == "__main__":
    main()
