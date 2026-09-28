#!/usr/bin/env python3
"""
Simple 2D Top-Down Projection of Stochastic Drone Trajectories.
Generates and plots multiple stochastic trajectories viewed from above.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure control_model is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from stochastic_trajectory import StochasticTargetTrajectory

ARTIFACT_DIR = Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/42b621c5-bd5c-42b8-ae83-87208090e0a7")


def plot_trajectories_2d():
    # 5 representative trajectories with varying dynamics
    trajectories_config = [
        {"name": "Cruising", "profile": "cruising", "seed": 10, "color": "#1f77b4", "linestyle": "-"},
        {"name": "Evasive", "profile": "evasive", "seed": 42, "color": "#ff7f0e", "linestyle": "-"},
        {"name": "Aerobatic", "profile": "aerobatic", "seed": 7, "color": "#2ca02c", "linestyle": "-"},
        {"name": "Hyper-Evasive", "profile": "hyper_evasive", "seed": 99, "color": "#d62728", "linestyle": "-"},
        {"name": "Sprint Break", "profile": "sprint_break", "seed": 55, "color": "#9467bd", "linestyle": "-"},
    ]

    duration = 25.0  # seconds
    time_eval = np.linspace(0.0, duration, 600)
    origin = np.array([0.0, 0.0, -5.0])  # All start from common origin (0, 0)

    fig, ax = plt.subplots(figsize=(9, 8), dpi=150)

    for cfg in trajectories_config:
        traj = StochasticTargetTrajectory(
            duration=duration,
            profile=cfg["profile"],
            initial_pos=origin,
            seed=cfg["seed"]
        )

        # Sample positions over time
        positions = np.array([traj.get_position(t) for t in time_eval])
        north = positions[:, 0]  # X in NED
        east = positions[:, 1]   # Y in NED

        # In top-down view: X-axis = East (m), Y-axis = North (m)
        ax.plot(
            east, north,
            label=f"{traj.p.mean_speed:.1f} m/s, a_max={traj.p.max_accel:.1f}",
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=2.0,
            alpha=0.85
        )

        # Mark endpoint
        ax.plot(east[-1], north[-1], marker="s", markersize=6, color=cfg["color"], alpha=0.9)

    # Mark common start point
    ax.plot(0, 0, marker="o", markersize=9, color="black", label="Start (0, 0)", zorder=10)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("East (m)", fontsize=11)
    ax.set_ylabel("North (m)", fontsize=11)
    ax.set_title("2D Stochastic Drone Trajectories (Top-Down Projection)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="best", framealpha=0.9, fontsize=9.5)

    plt.tight_layout()

    out_path = BASE_DIR / "stochastic_trajectories_2d.png"
    plt.savefig(out_path, dpi=200)
    print(f"[SUCCESS] Saved plot to: {out_path}")

    # Copy to artifact directory if available
    if ARTIFACT_DIR.exists():
        artifact_file = ARTIFACT_DIR / "stochastic_trajectories_2d.png"
        import shutil
        shutil.copy(out_path, artifact_file)
        print(f"[SUCCESS] Copied to artifacts: {artifact_file}")

    plt.close()


if __name__ == "__main__":
    plot_trajectories_2d()
