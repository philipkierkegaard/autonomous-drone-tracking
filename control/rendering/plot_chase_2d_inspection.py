#!/usr/bin/env python3
"""
2D Pursuit Trajectory & Aerospace Telemetry Plotter.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Generates a broadcast-quality 2D multi-panel visualization of a closed-loop chase
using the newly fine-tuned Recurrent PPO policy.
"""

import sys
import os
import shutil
from pathlib import Path
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon, Circle

from control.rendering.create_chase_movie import simulate_pursuit, get_target_trajectory
from control.predictive_controller import RecurrentVisualServoController

ARTIFACT_DIR = Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/53eb0b3e-b75c-4830-a9d4-c69e644ccb6a")


def generate_chase_2d_plot(
    model_path: str = "control/weights/recurrent_ppo_finetuned/best_model/best_model.zip",
    profile: str = "evasive",
    duration: float = 20.0,
    seed: int = 42,
    output_filename: str = "chase_2d_new_controller.png"
):
    print(f"[PLOT] Loading model: {model_path}...")
    controller = RecurrentVisualServoController(model_path=model_path)
    
    print(f"[PLOT] Simulating {profile} pursuit for {duration:.1f}s (seed {seed})...")
    traj_fn, title = get_target_trajectory(profile, duration=duration, seed=seed)
    
    sim_data = simulate_pursuit(
        trajectory_fn=traj_fn,
        duration=duration,
        dt=0.02,
        controller=controller,
    )
    
    times = sim_data["times"]
    c_pos = sim_data["chaser_pos"]  # [North, East, Down]
    t_pos = sim_data["target_pos"]
    dists = sim_data["distance"]
    err_xs = sim_data["err_x"]
    err_ys = sim_data["err_y"]
    in_views = sim_data["in_view"]
    c_vel = sim_data["chaser_vel"]
    c_yaw = sim_data["chaser_yaw"]
    cmds = sim_data["cmd"]
    
    # Coordinates in 2D Top-Down: X = East, Y = North
    c_east, c_north = c_pos[:, 1], c_pos[:, 0]
    t_east, t_north = t_pos[:, 1], t_pos[:, 0]
    
    # Setup Figure with dark aesthetics
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 11), dpi=160, facecolor="#0b0e14")
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0, 1.0],
                  left=0.05, right=0.97, top=0.92, bottom=0.07, hspace=0.35, wspace=0.25)
    
    fig.suptitle(f"Autonomous Drone Pursuit: 2D Flight Trajectory & Telemetry Analysis\n[Learned Recurrent PPO — {profile.upper()} Encounter]",
                 fontsize=14, fontweight="bold", color="#f1f2f6", y=0.98)
    
    # -------------------------------------------------------------------------
    # Panel 1: Main 2D Top-Down Flight Arena
    # -------------------------------------------------------------------------
    ax_arena = fig.add_subplot(gs[:, 0], facecolor="#10141d")
    ax_arena.grid(True, linestyle="--", alpha=0.25, color="#2c3444")
    
    # Trajectories
    ax_arena.plot(t_east, t_north, color="#ff4757", lw=2.4, label="Target Trajectory (Fleeing)", zorder=3)
    ax_arena.plot(c_east, c_north, color="#00d2d3", lw=2.6, label="Chaser Drone (Recurrent PPO)", zorder=4)
    
    # Start and End Markers
    ax_arena.scatter([t_east[0]], [t_north[0]], color="#ff4757", s=90, marker="o", edgecolors="#ffffff", lw=1.5, zorder=6, label="Target Start")
    ax_arena.scatter([c_east[0]], [c_north[0]], color="#00d2d3", s=90, marker="^", edgecolors="#ffffff", lw=1.5, zorder=6, label="Chaser Start")
    ax_arena.scatter([t_east[-1]], [t_north[-1]], color="#ff4757", s=110, marker="s", edgecolors="#ffffff", lw=1.5, zorder=6, label="Target End")
    ax_arena.scatter([c_east[-1]], [c_north[-1]], color="#00d2d3", s=130, marker="*", edgecolors="#ffffff", lw=1.5, zorder=6, label="Chaser End")
    
    # Draw Periodic Sightlines & FOV Camera Cones
    sample_steps = np.linspace(0, len(times) - 1, 10, dtype=int)
    half_hfov = np.deg2rad(60.0 / 2.0)
    fov_range = 15.0
    
    for idx in sample_steps:
        # Sightline laser
        ax_arena.plot([c_east[idx], t_east[idx]], [c_north[idx], t_north[idx]],
                      color="#2ed573" if in_views[idx] else "#ff6b81",
                      linestyle=":", lw=1.2, alpha=0.6, zorder=2)
        
        # Camera FOV wedge
        yaw = c_yaw[idx]
        p_chaser = np.array([c_east[idx], c_north[idx]])
        left_angle = yaw - half_hfov
        right_angle = yaw + half_hfov
        
        # In plot: X=East, Y=North, heading yaw=0 is North (+Y), 90 deg is East (+X)
        p_left = p_chaser + fov_range * np.array([np.sin(left_angle), np.cos(left_angle)])
        p_right = p_chaser + fov_range * np.array([np.sin(right_angle), np.cos(right_angle)])
        
        fov_poly = Polygon([p_chaser, p_left, p_right], closed=True,
                           facecolor="#2ed573", alpha=0.04, edgecolor="#2ed573",
                           linestyle="--", lw=0.6, zorder=1)
        ax_arena.add_patch(fov_poly)
        
        # Heading orientation arrow
        ax_arena.arrow(c_east[idx], c_north[idx], 2.2 * np.sin(yaw), 2.2 * np.cos(yaw),
                       head_width=1.0, head_length=1.2, fc="#00d2d3", ec="#00d2d3", alpha=0.85, zorder=5)
    
    ax_arena.set_xlabel("East (m)", color="#a4b0be", fontsize=11, fontweight="bold")
    ax_arena.set_ylabel("North (m)", color="#a4b0be", fontsize=11, fontweight="bold")
    ax_arena.set_title("TOP-DOWN 2D FLIGHT ARENA (with LOS Sightlines & Camera Cones)",
                       color="#f1f2f6", fontsize=11, fontweight="bold", pad=10)
    ax_arena.tick_params(colors="#747d8c", labelsize=9)
    ax_arena.legend(loc="upper left", facecolor="#10141d", edgecolor="#2c3444", fontsize=8.5, labelcolor="#dfe4ea")
    ax_arena.set_aspect("equal", adjustable="datalim")
    
    # -------------------------------------------------------------------------
    # Panel 2: Standoff Distance vs Time
    # -------------------------------------------------------------------------
    ax_dist = fig.add_subplot(gs[0, 1], facecolor="#10141d")
    ax_dist.grid(True, linestyle="--", alpha=0.25, color="#2c3444")
    
    ax_dist.plot(times, dists, color="#ffa502", lw=2.2, label="Actual Standoff")
    ax_dist.axhspan(5.0, 7.0, color="#2ed573", alpha=0.18, label="Goal Basket (5.0m - 7.0m)")
    ax_dist.axhline(6.0, color="#2ed573", linestyle="--", lw=1.2, alpha=0.8, label="Nominal Target (6.0m)")
    ax_dist.axhline(1.2, color="#ff4757", linestyle="-.", lw=1.2, alpha=0.8, label="Collision Threshold (1.2m)")
    
    # Highlight in-basket dwell intervals
    in_basket = (dists >= 5.0) & (dists <= 7.0) & in_views
    ax_dist.fill_between(times, 0, dists, where=in_basket, color="#2ed573", alpha=0.15)
    
    ax_dist.set_xlim(times[0], times[-1])
    ax_dist.set_ylim(0, max(16.0, np.max(dists) + 2.0))
    ax_dist.set_ylabel("Distance (m)", color="#a4b0be", fontsize=9.5, fontweight="bold")
    ax_dist.set_title("STANDOFF DISTANCE PROFILE & BASKET RETENTION", color="#f1f2f6", fontsize=10, fontweight="bold")
    ax_dist.tick_params(colors="#747d8c", labelsize=8)
    ax_dist.legend(loc="upper right", facecolor="#10141d", edgecolor="#2c3444", fontsize=7.5, labelcolor="#dfe4ea")
    
    # -------------------------------------------------------------------------
    # Panel 3: Optical Camera Centroid Error (e_x, e_y)
    # -------------------------------------------------------------------------
    ax_err = fig.add_subplot(gs[1, 1], facecolor="#10141d")
    ax_err.grid(True, linestyle="--", alpha=0.25, color="#2c3444")
    
    ax_err.plot(times, err_xs, color="#00d2d3", lw=1.8, label="Horizontal Error (ex)")
    ax_err.plot(times, err_ys, color="#ff9ff3", lw=1.8, label="Vertical Error (ey)")
    ax_err.axhspan(-0.25, 0.25, color="#2ed573", alpha=0.12, label="Center Sweet Spot (±25%)")
    ax_err.axhline(0.0, color="#747d8c", linestyle="--", lw=0.8, alpha=0.5)
    
    ax_err.set_xlim(times[0], times[-1])
    ax_err.set_ylim(-1.05, 1.05)
    ax_err.set_ylabel("Normalized Error [-1, 1]", color="#a4b0be", fontsize=9.5, fontweight="bold")
    ax_err.set_title("OPTICAL SENSOR VIEWPORT CENTERING (CAMERA FRAME)", color="#f1f2f6", fontsize=10, fontweight="bold")
    ax_err.tick_params(colors="#747d8c", labelsize=8)
    ax_err.legend(loc="upper right", facecolor="#10141d", edgecolor="#2c3444", fontsize=7.5, labelcolor="#dfe4ea")
    
    # -------------------------------------------------------------------------
    # Panel 4: Drone Speed & Commanded Velocities
    # -------------------------------------------------------------------------
    ax_vel = fig.add_subplot(gs[2, 1], facecolor="#10141d")
    ax_vel.grid(True, linestyle="--", alpha=0.25, color="#2c3444")
    
    c_speeds = np.linalg.norm(c_vel, axis=1)
    ax_vel.plot(times, c_speeds, color="#f1f2f6", lw=2.0, label="Total Speed (|v|)")
    ax_vel.plot(times, cmds[:, 0], color="#00d2d3", lw=1.6, linestyle="--", label="Cmd Forward (vx)")
    ax_vel.plot(times, cmds[:, 1], color="#54a0ff", lw=1.4, linestyle=":", label="Cmd Lateral (vy)")
    ax_vel.plot(times, cmds[:, 2], color="#ff6b6b", lw=1.4, linestyle=":", label="Cmd Vertical (vz)")
    
    ax_vel.set_xlim(times[0], times[-1])
    ax_vel.set_xlabel("Flight Time (s)", color="#a4b0be", fontsize=10, fontweight="bold")
    ax_vel.set_ylabel("Velocity (m/s)", color="#a4b0be", fontsize=9.5, fontweight="bold")
    ax_vel.set_title("AUTOPILOT VELOCITY SETPOINTS & FLIGHT SPEED", color="#f1f2f6", fontsize=10, fontweight="bold")
    ax_vel.tick_params(colors="#747d8c", labelsize=8)
    ax_vel.legend(loc="upper right", facecolor="#10141d", edgecolor="#2c3444", fontsize=7.5, labelcolor="#dfe4ea")
    
    # Save figure
    out_dir = PROJECT_ROOT / "outputs" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / output_filename
    
    plt.savefig(out_file, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"[SUCCESS] Saved 2D pursuit plot to: {out_file}")
    
    # Copy to artifact directory
    if ARTIFACT_DIR.exists():
        art_dest = ARTIFACT_DIR / output_filename
        shutil.copy(out_file, art_dest)
        print(f"[SUCCESS] Copied to artifact directory: {art_dest}")
        
    plt.close(fig)
    return out_file


if __name__ == "__main__":
    generate_chase_2d_plot()
