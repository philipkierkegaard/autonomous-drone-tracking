#!/usr/bin/env python3
"""
Yaw-Only Stationary Hover Orbit Simulation & Verification Node.
Simulates a stationary drone holding altitude and position in GUIDED mode (yaw-only tracking),
while a target moves in circles around it (emulating a person or drone orbiting during flight test).

Outputs:
1. Multi-panel publication diagnostic plot (PNG)
2. Synchronized 50fps/25fps broadcast MP4 video with top-down radar and FPV camera HUD
3. Quantitative tracking metrics (lag, FOV retention, rate saturation)
"""

import sys
import os
import math
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.simulation import FastPixhawkQuadSim
from control.pid_controller import KinematicVisualServoController

ARTIFACT_DIR = Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/7dc8da15-984b-4499-8ca6-5e32efd45198")


class CircularOrbitTargetOracle:
    """Target oracle generating smooth circular orbits or reversal patterns around the origin."""
    def __init__(
        self,
        radius: float = 4.0,
        speed: float = 0.8,
        altitude: float = -2.5,
        reversal: bool = False,
        reversal_period: float = 12.0,
    ):
        self.radius = float(radius)
        self.speed = float(speed)
        self.altitude = float(altitude)
        self.omega = self.speed / self.radius  # rad/s
        self.reversal = reversal
        self.reversal_period = float(reversal_period)

    def get_angle(self, t: float) -> float:
        if not self.reversal:
            return self.omega * t
        else:
            # First half: clockwise, pause 1s, second half: counter-clockwise
            p = self.reversal_period
            cycle = t % (2 * p + 2.0)
            if cycle <= p:
                return self.omega * cycle
            elif cycle <= p + 1.0:
                return self.omega * p
            else:
                dt_back = cycle - (p + 1.0)
                return self.omega * p - self.omega * dt_back

    def get_position(self, t: float) -> np.ndarray:
        theta = self.get_angle(t)
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        z = self.altitude
        return np.array([x, y, z], dtype=np.float64)

    def get_velocity(self, t: float) -> np.ndarray:
        theta = self.get_angle(t)
        # Numerical derivative
        eps = 1e-3
        p1 = self.get_position(t + eps)
        p0 = self.get_position(t - eps)
        return (p1 - p0) / (2.0 * eps)

    def get_acceleration(self, t: float) -> np.ndarray:
        eps = 1e-3
        v1 = self.get_velocity(t + eps)
        v0 = self.get_velocity(t - eps)
        return (v1 - v0) / (2.0 * eps)

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.get_position(t), self.get_velocity(t), self.get_acceleration(t)


def simulate_orbit_tracking(
    oracle: CircularOrbitTargetOracle,
    kp_yaw: float = 80.0,
    ki_yaw: float = 0.0,
    kd_yaw: float = 15.0,
    max_yawspeed_deg: float = 15.0,
    max_accel_yaw: float = 30.0,
    duration: float = 25.0,
    dt: float = 0.02,
    hfov_deg: float = 62.0,
    vfov_deg: float = 38.0,
    desired_bbox_size: float = 110.0,
) -> Dict[str, Any]:
    """Runs high-fidelity Pixhawk simulation of stationary yaw-only tracking on circular orbit."""
    sim = FastPixhawkQuadSim(dt=dt)
    drone_alt = oracle.altitude
    sim.reset(initial_pos=[0.0, 0.0, drone_alt], initial_yaw=0.0, initial_vel=np.zeros(3))

    controller = KinematicVisualServoController(
        camera_uptilt_deg=15.0,
        hfov_deg=hfov_deg,
        vfov_deg=vfov_deg,
        desired_bbox_size=desired_bbox_size,
        use_bbox_size=True,
        scale_mode="width",
        kp_yaw=kp_yaw,
        ki_yaw=ki_yaw,
        kd_yaw=kd_yaw,
        max_yawspeed_deg=max_yawspeed_deg,
        max_accel_yaw=max_accel_yaw,
        min_limits=np.array([0.0, 0.0, -max_yawspeed_deg]),
        max_limits=np.array([0.0, 0.0, max_yawspeed_deg]),
        enable_lateral_strafe=False
    )

    steps = int(round(duration / dt))
    t_hist, tgt_pos_hist, drone_yaw_hist, drone_yaw_rate_hist = [], [], [], []
    cmd_yaw_hist, err_x_hist, in_view_hist, bearing_hist = [], [], [], []
    bbox_hist = []

    half_hfov_rad = np.deg2rad(hfov_deg / 2.0)

    for i in range(steps):
        t = i * dt
        tgt_pos = oracle.get_position(t)

        # Camera projection
        err_x, err_y, in_view, dist, bw, bh = sim.project_to_camera(
            tgt_pos,
            hfov_deg=hfov_deg,
            vfov_deg=vfov_deg,
            target_w_m=0.35,
            target_h_m=0.20,
            img_w=1280,
            img_h=720,
            return_bbox=True,
            return_unclipped=True
        )

        ctrl_telem = {
            "status": "LOCKED" if in_view else "SEARCHING",
            "error_x": float(np.clip(err_x, -1.0, 1.0)),
            "error_y": float(np.clip(err_y, -1.0, 1.0)),
            "error_range": float((desired_bbox_size - bw) / desired_bbox_size),
            "target_size": float(bw),
            "desired_size": desired_bbox_size,
            "bbox_w": float(bw),
            "bbox_h": float(bh),
            "w_norm": float(bw / 1280.0),
            "scale_mode": "width"
        }

        # Compute controller command
        raw_cmd = controller.compute_cmd(
            telemetry=ctrl_telem,
            drone_pitch=sim.pitch,
            drone_roll=sim.roll,
            dt=dt
        )
        v_fwd, v_right, v_down, yawspeed = controller.to_velocity_body(raw_cmd)

        # Enforce strict Yaw-Only mode (0 forward, 0 strafe, 0 climb/descend)
        v_fwd = 0.0
        v_right = 0.0
        v_down = 0.0
        # Enforce yaw rate limits
        yawspeed = float(np.clip(yawspeed, -max_yawspeed_deg, max_yawspeed_deg))

        # Command to Pixhawk Simulator (in NED body frame: [vx, vy, vz, yawspeed])
        sim.step(np.array([v_fwd, v_right, v_down, yawspeed], dtype=np.float64))

        # True target bearing relative to world origin
        true_bearing_rad = math.atan2(tgt_pos[1] - sim.pos[1], tgt_pos[0] - sim.pos[0])

        t_hist.append(t)
        tgt_pos_hist.append(tgt_pos.copy())
        drone_yaw_hist.append(sim.yaw)
        drone_yaw_rate_hist.append(np.rad2deg(sim.yaw_rate))
        cmd_yaw_hist.append(yawspeed)
        err_x_hist.append(err_x)
        in_view_hist.append(in_view)
        bearing_hist.append(true_bearing_rad)
        bbox_hist.append((bw, bh, err_x, err_y))

    # Calculate metrics
    drone_yaw_arr = np.array(drone_yaw_hist)
    bearing_arr = np.array(bearing_hist)
    # Angular lag: shortest angle diff
    lag_rad = (bearing_arr - drone_yaw_arr + np.pi) % (2.0 * np.pi) - np.pi
    lag_deg = np.rad2deg(lag_rad)

    fov_retention_pct = (np.sum(in_view_hist) / len(in_view_hist)) * 100.0
    mean_abs_lag = float(np.mean(np.abs(lag_deg)))
    max_abs_lag = float(np.max(np.abs(lag_deg)))
    peak_yaw_rate = float(np.max(np.abs(drone_yaw_rate_hist)))

    return {
        "times": np.array(t_hist),
        "target_pos": np.array(tgt_pos_hist),
        "drone_yaw": drone_yaw_arr,
        "drone_yaw_rate": np.array(drone_yaw_rate_hist),
        "cmd_yaw": np.array(cmd_yaw_hist),
        "err_x": np.array(err_x_hist),
        "in_view": np.array(in_view_hist),
        "bearing": bearing_arr,
        "lag_deg": lag_deg,
        "bbox": bbox_hist,
        "metrics": {
            "fov_retention_pct": fov_retention_pct,
            "mean_abs_lag_deg": mean_abs_lag,
            "max_abs_lag_deg": max_abs_lag,
            "peak_yaw_rate_deg_s": peak_yaw_rate,
            "configured_max_yawspeed": max_yawspeed_deg,
            "configured_max_accel": max_accel_yaw,
            "kp_yaw": kp_yaw,
            "kd_yaw": kd_yaw,
        }
    }


def generate_diagnostic_plots(
    results_dict: Dict[str, Dict[str, Any]],
    output_png: Path
):
    """Creates a 4-panel publication diagnostic plot comparing tracking response."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor("#ffffff")

    colors = {
        "Slow Orbit (11.5°/s, Gentle: kp=80, kd=15)": "#1e90ff",
        "Moderate Orbit (18.6°/s, Tuned: kp=237, kd=43)": "#2ed573",
        "Reversal Orbit (14.3°/s, Gentle: kp=80, kd=15)": "#ff4757",
    }

    # Plot 1: True Bearing vs Drone Heading
    ax1 = axes[0, 0]
    ax1.set_facecolor("#f8f9fa")
    ax1.set_title("Target True Bearing vs Drone Yaw Angle", fontsize=12, fontweight="bold", pad=10)
    for name, res in results_dict.items():
        t = res["times"]
        c = colors.get(name, "#333333")
        bearing_deg = np.rad2deg(res["bearing"])
        yaw_deg = np.rad2deg(res["drone_yaw"])
        # Unwrap for clean continuous lines
        bearing_unwrapped = np.rad2deg(np.unwrap(res["bearing"]))
        yaw_unwrapped = np.rad2deg(np.unwrap(res["drone_yaw"]))
        ax1.plot(t, bearing_unwrapped, "--", color=c, alpha=0.5, label=f"Target ({name.split('(')[0].strip()})")
        ax1.plot(t, yaw_unwrapped, "-", color=c, lw=2, label=f"Drone Yaw ({name.split('(')[0].strip()})")
    ax1.set_xlabel("Time (s)", fontsize=10)
    ax1.set_ylabel("Cumulative Angle (deg)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=8)

    # Plot 2: Azimuth Tracking Lag Error
    ax2 = axes[0, 1]
    ax2.set_facecolor("#f8f9fa")
    ax2.set_title("Azimuth Tracking Error (Lag Angle)", fontsize=12, fontweight="bold", pad=10)
    hfov_bound = 62.0 / 2.0
    ax2.axhline(+hfov_bound, color="orange", linestyle="--", lw=1.5, label="Camera FOV Limit (+31°)")
    ax2.axhline(-hfov_bound, color="orange", linestyle="--", lw=1.5, label="Camera FOV Limit (-31°)")
    ax2.axhspan(-hfov_bound, +hfov_bound, color="green", alpha=0.06, label="Visible Camera Window")
    for name, res in results_dict.items():
        t = res["times"]
        c = colors.get(name, "#333333")
        ax2.plot(t, res["lag_deg"], "-", color=c, lw=1.8, label=name)
    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.set_ylabel("Lag Error (deg)", fontsize=10)
    ax2.set_ylim(-40, 40)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8)

    # Plot 3: Commanded Yaw Rate vs Actual Vehicle Yaw Rate
    ax3 = axes[1, 0]
    ax3.set_facecolor("#f8f9fa")
    ax3.set_title("Commanded vs Actual Drone Yaw Rate (Capped & Slewed)", fontsize=12, fontweight="bold", pad=10)
    for name, res in results_dict.items():
        t = res["times"]
        c = colors.get(name, "#333333")
        ax3.plot(t, res["cmd_yaw"], "--", color=c, alpha=0.5, label=f"Cmd ({name.split('(')[0].strip()})")
        ax3.plot(t, res["drone_yaw_rate"], "-", color=c, lw=2, label=f"Actual ({name.split('(')[0].strip()})")
    ax3.axhline(+15.0, color="gray", linestyle=":", label="Configured Max Speed (±15°/s)")
    ax3.axhline(-15.0, color="gray", linestyle=":")
    ax3.set_xlabel("Time (s)", fontsize=10)
    ax3.set_ylabel("Yaw Rate (deg/s)", fontsize=10)
    ax3.set_ylim(-20, 20)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="lower right", fontsize=8)

    # Plot 4: 2D Top-Down Radar Map with Orientation Vectors
    ax4 = axes[1, 1]
    ax4.set_facecolor("#f8f9fa")
    ax4.set_title("2D Top-Down Orbit Trajectory & Heading Vectors", fontsize=12, fontweight="bold", pad=10)
    # Circle radius 4m
    circle = patches.Circle((0, 0), 4.0, fill=False, color="#bbbbbb", linestyle="--", lw=1.5, label="Target 4m Orbit Path")
    ax4.add_patch(circle)
    # Chaser stationary hover position
    ax4.scatter([0.0], [0.0], color="#1e90ff", s=140, zorder=5, label="Stationary Drone (0,0)")

    # Sample arrows for one run
    sample_res = list(results_dict.values())[0]
    sample_times = sample_res["times"]
    step_sample = len(sample_times) // 8
    for idx in range(0, len(sample_times), step_sample):
        tgt = sample_res["target_pos"][idx]
        y_drone = sample_res["drone_yaw"][idx]
        # Target point
        ax4.scatter([tgt[0]], [tgt[1]], color="#ff4757", s=35, zorder=4)
        # Heading arrow
        arrow_len = 1.8
        dx = arrow_len * np.cos(y_drone)
        dy = arrow_len * np.sin(y_drone)
        ax4.arrow(0.0, 0.0, dx, dy, head_width=0.25, head_length=0.3, fc="#1e90ff", ec="#1e90ff", alpha=0.7, zorder=4)

    ax4.set_xlabel("X (m, Forward/North)", fontsize=10)
    ax4.set_ylabel("Y (m, Right/East)", fontsize=10)
    ax4.set_xlim(-5.5, 5.5)
    ax4.set_ylim(-5.5, 5.5)
    ax4.set_aspect("equal")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Exported orbit diagnostic plot: {output_png}")


def render_orbit_video(
    sim_data: Dict[str, Any],
    output_mp4: Path,
    fps: int = 25,
    title: str = "STATIONARY YAW-ONLY HOVER ORBIT TRACKING"
):
    """Renders a synchronized broadcast MP4 video of the stationary drone tracking an orbiting target."""
    times = sim_data["times"]
    target_pos = sim_data["target_pos"]
    drone_yaw = sim_data["drone_yaw"]
    drone_yaw_rate = sim_data["drone_yaw_rate"]
    cmd_yaw = sim_data["cmd_yaw"]
    lag_deg = sim_data["lag_deg"]
    in_view = sim_data["in_view"]
    bbox = sim_data["bbox"]

    dt_sim = float(times[1] - times[0])
    sim_steps_per_frame = max(1, int(round((1.0 / fps) / dt_sim)))
    frame_indices = np.arange(0, len(times), sim_steps_per_frame)

    fig_w, fig_h = 14.4, 8.0
    dpi = 100
    width_px = int(fig_w * dpi)
    height_px = int(fig_h * dpi)
    if width_px % 2 != 0: width_px += 1
    if height_px % 2 != 0: height_px += 1

    ffmpeg_bin = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    cmd = [
        ffmpeg_bin, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width_px}x{height_px}", "-pix_fmt", "rgba",
        "-r", str(fps), "-i", "-", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-crf", "18", str(output_mp4)
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    for frame_i, idx in enumerate(frame_indices):
        t = times[idx]
        t_pos = target_pos[idx]
        y_drone = drone_yaw[idx]
        y_rate = drone_yaw_rate[idx]
        c_rate = cmd_yaw[idx]
        lag = lag_deg[idx]
        iv = in_view[idx]
        bw, bh, ex, ey = bbox[idx]

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#ffffff")
        gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.2], height_ratios=[1.2, 0.8],
                               left=0.05, right=0.97, top=0.92, bottom=0.08, wspace=0.18, hspace=0.25)

        fig.suptitle(f"{title} | t = {t:5.2f} s", fontsize=15, fontweight="bold", color="#1e272e")

        # -------------------------------------------------------------
        # Left Panel: 2D Tactical Top-Down Radar Map
        # -------------------------------------------------------------
        ax_map = fig.add_subplot(gs[:, 0])
        ax_map.set_facecolor("#f1f2f6")
        ax_map.set_title("2D Top-Down Tactical Orbit Radar", fontsize=12, fontweight="bold", color="#2f3542", pad=8)

        # Concentric range rings
        for r_m in [2.0, 4.0, 6.0]:
            c_ring = patches.Circle((0, 0), r_m, fill=False, color="#ced6e0", linestyle=":", lw=1.2)
            ax_map.add_patch(c_ring)
            ax_map.text(0.1, r_m + 0.1, f"{r_m:.0f}m", color="#747d8c", fontsize=8)

        # Orbit Path
        c_orbit = patches.Circle((0, 0), 4.0, fill=False, color="#70a1ff", linestyle="--", lw=1.5, label="Target 4m Orbit")
        ax_map.add_patch(c_orbit)

        # Target breadcrumb trail
        tail_start = max(0, idx - 150)
        ax_map.plot(target_pos[tail_start:idx+1, 0], target_pos[tail_start:idx+1, 1], color="#ff6b81", lw=2, alpha=0.6)

        # Current target location
        ax_map.scatter([t_pos[0]], [t_pos[1]], color="#ff4757", s=140, edgecolors="#ffffff", lw=2, zorder=6, label="Orbiting Target")

        # Stationary Chaser Drone at (0, 0)
        ax_map.scatter([0.0], [0.0], color="#2ed573" if iv else "#ffa502", s=200, edgecolors="#ffffff", lw=2.5, zorder=7, label="Stationary Drone [0,0]")

        # Camera FOV wedge (62 deg)
        half_fov = np.deg2rad(62.0 / 2.0)
        wedge_r = 5.2
        w_left = y_drone - half_fov
        w_right = y_drone + half_fov
        fov_pts = np.array([
            [0.0, 0.0],
            [wedge_r * np.cos(w_left), wedge_r * np.sin(w_left)],
            [wedge_r * np.cos(w_right), wedge_r * np.sin(w_right)]
        ])
        fov_poly = patches.Polygon(fov_pts, closed=True, color="#2ed573" if iv else "#ffa502", alpha=0.15, zorder=3)
        ax_map.add_patch(fov_poly)

        # Heading vector line
        hx = wedge_r * np.cos(y_drone)
        hy = wedge_r * np.sin(y_drone)
        ax_map.plot([0, hx], [0, hy], color="#2ed573" if iv else "#ffa502", lw=2.5, zorder=5)

        ax_map.set_xlim(-5.5, 5.5)
        ax_map.set_ylim(-5.5, 5.5)
        ax_map.set_aspect("equal")
        ax_map.set_xlabel("X (meters, North/Forward)", fontsize=10)
        ax_map.set_ylabel("Y (meters, East/Right)", fontsize=10)
        ax_map.grid(True, linestyle=":", alpha=0.5)
        ax_map.legend(loc="upper left", fontsize=8)

        # -------------------------------------------------------------
        # Right Top Panel: On-Board FPV Camera Synthetic Projection
        # -------------------------------------------------------------
        ax_cam = fig.add_subplot(gs[0, 1])
        ax_cam.set_facecolor("#1e272e")
        ax_cam.set_title("On-Board Fixed Camera FPV (62° HFOV)", fontsize=12, fontweight="bold", color="#2f3542", pad=8)

        # Frame center crosshairs
        ax_cam.axhline(0, color="#57606f", linestyle="--", lw=1)
        ax_cam.axvline(0, color="#57606f", linestyle="--", lw=1)
        ax_cam.plot([0], [0], "+", color="#747d8c", ms=14, mew=1.5)

        # Draw target bbox if in view
        if iv:
            # normalized error [-1, 1] to normalized coord
            cx_norm = ex
            cy_norm = ey
            w_norm = (bw / 1280.0) * 2.0
            h_norm = (bh / 720.0) * 2.0
            rect = patches.Rectangle(
                (cx_norm - w_norm / 2.0, cy_norm - h_norm / 2.0),
                w_norm, h_norm,
                linewidth=2.5,
                edgecolor="#2ed573",
                facecolor="#2ed573",
                alpha=0.25,
                zorder=5
            )
            ax_cam.add_patch(rect)
            ax_cam.text(cx_norm, cy_norm - h_norm / 2.0 - 0.08, f"TARGET ({bw:.0f}px)", color="#2ed573", fontsize=9, fontweight="bold", ha="center")
            status_txt = "STATUS: LOCKED"
            status_col = "#2ed573"
        else:
            status_txt = "STATUS: OUT OF VIEW"
            status_col = "#ff4757"

        ax_cam.text(-0.95, 0.82, status_txt, color=status_col, fontsize=10, fontweight="bold")
        ax_cam.text(-0.95, 0.68, f"MODE: YAW-ONLY (Vx=0, Vy=0, Vz=0)", color="#70a1ff", fontsize=9)
        ax_cam.text(-0.95, 0.54, f"Azimuth Lag: {lag:+5.1f}°", color="#ffa502" if abs(lag) > 15 else "#2ed573", fontsize=9)

        ax_cam.set_xlim(-1.0, 1.0)
        ax_cam.set_ylim(1.0, -1.0)  # Image Y down
        ax_cam.set_xlabel("Normalized Horizontal Image Plane ($e_x$)", fontsize=9)
        ax_cam.set_ylabel("Vertical ($e_y$)", fontsize=9)

        # -------------------------------------------------------------
        # Right Bottom Panel: Live Yaw Dynamics Stripchart
        # -------------------------------------------------------------
        ax_dyn = fig.add_subplot(gs[1, 1])
        ax_dyn.set_facecolor("#f8f9fa")
        ax_dyn.set_title("Live Yaw Rate Response & Rate Limiter", fontsize=11, fontweight="bold", color="#2f3542", pad=6)

        window_start = max(0, idx - 250)
        w_t = times[window_start:idx+1]
        w_cmd = cmd_yaw[window_start:idx+1]
        w_actual = drone_yaw_rate[window_start:idx+1]

        ax_dyn.plot(w_t, w_cmd, "--", color="#70a1ff", lw=1.8, label="Cmd Yaw Rate")
        ax_dyn.plot(w_t, w_actual, "-", color="#2ed573", lw=2.2, label="Actual Yaw Rate")
        ax_dyn.axhline(+15.0, color="#ff4757", linestyle=":", lw=1.2, label="Rate Cap (±15°/s)")
        ax_dyn.axhline(-15.0, color="#ff4757", linestyle=":", lw=1.2)

        ax_dyn.set_xlim(times[window_start], times[window_start] + 5.5)
        ax_dyn.set_ylim(-20, 20)
        ax_dyn.set_xlabel("Time (s)", fontsize=9)
        ax_dyn.set_ylabel("Yaw Rate (°/s)", fontsize=9)
        ax_dyn.grid(True, linestyle=":", alpha=0.5)
        ax_dyn.legend(loc="upper right", fontsize=8)

        fig.canvas.draw()
        rgba_buf = np.asarray(fig.canvas.buffer_rgba())
        proc.stdin.write(rgba_buf.tobytes())
        plt.close(fig)

    proc.stdin.close()
    proc.wait()
    print(f"[VIDEO] Successfully encoded orbit simulation video: {output_mp4}")


def main():
    parser = argparse.ArgumentParser(description="Simulate and verify stationary yaw-only hover tracking on orbiting target")
    parser.add_argument("--radius", type=float, default=4.0, help="Orbit circle radius in meters (default: 4.0m)")
    parser.add_argument("--kp-yaw", type=float, default=80.0, help="Proportional yaw gain (default: 80.0)")
    parser.add_argument("--kd-yaw", type=float, default=15.0, help="Derivative yaw gain (default: 15.0)")
    parser.add_argument("--max-yawspeed", type=float, default=15.0, help="Max yaw rate in deg/s (default: 15.0)")
    parser.add_argument("--max-accel-yaw", type=float, default=30.0, help="Max yaw accel in deg/s^2 (default: 30.0)")
    parser.add_argument("--duration", type=float, default=25.0, help="Simulation duration in seconds (default: 25.0)")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("   YAW-ONLY STATIONARY HOVER ORBIT SIMULATION & VERIFICATION")
    print("="*65)
    print(f"• Drone Hover Position: [0.0, 0.0, -2.5m] (Stationary station-keeping)")
    print(f"• Target Orbit Radius:  {args.radius:.1f} m")
    print(f"• Yaw Controller:       kp={args.kp_yaw:.1f}, kd={args.kd_yaw:.1f}")
    print(f"• Kinematic Limits:     Max Speed: ±{args.max_yawspeed:.1f}°/s | Accel: {args.max_accel_yaw:.1f}°/s^2")
    print("="*65)

    # 1. Run 3 scenario simulations
    scenarios = {
        "Slow Orbit (11.5°/s, Gentle: kp=80, kd=15)": {
            "oracle": CircularOrbitTargetOracle(radius=args.radius, speed=0.8, altitude=-2.5, reversal=False),
            "kp": args.kp_yaw, "kd": args.kd_yaw, "max_yaw": args.max_yawspeed, "max_accel": args.max_accel_yaw
        },
        "Moderate Orbit (18.6°/s, Tuned: kp=237, kd=43)": {
            "oracle": CircularOrbitTargetOracle(radius=args.radius, speed=1.3, altitude=-2.5, reversal=False),
            "kp": 237.3, "kd": 43.2, "max_yaw": args.max_yawspeed, "max_accel": args.max_accel_yaw
        },
        "Reversal Orbit (14.3°/s, Gentle: kp=80, kd=15)": {
            "oracle": CircularOrbitTargetOracle(radius=args.radius, speed=1.0, altitude=-2.5, reversal=True, reversal_period=10.0),
            "kp": args.kp_yaw, "kd": args.kd_yaw, "max_yaw": args.max_yawspeed, "max_accel": args.max_accel_yaw
        }
    }

    results = {}
    print("\n[SIM] Executing closed-loop Pixhawk dynamic simulations...")
    for name, cfg in scenarios.items():
        res = simulate_orbit_tracking(
            oracle=cfg["oracle"],
            kp_yaw=cfg["kp"],
            kd_yaw=cfg["kd"],
            max_yawspeed_deg=cfg["max_yaw"],
            max_accel_yaw=cfg["max_accel"],
            duration=args.duration
        )
        results[name] = res
        m = res["metrics"]
        print(f"  • {name}:")
        print(f"      FOV Retention: {m['fov_retention_pct']:.1f}% | Mean Lag: {m['mean_abs_lag_deg']:.2f}° | Max Lag: {m['max_abs_lag_deg']:.2f}° | Peak Yaw: {m['peak_yaw_rate_deg_s']:.2f}°/s")

    # 2. Generate publication comparison plot
    out_dir = PROJECT_ROOT / "outputs" / "benchmarks" / "control"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "yaw_only_hover_circle_evaluation.png"
    generate_diagnostic_plots(results, out_png)

    # Also copy to artifact dir for direct user presentation
    art_png = ARTIFACT_DIR / "yaw_only_hover_circle_evaluation.png"
    shutil.copy(out_png, art_png)

    # 3. Render high-quality MP4 video of the primary safe deployment profile
    out_mp4 = out_dir / "yaw_only_hover_circle.mp4"
    render_orbit_video(
        results["Slow Orbit (11.5°/s, Gentle: kp=80, kd=15)"],
        out_mp4,
        fps=25,
        title="YAW-ONLY HOVER ORBIT TRACKING (4m RADIUS @ 15°/s CAP)"
    )
    art_mp4 = ARTIFACT_DIR / "yaw_only_hover_circle.mp4"
    shutil.copy(out_mp4, art_mp4)

    # 4. Render video for Reversal test as well
    out_rev_mp4 = out_dir / "yaw_only_reversal_circle.mp4"
    render_orbit_video(
        results["Reversal Orbit (14.3°/s, Gentle: kp=80, kd=15)"],
        out_rev_mp4,
        fps=25,
        title="YAW-ONLY HOVER ORBIT DIRECTION REVERSAL TEST"
    )
    art_rev_mp4 = ARTIFACT_DIR / "yaw_only_reversal_circle.mp4"
    shutil.copy(out_rev_mp4, art_rev_mp4)

    print("\n[COMPLETE] All orbit simulations and video renderings finished successfully.")


if __name__ == "__main__":
    main()
