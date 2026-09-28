#!/usr/bin/env python3
"""
Simple 2D Top-Down Stochastic Drone Chase Movie Generator.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Renders a minimalist, clean 2D top-down visualization of a quadrotor tracking
a target drone executing a stochastic evasive flight path.
Includes a clean side panel displaying real-time flight telemetry.
"""

import sys
import subprocess
from pathlib import Path
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from control.simulation import FastPixhawkQuadSim
from control.pid_controller import KinematicVisualServoController
from control.trajectory import StochasticTargetTrajectory


def render_simple_chase_video(
    output_path: Path,
    duration: float = 20.0,
    fps: int = 25,
    profile: str = "evasive",
    seed: int = 42
):
    dt_sim = 0.02  # 50 Hz simulation loop
    total_sim_steps = int(round(duration / dt_sim))
    sim_steps_per_frame = max(1, int(round((1.0 / fps) / dt_sim)))

    print(f"[SIM] Initializing simulation: {duration}s at {1.0/dt_sim:.0f} Hz ({total_sim_steps} steps)...")
    sim = FastPixhawkQuadSim(dt=dt_sim)
    controller = KinematicVisualServoController(
        camera_uptilt_deg=15.0,
        hfov_deg=60.0,
        vfov_deg=45.0,
        desired_bbox_size=35.0,
    )

    # Initial positions
    # Target starts 12m ahead at 2m altitude (NED: z = -2.0)
    target_oracle = StochasticTargetTrajectory(
        duration=duration + 5.0,
        profile=profile,
        initial_pos=np.array([12.0, 0.0, -2.0], dtype=np.float64),
        initial_heading_rad=0.0,
        seed=seed
    )
    sim.reset(initial_pos=[0.0, 0.0, -2.0], initial_yaw=0.0)

    # Pre-simulate entire flight trajectory for maximum speed and auto-scaling
    print("[SIM] Simulating flight dynamics...")
    times = []
    target_pos_all = []
    chaser_pos_all = []
    chaser_yaw_all = []
    chaser_vel_all = []
    chaser_pitch_all = []
    chaser_roll_all = []
    telemetry_all = []
    cmd_all = []

    for step in range(total_sim_steps):
        t = step * dt_sim
        target_world = target_oracle.get_position(t)

        telemetry = sim.get_camera_telemetry(
            target_world,
            hfov_deg=60.0,
            vfov_deg=45.0,
            target_w_m=0.35,
            target_h_m=0.20,
            img_w=640,
            img_h=480,
            desired_target_size=35.0
        )

        cmd = controller.compute_cmd(telemetry, drone_pitch=sim.pitch, dt=dt_sim)
        sim.step(cmd)

        times.append(t)
        target_pos_all.append(target_world.copy())
        chaser_pos_all.append(sim.pos.copy())
        chaser_yaw_all.append(sim.yaw)
        chaser_vel_all.append(sim.vel.copy())
        chaser_pitch_all.append(sim.pitch)
        chaser_roll_all.append(sim.roll)
        telemetry_all.append(telemetry.copy())
        cmd_all.append(cmd.copy())

    times = np.array(times)
    target_pos_all = np.array(target_pos_all)
    chaser_pos_all = np.array(chaser_pos_all)
    chaser_yaw_all = np.array(chaser_yaw_all)
    chaser_vel_all = np.array(chaser_vel_all)
    chaser_pitch_all = np.array(chaser_pitch_all)
    chaser_roll_all = np.array(chaser_roll_all)
    cmd_all = np.array(cmd_all)

    # Frame sampling stride
    frame_indices = np.arange(0, total_sim_steps, sim_steps_per_frame)
    num_frames = len(frame_indices)
    print(f"[RENDER] Preparing video: {num_frames} frames ({duration:.1f}s at {fps} FPS)...")

    # Dimensions for video (1280x720 16:9)
    fig_w, fig_h = 12.8, 7.2
    dpi = 100
    width_px = int(fig_w * dpi)
    height_px = int(fig_h * dpi)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Launch ffmpeg sub-process to pipe raw RGBA frames
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width_px}x{height_px}",
        "-pix_fmt", "rgba",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "18",
        "-movflags", "+faststart",
        str(output_path)
    ]
    pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Setup clean, simple Matplotlib figure (white background, no fancy theme)
    plt.style.use("default")
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#ffffff")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.0], left=0.08, right=0.96, top=0.92, bottom=0.08, wspace=0.18)

    ax_map = fig.add_subplot(gs[0, 0])
    ax_hud = fig.add_subplot(gs[0, 1])

    # Trajectory bounds for equal map view with padding
    all_east = np.concatenate([target_pos_all[:, 1], chaser_pos_all[:, 1]])
    all_north = np.concatenate([target_pos_all[:, 0], chaser_pos_all[:, 0]])
    e_min, e_max = np.min(all_east) - 5.0, np.max(all_east) + 5.0
    n_min, n_max = np.min(all_north) - 5.0, np.max(all_north) + 5.0
    
    # Keep square aspect ratio
    e_center = (e_min + e_max) / 2.0
    n_center = (n_min + n_max) / 2.0
    half_span = max((e_max - e_min), (n_max - n_min)) / 2.0 + 3.0

    half_hfov_rad = np.deg2rad(60.0 / 2.0)
    fov_cone_length = 6.0

    for f_idx, s_idx in enumerate(frame_indices):
        t_now = times[s_idx]
        chaser_pos = chaser_pos_all[s_idx]
        target_pos = target_pos_all[s_idx]
        yaw = chaser_yaw_all[s_idx]
        telem = telemetry_all[s_idx]
        cmd = cmd_all[s_idx]
        vel = chaser_vel_all[s_idx]

        # ------------------------------------------------------------------
        # 1. Clear axes for clean frame render
        # ------------------------------------------------------------------
        ax_map.clear()
        ax_hud.clear()

        # ------------------------------------------------------------------
        # 2. Main 2D Top-Down Map
        # ------------------------------------------------------------------
        ax_map.set_facecolor("#fafafa")
        ax_map.grid(True, linestyle="--", alpha=0.5, color="#c0c0c0")

        # Full intended trajectory paths (light dotted background)
        ax_map.plot(target_pos_all[:, 1], target_pos_all[:, 0], color="#d62728", linestyle=":", linewidth=1.0, alpha=0.35)
        ax_map.plot(chaser_pos_all[:, 1], chaser_pos_all[:, 0], color="#1f77b4", linestyle=":", linewidth=1.0, alpha=0.35)

        # Elapsed trail up to current step
        ax_map.plot(target_pos_all[:s_idx+1, 1], target_pos_all[:s_idx+1, 0], color="#d62728", linestyle="-", linewidth=1.8, label="Target Trajectory")
        ax_map.plot(chaser_pos_all[:s_idx+1, 1], chaser_pos_all[:s_idx+1, 0], color="#1f77b4", linestyle="-", linewidth=2.0, label="Chaser Trajectory")

        # Start markers
        ax_map.scatter(target_pos_all[0, 1], target_pos_all[0, 0], marker="o", color="#d62728", s=35, alpha=0.7)
        ax_map.scatter(chaser_pos_all[0, 1], chaser_pos_all[0, 0], marker="o", color="#1f77b4", s=35, alpha=0.7)

        # Visual Sightline (connecting line)
        line_color = "#2ca02c" if telem["in_view"] else "#7f7f7f"
        ax_map.plot([chaser_pos[1], target_pos[1]], [chaser_pos[0], target_pos[0]],
                    color=line_color, linestyle="--", linewidth=1.2, alpha=0.8, label="Sightline (Locked)" if telem["in_view"] else "Sightline (Lost)")

        # Camera 60-degree FOV Cone
        # p_origin = np.array([chaser_pos[1], chaser_pos[0]])
        # left_yaw = yaw - half_hfov_rad
        # right_yaw = yaw + half_hfov_rad
        # p_left = p_origin + fov_cone_length * np.array([np.sin(left_yaw), np.cos(left_yaw)])
        # p_right = p_origin + fov_cone_length * np.array([np.sin(right_yaw), np.cos(right_yaw)])

        # fov_wedge = Polygon([p_origin, p_left, p_right], closed=True,
        #                     facecolor="#1f77b4", edgecolor="#1f77b4", alpha=0.12)
        # ax_map.add_patch(fov_wedge)
        # ax_map.plot([p_origin[0], p_left[0]], [p_origin[1], p_left[1]], color="#1f77b4", linewidth=0.8, alpha=0.4)
        # ax_map.plot([p_origin[0], p_right[0]], [p_origin[1], p_right[1]], color="#1f77b4", linewidth=0.8, alpha=0.4)

        # Chaser Drone Marker & Heading Pointer
        ax_map.scatter(chaser_pos[1], chaser_pos[0], color="#1f77b4", s=70, zorder=5, label="Chaser Drone")
        arrow_len = 2.0
        ax_map.arrow(chaser_pos[1], chaser_pos[0], arrow_len * np.sin(yaw), arrow_len * np.cos(yaw),
                     head_width=0.7, head_length=0.7, fc="#1f77b4", ec="#1f77b4", zorder=6)

        # Target Drone Marker
        ax_map.scatter(target_pos[1], target_pos[0], color="#d62728", s=80, marker="o", edgecolors="black", linewidths=1.0, zorder=5, label="Target Drone")

        ax_map.set_xlim(e_center - half_span, e_center + half_span)
        ax_map.set_ylim(n_center - half_span, n_center + half_span)
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.set_xlabel("East Position Y (m)", fontsize=10)
        ax_map.set_ylabel("North Position X (m)", fontsize=10)
        ax_map.set_title(f"Top-Down Pursuit Arena (t = {t_now:.1f}s)", fontsize=11, fontweight="bold", pad=8)
        ax_map.legend(loc="upper left", fontsize=8.0, framealpha=0.9)

        # ------------------------------------------------------------------
        # 3. Telemetry Panel
        # ------------------------------------------------------------------
        ax_hud.axis("off")
        ax_hud.set_facecolor("#ffffff")

        # Telemetry Text
        status_str = telem.get("status", "LOCKED")
        current_dist = telem.get("distance", np.linalg.norm(target_pos - chaser_pos))
        target_size = telem.get("target_size", 0.0)
        err_x = telem.get("error_x", 0.0)
        err_y = telem.get("error_y", 0.0)

        chaser_speed = np.linalg.norm(vel)
        altitude_agl = -chaser_pos[2]  # NED: negative Z is altitude

        hud_text = (
            f"SIMULATION TELEMETRY\n"
            f"─────────────────────────────\n"
            f"Time:             {t_now:5.2f} s\n"
            f"Target Profile:   {profile.title()}\n"
            f"Tracking Status:  {status_str}\n"
            f"In Camera FOV:    {'YES' if telem['in_view'] else 'NO (COASTING)'}\n"
            f"\n"
            f"VISUAL SERVOING ERRORS\n"
            f"─────────────────────────────\n"
            f"Distance to Target: {current_dist:5.2f} m  (Goal: 6.0 m)\n"
            f"Bounding Box Size:  {target_size:5.1f} px (Goal: 35 px)\n"
            f"Azimuth Error (ex): {err_x:+5.2f}  [-1, +1]\n"
            f"Elevation Err (ey): {err_y:+5.2f}  [-1, +1]\n"
            f"\n"
            f"FLIGHT COMMANDS (MAVLink)\n"
            f"─────────────────────────────\n"
            f"cmd_vx (Forward):   {cmd[0]:+5.2f} m/s\n"
            f"cmd_vy (Lateral):   {cmd[1]:+5.2f} m/s\n"
            f"cmd_vz (Climb/Desc):{cmd[2]:+5.2f} m/s\n"
            f"cmd_yaw (Turn Rate):{cmd[3]:+5.1f} deg/s\n"
            f"\n"
            f"DRONE STATE (Pixhawk)\n"
            f"─────────────────────────────\n"
            f"Ground Speed:       {chaser_speed:5.2f} m/s\n"
            f"Altitude (AGL):     {altitude_agl:5.2f} m\n"
            f"Heading (Yaw):      {np.rad2deg(yaw):+5.1f}°\n"
            f"Body Pitch:         {np.rad2deg(chaser_pitch_all[s_idx]):+5.1f}°\n"
            f"Body Roll:          {np.rad2deg(chaser_roll_all[s_idx]):+5.1f}°\n"
            f"Pipeline Delay:     60 ms (3 steps)"
        )

        ax_hud.text(
            0.05, 0.95, hud_text,
            transform=ax_hud.transAxes,
            fontsize=9.2,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="square,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced4da", linewidth=1.0)
        )

        # Pipe frame directly to ffmpeg
        fig.canvas.draw()
        rgba_buffer = fig.canvas.buffer_rgba()
        pipe.stdin.write(rgba_buffer)

        if f_idx % 50 == 0 or f_idx == num_frames - 1:
            print(f"[RENDER] Frame {f_idx+1}/{num_frames} ({((f_idx+1)/num_frames)*100:.1f}%) rendered...")

    # Close pipe and finalize video
    pipe.stdin.close()
    pipe.wait()
    plt.close(fig)

    print(f"\n[SUCCESS] Rendered video successfully saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    out_dir = PROJECT_ROOT / "outputs" / "videos"
    out_file = out_dir / "simple_chase_hyper_evasive_stochastic_2d.mp4"
    render_simple_chase_video(out_file, duration=25.0, fps=20, profile="hyper_evasive", seed=42)
