#!/usr/bin/env python3
"""
2D Top-Down Autonomous Drone Chase Movie Generator.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Simulates closed-loop pursuit using FastPixhawkQuadSim (with dynamic roll-tilt,
60ms latency, and 15-degree camera up-tilt) and KinematicVisualServoController.

Generates a broadcast-quality, synchronized 2D flight arena visualization:
- Left: Smooth-tracking 2D top-down bird's-eye arena (North vs East) with
  fading trajectory trails, quadcopter orientation glyphs, dynamic FOV cone,
  and visual line-of-sight lock link.
- Right Top: On-board camera sensor FPV viewfinder with artificial horizon
  (reflecting dynamic pitch & roll) and YOLO target detection bounding box.
- Right Center: Live Pixhawk MAVSDK flight telemetry HUD and attitude metrics.
- Right Bottom: Rolling real-time strip charts of tracking errors and standoff range.

Directly pipes raw RGBA frames into ffmpeg for pristine, fast H.264/MP4 encoding,
and optionally exports animated GIFs for immediate documentation embedding.
"""

import sys
import os
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Callable
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle, Wedge
import matplotlib.patheffects as pe

from control.simulation import FastPixhawkQuadSim
from control.pid_controller import KinematicVisualServoController
from control.trajectory import StochasticTargetTrajectory, PROFILES

ARTIFACT_DIRS = [
    Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/42b621c5-bd5c-42b8-ae83-87208090e0a7"),
    Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/c2b8fddd-4624-46f7-9768-e1974371d76e"),
]


# ==============================================================================
# 1. Trajectory Resolution & Dispatch
# ==============================================================================

def get_target_trajectory(
    name_or_profile: str,
    duration: float = 30.0,
    seed: int = 42
) -> Tuple[Callable[[float], np.ndarray], str]:
    """
    Returns a trajectory evaluation callable `f(t) -> [x, y, z]` (world NED)
    and a display title for the specified trajectory name or stochastic profile.
    """
    clean_name = name_or_profile.lower().strip()

    # Extract embedded seed if present (e.g. "stochastic_evasive_seed42")
    if "seed" in clean_name:
        parts = clean_name.split("seed")
        clean_name = parts[0].rstrip("_")
        try:
            seed = int(parts[1])
        except (ValueError, IndexError):
            pass

    # Strip "stochastic_" prefix if present
    profile_candidate = clean_name.replace("stochastic_", "")

    # Check if this matches a known stochastic profile
    if profile_candidate in PROFILES:
        traj_oracle = StochasticTargetTrajectory(
            duration=duration,
            profile=profile_candidate,
            initial_pos=np.array([12.0, 0.0, -2.0]),
            seed=seed
        )
        title = f"Stochastic {profile_candidate.replace('_', ' ').title()} (Seed {seed})"
        return traj_oracle.get_position, title

    # Deterministic Benchmark Flight Trajectories
    if clean_name in ["figure8", "figure_8"]:
        w = 2.0 * np.pi / 18.0  # 18-second full loop
        def figure8_pos(t: float) -> np.ndarray:
            x = 18.0 + 12.0 * np.sin(w * t)
            y = 12.0 * np.sin(2.0 * w * t)
            z = -3.0 - 1.2 * np.sin(w * t)
            return np.array([x, y, z], dtype=np.float64)
        return figure8_pos, "3D Aerobatic Figure-8 Loop"

    elif clean_name == "slalom":
        w = 0.55
        def slalom_pos(t: float) -> np.ndarray:
            x = 12.0 + 4.2 * t
            y = 8.5 * np.sin(w * t)
            z = -3.0 + 0.8 * np.sin(0.25 * t)
            return np.array([x, y, z], dtype=np.float64)
        return slalom_pos, "High-Speed Evasive Slalom"

    elif clean_name in ["break_turn", "breakturn"]:
        def break_turn_pos(t: float) -> np.ndarray:
            if t < 6.0:
                # Straight high-speed sprint
                return np.array([12.0 + 5.5 * t, 0.0, -2.5], dtype=np.float64)
            else:
                # 90-degree tactical break right and rapid altitude plunge
                dt_turn = t - 6.0
                turn_radius = 10.0
                theta = min(np.pi / 2.0, 0.75 * dt_turn)
                x = 12.0 + 5.5 * 6.0 + turn_radius * np.sin(theta)
                y = turn_radius * (1.0 - np.cos(theta))
                if theta >= np.pi / 2.0:
                    dt_post = dt_turn - (np.pi / 2.0) / 0.75
                    y += 6.5 * dt_post
                z = -2.5 - 1.2 * dt_turn  # Dive into earth (climb in altitude)
                return np.array([x, y, z], dtype=np.float64)
        return break_turn_pos, "Tactical Break-Turn & Dive"

    elif clean_name in ["corkscrew", "spiral"]:
        w = 0.45
        def corkscrew_pos(t: float) -> np.ndarray:
            r = 8.0 + 0.22 * t
            x = 16.0 + r * np.cos(w * t)
            y = r * np.sin(w * t)
            z = -2.0 - 0.22 * t  # Ascending climb
            return np.array([x, y, z], dtype=np.float64)
        return corkscrew_pos, "3D Expanding Corkscrew Ascent"

    elif clean_name in ["circle", "orbit"]:
        w = 0.28
        def circle_pos(t: float) -> np.ndarray:
            r = 14.0
            x = 18.0 + r * np.cos(w * t)
            y = r * np.sin(w * t)
            z = -3.0
            return np.array([x, y, z], dtype=np.float64)
        return circle_pos, "Circular Orbital Track"

    elif clean_name in ["high_speed_cruise", "fast_cruise", "fast", "speed_test"]:
        def high_speed_cruise_pos(t: float) -> np.ndarray:
            # High speed straight cruising at 17.5 m/s (~63 km/h) with subtle aerodynamic drift
            x = 12.0 + 17.5 * t
            y = 2.0 * np.sin(0.35 * t)
            z = -2.5 + 0.4 * np.sin(0.20 * t)
            return np.array([x, y, z], dtype=np.float64)
        return high_speed_cruise_pos, "High-Speed Cruise (17.5 m/s / 63 km/h)"

    # Default fallback: stochastic evasive
    traj_oracle = StochasticTargetTrajectory(
        duration=duration,
        profile="evasive",
        initial_pos=np.array([12.0, 0.0, -2.0]),
        seed=seed
    )
    return traj_oracle.get_position, f"Stochastic Evasive (Seed {seed})"


# ==============================================================================
# 2. Closed-Loop Pursuit Simulation Runner
# ==============================================================================

def simulate_pursuit(
    trajectory_fn: Callable[[float], np.ndarray],
    duration: float = 25.0,
    dt: float = 0.02,
    enable_lateral_strafe: bool = True,
    desired_standoff: float = 6.0,
    desired_bbox_px: float = 55.0,
) -> dict:
    """
    Executes the full closed-loop 50 Hz simulation with FastPixhawkQuadSim and
    KinematicVisualServoController, logging state histories for movie rendering.
    """
    num_steps = int(duration / dt)

    sim = FastPixhawkQuadSim(
        dt=dt,
        max_vel_xy=18.0,
        max_accel_xy=6.5,
        max_vel_up=4.0,
        max_vel_down=2.5
    )
    sim.reset(initial_pos=[0.0, 0.0, -2.5], initial_yaw=0.0)

    controller = KinematicVisualServoController(
        camera_uptilt_deg=15.0,
        hfov_deg=60.0,
        vfov_deg=45.0,
        desired_bbox_size=desired_bbox_px,
        desired_standoff_dist=desired_standoff,
        use_bbox_size=True,
        enable_lateral_strafe=enable_lateral_strafe,
        kp_lat=2.5,
        kd_lat=0.35,
        max_lat_vel=4.0,
        max_lat_accel=5.0,
        max_limits=np.array([18.0, 4.0, 2.5, 120.0]),
        max_accel=6.5,
        max_decel=2.8,
    )

    times = []
    chaser_pos = []
    chaser_vel = []
    chaser_yaw = []
    chaser_pitch = []
    chaser_roll = []
    chaser_cam_pitch = []
    target_pos = []
    dist_history = []
    err_x_history = []
    err_y_history = []
    in_view_history = []
    cmd_history = []
    bbox_history = []

    for step in range(num_steps):
        t = step * dt
        p_target = trajectory_fn(t)

        # 1. Perspective projection & vision telemetry
        telem = sim.get_camera_telemetry(
            p_target,
            hfov_deg=60.0,
            vfov_deg=45.0,
            target_w_m=0.35,
            target_h_m=0.20,
            img_w=640,
            img_h=480,
            desired_target_size=desired_bbox_px
        )

        # 2. Compute 4-DOF velocity setpoint with 3D SO(3) attitude decoupling
        cmd = controller.compute_cmd(telem, drone_pitch=sim.pitch, drone_roll=sim.roll, dt=dt)

        # 3. Advance drone flight physics
        obs = sim.step(cmd)

        # Record histories
        times.append(t)
        chaser_pos.append(sim.pos.copy())
        chaser_vel.append(sim.vel.copy())
        chaser_yaw.append(sim.yaw)
        chaser_pitch.append(sim.pitch)
        chaser_roll.append(sim.roll)
        chaser_cam_pitch.append(sim.cam_pitch)
        target_pos.append(p_target.copy())
        dist_history.append(telem["distance"])
        err_x_history.append(telem["error_x"])
        err_y_history.append(telem["error_y"])
        in_view_history.append(telem["in_view"])
        cmd_history.append(cmd.copy())
        bbox_history.append(telem.get("bbox", [0, 0, 0, 0]))

    return {
        "times": np.array(times),
        "chaser_pos": np.array(chaser_pos),
        "chaser_vel": np.array(chaser_vel),
        "chaser_yaw": np.array(chaser_yaw),
        "chaser_pitch": np.array(chaser_pitch),
        "chaser_roll": np.array(chaser_roll),
        "chaser_cam_pitch": np.array(chaser_cam_pitch),
        "target_pos": np.array(target_pos),
        "distance": np.array(dist_history),
        "err_x": np.array(err_x_history),
        "err_y": np.array(err_y_history),
        "in_view": np.array(in_view_history),
        "cmd": np.array(cmd_history),
        "bbox": np.array(bbox_history),
        "duration": duration,
        "dt": dt,
    }


# ==============================================================================
# 3. Matplotlib 2D Frame Rendering Engine
# ==============================================================================

def render_movie_frames(
    sim_data: dict,
    output_mp4_path: Path,
    output_gif_path: Optional[Path] = None,
    trajectory_title: str = "Stochastic Pursuit",
    fps: int = 25,
    enable_lateral_strafe: bool = True,
):
    """
    Renders synchronized video frames and pipes directly into ffmpeg.
    """
    times = sim_data["times"]
    c_pos = sim_data["chaser_pos"]
    c_vel = sim_data["chaser_vel"]
    c_yaw = sim_data["chaser_yaw"]
    c_pitch = sim_data["chaser_pitch"]
    c_roll = sim_data["chaser_roll"]
    c_cam_pitch = sim_data["chaser_cam_pitch"]
    t_pos = sim_data["target_pos"]
    dists = sim_data["distance"]
    err_xs = sim_data["err_x"]
    err_ys = sim_data["err_y"]
    in_views = sim_data["in_view"]
    cmds = sim_data["cmd"]
    bboxes = sim_data["bbox"]

    total_duration = sim_data["duration"]
    dt_sim = sim_data["dt"]
    total_sim_steps = len(times)

    # Frame sampling stride
    sim_steps_per_frame = max(1, int(round((1.0 / fps) / dt_sim)))
    frame_indices = np.arange(0, total_sim_steps, sim_steps_per_frame)
    num_frames = len(frame_indices)

    print(f"[RENDER] Preparing 2D Pursuit Movie: {num_frames} frames ({total_duration:.1f}s at {fps} FPS)...")

    # Figure dimensions (16:9 1600x900)
    fig_w, fig_h = 16.0, 9.0
    dpi = 100
    width_px = int(fig_w * dpi)
    height_px = int(fig_h * dpi)

    # Ensure output parent directory exists
    output_mp4_path = Path(output_mp4_path)
    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    # Launch ffmpeg pipe
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
        "-movflags", "+faststart",
        str(output_mp4_path)
    ]

    pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Setup dark theme Matplotlib figure
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#0b0e14")
    gs = fig.add_gridspec(3, 3, width_ratios=[1.75, 1.0, 0.95], height_ratios=[1.15, 1.0, 1.0],
                           left=0.04, right=0.97, top=0.94, bottom=0.06, hspace=0.32, wspace=0.28)

    # 1. Main 2D Top-Down Flight Arena
    ax_arena = fig.add_subplot(gs[:, 0], facecolor="#10141d")

    # 2. On-board Camera Sensor Viewfinder (FPV)
    ax_cam = fig.add_subplot(gs[0, 1:], facecolor="#080a0f")

    # 3. Autopilot Telemetry & Flight State HUD
    ax_hud = fig.add_subplot(gs[1, 1:], facecolor="#10141d")
    ax_hud.axis("off")

    # 4. Rolling Tracking Error Strip Chart
    ax_err = fig.add_subplot(gs[2, 1], facecolor="#10141d")

    # 5. Range & Velocity Strip Chart
    ax_vel = fig.add_subplot(gs[2, 2], facecolor="#10141d")

    # Constant geometry
    half_hfov = np.deg2rad(60.0 / 2.0)
    fov_range = 14.0

    # Global trajectory bounds for camera auto-framing
    all_east = np.concatenate([c_pos[:, 1], t_pos[:, 1]])
    all_north = np.concatenate([c_pos[:, 0], t_pos[:, 0]])

    for frame_i, step_idx in enumerate(frame_indices):
        t_curr = times[step_idx]
        cx, cy, cz = c_pos[step_idx]
        tx, ty, tz = t_pos[step_idx]
        yaw = c_yaw[step_idx]
        pitch = c_pitch[step_idx]
        roll = c_roll[step_idx]
        cam_pitch = c_cam_pitch[step_idx]
        dist = dists[step_idx]
        in_view = in_views[step_idx]
        cmd = cmds[step_idx]
        ex = err_xs[step_idx]
        ey = err_ys[step_idx]
        bbox = bboxes[step_idx]

        # ----------------------------------------------------------------------
        # A. 2D Top-Down Flight Arena (East = X-axis, North = Y-axis)
        # ----------------------------------------------------------------------
        ax_arena.clear()
        ax_arena.set_facecolor("#10141d")
        ax_arena.grid(True, linestyle="--", alpha=0.22, color="#2c3444")

        # Trailing history window (last 6 seconds)
        trail_start = max(0, step_idx - int(6.0 / dt_sim))
        c_trail_east = c_pos[trail_start:step_idx + 1, 1]
        c_trail_north = c_pos[trail_start:step_idx + 1, 0]
        t_trail_east = t_pos[trail_start:step_idx + 1, 1]
        t_trail_north = t_pos[trail_start:step_idx + 1, 0]

        # Full target ghost trajectory
        ax_arena.plot(t_pos[:, 1], t_pos[:, 0], color="#ff4757", linestyle=":", linewidth=1.2, alpha=0.25, zorder=1)

        # Fading trails
        ax_arena.plot(t_trail_east, t_trail_north, color="#ff4757", linewidth=2.2, alpha=0.85, label="Target UAV", zorder=3)
        ax_arena.plot(c_trail_east, c_trail_north, color="#00d2d3", linewidth=2.8, alpha=0.90, label="Chaser (4-DOF PID)", zorder=4)

        # Camera FOV Wedge (Sensor Projection)
        left_yaw = yaw - half_hfov
        right_yaw = yaw + half_hfov
        p_chaser = np.array([cy, cx])
        p_left = p_chaser + fov_range * np.array([np.sin(left_yaw), np.cos(left_yaw)])
        p_right = p_chaser + fov_range * np.array([np.sin(right_yaw), np.cos(right_yaw)])

        fov_color = "#00d2d3" if in_view else "#ffa502"
        fov_alpha = 0.14 if in_view else 0.08
        wedge_poly = Polygon([p_chaser, p_left, p_right], closed=True, facecolor=fov_color, edgecolor=fov_color, alpha=fov_alpha, zorder=2)
        ax_arena.add_patch(wedge_poly)
        ax_arena.plot([cy, p_left[0]], [cx, p_left[1]], color=fov_color, linewidth=0.8, alpha=0.5, zorder=2)
        ax_arena.plot([cy, p_right[0]], [cx, p_right[1]], color=fov_color, linewidth=0.8, alpha=0.5, zorder=2)

        # Visual Line of Sight (LOS) Laser Link
        los_color = "#2ed573" if in_view else "#ff4757"
        los_style = "-" if in_view else ":"
        ax_arena.plot([cy, ty], [cx, tx], color=los_color, linestyle=los_style, linewidth=1.4, alpha=0.75, zorder=3)

        # Target Drone Marker (Pulsing Beacon)
        ax_arena.scatter([ty], [tx], s=110, marker="o", facecolor="#ff4757", edgecolor="white", linewidth=1.5, zorder=6)
        ax_arena.annotate(f"TARGET\nZ={tz:+.1f}m", (ty + 0.6, tx + 0.6), color="#ff6b81", fontsize=8, fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="#1e131d", edgecolor="#ff4757", alpha=0.85), zorder=7)

        # Chaser Quadcopter Glyphs (4-Rotor Disc Cross with true yaw)
        arm_len = 1.0
        rotor_r = 0.45
        c_cos, c_sin = np.cos(yaw), np.sin(yaw)
        r_cos, r_sin = -np.sin(yaw), np.cos(yaw)  # Body Right vector

        # 4 Rotor Hub positions in East-North plane: [East, North]
        front_right = p_chaser + (arm_len * 0.707) * (np.array([c_sin, c_cos]) + np.array([r_sin, r_cos]))
        front_left  = p_chaser + (arm_len * 0.707) * (np.array([c_sin, c_cos]) - np.array([r_sin, r_cos]))
        rear_right  = p_chaser + (arm_len * 0.707) * (-np.array([c_sin, c_cos]) + np.array([r_sin, r_cos]))
        rear_left   = p_chaser + (arm_len * 0.707) * (-np.array([c_sin, c_cos]) - np.array([r_sin, r_cos]))

        # Draw frame arms
        ax_arena.plot([front_left[0], rear_right[0]], [front_left[1], rear_right[1]], color="#dfe4ea", lw=2.2, zorder=5)
        ax_arena.plot([front_right[0], rear_left[0]], [front_right[1], rear_left[1]], color="#dfe4ea", lw=2.2, zorder=5)

        # Draw rotor discs
        for hub, is_front in [(front_right, True), (front_left, True), (rear_right, False), (rear_left, False)]:
            disc_col = "#00d2d3" if is_front else "#747d8c"
            ax_arena.add_patch(Circle(hub, rotor_r, facecolor=disc_col, edgecolor="white", lw=0.8, alpha=0.8, zorder=6))

        # Forward nose arrow
        ax_arena.arrow(cy, cx, 1.8 * c_sin, 1.8 * c_cos, head_width=0.6, head_length=0.6, fc="#00d2d3", ec="white", zorder=7)

        # Annotate chaser altitude & speed
        spd_mag = np.linalg.norm(c_vel[step_idx, :2])
        ax_arena.annotate(f"CHASER ({spd_mag:.1f}m/s)\nZ={cz:+.1f}m", (cy + 0.8, cx - 1.2), color="#00d2d3", fontsize=8, fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="#091f24", edgecolor="#00d2d3", alpha=0.85), zorder=7)

        # Smooth Auto-Framing Camera (Centered between Chaser and Target with padding)
        center_east = 0.5 * (cy + ty)
        center_north = 0.5 * (cx + tx)
        span_east = max(24.0, abs(cy - ty) * 1.6)
        span_north = max(24.0, abs(cx - tx) * 1.6)
        max_span = max(span_east, span_north)

        ax_arena.set_xlim(center_east - max_span / 2.0, center_east + max_span / 2.0)
        ax_arena.set_ylim(center_north - max_span / 2.0, center_north + max_span / 2.0)
        ax_arena.set_aspect("equal", adjustable="box")
        ax_arena.set_xlabel("East Position (m)", color="#a4b0be", fontsize=10, labelpad=4)
        ax_arena.set_ylabel("North Position (m)", color="#a4b0be", fontsize=10, labelpad=4)
        ax_arena.set_title(f"TOP-DOWN ARENA: {trajectory_title.upper()}", color="#f1f2f6", fontsize=11, fontweight="bold", pad=8)
        ax_arena.legend(loc="upper left", facecolor="#10141d", edgecolor="#2c3444", fontsize=8, labelcolor="#dfe4ea")

        # ----------------------------------------------------------------------
        # B. On-Board Camera FPV Viewfinder (Simulated 640x480 Sensor)
        # ----------------------------------------------------------------------
        ax_cam.clear()
        ax_cam.set_facecolor("#04060a")
        ax_cam.set_xlim(0, 640)
        ax_cam.set_ylim(480, 0)  # Image coordinates: top is 0, bottom is 480
        ax_cam.set_xticks([])
        ax_cam.set_yticks([])

        # Artificial Horizon Line (Tilted by roll, shifted by camera pitch)
        # Horizon y on sensor: pitch = 0 means y = 240 + 240 * (tan(-15 deg)/tan(22.5 deg))
        horiz_y_center = 240.0 + (np.tan(cam_pitch) / np.tan(np.deg2rad(22.5))) * 240.0
        tan_roll = np.tan(-roll)
        x_left, x_right = 0.0, 640.0
        y_left = horiz_y_center - (320.0) * tan_roll
        y_right = horiz_y_center + (320.0) * tan_roll
        ax_cam.plot([x_left, x_right], [y_left, y_right], color="#1e90ff", linestyle="--", linewidth=1.2, alpha=0.6)

        # Center Aiming Crosshairs (Optical Center)
        ax_cam.plot([300, 340], [240, 240], color="#00d2d3", lw=1.2, alpha=0.8)
        ax_cam.plot([320, 320], [220, 260], color="#00d2d3", lw=1.2, alpha=0.8)
        # 35px desired standoff circle
        ax_cam.add_patch(Circle((320, 240), 35.0 / 2.0, fill=False, edgecolor="#00d2d3", linestyle=":", lw=1.0, alpha=0.7))

        # Target Bounding Box
        if in_view and bbox[2] > bbox[0]:
            bx1, by1, bx2, by2 = bbox
            bw = bx2 - bx1
            bh = by2 - by1
            rect = Rectangle((bx1, by1), bw, bh, fill=False, edgecolor="#2ed573", linewidth=1.8)
            ax_cam.add_patch(rect)
            # Corner brackets
            b_arm = min(bw, bh) * 0.25
            ax_cam.plot([bx1, bx1 + b_arm], [by1, by1], color="#2ed573", lw=2.5)
            ax_cam.plot([bx1, bx1], [by1, by1 + b_arm], color="#2ed573", lw=2.5)
            ax_cam.plot([bx2, bx2 - b_arm], [by2, by2], color="#2ed573", lw=2.5)
            ax_cam.plot([bx2, bx2], [by2, by2 - b_arm], color="#2ed573", lw=2.5)

            ax_cam.text(bx1, max(14, by1 - 5), f"TRK-01 [LOCK: {dist:.1f}m]", color="#2ed573", fontsize=7.5, fontweight="bold",
                        bbox=dict(boxstyle="square,pad=0.1", facecolor="#081c10", edgecolor="#2ed573", alpha=0.9))
        else:
            ax_cam.text(320, 240, "TARGET OUT OF FOV\n[ COASTING / SEARCH ]", color="#ff4757", fontsize=10, fontweight="bold",
                        ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", facecolor="#240a0e", edgecolor="#ff4757", alpha=0.85))

        # Sensor HUD Labels
        ax_cam.text(12, 24, "FPV SENSOR: 1080p60 GLOBAL SHUTTER (+15° MOUNT)", color="#00d2d3", fontsize=8, family="monospace")
        ax_cam.text(12, 42, f"ERR X: {ex:+.3f} | ERR Y: {ey:+.3f} | RANGE ERR: {(dist - 6.0)/6.0:+.2f}", color="#dfe4ea", fontsize=7.5, family="monospace")
        ax_cam.text(628, 24, "LOCKED" if in_view else "SEARCHING", color="#2ed573" if in_view else "#ff4757", fontsize=8.5, fontweight="bold", ha="right",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#10141d", edgecolor="#2ed573" if in_view else "#ff4757"))
        ax_cam.set_title("ON-BOARD CAMERA SENSOR VIEW (HORIZON STABILIZED)", color="#f1f2f6", fontsize=10, fontweight="bold", pad=6)

        # ----------------------------------------------------------------------
        # C. Autopilot Telemetry & Flight State HUD Card
        # ----------------------------------------------------------------------
        ax_hud.clear()
        ax_hud.axis("off")

        v_fwd = cmd[0]
        v_lat = cmd[1]
        v_down = cmd[2]
        yawspeed = cmd[3]

        p_deg = np.rad2deg(pitch)
        p_label = "nose-down" if p_deg >= 0 else "nose-up"
        hud_text = (
            f"AUTOPILOT FLIGHT TELEMETRY (PX4 MAVSDK)               TIME: {t_curr:04.1f}s / {total_duration:04.1f}s\n"
            f"───────────────────────────────────────────────────────────────────────────────────\n"
            f"• PURSUIT STATUS:      {'LOCKED (100% IN FOV)' if in_view else 'COASTING / RE-ACQUIRING'}        TARGET DIST: {dist:5.2f} m  (GOAL: 6.00m)\n"
            f"• 4-DOF VELOCITIES:    FWD: {v_fwd:4.1f} m/s | LAT: {v_lat:+4.2f} m/s | VERT: {-v_down:+4.2f} m/s | YAW: {yawspeed:+5.1f} °/s\n"
            f"• KINEMATIC ATTITUDE:  PITCH: {p_deg:+4.1f}° ({p_label}) | ROLL: {np.rad2deg(roll):+4.1f}° (strafe) | CAM: {np.rad2deg(cam_pitch):+4.1f}°\n"
            f"• ALTITUDE ENVELOPE:   CHASER: {-cz:4.1f} m | TARGET: {-tz:4.1f} m | DELTA Z: {-(tz - cz):+4.2f} m ({'CLIMB' if v_down < 0 else 'DESCEND'})\n"
            f"• HARDWARE PIPELINE:   60ms Total Latency | 13.0ms YOLOv8n TensorRT FP16 | Global Shutter Active"
        )
        ax_hud.text(0.02, 0.90, hud_text, color="#f1f2f6", fontsize=8.5, family="monospace", va="top",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#10141d", edgecolor="#00d2d3", linewidth=1.1))

        # ----------------------------------------------------------------------
        # D. Rolling Visual Error Strip Chart (last 6 seconds)
        # ----------------------------------------------------------------------
        ax_err.clear()
        ax_err.set_facecolor("#10141d")
        ax_err.grid(True, linestyle="--", alpha=0.2, color="#2c3444")
        t_window = times[trail_start:step_idx + 1]
        ax_err.plot(t_window, err_xs[trail_start:step_idx + 1], color="#00d2d3", lw=1.6, label="Err X (Azimuth)")
        ax_err.plot(t_window, err_ys[trail_start:step_idx + 1], color="#ff4757", lw=1.6, label="Err Y (Elevation)")
        ax_err.axhline(0.0, color="#747d8c", linestyle=":", lw=0.8)
        ax_err.axhline(1.0, color="#ff4757", linestyle="--", lw=0.8, alpha=0.5)
        ax_err.axhline(-1.0, color="#ff4757", linestyle="--", lw=0.8, alpha=0.5)
        ax_err.set_ylim(-1.15, 1.15)
        ax_err.set_xlim(max(0.0, t_curr - 6.0), max(6.0, t_curr))
        ax_err.set_ylabel("Norm Error [-1, 1]", color="#a4b0be", fontsize=8)
        ax_err.set_xlabel("Time (s)", color="#a4b0be", fontsize=8)
        ax_err.set_title("OPTICAL TRACKING ERRORS", color="#f1f2f6", fontsize=9, fontweight="bold")
        ax_err.legend(loc="upper right", facecolor="#10141d", edgecolor="#2c3444", fontsize=7, labelcolor="#dfe4ea")

        # ----------------------------------------------------------------------
        # E. Rolling Distance & Lateral Velocity Chart
        # ----------------------------------------------------------------------
        ax_vel.clear()
        ax_vel.set_facecolor("#10141d")
        ax_vel.grid(True, linestyle="--", alpha=0.2, color="#2c3444")
        ax_vel.plot(t_window, dists[trail_start:step_idx + 1], color="#2ed573", lw=1.8, label="Distance (m)")
        ax_vel.axhline(6.0, color="#2ed573", linestyle=":", lw=1.0, label="Standoff Goal (6m)")
        if enable_lateral_strafe:
            lat_cmds = cmds[trail_start:step_idx + 1, 1]
            ax_vel.plot(t_window, lat_cmds, color="#ffa502", lw=1.4, linestyle="--", label="Strafe Vy (m/s)")
        ax_vel.set_xlim(max(0.0, t_curr - 6.0), max(6.0, t_curr))
        ax_vel.set_ylabel("Meters / m/s", color="#a4b0be", fontsize=8)
        ax_vel.set_xlabel("Time (s)", color="#a4b0be", fontsize=8)
        ax_vel.set_title("DISTANCE & LATERAL CONTROL", color="#f1f2f6", fontsize=9, fontweight="bold")
        ax_vel.legend(loc="upper right", facecolor="#10141d", edgecolor="#2c3444", fontsize=7, labelcolor="#dfe4ea")

        # Draw canvas and write raw RGBA bytes directly into ffmpeg pipe
        fig.canvas.draw()
        rgba_buffer = np.asarray(fig.canvas.buffer_rgba())
        pipe.stdin.write(rgba_buffer.tobytes())

        if (frame_i + 1) % max(1, num_frames // 10) == 0 or frame_i == num_frames - 1:
            pct = 100.0 * (frame_i + 1) / num_frames
            print(f"  [{pct:5.1f}%] Rendered frame {frame_i + 1}/{num_frames} (t = {t_curr:4.1f}s)")

    # Close pipe and finalize video file
    pipe.stdin.close()
    pipe.wait()
    plt.close(fig)
    print(f"[SUCCESS] MP4 Video generated: {output_mp4_path} ({output_mp4_path.stat().st_size / (1024*1024):.2f} MB)")

    # Optional GIF generation using ffmpeg palettegen
    if output_gif_path is not None:
        output_gif_path = Path(output_gif_path)
        print(f"[ENCODE] Generating optimized GIF: {output_gif_path.name}...")
        gif_cmd = [
            "ffmpeg", "-y",
            "-i", str(output_mp4_path),
            "-vf", "fps=15,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            str(output_gif_path)
        ]
        subprocess.run(gif_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"[SUCCESS] GIF generated: {output_gif_path} ({output_gif_path.stat().st_size / (1024*1024):.2f} MB)")

    # Copy to artifact directories for user review
    for adir in ARTIFACT_DIRS:
        if adir.exists():
            shutil.copy(output_mp4_path, adir / output_mp4_path.name)
            if output_gif_path and output_gif_path.exists():
                shutil.copy(output_gif_path, adir / output_gif_path.name)


# ==============================================================================
# 4. Public API & Command-Line Interface
# ==============================================================================

def generate_chase_movie(
    output_mp4_path: Path,
    output_gif_path: Optional[Path] = None,
    trajectory_name: str = "evasive",
    total_time_s: float = 25.0,
    fps: int = 25,
    seed: int = 42,
    enable_lateral_strafe: bool = True,
):
    """
    Primary interface for batch scripts and test suites.
    Simulates the pursuit and renders the 2D top-down movie.
    """
    traj_fn, title = get_target_trajectory(trajectory_name, duration=total_time_s, seed=seed)
    sim_data = simulate_pursuit(
        trajectory_fn=traj_fn,
        duration=total_time_s,
        dt=0.02,
        enable_lateral_strafe=enable_lateral_strafe
    )
    render_movie_frames(
        sim_data=sim_data,
        output_mp4_path=output_mp4_path,
        output_gif_path=output_gif_path,
        trajectory_title=title,
        fps=fps,
        enable_lateral_strafe=enable_lateral_strafe
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2D top-down pursuit video of autonomous drone tracking.")
    parser.add_argument("--profile", "--trajectory", type=str, default="cruising",
                        help="Trajectory profile: cruising, evasive, aerobatic, figure8, slalom, break_turn, corkscrew")
    parser.add_argument("--duration", type=float, default=25.0, help="Duration in seconds (default: 25.0)")
    parser.add_argument("--fps", type=int, default=25, help="Video framerate (default: 25)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stochastic trajectory")
    parser.add_argument("--out", type=str, default=None, help="Output MP4 file path")
    parser.add_argument("--gif", action="store_true", help="Also generate an animated GIF")
    parser.add_argument("--no-lateral-strafe", action="store_true", help="Disable 4-DOF lateral strafe (3-DOF classic mode)")
    args = parser.parse_args()

    # Determine output paths
    out_dir = PROJECT_ROOT / "outputs" / "videos" / "control"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.out:
        mp4_path = Path(args.out)
    else:
        mp4_path = out_dir / f"chase_2d_{args.profile}_seed{args.seed}.mp4"

    gif_path = mp4_path.with_suffix(".gif") if args.gif else None

    print(f"\n{'='*74}")
    print(f"  GENERATING 2D TOP-DOWN PURSUIT VIDEO: [{args.profile.upper()}]")
    print(f"  Duration: {args.duration}s | FPS: {args.fps} | Seed: {args.seed}")
    print(f"  4-DOF Lateral Strafe: {not args.no_lateral_strafe}")
    print(f"  Output MP4: {mp4_path}")
    if gif_path:
        print(f"  Output GIF: {gif_path}")
    print(f"{'='*74}\n")

    generate_chase_movie(
        output_mp4_path=mp4_path,
        output_gif_path=gif_path,
        trajectory_name=args.profile,
        total_time_s=args.duration,
        fps=args.fps,
        seed=args.seed,
        enable_lateral_strafe=(not args.no_lateral_strafe)
    )
