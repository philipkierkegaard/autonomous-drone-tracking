#!/usr/bin/env python3
"""
Simple 2D Top-Down Stochastic Drone Chase Movie Generator.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Renders a minimalist, clean 2D top-down visualization of a quadrotor tracking
a target drone executing a stochastic evasive flight path.
Includes a clean side panel displaying real-time flight telemetry.
"""

import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Union, Tuple
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

try:
    from control.predictive_controller import RecurrentVisualServoController
except Exception:
    RecurrentVisualServoController = None

ARTIFACT_DIRS = [
    Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/53eb0b3e-b75c-4830-a9d4-c69e644ccb6a"),
]


def get_encounter_init(
    encounter: str = "tail_chase",
    custom_pos: Optional[list] = None,
    custom_heading_deg: Optional[float] = None
) -> Tuple[np.ndarray, float, str]:
    """Resolves initial target position and heading for encounter type."""
    encounter = encounter.lower()
    if custom_pos is not None:
        init_pos = np.array(custom_pos, dtype=np.float64)
        init_heading = np.deg2rad(custom_heading_deg if custom_heading_deg is not None else 0.0)
        label = f"Custom ({init_pos[0]:.1f}, {init_pos[1]:.1f}, {init_pos[2]:.1f})"
    elif encounter == "crossing_right":
        # Target starts on the left flank (-Y), flying East (+Y) across the chaser's camera view
        init_pos = np.array([12.0, -6.0, -2.0], dtype=np.float64)
        init_heading = float(np.deg2rad(90.0))
        label = "Crossing from Left (Heading East +90°)"
    elif encounter == "crossing_left":
        # Target starts on the right flank (+Y), flying West (-Y) across the chaser's camera view
        init_pos = np.array([12.0, 6.0, -2.0], dtype=np.float64)
        init_heading = float(np.deg2rad(-90.0))
        label = "Crossing from Right (Heading West -90°)"
    else:
        # Default tail chase: target starts ahead flying along line of sight
        init_pos = np.array([12.0, 0.0, -2.0], dtype=np.float64)
        init_heading = 0.0
        label = "Tail Chase (Forward North 0°)"
    return init_pos, init_heading, label


def render_simple_chase_video(
    output_path: Path,
    duration: float = 20.0,
    fps: int = 25,
    profile: str = "evasive",
    encounter: str = "crossing_right",
    target_init_pos: Optional[list] = None,
    target_init_heading_deg: Optional[float] = None,
    seed: int = 42,
    controller_type: str = "pid",
    model_path: Optional[Union[str, Path]] = None,
    output_gif_path: Optional[Path] = None,
    fixed_bounds: Optional[Tuple[float, float, float]] = None,
):
    dt_sim = 0.02  # 50 Hz simulation loop
    total_sim_steps = int(round(duration / dt_sim))
    sim_steps_per_frame = max(1, int(round((1.0 / fps) / dt_sim)))

    is_rl = controller_type.lower() in ("recurrent_ppo", "rl", "recurrent")
    ctrl_label = "Learned Recurrent PPO (Gen 4)" if is_rl else "Classical Visual Servoing (PID)"
    ctrl_short = "Gen 4 RL" if is_rl else "PID"
    
    init_pos, init_heading, encounter_label = get_encounter_init(encounter, target_init_pos, target_init_heading_deg)

    print(f"\n[SIM] Initializing simulation: {duration}s at {1.0/dt_sim:.0f} Hz ({total_sim_steps} steps)...")
    print(f"      Profile: {profile.upper()} | Encounter: {encounter_label} | Seed: {seed}")
    print(f"      Controller: {ctrl_label}")
    sim = FastPixhawkQuadSim(dt=dt_sim)

    if is_rl:
        if model_path is None:
            default_p = PROJECT_ROOT / "control" / "weights" / "recurrent_ppo_continuous_potential" / "best_model" / "best_model.zip"
            if not default_p.exists():
                default_p = PROJECT_ROOT / "control" / "weights" / "recurrent_ppo_tail_chase_finetune" / "best_model" / "best_model.zip"
            model_path = default_p
        if RecurrentVisualServoController is not None:
            controller = RecurrentVisualServoController(model_path=str(model_path), w_nominal=0.0505)
        else:
            raise ImportError("RecurrentVisualServoController could not be imported.")
    else:
        controller = KinematicVisualServoController(
            camera_uptilt_deg=15.0,
            hfov_deg=60.0,
            vfov_deg=45.0,
            desired_bbox_size=32.33,
            desired_standoff_dist=6.0,
            use_bbox_size=False,
            enable_lateral_strafe=True,
            kp_lat=2.5,
            kd_lat=0.35,
            max_lat_vel=6.0,
            max_lat_accel=5.0,
            max_accel=6.5,
            max_decel=5.0
        )

    # Initial positions
    target_oracle = StochasticTargetTrajectory(
        duration=duration + 5.0,
        profile=profile,
        initial_pos=init_pos,
        initial_heading_rad=init_heading,
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
            desired_target_size=32.33
        )
        if is_rl:
            c, s = np.cos(sim.yaw), np.sin(sim.yaw)
            vx_b = c * sim.vel[0] + s * sim.vel[1]
            vy_b = -s * sim.vel[0] + c * sim.vel[1]
            vz_b = sim.vel[2]
            v_body = np.array([vx_b, vy_b, vz_b], dtype=np.float64)
            cmd = controller.compute_cmd(
                telemetry,
                drone_pitch=sim.pitch,
                drone_roll=sim.roll,
                dt=dt_sim,
                vehicle_vel=v_body,
                vehicle_yaw_rate=sim.yaw_rate,
                current_alt_m=-sim.pos[2]
            )
        else:
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
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0], left=0.08, right=0.96, top=0.92, bottom=0.08, wspace=0.18)

    ax_map = fig.add_subplot(gs[0, 0])
    ax_hud = fig.add_subplot(gs[0, 1])

    # Trajectory bounds for equal map view with padding
    if fixed_bounds is not None:
        e_center, n_center, half_span = fixed_bounds
    else:
        all_east = np.concatenate([target_pos_all[:, 1], chaser_pos_all[:, 1]])
        all_north = np.concatenate([target_pos_all[:, 0], chaser_pos_all[:, 0]])
        e_min, e_max = np.min(all_east) - 5.0, np.max(all_east) + 5.0
        n_min, n_max = np.min(all_north) - 5.0, np.max(all_north) + 5.0
        e_center = (e_min + e_max) / 2.0
        n_center = (n_min + n_max) / 2.0
        half_span = max((e_max - e_min), (n_max - n_min)) / 2.0 + 3.0

    trail_color = "#1f77b4" if not is_rl else "#2ca02c"
    drone_color = "#1f77b4" if not is_rl else "#1b7837"

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
        ax_map.plot(chaser_pos_all[:, 1], chaser_pos_all[:, 0], color=trail_color, linestyle=":", linewidth=1.0, alpha=0.35)

        # Elapsed trail up to current step
        ax_map.plot(target_pos_all[:s_idx+1, 1], target_pos_all[:s_idx+1, 0], color="#d62728", linestyle="-", linewidth=2.0, label="Target Trajectory")
        ax_map.plot(chaser_pos_all[:s_idx+1, 1], chaser_pos_all[:s_idx+1, 0], color=trail_color, linestyle="-", linewidth=2.2, label=f"Chaser ({ctrl_short})")

        # Start markers
        ax_map.scatter(target_pos_all[0, 1], target_pos_all[0, 0], marker="o", color="#d62728", s=45, alpha=0.8, label="Target Start")
        ax_map.scatter(chaser_pos_all[0, 1], chaser_pos_all[0, 0], marker="^", color="black", s=45, alpha=0.8, label="Chaser Origin")

        # Visual Sightline (connecting line)
        line_color = "#2ca02c" if telem["in_view"] else "#d95f02"
        line_lbl = "LOS Locked (In FOV)" if telem["in_view"] else "Target Lost (Coasting)"
        ax_map.plot([chaser_pos[1], target_pos[1]], [chaser_pos[0], target_pos[0]],
                    color=line_color, linestyle="--", linewidth=1.2, alpha=0.8, label=line_lbl)

        # Chaser Drone Marker & Heading Pointer
        ax_map.scatter(chaser_pos[1], chaser_pos[0], color=drone_color, s=75, zorder=5)
        arrow_len = 2.2
        ax_map.arrow(chaser_pos[1], chaser_pos[0], arrow_len * np.sin(yaw), arrow_len * np.cos(yaw),
                     head_width=0.8, head_length=0.8, fc=drone_color, ec=drone_color, zorder=6)

        # Target Drone Marker
        ax_map.scatter(target_pos[1], target_pos[0], color="#d62728", s=85, marker="o", edgecolors="black", linewidths=1.2, zorder=5)

        ax_map.set_xlim(e_center - half_span, e_center + half_span)
        ax_map.set_ylim(n_center - half_span, n_center + half_span)
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.set_xlabel("East Position Y (m)", fontsize=10)
        ax_map.set_ylabel("North Position X (m)", fontsize=10)
        ax_map.set_title(f"Top-Down Pursuit Arena | {ctrl_short} | t = {t_now:.1f}s", fontsize=11, fontweight="bold", pad=8)
        ax_map.legend(loc="upper left", fontsize=8.0, framealpha=0.92)

        # ------------------------------------------------------------------
        # 3. Telemetry Panel
        # ------------------------------------------------------------------
        ax_hud.axis("off")
        ax_hud.set_facecolor("#ffffff")

        status_str = telem.get("status", "LOCKED" if telem["in_view"] else "LOST")
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
            f"Profile:          {profile.title()}\n"
            f"Encounter:        {encounter_label[:22]}\n"
            f"Controller:       {ctrl_label}\n"
            f"Tracking Status:  {status_str}\n"
            f"In Camera FOV:    {'YES' if telem['in_view'] else 'NO (COASTING)'}\n"
            f"\n"
            f"VISUAL SERVOING ERRORS\n"
            f"─────────────────────────────\n"
            f"Distance to Target: {current_dist:5.2f} m  (Goal: 6.0 m)\n"
            f"Bounding Box Size:  {target_size:5.1f} px (Goal: 32 px)\n"
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
            f"Envelope:           vx[-5,15] vy[-6,6]"
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

    # Optional GIF generation using ffmpeg palettegen
    if output_gif_path is not None:
        output_gif_path = Path(output_gif_path)
        print(f"[ENCODE] Generating optimized GIF: {output_gif_path.name}...")
        gif_cmd = [
            "ffmpeg", "-y",
            "-i", str(output_path),
            "-vf", "fps=15,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            str(output_gif_path)
        ]
        subprocess.run(gif_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"[SUCCESS] GIF generated: {output_gif_path} ({output_gif_path.stat().st_size / (1024*1024):.2f} MB)")

    # Copy to artifact directories for user review
    for adir in ARTIFACT_DIRS:
        if adir.exists():
            dst_mp4 = adir / output_path.name
            if output_path.resolve() != dst_mp4.resolve():
                shutil.copy(output_path, dst_mp4)
            if output_gif_path and output_gif_path.exists():
                dst_gif = adir / output_gif_path.name
                if output_gif_path.resolve() != dst_gif.resolve():
                    shutil.copy(output_gif_path, dst_gif)

    return output_path


def render_dual_chase_video(
    output_path: Path,
    duration: float = 20.0,
    fps: int = 25,
    profile: str = "evasive",
    encounter: str = "crossing_right",
    target_init_pos: Optional[list] = None,
    target_init_heading_deg: Optional[float] = None,
    seed: int = 42,
    model_path: Optional[Union[str, Path]] = None,
    output_gif_path: Optional[Path] = None,
):
    """
    Renders a unified top-down video where BOTH PID and Learned RL (Gen 4) drones
    pursue the EXACT same target trajectory simultaneously on the same map.
    Includes a side-by-side telemetry HUD.
    """
    dt_sim = 0.02
    total_sim_steps = int(round(duration / dt_sim))
    sim_steps_per_frame = max(1, int(round((1.0 / fps) / dt_sim)))

    init_pos, init_heading, encounter_label = get_encounter_init(encounter, target_init_pos, target_init_heading_deg)

    print(f"\n[SIM-DUAL] Initializing Dual Controller Simulation: {duration}s...")
    print(f"           Profile: {profile.upper()} | Encounter: {encounter_label} | Seed: {seed}")

    sim_pid = FastPixhawkQuadSim(dt=dt_sim)
    sim_pid.reset(initial_pos=[0.0, 0.0, -2.0], initial_yaw=0.0)

    sim_rl = FastPixhawkQuadSim(dt=dt_sim)
    sim_rl.reset(initial_pos=[0.0, 0.0, -2.0], initial_yaw=0.0)

    # 1. Classical PID Controller
    ctrl_pid = KinematicVisualServoController(
        camera_uptilt_deg=15.0,
        hfov_deg=60.0,
        vfov_deg=45.0,
        desired_bbox_size=32.33,
        desired_standoff_dist=6.0,
        use_bbox_size=False,
        enable_lateral_strafe=True,
        kp_lat=2.5,
        kd_lat=0.35,
        max_lat_vel=6.0,
        max_lat_accel=5.0,
        max_accel=6.5,
        max_decel=5.0
    )

    # 2. Learned Recurrent PPO Controller
    if model_path is None:
        default_p = PROJECT_ROOT / "control" / "weights" / "recurrent_ppo_continuous_potential" / "best_model" / "best_model.zip"
        if not default_p.exists():
            default_p = PROJECT_ROOT / "control" / "weights" / "recurrent_ppo_tail_chase_finetune" / "best_model" / "best_model.zip"
        model_path = default_p

    if RecurrentVisualServoController is not None:
        ctrl_rl = RecurrentVisualServoController(model_path=str(model_path), w_nominal=0.0505)
    else:
        raise ImportError("RecurrentVisualServoController could not be imported.")

    # 3. Target Trajectory Oracle (Identical for both!)
    target_oracle = StochasticTargetTrajectory(
        duration=duration + 5.0,
        profile=profile,
        initial_pos=init_pos,
        initial_heading_rad=init_heading,
        seed=seed
    )

    print("[SIM-DUAL] Running synchronized flight simulations...")
    times = []
    target_pos_all = []
    pid_pos_all, pid_yaw_all, pid_vel_all, pid_telem_all, pid_cmd_all = [], [], [], [], []
    rl_pos_all, rl_yaw_all, rl_vel_all, rl_telem_all, rl_cmd_all = [], [], [], [], []

    for step in range(total_sim_steps):
        t = step * dt_sim
        tgt_world = target_oracle.get_position(t)

        # PID step
        telem_p = sim_pid.get_camera_telemetry(tgt_world, target_w_m=0.35, target_h_m=0.20, img_w=640, img_h=480, desired_target_size=32.33)
        cmd_p = ctrl_pid.compute_cmd(telem_p, drone_pitch=sim_pid.pitch, dt=dt_sim)
        sim_pid.step(cmd_p)

        # RL step
        telem_r = sim_rl.get_camera_telemetry(tgt_world, target_w_m=0.35, target_h_m=0.20, img_w=640, img_h=480, desired_target_size=32.33)
        c, s = np.cos(sim_rl.yaw), np.sin(sim_rl.yaw)
        vx_b = c * sim_rl.vel[0] + s * sim_rl.vel[1]
        vy_b = -s * sim_rl.vel[0] + c * sim_rl.vel[1]
        vz_b = sim_rl.vel[2]
        cmd_r = ctrl_rl.compute_cmd(
            telem_r,
            drone_pitch=sim_rl.pitch,
            drone_roll=sim_rl.roll,
            dt=dt_sim,
            vehicle_vel=np.array([vx_b, vy_b, vz_b]),
            vehicle_yaw_rate=sim_rl.yaw_rate,
            current_alt_m=-sim_rl.pos[2]
        )
        sim_rl.step(cmd_r)

        times.append(t)
        target_pos_all.append(tgt_world.copy())
        pid_pos_all.append(sim_pid.pos.copy())
        pid_yaw_all.append(sim_pid.yaw)
        pid_vel_all.append(sim_pid.vel.copy())
        pid_telem_all.append(telem_p.copy())
        pid_cmd_all.append(cmd_p.copy())

        rl_pos_all.append(sim_rl.pos.copy())
        rl_yaw_all.append(sim_rl.yaw)
        rl_vel_all.append(sim_rl.vel.copy())
        rl_telem_all.append(telem_r.copy())
        rl_cmd_all.append(cmd_r.copy())

    times = np.array(times)
    target_pos_all = np.array(target_pos_all)
    pid_pos_all = np.array(pid_pos_all)
    pid_yaw_all = np.array(pid_yaw_all)
    pid_vel_all = np.array(pid_vel_all)
    rl_pos_all = np.array(rl_pos_all)
    rl_yaw_all = np.array(rl_yaw_all)
    rl_vel_all = np.array(rl_vel_all)

    frame_indices = np.arange(0, total_sim_steps, sim_steps_per_frame)
    num_frames = len(frame_indices)
    print(f"[RENDER-DUAL] Rendering {num_frames} frames ({duration:.1f}s at {fps} FPS)...")

    fig_w, fig_h = 13.6, 7.6
    dpi = 100
    width_px = int(fig_w * dpi)
    height_px = int(fig_h * dpi)

    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    plt.style.use("default")
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#ffffff")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], left=0.07, right=0.96, top=0.92, bottom=0.08, wspace=0.16)

    ax_map = fig.add_subplot(gs[0, 0])
    ax_hud = fig.add_subplot(gs[0, 1])

    all_east = np.concatenate([target_pos_all[:, 1], pid_pos_all[:, 1], rl_pos_all[:, 1]])
    all_north = np.concatenate([target_pos_all[:, 0], pid_pos_all[:, 0], rl_pos_all[:, 0]])
    e_min, e_max = np.min(all_east) - 5.0, np.max(all_east) + 5.0
    n_min, n_max = np.min(all_north) - 5.0, np.max(all_north) + 5.0
    e_center = (e_min + e_max) / 2.0
    n_center = (n_min + n_max) / 2.0
    half_span = max((e_max - e_min), (n_max - n_min)) / 2.0 + 3.0

    for f_idx, s_idx in enumerate(frame_indices):
        t_now = times[s_idx]
        tgt_p = target_pos_all[s_idx]
        p_pos = pid_pos_all[s_idx]
        p_yaw = pid_yaw_all[s_idx]
        p_tel = pid_telem_all[s_idx]
        p_cmd = pid_cmd_all[s_idx]
        p_vel = pid_vel_all[s_idx]

        r_pos = rl_pos_all[s_idx]
        r_yaw = rl_yaw_all[s_idx]
        r_tel = rl_telem_all[s_idx]
        r_cmd = rl_cmd_all[s_idx]
        r_vel = rl_vel_all[s_idx]

        ax_map.clear()
        ax_hud.clear()

        ax_map.set_facecolor("#fafafa")
        ax_map.grid(True, linestyle="--", alpha=0.5, color="#c0c0c0")

        # Full planned paths (dotted background)
        ax_map.plot(target_pos_all[:, 1], target_pos_all[:, 0], color="#d62728", linestyle=":", linewidth=1.0, alpha=0.35)
        ax_map.plot(pid_pos_all[:, 1], pid_pos_all[:, 0], color="#1f77b4", linestyle=":", linewidth=1.0, alpha=0.35)
        ax_map.plot(rl_pos_all[:, 1], rl_pos_all[:, 0], color="#2ca02c", linestyle=":", linewidth=1.0, alpha=0.35)

        # Elapsed trails
        ax_map.plot(target_pos_all[:s_idx+1, 1], target_pos_all[:s_idx+1, 0], color="#d62728", linestyle="-", linewidth=2.2, label="Target Drone (Identical)")
        ax_map.plot(pid_pos_all[:s_idx+1, 1], pid_pos_all[:s_idx+1, 0], color="#1f77b4", linestyle="--", linewidth=2.0, label="Classical PID Chaser")
        ax_map.plot(rl_pos_all[:s_idx+1, 1], rl_pos_all[:s_idx+1, 0], color="#2ca02c", linestyle="-.", linewidth=2.0, label="Learned Gen 4 RL Chaser")

        # Markers
        ax_map.scatter(target_pos_all[0, 1], target_pos_all[0, 0], color="#d62728", s=50, marker="o", label="Target Start")
        ax_map.scatter(0.0, 0.0, color="black", s=50, marker="^", label="Chaser Start")

        # Sightlines
        color_p = "#1f77b4" if p_tel["in_view"] else "#a6cee3"
        color_r = "#2ca02c" if r_tel["in_view"] else "#b2df8a"
        ax_map.plot([p_pos[1], tgt_p[1]], [p_pos[0], tgt_p[0]], color=color_p, linestyle=":", linewidth=1.1, alpha=0.7)
        ax_map.plot([r_pos[1], tgt_p[1]], [r_pos[0], tgt_p[0]], color=color_r, linestyle=":", linewidth=1.1, alpha=0.7)

        # PID Drone marker & pointer
        ax_map.scatter(p_pos[1], p_pos[0], color="#1f77b4", s=75, zorder=6)
        arrow_len = 2.0
        ax_map.arrow(p_pos[1], p_pos[0], arrow_len * np.sin(p_yaw), arrow_len * np.cos(p_yaw),
                     head_width=0.7, head_length=0.7, fc="#1f77b4", ec="#1f77b4", zorder=7)

        # RL Drone marker & pointer
        ax_map.scatter(r_pos[1], r_pos[0], color="#2ca02c", s=75, zorder=6)
        ax_map.arrow(r_pos[1], r_pos[0], arrow_len * np.sin(r_yaw), arrow_len * np.cos(r_yaw),
                     head_width=0.7, head_length=0.7, fc="#2ca02c", ec="#2ca02c", zorder=7)

        # Target Drone
        ax_map.scatter(tgt_p[1], tgt_p[0], color="#d62728", s=85, marker="o", edgecolors="black", linewidths=1.2, zorder=6)

        ax_map.set_xlim(e_center - half_span, e_center + half_span)
        ax_map.set_ylim(n_center - half_span, n_center + half_span)
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.set_xlabel("East Position Y (m)", fontsize=10)
        ax_map.set_ylabel("North Position X (m)", fontsize=10)
        ax_map.set_title(f"Dual Head-to-Head Pursuit | t = {t_now:.1f}s", fontsize=11, fontweight="bold", pad=8)
        ax_map.legend(loc="upper left", fontsize=8.0, framealpha=0.92)

        # HUD Dual Table
        ax_hud.axis("off")
        ax_hud.set_facecolor("#ffffff")

        d_p = np.linalg.norm(tgt_p - p_pos)
        d_r = np.linalg.norm(tgt_p - r_pos)
        spd_p = np.linalg.norm(p_vel)
        spd_r = np.linalg.norm(r_vel)

        hud_text = (
            f"DUAL COMPARISON TELEMETRY\n"
            f"─────────────────────────────────────\n"
            f"Time:       {t_now:5.2f} s\n"
            f"Profile:    {profile.title()}\n"
            f"Encounter:  {encounter_label[:24]}\n"
            f"Target:     Red Drone (Exact Same)\n"
            f"\n"
            f"METRIC            PID (Blue)   RL (Green)\n"
            f"─────────────────────────────────────\n"
            f"In FOV:           {'YES' if p_tel['in_view'] else 'NO':<10}   {'YES' if r_tel['in_view'] else 'NO':<10}\n"
            f"Distance:         {d_p:5.2f} m       {d_r:5.2f} m\n"
            f"Goal Error:       {abs(d_p-6.0):+5.2f} m       {abs(d_r-6.0):+5.2f} m\n"
            f"Azimuth Err (ex): {p_tel.get('error_x',0.0):+5.2f}        {r_tel.get('error_x',0.0):+5.2f}\n"
            f"Elevation Err:    {p_tel.get('error_y',0.0):+5.2f}        {r_tel.get('error_y',0.0):+5.2f}\n"
            f"\n"
            f"COMMANDS & VELOCITIES\n"
            f"─────────────────────────────────────\n"
            f"Ground Speed:     {spd_p:5.2f} m/s     {spd_r:5.2f} m/s\n"
            f"cmd_vx (Forward): {p_cmd[0]:+5.2f} m/s     {r_cmd[0]:+5.2f} m/s\n"
            f"cmd_vy (Lateral): {p_cmd[1]:+5.2f} m/s     {r_cmd[1]:+5.2f} m/s\n"
            f"cmd_yaw (Rate):   {p_cmd[3]:+5.1f} °/s     {r_cmd[3]:+5.1f} °/s\n"
            f"Altitude (AGL):   {-p_pos[2]:5.2f} m       {-r_pos[2]:5.2f} m\n"
            f"Physical Limits:  Identical Envelope"
        )

        ax_hud.text(
            0.04, 0.95, hud_text,
            transform=ax_hud.transAxes,
            fontsize=8.8,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="square,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced4da", linewidth=1.0)
        )

        fig.canvas.draw()
        rgba_buffer = fig.canvas.buffer_rgba()
        pipe.stdin.write(rgba_buffer)

        if f_idx % 50 == 0 or f_idx == num_frames - 1:
            print(f"[RENDER-DUAL] Frame {f_idx+1}/{num_frames} ({((f_idx+1)/num_frames)*100:.1f}%) rendered...")

    pipe.stdin.close()
    pipe.wait()
    plt.close(fig)

    print(f"\n[SUCCESS] Dual overlay video saved to: {output_path}")

    if output_gif_path is not None:
        output_gif_path = Path(output_gif_path)
        print(f"[ENCODE] Generating optimized dual GIF: {output_gif_path.name}...")
        gif_cmd = [
            "ffmpeg", "-y",
            "-i", str(output_path),
            "-vf", "fps=15,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            str(output_gif_path)
        ]
        subprocess.run(gif_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"[SUCCESS] Dual GIF generated: {output_gif_path} ({output_gif_path.stat().st_size / (1024*1024):.2f} MB)")

    for adir in ARTIFACT_DIRS:
        if adir.exists():
            dst_mp4 = adir / output_path.name
            if output_path.resolve() != dst_mp4.resolve():
                shutil.copy(output_path, dst_mp4)
            if output_gif_path and output_gif_path.exists():
                dst_gif = adir / output_gif_path.name
                if output_gif_path.resolve() != dst_gif.resolve():
                    shutil.copy(output_gif_path, dst_gif)

    return output_path


def render_side_by_side_video(
    pid_video_path: Path,
    rl_video_path: Path,
    output_path: Path,
    output_gif_path: Optional[Path] = None,
) -> Path:
    """Stitches PID and RL videos side by side into a single comparison video using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[COMBINE] Generating Side-by-Side video from:\n  Left:  {pid_video_path.name}\n  Right: {rl_video_path.name}")

    # Scale each video to 960x540 so output is 1920x540
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", str(pid_video_path),
        "-i", str(rl_video_path),
        "-filter_complex",
        "[0:v]scale=960:540[v0];[1:v]scale=960:540[v1];[v0][v1]hstack=inputs=2[v]",
        "-map", "[v]",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path)
    ]
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"[SUCCESS] Side-by-Side comparison video saved to: {output_path}")

    if output_gif_path is not None:
        output_gif_path = Path(output_gif_path)
        print(f"[ENCODE] Generating side-by-side GIF: {output_gif_path.name}...")
        gif_cmd = [
            "ffmpeg", "-y",
            "-i", str(output_path),
            "-vf", "fps=12,scale=1280:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            str(output_gif_path)
        ]
        subprocess.run(gif_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"[SUCCESS] Side-by-Side GIF generated: {output_gif_path} ({output_gif_path.stat().st_size / (1024*1024):.2f} MB)")

    for adir in ARTIFACT_DIRS:
        if adir.exists():
            dst_mp4 = adir / output_path.name
            if output_path.resolve() != dst_mp4.resolve():
                shutil.copy(output_path, dst_mp4)
            if output_gif_path and output_gif_path.exists():
                dst_gif = adir / output_gif_path.name
                if output_gif_path.resolve() != dst_gif.resolve():
                    shutil.copy(output_gif_path, dst_gif)

    return output_path


def compute_shared_bounds(
    duration: float,
    profile: str,
    encounter: str,
    target_init_pos: Optional[list],
    target_init_heading_deg: Optional[float],
    seed: int,
    model_path: Optional[Union[str, Path]] = None,
) -> Tuple[float, float, float]:
    """Pre-runs both PID and RL to compute a unified bounding box for identical visual scaling."""
    dt_sim = 0.02
    total_sim_steps = int(round(duration / dt_sim))
    init_pos, init_heading, _ = get_encounter_init(encounter, target_init_pos, target_init_heading_deg)

    target_oracle = StochasticTargetTrajectory(
        duration=duration + 5.0, profile=profile, initial_pos=init_pos, initial_heading_rad=init_heading, seed=seed
    )

    all_e, all_n = [], []
    for ctype in ["pid", "rl"]:
        sim = FastPixhawkQuadSim(dt=dt_sim)
        sim.reset(initial_pos=[0.0, 0.0, -2.0], initial_yaw=0.0)

        if ctype == "pid":
            ctrl = KinematicVisualServoController(
                camera_uptilt_deg=15.0, hfov_deg=60.0, vfov_deg=45.0, desired_standoff_dist=6.0,
                enable_lateral_strafe=True, kp_lat=2.5, kd_lat=0.35, max_lat_vel=6.0, max_lat_accel=5.0, max_accel=6.5, max_decel=5.0
            )
        else:
            if model_path is None:
                model_path = PROJECT_ROOT / "control" / "weights" / "recurrent_ppo_continuous_potential" / "best_model" / "best_model.zip"
            ctrl = RecurrentVisualServoController(model_path=str(model_path), w_nominal=0.0505)

        for step in range(total_sim_steps):
            t = step * dt_sim
            tgt = target_oracle.get_position(t)
            all_e.append(tgt[1])
            all_n.append(tgt[0])
            telem = sim.get_camera_telemetry(tgt, target_w_m=0.35, target_h_m=0.20, img_w=640, img_h=480, desired_target_size=32.33)
            if ctype == "pid":
                cmd = ctrl.compute_cmd(telem, drone_pitch=sim.pitch, dt=dt_sim)
            else:
                c, s = np.cos(sim.yaw), np.sin(sim.yaw)
                vx_b = c * sim.vel[0] + s * sim.vel[1]
                vy_b = -s * sim.vel[0] + c * sim.vel[1]
                vz_b = sim.vel[2]
                cmd = ctrl.compute_cmd(telem, drone_pitch=sim.pitch, drone_roll=sim.roll, dt=dt_sim,
                                       vehicle_vel=np.array([vx_b, vy_b, vz_b]), vehicle_yaw_rate=sim.yaw_rate, current_alt_m=-sim.pos[2])
            sim.step(cmd)
            all_e.append(sim.pos[1])
            all_n.append(sim.pos[0])

    all_e = np.array(all_e)
    all_n = np.array(all_n)
    e_min, e_max = np.min(all_e) - 5.0, np.max(all_e) + 5.0
    n_min, n_max = np.min(all_n) - 5.0, np.max(all_n) + 5.0
    e_center = (e_min + e_max) / 2.0
    n_center = (n_min + n_max) / 2.0
    half_span = max((e_max - e_min), (n_max - n_min)) / 2.0 + 3.0
    return e_center, n_center, half_span


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Simple 2D Top-Down Drone Pursuit Movie")
    parser.add_argument("--profile", type=str, default="evasive",
                        help="Target trajectory profile: evasive, hyper_evasive, aerobatic, sprint_break, cruising")
    parser.add_argument("--encounter", type=str, default="crossing_right",
                        choices=["crossing_right", "crossing_left", "tail_chase", "custom"],
                        help="Encounter geometry: crossing_right, crossing_left, tail_chase, custom")
    parser.add_argument("--target-init-pos", type=float, nargs=3, default=None,
                        help="Custom initial target coordinate in NED [x, y, z]")
    parser.add_argument("--target-init-heading", type=float, default=None,
                        help="Custom initial target heading in degrees")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration in seconds (default: 20.0)")
    parser.add_argument("--fps", type=int, default=25, help="Video framerate (default: 25)")
    parser.add_argument("--controller", type=str, default="both",
                        choices=["pid", "rl", "recurrent_ppo", "both", "dual"],
                        help="Controller mode: 'pid', 'rl', 'both' (renders both + side-by-side), or 'dual' (overlay)")
    parser.add_argument("--model", type=str, default=None, help="Path to trained policy .zip")
    parser.add_argument("--out", type=str, default=None, help="Output MP4 path")
    parser.add_argument("--gif", action="store_true", help="Also generate animated GIF(s)")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "outputs" / "videos" / "control"
    out_dir.mkdir(parents=True, exist_ok=True)

    enc_tag = args.encounter
    if args.controller in ("both", "dual"):
        shared_bounds = compute_shared_bounds(
            duration=args.duration,
            profile=args.profile,
            encounter=args.encounter,
            target_init_pos=args.target_init_pos,
            target_init_heading_deg=args.target_init_heading,
            seed=args.seed,
            model_path=args.model,
        )

        if args.controller == "both":
            # 1. Render PID video
            pid_mp4 = out_dir / f"simple_chase_{args.profile}_{enc_tag}_pid_seed{args.seed}.mp4"
            pid_gif = pid_mp4.with_suffix(".gif") if args.gif else None
            render_simple_chase_video(
                output_path=pid_mp4, duration=args.duration, fps=args.fps, profile=args.profile,
                encounter=args.encounter, target_init_pos=args.target_init_pos, target_init_heading_deg=args.target_init_heading,
                seed=args.seed, controller_type="pid", output_gif_path=pid_gif, fixed_bounds=shared_bounds
            )

            # 2. Render RL video
            rl_mp4 = out_dir / f"simple_chase_{args.profile}_{enc_tag}_rl_seed{args.seed}.mp4"
            rl_gif = rl_mp4.with_suffix(".gif") if args.gif else None
            render_simple_chase_video(
                output_path=rl_mp4, duration=args.duration, fps=args.fps, profile=args.profile,
                encounter=args.encounter, target_init_pos=args.target_init_pos, target_init_heading_deg=args.target_init_heading,
                seed=args.seed, controller_type="rl", model_path=args.model, output_gif_path=rl_gif, fixed_bounds=shared_bounds
            )

            # 3. Side-by-side stitched video
            sbs_mp4 = out_dir / f"simple_chase_{args.profile}_{enc_tag}_side_by_side_seed{args.seed}.mp4"
            sbs_gif = sbs_mp4.with_suffix(".gif") if args.gif else None
            render_side_by_side_video(pid_mp4, rl_mp4, sbs_mp4, output_gif_path=sbs_gif)

            # 4. Also render dual-overlay
            dual_mp4 = out_dir / f"simple_chase_{args.profile}_{enc_tag}_dual_overlay_seed{args.seed}.mp4"
            dual_gif = dual_mp4.with_suffix(".gif") if args.gif else None
            render_dual_chase_video(
                output_path=dual_mp4, duration=args.duration, fps=args.fps, profile=args.profile,
                encounter=args.encounter, target_init_pos=args.target_init_pos, target_init_heading_deg=args.target_init_heading,
                seed=args.seed, model_path=args.model, output_gif_path=dual_gif
            )

        elif args.controller == "dual":
            dual_mp4 = out_dir / f"simple_chase_{args.profile}_{enc_tag}_dual_overlay_seed{args.seed}.mp4"
            dual_gif = dual_mp4.with_suffix(".gif") if args.gif else None
            render_dual_chase_video(
                output_path=dual_mp4, duration=args.duration, fps=args.fps, profile=args.profile,
                encounter=args.encounter, target_init_pos=args.target_init_pos, target_init_heading_deg=args.target_init_heading,
                seed=args.seed, model_path=args.model, output_gif_path=dual_gif
            )
    else:
        ctrl_tag = "rl" if args.controller in ("rl", "recurrent_ppo") else "pid"
        if args.out:
            mp4_path = Path(args.out)
        else:
            mp4_path = out_dir / f"simple_chase_{args.profile}_{enc_tag}_{ctrl_tag}_seed{args.seed}.mp4"

        gif_path = mp4_path.with_suffix(".gif") if args.gif else None

        render_simple_chase_video(
            output_path=mp4_path,
            duration=args.duration,
            fps=args.fps,
            profile=args.profile,
            encounter=args.encounter,
            target_init_pos=args.target_init_pos,
            target_init_heading_deg=args.target_init_heading,
            seed=args.seed,
            controller_type=args.controller,
            model_path=args.model,
            output_gif_path=gif_path,
        )

