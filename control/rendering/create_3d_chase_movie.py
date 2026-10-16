#!/usr/bin/env python3
"""
3D Spatial Autonomous Drone Chase Movie Generator.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Renders a broadcast-quality, synchronized 3D spatial pursuit animation:
- Left: Dynamic 3D flight arena with ground shadow projections, vertical altitude lines,
  3D airframe attitude orientation, orbiting cinematic camera, and 3D line-of-sight laser.
- Right Top: On-board camera sensor FPV viewfinder with artificial horizon and YOLO target bbox.
- Right Center: Live Pixhawk MAVSDK flight telemetry HUD.
- Right Bottom: Real-time 3D altitude profile and standoff distance strip charts.

Pipes raw RGBA frames directly into ffmpeg for fast, high-quality H.264 encoding.
"""

import sys
import os
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Callable, Union
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, Circle

from control.rendering.create_chase_movie import get_target_trajectory, simulate_pursuit


ARTIFACT_DIRS = [
    Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/1da038ce-1011-475e-b1de-1614f330a01a"),
    Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/53eb0b3e-b75c-4830-a9d4-c69e644ccb6a"),
    Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/42b621c5-bd5c-42b8-ae83-87208090e0a7"),
    Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/c2b8fddd-4624-46f7-9768-e1974371d76e"),
]


def render_3d_movie_frames(
    sim_data: dict,
    output_mp4_path: Path,
    output_gif_path: Optional[Path] = None,
    trajectory_title: str = "3D Pursuit",
    fps: int = 25,
):
    """
    Renders synchronized 3D video frames and pipes directly into ffmpeg.
    """
    times = sim_data["times"]
    c_pos = sim_data["chaser_pos"]       # [North, East, Down]
    c_vel = sim_data["chaser_vel"]
    c_yaw = sim_data["chaser_yaw"]
    c_pitch = sim_data["chaser_pitch"]
    c_roll = sim_data["chaser_roll"]
    c_cam_pitch = sim_data["chaser_cam_pitch"]
    t_pos = sim_data["target_pos"]       # [North, East, Down]
    dists = sim_data["distance"]
    err_xs = sim_data["err_x"]
    err_ys = sim_data["err_y"]
    in_views = sim_data["in_view"]
    cmds = sim_data["cmd"]
    bboxes = sim_data["bbox"]

    total_duration = sim_data["duration"]
    dt_sim = sim_data["dt"]
    total_sim_steps = len(times)

    # Empirical YOLO & Kalman Filter Perception Telemetry
    gt_bboxes = sim_data.get("gt_bbox", [None] * total_sim_steps)
    yolo_bboxes = sim_data.get("yolo_bbox", [None] * total_sim_steps)
    yolo_confs = sim_data.get("yolo_conf", np.zeros(total_sim_steps))
    yolo_drop_bursts = sim_data.get("yolo_drop_burst", np.zeros(total_sim_steps))
    kalman_is_coasting_all = sim_data.get("kalman_is_coasting", np.zeros(total_sim_steps, dtype=bool))
    kalman_coasting_ms_all = sim_data.get("kalman_coasting_ms", np.zeros(total_sim_steps))
    kalman_vels = sim_data.get("kalman_vel", [(0.0, 0.0)] * total_sim_steps)
    kalman_histories = sim_data.get("kalman_history", [[] for _ in range(total_sim_steps)])

    # Frame sampling stride
    sim_steps_per_frame = max(1, int(round((1.0 / fps) / dt_sim)))
    frame_indices = np.arange(0, total_sim_steps, sim_steps_per_frame)
    num_frames = len(frame_indices)

    print(f"[RENDER 3D] Preparing 3D Pursuit Movie: {num_frames} frames ({total_duration:.1f}s at {fps} FPS)...")

    # Figure dimensions (16:9 1600x900)
    fig_w, fig_h = 16.0, 9.0
    dpi = 100
    width_px = int(fig_w * dpi)
    height_px = int(fig_h * dpi)

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

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#0b0e14")
    gs = fig.add_gridspec(3, 3, width_ratios=[1.75, 1.0, 0.95], height_ratios=[1.15, 1.0, 1.0],
                           left=0.04, right=0.97, top=0.94, bottom=0.06, hspace=0.32, wspace=0.28)

    # Convert coordinates: In plot, X=East, Y=North, Z=Altitude (-Down)
    t_east = t_pos[:, 1]
    t_north = t_pos[:, 0]
    t_alt = -t_pos[:, 2]

    c_east = c_pos[:, 1]
    c_north = c_pos[:, 0]
    c_alt = -c_pos[:, 2]

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
        bbox = bboxes[step_idx]

        gt_box = gt_bboxes[step_idx]
        yolo_box = yolo_bboxes[step_idx]
        yolo_conf = float(yolo_confs[step_idx])
        yolo_burst = int(yolo_drop_bursts[step_idx])
        k_coasting = bool(kalman_is_coasting_all[step_idx])
        k_coast_ms = int(kalman_coasting_ms_all[step_idx])
        k_vel = kalman_vels[step_idx]
        k_hist = kalman_histories[step_idx]

        c_alt_curr = -cz
        t_alt_curr = -tz

        # ----------------------------------------------------------------------
        # 1. Main 3D Spatial Arena
        # ----------------------------------------------------------------------
        ax_3d = fig.add_subplot(gs[:, 0], projection="3d", facecolor="#10141d")
        ax_3d.set_facecolor("#10141d")

        # History window (last 6 seconds)
        trail_start = max(0, step_idx - int(6.0 / dt_sim))
        c_t_e = c_east[trail_start:step_idx + 1]
        c_t_n = c_north[trail_start:step_idx + 1]
        c_t_a = c_alt[trail_start:step_idx + 1]

        t_t_e = t_east[trail_start:step_idx + 1]
        t_t_n = t_north[trail_start:step_idx + 1]
        t_t_a = t_alt[trail_start:step_idx + 1]

        # Full target ghost trajectory
        ax_3d.plot(t_east, t_north, t_alt, color="#ff4757", linestyle=":", lw=1.0, alpha=0.25)

        # Fading 3D flight paths
        ax_3d.plot(t_t_e, t_t_n, t_t_a, color="#ff4757", lw=2.2, alpha=0.85, label="Target UAV")
        ax_3d.plot(c_t_e, c_t_n, c_t_a, color="#00d2d3", lw=2.8, alpha=0.90, label="Chaser")

        # Ground Shadows (Z = 0)
        ax_3d.plot(t_t_e, t_t_n, np.zeros_like(t_t_e), color="#ff4757", linestyle="--", lw=1.0, alpha=0.3)
        ax_3d.plot(c_t_e, c_t_n, np.zeros_like(c_t_e), color="#00d2d3", linestyle="--", lw=1.0, alpha=0.3)

        # Vertical Altitude Stems to Ground
        ax_3d.plot([ty, ty], [tx, tx], [0, t_alt_curr], color="#ff6b81", linestyle=":", lw=1.5, alpha=0.6)
        ax_3d.plot([cy, cy], [cx, cx], [0, c_alt_curr], color="#48dbfb", linestyle=":", lw=1.5, alpha=0.6)

        # Ground Shadow Discs
        ax_3d.scatter([ty], [tx], [0], color="#57606f", s=30, alpha=0.5)
        ax_3d.scatter([cy], [cx], [0], color="#57606f", s=30, alpha=0.5)

        # 3D Line-of-Sight Laser (Green = Active Lock, Amber = Coasting Dead Reckoning)
        if in_view:
            los_color = "#2ed573" if not k_coasting else "#ff7f50"
            los_style = "-" if not k_coasting else "--"
        else:
            los_color = "#ff4757"
            los_style = ":"
        ax_3d.plot([cy, ty], [cx, tx], [c_alt_curr, t_alt_curr], color=los_color, linestyle=los_style, lw=1.5, alpha=0.8)

        # Drone Markers
        ax_3d.scatter([ty], [tx], [t_alt_curr], color="#ff4757", edgecolor="white", s=90, depthshade=False)
        ax_3d.scatter([cy], [cx], [c_alt_curr], color="#00d2d3", edgecolor="white", s=110, depthshade=False)

        # Forward Heading Vector for Chaser
        h_len = 2.5
        h_e = cy + h_len * np.sin(yaw)
        h_n = cx + h_len * np.cos(yaw)
        h_a = c_alt_curr + h_len * np.sin(pitch)
        ax_3d.plot([cy, h_e], [cx, h_n], [c_alt_curr, h_a], color="#1dd1a1", lw=2.0)

        # Auto-framing 3D Bounds
        mid_e = 0.5 * (cy + ty)
        mid_n = 0.5 * (cx + tx)
        span = max(18.0, abs(cy - ty) * 1.5, abs(cx - tx) * 1.5)
        ax_3d.set_xlim(mid_e - span / 2.0, mid_e + span / 2.0)
        ax_3d.set_ylim(mid_n - span / 2.0, mid_n + span / 2.0)
        max_alt = max(5.0, c_alt_curr + 3.0, t_alt_curr + 3.0)
        ax_3d.set_zlim(0, max_alt)

        # Cinematic slowly orbiting camera angle
        elev = 26.0
        azim = 45.0 + 25.0 * np.sin(0.18 * t_curr)
        ax_3d.view_init(elev=elev, azim=azim)

        ax_3d.set_xlabel("East (m)", color="#a4b0be", fontsize=9, labelpad=2)
        ax_3d.set_ylabel("North (m)", color="#a4b0be", fontsize=9, labelpad=2)
        ax_3d.set_zlabel("Alt (m)", color="#a4b0be", fontsize=9, labelpad=2)
        ax_3d.tick_params(colors="#747d8c", labelsize=7)
        ax_3d.set_title(f"3D SPATIAL ARENA: {trajectory_title.upper()}", color="#f1f2f6", fontsize=11, fontweight="bold", pad=4)
        ax_3d.legend(loc="upper left", facecolor="#10141d", edgecolor="#2c3444", fontsize=7.5, labelcolor="#dfe4ea")

        # ----------------------------------------------------------------------
        # 2. On-Board Camera FPV Viewfinder
        # ----------------------------------------------------------------------
        ax_cam = fig.add_subplot(gs[0, 1:], facecolor="#04060a")
        ax_cam.set_xlim(0, 640)
        ax_cam.set_ylim(480, 0)
        ax_cam.set_xticks([])
        ax_cam.set_yticks([])

        horiz_y_center = 240.0 + (np.tan(cam_pitch) / np.tan(np.deg2rad(22.5))) * 240.0
        tan_roll = np.tan(-roll)
        ax_cam.plot([0, 640], [horiz_y_center - 320.0 * tan_roll, horiz_y_center + 320.0 * tan_roll],
                    color="#1e90ff", linestyle="--", lw=1.2, alpha=0.6)

        # Crosshairs & 35px circle
        ax_cam.plot([300, 340], [240, 240], color="#00d2d3", lw=1.2, alpha=0.8)
        ax_cam.plot([320, 320], [220, 260], color="#00d2d3", lw=1.2, alpha=0.8)
        ax_cam.add_patch(Circle((320, 240), 35.0 / 2.0, fill=False, edgecolor="#00d2d3", linestyle=":", lw=1.0, alpha=0.7))

        # Ground-Truth 3D Target Projection (Dashed Light Green)
        if gt_box is not None:
            gx1, gy1, gx2, gy2 = gt_box
            gw = max(1.0, gx2 - gx1)
            gh = max(1.0, gy2 - gy1)
            rect_gt = Rectangle((gx1, gy1), gw, gh, fill=False, edgecolor="#7bed9f", linestyle="--", linewidth=1.0, alpha=0.6)
            ax_cam.add_patch(rect_gt)
            ax_cam.text(gx1, max(12, gy1 - 3), "GT 3D", color="#7bed9f", fontsize=6.5, alpha=0.7)

        # Raw YOLO Measurement (Yellow Box when detected, or Dropout Alert)
        if yolo_box is not None:
            yx1, yy1, yx2, yy2 = yolo_box
            yw = max(1.0, yx2 - yx1)
            yh = max(1.0, yy2 - yy1)
            rect_yolo = Rectangle((yx1, yy1), yw, yh, fill=False, edgecolor="#ffa502", linewidth=1.4, alpha=0.85)
            ax_cam.add_patch(rect_yolo)
            ax_cam.text(yx1, min(475, yy2 + 10), f"YOLO [{yolo_conf*100:.0f}%]", color="#ffa502", fontsize=6.5, fontweight="bold",
                        bbox=dict(boxstyle="square,pad=0.1", facecolor="#141005", edgecolor="#ffa502", alpha=0.85))
        else:
            if yolo_burst > 0 and in_view:
                ax_cam.text(12, 60, f"● YOLO DROPOUT (Burst: {yolo_burst} frames / {yolo_burst*33.3:.0f}ms)",
                            color="#ff4757", fontsize=7.5, family="monospace", fontweight="bold")

        # Kalman Filter State Estimate (Neon Green when Locked, Amber when Coasting)
        if in_view and bbox is not None and bbox[2] > bbox[0]:
            bx1, by1, bx2, by2 = bbox
            bw = bx2 - bx1
            bh = by2 - by1

            if k_coasting:
                trk_color = "#ff7f50"  # Amber/Coral for Coasting
                trk_label = f"TRK-01 [COASTING +{k_coast_ms}ms]"
                status_badge = "DEAD RECKONING"
                badge_bg = "#2b1408"
            else:
                trk_color = "#2ed573"  # Neon Green for Locked
                trk_label = f"TRK-01 [LOCK: {dist:.1f}m]"
                status_badge = "LOCKED"
                badge_bg = "#081c10"

            rect = Rectangle((bx1, by1), bw, bh, fill=False, edgecolor=trk_color, linewidth=1.8)
            ax_cam.add_patch(rect)

            b_arm = min(bw, bh) * 0.25
            ax_cam.plot([bx1, bx1 + b_arm], [by1, by1], color=trk_color, lw=2.5)
            ax_cam.plot([bx1, bx1], [by1, by1 + b_arm], color=trk_color, lw=2.5)
            ax_cam.plot([bx2, bx2 - b_arm], [by2, by2], color=trk_color, lw=2.5)
            ax_cam.plot([bx2, bx2], [by2, by2 - b_arm], color=trk_color, lw=2.5)

            ax_cam.text(bx1, max(14, by1 - 5), trk_label, color=trk_color, fontsize=7.5, fontweight="bold",
                        bbox=dict(boxstyle="square,pad=0.1", facecolor=badge_bg, edgecolor=trk_color, alpha=0.9))

            # Kalman Velocity Arrow
            kcx = (bx1 + bx2) / 2.0
            kcy = (by1 + by2) / 2.0
            kvx, kvy = k_vel
            if abs(kvx) > 1.0 or abs(kvy) > 1.0:
                ax_cam.annotate(
                    "", xy=(kcx + kvx * 0.25, kcy + kvy * 0.25), xytext=(kcx, kcy),
                    arrowprops=dict(arrowstyle="->", color="#00d2d3", lw=1.5)
                )

            # Trajectory Breadcrumb trail
            if k_hist and len(k_hist) > 1:
                hist_arr = np.array(k_hist[-25:])
                ax_cam.plot(hist_arr[:, 0], hist_arr[:, 1], color=trk_color, lw=1.0, alpha=0.5, linestyle=":")
        else:
            status_badge = "SEARCHING"
            trk_color = "#ff4757"
            ax_cam.text(320, 240, "TARGET OUT OF FOV\n[ COASTING / SEARCH ]", color="#ff4757", fontsize=10, fontweight="bold",
                        ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", facecolor="#240a0e", edgecolor="#ff4757", alpha=0.85))

        ax_cam.text(12, 24, "FPV SENSOR: 1080p60 GLOBAL SHUTTER (+15° MOUNT)", color="#00d2d3", fontsize=8, family="monospace")
        ax_cam.text(628, 24, status_badge, color=trk_color, fontsize=8.5, fontweight="bold", ha="right",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#10141d", edgecolor=trk_color))
        ax_cam.set_title(f"ON-BOARD FPV SENSOR (640x480) | TIME: {t_curr:04.1f}s", color="#dfe4ea", fontsize=9, fontweight="bold")

        # ----------------------------------------------------------------------
        # 3. Autopilot Telemetry & Flight HUD
        # ----------------------------------------------------------------------
        ax_hud = fig.add_subplot(gs[1, 1:], facecolor="#10141d")
        ax_hud.axis("off")

        c_spd = np.linalg.norm(c_vel[step_idx])
        c_spd_xy = np.linalg.norm(c_vel[step_idx, :2])
        c_rate = (dists[max(0, step_idx - 5)] - dist) / (5 * dt_sim) if step_idx >= 5 else 0.0

        if in_view:
            status_str = f"COASTING (KALMAN DEAD RECKONING +{k_coast_ms}ms)" if k_coasting else "LOCKED (YOLO ACTIVE)"
        else:
            status_str = "SEARCH / RE-ACQUIRE COAST"

        hud_text = (
            f"AUTONOMOUS PURSUIT TELEMETRY — ARDUCOPTER GUIDED LINK\n"
            f"─────────────────────────────────────────────────────────────\n"
            f"Standoff Dist: {dist:5.1f} m  (Target: 6.0 m)  | Closure Rate: {c_rate:+4.1f} m/s\n"
            f"Chaser Speed:  {c_spd:5.1f} m/s ({c_spd*3.6:4.1f} km/h) | Horiz Speed:  {c_spd_xy:4.1f} m/s\n"
            f"Chaser Alt:    {c_alt_curr:5.1f} m              | Target Alt:   {t_alt_curr:4.1f} m\n"
            f"Attitude:      P={np.rad2deg(pitch):+4.1f}°  R={np.rad2deg(roll):+4.1f}°  Y={np.rad2deg(yaw):+4.1f}°\n"
            f"Target Status: [ {status_str} ]\n"
            f"Pipeline:      60ms Latency | 30Hz YOLOv8n TensorRT | 50Hz Kalman Dead Reckoning"
        )
        status_col = "#2ed573" if (in_view and not k_coasting) else ("#ff7f50" if k_coasting else "#ff4757")
        ax_hud.text(0.02, 0.95, hud_text, transform=ax_hud.transAxes, color="#f1f2f6",
                    fontfamily="monospace", fontsize=8.0, va="top",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#0b0e14", edgecolor=status_col, lw=1.2))

        # ----------------------------------------------------------------------
        # 4. Real-Time 3D Altitude Profile Strip Chart
        # ----------------------------------------------------------------------
        ax_alt = fig.add_subplot(gs[2, 1], facecolor="#10141d")
        ax_alt.set_facecolor("#10141d")
        ax_alt.grid(True, linestyle="--", alpha=0.2, color="#2c3444")
        hist_s = max(0, step_idx - int(10.0 / dt_sim))
        t_hist = times[hist_s:step_idx + 1]
        ax_alt.plot(t_hist, t_alt[hist_s:step_idx + 1], color="#ff4757", lw=1.8, label="Target Alt")
        ax_alt.plot(t_hist, c_alt[hist_s:step_idx + 1], color="#00d2d3", lw=1.8, label="Chaser Alt")
        ax_alt.set_xlim(max(0, t_curr - 10.0), max(10.0, t_curr + 0.5))
        min_a = min(np.min(t_alt[:step_idx + 1]), np.min(c_alt[:step_idx + 1])) - 0.5
        max_a = max(np.max(t_alt[:step_idx + 1]), np.max(c_alt[:step_idx + 1])) + 0.5
        ax_alt.set_ylim(min_a, max_a)
        ax_alt.set_title("ALTITUDE PROFILE (m)", color="#dfe4ea", fontsize=8.5, fontweight="bold")
        ax_alt.tick_params(colors="#747d8c", labelsize=7)
        ax_alt.legend(loc="upper right", facecolor="#10141d", edgecolor="#2c3444", fontsize=6.5, labelcolor="#dfe4ea")

        # ----------------------------------------------------------------------
        # 5. Standoff Distance Strip Chart
        # ----------------------------------------------------------------------
        ax_dist = fig.add_subplot(gs[2, 2], facecolor="#10141d")
        ax_dist.set_facecolor("#10141d")
        ax_dist.grid(True, linestyle="--", alpha=0.2, color="#2c3444")
        ax_dist.plot(t_hist, dists[hist_s:step_idx + 1], color="#ffa502", lw=1.8)
        ax_dist.axhline(6.0, color="#2ed573", linestyle="--", lw=1.0, alpha=0.8, label="6m Goal")
        ax_dist.axhspan(5.0, 7.0, color="#2ed573", alpha=0.10, label="5-7m Window")
        ax_dist.set_xlim(max(0, t_curr - 10.0), max(10.0, t_curr + 0.5))
        ax_dist.set_ylim(0, max(15.0, np.max(dists[:step_idx + 1]) + 2.0))
        ax_dist.set_title("STANDOFF DISTANCE (m)", color="#dfe4ea", fontsize=8.5, fontweight="bold")
        ax_dist.tick_params(colors="#747d8c", labelsize=7)
        ax_dist.legend(loc="upper right", facecolor="#10141d", edgecolor="#2c3444", fontsize=6.5, labelcolor="#dfe4ea")

        # Render frame buffer into ffmpeg
        fig.canvas.draw()
        rgba_buffer = fig.canvas.buffer_rgba()
        pipe.stdin.write(rgba_buffer)

        # Clear subplots to avoid memory accumulation
        fig.clf()

        if (frame_i + 1) % max(1, num_frames // 10) == 0 or frame_i == num_frames - 1:
            pct = ((frame_i + 1) / num_frames) * 100.0
            print(f"  [{pct:5.1f}%] Rendered frame {frame_i+1}/{num_frames} (t = {t_curr:4.1f}s)")

    plt.close(fig)
    pipe.stdin.close()
    pipe.wait()

    file_size_mb = output_mp4_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] 3D MP4 Video generated: {output_mp4_path} ({file_size_mb:.2f} MB)")

    if output_gif_path:
        print(f"[GIF] Generating animated preview GIF: {output_gif_path}...")
        gif_cmd = [
            "ffmpeg", "-y",
            "-i", str(output_mp4_path),
            "-vf", "fps=10,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            str(output_gif_path)
        ]
        subprocess.run(gif_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[SUCCESS] GIF generated: {output_gif_path}")

    for art_dir in ARTIFACT_DIRS:
        try:
            if art_dir.exists():
                shutil.copy2(output_mp4_path, art_dir / output_mp4_path.name)
                if output_gif_path and output_gif_path.exists():
                    shutil.copy2(output_gif_path, art_dir / output_gif_path.name)
                print(f"[ARTIFACT] Copied 3D visualization to {art_dir}")
        except Exception as e:
            print(f"[ARTIFACT] Could not copy to {art_dir}: {e}")


def generate_3d_chase_movie(
    output_mp4_path: Path,
    output_gif_path: Optional[Path] = None,
    trajectory_name: str = "evasive",
    total_time_s: float = 25.0,
    fps: int = 25,
    seed: int = 42,
    enable_lateral_strafe: bool = True,
    controller_type: str = "pid",
    model_path: Optional[Union[str, Path]] = None,
):
    """
    Primary interface for batch scripts and test suites.
    Simulates the pursuit and renders the 3D spatial movie.
    """
    traj_fn, title = get_target_trajectory(trajectory_name, duration=total_time_s, seed=seed)
    ctrl_label = "Learned Recurrent PPO" if controller_type.lower() in ("recurrent_ppo", "rl", "recurrent") else "Classical Visual Servoing (PID)"
    title = f"{title} [{ctrl_label}]"

    sim_data = simulate_pursuit(
        trajectory_fn=traj_fn,
        duration=total_time_s,
        dt=0.02,
        enable_lateral_strafe=enable_lateral_strafe,
        controller_type=controller_type,
        model_path=model_path,
    )
    render_3d_movie_frames(
        sim_data=sim_data,
        output_mp4_path=output_mp4_path,
        output_gif_path=output_gif_path,
        trajectory_title=title,
        fps=fps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D spatial pursuit video of autonomous drone tracking.")
    parser.add_argument("--profile", "--trajectory", type=str, default="evasive",
                        help="Trajectory profile: cruising, evasive, aerobatic, hyper_evasive, sprint_break, figure8, slalom")
    parser.add_argument("--duration", type=float, default=20.0, help="Duration in seconds (default: 20.0)")
    parser.add_argument("--fps", type=int, default=25, help="Video framerate (default: 25)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stochastic trajectory")
    parser.add_argument("--controller", type=str, default="pid", choices=["pid", "recurrent_ppo", "rl"],
                        help="Controller policy: 'pid' or 'recurrent_ppo' / 'rl'")
    parser.add_argument("--model", type=str, default=None, help="Path to trained policy .zip")
    parser.add_argument("--out", type=str, default=None, help="Output MP4 file path")
    parser.add_argument("--gif", action="store_true", help="Also generate an animated GIF")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "outputs" / "videos" / "control"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.out:
        mp4_path = Path(args.out)
    else:
        mp4_path = out_dir / f"chase_3d_{args.profile}_{args.controller}_seed{args.seed}.mp4"

    gif_path = mp4_path.with_suffix(".gif") if args.gif else None

    generate_3d_chase_movie(
        output_mp4_path=mp4_path,
        output_gif_path=gif_path,
        trajectory_name=args.profile,
        total_time_s=args.duration,
        fps=args.fps,
        seed=args.seed,
        controller_type=args.controller,
        model_path=args.model,
    )
