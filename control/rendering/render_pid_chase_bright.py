#!/usr/bin/env python3
"""
Single-Agent Bright-Theme Video Renderer: Tuned Kinematic PID with Optical Looming Feedforward.
Format:
- Left: High-Resolution 2D Tactical Top-Down Trajectory Radar Map
- Right Top: 16:9 Widescreen On-Board FPV Camera (Synthetic Perception, Kalman State, Target BBox)
- Right Bottom: Live Flight Telemetry & Optical Looming Dashboard (Standoff Basket Bar, Dwell Progress)

Theme: Clean Bright / Light Mode (Vivid Crimson Target, Electric Royal Blue Chaser, Emerald Green Lock).
Strictly MP4 format only (no GIFs).
"""

import sys
import os
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.dataset_generator import PursuitScenario
from benchmarks.control.benchmark_controller import create_pid_controller, evaluate_single_run

ARTIFACT_DIR = Path("/Users/philipkierkegaard/.gemini/antigravity/brain/3f6fb6d1-ed45-4315-9b5d-b3cca4e6bef7")


def render_pid_bright_video(
    scenario: PursuitScenario,
    pid_controller: Any,
    output_path: Path,
    fps: int = 25,
    controller_name: Optional[str] = None,
    tau_yaw: Optional[float] = 0.150
) -> str:
    """Renders a broadcast-quality bright-mode pursuit video of the controller."""
    if controller_name is None:
        if hasattr(pid_controller, "residual_scales") or "Residual" in type(pid_controller).__name__:
            controller_name = "Residual RL (PPO v9 Wide35)"
        else:
            controller_name = "Tuned Kinematic PID"

    print(f"\n[BENCH-RUN] Simulating benchmark perception & control for: {scenario.scenario_id}")
    print(f"            Controller: {controller_name} (tau_yaw = {tau_yaw:.3f}s)")
    
    pid_controller.reset()
    res = evaluate_single_run(
        scenario=scenario,
        controller=pid_controller,
        controller_type="residual" if ("Residual" in type(pid_controller).__name__) else "pid",
        noise_mode="empirical",
        frame_drop_rate=0.05,
        standoff_nominal=3.5,
        use_kalman=True,
        return_history=True,
        enable_ego_compensation=True,
        tau_yaw=tau_yaw,
    )

    h = res["history"]
    s1 = res["success_1s"]
    s3 = res["success_3s"]
    dwell_max = res["max_continuous_dwell_s"]
    fov_lock = res["fov_lock_pct"]
    min_dist = res["min_distance"]
    print(f"            PID 1s Success: {s1} | 3s Success: {s3} | Max Dwell: {dwell_max:.2f}s | FOV: {fov_lock:.1f}% | Min Dist: {min_dist:.2f}m")

    times = h["times"]
    target_pos_all = h["target_pos"]
    chaser_pos_all = h["chaser_pos"]
    chaser_yaw_all = h["chaser_yaw"]
    chaser_vel_all = h["chaser_vel"]
    chaser_cmd_all = h["chaser_cmd"]
    chaser_pid_cmd_all = h.get("chaser_pid_cmd", chaser_cmd_all)
    chaser_residual_cmd_all = h.get("chaser_residual_cmd", np.zeros_like(chaser_cmd_all))
    chaser_telem_all = h["chaser_telem"]
    dwell_all = h["dwell"]
    catch_t = h["catch_t_1s"]

    dt_sim = 0.02
    total_steps = len(times)
    sim_steps_per_frame = max(1, int(round((1.0 / fps) / dt_sim)))
    frame_indices = np.arange(0, total_steps, sim_steps_per_frame)
    num_frames = len(frame_indices)

    fig_w, fig_h = 14.4, 8.0
    dpi = 100
    width_px = int(fig_w * dpi)
    height_px = int(fig_h * dpi)
    if width_px % 2 != 0:
        width_px += 1
    if height_px % 2 != 0:
        height_px += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width_px}x{height_px}",
        "-pix_fmt", "rgba",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "20",
        str(output_path)
    ]

    pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#FFFFFF")

    # Global trajectory bounds for consistent spatial framing (East = X-axis, North = Y-axis)
    all_east = np.concatenate([target_pos_all[:, 1], chaser_pos_all[:, 1]])
    all_north = np.concatenate([target_pos_all[:, 0], chaser_pos_all[:, 0]])
    margin = 6.0
    xlim = (np.min(all_east) - margin, np.max(all_east) + margin)
    ylim = (np.min(all_north) - margin, np.max(all_north) + margin)

    print(f"[RENDER] Generating {num_frames} bright frames ({times[-1]:.1f}s at {fps} FPS) via FFmpeg pipe...")
    t0 = time.time()

    enc_str = scenario.encounter_type.replace("_", " ").title()

    for fi, idx in enumerate(frame_indices):
        fig.clf()
        t_curr = times[idx]
        tgt_p = target_pos_all[idx]
        c_p = chaser_pos_all[idx]
        c_yaw = chaser_yaw_all[idx]
        c_vel = chaser_vel_all[idx]
        c_cmd = chaser_cmd_all[idx]
        c_pid_cmd = chaser_pid_cmd_all[idx]
        c_res_cmd = chaser_residual_cmd_all[idx]
        telem = chaser_telem_all[idx]

        dist = float(np.linalg.norm(c_p[:2] - tgt_p[:2]))
        dwell = dwell_all[idx]
        succ_now = (catch_t is not None and t_curr >= catch_t)
        in_basket = (2.5 <= dist <= 4.5)

        gs = fig.add_gridspec(
            2, 2,
            width_ratios=[1.25, 1.0],
            height_ratios=[1.15, 0.85],
            wspace=0.18, hspace=0.28,
            left=0.05, right=0.96, top=0.91, bottom=0.07
        )

        # -------------------------------------------------------------
        # 1. 2D Tactical Top-Down Trajectory Map (Left Column)
        # -------------------------------------------------------------
        ax_map = fig.add_subplot(gs[:, 0])
        ax_map.set_facecolor("#FAFAFA")
        ax_map.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1")

        # Full intended background paths (light dotted): East = X-axis, North = Y-axis
        ax_map.plot(target_pos_all[:, 1], target_pos_all[:, 0], color="#DC2626", ls=":", lw=1.2, alpha=0.35)
        ax_map.plot(chaser_pos_all[:, 1], chaser_pos_all[:, 0], color="#2563EB", ls=":", lw=1.2, alpha=0.35)

        # History trails up to current time
        ax_map.plot(target_pos_all[:idx+1, 1], target_pos_all[:idx+1, 0], color="#DC2626", lw=2.6, label=f"Target ({scenario.profile.replace('_', ' ').title()})", alpha=0.95)
        ax_map.plot(chaser_pos_all[:idx+1, 1], chaser_pos_all[:idx+1, 0], color="#2563EB", lw=2.8, label=f"Chaser ({controller_name})", alpha=0.95)

        # Origin points
        ax_map.scatter([target_pos_all[0, 1]], [target_pos_all[0, 0]], color="#DC2626", marker="o", s=45, alpha=0.7, label="Target Start")
        ax_map.scatter([chaser_pos_all[0, 1]], [chaser_pos_all[0, 0]], color="#1E293B", marker="^", s=50, alpha=0.7, label="Drone Origin")

        # Sightline
        sight_color = "#10B981" if in_basket else "#F59E0B"
        ax_map.plot([c_p[1], tgt_p[1]], [c_p[0], tgt_p[0]], color=sight_color, ls="--", lw=2.0, alpha=0.85, label=f"LOS Sightline ({dist:.1f}m)")

        # Current positions
        ax_map.scatter([tgt_p[1]], [tgt_p[0]], color="#DC2626", s=120, edgecolors="#FFFFFF", linewidths=2.2, zorder=6)
        ax_map.scatter([c_p[1]], [c_p[0]], color="#2563EB", s=130, edgecolors="#FFFFFF", linewidths=2.2, zorder=6)

        # Heading vector arrow: dx = sin(yaw) (East), dy = cos(yaw) (North)
        arrow_len = 2.6
        ax_map.arrow(c_p[1], c_p[0], arrow_len * np.sin(c_yaw), arrow_len * np.cos(c_yaw),
                     head_width=0.75, head_length=0.75, fc="#1D4ED8", ec="#1D4ED8", zorder=7)

        # Standoff capture basket circle [2.5m - 4.5m]
        basket_color = "#10B981" if in_basket else "#93C5FD"
        basket_circle = patches.Circle((c_p[1], c_p[0]), 3.5, color=basket_color, fill=True, alpha=0.10 if in_basket else 0.04, zorder=4)
        basket_edge = patches.Circle((c_p[1], c_p[0]), 3.5, color=basket_color, fill=False, ls="--", lw=2.0 if in_basket else 1.2, alpha=0.9, zorder=5, label="Standoff Basket [2.5-4.5m]")
        ax_map.add_patch(basket_circle)
        ax_map.add_patch(basket_edge)

        ax_map.set_xlim(xlim)
        ax_map.set_ylim(ylim)
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.set_xlabel("East Position Y (m)", color="#475569", fontsize=8.5, labelpad=2)
        ax_map.set_ylabel("North Position X (m)", color="#475569", fontsize=8.5, labelpad=2)
        ax_map.tick_params(colors="#475569", labelsize=8.0)
        for spine in ax_map.spines.values():
            spine.set_color("#CBD5E1")

        status_txt = "INTERCEPTION LOCK ACHIEVED (>=1.0s)" if succ_now else (f"BASKET DWELL: {dwell:.2f}s / 3.0s" if (in_basket and telem.get("status") != "SEARCHING") else "PURSUING / SPEED MATCHING")
        color_status = "#059669" if succ_now else ("#D97706" if dwell > 0 else "#2563EB")
        ax_map.set_title(f"TOP-DOWN TACTICAL RADAR — {status_txt}", color=color_status, fontsize=11, fontweight="bold", pad=8)
        ax_map.legend(loc="upper left", facecolor="#FFFFFF", edgecolor="#E2E8F0", labelcolor="#1E293B", fontsize=8.5, framealpha=0.95)

        # -------------------------------------------------------------
        # 2. FPV Camera (Right Top)
        # -------------------------------------------------------------
        ax_cam = fig.add_subplot(gs[0, 1])
        ax_cam.set_facecolor("#F8FAFC")
        ax_cam.set_xlim(-1.0, 1.0)
        ax_cam.set_ylim(-0.5625, 0.5625)

        # Subtle crosshair
        ax_cam.axhline(0, color="#CBD5E1", ls=":", lw=0.8)
        ax_cam.axvline(0, color="#CBD5E1", ls=":", lw=0.8)
        ax_cam.plot([-0.07, 0.07], [0, 0], color="#64748B", lw=1.4)
        ax_cam.plot([0, 0], [-0.07, 0.07], color="#64748B", lw=1.4)

        # Camera HUD indicators
        ax_cam.text(-0.95, 0.48, "● REC  30 FPS", color="#DC2626", fontsize=7.5, fontweight="bold", va="top")
        ax_cam.text(-0.95, 0.40, "MODE: KALMAN (NO EGO-COMP)", color="#64748B", fontsize=7.0, va="top")
        ax_cam.text(0.95, 0.48, f"ALT: {-c_p[2]:.1f}m", color="#0F172A", fontsize=7.5, fontweight="bold", ha="right", va="top")
        ax_cam.text(0.95, 0.40, f"HDG: {np.rad2deg(c_yaw)%360:.0f}°", color="#64748B", fontsize=7.0, ha="right", va="top")

        status = telem.get("status", "SEARCHING")
        if status in ("LOCKED", "COASTING"):
            ex = float(np.clip(telem.get("error_x", 0.0), -1.0, 1.0))
            ey = float(np.clip(telem.get("error_y", 0.0), -1.0, 1.0))
            # Camera frame maps normalized [-1, 1] horizontally to [-1, 1] and vertically to [-0.5625, 0.5625]
            # Since camera ey is positive downwards, in matplotlib's positive-up system: y_center = -ey * 0.5625
            cam_y = -ey * 0.5625
            bw = (telem.get("bbox_w", 110.0) / 640.0) * 2.0
            bh = (telem.get("bbox_h", 70.0) / 360.0) * 1.125

            is_coasting = (status == "COASTING")
            if is_coasting:
                # Kalman prediction / dead-reckoning during frame dropout or inter-camera intervals
                rect_col = "#8B5CF6"  # Purple for Kalman prediction
                line_style = "--"
                tag_str = f"TGT {dist:.1f}m [KALMAN PRED]"
                tag_bg = "#F3E8FF"
            else:
                # Confirmed active YOLO measurement
                rect_col = "#059669" if in_basket else "#D97706"
                line_style = "-"
                tag_str = f"TGT {dist:.1f}m [MEAS LOCKED]"
                tag_bg = "#ECFDF5" if in_basket else "#FFFBEB"

            bbox_patch = patches.Rectangle(
                (ex - bw/2, cam_y - bh/2), bw, bh,
                fill=False, edgecolor=rect_col, lw=2.6, linestyle=line_style
            )
            ax_cam.add_patch(bbox_patch)
            tag_y = max(-0.50, min(0.48, cam_y - bh/2 - 0.07))
            ax_cam.text(ex, tag_y, tag_str, color=rect_col, fontsize=7.8, ha="center", weight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor=tag_bg, edgecolor=rect_col, lw=1.2))
        else:
            ax_cam.text(0, 0, "SEARCHING / TARGET LOST", color="#DC2626", fontsize=11, ha="center", weight="bold")

        ax_cam.set_xticks([])
        ax_cam.set_yticks([])
        cam_border_col = "#10B981" if in_basket else "#CBD5E1"
        for spine in ax_cam.spines.values():
            spine.set_color(cam_border_col)
            spine.set_linewidth(2.4 if in_basket else 1.2)
        ax_cam.set_title(f"FPV ON-BOARD CAMERA (640×360) | Distance={dist:.2f}m | Speed={np.linalg.norm(c_vel[:2]):.1f}m/s", color="#0F172A", fontsize=9.5, fontweight="bold", pad=6)

        # -------------------------------------------------------------
        # 3. Telemetry & Optical Looming Feedforward Dashboard (Right Bottom)
        # -------------------------------------------------------------
        ax_hud = fig.add_subplot(gs[1, 1])
        ax_hud.set_facecolor("#FFFFFF")
        ax_hud.set_xlim(0, 1)
        ax_hud.set_ylim(0, 1)
        ax_hud.set_xticks([])
        ax_hud.set_yticks([])
        for spine in ax_hud.spines.values():
            spine.set_color("#E2E8F0")

        ax_hud.set_title("FLIGHT TELEMETRY & OPTICAL LOOMING CONTROLLER", fontsize=9.5, fontweight="bold", color="#0F172A", pad=6)

        speed_drone = np.linalg.norm(c_vel[:2])

        # 1. Standoff Basket Gauge Bar
        ax_hud.text(0.04, 0.90, "Standoff Distance:", fontsize=7.5, color="#64748B", fontweight="bold", va="center")
        bar_x, bar_y, bar_w, bar_h = 0.30, 0.86, 0.65, 0.07
        ax_hud.add_patch(patches.Rectangle((bar_x, bar_y), bar_w, bar_h, facecolor="#F1F5F9", edgecolor="#CBD5E1", lw=1))
        # Target Basket Zone [2.5 to 4.5m]
        bx_start = bar_x + bar_w * (2.5 / 10.0)
        bx_width = bar_w * (2.0 / 10.0)
        ax_hud.add_patch(patches.Rectangle((bx_start, bar_y), bx_width, bar_h, facecolor="#D1FAE5", edgecolor="#10B981", lw=1.2))
        ax_hud.text(bx_start + bx_width/2, bar_y + bar_h/2, "BASKET 2.5-4.5m", color="#065F46", fontsize=6.5, ha="center", va="center", fontweight="bold")
        # Current distance indicator marker
        clamped_d = min(max(dist, 0.0), 10.0)
        marker_x = bar_x + bar_w * (clamped_d / 10.0)
        marker_col = "#059669" if in_basket else "#D97706"
        ax_hud.plot([marker_x, marker_x], [bar_y - 0.015, bar_y + bar_h + 0.015], color=marker_col, lw=2.4, zorder=5)
        ax_hud.text(marker_x, bar_y + bar_h + 0.035, f"{dist:.2f}m", color=marker_col, fontsize=7.2, ha="center", fontweight="bold")

        # 2. Continuous Dwell Gauge Bar
        ax_hud.text(0.04, 0.75, "Interception Dwell:", fontsize=7.5, color="#64748B", fontweight="bold", va="center")
        dwell_bar_y = 0.71
        ax_hud.add_patch(patches.Rectangle((bar_x, dwell_bar_y), bar_w, bar_h, facecolor="#F1F5F9", edgecolor="#CBD5E1", lw=1))
        dwell_fill_w = min(dwell / 3.0, 1.0) * bar_w
        if dwell_fill_w > 0:
            ax_hud.add_patch(patches.Rectangle((bar_x, dwell_bar_y), dwell_fill_w, bar_h, facecolor="#10B981", edgecolor="none"))
        ax_hud.text(bar_x + bar_w/2, dwell_bar_y + bar_h/2, f"{dwell:.2f}s / 3.00s", color="#0F172A", fontsize=7.2, ha="center", va="center", fontweight="bold")

        # Divider line
        ax_hud.axhline(0.63, color="#E2E8F0", lw=0.9)

        # 3. Command Decomposition Deck: PID Command + RL Residual Correction
        ax_hud.text(0.04, 0.57, "FLY-BY-WIRE ACTUATION: PID + RL CORRECTIONS", fontsize=7.5, color="#0F172A", fontweight="bold")
        # Mini-Legend
        ax_hud.add_patch(patches.Rectangle((0.68, 0.55), 0.035, 0.032, facecolor="#2563EB", edgecolor="none"))
        ax_hud.text(0.725, 0.565, "PID", fontsize=6.5, color="#475569", va="center", fontweight="bold")
        ax_hud.add_patch(patches.Rectangle((0.78, 0.55), 0.035, 0.032, facecolor="#10B981", edgecolor="none"))
        ax_hud.text(0.825, 0.565, "+RL", fontsize=6.5, color="#065F46", va="center", fontweight="bold")
        ax_hud.add_patch(patches.Rectangle((0.88, 0.55), 0.035, 0.032, facecolor="#F59E0B", edgecolor="none"))
        ax_hud.text(0.925, 0.565, "-RL", fontsize=6.5, color="#92400E", va="center", fontweight="bold")

        # Channels: (Label, PID value, Residual value, Total value, Min limit, Max limit, Unit, y_pos)
        p_vx, p_vy, p_vz, p_yaw = c_pid_cmd
        r_vx, r_vy, r_vz, r_yaw = c_res_cmd
        tot_vx, tot_vy, tot_vz, tot_yaw = c_cmd

        channels = [
            ("Surge (Vx)", p_vx, r_vx, tot_vx, 0.0, 8.0, "m/s", 0.42),
            ("Sway (Vy)",  p_vy, r_vy, tot_vy, -3.0, 3.0, "m/s", 0.28),
            ("Yaw Rate",   p_yaw, r_yaw, tot_yaw, -100.0, 100.0, "°/s", 0.14)
        ]

        for label, p_val, r_val, tot_val, min_v, max_v, unit, y_pos in channels:
            ch_bx, ch_by, ch_bw, ch_bh = 0.27, y_pos, 0.45, 0.085
            ax_hud.text(0.04, y_pos + ch_bh/2, label + ":", fontsize=7.2, color="#475569", fontweight="bold", va="center")
            ax_hud.add_patch(patches.Rectangle((ch_bx, ch_by), ch_bw, ch_bh, facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.8))

            if min_v < 0:
                # Bi-directional centered bar
                zero_x = ch_bx + ch_bw * (-min_v / (max_v - min_v))
                ax_hud.plot([zero_x, zero_x], [ch_by, ch_by + ch_bh], color="#94A3B8", ls="--", lw=0.8)
                px = ch_bx + ch_bw * ((np.clip(p_val, min_v, max_v) - min_v) / (max_v - min_v))
                if px >= zero_x:
                    ax_hud.add_patch(patches.Rectangle((zero_x, ch_by), px - zero_x, ch_bh, facecolor="#2563EB", edgecolor="none", alpha=0.85))
                else:
                    ax_hud.add_patch(patches.Rectangle((px, ch_by), zero_x - px, ch_bh, facecolor="#2563EB", edgecolor="none", alpha=0.85))

                cx = ch_bx + ch_bw * ((np.clip(tot_val, min_v, max_v) - min_v) / (max_v - min_v))
                res_col = "#10B981" if (r_val >= 0) else "#F59E0B"
                if cx >= px:
                    ax_hud.add_patch(patches.Rectangle((px, ch_by), cx - px, ch_bh, facecolor=res_col, edgecolor="none", alpha=0.95))
                else:
                    ax_hud.add_patch(patches.Rectangle((cx, ch_by), px - cx, ch_bh, facecolor=res_col, edgecolor="none", alpha=0.95))
                ax_hud.plot([cx, cx], [ch_by - 0.01, ch_by + ch_bh + 0.01], color="#0F172A", lw=2.0)
            else:
                zero_x = ch_bx
                px = ch_bx + ch_bw * (np.clip(p_val, min_v, max_v) / max_v)
                ax_hud.add_patch(patches.Rectangle((zero_x, ch_by), px - zero_x, ch_bh, facecolor="#2563EB", edgecolor="none", alpha=0.85))
                cx = ch_bx + ch_bw * (np.clip(tot_val, min_v, max_v) / max_v)
                res_col = "#10B981" if (r_val >= 0) else "#F59E0B"
                if cx >= px:
                    ax_hud.add_patch(patches.Rectangle((px, ch_by), cx - px, ch_bh, facecolor=res_col, edgecolor="none", alpha=0.95))
                else:
                    ax_hud.add_patch(patches.Rectangle((cx, ch_by), px - cx, ch_bh, facecolor=res_col, edgecolor="none", alpha=0.95))
                ax_hud.plot([cx, cx], [ch_by - 0.01, ch_by + ch_bh + 0.01], color="#0F172A", lw=2.0)

            # Text readout
            res_str = f"{r_val:+.2f}" if abs(r_val) >= 0.005 else "0.0"
            sign_p = "+" if (p_val >= 0 and min_v < 0) else ""
            ax_hud.text(0.74, y_pos + ch_bh/2, f"PID:{sign_p}{p_val:.1f} RL:{res_str} -> {tot_val:.1f}{unit}",
                        fontsize=6.8, color="#0F172A", va="center", fontweight="bold")

        # Bottom info line
        ax_hud.text(0.04, 0.04, f"Drone Speed: {speed_drone:.1f} m/s  |  FOV Lock: {fov_lock:.1f}%  |  Dwell Achieved: {dwell_max:.2f}s",
                    fontsize=7.2, color="#64748B", fontweight="bold", va="center")

        # Supertitle
        catch_str = f"INTERCEPTION SUCCESS (Max Dwell: {dwell_max:.2f}s | FOV Lock: {fov_lock:.1f}%)" if s1 else "PURSUING"
        fig.suptitle(
            f"{scenario.scenario_id} | Time: {t_curr:4.1f}s / {times[-1]:.1f}s | {controller_name} | {catch_str}",
            fontsize=9.0, fontweight="bold", color="#0F172A", x=0.5, ha="center", y=0.965
        )

        fig.canvas.draw()
        raw_rgba = fig.canvas.buffer_rgba()
        pipe.stdin.write(raw_rgba)

    pipe.stdin.close()
    pipe.wait()
    plt.close(fig)

    elapsed = time.time() - t0
    print(f"[COMPLETE] Rendered MP4 video in {elapsed:.1f}s -> {output_path}")

    # Copy to artifacts directory if available
    if ARTIFACT_DIR.exists():
        dest = ARTIFACT_DIR / output_path.name
        if output_path.resolve() != dest.resolve():
            shutil.copy(output_path, dest)
            print(f"[ARTIFACT] Copied video to artifact directory: {dest}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Render bright-mode pursuit video of tuned PID controller.")
    parser.add_argument(
        "--scenario-id",
        type=str,
        default="bench200_087_evasive_oblique_away_s40086",
        help="Scenario ID to render from evasive_200 suite."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "evasive_200_benchmark_suite.json"),
        help="Path to scenarios dataset JSON."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output MP4 path."
    )
    parser.add_argument(
        "--pid-gains",
        type=str,
        default=str(PROJECT_ROOT / "control" / "weights" / "tuned_pid_gains_slow_tau770.json"),
        help="Path to PID gains JSON (default: tuned_pid_gains_slow_tau770.json)."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Output video FPS."
    )
    args = parser.parse_args()

    with open(args.dataset) as f:
        scenarios = {s["scenario_id"]: s for s in json.load(f)["scenarios"]}

    if args.scenario_id not in scenarios:
        raise ValueError(f"Scenario '{args.scenario_id}' not found in {args.dataset}")

    sc = PursuitScenario.from_dict(scenarios[args.scenario_id])

    pid_controller = create_pid_controller(
        gains_path=args.pid_gains,
        standoff_nominal=3.5,
        desired_bbox_size=110.0,
    )

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = PROJECT_ROOT / "outputs" / "videos" / "control" / "success" / "pid_tuned"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pid_tuned_bright_{args.scenario_id}.mp4"

    render_pid_bright_video(
        scenario=sc,
        pid_controller=pid_controller,
        output_path=out_path,
        fps=args.fps
    )


if __name__ == "__main__":
    main()
