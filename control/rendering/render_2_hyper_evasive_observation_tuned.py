#!/usr/bin/env python3
"""
Dual Video Renderer: Learned Controller (RL Cloned Expert v2) vs. Observation-Tuned Kinematic PID.
Target Profile: Hyper-Evasive Targets (RL Succeeds with >= 1.0s Dwell, Tuned PID Fails).
Scenarios:
  1. bench200_133_hyper_evasive_crossing_right_s40132 (Crossing Right)
  2. bench200_174_hyper_evasive_head_on_s40173 (Head-On Pass)

Directly executes evaluate_single_run with return_history=True and enable_ego_compensation=True.
Strictly MP4 format only (NO GIFs).
"""

import sys
import os
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.dataset_generator import PursuitScenario
from control.predictive_controller import RecurrentVisualServoController
from benchmarks.control.benchmark_controller import create_pid_controller, evaluate_single_run


def render_scenario_from_benchmark_history(
    scenario: PursuitScenario,
    rl_controller: Any,
    pid_controller: Any,
    output_path: Path,
    fps: int = 25
) -> str:
    """Runs the exact benchmark evaluation with history and renders the dual pursuit video."""
    print(f"\n[BENCH-RUN] Simulating benchmark perception & control for: {scenario.scenario_id}")
    
    # 1. Evaluate RL under full benchmark conditions
    rl_controller.reset()
    res_rl = evaluate_single_run(
        scenario=scenario,
        controller=rl_controller,
        controller_type="rl",
        noise_mode="empirical",
        frame_drop_rate=0.05,
        standoff_nominal=3.5,
        use_kalman=True,
        return_history=True,
        enable_ego_compensation=True
    )
    
    # 2. Evaluate Tuned PID under full benchmark conditions
    pid_controller.reset()
    res_pid = evaluate_single_run(
        scenario=scenario,
        controller=pid_controller,
        controller_type="pid",
        noise_mode="empirical",
        frame_drop_rate=0.05,
        standoff_nominal=3.5,
        use_kalman=True,
        return_history=True,
        enable_ego_compensation=True
    )

    h_rl = res_rl["history"]
    h_pid = res_pid["history"]

    print(f"            RL  1s Success: {res_rl['success_1s']} | Max Dwell: {res_rl['max_continuous_dwell_s']:.2f}s | FOV: {res_rl['fov_lock_pct']:.1f}%")
    print(f"            PID 1s Success: {res_pid['success_1s']} | Max Dwell: {res_pid['max_continuous_dwell_s']:.2f}s | FOV: {res_pid['fov_lock_pct']:.1f}%")

    times = h_rl["times"]
    target_pos_all = h_rl["target_pos"]
    rl_pos_all = h_rl["chaser_pos"]
    rl_yaw_all = h_rl["chaser_yaw"]
    rl_vel_all = h_rl["chaser_vel"]
    rl_cmd_all = h_rl["chaser_cmd"]
    rl_telem_all = h_rl["chaser_telem"]
    rl_dwell_all = h_rl["dwell"]
    catch_t_r = h_rl["catch_t_1s"]

    pid_pos_all = h_pid["chaser_pos"]
    pid_yaw_all = h_pid["chaser_yaw"]
    pid_vel_all = h_pid["chaser_vel"]
    pid_cmd_all = h_pid["chaser_cmd"]
    pid_telem_all = h_pid["chaser_telem"]
    pid_dwell_all = h_pid["dwell"]
    catch_t_p = h_pid["catch_t_1s"]

    dt_sim = 0.02
    total_steps = len(times)
    sim_steps_per_frame = max(1, int(round((1.0 / fps) / dt_sim)))
    frame_indices = np.arange(0, total_steps, sim_steps_per_frame)
    num_frames = len(frame_indices)

    fig_w, fig_h = 13.6, 7.6
    dpi = 100
    width_px = int(fig_w * dpi)
    height_px = int(fig_h * dpi)
    if width_px % 2 != 0:
        width_px += 1
    if height_px % 2 != 0:
        height_px += 1

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
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#0B0F19")

    # Global trajectory bounds for consistent spatial context
    all_x = np.concatenate([target_pos_all[:, 0], rl_pos_all[:, 0], pid_pos_all[:, 0]])
    all_y = np.concatenate([target_pos_all[:, 1], rl_pos_all[:, 1], pid_pos_all[:, 1]])
    margin = 7.0
    xlim = (np.min(all_x) - margin, np.max(all_x) + margin)
    ylim = (np.min(all_y) - margin, np.max(all_y) + margin)

    print(f"[RENDER] Generating {num_frames} dual frames via FFmpeg pipe...")
    t0 = time.time()

    for fi, idx in enumerate(frame_indices):
        fig.clf()
        t_curr = times[idx]
        tgt_p = target_pos_all[idx]

        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1.3, 1], wspace=0.18, hspace=0.28, left=0.06, right=0.96, top=0.90, bottom=0.08)

        # -------------------------------------------------------------
        # Left Column: Learned Controller (RL Cloned Expert v2)
        # -------------------------------------------------------------
        # 1. Top-Down Trajectory
        ax_map_rl = fig.add_subplot(gs[0, 0])
        ax_map_rl.set_facecolor("#111827")
        ax_map_rl.plot(target_pos_all[:idx+1, 0], target_pos_all[:idx+1, 1], color="#EF4444", lw=2.2, label="Target (Hyper-Evasive)", alpha=0.9)
        ax_map_rl.plot(rl_pos_all[:idx+1, 0], rl_pos_all[:idx+1, 1], color="#10B981", lw=2.5, label="RL Chaser (Cloned v2)")
        
        # Current positions & orientations
        c_p = rl_pos_all[idx]
        c_yaw = rl_yaw_all[idx]
        ax_map_rl.scatter([tgt_p[0]], [tgt_p[1]], color="#EF4444", s=90, edgecolors="white", zorder=5)
        ax_map_rl.scatter([c_p[0]], [c_p[1]], color="#10B981", s=110, edgecolors="white", zorder=5)
        ax_map_rl.plot([c_p[0], c_p[0] + 2.5 * np.cos(c_yaw)], [c_p[1], c_p[1] + 2.5 * np.sin(c_yaw)], color="#34D399", lw=2.5)

        # 3.5m Standoff capture basket
        basket_circle = plt.Circle((c_p[0], c_p[1]), 3.5, color="#10B981", fill=False, ls="--", lw=1.2, alpha=0.5)
        ax_map_rl.add_patch(basket_circle)

        dist_r = float(np.linalg.norm(c_p[:2] - tgt_p[:2]))
        dwell_r = rl_dwell_all[idx]
        rl_succ_now = (catch_t_r is not None and t_curr >= catch_t_r)

        ax_map_rl.set_xlim(xlim)
        ax_map_rl.set_ylim(ylim)
        ax_map_rl.set_aspect("equal", adjustable="box")
        ax_map_rl.tick_params(colors="#9CA3AF", labelsize=8)
        for spine in ax_map_rl.spines.values():
            spine.set_color("#374151")
        status_txt = "INTERCEPTION LOCK ACHIEVED (>=1.0s)" if rl_succ_now else (f"BASKET DWELL: {dwell_r:.2f}s / 1.0s" if (2.5 <= dist_r <= 4.5 and rl_telem_all[idx].get("status") != "SEARCHING") else "PURSUING / POSITIONING")
        color_status = "#34D399" if rl_succ_now else ("#FBBF24" if dwell_r > 0 else "#9CA3AF")
        ax_map_rl.set_title(f"LEARNED RL (Cloned v2) — {status_txt}", color=color_status, fontsize=11, fontweight="bold", pad=8)
        ax_map_rl.legend(loc="upper left", facecolor="#1F2937", edgecolor="#374151", labelcolor="#F3F4F6", fontsize=8)

        # 2. Camera View (FPV Synthetic Perception)
        ax_cam_rl = fig.add_subplot(gs[1, 0])
        ax_cam_rl.set_facecolor("#030712")
        ax_cam_rl.set_xlim(-1.0, 1.0)
        ax_cam_rl.set_ylim(-1.0, 1.0)
        ax_cam_rl.axhline(0, color="#374151", ls=":", lw=0.8)
        ax_cam_rl.axvline(0, color="#374151", ls=":", lw=0.8)
        
        # Crosshair optical center
        ax_cam_rl.plot([ -0.1, 0.1 ], [0, 0], color="#6B7280", lw=1.2)
        ax_cam_rl.plot([0, 0], [-0.1, 0.1], color="#6B7280", lw=1.2)

        telem_r = rl_telem_all[idx]
        if telem_r.get("status") in ("LOCKED", "COASTING"):
            ex = telem_r.get("error_x", 0.0)
            ey = telem_r.get("error_y", 0.0)
            bw = telem_r.get("bbox_w", 110.0) / 640.0
            bh = telem_r.get("bbox_h", 70.0) / 360.0
            rect_color = "#10B981" if (2.5 <= dist_r <= 4.5) else "#F59E0B"
            bbox_patch = plt.Rectangle((ex - bw/2, ey - bh/2), bw, bh, fill=False, edgecolor=rect_color, lw=2.2)
            ax_cam_rl.add_patch(bbox_patch)
            ax_cam_rl.text(ex, ey - bh/2 - 0.08, f"TGT {dist_r:.1f}m", color=rect_color, fontsize=8, ha="center", weight="bold")
        else:
            ax_cam_rl.text(0, 0, "SEARCHING / TARGET LOST", color="#EF4444", fontsize=10, ha="center", weight="bold")

        ax_cam_rl.set_xticks([])
        ax_cam_rl.set_yticks([])
        for spine in ax_cam_rl.spines.values():
            spine.set_color("#10B981" if (2.5 <= dist_r <= 4.5) else "#374151")
            spine.set_linewidth(1.8 if (2.5 <= dist_r <= 4.5) else 1.0)
        ax_cam_rl.set_title(f"FPV Camera: Distance={dist_r:.2f}m | Speed={np.linalg.norm(rl_vel_all[idx][:2]):.1f}m/s", color="#E5E7EB", fontsize=9)

        # -------------------------------------------------------------
        # Right Column: Observation-Tuned Kinematic PID
        # -------------------------------------------------------------
        # 1. Top-Down Trajectory
        ax_map_pid = fig.add_subplot(gs[0, 1])
        ax_map_pid.set_facecolor("#111827")
        ax_map_pid.plot(target_pos_all[:idx+1, 0], target_pos_all[:idx+1, 1], color="#EF4444", lw=2.2, label="Target (Hyper-Evasive)", alpha=0.9)
        ax_map_pid.plot(pid_pos_all[:idx+1, 0], pid_pos_all[:idx+1, 1], color="#3B82F6", lw=2.5, label="Observation-Tuned PID")
        
        # Current positions & orientations
        p_p = pid_pos_all[idx]
        p_yaw = pid_yaw_all[idx]
        ax_map_pid.scatter([tgt_p[0]], [tgt_p[1]], color="#EF4444", s=90, edgecolors="white", zorder=5)
        ax_map_pid.scatter([p_p[0]], [p_p[1]], color="#3B82F6", s=110, edgecolors="white", zorder=5)
        ax_map_pid.plot([p_p[0], p_p[0] + 2.5 * np.cos(p_yaw)], [p_p[1], p_p[1] + 2.5 * np.sin(p_yaw)], color="#60A5FA", lw=2.5)

        basket_circle_p = plt.Circle((p_p[0], p_p[1]), 3.5, color="#3B82F6", fill=False, ls="--", lw=1.2, alpha=0.5)
        ax_map_pid.add_patch(basket_circle_p)

        dist_p = float(np.linalg.norm(p_p[:2] - tgt_p[:2]))
        dwell_p = pid_dwell_all[idx]
        pid_succ_now = (catch_t_p is not None and t_curr >= catch_t_p)

        ax_map_pid.set_xlim(xlim)
        ax_map_pid.set_ylim(ylim)
        ax_map_pid.set_aspect("equal", adjustable="box")
        ax_map_pid.tick_params(colors="#9CA3AF", labelsize=8)
        for spine in ax_map_pid.spines.values():
            spine.set_color("#374151")
        status_txt_p = "INTERCEPTION LOCK ACHIEVED (>=1.0s)" if pid_succ_now else (f"BASKET DWELL: {dwell_p:.2f}s / 1.0s" if (2.5 <= dist_p <= 4.5 and pid_telem_all[idx].get("status") != "SEARCHING") else "PURSUING / LAG")
        color_status_p = "#34D399" if pid_succ_now else ("#FBBF24" if dwell_p > 0 else "#9CA3AF")
        ax_map_pid.set_title(f"OBSERVATION-TUNED PID — {status_txt_p}", color=color_status_p, fontsize=11, fontweight="bold", pad=8)
        ax_map_pid.legend(loc="upper left", facecolor="#1F2937", edgecolor="#374151", labelcolor="#F3F4F6", fontsize=8)

        # 2. Camera View (FPV Synthetic Perception)
        ax_cam_pid = fig.add_subplot(gs[1, 1])
        ax_cam_pid.set_facecolor("#030712")
        ax_cam_pid.set_xlim(-1.0, 1.0)
        ax_cam_pid.set_ylim(-1.0, 1.0)
        ax_cam_pid.axhline(0, color="#374151", ls=":", lw=0.8)
        ax_cam_pid.axvline(0, color="#374151", ls=":", lw=0.8)
        ax_cam_pid.plot([ -0.1, 0.1 ], [0, 0], color="#6B7280", lw=1.2)
        ax_cam_pid.plot([0, 0], [-0.1, 0.1], color="#6B7280", lw=1.2)

        telem_p = pid_telem_all[idx]
        if telem_p.get("status") in ("LOCKED", "COASTING"):
            ex_p = telem_p.get("error_x", 0.0)
            ey_p = telem_p.get("error_y", 0.0)
            bw_p = telem_p.get("bbox_w", 110.0) / 640.0
            bh_p = telem_p.get("bbox_h", 70.0) / 360.0
            rect_color_p = "#3B82F6" if (2.5 <= dist_p <= 4.5) else "#F59E0B"
            bbox_patch_p = plt.Rectangle((ex_p - bw_p/2, ey_p - bh_p/2), bw_p, bh_p, fill=False, edgecolor=rect_color_p, lw=2.2)
            ax_cam_pid.add_patch(bbox_patch_p)
            ax_cam_pid.text(ex_p, ey_p - bh_p/2 - 0.08, f"TGT {dist_p:.1f}m", color=rect_color_p, fontsize=8, ha="center", weight="bold")
        else:
            ax_cam_pid.text(0, 0, "SEARCHING / TARGET LOST", color="#EF4444", fontsize=10, ha="center", weight="bold")

        ax_cam_pid.set_xticks([])
        ax_cam_pid.set_yticks([])
        for spine in ax_cam_pid.spines.values():
            spine.set_color("#3B82F6" if (2.5 <= dist_p <= 4.5) else "#374151")
            spine.set_linewidth(1.8 if (2.5 <= dist_p <= 4.5) else 1.0)
        ax_cam_pid.set_title(f"FPV Camera: Distance={dist_p:.2f}m | Speed={np.linalg.norm(pid_vel_all[idx][:2]):.1f}m/s", color="#E5E7EB", fontsize=9)

        # Super Title with Scenario Info & Dwell Metric
        rl_dwell_val = res_rl['max_continuous_dwell_s']
        pid_dwell_val = res_pid['max_continuous_dwell_s']
        rl_res_str = f"SUCCESS (Dwell {rl_dwell_val:.2f}s)" if res_rl['success_1s'] else "FAIL"
        pid_res_str = "SUCCESS" if res_pid['success_1s'] else f"FAIL (Max Dwell {pid_dwell_val:.2f}s)"

        fig.suptitle(
            f"Scenario: {scenario.scenario_id} | Time: {t_curr:4.1f}s / {times[-1]:.1f}s | RL Catch: {rl_res_str} vs PID: {pid_res_str}",
            fontsize=12, fontweight="bold", color="#F9FAFB", y=0.97
        )

        fig.canvas.draw()
        raw_rgba = fig.canvas.buffer_rgba()
        pipe.stdin.write(raw_rgba)

    pipe.stdin.close()
    pipe.wait()
    plt.close(fig)

    elapsed = time.time() - t0
    print(f"[COMPLETE] Rendered MP4 video in {elapsed:.1f}s -> {output_path}")
    return str(output_path)


def main():
    suite_file = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "evasive_200_benchmark_suite.json"
    with open(suite_file) as f:
        scenarios = {s["scenario_id"]: s for s in json.load(f)["scenarios"]}

    # Two distinct hyper-evasive encounters where RL succeeds and observation-tuned PID fails
    selected_ids = [
        "bench200_133_hyper_evasive_crossing_right_s40132",
        "bench200_174_hyper_evasive_head_on_s40173",
    ]

    out_dir = PROJECT_ROOT / "outputs" / "videos" / "control" / "success" / "expertv2" / "observation_tuned_hyper_evasive_2"
    out_dir.mkdir(parents=True, exist_ok=True)

    rl_model_path = PROJECT_ROOT / "control" / "weights" / "cloned_expert_v2" / "actor_weights.pt"
    rl_controller = RecurrentVisualServoController(model_path=str(rl_model_path), w_nominal=0.0859)
    pid_controller = create_pid_controller(
        gains_path=PROJECT_ROOT / "control" / "weights" / "tuned_pid_gains.json",
        standoff_nominal=3.5,
        desired_bbox_size=110.0,
    )

    rendered = []
    brain_dir = Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/5967d853-7a7e-4cdd-8792-4a77e6d704a2")

    for sc_id in selected_ids:
        sc = PursuitScenario.from_dict(scenarios[sc_id])
        out_path = out_dir / f"dual_{sc_id}.mp4"
        p = render_scenario_from_benchmark_history(
            scenario=sc,
            rl_controller=rl_controller,
            pid_controller=pid_controller,
            output_path=out_path
        )
        rendered.append(p)
        if brain_dir.exists():
            shutil.copy(out_path, brain_dir / out_path.name)

    # Concatenate into single side-by-side comparison video
    concat_list = out_dir / "concat_list.txt"
    with open(concat_list, "w") as f:
        for p in rendered:
            f.write(f"file '{Path(p).resolve()}'\n")

    comp_path = out_dir / "two_hyper_evasive_rl_success_pid_fail.mp4"
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(comp_path)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"\n[COMPLETE] 2 Hyper-Evasive Compilation saved to: {comp_path}")

    if brain_dir.exists():
        shutil.copy(comp_path, brain_dir / comp_path.name)


if __name__ == "__main__":
    main()
