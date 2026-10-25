#!/usr/bin/env python3
"""
True-to-Benchmark Dual Renderer for 4 Hyper-Evasive Success Encounters WITH Ego-Shifts.
Directly executes evaluate_single_run with return_history=True and enable_ego_compensation=True
(empirical YOLO noise, 5% frame drops, and SO(3) gyroscopic ego-compensated Kalman tracking).
Strictly NO GIFs.
"""

import sys
import os
import json
import time
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
    print(f"\n[BENCH-RUN] Simulating benchmark perception & control (WITH EGO-SHIFTS) for: {scenario.scenario_id}")
    
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
    
    # 2. Evaluate PID under full benchmark conditions
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

    print(f"[RENDER] Generating {num_frames} video frames...")
    for f_idx, s_idx in enumerate(frame_indices):
        t_now = times[s_idx]
        tgt_p = target_pos_all[s_idx]
        p_pos = pid_pos_all[s_idx]
        p_yaw = pid_yaw_all[s_idx]
        p_tel = pid_telem_all[s_idx]
        p_cmd = pid_cmd_all[s_idx]
        p_vel = pid_vel_all[s_idx]
        p_dw = pid_dwell_all[s_idx]

        r_pos = rl_pos_all[s_idx]
        r_yaw = rl_yaw_all[s_idx]
        r_tel = rl_telem_all[s_idx]
        r_cmd = rl_cmd_all[s_idx]
        r_vel = rl_vel_all[s_idx]
        r_dw = rl_dwell_all[s_idx]

        ax_map.clear()
        ax_hud.clear()

        ax_map.set_facecolor("#fafafa")
        ax_map.grid(True, linestyle="--", alpha=0.5, color="#c0c0c0")

        # Full planned paths (dotted background)
        ax_map.plot(target_pos_all[:, 1], target_pos_all[:, 0], color="#d62728", linestyle=":", linewidth=1.0, alpha=0.35)
        ax_map.plot(pid_pos_all[:, 1], pid_pos_all[:, 0], color="#1f77b4", linestyle=":", linewidth=1.0, alpha=0.35)
        ax_map.plot(rl_pos_all[:, 1], rl_pos_all[:, 0], color="#2ca02c", linestyle=":", linewidth=1.0, alpha=0.35)

        # Elapsed trails
        ax_map.plot(target_pos_all[:s_idx+1, 1], target_pos_all[:s_idx+1, 0], color="#d62728", linestyle="-", linewidth=2.2, label="6G Hyper-Evasive Target")
        ax_map.plot(pid_pos_all[:s_idx+1, 1], pid_pos_all[:s_idx+1, 0], color="#1f77b4", linestyle="--", linewidth=2.0, label="Kinematic PID (Ego-Compensated)")
        ax_map.plot(rl_pos_all[:s_idx+1, 1], rl_pos_all[:s_idx+1, 0], color="#2ca02c", linestyle="-", linewidth=2.2, label="Learned RL Chaser (Cloned v2)")

        # Markers
        ax_map.scatter(target_pos_all[0, 1], target_pos_all[0, 0], color="#d62728", s=50, marker="o", label="Target Start")
        ax_map.scatter(scenario.initial_chaser_pos[1], scenario.initial_chaser_pos[0], color="black", s=50, marker="^", label="Chaser Origin")

        d_p = np.linalg.norm(tgt_p - p_pos)
        d_r = np.linalg.norm(tgt_p - r_pos)
        in_b_p = (2.5 <= d_p <= 4.5) and (p_tel["status"] in ("LOCKED", "COASTING"))
        in_b_r = (2.5 <= d_r <= 4.5) and (r_tel["status"] in ("LOCKED", "COASTING"))

        # Sightlines
        color_p = "#2ca02c" if in_b_p else ("#1f77b4" if p_tel["status"] in ("LOCKED", "COASTING") else "#d95f02")
        color_r = "#2ca02c" if in_b_r else ("#238b45" if r_tel["status"] in ("LOCKED", "COASTING") else "#d95f02")
        ax_map.plot([p_pos[1], tgt_p[1]], [p_pos[0], tgt_p[0]], color=color_p, linestyle="--", linewidth=1.5, alpha=0.75)
        ax_map.plot([r_pos[1], tgt_p[1]], [r_pos[0], tgt_p[0]], color=color_r, linestyle="--", linewidth=1.8, alpha=0.75)

        # Drone markers & pointers
        arrow_len = 2.0
        ax_map.scatter(p_pos[1], p_pos[0], color="#1f77b4", s=75, zorder=6)
        ax_map.arrow(p_pos[1], p_pos[0], arrow_len * np.sin(p_yaw), arrow_len * np.cos(p_yaw),
                     head_width=0.7, head_length=0.7, fc="#1f77b4", ec="#1f77b4", zorder=7)

        ax_map.scatter(r_pos[1], r_pos[0], color="#2ca02c", s=75, zorder=6)
        ax_map.arrow(r_pos[1], r_pos[0], arrow_len * np.sin(r_yaw), arrow_len * np.cos(r_yaw),
                     head_width=0.7, head_length=0.7, fc="#2ca02c", ec="#2ca02c", zorder=7)

        ax_map.scatter(tgt_p[1], tgt_p[0], color="#d62728", s=85, marker="o", edgecolors="black", linewidths=1.2, zorder=6)

        # 1.0s Catch Badges
        catch_done_r = (catch_t_r is not None and t_now >= catch_t_r + 1.0)
        catch_done_p = (catch_t_p is not None and t_now >= catch_t_p + 1.0)

        if catch_done_r:
            ax_map.text(
                0.98, 0.04, f"★ Learned RL 1.0s LOCKED (t={catch_t_r+1.0:.2f}s)",
                transform=ax_map.transAxes, fontsize=8.8, fontweight="bold",
                color="#1b7837", ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", edgecolor="#2ca02c", linewidth=1.4)
            )
        if catch_done_p:
            ax_map.text(
                0.98, 0.11, f"★ Kinematic PID 1.0s LOCKED (t={catch_t_p+1.0:.2f}s)",
                transform=ax_map.transAxes, fontsize=8.8, fontweight="bold",
                color="#08519c", ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#eff3ff", edgecolor="#3182bd", linewidth=1.4)
            )

        ax_map.set_xlim(e_center - half_span, e_center + half_span)
        ax_map.set_ylim(n_center - half_span, n_center + half_span)
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.set_xlabel("East Position Y (m)", fontsize=10)
        ax_map.set_ylabel("North Position X (m)", fontsize=10)
        ax_map.set_title(f"{scenario.scenario_id} | t = {t_now:.1f}s", fontsize=10.5, fontweight="bold", pad=8)
        ax_map.legend(loc="upper left", fontsize=8.0, framealpha=0.92)

        # HUD Dual Table
        ax_hud.axis("off")
        ax_hud.set_facecolor("#ffffff")

        spd_p = np.linalg.norm(p_vel)
        spd_r = np.linalg.norm(r_vel)

        status_p = f"LOCKED({catch_t_p+1.0:.1f}s)" if catch_done_p else (f"DWELL({p_dw:.1f}s)" if in_b_p else p_tel["status"])
        status_r = f"LOCKED({catch_t_r+1.0:.1f}s)" if catch_done_r else (f"DWELL({r_dw:.1f}s)" if in_b_r else r_tel["status"])

        hud_text = (
            f"DUAL PURSUIT (EGO-COMPENSATED KALMAN)\n"
            f"────────────────────────────────────────────\n"
            f"Time:       {t_now:5.2f} s\n"
            f"Profile:    HYPER_EVASIVE (6G Maneuvers)\n"
            f"Encounter:  {scenario.encounter_type}\n"
            f"Sensor:     YOLO Noise + Drops + SO(3) Ego-Shift\n"
            f"\n"
            f"METRIC            PID (Blue)   RL (Green)\n"
            f"────────────────────────────────────────────\n"
            f"Perception:       {p_tel['status']:<10}   {r_tel['status']:<10}\n"
            f"Distance:         {d_p:5.2f} m       {d_r:5.2f} m\n"
            f"In Basket(2.5-4.5m):{'YES' if in_b_p else 'NO':<10}   {'YES' if in_b_r else 'NO':<10}\n"
            f"1.0s Strict Dwell:{p_dw:4.2f} / 1.0s    {r_dw:4.2f} / 1.0s\n"
            f"Lock Status:      {status_p:<10}   {status_r:<10}\n"
            f"\n"
            f"COMMANDS & VELOCITIES\n"
            f"────────────────────────────────────────────\n"
            f"Ground Speed:     {spd_p:5.2f} m/s     {spd_r:5.2f} m/s\n"
            f"cmd_vx (Forward): {p_cmd[0]:+5.2f} m/s     {r_cmd[0]:+5.2f} m/s\n"
            f"cmd_vy (Lateral): {p_cmd[1]:+5.2f} m/s     {r_cmd[1]:+5.2f} m/s\n"
            f"cmd_yaw (Rate):   {p_cmd[3]:+5.1f} °/s     {r_cmd[3]:+5.1f} °/s\n"
            f"Altitude (AGL):   {-p_pos[2]:5.2f} m       {-r_pos[2]:5.2f} m\n"
            f"Benchmark Result: {'FAILED (0.0s)' if not res_pid['success_1s'] else 'SUCCESS':<10}   {'SUCCESS (1s)' if res_rl['success_1s'] else 'FAILED':<10}"
        )

        ax_hud.text(
            0.04, 0.95, hud_text,
            transform=ax_hud.transAxes,
            fontsize=8.2,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="square,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced4da", linewidth=1.0)
        )

        fig.canvas.draw()
        rgba_buffer = fig.canvas.buffer_rgba()
        pipe.stdin.write(rgba_buffer)

    pipe.stdin.close()
    pipe.wait()
    plt.close(fig)

    print(f"[SUCCESS] Saved video: {output_path}")

    # Mirror to brain directory
    brain_dir = Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/5967d853-7a7e-4cdd-8792-4a77e6d704a2")
    if brain_dir.exists():
        import shutil
        shutil.copy(output_path, brain_dir / output_path.name)

    return str(output_path)


def main():
    suite_file = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "evasive_200_benchmark_suite.json"
    with open(suite_file) as f:
        scenarios = {s["scenario_id"]: s for s in json.load(f)["scenarios"]}

    selected_ids = [
        "bench200_108_hyper_evasive_tail_chase_s40107",
        "bench200_133_hyper_evasive_crossing_right_s40132",
        "bench200_147_hyper_evasive_crossing_left_s40146",
        "bench200_179_hyper_evasive_head_on_s40178",
    ]

    out_dir = PROJECT_ROOT / "outputs" / "videos" / "control" / "success" / "expertv2" / "hyper_evasive_egoshift_4"
    out_dir.mkdir(parents=True, exist_ok=True)

    rl_model_path = PROJECT_ROOT / "control" / "weights" / "cloned_expert_v2" / "actor_weights.pt"
    rl_controller = RecurrentVisualServoController(model_path=str(rl_model_path), w_nominal=0.0859)
    pid_controller = create_pid_controller(
        gains_path=PROJECT_ROOT / "control" / "weights" / "tuned_pid_gains.json",
        standoff_nominal=3.5,
        desired_bbox_size=110.0,
    )

    rendered = []
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

    # Concatenate the 4 hyper-evasive videos
    concat_list = out_dir / "concat_list.txt"
    with open(concat_list, "w") as f:
        for p in rendered:
            f.write(f"file '{Path(p).resolve()}'\n")

    comp_path = out_dir / "four_hyper_evasive_egoshift_chases.mp4"
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(comp_path)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"\n[COMPLETE] 4 Hyper-Evasive Compilation saved to: {comp_path}")

    brain_dir = Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/5967d853-7a7e-4cdd-8792-4a77e6d704a2")
    if brain_dir.exists():
        import shutil
        shutil.copy(comp_path, brain_dir / comp_path.name)


if __name__ == "__main__":
    main()
