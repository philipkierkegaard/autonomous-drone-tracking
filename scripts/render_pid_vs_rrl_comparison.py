#!/usr/bin/env python3
"""
Render PID vs. Residual RL (v9_wide35) Comparison Videos
Demonstrates the decisive performance divergence on an evasive scenario
where RRL achieves sustained 2.0s interception lock while PID completely fails.
"""

import sys
import os
import json
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.dataset_generator import PursuitScenario
from control.rendering.render_pid_chase_bright import render_pid_bright_video
from benchmarks.control.benchmark_controller import create_pid_controller
from control.predictive_controller import ResidualVisualServoController

ARTIFACT_DIR = Path("/Users/philipkierkegaard/.gemini/antigravity/brain/3f6fb6d1-ed45-4315-9b5d-b3cca4e6bef7")
DATASET_PATH = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "ci_10x600" / "ci_benchmark_600_fold_01.json"
OUT_DIR = PROJECT_ROOT / "outputs" / "videos" / "control" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SID = "eval_203_evasive_tail_chase_s103451"

def main():
    print("=" * 80)
    print(f"  RENDERING PURSUIT COMPARISON: PID vs. RRL")
    print(f"  Scenario: {SID}")
    print("=" * 80)

    # 1. Load scenario
    with open(DATASET_PATH) as f:
        scenarios = {s["scenario_id"]: s for s in json.load(f)["scenarios"]}

    if SID not in scenarios:
        raise ValueError(f"Scenario {SID} not found in {DATASET_PATH}")

    sc = PursuitScenario.from_dict(scenarios[SID])

    # 2. Render Tuned Kinematic PID
    print("\n>>> [1/3] Rendering Tuned Kinematic PID Video...")
    pid = create_pid_controller(
        gains_path=PROJECT_ROOT / "control" / "weights" / "tuned_pid_gains_pre_attitude_dynamics.json",
        standoff_nominal=3.5,
        desired_bbox_size=110.0,
        max_decel=6.0,
    )
    pid_mp4 = OUT_DIR / f"pid_tuned_{SID}.mp4"
    render_pid_bright_video(
        scenario=sc,
        pid_controller=pid,
        output_path=pid_mp4,
        fps=25,
        controller_name="Tuned Kinematic PID (Baseline)",
        tau_yaw=0.150,
    )
    pid_artifact = ARTIFACT_DIR / f"pid_bright_{SID}.mp4"
    shutil.copy2(pid_mp4, pid_artifact)
    print(f"[PID ARTIFACT] Copied to: {pid_artifact}")

    # 3. Render Residual RL (v9_wide35)
    print("\n>>> [2/3] Rendering Residual RL (v9_wide35) Video...")
    rrl = ResidualVisualServoController(
        model_path=PROJECT_ROOT / "outputs" / "models" / "residual_ppo_v9_wide35" / "best_model" / "best_model.zip",
        pid_gains_path=PROJECT_ROOT / "control" / "weights" / "tuned_pid_gains_pre_attitude_dynamics.json",
        standoff_nominal=3.5,
        desired_bbox_size=110.0,
        w_nominal=0.0859,
    )
    rrl_mp4 = OUT_DIR / f"rrl_v9_wide35_{SID}.mp4"
    render_pid_bright_video(
        scenario=sc,
        pid_controller=rrl,
        output_path=rrl_mp4,
        fps=25,
        controller_name="Residual RL (PPO v9 Wide35)",
        tau_yaw=0.150,
    )
    rrl_artifact = ARTIFACT_DIR / f"rrl_bright_{SID}.mp4"
    shutil.copy2(rrl_mp4, rrl_artifact)
    print(f"[RRL ARTIFACT] Copied to: {rrl_artifact}")

    # 4. Generate Synchronized Dual Split-Screen Comparison Video
    print("\n>>> [3/3] Generating Synchronized Dual Comparison Video via ffmpeg...")
    comparison_mp4 = OUT_DIR / f"comparison_pid_vs_rrl_{SID}.mp4"
    comparison_artifact = ARTIFACT_DIR / f"pid_vs_rrl_{SID}_comparison.mp4"

    # Scale both to 960x534 and stack side-by-side -> 1920x534
    cmd = [
        "ffmpeg", "-y",
        "-i", str(pid_mp4),
        "-i", str(rrl_mp4),
        "-filter_complex",
        "[0:v]scale=960:534,drawtext=text='TUNED KINEMATIC PID (FAILS: 10.8% FOV, 0.0s Dwell)':x=20:y=20:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=5[left];"
        "[1:v]scale=960:534,drawtext=text='RESIDUAL RL v9 (SUCCEEDS: 97.8% FOV, 5.98s Dwell)':x=20:y=20:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=5[right];"
        "[left][right]hstack=inputs=2[v]",
        "-map", "[v]",
        "-c:v", "libx264",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(comparison_mp4)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        shutil.copy2(comparison_mp4, comparison_artifact)
        print(f"[DUAL COMPARISON] Successfully created: {comparison_artifact}")
    except Exception as e:
        print(f"Warning: ffmpeg drawtext filter failed, falling back to clean hstack: {e}")
        cmd_fallback = [
            "ffmpeg", "-y",
            "-i", str(pid_mp4),
            "-i", str(rrl_mp4),
            "-filter_complex",
            "[0:v]scale=960:534[left];[1:v]scale=960:534[right];[left][right]hstack=inputs=2[v]",
            "-map", "[v]",
            "-c:v", "libx264",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(comparison_mp4)
        ]
        subprocess.run(cmd_fallback, check=True)
        shutil.copy2(comparison_mp4, comparison_artifact)
        print(f"[DUAL COMPARISON FALLBACK] Created: {comparison_artifact}")

    print("\n" + "=" * 80)
    print("  ALL COMPARISON VIDEOS SUCCESSFULLY RENDERED & ARTIFACTS CREATED")
    print(f"  PID Individual Video:  {pid_artifact}")
    print(f"  RRL Individual Video:  {rrl_artifact}")
    print(f"  Dual Comparison Video: {comparison_artifact}")
    print("=" * 80)

if __name__ == "__main__":
    main()
