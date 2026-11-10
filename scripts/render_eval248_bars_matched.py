#!/usr/bin/env python3
"""
Dedicated renderer for Scenario eval_248_evasive_crossing_right_s104216:
- Uses 100% matched canonical gains: control/weights/tuned_pid_gains_pre_attitude_dynamics.json
- Renders Fly-By-Wire Actuation Bar Deck (Surge, Sway, Yaw Rate with PID + RL breakdown)
- Generates:
    1. PID Individual Full-Resolution MP4 (1440x800)
    2. RRL Individual Full-Resolution MP4 (1440x800)
    3. Dual Synchronized Side-by-Side MP4 (1920x534)
    4. Diagnostic Frame Stills at t = 2.4s, 4.0s, 7.0s, 12.0s
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

import cv2
from control.dataset_generator import PursuitScenario
from control.rendering.render_pid_chase_bright import render_pid_bright_video
from benchmarks.control.benchmark_controller import create_pid_controller
from control.predictive_controller import ResidualVisualServoController

ARTIFACT_DIR = Path("/Users/philipkierkegaard/.gemini/antigravity/brain/3f6fb6d1-ed45-4315-9b5d-b3cca4e6bef7")
DATASET_PATH = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "ci_10x600" / "ci_benchmark_600_fold_01.json"
OUT_DIR = PROJECT_ROOT / "outputs" / "videos" / "control" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SID = "eval_248_evasive_crossing_right_s104216"

def main():
    print("=" * 80)
    print(f"  RENDERING EVAL_248 WITH MATCHED PRE-ATTITUDE DYNAMICS GAINS & ACTUATION BARS")
    print(f"  Scenario: {SID}")
    print("=" * 80)

    # 1. Load scenario definition
    with open(DATASET_PATH) as f:
        scenarios = {s["scenario_id"]: s for s in json.load(f)["scenarios"]}

    if SID not in scenarios:
        raise ValueError(f"Scenario {SID} not found in {DATASET_PATH}")

    sc = PursuitScenario.from_dict(scenarios[SID])

    canonical_gains = PROJECT_ROOT / "control" / "weights" / "tuned_pid_gains_pre_attitude_dynamics.json"
    print(f"[GAINS] Using canonical matched gains: {canonical_gains}")

    # 2. Render Tuned Kinematic PID (Baseline)
    print("\n>>> [1/3] Rendering Tuned Kinematic PID with Matched Gains & Actuation Bars...")
    pid = create_pid_controller(
        gains_path=canonical_gains,
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
    pid_artifact = ARTIFACT_DIR / f"pid_eval_248_matched_bars.mp4"
    shutil.copy2(pid_mp4, pid_artifact)
    print(f"[PID ARTIFACT] Saved to: {pid_artifact}")

    # 3. Render Residual RL (v9 Wide35)
    print("\n>>> [2/3] Rendering Residual RL (PPO v9 Wide35) with Matched Gains & Actuation Bars...")
    rrl = ResidualVisualServoController(
        model_path=PROJECT_ROOT / "outputs" / "models" / "residual_ppo_v9_wide35" / "best_model" / "best_model.zip",
        pid_gains_path=canonical_gains,
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
    rrl_artifact = ARTIFACT_DIR / f"rrl_eval_248_matched_bars.mp4"
    shutil.copy2(rrl_mp4, rrl_artifact)
    print(f"[RRL ARTIFACT] Saved to: {rrl_artifact}")

    # 4. Generate Synchronized Dual Comparison Video
    print("\n>>> [3/3] Generating Synchronized Dual Comparison Video via FFmpeg...")
    comp_mp4 = OUT_DIR / f"comparison_pid_vs_rrl_{SID}.mp4"
    comp_artifact = ARTIFACT_DIR / f"pid_vs_rrl_eval_248_matched_gains_with_bars.mp4"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(pid_mp4),
        "-i", str(rrl_mp4),
        "-filter_complex",
        "[0:v]scale=960:534[left];[1:v]scale=960:534[right];[left][right]hstack=inputs=2[v]",
        "-map", "[v]",
        "-c:v", "libx264",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(comp_mp4)
    ]
    subprocess.run(cmd, check=True)
    shutil.copy2(comp_mp4, comp_artifact)
    # Also overwrite the comparison artifact filename so any existing link works
    shutil.copy2(comp_mp4, ARTIFACT_DIR / f"pid_vs_rrl_{SID}_comparison.mp4")
    print(f"[DUAL ARTIFACT] Saved to: {comp_artifact}")

    # 5. Extract verification stills
    print("\n>>> Extracting verification frames (2.4s, 4.0s, 7.0s, 12.0s)...")
    cap = cv2.VideoCapture(str(comp_mp4))
    for t_sec in [2.4, 4.0, 7.0, 12.0]:
        frame_idx = int(t_sec * 25.0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            still_path = ARTIFACT_DIR / f"eval_248_matched_bars_t{int(t_sec*10):03d}.png"
            cv2.imwrite(str(still_path), frame)
            print(f"  Frame at t={t_sec:.1f}s saved: {still_path}")
    cap.release()

    print("\n" + "=" * 80)
    print("  EVAL_248 RENDERING COMPLETE!")
    print(f"  Dual Video: {comp_artifact}")
    print(f"  PID Video:  {pid_artifact}")
    print(f"  RRL Video:  {rrl_artifact}")
    print("=" * 80)

if __name__ == "__main__":
    main()

