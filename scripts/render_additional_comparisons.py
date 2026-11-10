#!/usr/bin/env python3
"""
Batch Video Renderer for 2 Additional High-Contrast Scenarios:
1. eval_445_hyper_evasive_crossing_right_s107565 (Hyper-Evasive Crossing: PID 0.0s dwell / 26.6% FOV vs. RRL 5.02s dwell / 97.1% FOV)
2. eval_354_evasive_head_on_s106018 (Evasive Head-On Merger: PID 0.98s dwell / 23.5% FOV vs. RRL 8.16s dwell / 98.7% FOV)
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

SCENARIOS_TO_RENDER = [
    {
        "id": "eval_248_evasive_crossing_right_s104216",
        "label": "Evasive Crossing Right",
        "pid_meta": "TUNED KINEMATIC PID (FAILS: 43.8% FOV, 0.44s Dwell, Dropped Track)",
        "rrl_meta": "RESIDUAL RL v9 (SUCCEEDS: 94.3% FOV, 7.40s Continuous Dwell)",
    },
    {
        "id": "eval_445_hyper_evasive_crossing_right_s107565",
        "label": "Hyper-Evasive Crossing Right",
        "pid_meta": "TUNED KINEMATIC PID (FAILS: 86.3% FOV, 0.38s Dwell Shortfall)",
        "rrl_meta": "RESIDUAL RL v9 (SUCCEEDS: 97.1% FOV, 5.02s Continuous Dwell)",
    }
]

def main():
    print("=" * 80)
    print("  RENDERING 2 ADDITIONAL PURSUIT COMPARISONS")
    print("=" * 80)

    with open(DATASET_PATH) as f:
        scenarios_map = {s["scenario_id"]: s for s in json.load(f)["scenarios"]}

    # Instantiate controllers once with matched canonical pre_attitude_dynamics gains
    pid = create_pid_controller(
        gains_path=PROJECT_ROOT / "control" / "weights" / "tuned_pid_gains_pre_attitude_dynamics.json",
        standoff_nominal=3.5,
        desired_bbox_size=110.0,
        max_decel=6.0,
    )

    rrl = ResidualVisualServoController(
        model_path=PROJECT_ROOT / "outputs" / "models" / "residual_ppo_v9_wide35" / "best_model" / "best_model.zip",
        pid_gains_path=PROJECT_ROOT / "control" / "weights" / "tuned_pid_gains_pre_attitude_dynamics.json",
        standoff_nominal=3.5,
        desired_bbox_size=110.0,
        w_nominal=0.0859,
    )

    for idx, sc_info in enumerate(SCENARIOS_TO_RENDER, 1):
        sid = sc_info["id"]
        label = sc_info["label"]
        print(f"\n========================================================")
        print(f"  SCENARIO [{idx}/2]: {sid} ({label})")
        print(f"========================================================")

        sc = PursuitScenario.from_dict(scenarios_map[sid])

        # 1. Render PID
        print(f">>> [{idx}.1] Rendering Tuned Kinematic PID...")
        pid_mp4 = OUT_DIR / f"pid_tuned_{sid}.mp4"
        render_pid_bright_video(
            scenario=sc,
            pid_controller=pid,
            output_path=pid_mp4,
            fps=25,
            controller_name="Tuned Kinematic PID (Baseline)",
            tau_yaw=0.150,
        )
        pid_artifact = ARTIFACT_DIR / f"pid_bright_{sid}.mp4"
        shutil.copy2(pid_mp4, pid_artifact)
        print(f"[PID ARTIFACT] Saved: {pid_artifact}")

        # 2. Render RRL
        print(f">>> [{idx}.2] Rendering Residual RL (v9)...")
        rrl_mp4 = OUT_DIR / f"rrl_v9_wide35_{sid}.mp4"
        render_pid_bright_video(
            scenario=sc,
            pid_controller=rrl,
            output_path=rrl_mp4,
            fps=25,
            controller_name="Residual RL (PPO v9 Wide35)",
            tau_yaw=0.150,
        )
        rrl_artifact = ARTIFACT_DIR / f"rrl_bright_{sid}.mp4"
        shutil.copy2(rrl_mp4, rrl_artifact)
        print(f"[RRL ARTIFACT] Saved: {rrl_artifact}")

        # 3. Side-by-side Dual Comparison
        print(f">>> [{idx}.3] Stacking Dual Side-by-Side Comparison Video...")
        comp_mp4 = OUT_DIR / f"comparison_pid_vs_rrl_{sid}.mp4"
        comp_artifact = ARTIFACT_DIR / f"pid_vs_rrl_{sid}_comparison.mp4"

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
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        shutil.copy2(comp_mp4, comp_artifact)
        print(f"[DUAL ARTIFACT] Saved: {comp_artifact}")

    print("\n" + "=" * 80)
    print("  ALL 2 ADDITIONAL PURSUIT COMPARISONS FULLY RENDERED!")
    print("=" * 80)

if __name__ == "__main__":
    main()

