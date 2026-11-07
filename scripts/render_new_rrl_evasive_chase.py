#!/usr/bin/env python3
"""
Renders broadcast-quality bright-mode video of the New RRL Model (residual_ppo_v9_wide35)
tracking an Evasive target using render_pid_chase_bright.
"""

import sys
import os
import json
import shutil
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.dataset_generator import PursuitScenario, PursuitDataset
from control.predictive_controller import ResidualVisualServoController
from control.rendering.render_pid_chase_bright import render_pid_bright_video

ARTIFACT_DIR = Path("/Users/philipkierkegaard/.gemini/antigravity/brain/3f6fb6d1-ed45-4315-9b5d-b3cca4e6bef7")
DEFAULT_DATASET = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "ci_10x600" / "ci_benchmark_600_fold_01.json"
DEFAULT_MODEL = PROJECT_ROOT / "outputs" / "models" / "residual_ppo_v9_wide35" / "best_model" / "best_model.zip"
DEFAULT_PID_GAINS = PROJECT_ROOT / "control" / "weights" / "tuned_pid_gains_pre_attitude_dynamics.json"
OUT_DIR = PROJECT_ROOT / "outputs" / "videos" / "control" / "rrl_v9_evasive"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Render New RRL v9 chase video on evasive target.")
    parser.add_argument("--scenario-id", type=str, default="eval_256_evasive_crossing_right_s104352",
                        help="Scenario ID to render")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET),
                        help="Path to dataset JSON")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL),
                        help="Path to trained RRL model")
    parser.add_argument("--pid-gains", type=str, default=str(DEFAULT_PID_GAINS),
                        help="Path to PID gains JSON")
    parser.add_argument("--fps", type=int, default=25,
                        help="Output video FPS")
    parser.add_argument("--tau-yaw", type=float, default=0.150,
                        help="Chaser yaw time constant (default: 0.150)")
    args = parser.parse_args()

    # Load dataset
    ds = PursuitDataset.load_json(args.dataset)
    scenarios_by_id = {s.scenario_id: s for s in ds.scenarios}

    if args.scenario_id not in scenarios_by_id:
        raise ValueError(f"Scenario {args.scenario_id} not found in {args.dataset}")

    sc = scenarios_by_id[args.scenario_id]

    print(f"\n=======================================================")
    print(f"  RENDERING NEW RRL (PPO v9 Wide35) EVASIVE CHASE")
    print(f"  Scenario:   {sc.scenario_id} ({sc.encounter_type}, {sc.profile})")
    print(f"  Model:      {args.model_path}")
    print(f"  PID Gains:  {args.pid_gains}")
    print(f"  tau_yaw:    {args.tau_yaw:.3f}s")
    print(f"=======================================================\n")

    controller = ResidualVisualServoController(
        model_path=args.model_path,
        pid_gains_path=args.pid_gains,
        residual_scales=[3.5, 3.5, 3.5, 35.0],
        standoff_nominal=3.5,
        desired_bbox_size=110.0,
    )

    out_file = OUT_DIR / f"rrl_v9_wide35_bright_{sc.scenario_id}.mp4"

    rendered_path = render_pid_bright_video(
        scenario=sc,
        pid_controller=controller,
        output_path=out_file,
        fps=args.fps,
        controller_name="Residual PPO v9 (Wide35)",
        tau_yaw=args.tau_yaw
    )

    print(f"\n[SUCCESS] Rendered video to: {rendered_path}")

    # Copy to conversation artifact directory
    if ARTIFACT_DIR.exists():
        dest = ARTIFACT_DIR / out_file.name
        shutil.copy2(rendered_path, dest)
        print(f"[ARTIFACT] Successfully copied video to artifact directory: {dest}")


if __name__ == "__main__":
    main()

