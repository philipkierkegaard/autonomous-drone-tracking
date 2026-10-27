#!/usr/bin/env python3
"""
Renders a broadcast-quality video of the Residual RL Champion (50k checkpoint)
tracking a Hyper-Evasive target (bench200_109_hyper_evasive_tail_chase_s40108).
"""

import sys
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.dataset_generator import PursuitScenario
from control.rendering.render_pid_chase_bright import render_pid_bright_video
from control.predictive_controller import ResidualVisualServoController

ARTIFACT_DIR = Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/7dc8da15-984b-4499-8ca6-5e32efd45198")
DATASET_PATH = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "benchmark_300_suite.json"
OUT_DIR = PROJECT_ROOT / "outputs" / "videos" / "control" / "success" / "residual_champion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SID = "bench200_109_hyper_evasive_tail_chase_s40108"

def main():
    with open(DATASET_PATH) as f:
        scenarios = {s["scenario_id"]: s for s in json.load(f)["scenarios"]}

    sc = PursuitScenario.from_dict(scenarios[SID])

    controller = ResidualVisualServoController(
        model_path=PROJECT_ROOT / "outputs" / "models" / "residual_ppo_v1" / "champion_50k" / "actor_weights.pt",
        pid_gains_path=PROJECT_ROOT / "control" / "weights" / "tuned_pid_gains.json",
        w_nominal=0.0859
    )

    out_file = OUT_DIR / f"residual_champion_bright_{SID}.mp4"
    rendered_path = render_pid_bright_video(
        scenario=sc,
        pid_controller=controller,
        output_path=out_file,
        fps=25
    )
    print(f"\n[SUCCESS] Rendered Residual Champion video to: {rendered_path}")

    # Copy to artifact directory
    artifact_copy = ARTIFACT_DIR / f"residual_champion_bright_{SID}.mp4"
    shutil.copy2(rendered_path, artifact_copy)
    print(f"[ARTIFACT] Copied video to artifact directory: {artifact_copy}")

if __name__ == "__main__":
    main()
