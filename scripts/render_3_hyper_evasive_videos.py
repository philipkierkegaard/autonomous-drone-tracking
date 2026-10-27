#!/usr/bin/env python3
"""
Renders 3 distinct successful Hyper-Evasive pursuit videos using the tuned PID controller
under bright mode broadcast visualization with fresh seeds.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.dataset_generator import PursuitScenario
from control.rendering.render_pid_chase_bright import render_pid_bright_video
from benchmarks.control.benchmark_controller import create_pid_controller
import json

TARGET_SCENARIOS = [
    {
        "id": "bench200_176_hyper_evasive_head_on_s40175",
        "name": "Head-On Inward Charge",
        "description": "Target charges inward head-on with sharp jinking and evasive banking (Seed 40175)."
    },
    {
        "id": "bench200_120_hyper_evasive_tail_chase_s40119",
        "name": "Violent Tail Chase",
        "description": "High-speed tail chase with rapid zig-zag jinks and sharp velocity pulses (Seed 40119)."
    },
    {
        "id": "bench200_131_hyper_evasive_crossing_right_s40130",
        "name": "Lateral Crossing Break",
        "description": "Aggressive rightward crossing sprint across the visual field of view (Seed 40130)."
    }
]

DATASET_PATH = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "benchmark_300_suite.json"
OUT_DIR = PROJECT_ROOT / "outputs" / "videos" / "control" / "success" / "pid_tuned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    with open(DATASET_PATH) as f:
        scenarios = {s["scenario_id"]: s for s in json.load(f)["scenarios"]}

    pid_controller = create_pid_controller(
        gains_path=PROJECT_ROOT / "control" / "weights" / "tuned_pid_gains.json",
        standoff_nominal=3.5,
        desired_bbox_size=110.0,
    )

    rendered_files = []
    for item in TARGET_SCENARIOS:
        sid = item["id"]
        print(f"\n{'='*80}")
        print(f"RENDERING HYPER-EVASIVE VIDEO: {item['name']} ({sid})")
        print(f"{'='*80}")

        sc = PursuitScenario.from_dict(scenarios[sid])
        out_path = OUT_DIR / f"pid_tuned_bright_{sid}.mp4"

        res_path = render_pid_bright_video(
            scenario=sc,
            pid_controller=pid_controller,
            output_path=out_path,
            fps=25
        )
        rendered_files.append((item, res_path))

    print("\n" + "="*80)
    print("ALL 3 HYPER-EVASIVE VIDEOS RENDERED SUCCESSFULLY!")
    print("="*80)
    for item, path in rendered_files:
        print(f"- {item['name']}: {path}")

if __name__ == "__main__":
    main()
