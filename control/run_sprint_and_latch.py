#!/usr/bin/env python3
"""
Sprint & Latch RL Calibration Run.
Trains 1,000,000 timesteps from cloned expert weights (control/weights/cloned_expert_v1/actor_weights.pt)
with the Sprint & Latch reward formulation:
1. Active differential closing potential (sprint_weight = 3.5) driving forward throttle when far.
2. Reverse-exponential arrival jackpot (R_0 = 140.0, tau = 3.5s) upon first bucket entry.
3. Relative velocity matching / damping (r_match = 2.0 * exp(-0.5 * (closing_speed/1.0)^2)) inside bucket.
4. Dense bucket reward blast (base 25.0 + 10.0 * t_dwell^2) escalating toward 3.0s hold.
5. Zero collision penalty (p_collision = 0.0) to completely prevent cautious hang-back behaviour.
6. Proportional Navigation lead (w_pn = 3.5) for evasive turn anticipation.
"""

import sys
import os
import time
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"
if not PYTHON_BIN.exists():
    PYTHON_BIN = Path(sys.executable)

from control.run_1m5_sweep import evaluate_policy_detailed

OUT_DIR = PROJECT_ROOT / "control" / "weights" / "sprint_and_latch_v1"
LOG_FILE = PROJECT_ROOT / "outputs" / "benchmarks" / "sprint_and_latch_v1.log"
VAL_SUITE = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "validation_suite_30.json"
WARM_START_WEIGHTS = PROJECT_ROOT / "control" / "weights" / "cloned_expert_v1" / "actor_weights.pt"
REPORT_PATH = PROJECT_ROOT / "docs" / "SPRINT_AND_LATCH_EVALUATION_REPORT.md"


def main():
    print("=" * 80)
    print("  SPRINT & LATCH RECURRENT RL TRAINING & CALIBRATION")
    print(f"  Warm-Start: {WARM_START_WEIGHTS}")
    print(f"  Output Directory: {OUT_DIR}")
    print(f"  Validation Suite: {VAL_SUITE}")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PYTHON_BIN),
        str(PROJECT_ROOT / "control" / "train_rl.py"),
        "--timesteps", "1000000",
        "--envs", "8",
        "--hidden", "128",
        "--lr", "2e-4",
        "--ent-coef", "0.002",
        "--gamma", "0.995",
        "--epochs", "4",
        "--eval-freq", "25000",
        "--checkpoint-freq", "50000",
        "--seed", "42",
        "--warm-start-weights", str(WARM_START_WEIGHTS),
        "--warm-start-log-std", "-0.8",
        "--output-dir", str(OUT_DIR),
        "--reward-mode", "expressive_iterative",
        "--hyper-weight", "0.45",
        "--wandb-run-name", "sprint_and_latch_v1",
        "--wandb-notes", "Sprint & Latch synthesis: differential sprint closing + arrival jackpot + speed match inside + dwell blast",
    ]

    t0 = time.time()
    with open(LOG_FILE, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT)
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()
        proc.wait()

    elapsed_s = time.time() - t0
    print(f"\n[TRAIN] Training finished in {elapsed_s / 60.0:.1f} minutes with exit code {proc.returncode}")

    # Evaluate best candidate model
    model_weights = OUT_DIR / "best_model" / "actor_weights.pt"
    if not model_weights.exists():
        model_weights = OUT_DIR / "actor_weights.pt"
    if not model_weights.exists():
        model_weights = OUT_DIR / "best_model" / "best_model.zip"

    print(f"\n[EVAL] Evaluating policy on 30-scenario validation suite: {model_weights}")
    m = evaluate_policy_detailed(model_weights, VAL_SUITE)
    print(f"[METRICS] Hyper FOV: {m.get('hyper_mean_fov', 0.0):.1f}%")
    print(f"[METRICS] Mean Distance: {m.get('hyper_mean_dist', 99.0):.2f}m (Closest: {m.get('hyper_min_dist', 99.0):.2f}m)")
    print(f"[METRICS] Peak Continuous Dwell: {m.get('hyper_continuous_dwell', 0.0):.2f}s")
    print(f"[METRICS] Success Rate (>=3.0s dwell): {m.get('hyper_success_rate', 0.0):.1f}%")
    print(f"[METRICS] Collisions: {m.get('hyper_collisions', 0)}")
    print(f"[METRICS] Speed Range: {m.get('hyper_min_speed', 0.0):.1f}m/s (turns) <-> {m.get('hyper_max_speed', 0.0):.1f}m/s (straights)")

    # Write evaluation report
    report_lines = [
        "# Sprint & Latch Recurrent Guidance Evaluation Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "**Pipeline:** RecurrentPPO fine-tuned from `cloned_expert_v1` with Sprint & Latch Reward Shaping",
        f"**Training Timesteps:** 1,000,000 steps ({elapsed_s / 60.0:.1f} minutes on 8 vectorized environments)",
        "**Reward Mode:** `expressive_iterative` with active differential sprint closing potential",
        "**Model Weights:** [`control/weights/sprint_and_latch_v1/best_model/actor_weights.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/sprint_and_latch_v1/best_model/actor_weights.pt)",
        "",
        "---",
        "",
        "## 1. Comparative Validation Suite Benchmark",
        "",
        "| Architecture | Timesteps | Hyper FOV% | Mean Dist | Closest Approach | Speed Range | Max Continuous Dwell | Success Rate (≥3.0s) | Collisions |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        "| Cloned Expert Baseline | 0 | 41.1% | 21.8 m | 1.12 m | 3.4 – 7.9 m/s | 0.26 s | 0.0% | 0 |",
        "| Overnight Run 01 (Pure Bucket Blast) | 1.5M | 89.9% | 51.3 m | 3.05 m | 0.3 – 4.0 m/s | 0.20 s | 0.0% | 0 |",
        "| Overnight Run 03 (Bucket Magnet) | 1.5M | 14.6% | 22.4 m | 1.72 m | 4.0 – 6.0 m/s | 0.14 s | 0.0% | 0 |",
        "| Overnight Run 14 (Peak Lock Pursuit) | 1.5M | 5.0% | 47.2 m | 2.87 m | 0.0 – 4.1 m/s | 0.26 s | 0.0% | 0 |",
        f"| **Sprint & Latch (Current)** | **1.0M** | **{m.get('hyper_mean_fov', 0.0):.1f}%** | **{m.get('hyper_mean_dist', 99.0):.2f} m** | **{m.get('hyper_min_dist', 99.0):.2f} m** | **{m.get('hyper_min_speed', 0.0):.1f} – {m.get('hyper_max_speed', 0.0):.1f} m/s** | **{m.get('hyper_continuous_dwell', 0.0):.2f} s** | **{m.get('hyper_success_rate', 0.0):.1f}%** | **{m.get('hyper_collisions', 0)}** |",
        "",
        "---",
        "",
        "## 2. Key Synthesis Architectural Elements",
        "",
        "1. **Active Closing Sprint Potential ($w_{\\text{sprint}} = 3.5$):** Directly rewards closing speed $\\dot{d}_{\\text{close}}$ when outside the bucket ($d > 4.5\\text{m}$), eliminating the low-speed telescope trap.",
        "2. **Reverse-Exponential Arrival Jackpot ($R_0 = 140.0, \\tau = 3.5\\text{s}$):** Awarded once upon first bucket entry, strongly incentivizing an immediate arrival sprint.",
        "3. **Speed Matching & Braking Inside Bucket:** Transitions from sprint drive into Gaussian velocity matching ($2.0 \\cdot \\exp(-0.5 (\\dot{d}/1.0)^2)$), helping the drone match speed and avoid overshooting.",
        "4. **Dense Escalating Bucket Blast ($B = 25.0, S = 10.0$):** Massive reward acceleration inside the $2.5\\text{--}4.5\\text{m}$ formation basket.",
        "5. **Zero Collision Penalty ($P_{\\text{collision}} = 0.0$):** Zero punishment for drone contact, preventing cautious hang-back behaviour.",
        "6. **Proportional Navigation Lead Guidance ($w_{\\text{pn}} = 3.5$):** Anticipates evasive flight paths to slice corner apexes.",
    ]
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n[REPORT] Saved benchmark report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
