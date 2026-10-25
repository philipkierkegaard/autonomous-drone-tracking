#!/usr/bin/env python3
"""
Calibrated Expressive RL Training & Evaluation Pipeline.
Trains 1,500,000 timesteps from cloned expert weights (control/weights/cloned_expert_v1/actor_weights.pt)
with the calibrated expressive reward formulation:
1. Continuous distance pull peaking at 3.5m center with linear sprint closing potential outside (> 4.5m).
2. Altitude floor breach penalty (-20.0) preventing ground strikes (< 1.0m AGL).
3. Target FOV loss step penalty (-1.5/step) ensuring continuous optical contact.
4. 0.5s dwell-gated reverse-exponential fast-arrival jackpot (R_0 = 140, tau = 3.5s) preventing fly-by exploits.
5. Dense bucket reward blast inside [2.5m, 4.5m], peaking at 3.5m center (base 6.0 + 2.0 * t_dwell).
6. Target collision penalty (-20.0) at 0.5m boundary, expanding braking cushion to 2.0m.
7. Proportional Navigation velocity vector alignment (w_pn = 2.5) for turn anticipation.
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

OUT_DIR = PROJECT_ROOT / "control" / "weights" / "calibrated_expressive_1m5"
LOG_FILE = PROJECT_ROOT / "outputs" / "benchmarks" / "calibrated_expressive_1m5.log"
VAL_SUITE = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "validation_suite_30.json"
WARM_START_WEIGHTS = PROJECT_ROOT / "control" / "weights" / "cloned_expert_v1" / "actor_weights.pt"
REPORT_PATH = PROJECT_ROOT / "docs" / "CALIBRATED_EXPRESSIVE_1M5_EVALUATION_REPORT.md"


def main():
    print("=" * 80)
    print("  CALIBRATED EXPRESSIVE RECURRENT RL TRAINING & BENCHMARK (1.5M STEPS)")
    print(f"  Warm-Start Expert: {WARM_START_WEIGHTS}")
    print(f"  Output Directory:  {OUT_DIR}")
    print(f"  Validation Suite:  {VAL_SUITE}")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PYTHON_BIN),
        str(PROJECT_ROOT / "control" / "train_rl.py"),
        "--timesteps", "1500000",
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
        "--reward-mode", "expressive",
        "--hyper-weight", "0.45",
        "--wandb-run-name", "calibrated_expressive_1m5",
        "--wandb-notes", "Calibrated Expressive: 0.5s dwell-gated arrival jackpot + 3.5m center Gaussian well + 0.5m collision + FOV loss penalty + PN lead",
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
        "# Calibrated Expressive Guidance Evaluation Report (1.5M Steps)",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "**Pipeline:** RecurrentPPO fine-tuned from `cloned_expert_v1` with Calibrated Expressive Reward Shaping",
        f"**Training Timesteps:** 1,500,000 steps ({elapsed_s / 60.0:.1f} minutes on 8 vectorized environments)",
        "**Reward Mode:** `expressive` (0.5s dwell-gated arrival jackpot, 3.5m center Gaussian well, 0.5m collision, FOV loss penalty)",
        f"**Model Weights:** [`control/weights/calibrated_expressive_1m5/best_model/actor_weights.pt`](file://{OUT_DIR}/best_model/actor_weights.pt)",
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
        "| Sprint & Latch (Ungated) | 1.0M | 15.3% | 50.15 m | 2.91 m | 3.1 – 4.5 m/s | 0.14 s | 0.0% | 0 |",
        f"| **Calibrated Expressive (Current)** | **1.5M** | **{m.get('hyper_mean_fov', 0.0):.1f}%** | **{m.get('hyper_mean_dist', 99.0):.2f} m** | **{m.get('hyper_min_dist', 99.0):.2f} m** | **{m.get('hyper_min_speed', 0.0):.1f} – {m.get('hyper_max_speed', 0.0):.1f} m/s** | **{m.get('hyper_continuous_dwell', 0.0):.2f} s** | **{m.get('hyper_success_rate', 0.0):.1f}%** | **{m.get('hyper_collisions', 0)}** |",
        "",
        "---",
        "",
        "## 2. Key Synthesis Architectural Elements",
        "",
        "1. **Continuous Distance Pull Peaking at 3.5m ($R_{\\text{dist}} = 8.0 \\cdot \\exp(-0.5 ((d - 3.5)/1.0)^2)$):** Drives chaser toward the 3.5m bucket center.",
        "2. **Active Sprint Closing Potential ($w_{\\text{sprint}} = 3.5$):** Prevents the $1/d^2$ pinhole vanishing gradient trap at long distances (> 4.5m).",
        "3. **0.5s Dwell-Gated Fast-Arrival Jackpot ($R_0 = 140.0, \\tau = 3.5\\text{s}$):** Awarded once after holding continuous lock for 0.5s, eliminating the bullet fly-by exploit.",
        "4. **Dense Bucket Bombardment Scaled to Center ($B = 6.0 \\cdot \\exp(-0.5 ((d - 3.5)/0.6)^2) + 2.0 \\cdot t_{\\text{dwell}}$):** Floods the agent with dense rewards inside the bucket without blowing up PPO advantage variance.",
        "5. **0.5m Collision Clearance Radius ($P_{\\text{collision}} = -20.0$):** Expands the braking cushion inside the bucket from 1.3m to 2.0m.",
        "6. **Sightline FOV Loss Step Penalty ($P_{\\text{lost}} = -1.5$):** Explicit negative reward while blind, training aggressive re-acquisition.",
        "7. **Proportional Navigation Lead Guidance ($w_{\\text{pn}} = 2.5$):** Aligns velocity vectors ($\\vec{v}_c \\cdot \\vec{v}_t$) for corner apex anticipation.",
    ]
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n[REPORT] Saved benchmark report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
