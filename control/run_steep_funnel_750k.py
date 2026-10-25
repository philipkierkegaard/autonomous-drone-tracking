#!/usr/bin/env python3
"""
Dedicated Training & Evaluation Runner: Steep Variable-Curvature Funnel RL Controller (750k Steps).
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Key Architectural Principles:
1. Warm-start initialized from cloned expert weights (control/weights/cloned_expert_v1/actor_weights.pt)
2. Low exploration noise: warm_start_log_std = -2.0 (std ~ 0.135) to protect forward throttle and tracking reflex
3. Steep Funnel Reward (steep_funnel_bullseye):
   - Continuous piece-wise standoff potential peaked at 3.5m (+10 max, bleeds negative past 8.5m)
   - Distance-adaptive Lorentzian centering funnel (steep at 3.5m, sigma=0.15)
   - Multiplicative gating: centering error drastically diminishes standoff rewards
   - Unconditional distance bleed: camping at >8.5m bleeds negative rewards regardless of centering
   - Zero alive/FOV subsidy: eliminates passive standoff hedging exploit
4. 750,000 steps with 8 parallel environments (~10-12 minutes on Apple Silicon)
5. Comprehensive post-training evaluation on both 30-scenario validation suite and 50-scenario held-out test suite.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"
VAL_SUITE_PATH = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "validation_suite_30.json"
HELDOUT_SUITE_PATH = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "heldout_test_suite_50.json"
EXPERT_WEIGHTS_PATH = PROJECT_ROOT / "control" / "weights" / "cloned_expert_v1" / "actor_weights.pt"
OUTPUT_DIR = PROJECT_ROOT / "control" / "weights" / "steep_funnel_750k"
LOG_FILE = PROJECT_ROOT / "outputs" / "benchmarks" / "control" / "steep_funnel_750k.log"
RESULTS_JSON_PATH = PROJECT_ROOT / "outputs" / "benchmarks" / "control" / "steep_funnel_750k_evaluation.json"


def evaluate_policy_on_suite(model_path: Path, dataset_path: Path, noise_mode: str = "empirical", frame_drop_rate: float = 0.05) -> Dict[str, Any]:
    """Evaluates a policy on a standardized dataset manifest and returns comprehensive metrics."""
    from control.predictive_controller import RecurrentVisualServoController
    from benchmarks.control.benchmark_controller import PursuitDataset, evaluate_single_run

    dataset = PursuitDataset.load_json(dataset_path)
    ctrl = RecurrentVisualServoController(model_path=str(model_path), w_nominal=0.0859)

    runs = []
    for sc in dataset.scenarios:
        ctrl.reset()
        r = evaluate_single_run(
            scenario=sc,
            controller=ctrl,
            controller_type="rl",
            noise_mode=noise_mode,
            frame_drop_rate=frame_drop_rate,
            standoff_nominal=3.5,
            use_kalman=True
        )
        runs.append(r)

    n = len(runs)
    if n == 0:
        return {}

    succ = sum(1 for r in runs if r["success"])
    success_rate = (succ / n) * 100.0
    succ_1s = sum(1 for r in runs if r.get("success_1s", r["success"]))
    succ_1s_rate = (succ_1s / n) * 100.0
    succ_3s = sum(1 for r in runs if r.get("success_3s", False))
    succ_3s_rate = (succ_3s / n) * 100.0
    max_dwell = float(np.mean([r.get("max_continuous_dwell_s", 0.0) for r in runs]))
    fov_retention = float(np.mean([r["fov_lock_pct"] for r in runs]))
    firing_win = float(np.mean([r["firing_window_s"] for r in runs]))
    min_dist = float(np.mean([r["min_distance"] for r in runs]))
    colls = sum(1 for r in runs if r["collision_breach"])
    collision_rate = (colls / n) * 100.0

    by_profile = {}
    for p in ["cruising", "evasive", "hyper_evasive"]:
        sub = [r for r in runs if r["profile"] == p]
        if sub:
            by_profile[p] = {
                "n": len(sub),
                "success_rate": float((sum(1 for r in sub if r["success"]) / len(sub)) * 100.0),
                "success_1s": float((sum(1 for r in sub if r.get("success_1s", r["success"])) / len(sub)) * 100.0),
                "success_3s": float((sum(1 for r in sub if r.get("success_3s", False)) / len(sub)) * 100.0),
                "fov_retention": float(np.mean([r["fov_lock_pct"] for r in sub])),
                "firing_window_s": float(np.mean([r["firing_window_s"] for r in sub])),
                "max_dwell_s": float(np.mean([r.get("max_continuous_dwell_s", 0.0) for r in sub])),
                "min_distance_m": float(np.mean([r["min_distance"] for r in sub])),
            }

    return {
        "n_scenarios": n,
        "success_rate_pct": success_rate,
        "success_1s_pct": succ_1s_rate,
        "success_3s_pct": succ_3s_rate,
        "max_dwell_s": max_dwell,
        "fov_retention_pct": fov_retention,
        "firing_window_s": firing_win,
        "min_distance_m": min_dist,
        "collision_breaches": colls,
        "collision_rate_pct": collision_rate,
        "by_profile": by_profile,
    }


def main():
    print("=" * 80)
    print("STEEP VARIABLE-CURVATURE FUNNEL RL TRAINING CAMPAIGN (750,000 STEPS)")
    print("DTU Autonomous Drone Pursuit & Interception System")
    print("=" * 80)
    print(f"Timestamp:          {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target Timesteps:   750,000")
    print(f"Parallel Envs:      8")
    print(f"Learning Rate:      8e-5 (annealing to 1.5e-5)")
    print(f"Entropy Coef:       0.001")
    print(f"Warm-Start Weights: {EXPERT_WEIGHTS_PATH}")
    print(f"Warm-Start Log Std: -2.0 (std ~ 0.135)")
    print(f"Reward Mode:        steep_funnel_bullseye")
    print(f"Evaluation Freq:    15,000 timesteps")
    print(f"Output Directory:   {OUTPUT_DIR}")
    print(f"Log File:           {LOG_FILE}")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not EXPERT_WEIGHTS_PATH.exists():
        print(f"[FATAL] Expert weights not found at {EXPERT_WEIGHTS_PATH}!")
        sys.exit(1)

    train_cmd = [
        str(PYTHON_BIN),
        str(PROJECT_ROOT / "control" / "train_rl.py"),
        "--timesteps", "750000",
        "--envs", "8",
        "--lr", "8e-5",
        "--ent-coef", "0.001",
        "--reward-mode", "steep_funnel_bullseye",
        "--hyper-weight", "0.45",
        "--eval-freq", "15000",
        "--checkpoint-freq", "50000",
        "--output-dir", str(OUTPUT_DIR),
        "--warm-start-weights", str(EXPERT_WEIGHTS_PATH),
        "--warm-start-log-std", "-2.0",
        "--no-wandb",
        "--seed", "42"
    ]

    print("\n[1/3] Launching Policy Gradient Optimization...")
    start_time = time.time()

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== Command: {' '.join(str(c) for c in train_cmd)} ===\n\n")
        proc = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT)
        )
        for line in proc.stdout:
            f.write(line)
            f.flush()
            if any(k in line for k in ["rollout/", "eval/", "TASK-LEVEL", "DONE", "BEST", "WEIGHTS", "WARM_START", "Error", "Traceback", "Exception"]):
                sys.stdout.write(f"  {line}")
                sys.stdout.flush()
        proc.wait()

    ret_code = proc.returncode
    elapsed_min = (time.time() - start_time) / 60.0

    if ret_code != 0:
        print(f"\n[FATAL] Training exited with non-zero status code: {ret_code}")
        print(f"Check log file at: {LOG_FILE}")
        sys.exit(ret_code)

    print(f"\n[TRAIN] Training successfully completed in {elapsed_min:.1f} minutes!")

    # Locate best model or fallback
    best_model_path = OUTPUT_DIR / "best_model" / "best_model.zip"
    if not best_model_path.exists():
        final_p = OUTPUT_DIR / "recurrent_ppo_drone_final.zip"
        if final_p.exists():
            print(f"[WARN] best_model.zip not found, falling back to final model: {final_p}")
            best_model_path = final_p
        else:
            print(f"[FATAL] No model zip found in {OUTPUT_DIR}!")
            sys.exit(1)

    print(f"\n[2/3] Evaluating Locked-in Policy ({best_model_path.name})...")
    print("  -> Evaluating on Clean Validation Suite (30 scenarios)...")
    val_metrics = evaluate_policy_on_suite(best_model_path, VAL_SUITE_PATH, noise_mode="none", frame_drop_rate=0.0)

    print("  -> Evaluating on Held-Out Test Suite (50 scenarios, empirical YOLO noise + 5% frame drops)...")
    heldout_metrics = evaluate_policy_on_suite(best_model_path, HELDOUT_SUITE_PATH, noise_mode="empirical", frame_drop_rate=0.05)

    # Read callback meta score if present
    best_score = 0.0
    meta_file = OUTPUT_DIR / "best_model" / "best_metrics.json"
    if meta_file.exists():
        try:
            m_data = json.loads(meta_file.read_text(encoding="utf-8"))
            best_score = float(m_data.get("task_score", 0.0))
        except Exception:
            pass

    results_data = {
        "model_id": "steep_funnel_750k",
        "reward_mode": "steep_funnel_bullseye",
        "timesteps": 750000,
        "elapsed_min": elapsed_min,
        "best_task_score": best_score,
        "best_model_path": str(best_model_path),
        "validation_metrics": val_metrics,
        "heldout_metrics": heldout_metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    RESULTS_JSON_PATH.write_text(json.dumps(results_data, indent=2), encoding="utf-8")
    print(f"[EVAL] Results persisted to: {RESULTS_JSON_PATH}")

    # Print summary table
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS SUMMARY: STEEP FUNNEL RL (750k STEPS)")
    print("=" * 80)
    print(f"Validation Suite (30 scenarios, Clean):")
    print(f"  Success Rate (Catch):   {val_metrics.get('success_rate_pct', 0.0):.1f}%")
    print(f"  FOV Retention:          {val_metrics.get('fov_retention_pct', 0.0):.1f}%")
    print(f"  Mean Firing Window:     {val_metrics.get('firing_window_s', 0.0):.2f}s")
    print(f"  Mean Min Distance:      {val_metrics.get('min_distance_m', 0.0):.2f}m")
    print(f"  Collision Rate:         {val_metrics.get('collision_rate_pct', 0.0):.1f}%")
    print("-" * 80)
    print(f"Held-Out Test Suite (50 scenarios, Jetson YOLO Noise + 5% Frame Drop):")
    print(f"  Success Rate (Catch):   {heldout_metrics.get('success_rate_pct', 0.0):.1f}%")
    print(f"  FOV Retention:          {heldout_metrics.get('fov_retention_pct', 0.0):.1f}%")
    print(f"  Mean Firing Window:     {heldout_metrics.get('firing_window_s', 0.0):.2f}s")
    print(f"  Mean Min Distance:      {heldout_metrics.get('min_distance_m', 0.0):.2f}m")
    print(f"  Collision Rate:         {heldout_metrics.get('collision_rate_pct', 0.0):.1f}%")
    print("-" * 80)
    print("Performance by Profile (Held-Out Test):")
    for prof, pdata in heldout_metrics.get("by_profile", {}).items():
        print(f"  [{prof.upper()}] (n={pdata['n']}): Catch={pdata['success_rate']:.1f}% | FOV={pdata['fov_retention']:.1f}% | Dwell={pdata['firing_window_s']:.2f}s | MinDist={pdata['min_distance_m']:.2f}m")
    print("=" * 80)


if __name__ == "__main__":
    main()
