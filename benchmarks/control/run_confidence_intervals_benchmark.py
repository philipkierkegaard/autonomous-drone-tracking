#!/usr/bin/env python3
"""
Orchestrator for 10 x 600 (12,000 Runs) Paired Confidence Interval Benchmarking.
Evaluates Tuned Kinematic PID vs Residual Reinforcement Learning (v6) across
10 independent, seed-stratified 600-scenario folds (3 profiles x 200 = 600/fold).
Computes:
- Mean and 95% Student-t Confidence Intervals for PID and RRL
- Paired differences (Delta = RRL - PID) with 95% CI
- Paired Student-t test p-values for statistical significance.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.control.benchmark_controller import run_dataset_benchmark
from control.dataset_generator import PursuitDataset


def compute_ci(data: List[float], confidence: float = 0.95):
    arr = np.array(data, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    if n < 2:
        return mean, 0.0, (mean, mean)
    se = float(stats.sem(arr))
    t_crit = float(stats.t.ppf((1 + confidence) / 2.0, df=n - 1))
    margin = t_crit * se
    return mean, se, (mean - margin, mean + margin)


def compute_paired_stats(data_rrl: List[float], data_pid: List[float], confidence: float = 0.95):
    diffs = np.array(data_rrl) - np.array(data_pid)
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    if n < 2 or np.all(diffs == 0.0):
        return mean_diff, 0.0, (mean_diff, mean_diff), 1.0
    se = float(stats.sem(diffs))
    t_crit = float(stats.t.ppf((1 + confidence) / 2.0, df=n - 1))
    margin = t_crit * se
    t_stat = mean_diff / max(1e-12, se)
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1)))
    return mean_diff, se, (mean_diff - margin, mean_diff + margin), p_val


def main():
    dataset_dir = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "ci_10x600"
    output_dir = PROJECT_ROOT / "outputs" / "benchmarks" / "control"
    model_path = PROJECT_ROOT / "outputs" / "models" / "residual_ppo_v6_hard_2s_dwell" / "best_model" / "best_model.zip"
    pid_gains_path = PROJECT_ROOT / "control" / "weights" / "tuned_pid_gains.json"

    fold_files = sorted(list(dataset_dir.glob("ci_benchmark_600_fold_*.json")))
    if not fold_files:
        print("[ERROR] No fold files found in", dataset_dir)
        sys.exit(1)

    print("=" * 95)
    print("  AUTONOMOUS DRONE PURSUIT: 10 x 600 PAIRED CONFIDENCE INTERVAL PIPELINE")
    print(f"  Folds to Benchmark:  {len(fold_files)} folds (600 scenarios/fold)")
    print(f"  Total Runs:          {len(fold_files) * 600 * 2:,} simulations (6,000 PID + 6,000 RRL)")
    print(f"  PID Gains:           {pid_gains_path.relative_to(PROJECT_ROOT)}")
    print(f"  RRL Model:           {model_path.relative_to(PROJECT_ROOT)}")
    print("=" * 95)

    fold_results_pid = []
    fold_results_rrl = []

    start_all = time.perf_counter()

    for idx, fold_file in enumerate(fold_files, 1):
        print(f"\n=========================================================================================")
        print(f"  EXECUTING FOLD [{idx:02d}/10]: {fold_file.stem}")
        print(f"=========================================================================================")

        # 1. PID Benchmark on Fold
        pid_json_expected = output_dir / f"benchmark_results_{fold_file.stem}_pid_tuned_kalman_empirical_d5.json"
        if pid_json_expected.exists():
            print(f"---> [FOLD {idx:02d}/10] Found existing PID results: {pid_json_expected.name}. Loading...")
            with open(pid_json_expected, "r", encoding="utf-8") as f:
                pid_res = json.load(f)
        else:
            print(f"\n---> [FOLD {idx:02d}/10] Running PID Baseline...")
            t0 = time.perf_counter()
            _ = run_dataset_benchmark(
                dataset_path=fold_file,
                controller_type="pid",
                pid_gains=str(pid_gains_path),
                use_kalman=True,
                noise_mode="empirical",
                frame_drop_rate=0.05,
                output_dir=output_dir
            )
            with open(pid_json_expected, "r", encoding="utf-8") as f:
                pid_res = json.load(f)
            print(f"---> [FOLD {idx:02d}/10] PID finished in {time.perf_counter() - t0:.1f}s")
        fold_results_pid.append(pid_res)

        # 2. Residual RL Benchmark on Fold
        rrl_json_expected = output_dir / f"benchmark_results_{fold_file.stem}_residual_ppo_v6_hard_2s_dwell_best_model_kalman_empirical_d5.json"
        if rrl_json_expected.exists():
            print(f"---> [FOLD {idx:02d}/10] Found existing RRL results: {rrl_json_expected.name}. Loading...")
            with open(rrl_json_expected, "r", encoding="utf-8") as f:
                rrl_res = json.load(f)
        else:
            print(f"\n---> [FOLD {idx:02d}/10] Running Residual RL Policy...")
            t1 = time.perf_counter()
            _ = run_dataset_benchmark(
                dataset_path=fold_file,
                controller_type="residual",
                model_path=str(model_path),
                pid_gains=str(pid_gains_path),
                use_kalman=True,
                noise_mode="empirical",
                frame_drop_rate=0.05,
                output_dir=output_dir
            )
            with open(rrl_json_expected, "r", encoding="utf-8") as f:
                rrl_res = json.load(f)
            print(f"---> [FOLD {idx:02d}/10] RRL finished in {time.perf_counter() - t1:.1f}s")
        fold_results_rrl.append(rrl_res)

    total_time = time.perf_counter() - start_all
    print(f"\n[DONE] All 10 folds (12,000 runs) evaluated in {total_time/60:.1f} minutes!")

    # =========================================================================
    # STATISTICAL SYNTHESIS ACROSS 10 FOLDS
    # =========================================================================
    profiles = ["cruising", "evasive", "hyper_evasive"]
    metrics_to_track = [
        ("success_rate_2s_pct", "2.0s Dwell Catch (%)", "{:.1f}%"),
        ("success_rate_1s_pct", "1.0s Dwell Catch (%)", "{:.1f}%"),
        ("success_rate_3s_pct", "3.0s Dwell Catch (%)", "{:.1f}%"),
        ("mean_time_to_lock_2s", "Time-to-Lock 2s (s)", "{:.2f}s"),
        ("mean_dwell_s", "Mean Max Dwell (s)", "{:.2f}s"),
        ("mean_fov_lock_pct", "FOV Retention (%)", "{:.1f}%"),
        ("mean_jerk_yaw_deg_s2", "Yaw Jerk (deg/s^2)", "{:.1f}"),
        ("mean_jerk_surge_m_s2", "Surge Jerk (m/s^2)", "{:.2f}"),
        ("collision_breach_pct", "Collision Breaches (%)", "{:.1f}%"),
    ]

    analysis = {
        "metadata": {
            "num_folds": len(fold_files),
            "scenarios_per_fold": 600,
            "total_runs_per_controller": len(fold_files) * 600,
            "confidence_level": 0.95,
            "total_evaluation_time_min": total_time / 60.0,
        },
        "overall": {},
        "by_profile": {}
    }

    # Aggregate Overall stats
    for m_key, m_name, fmt in metrics_to_track:
        p_vals = [f["overall"][m_key] for f in fold_results_pid if f["overall"][m_key] is not None]
        r_vals = [f["overall"][m_key] for f in fold_results_rrl if f["overall"][m_key] is not None]

        p_m, p_se, p_ci = compute_ci(p_vals)
        r_m, r_se, r_ci = compute_ci(r_vals)
        d_m, d_se, d_ci, p_val = compute_paired_stats(r_vals, p_vals)

        analysis["overall"][m_key] = {
            "name": m_name,
            "pid_mean": p_m, "pid_ci_95": p_ci, "pid_se": p_se,
            "rrl_mean": r_m, "rrl_ci_95": r_ci, "rrl_se": r_se,
            "paired_delta_mean": d_m, "paired_delta_ci_95": d_ci,
            "paired_delta_se": d_se, "p_value": p_val
        }

    # By Profile stats
    for prof in profiles:
        analysis["by_profile"][prof] = {}
        for m_key, m_name, fmt in metrics_to_track:
            p_vals = []
            for f in fold_results_pid:
                sub = next((item for item in f["profile_summaries"] if item["profile"] == prof), None)
                if sub and sub.get(m_key) is not None:
                    p_vals.append(sub[m_key])

            r_vals = []
            for f in fold_results_rrl:
                sub = next((item for item in f["profile_summaries"] if item["profile"] == prof), None)
                if sub and sub.get(m_key) is not None:
                    r_vals.append(sub[m_key])

            p_m, p_se, p_ci = compute_ci(p_vals)
            r_m, r_se, r_ci = compute_ci(r_vals)
            d_m, d_se, d_ci, p_val = compute_paired_stats(r_vals, p_vals)

            analysis["by_profile"][prof][m_key] = {
                "name": m_name,
                "pid_mean": p_m, "pid_ci_95": p_ci,
                "rrl_mean": r_m, "rrl_ci_95": r_ci,
                "paired_delta_mean": d_m, "paired_delta_ci_95": d_ci,
                "p_value": p_val
            }

    # Save JSON report
    report_json_path = output_dir / "ci_10x600_final_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    # Generate Markdown Table Report
    report_md_path = output_dir / "ci_10x600_final_report.md"
    md_lines = [
        "# Controller Comparison with 95% Confidence Intervals (10 x 600 Trajectories)",
        f"**Sample Size**: 10 folds x 600 scenarios = 6,000 trajectories per controller (12,000 total runs)  ",
        f"**Method**: Paired Student's t 95% Confidence Intervals & Two-Tailed Significance Testing  ",
        "",
        "## 1. Overall Aggregate Metrics (6,000 Trajectories)",
        "",
        "| Evaluation Metric | Tuned PID Baseline (95% CI) | Residual RL v6 (95% CI) | Paired Difference (Δ, 95% CI) | p-value | Significance |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    for m_key, m_name, fmt in metrics_to_track:
        ov = analysis["overall"][m_key]
        p_str = f"{ov['pid_mean']:.2f} [{ov['pid_ci_95'][0]:.2f}, {ov['pid_ci_95'][1]:.2f}]"
        r_str = f"{ov['rrl_mean']:.2f} [{ov['rrl_ci_95'][0]:.2f}, {ov['rrl_ci_95'][1]:.2f}]"
        d_sign = "+" if ov["paired_delta_mean"] >= 0 else ""
        d_str = f"{d_sign}{ov['paired_delta_mean']:.2f} [{ov['paired_delta_ci_95'][0]:.2f}, {ov['paired_delta_ci_95'][1]:.2f}]"
        p_val = ov["p_value"]
        p_str_val = f"< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
        sig_str = "**p < 0.001 (***)**" if p_val < 0.001 else ("**p < 0.01 (**)**" if p_val < 0.01 else ("*p < 0.05 (*)*" if p_val < 0.05 else "n.s."))
        md_lines.append(f"| **{m_name}** | {p_str} | **{r_str}** | {d_str} | {p_str_val} | {sig_str} |")

    md_lines.extend([
        "",
        "## 2. Profile Breakdown (2.0s Dwell Catch Success Rate)",
        "",
        "| Target Profile | Tuned PID Baseline (95% CI) | Residual RL v6 (95% CI) | Paired Difference (Δ, 95% CI) | p-value |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ])

    for prof in profiles:
        pr = analysis["by_profile"][prof]["success_rate_2s_pct"]
        p_str = f"{pr['pid_mean']:.1f}% [{pr['pid_ci_95'][0]:.1f}%, {pr['pid_ci_95'][1]:.1f}%]"
        r_str = f"{pr['rrl_mean']:.1f}% [{pr['rrl_ci_95'][0]:.1f}%, {pr['rrl_ci_95'][1]:.1f}%]"
        d_sign = "+" if pr["paired_delta_mean"] >= 0 else ""
        d_str = f"{d_sign}{pr['paired_delta_mean']:.1f}% [{pr['paired_delta_ci_95'][0]:.1f}%, {pr['paired_delta_ci_95'][1]:.1f}%]"
        p_val = pr["p_value"]
        p_str_val = f"< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
        md_lines.append(f"| **{prof.capitalize()}** (2,000 runs) | {p_str} | **{r_str}** | {d_str} | {p_str_val} |")

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print("\n" + "=" * 95)
    print("  FINAL 10 x 600 CONFIDENCE INTERVAL SYNTHESIS")
    print("=" * 95)
    print("\n".join(md_lines))
    print(f"\n[REPORT] Saved full statistical reports to:")
    print(f"  - {report_json_path}")
    print(f"  - {report_md_path}")


if __name__ == "__main__":
    main()
