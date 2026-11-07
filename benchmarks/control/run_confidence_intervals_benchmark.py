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
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from scipy import stats
from concurrent.futures import ProcessPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.control.benchmark_controller import run_dataset_benchmark
from control.dataset_generator import PursuitDataset


def _eval_fold_worker(args):
    (
        fold_file,
        controller_type,
        model_path,
        pid_gains_path,
        output_dir,
        tau_yaw,
    ) = args
    return run_dataset_benchmark(
        dataset_path=fold_file,
        controller_type=controller_type,
        model_path=str(model_path) if model_path else None,
        pid_gains=str(pid_gains_path) if pid_gains_path else None,
        use_kalman=True,
        noise_mode="empirical",
        frame_drop_rate=0.05,
        output_dir=output_dir,
        tau_yaw=tau_yaw,
    )


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
    parser = argparse.ArgumentParser(description="10x600 Paired Confidence Interval Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/models/residual_ppo_v9_wide35/best_model/best_model.zip",
        help="Path to residual RL model checkpoint"
    )
    parser.add_argument(
        "--pid-gains",
        type=str,
        default="control/weights/tuned_pid_gains_pre_attitude_dynamics.json",
        help="Path to baseline PID gains JSON"
    )
    parser.add_argument(
        "--tau-yaw",
        type=float,
        default=0.150,
        help="Sim yaw time constant in seconds (default: 0.150)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel worker processes for evaluating folds (default: 3)"
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="ci_10x600_v9_final_report",
        help="Base filename for output JSON and Markdown reports"
    )
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "ci_10x600"
    output_dir = PROJECT_ROOT / "outputs" / "benchmarks" / "control"
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    pid_gains_path = Path(args.pid_gains)
    if not pid_gains_path.is_absolute():
        pid_gains_path = PROJECT_ROOT / pid_gains_path

    fold_files = sorted(list(dataset_dir.glob("ci_benchmark_600_fold_*.json")))
    if not fold_files:
        print("[ERROR] No fold files found in", dataset_dir)
        sys.exit(1)

    # Determine RRL ctrl_tag matching benchmark_controller.py
    p = model_path
    if p.parent.name in ["best_model", "checkpoints"]:
        tag_name = f"{p.parent.parent.name}_{p.parent.name}" if p.parent.name == "best_model" else f"{p.parent.parent.name}_{p.stem}"
    else:
        tag_name = p.parent.name if p.parent.name not in ["weights", "."] else p.stem
    rrl_ctrl_tag = tag_name if tag_name.startswith("residual_") else f"residual_{tag_name}"
    tau_suffix = f"_tau{int(args.tau_yaw * 1000)}" if args.tau_yaw is not None else ""

    print("=" * 95)
    print("  AUTONOMOUS DRONE PURSUIT: 10 x 600 PAIRED CONFIDENCE INTERVAL PIPELINE")
    print(f"  Folds to Benchmark:  {len(fold_files)} folds (600 scenarios/fold)")
    print(f"  Total Runs:          {len(fold_files) * 600 * 2:,} simulations (6,000 PID + 6,000 RRL)")
    print(f"  PID Gains:           {pid_gains_path.relative_to(PROJECT_ROOT) if pid_gains_path.is_relative_to(PROJECT_ROOT) else pid_gains_path}")
    print(f"  RRL Model:           {model_path.relative_to(PROJECT_ROOT) if model_path.is_relative_to(PROJECT_ROOT) else model_path}")
    print(f"  RRL Tag:             {rrl_ctrl_tag}")
    print(f"  Parallel Workers:    {args.workers}")
    print("=" * 95)

    start_all = time.perf_counter()

    # 1. PID Fold Results (check existing or compute)
    fold_results_pid = []
    pid_missing_tasks = []
    for idx, fold_file in enumerate(fold_files, 1):
        pid_json_expected = output_dir / f"benchmark_results_{fold_file.stem}_pid_tuned_kalman_empirical_d5.json"
        if pid_json_expected.exists():
            print(f"---> [PID FOLD {idx:02d}/10] Found existing PID results: {pid_json_expected.name}")
            with open(pid_json_expected, "r", encoding="utf-8") as f:
                fold_results_pid.append(json.load(f))
        else:
            pid_missing_tasks.append((idx, fold_file, pid_json_expected))

    if pid_missing_tasks:
        print(f"\n[INFO] Running {len(pid_missing_tasks)} missing PID folds with {args.workers} workers...")
        worker_args = [
            (f_file, "pid", None, pid_gains_path, output_dir, args.tau_yaw)
            for _, f_file, _ in pid_missing_tasks
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pool.map(_eval_fold_worker, worker_args)
        # Reload all
        fold_results_pid = []
        for idx, fold_file in enumerate(fold_files, 1):
            pid_json_expected = output_dir / f"benchmark_results_{fold_file.stem}_pid_tuned_kalman_empirical_d5.json"
            with open(pid_json_expected, "r", encoding="utf-8") as f:
                fold_results_pid.append(json.load(f))

    # 2. RRL Fold Results (check existing or compute)
    rrl_missing_tasks = []
    for idx, fold_file in enumerate(fold_files, 1):
        rrl_json_expected = output_dir / f"benchmark_results_{fold_file.stem}_{rrl_ctrl_tag}_kalman_empirical_d5{tau_suffix}.json"
        if rrl_json_expected.exists():
            print(f"---> [RRL FOLD {idx:02d}/10] Found existing RRL results: {rrl_json_expected.name}")
        else:
            rrl_missing_tasks.append((idx, fold_file, rrl_json_expected))

    if rrl_missing_tasks:
        print(f"\n[INFO] Evaluating {len(rrl_missing_tasks)} missing RRL folds using {args.workers} parallel workers...")
        worker_args = [
            (f_file, "residual", model_path, pid_gains_path, output_dir, args.tau_yaw)
            for _, f_file, _ in rrl_missing_tasks
        ]
        t_rrl_start = time.perf_counter()
        if args.workers > 1 and len(worker_args) > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(_eval_fold_worker, worker_args))
        else:
            for w_arg in worker_args:
                _eval_fold_worker(w_arg)
        print(f"[INFO] Completed {len(rrl_missing_tasks)} RRL folds in {(time.perf_counter() - t_rrl_start)/60:.1f} minutes!")

    # Reload all 10 RRL fold results
    fold_results_rrl = []
    for idx, fold_file in enumerate(fold_files, 1):
        rrl_json_expected = output_dir / f"benchmark_results_{fold_file.stem}_{rrl_ctrl_tag}_kalman_empirical_d5{tau_suffix}.json"
        if not rrl_json_expected.exists():
            # Try without tau suffix if not found
            fallback = output_dir / f"benchmark_results_{fold_file.stem}_{rrl_ctrl_tag}_kalman_empirical_d5.json"
            if fallback.exists():
                rrl_json_expected = fallback
        with open(rrl_json_expected, "r", encoding="utf-8") as f:
            fold_results_rrl.append(json.load(f))

    total_time = time.perf_counter() - start_all
    print(f"\n[DONE] All 10 folds (12,000 runs) evaluated in {total_time/60:.1f} minutes!")

    # =========================================================================
    # STATISTICAL SYNTHESIS ACROSS 10 FOLDS
    # =========================================================================
    profiles = ["cruising", "evasive", "hyper_evasive"]

    # Align Time-to-Success to use raw time_to_lock without hardcoded offsets
    for f in fold_results_pid + fold_results_rrl:
        if "results" in f:
            ttls_2s = [float(r["time_to_lock_2s"]) for r in f["results"] if r.get("time_to_lock_2s") is not None]
            ttls_1s = [float(r["time_to_lock_1s"]) for r in f["results"] if r.get("time_to_lock_1s") is not None]
            f["overall"]["mean_time_to_lock_2s"] = float(np.mean(ttls_2s)) if ttls_2s else None
            f["overall"]["mean_time_to_lock_1s"] = float(np.mean(ttls_1s)) if ttls_1s else None

            for prof in profiles:
                p_sub_2s = [float(r["time_to_lock_2s"]) for r in f["results"] if r["profile"] == prof and r.get("time_to_lock_2s") is not None]
                p_sub_1s = [float(r["time_to_lock_1s"]) for r in f["results"] if r["profile"] == prof and r.get("time_to_lock_1s") is not None]
                prof_entry = next((item for item in f["profile_summaries"] if item["profile"] == prof), None)
                if prof_entry:
                    prof_entry["mean_time_to_lock_2s"] = float(np.mean(p_sub_2s)) if p_sub_2s else None
                    prof_entry["mean_time_to_lock_1s"] = float(np.mean(p_sub_1s)) if p_sub_1s else None

    metrics_to_track = [
        ("success_rate_2s_pct", "2.0s Dwell Catch (%)", "{:.1f}%"),
        ("success_rate_1s_pct", "1.0s Dwell Catch (%)", "{:.1f}%"),
        ("success_rate_3s_pct", "3.0s Dwell Catch (%)", "{:.1f}%"),
        ("mean_time_to_lock_2s", "Time-to-Success (2.0s Dwell) (s)", "{:.2f}s"),
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
    report_json_path = output_dir / f"{args.report_name}.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    # Generate Markdown Table Report
    report_md_path = output_dir / f"{args.report_name}.md"
    rrl_label = rrl_ctrl_tag.replace("_", " ").title()
    md_lines = [
        f"# Controller Comparison with 95% Confidence Intervals (10 x 600 Trajectories: Tuned PID vs {rrl_label})",
        f"**Sample Size**: 10 folds x 600 scenarios = 6,000 trajectories per controller (12,000 total runs)  ",
        f"**Method**: Paired Student's t 95% Confidence Intervals & Two-Tailed Significance Testing  ",
        f"**RRL Model**: `{model_path.name}` (`{rrl_ctrl_tag}`)  ",
        f"**PID Base**: `{pid_gains_path.name}`  ",
        "",
        "## 1. Overall Aggregate Metrics (6,000 Trajectories)",
        "",
        f"| Evaluation Metric | Tuned PID Baseline (95% CI) | {rrl_label} (95% CI) | Paired Difference (Δ, 95% CI) | p-value | Significance |",
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
        f"| Target Profile | Tuned PID Baseline (95% CI) | {rrl_label} (95% CI) | Paired Difference (Δ, 95% CI) | p-value |",
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

    md_lines.extend([
        "",
        "## 3. Profile Breakdown (Time-to-Success for 2.0s Dwell, including 2s dwell)",
        "",
        f"| Target Profile | Tuned PID Baseline (95% CI) | {rrl_label} (95% CI) | Paired Difference (Δ, 95% CI) | p-value |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ])

    for prof in profiles:
        pr_ttl = analysis["by_profile"][prof]["mean_time_to_lock_2s"]
        p_str = f"{pr_ttl['pid_mean']:.2f}s [{pr_ttl['pid_ci_95'][0]:.2f}s, {pr_ttl['pid_ci_95'][1]:.2f}s]"
        r_str = f"{pr_ttl['rrl_mean']:.2f}s [{pr_ttl['rrl_ci_95'][0]:.2f}s, {pr_ttl['rrl_ci_95'][1]:.2f}s]"
        d_sign = "+" if pr_ttl["paired_delta_mean"] >= 0 else ""
        d_str = f"{d_sign}{pr_ttl['paired_delta_mean']:.2f}s [{pr_ttl['paired_delta_ci_95'][0]:.2f}s, {pr_ttl['paired_delta_ci_95'][1]:.2f}s]"
        p_val = pr_ttl["p_value"]
        p_str_val = f"< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
        md_lines.append(f"| **{prof.capitalize()}** | {p_str} | **{r_str}** | {d_str} | {p_str_val} |")

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
