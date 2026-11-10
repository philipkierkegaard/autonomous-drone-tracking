#!/usr/bin/env python3
"""
Trajectory-Level Engagement Dynamics Analysis: PID vs. RRL (v9_wide35)

Analyzes mutually successful trajectories across the standardized 10-fold benchmark (N=2,231)
to decompose why RRL requires more time to achieve 2.0s sustained dwell:
1. Time to Initial Engagement (T_initial): First reaching 2.5-4.5m standoff while in FOV.
2. Time Spent Recovering from Interrupted Attempts (T_recovery = T_success - T_initial - 2.0s).
3. Interruption frequency and root causes (Overshoot <2.5m, Lag >4.5m, FOV loss).
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import multiprocessing as mp

import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.control.benchmark_controller import (
    PursuitDataset,
    evaluate_single_run,
    create_pid_controller,
)
from control.predictive_controller import ResidualVisualServoController

# Worker global variables for multiprocessing
_GLOBAL_PID = None
_GLOBAL_RRL = None

def init_worker(model_path: str, pid_gains_path: str):
    global _GLOBAL_PID, _GLOBAL_RRL
    _GLOBAL_PID = create_pid_controller(gains_path=pid_gains_path, max_decel=6.0)
    _GLOBAL_RRL = ResidualVisualServoController(
        model_path=model_path,
        pid_gains_path=pid_gains_path,
        standoff_nominal=3.5,
        desired_bbox_size=110.0,
        w_nominal=0.0859
    )

def extract_trajectory_dynamics(res: Dict[str, Any], standoff_min: float = 2.5, standoff_max: float = 4.5) -> Dict[str, Any]:
    hist = res["history"]
    times = hist["times"]
    dist = hist["distance"]
    in_view = hist["in_view"]
    dwell = hist["dwell"]
    
    in_envelope = (dist >= standoff_min) & (dist <= standoff_max) & in_view
    
    # First timestamp entering envelope
    first_idx = np.where(in_envelope)[0]
    t_initial = float(times[first_idx[0]]) if len(first_idx) > 0 else None
    ttl_2s = res.get("time_to_lock_2s")
    
    if ttl_2s is None or t_initial is None:
        return {
            "success_2s": False,
            "t_initial": t_initial,
            "ttl_2s": ttl_2s,
            "t_post_initial": None,
            "t_recovery": None,
            "num_interruptions": None,
            "fov_loss_drops": 0,
            "overshoot_drops": 0,
            "lag_drops": 0,
            "both_drops": 0,
            "clean_catch": False
        }
        
    t_post_initial = max(2.0, float(ttl_2s - t_initial))
    t_recovery = max(0.0, float(t_post_initial - 2.0))
    
    # Analyze interruptions strictly before completion of 2s dwell
    mask_before = times <= (ttl_2s + 1e-4)
    t_sub = times[mask_before]
    env_sub = in_envelope[mask_before]
    dist_sub = dist[mask_before]
    view_sub = in_view[mask_before]
    
    interruptions = []
    in_streak = False
    streak_start = 0.0
    streak_len = 0
    
    for i in range(len(t_sub)):
        if env_sub[i]:
            if not in_streak:
                in_streak = True
                streak_start = t_sub[i]
                streak_len = 1
            else:
                streak_len += 1
        else:
            if in_streak:
                # Interruption occurred!
                streak_dur = t_sub[i] - streak_start
                # Classify drop reason based on the step that fell out
                lost_fov = not bool(view_sub[i])
                overshoot = bool(dist_sub[i] < standoff_min)
                lag = bool(dist_sub[i] > standoff_max)
                
                if lost_fov and (overshoot or lag):
                    reason = "both"
                elif lost_fov:
                    reason = "fov_loss"
                elif overshoot:
                    reason = "overshoot"
                elif lag:
                    reason = "lag"
                else:
                    reason = "unknown"
                    
                interruptions.append({
                    "start": streak_start,
                    "end": t_sub[i],
                    "duration": streak_dur,
                    "reason": reason
                })
                in_streak = False

    fov_loss_count = sum(1 for it in interruptions if it["reason"] == "fov_loss")
    overshoot_count = sum(1 for it in interruptions if it["reason"] == "overshoot")
    lag_count = sum(1 for it in interruptions if it["reason"] == "lag")
    both_count = sum(1 for it in interruptions if it["reason"] == "both")
    
    return {
        "success_2s": True,
        "t_initial": t_initial,
        "ttl_2s": ttl_2s,
        "t_post_initial": t_post_initial,
        "t_recovery": t_recovery,
        "num_interruptions": len(interruptions),
        "fov_loss_drops": fov_loss_count,
        "overshoot_drops": overshoot_count,
        "lag_drops": lag_count,
        "both_drops": both_count,
        "clean_catch": (len(interruptions) == 0),
        "interruptions_detail": interruptions
    }

def run_paired_trajectory(task_args: Tuple[Any, float, float]) -> Optional[Dict[str, Any]]:
    scenario, frame_drop_rate, tau_yaw = task_args
    global _GLOBAL_PID, _GLOBAL_RRL
    
    # Run PID
    res_p = evaluate_single_run(
        scenario=scenario,
        controller=_GLOBAL_PID,
        frame_drop_rate=frame_drop_rate,
        return_history=True,
    )
    
    # Fast check: if PID failed 2s dwell, no need to run RRL for mutual analysis
    if not res_p["success_2s"]:
        return None
        
    # Run RRL
    res_r = evaluate_single_run(
        scenario=scenario,
        controller=_GLOBAL_RRL,
        frame_drop_rate=frame_drop_rate,
        tau_yaw=tau_yaw,
        return_history=True,
    )
    
    if not res_r["success_2s"]:
        return None
        
    # Both succeeded! Analyze dynamics
    dyn_p = extract_trajectory_dynamics(res_p)
    dyn_r = extract_trajectory_dynamics(res_r)
    
    return {
        "scenario_id": scenario.scenario_id,
        "profile": scenario.profile,
        "encounter_type": scenario.encounter_type,
        "seed": scenario.seed,
        "pid": dyn_p,
        "rrl": dyn_r,
        "history_p": {
            "times": np.array(res_p["history"]["times"], dtype=np.float32),
            "dist": np.array(res_p["history"]["distance"], dtype=np.float32),
            "in_view": np.array(res_p["history"]["in_view"], dtype=bool),
            "dwell": np.array(res_p["history"]["dwell"], dtype=np.float32),
        },
        "history_r": {
            "times": np.array(res_r["history"]["times"], dtype=np.float32),
            "dist": np.array(res_r["history"]["distance"], dtype=np.float32),
            "in_view": np.array(res_r["history"]["in_view"], dtype=bool),
            "dwell": np.array(res_r["history"]["dwell"], dtype=np.float32),
        }
    }


def compute_paired_stats(p_vals: List[float], r_vals: List[float]) -> Dict[str, Any]:
    p = np.array(p_vals, dtype=np.float64)
    r = np.array(r_vals, dtype=np.float64)
    n = len(p)
    if n == 0:
        return {}
    diff = r - p
    mean_p = float(np.mean(p))
    mean_r = float(np.mean(r))
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    sem_diff = std_diff / np.sqrt(n) if n > 1 else 0.0
    
    # 95% CI
    tcrit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else 1.96
    ci_lower = mean_diff - tcrit * sem_diff
    ci_upper = mean_diff + tcrit * sem_diff
    
    t_stat, p_val = (stats.ttest_rel(r, p) if n > 1 else (0.0, 1.0))
    
    return {
        "n": n,
        "mean_pid": mean_p,
        "mean_rrl": mean_r,
        "delta": mean_diff,
        "sem": sem_diff,
        "ci_95": (float(ci_lower), float(ci_upper)),
        "ci_str": f"{mean_diff:+.2f} [{ci_lower:+.2f}, {ci_upper:+.2f}]",
        "t_stat": float(t_stat),
        "p_val": float(p_val)
    }


def render_trajectory_comparison_plot(
    results: List[Dict[str, Any]],
    output_path: Path
):
    """
    Renders publication-grade 3-panel comparative figures for representative trajectories:
    Panel 1: Target Separation d(t) [m] with [2.5, 4.5m] engagement basket
    Panel 2: Target Visibility in_view(t) (binary step)
    Panel 3: Continuous Dwell Accumulator t_dwell(t) [s] with 2.0s threshold
    """
    # Find two compelling representative trajectories:
    # 1. Clean Sprint: Cruising or steady where both succeed with 0 interruptions, showing RRL's conservative entry delay
    # 2. Evasive Interrupted Dwell: Evasive scenario where PID overshoots/loses FOV causing resets, or where RRL recovers
    
    candidates_clean = [
        r for r in results 
        if r["profile"] in ["cruising", "steady_slow"] 
        and r["pid"]["clean_catch"] and r["rrl"]["clean_catch"]
        and (r["rrl"]["ttl_2s"] - r["pid"]["ttl_2s"] > 1.0)
    ]
    
    candidates_interrupted = [
        r for r in results 
        if r["profile"] in ["evasive", "hyper_evasive"] 
        and (r["pid"]["num_interruptions"] > 0 or r["rrl"]["num_interruptions"] > 0)
        and abs(r["rrl"]["ttl_2s"] - r["pid"]["ttl_2s"]) > 1.5
    ]
    
    # Pick best representatives
    clean_case = candidates_clean[0] if candidates_clean else results[0]
    interrupted_case = candidates_interrupted[0] if candidates_interrupted else results[1]
    
    cases = [
        (clean_case, "Case A: Clean Tracking (Cruising)\nSustained Lock Without Interruptions"),
        (interrupted_case, "Case B: High-Agility Evasion\nEngagement Interruption & Dwell Resets")
    ]
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex="col", dpi=300)
    
    c_pid = "#1f77b4"  # Blue
    c_rrl = "#d62728"  # Crimson Red
    c_basket = "#2ca02c" # Green
    
    for col_idx, (case, title) in enumerate(cases):
        t_max = max(case["pid"]["ttl_2s"], case["rrl"]["ttl_2s"]) + 2.5
        t_max = min(t_max, 25.0)
        
        hp = case["history_p"]
        hr = case["history_r"]
        
        mask_p = hp["times"] <= t_max
        mask_r = hr["times"] <= t_max
        
        tp = hp["times"][mask_p]
        tr = hr["times"][mask_r]
        
        dp = hp["dist"][mask_p]
        dr = hr["dist"][mask_r]
        
        vp = hp["in_view"][mask_p]
        vr = hr["in_view"][mask_r]
        
        wp = hp["dwell"][mask_p]
        wr = hr["dwell"][mask_r]
        
        # ----------------------------------------------------
        # Panel 1: Distance d(t)
        # ----------------------------------------------------
        ax0 = axes[0, col_idx]
        ax0.set_title(title, fontsize=12, fontweight="bold", pad=12)
        
        # Basket shaded region
        ax0.axhspan(2.5, 4.5, color=c_basket, alpha=0.15, label="Engagement Basket [2.5, 4.5 m]")
        ax0.axhline(3.5, color=c_basket, linestyle=":", alpha=0.7, label="Nominal Standoff (3.5 m)")
        ax0.axhline(2.5, color=c_basket, linestyle="--", alpha=0.5, linewidth=0.8)
        ax0.axhline(4.5, color=c_basket, linestyle="--", alpha=0.5, linewidth=0.8)
        
        # Controller trajectories
        ax0.plot(tp, dp, color=c_pid, linewidth=2.0, label=f"Kinematic PID (T_init={case['pid']['t_initial']:.2f}s)")
        ax0.plot(tr, dr, color=c_rrl, linewidth=2.0, linestyle="--", label=f"RRL v9 (T_init={case['rrl']['t_initial']:.2f}s)")
        
        # Mark initial engagement and success
        ax0.scatter([case["pid"]["t_initial"]], [dp[np.argmin(np.abs(tp - case["pid"]["t_initial"]))]], 
                    color=c_pid, marker="o", s=60, zorder=5)
        ax0.scatter([case["rrl"]["t_initial"]], [dr[np.argmin(np.abs(tr - case["rrl"]["t_initial"]))]], 
                    color=c_rrl, marker="s", s=60, zorder=5)
        
        ax0.set_ylabel("Target Distance [m]", fontsize=10, fontweight="bold")
        ax0.set_ylim(1.5, max(7.0, np.max(dp[:int(3.0/0.02)]) + 1.0))
        ax0.grid(True, linestyle="--", alpha=0.5)
        if col_idx == 0:
            ax0.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
            
        # ----------------------------------------------------
        # Panel 2: Visibility in_view
        # ----------------------------------------------------
        ax1 = axes[1, col_idx]
        ax1.step(tp, vp.astype(int), color=c_pid, linewidth=1.8, where="post", label="PID in FOV")
        ax1.step(tr, vr.astype(int) * 0.92, color=c_rrl, linewidth=1.8, linestyle="--", where="post", label="RRL in FOV")
        ax1.set_ylabel("Target in FOV", fontsize=10, fontweight="bold")
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(["LOST", "LOCKED"])
        ax1.set_ylim(-0.15, 1.25)
        ax1.grid(True, linestyle="--", alpha=0.5)
        if col_idx == 0:
            ax1.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
            
        # ----------------------------------------------------
        # Panel 3: Continuous Dwell Accumulator t_dwell(t)
        # ----------------------------------------------------
        ax2 = axes[2, col_idx]
        ax2.plot(tp, wp, color=c_pid, linewidth=2.0, label=f"PID Dwell (Succ @ {case['pid']['ttl_2s']:.2f}s)")
        ax2.plot(tr, wr, color=c_rrl, linewidth=2.0, linestyle="--", label=f"RRL Dwell (Succ @ {case['rrl']['ttl_2s']:.2f}s)")
        
        # 2.0s threshold line
        ax2.axhline(2.0, color="darkred", linestyle=":", linewidth=1.5, label="2.0s Dwell Threshold")
        
        # Mark 2s success points
        ax2.scatter([case["pid"]["ttl_2s"]], [2.0], color=c_pid, marker="*", s=140, zorder=6, label="PID 2s Catch")
        ax2.scatter([case["rrl"]["ttl_2s"]], [2.0], color=c_rrl, marker="*", s=140, zorder=6, label="RRL 2s Catch")
        
        # Shade recovery intervals if interruptions occurred
        if case["pid"]["num_interruptions"] > 0:
            t_init_p = case["pid"]["t_initial"]
            t_succ_p = case["pid"]["ttl_2s"]
            ax2.axvspan(t_init_p, t_succ_p - 2.0, color=c_pid, alpha=0.12, label="PID Recovery Window")
            
        if case["rrl"]["num_interruptions"] > 0:
            t_init_r = case["rrl"]["t_initial"]
            t_succ_r = case["rrl"]["ttl_2s"]
            ax2.axvspan(t_init_r, t_succ_r - 2.0, color=c_rrl, alpha=0.12, label="RRL Recovery Window")
            
        ax2.set_xlabel("Simulation Time [s]", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Continuous Dwell [s]", fontsize=10, fontweight="bold")
        ax2.set_ylim(-0.1, 2.5)
        ax2.grid(True, linestyle="--", alpha=0.5)
        if col_idx == 0:
            ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
            
        # Add scenario metadata annotation
        meta_text = (
            f"Scenario: {case['scenario_id']}\n"
            f"PID: T_init={case['pid']['t_initial']:.2f}s, T_rec={case['pid']['t_recovery']:.2f}s, Resets={case['pid']['num_interruptions']}\n"
            f"RRL: T_init={case['rrl']['t_initial']:.2f}s, T_rec={case['rrl']['t_recovery']:.2f}s, Resets={case['rrl']['num_interruptions']}"
        )
        ax2.text(0.98, 0.05, meta_text, transform=ax2.transAxes, fontsize=8,
                 verticalalignment="bottom", horizontalalignment="right",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#ccc"))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Trajectory comparison figure saved to: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze Trajectory-Level Engagement Dynamics")
    parser.add_argument("--folds", type=int, default=10, help="Number of folds to analyze (1-10)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes")
    parser.add_argument("--tau-yaw", type=float, default=0.150, help="Yaw time constant for simulation")
    parser.add_argument("--frame-drop-rate", type=float, default=0.05, help="Empirical frame drop rate")
    parser.add_argument("--output-json", type=str, default="outputs/benchmarks/control/trajectory_engagement_dynamics_10fold.json")
    parser.add_argument("--output-plot", type=str, default="/Users/philipkierkegaard/.gemini/antigravity/brain/3f6fb6d1-ed45-4315-9b5d-b3cca4e6bef7/trajectory_engagement_dynamics_comparison.png")
    args = parser.parse_args()

    model_path = PROJECT_ROOT / "outputs" / "models" / "residual_ppo_v9_wide35" / "best_model" / "best_model.zip"
    pid_gains_path = PROJECT_ROOT / "control" / "weights" / "tuned_pid_gains.json"

    print("=" * 80)
    print("  TRAJECTORY-LEVEL ENGAGEMENT DYNAMICS INVESTIGATION")
    print(f"  Folds to evaluate: {args.folds} (ci_benchmark_600_fold_01 to fold_{args.folds:02d})")
    print(f"  Parallel workers:  {args.workers}")
    print(f"  Model Path:        {model_path}")
    print(f"  Gains Path:        {pid_gains_path}")
    print("=" * 80)

    # 1. Load scenarios
    all_scenarios = []
    for f_idx in range(1, args.folds + 1):
        fpath = PROJECT_ROOT / f"outputs/benchmarks/datasets/ci_10x600/ci_benchmark_600_fold_{f_idx:02d}.json"
        if not fpath.exists():
            print(f"Warning: {fpath} does not exist, skipping.")
            continue
        ds = PursuitDataset.load_json(fpath)
        all_scenarios.extend(ds.scenarios)
    print(f"Loaded total {len(all_scenarios)} candidate scenarios across {args.folds} folds.")

    tasks = [(sc, args.frame_drop_rate, args.tau_yaw) for sc in all_scenarios]

    t0 = time.perf_counter()
    with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(str(model_path), str(pid_gains_path))) as pool:
        raw_results = pool.map(run_paired_trajectory, tasks)
    t1 = time.perf_counter()

    mutual_results = [r for r in raw_results if r is not None]
    print(f"Completed analysis in {t1 - t0:.1f} s.")
    print(f"Mutually successful scenarios: {len(mutual_results)} / {len(all_scenarios)} ({len(mutual_results)/len(all_scenarios)*100:.1f}%)")

    # Group by profile and overall
    profiles = ["all", "steady_slow", "cruising", "evasive", "hyper_evasive"]
    grouped_stats = {}

    for prof in profiles:
        if prof == "all":
            subset = mutual_results
        else:
            subset = [r for r in mutual_results if r["profile"] == prof]
            
        n = len(subset)
        if n == 0:
            continue
            
        t_init_p = [r["pid"]["t_initial"] for r in subset]
        t_init_r = [r["rrl"]["t_initial"] for r in subset]
        
        ttl_2s_p = [r["pid"]["ttl_2s"] for r in subset]
        ttl_2s_r = [r["rrl"]["ttl_2s"] for r in subset]
        
        t_post_p = [r["pid"]["t_post_initial"] for r in subset]
        t_post_r = [r["rrl"]["t_post_initial"] for r in subset]
        
        t_rec_p = [r["pid"]["t_recovery"] for r in subset]
        t_rec_r = [r["rrl"]["t_recovery"] for r in subset]
        
        n_int_p = [r["pid"]["num_interruptions"] for r in subset]
        n_int_r = [r["rrl"]["num_interruptions"] for r in subset]
        
        clean_p = [1.0 if r["pid"]["clean_catch"] else 0.0 for r in subset]
        clean_r = [1.0 if r["rrl"]["clean_catch"] else 0.0 for r in subset]
        
        # Interruption causes per trajectory
        fov_p = [r["pid"]["fov_loss_drops"] for r in subset]
        fov_r = [r["rrl"]["fov_loss_drops"] for r in subset]
        
        over_p = [r["pid"]["overshoot_drops"] for r in subset]
        over_r = [r["rrl"]["overshoot_drops"] for r in subset]
        
        lag_p = [r["pid"]["lag_drops"] for r in subset]
        lag_r = [r["rrl"]["lag_drops"] for r in subset]

        both_p = [r["pid"]["both_drops"] for r in subset]
        both_r = [r["rrl"]["both_drops"] for r in subset]
        
        grouped_stats[prof] = {
            "n": n,
            "t_initial": compute_paired_stats(t_init_p, t_init_r),
            "ttl_2s": compute_paired_stats(ttl_2s_p, ttl_2s_r),
            "t_post_initial": compute_paired_stats(t_post_p, t_post_r),
            "t_recovery": compute_paired_stats(t_rec_p, t_rec_r),
            "num_interruptions": compute_paired_stats(n_int_p, n_int_r),
            "clean_catch_rate_pct": compute_paired_stats([c * 100 for c in clean_p], [c * 100 for c in clean_r]),
            "fov_loss_drops": compute_paired_stats(fov_p, fov_r),
            "overshoot_drops": compute_paired_stats(over_p, over_r),
            "lag_drops": compute_paired_stats(lag_p, lag_r),
            "both_drops": compute_paired_stats(both_p, both_r),
        }

    # Print markdown table
    print("\n" + "=" * 100)
    print("### TABLE 1: Trajectory Engagement Decomposition on Mutually Successful Scenarios (Δ = RRL - PID)")
    print("=" * 100)
    print(f"{'Profile':<16} | {'N':<5} | {'PID T_init (s)':<14} | {'RRL T_init (s)':<14} | {'Δ T_init [95% CI]':<20} | {'PID T_rec (s)':<14} | {'RRL T_rec (s)':<14} | {'Δ T_rec [95% CI]':<20}")
    print("-" * 125)
    for prof in profiles:
        if prof not in grouped_stats:
            continue
        st = grouped_stats[prof]
        n_val = st["n"]
        ti = st["t_initial"]
        tr = st["t_recovery"]
        print(f"{prof:<16} | {n_val:<5} | {ti['mean_pid']:<14.2f} | {ti['mean_rrl']:<14.2f} | {ti['ci_str']:<20} | {tr['mean_pid']:<14.2f} | {tr['mean_rrl']:<14.2f} | {tr['ci_str']:<20}")

    print("\n" + "=" * 100)
    print("### TABLE 2: Engagement Interruptions and Clean Catch Rates (Δ = RRL - PID)")
    print("=" * 100)
    print(f"{'Profile':<16} | {'N':<5} | {'PID Interr.':<12} | {'RRL Interr.':<12} | {'Δ Interr. [95% CI]':<20} | {'PID Clean%':<10} | {'RRL Clean%':<10} | {'Δ Clean% [95% CI]':<20}")
    print("-" * 115)
    for prof in profiles:
        if prof not in grouped_stats:
            continue
        st = grouped_stats[prof]
        n_val = st["n"]
        ni = st["num_interruptions"]
        cc = st["clean_catch_rate_pct"]
        print(f"{prof:<16} | {n_val:<5} | {ni['mean_pid']:<12.2f} | {ni['mean_rrl']:<12.2f} | {ni['ci_str']:<20} | {cc['mean_pid']:<10.1f} | {cc['mean_rrl']:<10.1f} | {cc['ci_str']:<20}")

    print("\n" + "=" * 100)
    print("### TABLE 3: Root Cause Breakdown of Engagement Interruptions (Mean Count per Trajectory, Δ = RRL - PID)")
    print("=" * 100)
    print(f"{'Profile':<16} | {'FOV Loss Δ [95% CI]':<25} | {'Overshoot (<2.5m) Δ [95% CI]':<30} | {'Lag (>4.5m) Δ [95% CI]':<25}")
    print("-" * 105)
    for prof in profiles:
        if prof not in grouped_stats:
            continue
        st = grouped_stats[prof]
        fov = st["fov_loss_drops"]
        over = st["overshoot_drops"]
        lag = st["lag_drops"]
        print(f"{prof:<16} | {fov['ci_str']:<25} | {over['ci_str']:<30} | {lag['ci_str']:<25}")

    # Render visualization
    output_plot_path = Path(args.output_plot)
    render_trajectory_comparison_plot(mutual_results, output_plot_path)

    # Save json summary
    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        # Save grouped stats (stripping non-serializable)
        json.dump(grouped_stats, f, indent=2)
    print(f"\n[DONE] Saved statistical results to {output_json_path}")

if __name__ == "__main__":
    main()

