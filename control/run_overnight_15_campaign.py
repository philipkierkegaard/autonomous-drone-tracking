#!/usr/bin/env python3
"""
Overnight 15-Run Iterative Reward Shaping Campaign.
Trains 15 sequential RecurrentPPO models (1,500,000 steps each, ~27 min/run)
warm-started from cloned expert weights (control/weights/cloned_expert_v1/actor_weights.pt).

Key Reward Design Principles (as instructed by user):
1. Reaching target bucket (2.5m - 4.5m) as fast as possible.
2. Reverse-exponential arrival jackpot: awarded once per episode upon first bucket entry.
3. Continuous distance pull peaking at 3.5m standoff.
4. Reward blast inside bucket escalating with dwell towards 3.0s continuous lock.
5. Mild collision penalty (p_collision = 0.0 to -2.0) to prevent cautious hang-back behaviour.
6. Iterative evaluation & dynamic adaptation after each run.
"""

import sys
import os
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"
if not PYTHON_BIN.exists():
    PYTHON_BIN = Path(sys.executable)

from control.run_1m5_sweep import evaluate_policy_detailed

CONFIG_PATH = PROJECT_ROOT / "control" / "current_expressive_reward.json"
HISTORY_PATH = PROJECT_ROOT / "outputs" / "benchmarks" / "overnight_expressive_campaign_history.json"
REPORT_PATH = PROJECT_ROOT / "docs" / "OVERNIGHT_ITERATIVE_1M5_REWARD_REPORT.md"
WEIGHTS_BASE = PROJECT_ROOT / "control" / "weights" / "overnight_expressive"
LOGS_BASE = PROJECT_ROOT / "outputs" / "benchmarks" / "overnight_expressive"
VAL_SUITE_PATH = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "validation_suite_30.json"
WARM_START_WEIGHTS = PROJECT_ROOT / "control" / "weights" / "cloned_expert_v1" / "actor_weights.pt"


# Pre-defined candidate roadmaps (which can also be dynamically adjusted based on telemetry)
INITIAL_CANDIDATES = [
    {
        "name": "run_01_pure_bucket_blast",
        "description": "Baseline Expressive: Zero collision penalty, R_arr=120, tau=4.5, blast_base=15.0, dwell_scale=6.0",
        "hyper_weight": 0.45,
        "config": {
            "r_arrival": 120.0,
            "tau_arrival": 4.5,
            "dist_peak": 8.0,
            "dist_sigma": 1.5,
            "dist_pull": 0.35,
            "blast_base": 15.0,
            "dwell_scale": 6.0,
            "p_collision": 0.0,
            "p_ground": -20.0,
            "pn_lead_weight": 2.0,
            "collision_dist": 1.0,
        }
    },
    {
        "name": "run_02_turbo_sprint_arrival",
        "description": "Fast-Arrival Turbo: tau=3.5, R_arr=160.0 to aggressively penalize sluggish closing times",
        "hyper_weight": 0.45,
        "config": {
            "r_arrival": 160.0,
            "tau_arrival": 3.5,
            "dist_peak": 8.0,
            "dist_sigma": 1.5,
            "dist_pull": 0.50,
            "blast_base": 15.0,
            "dwell_scale": 6.0,
            "p_collision": 0.0,
            "p_ground": -20.0,
            "pn_lead_weight": 2.0,
            "collision_dist": 1.0,
        }
    },
    {
        "name": "run_03_massive_bucket_magnet",
        "description": "Bucket Magnet: blast_base=25.0, dwell_scale=10.0 to lock chaser like glue once inside 2.5-4.5m",
        "hyper_weight": 0.45,
        "config": {
            "r_arrival": 130.0,
            "tau_arrival": 4.0,
            "dist_peak": 8.0,
            "dist_sigma": 1.5,
            "dist_pull": 0.35,
            "blast_base": 25.0,
            "dwell_scale": 10.0,
            "p_collision": 0.0,
            "p_ground": -20.0,
            "pn_lead_weight": 2.5,
            "collision_dist": 1.0,
        }
    },
    {
        "name": "run_04_feather_cushion_tap",
        "description": "Feather Cushion: p_collision=-1.0, collision_dist=0.9m to test gentle tap vs overshoot",
        "hyper_weight": 0.45,
        "config": {
            "r_arrival": 140.0,
            "tau_arrival": 4.0,
            "dist_peak": 8.0,
            "dist_sigma": 1.5,
            "dist_pull": 0.40,
            "blast_base": 20.0,
            "dwell_scale": 8.0,
            "p_collision": -1.0,
            "p_ground": -20.0,
            "pn_lead_weight": 2.0,
            "collision_dist": 0.9,
        }
    },
    {
        "name": "run_05_proportional_nav_apex",
        "description": "High PN Lead (w_pn=4.0): Antennas velocity vector to slice turns and maintain lock on evasive bursts",
        "hyper_weight": 0.50,
        "config": {
            "r_arrival": 140.0,
            "tau_arrival": 4.0,
            "dist_peak": 8.0,
            "dist_sigma": 1.5,
            "dist_pull": 0.40,
            "blast_base": 20.0,
            "dwell_scale": 8.0,
            "p_collision": 0.0,
            "p_ground": -20.0,
            "pn_lead_weight": 4.0,
            "collision_dist": 1.0,
        }
    },
    {
        "name": "run_06_steep_distance_suction",
        "description": "Steep Suction: dist_pull=0.65, dist_peak=10.0 to create a steep potential funnel toward 3.5m",
        "hyper_weight": 0.45,
        "config": {
            "r_arrival": 140.0,
            "tau_arrival": 4.0,
            "dist_peak": 10.0,
            "dist_sigma": 1.6,
            "dist_pull": 0.65,
            "blast_base": 20.0,
            "dwell_scale": 8.0,
            "p_collision": 0.0,
            "p_ground": -20.0,
            "pn_lead_weight": 2.5,
            "collision_dist": 1.0,
        }
    },
    {
        "name": "run_07_crescendo_dwell_escalator",
        "description": "Crescendo Dwell: blast_base=30.0, dwell_scale=12.0 for exponential reward ramp towards 3.0s mark",
        "hyper_weight": 0.45,
        "config": {
            "r_arrival": 130.0,
            "tau_arrival": 4.2,
            "dist_peak": 8.0,
            "dist_sigma": 1.5,
            "dist_pull": 0.40,
            "blast_base": 30.0,
            "dwell_scale": 12.0,
            "p_collision": 0.0,
            "p_ground": -20.0,
            "pn_lead_weight": 2.5,
            "collision_dist": 1.0,
        }
    },
    {
        "name": "run_08_deep_basket_08m_clearance",
        "description": "Deep Clearance: collision_dist=0.8m giving 1.7m deep operating pocket inside bucket",
        "hyper_weight": 0.50,
        "config": {
            "r_arrival": 140.0,
            "tau_arrival": 3.8,
            "dist_peak": 9.0,
            "dist_sigma": 1.5,
            "dist_pull": 0.45,
            "blast_base": 22.0,
            "dwell_scale": 9.0,
            "p_collision": 0.0,
            "p_ground": -20.0,
            "pn_lead_weight": 3.0,
            "collision_dist": 0.8,
        }
    },
    {
        "name": "run_09_hyper_evasive_dogfighter",
        "description": "Hyper-Evasive Master: 55% hyper training, high PN lead + massive dwell bonus",
        "hyper_weight": 0.55,
        "config": {
            "r_arrival": 150.0,
            "tau_arrival": 3.6,
            "dist_peak": 9.0,
            "dist_sigma": 1.5,
            "dist_pull": 0.50,
            "blast_base": 25.0,
            "dwell_scale": 10.0,
            "p_collision": 0.0,
            "p_ground": -20.0,
            "pn_lead_weight": 3.5,
            "collision_dist": 0.9,
        }
    },
    {
        "name": "run_10_balanced_elite_synthesis",
        "description": "Balanced Elite Synthesis combining top characteristics of runs 1-9",
        "hyper_weight": 0.50,
        "config": {
            "r_arrival": 140.0,
            "tau_arrival": 3.8,
            "dist_peak": 9.0,
            "dist_sigma": 1.5,
            "dist_pull": 0.45,
            "blast_base": 24.0,
            "dwell_scale": 9.0,
            "p_collision": 0.0,
            "p_ground": -20.0,
            "pn_lead_weight": 3.0,
            "collision_dist": 0.9,
        }
    },
    {
        "name": "run_11_adaptive_refinement_1",
        "description": "Adaptive tuning based on telemetry from runs 1-10",
        "hyper_weight": 0.50,
        "config": {
            "r_arrival": 150.0,
            "tau_arrival": 3.5,
            "dist_peak": 9.0,
            "dist_sigma": 1.5,
            "dist_pull": 0.50,
            "blast_base": 26.0,
            "dwell_scale": 10.0,
            "p_collision": 0.0,
            "p_ground": -20.0,
            "pn_lead_weight": 3.5,
            "collision_dist": 0.9,
        }
    },
    {
        "name": "run_12_adaptive_refinement_2",
        "description": "Adaptive tuning focused on maximizing 3.0s continuous lock completion",
        "hyper_weight": 0.50,
        "config": {
            "r_arrival": 150.0,
            "tau_arrival": 3.5,
            "dist_peak": 10.0,
            "dist_sigma": 1.5,
            "dist_pull": 0.55,
            "blast_base": 28.0,
            "dwell_scale": 11.0,
            "p_collision": 0.0,
            "p_ground": -20.0,
            "pn_lead_weight": 3.5,
            "collision_dist": 0.85,
        }
    },
    {
        "name": "run_13_high_velocity_match",
        "description": "High Velocity Lead & Dynamic Decel Tuning",
        "hyper_weight": 0.52,
        "config": {
            "r_arrival": 160.0,
            "tau_arrival": 3.2,
            "dist_peak": 10.0,
            "dist_sigma": 1.4,
            "dist_pull": 0.60,
            "blast_base": 30.0,
            "dwell_scale": 12.0,
            "p_collision": 0.0,
            "p_ground": -20.0,
            "pn_lead_weight": 4.0,
            "collision_dist": 0.85,
        }
    },
    {
        "name": "run_14_peak_lock_pursuit",
        "description": "Peak formation hold tuning with wide basket dwell acceleration",
        "hyper_weight": 0.50,
        "config": {
            "r_arrival": 150.0,
            "tau_arrival": 3.5,
            "dist_peak": 9.5,
            "dist_sigma": 1.5,
            "dist_pull": 0.55,
            "blast_base": 32.0,
            "dwell_scale": 14.0,
            "p_collision": 0.0,
            "p_ground": -20.0,
            "pn_lead_weight": 3.8,
            "collision_dist": 0.85,
        }
    },
    {
        "name": "run_15_grand_champion_convergence",
        "description": "Final convergence run unifying all optimal parameters from campaign",
        "hyper_weight": 0.50,
        "config": {
            "r_arrival": 160.0,
            "tau_arrival": 3.4,
            "dist_peak": 10.0,
            "dist_sigma": 1.5,
            "dist_pull": 0.55,
            "blast_base": 35.0,
            "dwell_scale": 15.0,
            "p_collision": 0.0,
            "p_ground": -20.0,
            "pn_lead_weight": 4.0,
            "collision_dist": 0.85,
        }
    }
]


def load_history() -> List[Dict[str, Any]]:
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_history(history: List[Dict[str, Any]]):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


def write_leaderboard_report(history: List[Dict[str, Any]], total_elapsed_s: float):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Overnight 15-Run Iterative Reward Shaping Campaign Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "**Campaign Type:** Autonomous Iterative RL Loop (`/goal`) with Warm-Started RecurrentPPO",
        "**Warm-Start Baseline:** Cloned Expert (`control/weights/cloned_expert_v1/actor_weights.pt`)",
        "**Timesteps per Candidate:** 1,500,000 steps (~27.5 minutes on 8 vectorized environments)",
        "**Total Completed Runs:** " + f"{len(history)} / 15",
        f"**Total Campaign Runtime:** {total_elapsed_s / 3600.0:.2f} hours ({total_elapsed_s / 60.0:.1f} mins)",
        "",
        "### User Reward Objectives & Design Constraints:",
        "1. **Fast-Arrival Jackpot:** Reverse-exponential arrival reward $R_{\\text{arrival}} \\cdot \\exp(-t_{\\text{arrival}} / \\tau)$ awarded once upon first bucket entry ($2.5\\text{ m} \\le d \\le 4.5\\text{ m}$).",
        "2. **Continuous Distance Pull:** Peak reward at $3.5\\text{ m}$ pulling chaser smoothly inward.",
        "3. **Reward Blast in Bucket:** Dense rewards ($B_{\\text{base}} + S_{\\text{dwell}} \\cdot t^2$) while inside bucket.",
        "4. **No Cautious Hang-Back:** Collision penalty deliberately kept mild ($P_{\\text{collision}} = 0.0$ to $-1.0$) so agent never becomes overly cautious or hesitates to close.",
        "5. **Hold 3 Consecutive Seconds:** Interception lock terminal jackpot ($+120$ to $+220$) awarded at 3.0s unbroken dwell.",
        "",
        "---",
        "",
        "## 1. Candidate Leaderboard (Standardized 30-Scenario Validation Suite)",
        "",
        "| Rank | Run Name | Hyper FOV% | Mean Dist | Min Dist | Turn Speed | Straight Speed | Continuous Dwell | Success Rate (≥3.0s) | Collisions |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    # Sort primarily by success rate, then max continuous dwell, then hyper FOV%
    sorted_history = sorted(
        history,
        key=lambda x: (
            x["metrics"].get("hyper_success_rate", 0.0),
            x["metrics"].get("hyper_continuous_dwell", 0.0),
            x["metrics"].get("hyper_mean_fov", 0.0)
        ),
        reverse=True
    )

    for rank, item in enumerate(sorted_history, 1):
        m = item["metrics"]
        lines.append(
            f"| **{rank}** | `{item['name']}` | **{m.get('hyper_mean_fov', 0.0):.1f}%** | "
            f"{m.get('hyper_mean_dist', 99.0):.2f} m | {m.get('hyper_min_dist', 99.0):.2f} m | "
            f"{m.get('hyper_min_speed', 0.0):.1f} m/s | {m.get('hyper_max_speed', 0.0):.1f} m/s | "
            f"**{m.get('hyper_continuous_dwell', 0.0):.2f} s** | **{m.get('hyper_success_rate', 0.0):.1f}%** | "
            f"{m.get('hyper_collisions', 0)} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Iteration Chronology & Telemetry Breakdown",
        ""
    ])

    for item in history:
        cfg = item["config"]
        m = item["metrics"]
        lines.extend([
            f"### `{item['name']}` (Iteration {item['iteration']})",
            f"- **Hypothesis:** {item['description']}",
            f"- **Reward Parameters:**",
            f"  - Arrival Jackpot: $R_0 = {cfg.get('r_arrival', 120.0)}$, $\\tau = {cfg.get('tau_arrival', 4.5)}\\text{{ s}}$",
            f"  - Continuous Pull: $\\text{{Peak}} = {cfg.get('dist_peak', 8.0)}$ at $3.5\\text{{ m}}$, $\\text{{Pull}} = {cfg.get('dist_pull', 0.35)}$",
            f"  - Bucket Blast: $\\text{{Base}} = {cfg.get('blast_base', 15.0)}$, $\\text{{Dwell Scale}} = {cfg.get('dwell_scale', 6.0)}$",
            f"  - Collision Safety: $P_{{\\text{{collision}}}} = {cfg.get('p_collision', 0.0)}$, Clearance Radius $= {cfg.get('collision_dist', 1.0)}\\text{{ m}}$ (Non-punitive; no cautious hang-back)",
            f"  - Proportional Nav Lead: $w_{{\\text{{pn}}}} = {cfg.get('pn_lead_weight', 2.0)}$",
            f"- **Validation Performance:**",
            f"  - Hyper FOV Retention: **{m.get('hyper_mean_fov', 0.0):.1f}%**",
            f"  - Standoff Distance: Mean={m.get('hyper_mean_dist', 99.0):.2f}m, Min={m.get('hyper_min_dist', 99.0):.2f}m",
            f"  - Peak Continuous Dwell: **{m.get('hyper_continuous_dwell', 0.0):.2f} s** (Success Rate: **{m.get('hyper_success_rate', 0.0):.1f}%**)",
            f"  - Speed Envelope: {m.get('hyper_min_speed', 0.0):.1f} m/s (turns) $\\longleftrightarrow$ {m.get('hyper_max_speed', 0.0):.1f} m/s (straights)",
            f"  - Collisions: {m.get('hyper_collisions', 0)}",
            f"- **Checkpoint Path:** `{item.get('model_path', '')}`",
            ""
        ])

    lines.extend([
        "---",
        "",
        "## 3. Morning Analysis & Next Actionable Recommendations",
        ""
    ])

    if sorted_history:
        best = sorted_history[0]
        bm = best["metrics"]
        lines.extend([
            f"**Best Model to Date:** `{best['name']}`",
            f"- **Success Rate:** **{bm.get('hyper_success_rate', 0.0):.1f}%**",
            f"- **Max Continuous Formation Dwell:** **{bm.get('hyper_continuous_dwell', 0.0):.2f} s**",
            f"- **Mean Distance to Target:** **{bm.get('hyper_mean_dist', 99.0):.2f} m**",
            f"- **Hyper-Evasive FOV Retention:** **{bm.get('hyper_mean_fov', 0.0):.1f}%**",
            "",
            "To deploy or benchmark the best policy interactively:",
            "```bash",
            f".venv/bin/python control/predictive_controller.py --weights {best.get('model_path', '')}",
            "```"
        ])

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[REPORT] Updated report written to: {REPORT_PATH}")


def run_single_iteration(idx: int, cand: Dict[str, Any]) -> Dict[str, Any]:
    name = cand["name"]
    cfg = cand["config"]
    hyper_w = cand.get("hyper_weight", 0.45)
    out_dir = WEIGHTS_BASE / name
    log_file = LOGS_BASE / f"{name}.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"  LAUNCHING RUN {idx + 1}/15: {name}")
    print(f"  Description: {cand['description']}")
    print(f"  Collision Penalty: {cfg.get('p_collision')} (Safe/non-cautious)")
    print("=" * 80)

    # 1. Write expressive reward configuration
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[CONFIG] Wrote expressive config to {CONFIG_PATH}")

    # 2. Build training command
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
        "--seed", str(42 + idx),
        "--warm-start-weights", str(WARM_START_WEIGHTS),
        "--warm-start-log-std", "-0.8",
        "--output-dir", str(out_dir),
        "--reward-mode", "expressive_iterative",
        "--hyper-weight", str(hyper_w),
        "--wandb-run-name", f"overnight_expressive_{name}",
        "--wandb-notes", cand["description"],
    ]

    t0 = time.time()
    with open(log_file, "w", encoding="utf-8") as lf:
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
    print(f"[RUN {idx + 1}] Finished training in {elapsed_s / 60.0:.1f} minutes with exit code {proc.returncode}")

    # 3. Locate best model weights for evaluation
    model_weights = out_dir / "best_model" / "actor_weights.pt"
    if not model_weights.exists():
        model_weights = out_dir / "actor_weights.pt"
    if not model_weights.exists():
        model_weights = out_dir / "best_model" / "best_model.zip"

    # 4. Evaluate against validation suite
    print(f"[EVAL] Evaluating candidate policy: {model_weights}")
    metrics = evaluate_policy_detailed(model_weights, VAL_SUITE_PATH)
    print(f"[METRICS] Hyper FOV: {metrics.get('hyper_mean_fov', 0.0):.1f}%, Mean Dist: {metrics.get('hyper_mean_dist', 99.0):.2f}m, Max Dwell: {metrics.get('hyper_continuous_dwell', 0.0):.2f}s, Success: {metrics.get('hyper_success_rate', 0.0):.1f}%")

    record = {
        "iteration": idx + 1,
        "name": name,
        "description": cand["description"],
        "hyper_weight": hyper_w,
        "config": cfg,
        "elapsed_s": elapsed_s,
        "metrics": metrics,
        "model_path": str(model_weights)
    }
    return record


def adapt_candidate(prev_record: Dict[str, Any], next_cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dynamic feedback rule: adapts next_cand's reward parameters based on prev_record's telemetry.
    """
    m = prev_record["metrics"]
    cfg = next_cand["config"].copy()

    # Rule A: If drone is staying too far away (>8m), intensify arrival jackpot and distance pull
    if m.get("hyper_mean_dist", 99.0) > 8.0:
        cfg["r_arrival"] = min(200.0, cfg.get("r_arrival", 120.0) * 1.15)
        cfg["dist_pull"] = min(0.80, cfg.get("dist_pull", 0.35) + 0.10)
        cfg["tau_arrival"] = max(2.5, cfg.get("tau_arrival", 4.5) * 0.90)

    # Rule B: If drone enters bucket but continuous dwell is short (<1.5s), boost bucket dwell rewards
    if m.get("hyper_continuous_dwell", 0.0) < 1.5:
        cfg["blast_base"] = min(40.0, cfg.get("blast_base", 15.0) + 5.0)
        cfg["dwell_scale"] = min(18.0, cfg.get("dwell_scale", 6.0) + 2.0)
        cfg["pn_lead_weight"] = min(5.0, cfg.get("pn_lead_weight", 2.0) + 0.5)

    # Rule C: If drone collides with target (>2 collisions), apply mild cushion (-0.5 to -1.0)
    # but never make it large so it never causes cautious hang-back behaviour!
    if m.get("hyper_collisions", 0) > 2:
        cfg["p_collision"] = -1.0
        cfg["collision_dist"] = 0.9
    elif m.get("hyper_collisions", 0) == 0 and cfg.get("p_collision", 0.0) < 0:
        # If zero collisions, keep collision penalty at 0.0
        cfg["p_collision"] = 0.0

    next_cand["config"] = cfg
    return next_cand


def main():
    print("=" * 80)
    print("  STARTING AUTONOMOUS OVERNIGHT 15-RUN REWARD SHAPING CAMPAIGN")
    print(f"  Target Timesteps per Run: 1,500,000 | Total Steps: 22,500,000")
    print(f"  Warm-Start Source: {WARM_START_WEIGHTS}")
    print("=" * 80)

    campaign_start = time.time()
    history = load_history()
    completed_names = {item["name"] for item in history}

    for idx in range(len(INITIAL_CANDIDATES)):
        cand = INITIAL_CANDIDATES[idx]
        if cand["name"] in completed_names:
            print(f"[SKIP] Run {idx + 1} ({cand['name']}) already completed, skipping.")
            continue

        # If previous run exists in history, apply dynamic adaptation
        if history:
            cand = adapt_candidate(history[-1], cand)

        record = run_single_iteration(idx, cand)
        history.append(record)
        completed_names.add(cand["name"])
        save_history(history)

        total_elapsed = time.time() - campaign_start
        write_leaderboard_report(history, total_elapsed)

    print("\n" + "=" * 80)
    print(f"  OVERNIGHT CAMPAIGN COMPLETE! Total Time: {(time.time() - campaign_start) / 3600.0:.2f} hours")
    print(f"  Final Report saved at: {REPORT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
