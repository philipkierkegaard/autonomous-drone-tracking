#!/usr/bin/env python3
"""
Comprehensive 200-Scenario Benchmark: Learned Controller vs. Kinematic Controller
Profiles: 100 Evasive, 100 Hyper-Evasive
Encounter Breakdown: Tail Chase, Crossing Right, Crossing Left, Head On, Oblique Away
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.dataset_generator import PursuitDatasetGenerator, PursuitScenario, PursuitDataset, EncounterType
from control.predictive_controller import RecurrentVisualServoController
from benchmarks.control.benchmark_controller import create_pid_controller, evaluate_single_run


def generate_evasive_200_dataset(output_path: Path) -> PursuitDataset:
    """Generates the 200-scenario dataset: 100 evasive + 100 hyper-evasive, broken down by 5 encounters."""
    gen = PursuitDatasetGenerator()
    scenarios = []

    profiles = ["evasive", "hyper_evasive"]
    encounters = [
        EncounterType.TAIL_CHASE,
        EncounterType.CROSSING_RIGHT,
        EncounterType.CROSSING_LEFT,
        EncounterType.HEAD_ON,
        EncounterType.OBLIQUE_AWAY,
    ]
    # 2 profiles x 5 encounters x 20 scenarios = 200 scenarios
    scenarios_per_cell = 20
    base_seed = 40000

    idx = 0
    for profile in profiles:
        for enc in encounters:
            enc_name = enc.value
            for i in range(scenarios_per_cell):
                seed = base_seed + idx
                sc_id = f"bench200_{idx+1:03d}_{profile}_{enc_name}_s{seed}"
                sc = gen.generate_scenario(
                    scenario_id=sc_id,
                    profile=profile,
                    encounter_type=enc,
                    seed=seed,
                    duration=25.0,
                    chaser_alt_m=3.0,
                )
                scenarios.append(sc)
                idx += 1

    dataset = PursuitDataset(
        name="evasive_200_benchmark_suite",
        scenarios=scenarios,
        metadata={
            "description": "Standardized 200-scenario benchmark evaluating 100 evasive and 100 hyper-evasive targets across 5 encounter trajectories.",
            "total_scenarios": len(scenarios),
            "profile_distribution": {"evasive": 100, "hyper_evasive": 100},
            "encounter_distribution": {
                "tail_chase": 40,
                "crossing_right": 40,
                "crossing_left": 40,
                "head_on": 40,
                "oblique_away": 40,
            },
            "created_timestamp": time.time(),
        }
    )
    dataset.save_json(output_path)
    return dataset


def run_evaluation(
    dataset: PursuitDataset,
    controller_type: str,
    controller_instance: Any,
    noise_mode: str = "empirical",
    frame_drop_rate: float = 0.05,
) -> List[Dict[str, Any]]:
    """Evaluates a controller instance across all scenarios in the dataset."""
    results = []
    total = len(dataset.scenarios)
    print(f"\n--- Starting Evaluation: {controller_type.upper()} ({total} scenarios) ---")
    t0 = time.perf_counter()

    for i, sc in enumerate(dataset.scenarios):
        if hasattr(controller_instance, "reset"):
            controller_instance.reset()

        r = evaluate_single_run(
            scenario=sc,
            controller=controller_instance,
            controller_type="rl" if "rl" in controller_type.lower() else "pid",
            noise_mode=noise_mode,
            frame_drop_rate=frame_drop_rate,
            standoff_nominal=3.5,
            use_kalman=True,
        )
        results.append(r)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.perf_counter() - t0
            print(f"[{controller_type.upper()}] Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%) in {elapsed:.1f}s")

    return results


def summarize_results(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes comprehensive aggregations across profiles, trajectories, and cross-tabulations."""
    n = len(runs)
    if n == 0:
        return {}

    def calc_group(sub: List[Dict[str, Any]]) -> Dict[str, Any]:
        count = len(sub)
        if count == 0:
            return {}
        succ_1s = sum(1 for r in sub if r.get("success_1s", r["success"]))
        succ_3s = sum(1 for r in sub if r.get("success_3s", False))
        dwells = [r.get("max_continuous_dwell_s", 0.0) for r in sub]
        fovs = [r["fov_lock_pct"] for r in sub]
        min_dists = [r["min_distance"] for r in sub]
        colls = sum(1 for r in sub if r["collision_breach"])
        locks = [r["time_to_lock"] for r in sub if r.get("time_to_lock") is not None]
        efforts = [r.get("control_effort", 0.0) for r in sub]

        return {
            "n": count,
            "catch_rate_1s_pct": float((succ_1s / count) * 100.0),
            "catch_rate_3s_pct": float((succ_3s / count) * 100.0),
            "mean_max_dwell_s": float(np.mean(dwells)),
            "median_max_dwell_s": float(np.median(dwells)),
            "fov_retention_pct": float(np.mean(fovs)),
            "mean_min_dist_m": float(np.mean(min_dists)),
            "collision_rate_pct": float((colls / count) * 100.0),
            "mean_time_to_lock_s": float(np.mean(locks)) if len(locks) > 0 else None,
            "mean_control_effort": float(np.mean(efforts)),
        }

    overall = calc_group(runs)

    # Breakdown by profile
    by_profile = {}
    for p in ["evasive", "hyper_evasive"]:
        sub = [r for r in runs if r["profile"] == p]
        by_profile[p] = calc_group(sub)

    # Breakdown by encounter trajectory
    enc_types = ["tail_chase", "crossing_right", "crossing_left", "head_on", "oblique_away"]
    by_encounter = {}
    for enc in enc_types:
        sub = [r for r in runs if r["encounter_type"] == enc]
        by_encounter[enc] = calc_group(sub)

    # Cross-tabulation (Profile x Encounter)
    cross_tab = {}
    for p in ["evasive", "hyper_evasive"]:
        cross_tab[p] = {}
        for enc in enc_types:
            sub = [r for r in runs if r["profile"] == p and r["encounter_type"] == enc]
            cross_tab[p][enc] = calc_group(sub)

    return {
        "overall": overall,
        "by_profile": by_profile,
        "by_encounter": by_encounter,
        "cross_tabulation": cross_tab,
    }


def main():
    print("=" * 80)
    print("  200-SCENARIO BENCHMARK: LEARNED CONTROLLER vs. KINEMATIC CONTROLLER")
    print("  Dataset: 100 Evasive + 100 Hyper-Evasive | 5 Encounter Geometries (20 each)")
    print("  Perception: 8D Kalman Filter, AR(1) Empirical Noise, 5% Random Frame Drops")
    print("=" * 80)

    dataset_path = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "evasive_200_benchmark_suite.json"
    dataset = generate_evasive_200_dataset(dataset_path)

    # 1. Evaluate Learned Controller (Cloned Expert v2)
    rl_model_path = PROJECT_ROOT / "control" / "weights" / "cloned_expert_v2" / "actor_weights.pt"
    print(f"\n[INIT] Instantiating Learned Controller from: {rl_model_path}")
    rl_controller = RecurrentVisualServoController(model_path=str(rl_model_path), w_nominal=0.0859)

    rl_runs = run_evaluation(
        dataset=dataset,
        controller_type="rl",
        controller_instance=rl_controller,
        noise_mode="empirical",
        frame_drop_rate=0.05,
    )
    rl_summary = summarize_results(rl_runs)

    # 2. Evaluate Kinematic Controller (Tuned PID)
    print(f"\n[INIT] Instantiating Tuned Kinematic PID Controller (Zero Integrals)")
    pid_controller = create_pid_controller(
        gains_path=PROJECT_ROOT / "control" / "weights" / "tuned_pid_gains.json",
        standoff_nominal=3.5,
        desired_bbox_size=110.0,
    )

    pid_runs = run_evaluation(
        dataset=dataset,
        controller_type="pid",
        controller_instance=pid_controller,
        noise_mode="empirical",
        frame_drop_rate=0.05,
    )
    pid_summary = summarize_results(pid_runs)

    # 3. Export Raw Results & Aggregations
    out_dir = PROJECT_ROOT / "outputs" / "benchmarks" / "control"
    out_dir.mkdir(parents=True, exist_ok=True)

    rl_out_path = out_dir / "benchmark_results_evasive_200_rl_cloned_expert_v2.json"
    with open(rl_out_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": str(dataset_path.relative_to(PROJECT_ROOT)),
            "controller": "rl_cloned_expert_v2",
            "model_path": str(rl_model_path.relative_to(PROJECT_ROOT)),
            "noise_mode": "empirical",
            "frame_drop_rate": 0.05,
            "summary": rl_summary,
            "results": rl_runs,
        }, f, indent=2)

    pid_out_path = out_dir / "benchmark_results_evasive_200_pid_tuned.json"
    with open(pid_out_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": str(dataset_path.relative_to(PROJECT_ROOT)),
            "controller": "pid_tuned",
            "gains_path": "control/weights/tuned_pid_gains.json",
            "noise_mode": "empirical",
            "frame_drop_rate": 0.05,
            "summary": pid_summary,
            "results": pid_runs,
        }, f, indent=2)

    # Comparative Summary JSON
    comparative = {
        "dataset": "evasive_200_benchmark_suite",
        "num_scenarios": len(dataset.scenarios),
        "overall": {
            "rl": rl_summary["overall"],
            "pid": pid_summary["overall"],
        },
        "by_profile": {
            "evasive": {
                "rl": rl_summary["by_profile"]["evasive"],
                "pid": pid_summary["by_profile"]["evasive"],
            },
            "hyper_evasive": {
                "rl": rl_summary["by_profile"]["hyper_evasive"],
                "pid": pid_summary["by_profile"]["hyper_evasive"],
            },
        },
        "by_encounter": {
            enc: {
                "rl": rl_summary["by_encounter"][enc],
                "pid": pid_summary["by_encounter"][enc],
            } for enc in ["tail_chase", "crossing_right", "crossing_left", "head_on", "oblique_away"]
        },
        "cross_tabulation": {
            p: {
                enc: {
                    "rl": rl_summary["cross_tabulation"][p][enc],
                    "pid": pid_summary["cross_tabulation"][p][enc],
                } for enc in ["tail_chase", "crossing_right", "crossing_left", "head_on", "oblique_away"]
            } for p in ["evasive", "hyper_evasive"]
        }
    }

    comp_path = out_dir / "evasive_200_comparative_summary.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparative, f, indent=2)

    print("\n" + "=" * 80)
    print("  BENCHMARK COMPLETE!")
    print(f"  Dataset:      {dataset_path}")
    print(f"  RL Results:   {rl_out_path}")
    print(f"  PID Results:  {pid_out_path}")
    print(f"  Comparison:   {comp_path}")
    print("=" * 80)

    # Print high-level comparison
    print("\n>>> OVERALL SUMMARY (200 SCENARIOS)")
    print(f"{'Metric':<28} | {'Learned (Cloned v2)':<20} | {'Kinematic (Tuned PID)':<20} | {'Delta':<12}")
    print("-" * 88)
    for k, label in [
        ("catch_rate_1s_pct", "1.0s Catch Rate (%)"),
        ("catch_rate_3s_pct", "3.0s Catch Rate (%)"),
        ("mean_max_dwell_s", "Mean Max Dwell (s)"),
        ("fov_retention_pct", "FOV Retention (%)"),
        ("collision_rate_pct", "Collision Rate (%)"),
        ("mean_min_dist_m", "Mean Min Dist (m)"),
    ]:
        v_rl = rl_summary["overall"][k]
        v_pid = pid_summary["overall"][k]
        diff = v_rl - v_pid
        sign = "+" if diff >= 0 else ""
        print(f"{label:<28} | {v_rl:<20.2f} | {v_pid:<20.2f} | {sign}{diff:<11.2f}")

    print("\n>>> PROFILE BREAKDOWN")
    for p in ["evasive", "hyper_evasive"]:
        print(f"\n--- Profile: {p.upper()} (100 scenarios) ---")
        print(f"{'Metric':<28} | {'Learned (Cloned v2)':<20} | {'Kinematic (Tuned PID)':<20} | {'Delta':<12}")
        print("-" * 88)
        for k, label in [
            ("catch_rate_1s_pct", "1.0s Catch Rate (%)"),
            ("catch_rate_3s_pct", "3.0s Catch Rate (%)"),
            ("mean_max_dwell_s", "Mean Max Dwell (s)"),
            ("fov_retention_pct", "FOV Retention (%)"),
        ]:
            v_rl = rl_summary["by_profile"][p][k]
            v_pid = pid_summary["by_profile"][p][k]
            diff = v_rl - v_pid
            sign = "+" if diff >= 0 else ""
            print(f"{label:<28} | {v_rl:<20.2f} | {v_pid:<20.2f} | {sign}{diff:<11.2f}")

    print("\n>>> ENCOUNTER TRAJECTORY BREAKDOWN (40 scenarios each)")
    print(f"{'Trajectory':<16} | {'RL 1s Catch':<12} | {'PID 1s Catch':<12} | {'RL FOV':<10} | {'PID FOV':<10} | {'RL Dwell':<10} | {'PID Dwell':<10}")
    print("-" * 90)
    for enc in ["tail_chase", "crossing_right", "crossing_left", "head_on", "oblique_away"]:
        r_c1 = rl_summary["by_encounter"][enc]["catch_rate_1s_pct"]
        p_c1 = pid_summary["by_encounter"][enc]["catch_rate_1s_pct"]
        r_fov = rl_summary["by_encounter"][enc]["fov_retention_pct"]
        p_fov = pid_summary["by_encounter"][enc]["fov_retention_pct"]
        r_dw = rl_summary["by_encounter"][enc]["mean_max_dwell_s"]
        p_dw = pid_summary["by_encounter"][enc]["mean_max_dwell_s"]
        print(f"{enc:<16} | {r_c1:>10.1f}% | {p_c1:>10.1f}% | {r_fov:>8.1f}% | {p_fov:>8.1f}% | {r_dw:>8.2f}s | {p_dw:>8.2f}s")


if __name__ == "__main__":
    main()
