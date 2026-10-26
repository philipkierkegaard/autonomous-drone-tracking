#!/usr/bin/env python3
"""
Comprehensive 200-Scenario Benchmark: Learned Controller vs. Kinematic Controller
Profiles: 100 Evasive, 100 Hyper-Evasive
Encounter Breakdown: Tail Chase, Crossing Right, Crossing Left, Head On, Oblique Away
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.dataset_generator import PursuitDatasetGenerator, PursuitScenario, PursuitDataset, EncounterType
from control.predictive_controller import RecurrentVisualServoController
from benchmarks.control.benchmark_controller import create_pid_controller, evaluate_single_run


def generate_benchmark_300_dataset(output_path: Path) -> PursuitDataset:
    """Generates the 300-scenario dataset: 100 evasive + 100 hyper-evasive + 100 cruising, across 5 encounter geometries."""
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
    scenarios_per_cell = 20
    base_seed = 40000

    idx = 0
    # 1. 100 Evasive & 100 Hyper-Evasive (seeds 40000..40199)
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

    # 2. 100 Cruising (seeds 40200..40299)
    for enc in encounters:
        enc_name = enc.value
        for i in range(scenarios_per_cell):
            seed = base_seed + idx
            sc_id = f"bench300_{idx+1:03d}_cruising_{enc_name}_s{seed}"
            sc = gen.generate_scenario(
                scenario_id=sc_id,
                profile="cruising",
                encounter_type=enc,
                seed=seed,
                duration=25.0,
                chaser_alt_m=3.0,
            )
            scenarios.append(sc)
            idx += 1

    dataset = PursuitDataset(
        name="benchmark_300_suite",
        scenarios=scenarios,
        metadata={
            "description": "Standardized 300-scenario benchmark evaluating 100 cruising, 100 evasive, and 100 hyper-evasive targets across 5 encounter trajectories.",
            "total_scenarios": len(scenarios),
            "profile_distribution": {"cruising": 100, "evasive": 100, "hyper_evasive": 100},
            "encounter_distribution": {
                "tail_chase": 60,
                "crossing_right": 60,
                "crossing_left": 60,
                "head_on": 60,
                "oblique_away": 60,
            },
            "created_timestamp": time.time(),
        }
    )
    dataset.save_json(output_path)
    return dataset


generate_evasive_200_dataset = generate_benchmark_300_dataset


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

    # Breakdown by profile (dynamic for any dataset)
    profiles = sorted(list(set(r["profile"] for r in runs)))
    by_profile = {p: calc_group([r for r in runs if r["profile"] == p]) for p in profiles}

    # Breakdown by encounter trajectory (dynamic for any dataset)
    enc_types = sorted(list(set(r["encounter_type"] for r in runs)))
    by_encounter = {enc: calc_group([r for r in runs if r["encounter_type"] == enc]) for enc in enc_types}

    # Cross-tabulation (Profile x Encounter)
    cross_tab = {
        p: {enc: calc_group([r for r in runs if r["profile"] == p and r["encounter_type"] == enc]) for enc in enc_types}
        for p in profiles
    }

    return {
        "overall": overall,
        "by_profile": by_profile,
        "by_encounter": by_encounter,
        "cross_tabulation": cross_tab,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Learned Controller vs. Kinematic PID")
    parser.add_argument("--dataset", type=str, default="outputs/benchmarks/datasets/evasive_200_benchmark_suite.json", help="Path to benchmark dataset JSON")
    parser.add_argument("--rl-model", type=str, default="control/weights/cloned_expert_v4/actor_weights.pt", help="Path to RL model weights")
    parser.add_argument("--pid-gains", type=str, default="control/weights/tuned_pid_gains.json", help="Path to tuned PID gains JSON")
    parser.add_argument("--noise-mode", type=str, default="empirical", choices=["empirical", "none", "bernoulli"], help="Perception noise model")
    parser.add_argument("--drop-rate", type=float, default=0.05, help="Frame drop rate")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    if dataset_path.exists():
        dataset = PursuitDataset.load_json(dataset_path)
    else:
        dataset = generate_evasive_200_dataset(dataset_path)

    rl_model_path = Path(args.rl_model)
    if not rl_model_path.is_absolute():
        rl_model_path = PROJECT_ROOT / rl_model_path

    model_tag = rl_model_path.parent.name
    if model_tag in ("weights", ".", ""):
        model_tag = rl_model_path.stem

    pid_gains_path = Path(args.pid_gains)
    if not pid_gains_path.is_absolute():
        pid_gains_path = PROJECT_ROOT / pid_gains_path

    print("=" * 80)
    print(f"  BENCHMARK: LEARNED ({model_tag}) vs. KINEMATIC PID (Tuned 4-DOF)")
    print(f"  Dataset: {dataset.name} ({len(dataset.scenarios)} scenarios)")
    print(f"  Perception: 8D Kalman Filter, {args.noise_mode.upper()} Noise, {int(args.drop_rate*100)}% Random Frame Drops")
    print("=" * 80)

    # 1. Evaluate Learned Controller
    print(f"\n[INIT] Instantiating Learned Controller from: {rl_model_path}")
    rl_controller = RecurrentVisualServoController(model_path=str(rl_model_path), w_nominal=0.0859)

    rl_runs = run_evaluation(
        dataset=dataset,
        controller_type="rl",
        controller_instance=rl_controller,
        noise_mode=args.noise_mode,
        frame_drop_rate=args.drop_rate,
    )
    rl_summary = summarize_results(rl_runs)

    # 2. Evaluate Kinematic Controller (Tuned PID)
    print(f"\n[INIT] Instantiating Tuned Kinematic PID Controller from: {pid_gains_path}")
    pid_controller = create_pid_controller(
        gains_path=pid_gains_path,
        standoff_nominal=3.5,
        desired_bbox_size=110.0,
    )

    pid_runs = run_evaluation(
        dataset=dataset,
        controller_type="pid",
        controller_instance=pid_controller,
        noise_mode=args.noise_mode,
        frame_drop_rate=args.drop_rate,
    )
    pid_summary = summarize_results(pid_runs)

    # 3. Export Raw Results & Aggregations
    out_dir = PROJECT_ROOT / "outputs" / "benchmarks" / "control"
    out_dir.mkdir(parents=True, exist_ok=True)

    d_tag = dataset.name
    rl_out_path = out_dir / f"benchmark_results_{d_tag}_rl_{model_tag}.json"
    with open(rl_out_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": str(dataset_path.relative_to(PROJECT_ROOT) if dataset_path.is_relative_to(PROJECT_ROOT) else dataset_path),
            "controller": f"rl_{model_tag}",
            "model_path": str(rl_model_path.relative_to(PROJECT_ROOT) if rl_model_path.is_relative_to(PROJECT_ROOT) else rl_model_path),
            "noise_mode": args.noise_mode,
            "frame_drop_rate": args.drop_rate,
            "summary": rl_summary,
            "results": rl_runs,
        }, f, indent=2)

    pid_out_path = out_dir / f"benchmark_results_{d_tag}_pid_tuned.json"
    with open(pid_out_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": str(dataset_path.relative_to(PROJECT_ROOT) if dataset_path.is_relative_to(PROJECT_ROOT) else dataset_path),
            "controller": "pid_tuned",
            "gains_path": str(pid_gains_path.relative_to(PROJECT_ROOT) if pid_gains_path.is_relative_to(PROJECT_ROOT) else pid_gains_path),
            "noise_mode": args.noise_mode,
            "frame_drop_rate": args.drop_rate,
            "summary": pid_summary,
            "results": pid_runs,
        }, f, indent=2)

    comparative = {
        "dataset": d_tag,
        "num_scenarios": len(dataset.scenarios),
        "overall": {
            "rl": rl_summary["overall"],
            "pid": pid_summary["overall"],
        },
        "by_profile": {
            p: {
                "rl": rl_summary["by_profile"][p],
                "pid": pid_summary["by_profile"][p],
            } for p in rl_summary["by_profile"] if p in pid_summary["by_profile"]
        },
        "by_encounter": {
            enc: {
                "rl": rl_summary["by_encounter"][enc],
                "pid": pid_summary["by_encounter"][enc],
            } for enc in rl_summary["by_encounter"] if enc in pid_summary["by_encounter"]
        },
        "cross_tabulation": {
            p: {
                enc: {
                    "rl": rl_summary["cross_tabulation"][p][enc],
                    "pid": pid_summary["cross_tabulation"][p][enc],
                } for enc in rl_summary["cross_tabulation"][p] if enc in pid_summary["cross_tabulation"].get(p, {})
            } for p in rl_summary["cross_tabulation"] if p in pid_summary["cross_tabulation"]
        }
    }

    comp_path = out_dir / f"{d_tag}_comparative_summary.json"
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
    rl_label = f"Learned ({model_tag})"
    pid_label = "Kinematic (Tuned PID)"
    print(f"\n>>> OVERALL SUMMARY ({len(dataset.scenarios)} SCENARIOS)")
    print(f"{'Metric':<28} | {rl_label:<22} | {pid_label:<22} | {'Delta':<12}")
    print("-" * 90)
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
        print(f"{label:<28} | {v_rl:<22.2f} | {v_pid:<22.2f} | {sign}{diff:<11.2f}")

    print("\n>>> PROFILE BREAKDOWN")
    for p in rl_summary["by_profile"]:
        n_p = rl_summary["by_profile"][p]["n"]
        print(f"\n--- Profile: {p.upper()} ({n_p} scenarios) ---")
        print(f"{'Metric':<28} | {rl_label:<22} | {pid_label:<22} | {'Delta':<12}")
        print("-" * 90)
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
            print(f"{label:<28} | {v_rl:<22.2f} | {v_pid:<22.2f} | {sign}{diff:<11.2f}")

    print("\n>>> ENCOUNTER TRAJECTORY BREAKDOWN")
    print(f"{'Trajectory':<16} | {'RL 1s Catch':<12} | {'PID 1s Catch':<12} | {'RL FOV':<10} | {'PID FOV':<10} | {'RL Dwell':<10} | {'PID Dwell':<10}")
    print("-" * 90)
    for enc in rl_summary["by_encounter"]:
        r_c1 = rl_summary["by_encounter"][enc]["catch_rate_1s_pct"]
        p_c1 = pid_summary["by_encounter"][enc]["catch_rate_1s_pct"]
        r_fov = rl_summary["by_encounter"][enc]["fov_retention_pct"]
        p_fov = pid_summary["by_encounter"][enc]["fov_retention_pct"]
        r_dw = rl_summary["by_encounter"][enc]["mean_max_dwell_s"]
        p_dw = pid_summary["by_encounter"][enc]["mean_max_dwell_s"]
        print(f"{enc:<16} | {r_c1:>10.1f}% | {p_c1:>10.1f}% | {r_fov:>8.1f}% | {p_fov:>8.1f}% | {r_dw:>8.2f}s | {p_dw:>8.2f}s")


if __name__ == "__main__":
    main()
