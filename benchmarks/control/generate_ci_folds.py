#!/usr/bin/env python3
"""
Generate 10 Seed-Stratified Folds of 600 Scenarios for Confidence Interval Benchmarking.
Each fold contains:
- 3 Trajectory Profiles: cruising (200), evasive (200), hyper_evasive (200) = 600 scenarios
- 5 Geometric Encounter Types: 40 seeds per condition
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.dataset_generator import PursuitDatasetGenerator, EncounterType


def generate_10_folds(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = PursuitDatasetGenerator(
        camera_uptilt_deg=15.0,
        hfov_deg=62.0,
        vfov_deg=38.0,
        min_distance_m=5.0,
        max_distance_m=12.0
    )

    profiles = ["cruising", "evasive", "hyper_evasive"]
    encounter_types = [
        EncounterType.TAIL_CHASE,
        EncounterType.CROSSING_RIGHT,
        EncounterType.CROSSING_LEFT,
        EncounterType.HEAD_ON,
        EncounterType.OBLIQUE_AWAY
    ]
    seeds_per_condition = 40  # 5 * 40 = 200 per profile; 3 * 200 = 600 total

    for fold_idx in range(1, 11):
        fold_name = f"ci_benchmark_600_fold_{fold_idx:02d}"
        fold_path = output_dir / f"{fold_name}.json"
        if fold_path.exists():
            print(f"[{fold_idx:02d}/10] {fold_name} already exists. Skipping.")
            continue

        base_seed = fold_idx * 100_000
        dataset = generator.generate_balanced_benchmark_suite(
            suite_name=fold_name,
            profiles=profiles,
            encounter_types=encounter_types,
            seeds_per_condition=seeds_per_condition,
            duration=25.0,
            base_seed=base_seed
        )
        dataset.save_json(fold_path)
        print(f"[{fold_idx:02d}/10] Generated {fold_name} ({len(dataset)} scenarios) -> {fold_path}")


if __name__ == "__main__":
    out_dir = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "ci_10x600"
    generate_10_folds(out_dir)

