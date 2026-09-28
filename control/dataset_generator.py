#!/usr/bin/env python3
"""
Autonomous Drone Pursuit Dataset & Encounter Scenario Generator.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Generates standardized, deterministic, and diverse multi-angle encounter scenarios
for both classical controller benchmarking and future predictive / learned (RL/MPC) training.

Guarantees:
1. FOV Inception: 100% of targets initialize within the camera frustum (HFOV 60°, VFOV 45°, +15° uptilt).
2. Diverse Encounter Geometries:
   - Tail-Chase (0° ± 30°): Target fleeing ahead
   - Crossing Left / Right (90° / 270° ± 25°): Target crossing perpendicular to LOS
   - Head-On / Oncoming (180° ± 30°): Target closing at high relative speed
   - Oblique (45°, 135°): Angled pursuit
   - Omnidirectional: Uniform random heading
3. Reproducibility: Fully deterministic scenario generation with seeded random number generators.
4. Exportable JSON Manifests: Standalone serialized test/train splits for exact apples-to-apples evaluation.
"""

import sys
import json
import argparse
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.trajectory import StochasticTargetTrajectory, PROFILES, FlightProfile


class EncounterType(str, Enum):
    TAIL_CHASE = "tail_chase"
    CROSSING_RIGHT = "crossing_right"
    CROSSING_LEFT = "crossing_left"
    HEAD_ON = "head_on"
    OBLIQUE_AWAY = "oblique_away"
    OBLIQUE_INWARD = "oblique_inward"
    OMNIDIRECTIONAL = "omnidirectional"


@dataclass
class PursuitScenario:
    """Structured descriptor for a single closed-loop pursuit evaluation run."""
    scenario_id: str
    profile: str
    encounter_type: str
    duration: float
    seed: int
    initial_distance: float
    initial_azimuth_deg: float
    initial_elevation_deg: float
    initial_target_pos: List[float]       # [x, y, z] in NED
    initial_target_heading_rad: float     # Heading angle in radians
    initial_chaser_pos: List[float]       # [x, y, z] in NED
    initial_chaser_yaw: float             # Yaw angle in radians

    def create_target_oracle(self) -> StochasticTargetTrajectory:
        """Instantiates the continuous C^2 B-spline stochastic trajectory oracle."""
        return StochasticTargetTrajectory(
            duration=self.duration + 5.0,
            profile=self.profile,
            initial_pos=np.array(self.initial_target_pos, dtype=np.float64),
            initial_heading_rad=self.initial_target_heading_rad,
            seed=self.seed
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PursuitScenario":
        return cls(**data)


class PursuitDataset:
    """Container holding a collection of pursuit scenarios with metadata and export capabilities."""
    def __init__(self, name: str, scenarios: List[PursuitScenario], metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.scenarios = scenarios
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self.scenarios)

    def __getitem__(self, idx: int) -> PursuitScenario:
        return self.scenarios[idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_scenarios": len(self.scenarios),
            "metadata": self.metadata,
            "scenarios": [s.to_dict() for s in self.scenarios]
        }

    def save_json(self, output_path: Union[str, Path]):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[DATASET] Saved {len(self.scenarios)} scenarios to: {path}")

    @classmethod
    def load_json(cls, json_path: Union[str, Path]) -> "PursuitDataset":
        path = Path(json_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scenarios = [PursuitScenario.from_dict(s) for s in data["scenarios"]]
        return cls(name=data["name"], scenarios=scenarios, metadata=data.get("metadata", {}))


class PursuitDatasetGenerator:
    """
    Synthesizes diverse, geometrically rigorous encounter datasets guaranteed to start within FOV.
    """
    def __init__(
        self,
        camera_uptilt_deg: float = 15.0,
        hfov_deg: float = 60.0,
        vfov_deg: float = 45.0,
        fov_margin_deg: float = 5.0,
        min_distance_m: float = 8.0,
        max_distance_m: float = 20.0,
        nominal_altitude_m: float = 2.5
    ):
        self.camera_uptilt_deg = camera_uptilt_deg
        self.camera_uptilt = np.deg2rad(camera_uptilt_deg)
        self.hfov_deg = hfov_deg
        self.vfov_deg = vfov_deg
        self.fov_margin_deg = fov_margin_deg

        # Safe FOV bounds within camera sensor
        self.max_az_deg = (hfov_deg / 2.0) - fov_margin_deg    # e.g., 25.0 deg
        self.max_el_deg = (vfov_deg / 2.0) - fov_margin_deg    # e.g., 17.5 deg

        self.min_dist = min_distance_m
        self.max_dist = max_distance_m
        self.nominal_alt = nominal_altitude_m

        # Camera to Body transformation matrix
        cu, su = np.cos(self.camera_uptilt), np.sin(self.camera_uptilt)
        self.M_cam_to_body = np.array([
            [0.0, su, cu],
            [1.0, 0.0, 0.0],
            [0.0, cu, -su]
        ])

    def sample_initial_target_pos(
        self,
        chaser_pos: np.ndarray,
        chaser_yaw: float,
        rng: np.random.Generator,
        distance_range: Optional[Tuple[float, float]] = None,
        azimuth_deg_range: Optional[Tuple[float, float]] = None,
        elevation_deg_range: Optional[Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, float, float, float]:
        """
        Samples a 3D target coordinate strictly within the camera frustum.

        Returns:
            target_pos_world: [x, y, z] in NED
            range_m: Distance in meters
            azimuth_deg: Bearing relative to camera optical axis
            elevation_deg: Pitch angle relative to camera optical axis (positive is up)
        """
        d_min, d_max = distance_range or (self.min_dist, self.max_dist)
        az_min, az_max = azimuth_deg_range or (-self.max_az_deg, self.max_az_deg)
        el_min, el_max = elevation_deg_range or (-self.max_el_deg, self.max_el_deg)

        R = rng.uniform(d_min, d_max)
        az_deg = rng.uniform(az_min, az_max)
        el_deg = rng.uniform(el_min, el_max)

        az = np.deg2rad(az_deg)
        el = np.deg2rad(el_deg)

        # Coordinate in camera optical frame: X_cam: right, Y_cam: down, Z_cam: forward optical axis
        p_cam = np.array([
            R * np.sin(az) * np.cos(el),
            -R * np.sin(el),
            R * np.cos(az) * np.cos(el)
        ])

        # Convert to body frame [Forward, Right, Down]
        p_body = self.M_cam_to_body @ p_cam

        # Convert to world NED frame via chaser heading
        c, s = np.cos(chaser_yaw), np.sin(chaser_yaw)
        R_body_to_world = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ])
        p_world = chaser_pos + R_body_to_world @ p_body

        return p_world, R, az_deg, el_deg

    def compute_encounter_heading(
        self,
        chaser_pos: np.ndarray,
        target_pos: np.ndarray,
        encounter_type: Union[str, EncounterType],
        rng: np.random.Generator
    ) -> float:
        """
        Computes the target's initial horizontal heading angle relative to Line of Sight (LOS).
        """
        if isinstance(encounter_type, str):
            encounter_type = EncounterType(encounter_type)

        # Line of Sight vector in horizontal plane
        dx = target_pos[0] - chaser_pos[0]
        dy = target_pos[1] - chaser_pos[1]
        los_bearing = np.arctan2(dy, dx)

        if encounter_type == EncounterType.TAIL_CHASE:
            # Target flies away along/near LOS
            delta_heading = rng.uniform(-np.deg2rad(30.0), np.deg2rad(30.0))

        elif encounter_type == EncounterType.CROSSING_RIGHT:
            # Target crosses perpendicularly from left to right (+90 deg relative to LOS)
            delta_heading = np.deg2rad(90.0) + rng.uniform(-np.deg2rad(20.0), np.deg2rad(20.0))

        elif encounter_type == EncounterType.CROSSING_LEFT:
            # Target crosses perpendicularly from right to left (-90 deg relative to LOS)
            delta_heading = -np.deg2rad(90.0) + rng.uniform(-np.deg2rad(20.0), np.deg2rad(20.0))

        elif encounter_type == EncounterType.HEAD_ON:
            # Target flies directly towards chaser (+180 deg relative to LOS)
            delta_heading = np.pi + rng.uniform(-np.deg2rad(25.0), np.deg2rad(25.0))

        elif encounter_type == EncounterType.OBLIQUE_AWAY:
            # Angled away at ~45 deg
            sign = rng.choice([-1.0, 1.0])
            delta_heading = sign * np.deg2rad(45.0) + rng.uniform(-np.deg2rad(15.0), np.deg2rad(15.0))

        elif encounter_type == EncounterType.OBLIQUE_INWARD:
            # Angled toward at ~135 deg
            sign = rng.choice([-1.0, 1.0])
            delta_heading = sign * np.deg2rad(135.0) + rng.uniform(-np.deg2rad(15.0), np.deg2rad(15.0))

        elif encounter_type == EncounterType.OMNIDIRECTIONAL:
            # Completely random 360 deg heading
            delta_heading = rng.uniform(-np.pi, np.pi)

        else:
            delta_heading = 0.0

        target_heading = float((los_bearing + delta_heading + np.pi) % (2.0 * np.pi) - np.pi)
        return target_heading

    def generate_scenario(
        self,
        scenario_id: str,
        profile: str = "evasive",
        encounter_type: Union[str, EncounterType] = EncounterType.TAIL_CHASE,
        seed: int = 42,
        duration: float = 25.0,
        chaser_alt_m: float = 2.5
    ) -> PursuitScenario:
        """Generates a single deterministic, fully-validated pursuit scenario."""
        rng = np.random.default_rng(seed)

        chaser_pos = np.array([0.0, 0.0, -chaser_alt_m], dtype=np.float64)
        chaser_yaw = 0.0

        target_pos, R, az_deg, el_deg = self.sample_initial_target_pos(
            chaser_pos=chaser_pos,
            chaser_yaw=chaser_yaw,
            rng=rng
        )

        target_heading = self.compute_encounter_heading(
            chaser_pos=chaser_pos,
            target_pos=target_pos,
            encounter_type=encounter_type,
            rng=rng
        )

        enc_str = encounter_type.value if isinstance(encounter_type, EncounterType) else str(encounter_type)

        return PursuitScenario(
            scenario_id=scenario_id,
            profile=profile,
            encounter_type=enc_str,
            duration=duration,
            seed=seed,
            initial_distance=float(R),
            initial_azimuth_deg=float(az_deg),
            initial_elevation_deg=float(el_deg),
            initial_target_pos=[float(p) for p in target_pos],
            initial_target_heading_rad=float(target_heading),
            initial_chaser_pos=[float(p) for p in chaser_pos],
            initial_chaser_yaw=float(chaser_yaw)
        )

    def generate_balanced_benchmark_suite(
        self,
        suite_name: str = "drone_pursuit_benchmark_v1",
        profiles: Optional[List[str]] = None,
        encounter_types: Optional[List[EncounterType]] = None,
        seeds_per_condition: int = 10,
        duration: float = 25.0,
        base_seed: int = 1000
    ) -> PursuitDataset:
        """
        Creates a balanced, multi-axis benchmark dataset covering all profiles and encounter types.
        """
        if profiles is None:
            profiles = ["cruising", "evasive", "hyper_evasive"]

        if encounter_types is None:
            encounter_types = [
                EncounterType.TAIL_CHASE,
                EncounterType.CROSSING_RIGHT,
                EncounterType.CROSSING_LEFT,
                EncounterType.HEAD_ON
            ]

        scenarios = []
        scen_idx = 1

        for prof in profiles:
            for enc in encounter_types:
                for s_i in range(seeds_per_condition):
                    seed = base_seed + scen_idx * 17
                    scen_id = f"eval_{scen_idx:03d}_{prof}_{enc.value}_s{seed}"

                    scenario = self.generate_scenario(
                        scenario_id=scen_id,
                        profile=prof,
                        encounter_type=enc,
                        seed=seed,
                        duration=duration
                    )
                    scenarios.append(scenario)
                    scen_idx += 1

        metadata = {
            "suite_name": suite_name,
            "profiles": profiles,
            "encounter_types": [e.value for e in encounter_types],
            "seeds_per_condition": seeds_per_condition,
            "total_scenarios": len(scenarios),
            "camera_hfov_deg": self.hfov_deg,
            "camera_vfov_deg": self.vfov_deg,
            "camera_uptilt_deg": self.camera_uptilt_deg,
            "duration_s": duration
        }

        return PursuitDataset(name=suite_name, scenarios=scenarios, metadata=metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark encounter dataset for drone pursuit.")
    parser.add_argument("--name", type=str, default="benchmark_suite_v1", help="Dataset name")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds per condition (default: 10)")
    parser.add_argument("--duration", type=float, default=25.0, help="Flight duration (default: 25.0s)")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    generator = PursuitDatasetGenerator()
    dataset = generator.generate_balanced_benchmark_suite(
        suite_name=args.name,
        seeds_per_condition=args.seeds,
        duration=args.duration
    )

    out_file = args.out or str(PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / f"{args.name}.json")
    dataset.save_json(out_file)

    # Print summary statistics
    print(f"\n{'='*75}")
    print(f"  DATASET GENERATED: {args.name} ({len(dataset)} SCENARIOS)")
    print(f"{'='*75}")
    enc_counts = {}
    prof_counts = {}
    for s in dataset.scenarios:
        enc_counts[s.encounter_type] = enc_counts.get(s.encounter_type, 0) + 1
        prof_counts[s.profile] = prof_counts.get(s.profile, 0) + 1

    print("  Encounter Geometries:")
    for enc, count in enc_counts.items():
        print(f"    • {enc:<16}: {count} scenarios")

    print("\n  Target Flight Profiles:")
    for prof, count in prof_counts.items():
        print(f"    • {prof:<16}: {count} scenarios")

    print(f"\n  Manifest saved to: {out_file}\n")
