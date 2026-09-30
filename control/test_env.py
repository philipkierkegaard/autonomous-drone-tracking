#!/usr/bin/env python3
"""
Test & Benchmarking Script for DronePursuitEnv.
Verifies environment reset distribution, stepping logic, and throughput speed.
"""

import sys
import time
from pathlib import Path
from collections import Counter
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.environment import DronePursuitEnv


def test_environment_stepping():
    print("\n" + "=" * 70)
    print("  TEST 1: Step Logic, Reward Components & Observation Sanity")
    print("=" * 70)

    env = DronePursuitEnv()
    obs, info = env.reset(seed=42)

    assert obs.shape == (13,), f"Expected obs shape (13,), got {obs.shape}"
    assert np.all(np.isfinite(obs)), "Observation contains NaN or Inf!"
    print(f"✓ Initial Observation: {np.round(obs, 3)}")
    print(f"✓ Initial Info: {info}")

    total_reward = 0.0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        assert obs.shape == (13,), f"Step {step}: Bad obs shape {obs.shape}"
        assert np.isfinite(reward), f"Step {step}: Reward is not finite: {reward}"

        if terminated or truncated:
            print(f"✓ Episode ended at step {step+1}: reason='{info['termination_reason']}'")
            obs, info = env.reset()
            break

    print(f"✓ Successfully executed steps. Total accumulated reward: {total_reward:.2f}")


def test_scenario_distribution():
    print("\n" + "=" * 70)
    print("  TEST 2: 70/30 Encounter & Profile Distribution Verification (500 Resets)")
    print("=" * 70)

    env = DronePursuitEnv(in_medias_res_ratio=0.30)
    modes = Counter()
    profiles = Counter()
    distances = []

    for i in range(500):
        obs, info = env.reset(seed=i)
        mode = info["mode"]
        prof = info["profile"]
        dist = info["initial_distance"]

        modes[mode] += 1
        profiles[prof] += 1
        distances.append(dist)

    total = 500
    canonical_pct = (modes["canonical"] / total) * 100.0
    medias_res_pct = 100.0 - canonical_pct

    print(f"Modes Breakdown (500 resets):")
    print(f"  • Canonical Full Flights: {modes['canonical']} ({canonical_pct:.1f}% vs. target ~70%)")
    for m, c in modes.items():
        if m != "canonical":
            print(f"    - {m}: {c} ({(c/total)*100.0:.1f}%)")
    print(f"  • Total In Medias Res: {medias_res_pct:.1f}% (target ~30%)")

    print(f"\nProfiles Breakdown:")
    for p, c in profiles.items():
        print(f"  • {p:15s}: {c:3d} ({(c/total)*100.0:.1f}%)")

    print(f"\nInitial Distances:")
    print(f"  • Mean: {np.mean(distances):.1f}m, Min: {np.min(distances):.1f}m, Max: {np.max(distances):.1f}m")

    assert 60.0 <= canonical_pct <= 80.0, f"Canonical pct out of expected range: {canonical_pct:.1f}%"
    print("✓ Scenario distribution matches target specifications!")


def test_simulation_speed():
    print("\n" + "=" * 70)
    print("  TEST 3: Simulation Stepping Throughput (Speed Benchmark)")
    print("=" * 70)

    env = DronePursuitEnv()
    env.reset(seed=42)

    num_steps = 10_000
    t0 = time.perf_counter()

    for _ in range(num_steps):
        action = np.zeros(4, dtype=np.float32)  # hover action
        obs, reward, term, trunc, _ = env.step(action)
        if term or trunc:
            env.reset()

    elapsed = time.perf_counter() - t0
    steps_per_sec = num_steps / elapsed
    sim_time_sec = num_steps * 0.02
    realtime_factor = sim_time_sec / elapsed

    print(f"Executed {num_steps:,} simulation steps in {elapsed:.2f}s")
    print(f"  • Step Throughput: {steps_per_sec:,.0f} steps/second")
    print(f"  • Real-Time Factor: {realtime_factor:.1f}x real-time (50 Hz simulation)")
    assert steps_per_sec > 2_000, f"Simulation too slow: {steps_per_sec} steps/s"
    print("✓ Throughput is exceptionally high for vectorized RL training!")


if __name__ == "__main__":
    test_environment_stepping()
    test_scenario_distribution()
    test_simulation_speed()
    print("\n" + "=" * 70)
    print("  ALL DRONE PURSUIT ENVIRONMENT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70 + "\n")
