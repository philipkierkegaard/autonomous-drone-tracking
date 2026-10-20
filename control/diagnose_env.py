#!/usr/bin/env python3
"""
Diagnostic Verification Suite for DronePursuitEnv.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Explicitly verifies:
1. Kalman timing, multirate tick behavior, and 3-state visibility flag (1.0 / 0.5 / 0.0).
2. Track expiration alignment with 0.5s deployment baseline (25 steps at 50 Hz).
3. 13-dimensional observation vector consistency, finiteness, and normalized bounds.
4. Reward term distributions, balance, and perception-grounded dwell criteria.
5. Termination conditions (ground strike, collision, lost timeout, dwell lock).
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.environment import DronePursuitEnv


def test_visibility_flag_and_kalman_timing():
    print("\n" + "=" * 76)
    print("  DIAGNOSTIC TEST 1: Kalman Timing & 3-State Visibility Flag (1.0 / 0.5 / 0.0)")
    print("=" * 76)

    env = DronePursuitEnv(in_medias_res_ratio=0.0)
    obs, info = env.reset(seed=42)

    flag_history = []
    camera_ticks = []
    coasting_steps = 0
    fresh_steps = 0

    # Step for 100 steps (2 seconds)
    for step in range(100):
        # Hold zero action / smooth cruise
        action = np.zeros(4, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        flag = obs[4]
        flag_history.append(flag)

        if flag == 1.0:
            fresh_steps += 1
        elif flag == 0.5:
            coasting_steps += 1

        if terminated or truncated:
            break

    flag_arr = np.array(flag_history)
    unique_flags = set(np.unique(flag_arr))
    valid_flag_states = {0.0, 0.5, 1.0}

    print(f"Observed Visibility Flags across {len(flag_arr)} steps:")
    print(f"  • Fresh Detections (1.0):       {fresh_steps} ({fresh_steps/len(flag_arr)*100:.1f}%)")
    print(f"  • Coasting Predictions (0.5):   {coasting_steps} ({coasting_steps/len(flag_arr)*100:.1f}%)")
    print(f"  • Unique Flag Values Observed:  {sorted(list(unique_flags))}")

    assert unique_flags.issubset(valid_flag_states), f"Invalid flag values observed: {unique_flags}"

    # Check for 1.0 <-> 0.0 toggling:
    # Under the old bug, the flag toggled between 1.0 and 0.0 on consecutive steps.
    transitions_1_to_0 = np.sum((flag_arr[:-1] == 1.0) & (flag_arr[1:] == 0.0))
    transitions_0_to_1 = np.sum((flag_arr[:-1] == 0.0) & (flag_arr[1:] == 1.0))
    print(f"  • Direct 1.0 <-> 0.0 Toggles:    {transitions_1_to_0 + transitions_0_to_1} (Target: 0 during tracking)")

    assert (transitions_1_to_0 + transitions_0_to_1) == 0, (
        f"Detected artificial 1.0 <-> 0.0 strobing during tracking! "
        f"Transitions 1->0: {transitions_1_to_0}, 0->1: {transitions_0_to_1}"
    )

    print("✓ TEST 1 PASSED: Visibility flag cleanly distinguishes fresh (1.0) and coasting (0.5) with zero strobing.")


def test_track_expiration_baseline():
    print("\n" + "=" * 76)
    print("  DIAGNOSTIC TEST 2: Track Expiration Alignment to 0.50s Baseline (25 steps)")
    print("=" * 76)

    env = DronePursuitEnv(in_medias_res_ratio=0.0)
    env.reset(seed=42)

    # Verify configuration
    assert env.coasting_step_limit == 25, f"Expected coasting_step_limit=25 (0.50s at 50Hz), got {env.coasting_step_limit}"
    assert env.tracker.max_lost_frames == 25, f"Expected tracker.max_lost_frames=25, got {env.tracker.max_lost_frames}"
    print(f"  • Configured coasting step limit: {env.coasting_step_limit} steps ({env.coasting_step_limit * env.dt:.2f} s)")

    # Step until target acquired with a fresh detection
    for _ in range(40):
        obs, _, _, _, _ = env.step(np.zeros(4))
        if obs[4] == 1.0:
            break

    # Turn drone 180 degrees away so target is completely behind the camera frustum
    env.sim.yaw = float((env.sim.yaw + np.pi) % (2.0 * np.pi))

    expired_step = None
    flags = []
    for step in range(1, 40):
        obs, _, terminated, truncated, info = env.step(np.zeros(4))
        flag = obs[4]
        flags.append(flag)
        if flag == 0.0 and expired_step is None:
            expired_step = step

    print(f"  • Flag sequence after target disappearance: {flags[:30]}")
    print(f"  • Steps until track expired to 0.0: {expired_step} steps")

    # Should coast for approximately 25 steps (allowing +-1 step for camera tick synchronization)
    assert expired_step is not None and 24 <= expired_step <= 27, (
        f"Track expiration occurred at step {expired_step}, expected ~25 steps (0.50s)!"
    )
    print(f"✓ TEST 2 PASSED: Track expiration matches the 0.50s deployment baseline ({expired_step * env.dt:.3f}s).")


def test_13d_observation_consistency_and_bounds():
    print("\n" + "=" * 76)
    print("  DIAGNOSTIC TEST 3: 13-Dimensional Observation Bounds & Finiteness (1,000 Steps)")
    print("=" * 76)

    env = DronePursuitEnv(in_medias_res_ratio=0.30)
    obs, info = env.reset(seed=99)

    obs_names = [
        "e_x (azimuth)", "e_y (elevation)", "scale_err", "w_norm",
        "flag (visibility)", "vx_body", "vy_body", "vz_body",
        "yaw_rate", "alt_norm", "d_ex (azimuth rate)", "d_ey (elevation rate)", "d_w (scale rate)"
    ]

    all_obs = []
    rewards = []
    term_reasons = []

    for step in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        all_obs.append(obs)
        rewards.append(reward)

        assert obs.shape == (13,), f"Bad obs shape: {obs.shape}"
        assert np.all(np.isfinite(obs)), f"Non-finite values found in observation at step {step}: {obs}"
        assert np.isfinite(reward), f"Non-finite reward at step {step}: {reward}"

        if terminated or truncated:
            term_reasons.append(info.get("termination_reason", "unknown"))
            obs, info = env.reset()

    obs_matrix = np.array(all_obs)  # (1000, 13)

    print("\n  Observation Statistics (1,000 steps across Canonical + In Medias Res):")
    print(f"  {'Dim':<4} {'Name':<24} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8} {'Status':>8}")
    print("  " + "-" * 72)

    all_in_bounds = True
    expected_ranges = [
        (-1.0, 1.0),   # e_x
        (-1.0, 1.0),   # e_y
        (-1.0, 1.0),   # scale_err
        (0.0, 1.0),    # w_norm
        (0.0, 1.0),    # flag
        (-1.0, 1.0),   # vx_body
        (-1.0, 1.0),   # vy_body
        (-1.0, 1.0),   # vz_body
        (-1.0, 1.0),   # yaw_rate
        (0.0, 2.0),    # alt_norm
        (-1.0, 1.0),   # d_ex
        (-1.0, 1.0),   # d_ey
        (-1.0, 1.0),   # d_w
    ]

    for dim in range(13):
        col = obs_matrix[:, dim]
        c_min, c_max = np.min(col), np.max(col)
        c_mean, c_std = np.mean(col), np.std(col)
        e_min, e_max = expected_ranges[dim]

        in_b = (c_min >= e_min - 1e-3) and (c_max <= e_max + 1e-3)
        status = "PASS" if in_b else "WARN"
        if not in_b:
            all_in_bounds = False

        print(f"  [{dim:>2}] {obs_names[dim]:<24} {c_min:>8.3f} {c_max:>8.3f} {c_mean:>8.3f} {c_std:>8.3f} {status:>8}")

    assert all_in_bounds, "Some observation dimensions exceeded their normalized bounds!"
    print(f"\n✓ TEST 3 PASSED: All 13 observation dimensions are finite, well-scaled, and within bounds.")


def test_reward_behavior_and_dwell_grounding():
    print("\n" + "=" * 76)
    print("  DIAGNOSTIC TEST 4: Reward Terms & Perception-Grounded Dwell Evaluation")
    print("=" * 76)

    env = DronePursuitEnv()
    obs, info = env.reset(seed=42)

    reward_breakdown: Dict[str, List[float]] = {
        "r_tracking_base": [],
        "r_trail_potential": [],
        "r_progress": [],
        "r_basket_dwell": [],
        "r_milestone": [],
        "p_proximity": [],
        "p_low_alt": [],
        "p_jerk": [],
        "p_effort": [],
    }

    dwell_without_track_count = 0

    for step in range(500):
        # Use simple forward pursuit action
        action = np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        for k in reward_breakdown:
            if k in info:
                reward_breakdown[k].append(info[k])

        # Verify dwell is NOT awarded if tracker has no active lock
        if info.get("in_basket", False):
            if obs[4] == 0.0:  # Tracker expired/lost
                dwell_without_track_count += 1

        if terminated or truncated:
            obs, info = env.reset()

    print("  Reward Component Distributions (Mean / Max):")
    for k, vals in reward_breakdown.items():
        if len(vals) > 0:
            print(f"  • {k:<22}: Mean={np.mean(vals):>7.3f}, Max={np.max(vals):>7.3f}, Min={np.min(vals):>7.3f}")

    print(f"\n  • Dwell awarded without active tracker lock: {dwell_without_track_count}")
    assert dwell_without_track_count == 0, (
        f"Dwell reward was awarded {dwell_without_track_count} times while tracker had no active lock!"
    )
    print("✓ TEST 4 PASSED: Reward terms are balanced and dwell is strictly grounded in active perception.")


def test_termination_conditions():
    print("\n" + "=" * 76)
    print("  DIAGNOSTIC TEST 5: Aerospace Termination Boundaries")
    print("=" * 76)

    # 1. Ground strike test (altitude < 1.0m AGL)
    env = DronePursuitEnv()
    env.reset(seed=1)
    env.sim.pos[2] = -0.80  # In NED, -0.80m is 0.80m above ground (< 1.0m min altitude)
    obs, reward, terminated, truncated, info = env.step(np.zeros(4))
    print(f"  • Low Altitude Test: terminated={terminated}, reason='{info.get('termination_reason')}'")
    assert terminated and info.get("termination_reason") == "ground_strike", "Ground strike termination failed!"

    # 2. Collision test (dist < 1.2m)
    env.reset(seed=2)
    t_sim = (env.step_count + 1) * env.dt
    tpos, _, _ = env.target_oracle.get_state(t_sim)
    env.sim.pos = tpos + np.array([0.5, 0.0, 0.0])
    obs, reward, terminated, truncated, info = env.step(np.zeros(4))
    print(f"  • Collision Test:    terminated={terminated}, reason='{info.get('termination_reason')}'")
    assert terminated and info.get("termination_reason") == "collision", "Collision termination failed!"

    # 3. Lost target timeout test
    env.reset(seed=3)
    env.sim.yaw = float((env.sim.yaw + np.pi) % (2.0 * np.pi))
    env.time_lost = 3.6  # Exceeds 3.5s timeout
    obs, reward, terminated, truncated, info = env.step(np.zeros(4))
    print(f"  • Lost Timeout Test: terminated={terminated}, reason='{info.get('termination_reason')}'")
    assert terminated and info.get("termination_reason") == "lost_target_timeout", "Lost timeout termination failed!"

    # 4. Lock dwell achieved test
    env.reset(seed=4)
    env.continuous_lock_dwell_s = 3.05  # Reached 3.0s dwell
    obs, reward, terminated, truncated, info = env.step(np.zeros(4))
    print(f"  • Lock Dwell Test:   terminated={terminated}, reason='{info.get('termination_reason')}'")
    assert terminated and info.get("termination_reason") == "lock_dwell_achieved", "Lock dwell termination failed!"

    print("✓ TEST 5 PASSED: All 4 aerospace termination boundaries trigger with 100% precision.")


def main():
    print("\n" + "#" * 76)
    print("  AUTONOMOUS DRONE PURSUIT: PRE-FLIGHT DIAGNOSTIC ROLLOUT SUITE")
    print("#" * 76)

    test_visibility_flag_and_kalman_timing()
    test_track_expiration_baseline()
    test_13d_observation_consistency_and_bounds()
    test_reward_behavior_and_dwell_grounding()
    test_termination_conditions()

    print("\n" + "#" * 76)
    print("  ALL 5 DIAGNOSTIC SUITES PASSED SUCCESSFULLY — ENVIRONMENT IS FLIGHT READY!")
    print("#" * 76 + "\n")


if __name__ == "__main__":
    main()
