#!/usr/bin/env python3
"""
Decelerate-to-Turn (Trail-Braking & Cornering Control) RL Sweep
Tests 6 distinct paradigms starting from cloned expert weights (cloned_expert_v1).
"""

import sys
import time
import json
import subprocess
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"
if not PYTHON_BIN.exists():
    PYTHON_BIN = Path(sys.executable)

from control.predictive_controller import RecurrentVisualServoController
from control.simulation import FastPixhawkQuadSim
from control.trajectory import StochasticTargetTrajectory


def run_command_logged(cmd, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as lf:
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
        return proc.returncode


def evaluate_policy_detailed(model_path: Path, val_suite_path: Path):
    suite = json.loads(val_suite_path.read_text())
    scenarios = suite.get("scenarios", suite if isinstance(suite, list) else [])
    hyper_scenarios = [s for s in scenarios if "hyper_evasive" in s.get("scenario_id", "")]
    evasive_scenarios = [s for s in scenarios if "evasive" in s.get("scenario_id", "") and "hyper" not in s.get("scenario_id", "")]

    try:
        ctrl = RecurrentVisualServoController(model_path=str(model_path), w_nominal=0.0859)
    except Exception as e:
        print(f"[WARN] Failed to load controller for evaluation: {e}")
        return {
            "hyper_mean_fov": 0.0, "hyper_mean_dist": 99.0, "hyper_min_dist": 99.0,
            "hyper_min_speed": 0.0, "hyper_max_speed": 0.0, "hyper_mean_speed": 0.0,
            "hyper_catch_dwell": 0.0, "hyper_collisions": 0,
            "evasive_mean_fov": 0.0, "evasive_mean_dist": 99.0
        }

    def eval_subset(sc_list):
        fovs, min_dists, mean_dists, dwells, colls = [], [], [], [], 0
        min_speeds, max_speeds, mean_speeds = [], [], []

        for sc in sc_list:
            ctrl.reset()
            ctrl.prev_cmd = np.array([4.0, 0.0, 0.0, 0.0], dtype=np.float64)
            c_init = sc.get("initial_chaser_pos", [0.0, 0.0, -2.5])
            c_yaw = float(sc.get("initial_chaser_yaw", 0.0))
            c_vel = np.array([np.cos(c_yaw) * 4.0, np.sin(c_yaw) * 4.0, 0.0], dtype=np.float64)

            sim = FastPixhawkQuadSim(dt=0.02, max_vel_xy=12.0, max_accel_xy=6.0, max_vel_up=2.5, max_vel_down=1.5)
            sim.reset(initial_pos=np.array(c_init, dtype=np.float64), initial_yaw=c_yaw, initial_vel=c_vel)

            duration = float(sc.get("duration", 20.0))
            steps = int(round(duration / 0.02))
            t_pos = sc["initial_target_pos"]
            t_head = sc["initial_target_heading_rad"]
            traj = StochasticTargetTrajectory(
                duration=duration + 5.0,
                profile=sc["profile"],
                seed=sc["seed"],
                initial_pos=t_pos,
                initial_heading_rad=t_head
            )

            sc_dists, sc_fov, sc_speeds = [], [], []
            dwell_s = 0.0
            for step in range(steps):
                t = step * 0.02
                tgt_w = traj.get_position(t)
                telem = sim.get_camera_telemetry(tgt_w, target_w_m=0.35, target_h_m=0.20, img_w=1280, img_h=720, desired_target_size=110.0)
                c, s = np.cos(sim.yaw), np.sin(sim.yaw)
                vx_b = c * sim.vel[0] + s * sim.vel[1]
                vy_b = -s * sim.vel[0] + c * sim.vel[1]
                telem_clean = {k: v for k, v in telem.items() if k != "distance"}
                cmd = ctrl.compute_cmd(telem_clean, drone_pitch=sim.pitch, drone_roll=sim.roll, dt=0.02, vehicle_vel=np.array([vx_b, vy_b, vz_b]), vehicle_yaw_rate=sim.yaw_rate, current_alt_m=-sim.pos[2])
                sim.step(cmd)
                sc_dists.append(telem["distance"])
                sc_fov.append(telem["in_view"])
                speed_xy = float(np.linalg.norm(sim.vel[:2]))
                sc_speeds.append(speed_xy)
                if 2.5 <= telem["distance"] <= 4.5 and telem["in_view"]:
                    dwell_s += 0.02

            m_dist = float(np.min(sc_dists))
            if m_dist < 1.0:
                colls += 1
            fovs.append(float(np.mean(sc_fov) * 100))
            min_dists.append(m_dist)
            mean_dists.append(float(np.mean(sc_dists)))
            dwells.append(dwell_s)
            min_speeds.append(float(np.min(sc_speeds)))
            max_speeds.append(float(np.max(sc_speeds)))
            mean_speeds.append(float(np.mean(sc_speeds)))

        return {
            "mean_fov": float(np.mean(fovs)),
            "min_dist": float(np.min(min_dists)),
            "mean_dist": float(np.mean(mean_dists)),
            "catch_dwell": float(np.mean(dwells)),
            "collisions": colls,
            "min_speed": float(np.mean(min_speeds)),
            "max_speed": float(np.mean(max_speeds)),
            "mean_speed": float(np.mean(mean_speeds)),
        }

    hyper_m = eval_subset(hyper_scenarios)
    evasive_m = eval_subset(evasive_scenarios)

    return {
        "hyper_mean_fov": hyper_m["mean_fov"],
        "hyper_mean_dist": hyper_m["mean_dist"],
        "hyper_min_dist": hyper_m["min_dist"],
        "hyper_min_speed": hyper_m["min_speed"],
        "hyper_max_speed": hyper_m["max_speed"],
        "hyper_mean_speed": hyper_m["mean_speed"],
        "hyper_catch_dwell": hyper_m["catch_dwell"],
        "hyper_collisions": hyper_m["collisions"],
        "evasive_mean_fov": evasive_m["mean_fov"],
        "evasive_mean_dist": evasive_m["mean_dist"],
    }


def main():
    print("=" * 80)
    print("  DECELERATE-TO-TURN & CORNERING CONTROL RL SWEEP (6 CANDIDATES)")
    print("=" * 80)
    start_time = time.time()

    warm_start_path = PROJECT_ROOT / "control" / "weights" / "cloned_expert_v1" / "actor_weights.pt"
    val_suite_path = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "validation_suite_30.json"

    candidate_specs = [
        (
            "run_01_trail_brake_governor",
            "trail_brake_corner_governor",
            0.50,
            "Physics Centripetal Governor + Active Trail-Braking Incentive + Zero Cornering Penalty"
        ),
        (
            "run_02_pn_apex_cutter",
            "proportional_nav_apex_cutter",
            0.50,
            "Proportional Navigation Velocity Vector Alignment (Apex Cutting with Misalignment Drag)"
        ),
        (
            "run_03_retinal_slip_barrier",
            "retinal_slip_kinetic_barrier",
            0.50,
            "Biological Raptor Attack Model: Kinetic Energy Penalty Scaled to Retinal Slip Rate"
        ),
        (
            "run_04_curvature_energy_match",
            "curvature_adaptive_energy_matching",
            0.50,
            "3D Relative Velocity Vector Nulling with Curvature Standoff Buffer Relaxation"
        ),
        (
            "run_05_curriculum_speed_envelope",
            "staged_curriculum_trail_brake",
            0.35,
            "Analytical Cornering Speed Envelope Matching with Balanced Evasive Curriculum"
        ),
        (
            "run_06_unified_apex_dogfight_v2",
            "unified_apex_dogfight_v2",
            0.50,
            "Unified Synthesis: Huber Potential + PN Vector Lead + Deceleration on Shearing + Quadratic Dwell Jackpot"
        ),
    ]

    candidates = []
    for tag, r_mode, h_weight, notes in candidate_specs:
        candidates.append({
            "name": tag,
            "reward_mode": r_mode,
            "hyper_weight": h_weight,
            "timesteps": 200_000,
            "lr": 5e-5,
            "output_dir": PROJECT_ROOT / "control" / "weights" / f"sweep_{tag}",
            "warm_start": warm_start_path,
            "notes": notes,
        })

    results = {}
    for idx, cand in enumerate(candidates, 1):
        print(f"\n[{idx}/6] Processing Candidate {idx}: {cand['name']}")
        print(f"      Reward Mode:  {cand['reward_mode']}")
        print(f"      Hypothesis:   {cand['notes']}")
        print(f"      Hyper Weight: {cand['hyper_weight'] * 100:.0f}%")
        print(f"      Timesteps:    {cand['timesteps']:,}")

        best_p = cand["output_dir"] / "best_model" / "best_model.zip"
        final_p = cand["output_dir"] / "recurrent_ppo_drone_final.zip"

        if best_p.exists() or final_p.exists():
            print(f"      [CACHE] Found pre-trained model checkpoint in {cand['output_dir']}, evaluating directly...")
        else:
            train_cmd = [
                str(PYTHON_BIN),
                "control/train_rl.py",
                "--timesteps", str(cand["timesteps"]),
                "--envs", "4",
                "--lr", str(cand["lr"]),
                "--ent-coef", "0.0005",
                "--gamma", "0.995",
                "--epochs", "4",
                "--eval-freq", "25000",
                "--checkpoint-freq", "50000",
                "--warm-start-weights", str(cand["warm_start"]),
                "--output-dir", str(cand["output_dir"]),
                "--reward-mode", cand["reward_mode"],
                "--hyper-weight", str(cand["hyper_weight"]),
                "--no-wandb",
            ]

            log_file = PROJECT_ROOT / "runs" / "decel_turn_logs" / f"train_{cand['name']}.log"
            print(f"      [TRAINING] Launching PPO training (log: {log_file})...")
            t_train_start = time.time()
            ret = run_command_logged(train_cmd, log_file)
            t_train_elapsed = time.time() - t_train_start
            print(f"      [TRAINING] Completed in {t_train_elapsed:.1f}s (exit code {ret})")
            if ret != 0:
                print(f"[ERROR] Training failed for candidate {cand['name']} (exit code {ret})")
                continue

        eval_model = best_p if best_p.exists() else final_p
        print(f"\n[EVAL] Benchmarking candidate {cand['name']} on Validation Suite...")
        metrics = evaluate_policy_detailed(eval_model, val_suite_path)
        metrics["model_path"] = str(eval_model)
        metrics["reward_mode"] = cand["reward_mode"]
        metrics["notes"] = cand["notes"]
        results[cand["name"]] = metrics
        print(f"      Hyper-Evasive: FOV Lock={metrics['hyper_mean_fov']:.1f}%, Mean Dist={metrics['hyper_mean_dist']:.2f}m, Min Dist={metrics['hyper_min_dist']:.2f}m")
        print(f"      Speed Dynamic: Min={metrics['hyper_min_speed']:.1f} m/s (corner brake), Max={metrics['hyper_max_speed']:.1f} m/s (straight sprint)")
        print(f"      Evasive:       FOV Lock={metrics['evasive_mean_fov']:.1f}%, Mean Dist={metrics['evasive_mean_dist']:.2f}m")

    print("\n" + "=" * 80)
    print("  ALL 6 DECELERATE-TO-TURN EXPERIMENTS COMPLETE — LEADERBOARD")
    print("=" * 80)

    # Sort leaderboard by hyper_mean_fov descending
    leaderboard = sorted(results.items(), key=lambda x: x[1]["hyper_mean_fov"], reverse=True)
    print(f"{'Rank':<4} | {'Candidate Name':<34} | {'Hyper FOV%':<10} | {'Mean Dist':<10} | {'Min Speed':<10} | {'Max Speed':<10}")
    print("-" * 92)
    for rank, (name, m) in enumerate(leaderboard, 1):
        print(f"{rank:<4} | {name:<34} | {m['hyper_mean_fov']:>9.1f}% | {m['hyper_mean_dist']:>8.2f}m | {m['hyper_min_speed']:>8.1f}m/s | {m['hyper_max_speed']:>8.1f}m/s")

    # Save to JSON
    out_json = PROJECT_ROOT / "outputs" / "benchmarks" / "control" / "decel_turn_sweep_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[SAVED] Benchmark results saved to: {out_json}")
    print(f"[TOTAL TIME] Elapsed time: {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
