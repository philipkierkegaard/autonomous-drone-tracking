#!/usr/bin/env python3
"""
1.5 Million Timestep Decelerate-to-Turn & Standoff Capture RL Sweep.
Trains 5 elite reward function candidates starting from cloned expert weights (cloned_expert_v1)
with 1,500,000 timesteps each, 8 parallel environments, loosened dynamic pitch/speed gates,
and un-clamped exploration variance (log_std = -0.8).
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
            "hyper_catch_dwell": 0.0, "hyper_continuous_dwell": 0.0, "hyper_success_rate": 0.0,
            "hyper_collisions": 0, "evasive_mean_fov": 0.0, "evasive_mean_dist": 99.0
        }

    def eval_subset(sc_list):
        fovs, min_dists, mean_dists, dwells, cont_dwells, colls = [], [], [], [], [], 0
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
            cur_cont_dwell = 0.0
            max_cont_dwell = 0.0

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

                in_basket = (2.5 <= telem["distance"] <= 4.5 and telem["in_view"])
                if in_basket:
                    dwell_s += 0.02
                    cur_cont_dwell += 0.02
                    if cur_cont_dwell > max_cont_dwell:
                        max_cont_dwell = cur_cont_dwell
                else:
                    cur_cont_dwell = 0.0

            m_dist = float(np.min(sc_dists))
            if m_dist < 1.0:
                colls += 1
            fovs.append(float(np.mean(sc_fov) * 100))
            min_dists.append(m_dist)
            mean_dists.append(float(np.mean(sc_dists)))
            dwells.append(dwell_s)
            cont_dwells.append(max_cont_dwell)
            min_speeds.append(float(np.min(sc_speeds)))
            max_speeds.append(float(np.max(sc_speeds)))
            mean_speeds.append(float(np.mean(sc_speeds)))

        n_success = sum(1 for d in cont_dwells if d >= 3.0)
        return {
            "mean_fov": float(np.mean(fovs)),
            "min_dist": float(np.min(min_dists)),
            "mean_dist": float(np.mean(mean_dists)),
            "catch_dwell": float(np.mean(dwells)),
            "continuous_dwell": float(np.max(cont_dwells)) if cont_dwells else 0.0,
            "success_rate": float(n_success / len(sc_list) * 100.0) if sc_list else 0.0,
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
        "hyper_continuous_dwell": hyper_m["continuous_dwell"],
        "hyper_success_rate": hyper_m["success_rate"],
        "hyper_collisions": hyper_m["collisions"],
        "evasive_mean_fov": evasive_m["mean_fov"],
        "evasive_mean_dist": evasive_m["mean_dist"],
    }


def write_report(results: dict, elapsed_total_s: float):
    report_path = PROJECT_ROOT / "docs" / "1M5_DECEL_TURN_RL_TRAINING_REPORT.md"
    lines = [
        "# 1.5M Timestep Decelerate-to-Turn & Standoff Capture RL Training Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "**Pipeline:** RecurrentPPO (MlpLstmPolicy) fine-tuning from `cloned_expert_v1` actor weights",
        f"**Timesteps per Candidate:** 1,500,000 steps (~7.5× more than the 200k sweep)",
        "**Parallel Environments:** 8 vectorized workers",
        "**Exploration Variance:** `log_std = -0.8` (std ≈ 0.45, un-clamping throttle dynamics)",
        "**Deceleration Physics:** `q_pitch` relaxed to -18° backward pitch, `q_speed` relaxed to 5.0 m/s closing speed",
        "**Autonomous Policy:** Zero manual regulation / overrides. Pure end-to-end neural actor.",
        f"**Total Sweep Runtime:** {elapsed_total_s / 60.0:.1f} minutes",
        "",
        "---",
        "",
        "## 1. Candidate Leaderboard (Hyper-Evasive Validation Suite)",
        "",
        "| Rank | Candidate Name | Reward Mode | Hyper FOV% | Mean Dist | Turn Speed (Min) | Straight Sprint (Max) | Max Continuous Dwell | Success Rate (≥3.0s Dwell) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    leaderboard = sorted(results.items(), key=lambda x: (x[1].get("hyper_success_rate", 0.0), x[1].get("hyper_mean_fov", 0.0)), reverse=True)
    for rank, (name, m) in enumerate(leaderboard, 1):
        lines.append(
            f"| **{rank}** | `{name}` | `{m['reward_mode']}` | **{m['hyper_mean_fov']:.1f}%** | {m['hyper_mean_dist']:.2f} m | {m['hyper_min_speed']:.1f} m/s | {m['hyper_max_speed']:.1f} m/s | **{m.get('hyper_continuous_dwell', 0.0):.2f} s** | **{m.get('hyper_success_rate', 0.0):.1f}%** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Detailed Candidate Formulations & Findings",
        ""
    ])

    for name, m in leaderboard:
        lines.extend([
            f"### `{name}`",
            f"- **Reward Mode:** `{m['reward_mode']}`",
            f"- **Core Hypothesis:** {m['notes']}",
            f"- **Hyper-Evasive Optical Retention (FOV%):** {m['hyper_mean_fov']:.1f}%",
            f"- **Hyper-Evasive Mean Distance:** {m['hyper_mean_dist']:.2f} m (Min Distance: {m['hyper_min_dist']:.2f} m)",
            f"- **Speed Modulation:** {m['hyper_min_speed']:.1f} m/s in turns $\\longleftrightarrow$ {m['hyper_max_speed']:.1f} m/s on straights (Mean: {m['hyper_mean_speed']:.1f} m/s)",
            f"- **Standoff Capture Dwell:** Cumulative: {m['hyper_catch_dwell']:.2f} s, Peak Continuous Dwell: **{m.get('hyper_continuous_dwell', 0.0):.2f} s**",
            f"- **Interception Success Rate:** **{m.get('hyper_success_rate', 0.0):.1f}%**",
            f"- **Evasive Benchmark Baseline:** FOV={m['evasive_mean_fov']:.1f}%, Mean Dist={m['evasive_mean_dist']:.2f} m",
            f"- **Model Checkpoint:** `{m.get('model_path', '')}`",
            ""
        ])

    lines.extend([
        "---",
        "",
        "## 3. Comparative Summary vs. Baseline and 200k Sweep",
        "",
        "| Architecture | Timesteps | Hyper FOV% | Mean Dist | Speed Range | Success Rate |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        "| Classical PID | N/A | 32.5% | 29.8 m | Oscillatory | 0.0% (Crash in val_26) |",
        "| Untrained Cloned Expert | 0 | 44.6% | 23.4 m | Flat (~5.1 m/s) | 0.0% |",
        "| Overnight Sweep Best | 200k | 31.0% | 41.0 m | Flat | 0.0% |",
        "| Decel-to-Turn Run 02 (PN Apex) | 200k | 68.0% | 18.7 m | 3.1 - 7.1 m/s | 0.0% |",
    ])
    if leaderboard:
        top_name, top_m = leaderboard[0]
        lines.append(f"| **{top_name} (Current 1.5M)** | **1.5M** | **{top_m['hyper_mean_fov']:.1f}%** | **{top_m['hyper_mean_dist']:.2f} m** | **{top_m['hyper_min_speed']:.1f} - {top_m['hyper_max_speed']:.1f} m/s** | **{top_m.get('hyper_success_rate', 0.0):.1f}%** |")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[REPORT] Updated report written to: {report_path}")


def main():
    print("=" * 80)
    print("  1.5M TIMESTEP DECELERATE-TO-TURN & INTERCEPTION RL SWEEP (5 CANDIDATES)")
    print("=" * 80)
    start_time = time.time()

    warm_start_path = PROJECT_ROOT / "control" / "weights" / "cloned_expert_v1" / "actor_weights.pt"
    val_suite_path = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "validation_suite_30.json"

    candidate_specs = [
        (
            "run_01_pn_sprint_apex_v3",
            "pn_sprint_apex_v3",
            0.50,
            "Proportional Navigation Vector Alignment + Monotonic Closing Potential (Unconstrained Sprint on Straights)"
        ),
        (
            "run_02_centripetal_load_envelope",
            "centripetal_load_envelope",
            0.50,
            "Physics-Grounded Centripetal Safe Speed Envelope Matching (v_safe = a_max / omega_turn)"
        ),
        (
            "run_03_retinal_slip_raptor_v3",
            "retinal_slip_raptor_v3",
            0.50,
            "Biological Raptor Pursuit: Low Retinal Slip Sprint Drive vs High Slip Kinetic Braking"
        ),
        (
            "run_04_huber_standoff_dogfight",
            "huber_standoff_dogfight",
            0.50,
            "Huber Standoff Potential Well + 3D Relative Velocity Matching at Standoff + Superlinear Dwell Jackpot"
        ),
        (
            "run_05_aerodynamic_sideslip_governor",
            "aerodynamic_sideslip_governor",
            0.50,
            "Aerodynamic Sideslip Angle Nulling + Boresight Sprint Drive + Active Trail-Brake Turn Bonus"
        ),
    ]

    candidates = []
    for tag, r_mode, h_weight, notes in candidate_specs:
        candidates.append({
            "name": tag,
            "reward_mode": r_mode,
            "hyper_weight": h_weight,
            "timesteps": 1_500_000,
            "lr": 1e-4,
            "ent_coef": 0.003,
            "log_std": -0.8,
            "envs": 8,
            "output_dir": PROJECT_ROOT / "control" / "weights" / f"sweep_1m5_{tag}",
            "warm_start": warm_start_path,
            "notes": notes,
        })

    results = {}
    for idx, cand in enumerate(candidates, 1):
        print(f"\n[{idx}/5] Processing Candidate {idx}: {cand['name']}")
        print(f"      Reward Mode:  {cand['reward_mode']}")
        print(f"      Hypothesis:   {cand['notes']}")
        print(f"      Hyper Weight: {cand['hyper_weight'] * 100:.0f}%")
        print(f"      Timesteps:    {cand['timesteps']:,} (Envs: {cand['envs']})")
        print(f"      Log Std Init: {cand['log_std']} (Exploration std ≈ 0.45)")

        best_p = cand["output_dir"] / "best_model" / "best_model.zip"
        final_p = cand["output_dir"] / "recurrent_ppo_drone_final.zip"

        if best_p.exists() or final_p.exists():
            print(f"      [CACHE] Found pre-trained checkpoint in {cand['output_dir']}, skipping training...")
        else:
            train_cmd = [
                str(PYTHON_BIN),
                "control/train_rl.py",
                "--timesteps", str(cand["timesteps"]),
                "--envs", str(cand["envs"]),
                "--lr", str(cand["lr"]),
                "--ent-coef", str(cand["ent_coef"]),
                "--gamma", "0.995",
                "--epochs", "4",
                "--eval-freq", "50000",
                "--checkpoint-freq", "100000",
                "--warm-start-weights", str(cand["warm_start"]),
                "--warm-start-log-std", str(cand["log_std"]),
                "--output-dir", str(cand["output_dir"]),
                "--reward-mode", cand["reward_mode"],
                "--hyper-weight", str(cand["hyper_weight"]),
                "--no-wandb",
            ]

            log_file = PROJECT_ROOT / "runs" / "1m5_decel_turn_logs" / f"train_{cand['name']}.log"
            print(f"      [TRAINING] Launching PPO training (log: {log_file})...")
            t_train_start = time.time()
            ret = run_command_logged(train_cmd, log_file)
            t_train_elapsed = time.time() - t_train_start
            print(f"      [TRAINING] Completed in {t_train_elapsed:.1f}s ({t_train_elapsed/60:.1f}m, exit code {ret})")
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
        print(f"      Dwell & Success: Catch Dwell={metrics['hyper_catch_dwell']:.2f}s, Peak Continuous Dwell={metrics['hyper_continuous_dwell']:.2f}s, Success Rate={metrics['hyper_success_rate']:.1f}%")

        # Save intermediate results & report
        out_json = PROJECT_ROOT / "outputs" / "benchmarks" / "control" / "1m5_decel_turn_sweep_results.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(results, indent=2))
        write_report(results, time.time() - start_time)

    print("\n" + "=" * 80)
    print("  ALL 5 CANDIDATES (1.5M TIMESTEPS EACH) TRAINING & BENCHMARKING COMPLETE")
    print("=" * 80)
    print(f"[TOTAL TIME] Elapsed time: {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
