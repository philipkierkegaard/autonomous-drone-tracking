#!/usr/bin/env python3
"""
Overnight 10-Candidate Hyper-Evasive Reward Function Optimization Suite.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Sequentially trains, evaluates, and benchmarks 10 distinct reward function architectures:
1.  01_monotone_potential:      Linear-Exponential potential eliminating distance deadzones.
2.  02_huber_loss:              Smooth Huber potential field with constant linear pull outside.
3.  03_lead_pursuit:            Proportional navigation velocity vector alignment (apex cutting).
4.  04_speed_adaptive:          Dynamic speed-dependent formation standoff (3.5m - 4.5m).
5.  05_closing_speed_bell:      Differential closing sprint with relative velocity matching.
6.  06_tight_centering:         High-gain optical crosshair centering with steep border penalties.
7.  07_optical_expansion_lead:  Optical bounding-box expansion rate divergence feedback.
8.  08_cumulative_dwell_jackpot: Quadratic dwell acceleration rewarding unbroken formation hold.
9.  09_terminal_bonus_steep:    Steep linear pull with high milestone rewards every 0.5s in basket.
10. 10_unified_dogfight_optimal: Unified synthesis of Huber potential, lead pursuit, and sprint closing.

All runs focus on Hyper-Evasive (60% hyper_evasive, 30% evasive, 10% cruising).
Generates benchmark reports, selects the top-performing policy, and renders side-by-side comparison videos.
"""

import sys
import time
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python3"
if not PYTHON_BIN.exists():
    PYTHON_BIN = Path(sys.executable)

ARTIFACT_DIRS = [
    Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/5967d853-7a7e-4cdd-8792-4a77e6d704a2"),
]


def run_command_logged(cmd, log_file: Path):
    """Executes a command and writes real-time logs to a file and stdout."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[EXEC] Running: {' '.join(str(c) for c in cmd)}")
    print(f"[LOG] Streaming output to: {log_file}")
    with open(log_file, "w") as f:
        f.write(f"=== Command: {' '.join(str(c) for c in cmd)} ===\n\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(PROJECT_ROOT))
        for line in proc.stdout:
            f.write(line)
            f.flush()
            if any(k in line for k in ["rollout/", "eval/", "TASK-LEVEL", "DONE", "BEST", "Error", "Traceback"]):
                print(f"  {line.strip()}")
        proc.wait()
        return proc.returncode


def evaluate_policy_on_hyper_evasive(model_path: Path) -> dict:
    """Directly evaluates a trained policy across all 7 hyper-evasive benchmark validation scenarios."""
    from control.simulation import FastPixhawkQuadSim
    from control.predictive_controller import RecurrentVisualServoController
    from control.trajectory import StochasticTargetTrajectory

    dataset_path = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "validation_suite_30.json"
    with open(dataset_path) as f:
        d = json.load(f)
    scenarios = d["scenarios"] if isinstance(d, dict) and "scenarios" in d else d
    hyper_scenarios = [s for s in scenarios if "hyper_evasive" in s.get("scenario_id", "")]

    try:
        ctrl = RecurrentVisualServoController(model_path=str(model_path), w_nominal=0.0859)
    except Exception as e:
        print(f"[WARN] Failed to load controller for evaluation: {e}")
        return {"mean_fov": 0.0, "mean_dist": 99.0, "min_dist": 99.0, "collisions": 0, "catch_dwell": 0.0}

    fovs, min_dists, mean_dists, dwells, colls = [], [], [], [], 0

    for sc in hyper_scenarios:
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

        sc_dists, sc_fov = [], []
        dwell_s = 0.0
        for step in range(steps):
            t = step * 0.02
            tgt_w = traj.get_position(t)
            telem = sim.get_camera_telemetry(tgt_w, target_w_m=0.35, target_h_m=0.20, img_w=1280, img_h=720, desired_target_size=110.0)
            c, s = np.cos(sim.yaw), np.sin(sim.yaw)
            vx_b = c * sim.vel[0] + s * sim.vel[1]
            vy_b = -s * sim.vel[0] + c * sim.vel[1]
            vz_b = sim.vel[2]
            cmd = ctrl.compute_cmd(telem, drone_pitch=sim.pitch, drone_roll=sim.roll, dt=0.02, vehicle_vel=np.array([vx_b, vy_b, vz_b]), vehicle_yaw_rate=sim.yaw_rate, current_alt_m=-sim.pos[2])
            sim.step(cmd)
            sc_dists.append(telem["distance"])
            sc_fov.append(telem["in_view"])
            if 2.5 <= telem["distance"] <= 4.5 and telem["in_view"]:
                dwell_s += 0.02

        m_dist = float(np.min(sc_dists))
        if m_dist < 1.0:
            colls += 1
        fovs.append(float(np.mean(sc_fov) * 100))
        min_dists.append(m_dist)
        mean_dists.append(float(np.mean(sc_dists)))
        dwells.append(dwell_s)

    return {
        "mean_fov": float(np.mean(fovs)),
        "min_dist": float(np.min(min_dists)),
        "mean_dist": float(np.mean(mean_dists)),
        "catch_dwell": float(np.mean(dwells)),
        "collisions": colls,
    }


def main():
    print("=" * 80)
    print("  OVERNIGHT 10-CANDIDATE HYPER-EVASIVE REWARD OPTIMIZATION SWEEP")
    print("=" * 80)
    start_time = time.time()

    warm_start_path = PROJECT_ROOT / "control" / "weights" / "cloned_expert_v1" / "actor_weights.pt"

    candidate_specs = [
        ("01_monotone_potential", "monotone_potential", "Linear-Exponential potential eliminating distance deadzones"),
        ("02_huber_loss", "huber_loss", "Smooth Huber potential field with constant linear pull outside"),
        ("03_lead_pursuit", "lead_pursuit", "Proportional navigation velocity vector alignment (apex cutting)"),
        ("04_speed_adaptive", "speed_adaptive", "Dynamic speed-dependent formation standoff (3.5m - 4.5m)"),
        ("05_closing_speed_bell", "closing_speed_bell", "Differential closing sprint with relative velocity matching"),
        ("06_tight_centering", "tight_centering", "High-gain optical crosshair centering with steep border penalties"),
        ("07_optical_expansion_lead", "optical_expansion_lead", "Optical bounding-box expansion rate divergence feedback"),
        ("08_cumulative_dwell_jackpot", "cumulative_dwell_jackpot", "Quadratic dwell acceleration rewarding unbroken formation hold"),
        ("09_terminal_bonus_steep", "terminal_bonus_steep", "Steep linear pull with high milestone rewards every 0.5s in basket"),
        ("10_unified_dogfight_optimal", "unified_dogfight_optimal", "Unified synthesis of Huber potential, lead pursuit, and sprint closing"),
    ]

    candidates = []
    for tag, r_mode, notes in candidate_specs:
        candidates.append({
            "name": tag,
            "reward_mode": r_mode,
            "hyper_weight": 0.60,
            "timesteps": 250_000,
            "lr": 4e-5,
            "output_dir": PROJECT_ROOT / "control" / "weights" / f"sweep_{tag}",
            "warm_start": warm_start_path,
            "notes": notes,
        })

    results = {}
    for idx, cand in enumerate(candidates, 1):
        print(f"\n[{idx}/10] Processing Candidate {idx}: {cand['name']}")
        print(f"      Reward Mode:  {cand['reward_mode']}")
        print(f"      Hypothesis:   {cand['notes']}")
        print(f"      Timesteps:    {cand['timesteps']:,}")

        best_p = cand["output_dir"] / "best_model" / "best_model.zip"
        final_p = cand["output_dir"] / "recurrent_ppo_drone_final.zip"

        # Check if already trained to save time
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

            log_file = PROJECT_ROOT / "runs" / "overnight_logs" / f"train_{cand['name']}.log"
            ret = run_command_logged(train_cmd, log_file)
            if ret != 0:
                print(f"[ERROR] Training failed for candidate {cand['name']} (exit code {ret})")
                continue

        eval_model = best_p if best_p.exists() else final_p
        print(f"\n[EVAL] Benchmarking candidate {cand['name']} on Hyper-Evasive Suite...")
        metrics = evaluate_policy_on_hyper_evasive(eval_model)
        metrics["model_path"] = str(eval_model)
        metrics["reward_mode"] = cand["reward_mode"]
        metrics["notes"] = cand["notes"]
        results[cand["name"]] = metrics
        print(f"      Metrics: FOV Lock={metrics['mean_fov']:.1f}%, Mean Dist={metrics['mean_dist']:.2f}m, Min Dist={metrics['min_dist']:.2f}m, Collisions={metrics['collisions']}")

    print("\n" + "=" * 80)
    print("  ALL 10 OVERNIGHT EXPERIMENTS COMPLETE — COMPREHENSIVE LEADERBOARD")
    print("=" * 80)

    # Save sweep results to JSON
    summary_path = PROJECT_ROOT / "outputs" / "benchmarks" / "control" / "overnight_sweep_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SUMMARY] Saved benchmark summary to: {summary_path}")

    # Print ranked leaderboard
    ranked = sorted(
        results.items(),
        key=lambda item: (item[1]["collisions"] == 0, item[1]["mean_fov"], -item[1]["mean_dist"]),
        reverse=True
    )

    print("\nRANK | CANDIDATE                      | FOV LOCK % | MEAN DIST | MIN DIST | COLLISIONS | NOTES")
    print("-" * 98)
    for rank, (name, r) in enumerate(ranked, 1):
        coll_str = "SAFE (0)" if r["collisions"] == 0 else f"FAIL ({r['collisions']})"
        print(f"{rank:4d} | {name:<30s} | {r['mean_fov']:9.1f}% | {r['mean_dist']:8.2f}m | {r['min_dist']:7.2f}m | {coll_str:<10s} | {r['notes'][:25]}")

    best_candidate_name = ranked[0][0] if ranked else None
    print(f"\n[WINNER] Champion Policy: {best_candidate_name}")

    if best_candidate_name:
        best_model = results[best_candidate_name]["model_path"]
        print(f"[RENDER] Generating validation videos for champion {best_candidate_name}...")

        v1_cmd = [
            str(PYTHON_BIN),
            "control/rendering/render_simple_chase.py",
            "--scenario", "val_24_hyper_evasive_tail_chase_s10023",
            "--controller", "both",
            "--gif",
            "--model", str(best_model),
        ]
        run_command_logged(v1_cmd, PROJECT_ROOT / "runs" / "overnight_logs" / "render_val24.log")

        v2_cmd = [
            str(PYTHON_BIN),
            "control/rendering/render_simple_chase.py",
            "--scenario", "val_25_hyper_evasive_crossing_right_s10024",
            "--controller", "both",
            "--gif",
            "--model", str(best_model),
        ]
        run_command_logged(v2_cmd, PROJECT_ROOT / "runs" / "overnight_logs" / "render_val25.log")

        # Copy newly generated GIFs to artifact directory
        out_vid_dir = PROJECT_ROOT / "outputs" / "videos" / "control"
        for gif_file in out_vid_dir.glob("*val_2*.gif"):
            for art_dir in ARTIFACT_DIRS:
                if art_dir.exists():
                    shutil.copy2(gif_file, art_dir / gif_file.name)
                    print(f"[ARTIFACT] Copied {gif_file.name} to {art_dir}")

    elapsed_m = (time.time() - start_time) / 60.0
    print(f"\n[COMPLETE] 10-candidate overnight suite finished in {elapsed_m:.1f} minutes!")


if __name__ == "__main__":
    main()
