#!/usr/bin/env python3
"""
Autonomous Drone Pursuit & Catch Benchmarking Suite
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System

Evaluates controllers on the 120-scenario standardized test suite:
Primary Measures:
  1. Catch Success Rate (%): Percentage of flights sustaining continuous target standoff (5.0m - 7.0m) in-view for >= 3.0 seconds.
  2. Mean Time-to-Catch (s): Average elapsed flight time taken to establish the 3.0s sustained catch.

Secondary Measures:
  - Visual In-FOV Retention Rate (%)
  - Standoff Distance RMSE (m) relative to nominal 6.0m
  - Cumulative Firing Basket Dwell Time (s)
  - Yaw Jerk (°/s²) / Control Smoothness
  - Physical Collision Breach Rate (d < 1.2m)
  - Mean Standoff Distance (m) & Minimum Separation (m)
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.simulation import FastPixhawkQuadSim
from control.pid_controller import KinematicVisualServoController
from control.predictive_controller import RecurrentVisualServoController
from control.dataset_generator import PursuitDataset, PursuitScenario


def run_scenario_evaluation(
    scenario: PursuitScenario,
    controller_type: str,
    rl_controller: Optional[RecurrentVisualServoController] = None,
    desired_standoff: float = 6.0,
    standoff_min: float = 5.0,
    standoff_max: float = 7.0,
    required_dwell: float = 3.0,
    dt: float = 0.02
) -> Dict[str, Any]:
    """
    Executes a single simulation flight and evaluates Catch and Secondary metrics.
    """
    sim = FastPixhawkQuadSim(
        dt=dt,
        max_vel_xy=18.0,
        max_accel_xy=6.5,
        max_vel_up=4.0,
        max_vel_down=2.5
    )
    sim.reset(
        initial_pos=np.array(scenario.initial_chaser_pos, dtype=np.float64),
        initial_yaw=float(scenario.initial_chaser_yaw)
    )
    target_oracle = scenario.create_target_oracle()

    if controller_type == "pid":
        controller = KinematicVisualServoController(
            camera_uptilt_deg=15.0,
            hfov_deg=60.0,
            vfov_deg=45.0,
            desired_bbox_size=32.33,
            desired_standoff_dist=desired_standoff,
            use_bbox_size=False,
            enable_lateral_strafe=True,
            kp_lat=2.5,
            kd_lat=0.35,
            max_lat_vel=6.0,
            max_lat_accel=5.0,
            max_accel=6.5,
            max_decel=5.0,
            min_limits=np.array([-5.0, -6.0, -4.0, -120.0]),
            max_limits=np.array([15.0,  6.0,  2.5,  120.0]),
        )
    else:
        controller = rl_controller
        controller.reset()

    num_steps = int(round(scenario.duration / dt))
    
    dists = []
    in_views = []
    in_baskets = []
    cmds = []
    
    # Strict continuous dwell tracker
    strict_dwell = 0.0
    catch_achieved_strict = False
    time_to_catch_strict: Optional[float] = None

    # Leaky continuous dwell tracker (with 1.5*dt grace decay)
    leaky_dwell = 0.0
    catch_achieved_leaky = False
    time_to_catch_leaky: Optional[float] = None

    for step in range(num_steps):
        t = step * dt
        p_target = target_oracle.get_position(t)

        telem = sim.get_camera_telemetry(
            p_target,
            hfov_deg=60.0,
            vfov_deg=45.0,
            target_w_m=0.35,
            target_h_m=0.20,
            img_w=640,
            img_h=480,
            desired_target_size=32.33
        )
        dist = telem["distance"]
        in_view = telem["in_view"]
        
        dists.append(dist)
        in_views.append(in_view)
        
        # Target distance envelope: 5.0m - 7.0m in FOV
        in_basket = (standoff_min <= dist <= standoff_max) and in_view
        in_baskets.append(in_basket)
        
        # 1. Strict continuous dwell tracking
        if in_basket:
            strict_dwell += dt
            if strict_dwell >= required_dwell and not catch_achieved_strict:
                catch_achieved_strict = True
                # Time to catch is the timestamp when the drone reached the envelope that began the 3.0s dwell
                time_to_catch_strict = float(t - required_dwell)
        else:
            strict_dwell = 0.0

        # 2. Leaky continuous dwell tracking (allowing momentary 1-frame dropout grace)
        if in_basket:
            leaky_dwell = min(required_dwell, leaky_dwell + dt)
            if leaky_dwell >= required_dwell and not catch_achieved_leaky:
                catch_achieved_leaky = True
                time_to_catch_leaky = float(t - required_dwell)
        else:
            leaky_dwell = max(0.0, leaky_dwell - 1.5 * dt)

        # Compute control command
        if controller_type == "pid":
            cmd = controller.compute_cmd(telem, drone_pitch=sim.pitch, drone_roll=sim.roll, dt=dt)
        else:
            c, s = np.cos(sim.yaw), np.sin(sim.yaw)
            vx_b = c * sim.vel[0] + s * sim.vel[1]
            vy_b = -s * sim.vel[0] + c * sim.vel[1]
            vz_b = sim.vel[2]
            v_body = np.array([vx_b, vy_b, vz_b], dtype=np.float64)
            alt_agl = -sim.pos[2]
            cmd = controller.compute_cmd(
                telem,
                drone_pitch=sim.pitch,
                drone_roll=sim.roll,
                dt=dt,
                vehicle_vel=v_body,
                vehicle_yaw_rate=sim.yaw_rate,
                current_alt_m=alt_agl
            )
            
        cmds.append(cmd)
        sim.step(cmd)

    dists = np.array(dists)
    cmds = np.array(cmds)
    
    # Actuator Jerk (rate of change of commands per second)
    dcmds = np.diff(cmds, axis=0) / dt
    jerk_surge = float(np.mean(np.abs(dcmds[:, 0])))
    jerk_lat = float(np.mean(np.abs(dcmds[:, 1])))
    jerk_vert = float(np.mean(np.abs(dcmds[:, 2])))
    jerk_yaw = float(np.mean(np.abs(dcmds[:, 3])))
    
    # Standoff statistics
    rmse_standoff = float(np.sqrt(np.mean((dists - desired_standoff) ** 2)))
    mean_standoff = float(np.mean(dists))
    min_dist = float(np.min(dists))
    in_view_pct = float(np.mean(in_views) * 100.0)
    in_basket_s = float(np.sum(in_baskets) * dt)
    in_basket_pct = float(np.mean(in_baskets) * 100.0)
    
    return {
        "scenario_id": scenario.scenario_id,
        "profile": scenario.profile,
        "encounter_type": scenario.encounter_type,
        "controller": controller_type,
        # PRIMARY METRICS
        "catch_success_strict": catch_achieved_strict,
        "time_to_catch_strict": time_to_catch_strict,
        "catch_success_leaky": catch_achieved_leaky,
        "time_to_catch_leaky": time_to_catch_leaky,
        # SECONDARY METRICS
        "in_view_pct": in_view_pct,
        "rmse_standoff": rmse_standoff,
        "mean_standoff": mean_standoff,
        "in_basket_s": in_basket_s,
        "in_basket_pct": in_basket_pct,
        "min_distance": min_dist,
        "collision_breach": bool(min_dist < 1.2),
        "jerk_yaw": jerk_yaw,
        "jerk_surge": jerk_surge,
        "jerk_lat": jerk_lat,
        "jerk_vert": jerk_vert,
    }


def aggregate_controller_metrics(results: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    """
    Computes summary statistics with catch metrics front and center.
    """
    n = len(results)
    
    # Primary Metrics (Strict Continuous 3s Catch)
    succ_strict = sum(1 for d in results if d["catch_success_strict"])
    catch_rate_strict = (succ_strict / n) * 100.0
    times_strict = [d["time_to_catch_strict"] for d in results if d["time_to_catch_strict"] is not None]
    mean_ttc_strict = float(np.mean(times_strict)) if times_strict else float("nan")
    median_ttc_strict = float(np.median(times_strict)) if times_strict else float("nan")
    std_ttc_strict = float(np.std(times_strict)) if times_strict else float("nan")

    # Leaky Catch (for tolerance comparison)
    succ_leaky = sum(1 for d in results if d["catch_success_leaky"])
    catch_rate_leaky = (succ_leaky / n) * 100.0
    times_leaky = [d["time_to_catch_leaky"] for d in results if d["time_to_catch_leaky"] is not None]
    mean_ttc_leaky = float(np.mean(times_leaky)) if times_leaky else float("nan")

    # Secondary Metrics
    fov_retention = float(np.mean([d["in_view_pct"] for d in results]))
    rmse_standoff = float(np.mean([d["rmse_standoff"] for d in results]))
    mean_standoff = float(np.mean([d["mean_standoff"] for d in results]))
    basket_time_s = float(np.mean([d["in_basket_s"] for d in results]))
    basket_share = float(np.mean([d["in_basket_pct"] for d in results]))
    yaw_jerk = float(np.mean([d["jerk_yaw"] for d in results]))
    min_dist = float(np.mean([d["min_distance"] for d in results]))
    collisions = sum(1 for d in results if d["collision_breach"])

    # Profile Breakdown
    profiles = ["cruising", "evasive", "hyper_evasive"]
    profile_stats = {}
    for p in profiles:
        sub = [d for d in results if d["profile"] == p]
        n_sub = len(sub)
        p_succ_strict = sum(1 for d in sub if d["catch_success_strict"])
        p_catch_rate_strict = (p_succ_strict / n_sub) * 100.0
        p_times_strict = [d["time_to_catch_strict"] for d in sub if d["time_to_catch_strict"] is not None]
        p_mean_ttc_strict = float(np.mean(p_times_strict)) if p_times_strict else float("nan")

        p_succ_leaky = sum(1 for d in sub if d["catch_success_leaky"])
        p_catch_rate_leaky = (p_succ_leaky / n_sub) * 100.0
        p_times_leaky = [d["time_to_catch_leaky"] for d in sub if d["time_to_catch_leaky"] is not None]
        p_mean_ttc_leaky = float(np.mean(p_times_leaky)) if p_times_leaky else float("nan")

        p_fov = float(np.mean([d["in_view_pct"] for d in sub]))
        p_dwell = float(np.mean([d["in_basket_s"] for d in sub]))
        p_rmse = float(np.mean([d["rmse_standoff"] for d in sub]))
        
        profile_stats[p] = {
            "n_runs": n_sub,
            "catch_rate_strict": p_catch_rate_strict,
            "catch_success_strict_count": p_succ_strict,
            "mean_ttc_strict": p_mean_ttc_strict,
            "catch_rate_leaky": p_catch_rate_leaky,
            "catch_success_leaky_count": p_succ_leaky,
            "mean_ttc_leaky": p_mean_ttc_leaky,
            "fov_retention": p_fov,
            "dwell_time_s": p_dwell,
            "rmse_standoff": p_rmse,
        }

    return {
        "controller_name": name,
        "n_scenarios": n,
        # PRIMARY
        "catch_rate_strict": catch_rate_strict,
        "catch_count_strict": succ_strict,
        "mean_time_to_catch_strict": mean_ttc_strict,
        "median_time_to_catch_strict": median_ttc_strict,
        "std_time_to_catch_strict": std_ttc_strict,
        "catch_rate_leaky": catch_rate_leaky,
        "catch_count_leaky": succ_leaky,
        "mean_time_to_catch_leaky": mean_ttc_leaky,
        # SECONDARY
        "fov_retention_pct": fov_retention,
        "rmse_standoff_m": rmse_standoff,
        "mean_standoff_m": mean_standoff,
        "basket_dwell_time_s": basket_time_s,
        "basket_dwell_share_pct": basket_share,
        "yaw_jerk_deg_s2": yaw_jerk,
        "mean_min_distance_m": min_dist,
        "collision_count": collisions,
        "collision_rate_pct": (collisions / n) * 100.0,
        "profiles": profile_stats,
    }


def generate_benchmark_visualizations(summary_stats: Dict[str, Dict[str, Any]], output_path: Path):
    """
    Renders academic publication-ready figures highlighting Primary and Secondary metrics.
    """
    controllers = list(summary_stats.keys())
    labels = [summary_stats[c]["controller_name"] for c in controllers]
    
    catch_rates = [summary_stats[c]["catch_rate_strict"] for c in controllers]
    ttcs = [summary_stats[c]["mean_time_to_catch_strict"] for c in controllers]
    fovs = [summary_stats[c]["fov_retention_pct"] for c in controllers]
    rmses = [summary_stats[c]["rmse_standoff_m"] for c in controllers]
    jerks = [summary_stats[c]["yaw_jerk_deg_s2"] for c in controllers]

    colors = ["#4A5568", "#ED8936", "#3182CE", "#805AD5", "#38A169"]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#FFFFFF")

    # Plot 1: Primary Metric 1 — Catch Success Rate (%)
    bars1 = axs[0, 0].bar(range(len(labels)), catch_rates, color=colors, width=0.55, edgecolor="#2D3748", linewidth=1.2)
    axs[0, 0].set_title("PRIMARY METRIC: Catch Success Rate (3.0s Continuous Standoff)", fontsize=12, fontweight="bold", pad=12)
    axs[0, 0].set_ylabel("Catch Success Rate (%)", fontsize=11)
    axs[0, 0].set_xticks(range(len(labels)))
    axs[0, 0].set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    axs[0, 0].set_ylim(0, 70)
    axs[0, 0].grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        axs[0, 0].text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

    # Plot 2: Primary Metric 2 — Mean Time-to-Catch (s)
    bars2 = axs[0, 1].bar(range(len(labels)), ttcs, color=colors, width=0.55, edgecolor="#2D3748", linewidth=1.2)
    axs[0, 1].set_title("PRIMARY METRIC: Mean Time-to-Catch (Elapsed Flight Seconds)", fontsize=12, fontweight="bold", pad=12)
    axs[0, 1].set_ylabel("Time to Catch (seconds)", fontsize=11)
    axs[0, 1].set_xticks(range(len(labels)))
    axs[0, 1].set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    axs[0, 1].set_ylim(0, 14)
    axs[0, 1].grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        axs[0, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}s", ha="center", va="bottom", fontweight="bold", fontsize=10)

    # Plot 3: Secondary Metric — Visual In-FOV Retention (%) vs Standoff RMSE
    x_pos = np.arange(len(labels))
    w = 0.35
    b_fov = axs[1, 0].bar(x_pos - w/2, fovs, width=w, label="In-FOV Retention (%)", color="#4299E1", edgecolor="#2B6CB0")
    b_dwell = axs[1, 0].bar(x_pos + w/2, [summary_stats[c]["basket_dwell_time_s"] for c in controllers], width=w, label="Basket Dwell (s)", color="#48BB78", edgecolor="#2F855A")
    axs[1, 0].set_title("SECONDARY METRICS: FOV Retention vs Basket Dwell", fontsize=12, fontweight="bold", pad=12)
    axs[1, 0].set_xticks(x_pos)
    axs[1, 0].set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    axs[1, 0].set_ylim(0, 100)
    axs[1, 0].grid(axis="y", linestyle="--", alpha=0.5)
    axs[1, 0].legend(loc="upper right", frameon=True)

    # Plot 4: Secondary Metric — Yaw Jerk Smoothness
    bars4 = axs[1, 1].bar(range(len(labels)), jerks, color=colors, width=0.55, edgecolor="#2D3748", linewidth=1.2)
    axs[1, 1].set_title("SECONDARY METRIC: Yaw Jerk Smoothness (°/s²)", fontsize=12, fontweight="bold", pad=12)
    axs[1, 1].set_ylabel("Yaw Jerk (°/s²)", fontsize=11)
    axs[1, 1].set_xticks(range(len(labels)))
    axs[1, 1].set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    axs[1, 1].set_ylim(0, 45)
    axs[1, 1].grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars4:
        yval = bar.get_height()
        axs[1, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f"{yval:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Drone Catch Benchmarking Suite")
    parser.add_argument("--dataset", type=str, default="outputs/benchmarks/datasets/benchmark_suite_v1.json")
    parser.add_argument("--output-json", type=str, default="outputs/benchmarks/control/catch_benchmark_results.json")
    parser.add_argument("--output-img", type=str, default="outputs/benchmarks/control/catch_benchmark_comparison.png")
    args = parser.parse_args()

    dataset_path = PROJECT_ROOT / args.dataset
    print("=" * 90)
    print("   AUTONOMOUS DRONE PURSUIT & CATCH BENCHMARK SUITE")
    print(f"   Target Dataset: {dataset_path}")
    print("   Primary Measure: Catch Success Rate (3.0s Standoff) & Mean Time-to-Catch")
    print("=" * 90)

    dataset = PursuitDataset.load_json(str(dataset_path))
    print(f"Loaded {len(dataset)} standardized scenarios.\n")

    models_to_test = [
        ("pid", "Classical PID Baseline", None, 0.0505),
        ("gen3_rl", "Gen 3 RL (Tail Chase)", PROJECT_ROOT / "control/weights/recurrent_ppo_tail_chase_finetune/best_model/best_model.zip", 0.048),
        ("gen4_rl", "Gen 4 RL (Continuous Potential)", PROJECT_ROOT / "control/weights/recurrent_ppo_continuous_potential/best_model/best_model.zip", 0.0505),
        ("gen5b_rl", "Gen 5B RL (Filtered Rates)", PROJECT_ROOT / "control/weights/recurrent_ppo_gen5/best_model/best_model.zip", 0.0505),
        ("gen5c_rl", "Gen 5C RL (Calibrated Production)", PROJECT_ROOT / "control/weights/recurrent_ppo_gen5_calibrated/best_model/best_model.zip", 0.0505),
    ]

    all_raw_results = {}
    summary_stats = {}

    for c_id, c_name, m_path, w_nom in models_to_test:
        print(f"\n>>> Running Evaluation for: [{c_name}]")
        rl_ctrl = None
        if m_path is not None:
            if not m_path.exists():
                print(f"  [WARNING] Model file {m_path} does not exist, skipping {c_id}!")
                continue
            rl_ctrl = RecurrentVisualServoController(model_path=str(m_path), w_nominal=w_nom)
        
        t0 = time.perf_counter()
        raw_runs = []
        for i, scen in enumerate(dataset.scenarios):
            res = run_scenario_evaluation(
                scenario=scen,
                controller_type="pid" if c_id == "pid" else "rl",
                rl_controller=rl_ctrl,
                desired_standoff=6.0,
                standoff_min=5.0,
                standoff_max=7.0,
                required_dwell=3.0,
                dt=0.02
            )
            raw_runs.append(res)
            if (i + 1) % 30 == 0 or (i + 1) == len(dataset):
                print(f"    [{i+1:3d}/{len(dataset)}] flights executed...")
        
        elapsed = time.perf_counter() - t0
        all_raw_results[c_id] = raw_runs
        summary_stats[c_id] = aggregate_controller_metrics(raw_runs, c_name)
        print(f"    Completed in {elapsed:.2f}s! Catch Rate: {summary_stats[c_id]['catch_rate_strict']:.1f}%, Mean TTC: {summary_stats[c_id]['mean_time_to_catch_strict']:.2f}s")

    # Save complete JSON
    out_json = PROJECT_ROOT / args.output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"summary": summary_stats, "raw": all_raw_results}, f, indent=2)
    print(f"\nRaw benchmark metrics exported to: {out_json}")

    # Generate visualization
    out_img = PROJECT_ROOT / args.output_img
    generate_benchmark_visualizations(summary_stats, out_img)

    # Print Table 1: Primary Metrics Front and Center
    print("\n" + "=" * 120)
    print(f"{'CONTROLLER':<30} | {'PRIMARY: CATCH RATE':^22} | {'PRIMARY: TIME-TO-CATCH':^22} | {'SECONDARY: IN-FOV':^18} | {'SECONDARY: RMSE':^16}")
    print(f"{'':<30} | {'(Strict 3.0s Dwell)':^22} | {'(Mean Elapsed s)':^22} | {'(Retention %)':^18} | {'(Standoff m)':^16}")
    print("-" * 120)
    for c_id, stats in summary_stats.items():
        name = stats["controller_name"]
        cr = f"{stats['catch_rate_strict']:5.1f}% ({stats['catch_count_strict']}/{stats['n_scenarios']})"
        ttc = f"{stats['mean_time_to_catch_strict']:5.2f}s (±{stats['std_time_to_catch_strict']:4.2f}s)" if not np.isnan(stats['mean_time_to_catch_strict']) else "N/A"
        fov = f"{stats['fov_retention_pct']:5.1f}%"
        rmse = f"{stats['rmse_standoff_m']:5.2f}m"
        print(f"{name:<30} | {cr:^22} | {ttc:^22} | {fov:^18} | {rmse:^16}")
    print("=" * 120)

    # Print Table 2: Profile Breakdown
    print("\n" + "=" * 120)
    print(f"{'FLIGHT PROFILE':<18} | {'METRIC':<20} | {'PID Baseline':^16} | {'Gen 3 RL':^16} | {'Gen 4 RL':^16} | {'Gen 5B RL':^16} | {'Gen 5C RL':^16}")
    print("-" * 120)
    for p in ["cruising", "evasive", "hyper_evasive"]:
        p_title = p.replace("_", " ").title()
        
        # Row 1: Catch Rate
        row_cr = [f"{summary_stats[c]['profiles'][p]['catch_rate_strict']:5.1f}%" if c in summary_stats else "N/A" for c in ["pid", "gen3_rl", "gen4_rl", "gen5b_rl", "gen5c_rl"]]
        print(f"{p_title:<18} | {'Catch Success %':<20} | {row_cr[0]:^16} | {row_cr[1]:^16} | {row_cr[2]:^16} | {row_cr[3]:^16} | {row_cr[4]:^16}")
        
        # Row 2: Time to catch
        row_ttc = [f"{summary_stats[c]['profiles'][p]['mean_ttc_strict']:5.2f}s" if (c in summary_stats and not np.isnan(summary_stats[c]['profiles'][p]['mean_ttc_strict'])) else "N/A" for c in ["pid", "gen3_rl", "gen4_rl", "gen5b_rl", "gen5c_rl"]]
        print(f"{'':<18} | {'Mean Time-to-Catch':<20} | {row_ttc[0]:^16} | {row_ttc[1]:^16} | {row_ttc[2]:^16} | {row_ttc[3]:^16} | {row_ttc[4]:^16}")

        # Row 3: In FOV
        row_fov = [f"{summary_stats[c]['profiles'][p]['fov_retention']:5.1f}%" if c in summary_stats else "N/A" for c in ["pid", "gen3_rl", "gen4_rl", "gen5b_rl", "gen5c_rl"]]
        print(f"{'':<18} | {'In-FOV Retention %':<20} | {row_fov[0]:^16} | {row_fov[1]:^16} | {row_fov[2]:^16} | {row_fov[3]:^16} | {row_fov[4]:^16}")
        
        # Row 4: Basket Dwell
        row_dw = [f"{summary_stats[c]['profiles'][p]['dwell_time_s']:5.2f}s" if c in summary_stats else "N/A" for c in ["pid", "gen3_rl", "gen4_rl", "gen5b_rl", "gen5c_rl"]]
        print(f"{'':<18} | {'Basket Dwell (s)':<20} | {row_dw[0]:^16} | {row_dw[1]:^16} | {row_dw[2]:^16} | {row_dw[3]:^16} | {row_dw[4]:^16}")
        print("-" * 120)
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
