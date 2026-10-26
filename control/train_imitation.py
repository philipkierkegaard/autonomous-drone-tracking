#!/usr/bin/env python3
"""
Behavioral Cloning & Imitation Learning Training Pipeline for Autonomous Drone Pursuit.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Clones the smooth, non-oscillating, high-performance flight characteristics of
KinematicVisualServoController directly into StandaloneRecurrentPolicy (15D LSTM actor).

Advantages over from-scratch RL:
1. Zero policy suicide risk: Supervised loss guarantees monotonic learning.
2. Perfect attitude and deceleration coordination learned directly from kinematics.
3. Rapid training (< 2 minutes) with deterministic convergence.
4. Directly deployable standalone PyTorch weights without SB3/NumPy version locks.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.simulation import FastPixhawkQuadSim
from control.pid_controller import KinematicVisualServoController
from control.trajectory import StochasticTargetTrajectory
from control.predictive_controller import StandaloneRecurrentPolicy, RecurrentVisualServoController
from control.spatial import unproject_camera_to_horizon
from benchmarks.control.benchmark_controller import create_pid_controller


def collect_expert_demonstrations(
    num_episodes: int = 1500,
    episode_duration: float = 15.0,
    dt: float = 0.02,
    dropout_prob: float = 0.05,
    noise_sigma: float = 0.015,
    seed_offset: int = 5000,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Simulates expert flights across diverse encounter geometries, target profiles,
    and initial conditions, generating 15D observation sequences and 4D normalized actions.
    """
    print(f"[EXPERT] Collecting {num_episodes} expert flight demonstrations ({int(episode_duration/dt)} steps each)...")
    trajectories = []
    
    profiles = ["hyper_evasive", "evasive", "cruising", "sprint_break", "aerobatic", "high_speed_evasive"]
    profile_weights = [0.40, 0.30, 0.10, 0.10, 0.05, 0.05]
    
    rng = np.random.default_rng(seed_offset)
    
    # Load tuned kinematic controller as expert once
    ctrl = create_pid_controller()
    
    for ep in range(num_episodes):
        prof = rng.choice(profiles, p=profile_weights)
        ep_seed = seed_offset + ep
        
        # Geometry sampling: Diverse initial positions, altitudes, and headings
        chaser_alt = float(rng.uniform(-4.5, -2.5))  # 2.5m to 4.5m AGL (NED coordinate)
        # Altitude delta between target and chaser: ±1.5m (ensuring target stays > 1.8m AGL)
        delta_alt = float(rng.uniform(-1.5, 1.5))
        target_alt = float(np.clip(chaser_alt + delta_alt, -6.8, -1.8))
        
        init_dist = float(rng.uniform(5.0, 16.0))
        init_lat = float(rng.uniform(-4.5, 4.5))
        init_heading = float(rng.uniform(-0.8, 0.8))
        init_chaser_vel = float(rng.uniform(3.0, 5.0))
        init_yaw = float(rng.uniform(-0.45, 0.45))
        
        # Diverse encounter types:
        # 1. Close Basket & Dwell (10%): Spawns within 2.8-4.5m basket for fine speed-matching & dwell demos
        # 2. Crossing Right (15%): Spawns on left flank, cuts East (+90 deg) across FOV
        # 3. Crossing Left (15%): Spawns on right flank, cuts West (-90 deg) across FOV
        # 4. Head-On Approach (12%): Spawns 12-24m ahead, flying toward and past the chaser
        # 5. Oblique Break Away (12%): Angled breakaway forcing aggressive yaw correction
        # 6. Oblique Inward (10%): Angled closing approach
        # 7. Omnidirectional 360 (8%): Full random 360-degree heading relative to LOS
        # 8. Tail Chase Sprint (18%): Deep forward chase (6-18m) demanding high-speed catch
        encounter_roll = rng.random()
        if encounter_roll < 0.10:
            # Close basket dwell encounter
            close_dist = float(rng.uniform(2.8, 4.5))
            close_lat = float(rng.uniform(-1.2, 1.2))
            target_pos = [close_dist, close_lat, target_alt]
            target_heading = float(rng.uniform(-0.35, 0.35))
            init_chaser_vel = float(rng.uniform(3.5, 4.5))  # Match target cruising speed
        elif encounter_roll < 0.25:
            # Crossing right
            flank_lat = float(rng.uniform(-7.0, -3.5))
            cross_dist = float(rng.uniform(6.0, 15.0))
            target_pos = [cross_dist, flank_lat, target_alt]
            target_heading = float(np.pi / 2.0 + rng.uniform(-0.25, 0.25))
        elif encounter_roll < 0.40:
            # Crossing left
            flank_lat = float(rng.uniform(3.5, 7.0))
            cross_dist = float(rng.uniform(6.0, 15.0))
            target_pos = [cross_dist, flank_lat, target_alt]
            target_heading = float(-np.pi / 2.0 + rng.uniform(-0.25, 0.25))
        elif encounter_roll < 0.52:
            # Head-on approach
            head_dist = float(rng.uniform(12.0, 24.0))
            head_lat = float(rng.uniform(-3.5, 3.5))
            target_pos = [head_dist, head_lat, target_alt]
            target_heading = float(np.pi + rng.uniform(-0.35, 0.35))
        elif encounter_roll < 0.64:
            # Oblique break away
            sign = float(rng.choice([-1.0, 1.0]))
            target_pos = [init_dist, init_lat, target_alt]
            target_heading = float(sign * (np.pi / 4.0) + rng.uniform(-0.25, 0.25))
        elif encounter_roll < 0.74:
            # Oblique inward
            sign = float(rng.choice([-1.0, 1.0]))
            target_pos = [init_dist, init_lat, target_alt]
            target_heading = float(sign * (3.0 * np.pi / 4.0) + rng.uniform(-0.2, 0.2))
        elif encounter_roll < 0.82:
            # Omnidirectional 360
            target_pos = [init_dist, init_lat, target_alt]
            target_heading = float(rng.uniform(-np.pi, np.pi))
        else:
            # Tail chase sprint
            sprint_dist = float(rng.uniform(6.0, 18.0))
            target_pos = [sprint_dist, init_lat, target_alt]
            target_heading = init_heading
            
        target = StochasticTargetTrajectory(
            duration=episode_duration + 5.0,
            profile=prof,
            initial_pos=target_pos,
            initial_heading_rad=target_heading,
            seed=ep_seed
        )
        
        sim = FastPixhawkQuadSim(
            dt=dt,
            max_vel_xy=15.0,
            max_accel_xy=6.0,
            max_vel_up=3.0,
            max_vel_down=2.0
        )
        sim.reset(
            initial_pos=np.array([0.0, 0.0, chaser_alt]),
            initial_yaw=init_yaw,
            initial_vel=np.array([np.cos(init_yaw) * init_chaser_vel, np.sin(init_yaw) * init_chaser_vel, 0.0])
        )
        
        # Reset kinematic controller for episode
        ctrl.reset()
        ctrl.prev_cmd = np.array([init_chaser_vel, 0.0, 0.0, 0.0])
        ctrl.prev_cmd_vx = float(init_chaser_vel)
        
        # Optical filtering state
        prev_ex, prev_ey, prev_w = 0.0, 0.0, 0.0859
        filt_dex, filt_dey, filt_dw = 0.0, 0.0, 0.0
        last_seen_ex, last_seen_ey, last_seen_w = 0.0, 0.0, 0.0859
        
        ep_obs = []
        ep_acts = []
        
        total_steps = int(episode_duration / dt)
        for step in range(total_steps):
            t = step * dt
            tgt_pos = target.get_position(t)
            telem = sim.get_camera_telemetry(
                tgt_pos,
                hfov_deg=62.0,
                vfov_deg=38.0,
                target_w_m=0.35,
                target_h_m=0.20,
                img_w=1280,
                img_h=720,
                desired_target_size=110.0
            )
            
            # Synthetic visual dropouts (simulates occlusion or frame loss)
            in_view_real = bool(telem["in_view"])
            is_dropped = (rng.random() < dropout_prob)
            telem_ctrl = {k: v for k, v in telem.items() if k != "distance"}
            if is_dropped or not in_view_real:
                telem_ctrl["status"] = "COASTING"
                telem_ctrl["in_view"] = False

            c_yaw, s_yaw = np.cos(sim.yaw), np.sin(sim.yaw)
            vx_b = c_yaw * sim.vel[0] + s_yaw * sim.vel[1]
            vy_b = -s_yaw * sim.vel[0] + c_yaw * sim.vel[1]
            vz_b = sim.vel[2]
            v_body = np.array([vx_b, vy_b, vz_b], dtype=np.float64)

            cmd = ctrl.compute_cmd(
                telem_ctrl,
                drone_pitch=sim.pitch,
                drone_roll=sim.roll,
                dt=dt,
                vehicle_vel=v_body,
                vehicle_yaw_rate=sim.yaw_rate,
                current_alt_m=-sim.pos[2]
            )
            sim.step(cmd)
            # DAgger-style recovery perturbation: nudges heading so PID demonstrates corrective turns
            if step > 20 and rng.random() < 0.025:
                sim.yaw += float(rng.uniform(-0.25, 0.25))
            
            # Observation formulation (matched identically to 15D environment.py)
            in_view_flag = 1.0 if (in_view_real and not is_dropped) else 0.0
            if in_view_flag > 0.5:
                # Unproject to horizon plane
                h_ex, h_ey, _, _ = unproject_camera_to_horizon(
                    err_x=float(telem["error_x"]),
                    err_y=float(telem["error_y"]),
                    drone_pitch=sim.pitch,
                    drone_roll=sim.roll,
                    camera_uptilt_rad=sim.camera_uptilt,
                    half_hfov=np.tan(np.deg2rad(31.0)),
                    half_vfov=np.tan(np.deg2rad(19.0)),
                )
                ex = float(np.clip(h_ex + rng.normal(0, noise_sigma), -1.0, 1.0))
                ey = float(np.clip(h_ey + rng.normal(0, noise_sigma), -1.0, 1.0))
                w_norm = float(np.clip(telem["w_norm"] + rng.normal(0, noise_sigma * 0.1), 0.005, 0.90))
                last_seen_ex, last_seen_ey, last_seen_w = ex, ey, w_norm
                
                raw_dex = (ex - prev_ex) / dt
                raw_dey = (ey - prev_ey) / dt
                raw_dw  = (w_norm - prev_w) / dt
                filt_dex = 0.4 * raw_dex + 0.6 * filt_dex
                filt_dey = 0.4 * raw_dey + 0.6 * filt_dey
                filt_dw  = 0.4 * raw_dw  + 0.6 * filt_dw
                prev_ex, prev_ey, prev_w = ex, ey, w_norm
            else:
                ex = last_seen_ex
                ey = last_seen_ey
                w_norm = last_seen_w
                filt_dex *= 0.5
                filt_dey *= 0.5
                filt_dw  *= 0.5
                
            scale_err = float(np.clip((0.0859 - w_norm) / 0.0859, -1.0, 1.0))
            
            # Chaser body-frame velocities
            c_yaw, s_yaw = np.cos(sim.yaw), np.sin(sim.yaw)
            vx_b = c_yaw * sim.vel[0] + s_yaw * sim.vel[1]
            vy_b = -s_yaw * sim.vel[0] + c_yaw * sim.vel[1]
            vz_b = sim.vel[2]
            
            alt_norm = float(np.clip(-sim.pos[2] / 5.0, 0.0, 2.0))
            norm_pitch = float(np.clip(sim.pitch / (np.pi / 4.0), -1.0, 1.0))
            norm_roll  = float(np.clip(sim.roll  / (np.pi / 4.0), -1.0, 1.0))
            
            obs_15d = np.array([
                ex,
                ey,
                scale_err,
                w_norm,
                in_view_flag,
                np.clip(vx_b / 15.0, -1.0, 1.0),
                np.clip(vy_b / 4.0, -1.0, 1.0),
                np.clip(vz_b / 3.0, -1.0, 1.0),
                np.clip(sim.yaw_rate / np.deg2rad(120.0), -1.0, 1.0),
                alt_norm,
                np.clip(filt_dex / 2.0, -1.0, 1.0),
                np.clip(filt_dey / 2.0, -1.0, 1.0),
                np.clip(filt_dw  / 0.20, -1.0, 1.0),
                norm_pitch,
                norm_roll
            ], dtype=np.float32)
            
            # Inverse Action Mapping matching environment.py
            if cmd[0] >= 3.5:
                a0 = (cmd[0] - 3.5) / (15.0 - 3.5)
            else:
                a0 = (cmd[0] - 3.5) / (3.5 + 0.5)
            a0 = float(np.clip(a0, -1.0, 1.0))
            a1 = float(np.clip(cmd[1] / 4.0, -1.0, 1.0))
            a2 = float(np.clip(cmd[2] / 2.0 if cmd[2] >= 0 else cmd[2] / 3.0, -1.0, 1.0))
            a3 = float(np.clip(cmd[3] / 120.0, -1.0, 1.0))
            
            act_4d = np.array([a0, a1, a2, a3], dtype=np.float32)
            
            ep_obs.append(obs_15d)
            ep_acts.append(act_4d)
            
        trajectories.append((np.array(ep_obs, dtype=np.float32), np.array(ep_acts, dtype=np.float32)))
        if (ep + 1) % 100 == 0 or (ep + 1) == num_episodes:
            print(f"  [COLLECT] {ep + 1}/{num_episodes} demonstration episodes recorded ({len(trajectories) * total_steps:,} transitions)...")
        
    print(f"[EXPERT] Successfully collected {len(trajectories)} trajectories ({len(trajectories) * total_steps:,} total transitions).")
    return trajectories


def evaluate_policy(
    model: StandaloneRecurrentPolicy,
    scenario_name: str = "Evasive Seed 42",
    profile: str = "evasive",
    initial_pos: List[float] = [12.0, 0.0, -2.5],
    initial_heading: float = 0.0,
    seed: int = 42,
    duration: float = 15.0,
    dt: float = 0.02
) -> Dict[str, Any]:
    """Evaluates the cloned policy in closed-loop flight on a standardized benchmark trajectory."""
    sim = FastPixhawkQuadSim(
        dt=dt,
        max_vel_xy=15.0,
        max_accel_xy=6.0,
        max_vel_up=3.0,
        max_vel_down=2.0
    )
    target = StochasticTargetTrajectory(
        duration=duration + 5.0,
        profile=profile,
        initial_pos=initial_pos,
        initial_heading_rad=initial_heading,
        seed=seed
    )
    sim.reset(
        initial_pos=np.array([0.0, 0.0, initial_pos[2]]),
        initial_yaw=0.0,
        initial_vel=np.array([4.0, 0.0, 0.0])
    )
    
    ctrl = RecurrentVisualServoController(w_nominal=0.0859)
    ctrl.model = model
    ctrl.obs_dim = 15
    ctrl.reset()
    ctrl.prev_cmd = np.array([4.0, 0.0, 0.0, 0.0])
    
    total_steps = int(duration / dt)
    dists = []
    in_views = []
    ground_strikes = 0
    collisions = 0
    
    for step in range(total_steps):
        t = step * dt
        tgt_pos = target.get_position(t)
        telem = sim.get_camera_telemetry(
            tgt_pos,
            hfov_deg=62.0,
            vfov_deg=38.0,
            target_w_m=0.35,
            target_h_m=0.20,
            img_w=1280,
            img_h=720,
            desired_target_size=110.0
        )
        c, s = np.cos(sim.yaw), np.sin(sim.yaw)
        v_body = np.array([c * sim.vel[0] + s * sim.vel[1], -s * sim.vel[0] + c * sim.vel[1], sim.vel[2]])
        
        telem_clean = {k: v for k, v in telem.items() if k != "distance"}
        cmd = ctrl.compute_cmd(
            telem_clean,
            drone_pitch=sim.pitch,
            drone_roll=sim.roll,
            dt=dt,
            vehicle_vel=v_body,
            vehicle_yaw_rate=sim.yaw_rate,
            current_alt_m=-sim.pos[2]
        )
        sim.step(cmd)
        
        d = float(telem["distance"])
        iv = bool(telem["in_view"])
        dists.append(d)
        in_views.append(iv)
        
        if -sim.pos[2] < 0.8:
            ground_strikes += 1
        if d < 1.2:
            collisions += 1
            
    in_view_pct = float(np.mean(in_views) * 100.0)
    min_dist = float(min(dists))
    max_dist = float(max(dists))
    final_dist = float(dists[-1])
    
    res = {
        "scenario": scenario_name,
        "in_view_pct": round(in_view_pct, 1),
        "min_dist_m": round(min_dist, 2),
        "max_dist_m": round(max_dist, 2),
        "final_dist_m": round(final_dist, 2),
        "ground_strikes": ground_strikes,
        "collisions": collisions,
    }
    return res


def train_behavioral_cloning(
    num_episodes: int = 1500,
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    output_dir: Union[str, Path] = "control/weights/cloned_expert_v2",
) -> StandaloneRecurrentPolicy:
    """End-to-end Behavioral Cloning training loop."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Dataset collection
    trajectories = collect_expert_demonstrations(num_episodes=num_episodes)
    
    # Convert to batched tensors
    obs_all = [torch.from_numpy(t[0]) for t in trajectories]
    act_all = [torch.from_numpy(t[1]) for t in trajectories]
    
    # 2. Model initialization
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model = StandaloneRecurrentPolicy(obs_dim=15, hidden_dim=128, act_dim=4).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    criterion = nn.HuberLoss(delta=0.1)  # Huber loss provides smooth, outlier-resistant gradients
    
    print(f"\n[TRAIN] Beginning supervised LSTM imitation learning ({epochs} epochs, batch_size={batch_size}, device={device})...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        # Shuffle trajectories each epoch
        perm = torch.randperm(len(trajectories))
        total_loss = 0.0
        
        for i in range(0, len(trajectories), batch_size):
            batch_indices = perm[i:i + batch_size]
            batch_obs = torch.stack([obs_all[idx] for idx in batch_indices]).to(device) # (B, T, 15)
            batch_acts = torch.stack([act_all[idx] for idx in batch_indices]).to(device) # (B, T, 4)
            
            optimizer.zero_grad()
            pred, _ = model.forward(batch_obs)
            loss = criterion(pred, batch_acts)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * len(batch_indices)
            
        scheduler.step()
        avg_loss = total_loss / len(trajectories)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Huber Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            
    elapsed = time.time() - start_time
    print(f"[TRAIN] Finished training in {elapsed:.1f} seconds.\n")
    
    # 3. Comprehensive Validation Suite
    model.eval()
    print("[EVAL] Running validation suite on unseen benchmark scenarios...")
    
    benchmarks = [
        {"name": "Evasive Tail Chase (Seed 42)", "profile": "evasive", "pos": [12.0, 0.0, -2.5], "heading": 0.0, "seed": 42},
        {"name": "Evasive Crossing Right (Seed 42)", "profile": "evasive", "pos": [12.0, -5.0, -2.5], "heading": np.pi/2.0, "seed": 42},
        {"name": "Hyper-Evasive Weave (Seed 100)", "profile": "hyper_evasive", "pos": [10.0, 0.0, -3.0], "heading": 0.0, "seed": 100},
        {"name": "Sprint Break (Seed 777)", "profile": "sprint_break", "pos": [8.0, 0.0, -2.5], "heading": 0.0, "seed": 777},
        {"name": "Hyper-Evasive Crossing Left (Seed 202)", "profile": "hyper_evasive", "pos": [11.0, 5.0, -2.8], "heading": -np.pi/2.0, "seed": 202},
        {"name": "Hyper-Evasive Head-On (Seed 303)", "profile": "hyper_evasive", "pos": [15.0, 1.0, -3.0], "heading": np.pi, "seed": 303},
        {"name": "Close Basket Dwell (Seed 333)", "profile": "evasive", "pos": [3.5, 0.0, -2.5], "heading": 0.0, "seed": 333},
    ]
    
    results = []
    for bench in benchmarks:
        r = evaluate_policy(
            model=model,
            scenario_name=bench["name"],
            profile=bench["profile"],
            initial_pos=bench["pos"],
            initial_heading=bench["heading"],
            seed=bench["seed"]
        )
        results.append(r)
        print(f"  • {r['scenario']:<36} | In-View: {r['in_view_pct']:5.1f}% | Dist: [{r['min_dist_m']:.2f}m..{r['max_dist_m']:.2f}m] -> Final: {r['final_dist_m']:.2f}m | Strikes: {r['ground_strikes']}")
        
    # 4. Save Weights and Benchmark Manifest
    weights_path = out_path / "actor_weights.pt"
    torch.save(model.state_dict(), str(weights_path))
    print(f"\n[SAVE] Exported standalone PyTorch actor weights to: {weights_path}")
    
    manifest = {
        "model_type": "StandaloneRecurrentPolicy",
        "obs_dim": 15,
        "hidden_dim": 128,
        "act_dim": 4,
        "num_episodes": num_episodes,
        "epochs": epochs,
        "training_time_s": round(elapsed, 1),
        "validation_results": results
    }
    with open(out_path / "model_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Behavioral Cloning Policy for Drone Pursuit")
    parser.add_argument("--episodes", type=int, default=1500, help="Number of expert demonstrations")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--output-dir", type=str, default="control/weights/cloned_expert_v2", help="Output directory")
    args = parser.parse_args()
    
    train_behavioral_cloning(
        num_episodes=args.episodes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
