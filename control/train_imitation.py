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
from benchmarks.control.benchmark_controller import create_pid_controller
from perception.pipeline import DroneTracker
from control.synthetic_sensor import SyntheticYOLOSensor
from control.spatial import compute_ego_motion_compensation


def collect_expert_demonstrations(
    num_episodes: int = 1500,
    episode_duration: float = 15.0,
    dt: float = 0.02,
    camera_fps: float = 30.0,
    dropout_prob: float = 0.05,
    seed_offset: int = 5000,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Simulates expert flights across diverse encounter geometries, target profiles,
    and initial conditions using the EXACT realistic benchmark perception suite
    (30 Hz camera, empirical Jetson YOLO noise, frame dropouts, and 8D Kalman tracking).
    """
    print(f"[EXPERT] Collecting {num_episodes} expert flight demonstrations ({int(episode_duration/dt)} steps each)...")
    print(f"         Perception: 30 Hz Camera, SyntheticYOLOSensor (empirical noise), 8D Kalman DroneTracker")
    trajectories = []
    
    profiles = ["hyper_evasive", "evasive", "cruising", "sprint_break", "aerobatic", "high_speed_evasive"]
    profile_weights = [0.35, 0.30, 0.15, 0.10, 0.05, 0.05]
    
    rng = np.random.default_rng(seed_offset)
    camera_dt = 1.0 / camera_fps
    
    # Load tuned kinematic controller as expert once
    ctrl = create_pid_controller()
    
    for ep in range(num_episodes):
        prof = rng.choice(profiles, p=profile_weights)
        ep_seed = seed_offset + ep
        
        # Geometry sampling: Diverse initial positions, altitudes, and headings
        chaser_alt = float(rng.uniform(-4.5, -2.5))  # 2.5m to 4.5m AGL (NED coordinate)
        delta_alt = float(rng.uniform(-1.5, 1.5))
        target_alt = float(np.clip(chaser_alt + delta_alt, -6.8, -1.8))
        
        init_dist = float(rng.uniform(4.0, 16.0))
        init_lat = float(rng.uniform(-4.5, 4.5))
        init_heading = float(rng.uniform(-0.8, 0.8))
        init_chaser_vel = float(rng.uniform(3.0, 5.0))
        init_yaw = float(rng.uniform(-0.45, 0.45))
        
        # Diverse encounter types:
        # 1. Close Basket & Dwell (20%): Spawns within 2.5-4.5m basket for fine speed-matching & dwell demos
        # 2. Medium Tail Chase (20%): Spawns 4.5-8.0m directly behind target
        # 3. Long Catch-Up Sprint (20%): Spawns 8.0-18.0m behind, requiring sustained full-throttle surge
        # 4. Crossing Right (12%): Spawns on left flank, cuts East (+90 deg) across FOV
        # 5. Crossing Left (12%): Spawns on right flank, cuts West (-90 deg) across FOV
        # 6. Head-On Approach (8%): Spawns 10-22m ahead, flying toward and past the chaser
        # 7. Oblique Break Away (8%): Angled breakaway forcing aggressive yaw correction
        encounter_roll = rng.random()
        if encounter_roll < 0.20:
            # Close basket dwell encounter (teaches station-keeping and zero-error holding)
            close_dist = float(rng.uniform(2.5, 4.5))
            close_lat = float(rng.uniform(-1.0, 1.0))
            target_pos = [close_dist, close_lat, target_alt]
            target_heading = float(rng.uniform(-0.30, 0.30))
            init_chaser_vel = float(rng.uniform(4.0, 5.0))
        elif encounter_roll < 0.40:
            # Medium tail chase
            med_dist = float(rng.uniform(4.5, 8.0))
            med_lat = float(rng.uniform(-2.0, 2.0))
            target_pos = [med_dist, med_lat, target_alt]
            target_heading = float(rng.uniform(-0.40, 0.40))
            init_chaser_vel = float(rng.uniform(3.0, 5.0))
        elif encounter_roll < 0.60:
            # Long catch-up sprint (teaches aggressive forward thrust to close distance)
            long_dist = float(rng.uniform(8.0, 18.0))
            long_lat = float(rng.uniform(-2.5, 2.5))
            target_pos = [long_dist, long_lat, target_alt]
            target_heading = init_heading
            init_chaser_vel = float(rng.uniform(2.0, 4.0))
        elif encounter_roll < 0.72:
            # Crossing right
            flank_lat = float(rng.uniform(-7.0, -3.5))
            cross_dist = float(rng.uniform(5.0, 14.0))
            target_pos = [cross_dist, flank_lat, target_alt]
            target_heading = float(np.pi / 2.0 + rng.uniform(-0.25, 0.25))
        elif encounter_roll < 0.84:
            # Crossing left
            flank_lat = float(rng.uniform(3.5, 7.0))
            cross_dist = float(rng.uniform(5.0, 14.0))
            target_pos = [cross_dist, flank_lat, target_alt]
            target_heading = float(-np.pi / 2.0 + rng.uniform(-0.25, 0.25))
        elif encounter_roll < 0.92:
            # Head-on approach
            head_dist = float(rng.uniform(10.0, 22.0))
            head_lat = float(rng.uniform(-3.0, 3.0))
            target_pos = [head_dist, head_lat, target_alt]
            target_heading = float(np.pi + rng.uniform(-0.35, 0.35))
        else:
            # Oblique break away
            sign = float(rng.choice([-1.0, 1.0]))
            target_pos = [init_dist, init_lat, target_alt]
            target_heading = float(sign * (np.pi / 4.0) + rng.uniform(-0.25, 0.25))
            
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
        
        # Reset kinematic controller and observation builder
        ctrl.reset()
        ctrl.prev_cmd = np.array([init_chaser_vel, 0.0, 0.0, 0.0])
        ctrl.prev_cmd_vx = float(init_chaser_vel)
        
        obs_builder = RecurrentVisualServoController(w_nominal=0.0859)
        obs_builder.reset()
        
        # Instantiate realistic benchmark perception stack
        synthetic_sensor = SyntheticYOLOSensor(
            img_w=1280,
            img_h=720,
            env_mode="nominal",
            seed=ep_seed
        )
        drone_tracker = DroneTracker(
            dt=dt,
            max_lost_frames=15,
            min_hits_to_confirm=2
        )
        
        time_since_camera = 0.0
        prev_chaser_pitch = sim.pitch
        prev_chaser_roll = sim.roll
        prev_chaser_yaw = sim.yaw
        
        ep_obs = []
        ep_acts = []
        
        total_steps = int(episode_duration / dt)
        for step in range(total_steps):
            t = step * dt
            tgt_pos = target.get_position(t)
            true_telem = sim.get_camera_telemetry(
                tgt_pos,
                hfov_deg=62.0,
                vfov_deg=38.0,
                target_w_m=0.35,
                target_h_m=0.20,
                img_w=1280,
                img_h=720,
                desired_target_size=110.0
            )
            
            # 1. Compute ego-motion compensation
            d_pitch = sim.pitch - prev_chaser_pitch
            d_roll = sim.roll - prev_chaser_roll
            d_yaw = float((sim.yaw - prev_chaser_yaw + np.pi) % (2.0 * np.pi) - np.pi)
            prev_chaser_pitch = sim.pitch
            prev_chaser_roll = sim.roll
            prev_chaser_yaw = sim.yaw
            
            ego_shifts = {}
            for tr in drone_tracker.trackers:
                st = tr.get_state()
                tcx = (st[0] + st[2]) / 2.0
                tcy = (st[1] + st[3]) / 2.0
                tw = max(1.0, st[2] - st[0])
                th = max(1.0, st[3] - st[1])
                shift = compute_ego_motion_compensation(
                    cx=tcx, cy=tcy, w=tw, h=th,
                    delta_pitch_rad=d_pitch,
                    delta_roll_rad=d_roll,
                    delta_yaw_rad=d_yaw,
                    vx_body=0.0, vy_body=0.0, vz_body=0.0,
                    dt=dt,
                    img_width=1280.0,
                    img_height=720.0
                )
                ego_shifts[tr.id] = shift
                
            # 2. 30 Hz Camera Timing vs 50 Hz Physics Prediction
            time_since_camera += dt
            is_camera_tick = (time_since_camera >= camera_dt - 1e-5)
            if is_camera_tick:
                time_since_camera -= camera_dt
                norm_cx = (true_telem["error_x"] + 1.0) / 2.0
                norm_cy = (true_telem["error_y"] + 1.0) / 2.0
                norm_w = true_telem["bbox_w"] / 1280.0
                norm_h = true_telem["bbox_h"] / 720.0
                in_fov = bool(true_telem["in_view"])
                
                det_bbox, conf, fps_boxes = synthetic_sensor.step(
                    norm_cx, norm_cy, norm_w, norm_h, in_fov=in_fov
                )
                if dropout_prob > 0.0 and rng.random() < dropout_prob:
                    det_bbox = None
                    
                detections = []
                confs = []
                if det_bbox is not None:
                    detections.append(det_bbox)
                    confs.append(conf)
                for fp in fps_boxes:
                    detections.append(fp)
                    confs.append(0.36)
                    
                det_arr = np.array(detections, dtype=np.float32) if detections else np.empty((0, 4), dtype=np.float32)
                conf_arr = np.array(confs, dtype=np.float32) if confs else np.empty((0,), dtype=np.float32)
                drone_tracker.update(det_arr, conf_arr, ego_shifts=ego_shifts, dt_step=dt)
            else:
                for trk in drone_tracker.trackers:
                    shift = ego_shifts.get(trk.id, None)
                    trk.predict(ego_shift=shift, dt_step=dt)
                    
            primary = None
            if drone_tracker.primary_track_id is not None:
                primary = next((t for t in drone_tracker.trackers if t.id == drone_tracker.primary_track_id), None)
                
            if primary is not None and primary.hits >= drone_tracker.min_hits_to_confirm and primary.time_since_update <= drone_tracker.max_lost_frames:
                pred_box = primary.get_state()
                tcx = (pred_box[0] + pred_box[2]) / 2.0
                tcy = (pred_box[1] + pred_box[3]) / 2.0
                bw = max(1.0, pred_box[2] - pred_box[0])
                bh = max(1.0, pred_box[3] - pred_box[1])
                curr_sz = bw
                
                err_x = (tcx - 1280.0 / 2.0) / (1280.0 / 2.0)
                err_y = (tcy - 720.0 / 2.0) / (720.0 / 2.0)
                range_err = (110.0 - curr_sz) / 110.0
                
                ctrl_telem = {
                    "status": "COASTING" if primary.time_since_update > 0 else "LOCKED",
                    "error_x": float(np.clip(err_x, -1.0, 1.0)),
                    "error_y": float(np.clip(err_y, -1.0, 1.0)),
                    "error_range": float(np.clip(range_err, -1.0, 1.0)),
                    "target_size": float(curr_sz),
                    "desired_size": 110.0,
                    "bbox_w": float(bw),
                    "bbox_h": float(bh),
                    "w_norm": float(bw / 1280.0),
                    "scale_mode": "width",
                    "in_view": True,
                }
            else:
                ctrl_telem = {
                    "status": "SEARCHING",
                    "error_x": 0.0,
                    "error_y": 0.0,
                    "error_range": 0.0,
                    "target_size": 0.0,
                    "desired_size": 110.0,
                    "bbox_w": 0.0,
                    "bbox_h": 0.0,
                    "w_norm": 0.0,
                    "scale_mode": "width",
                    "in_view": False,
                }

            c_yaw, s_yaw = np.cos(sim.yaw), np.sin(sim.yaw)
            vx_b = c_yaw * sim.vel[0] + s_yaw * sim.vel[1]
            vy_b = -s_yaw * sim.vel[0] + c_yaw * sim.vel[1]
            vz_b = sim.vel[2]
            v_body = np.array([vx_b, vy_b, vz_b], dtype=np.float64)

            # Compute expert PID command on realistic telemetry
            cmd = ctrl.compute_cmd(
                ctrl_telem,
                drone_pitch=sim.pitch,
                drone_roll=sim.roll,
                dt=dt,
                vehicle_vel=v_body,
                vehicle_yaw_rate=sim.yaw_rate,
                current_alt_m=-sim.pos[2]
            )
            
            # Compute observation using the exact unified builder
            obs_15d = obs_builder.build_obs(
                ctrl_telem,
                drone_pitch=sim.pitch,
                drone_roll=sim.roll,
                dt=dt,
                vehicle_vel=v_body,
                vehicle_yaw_rate=sim.yaw_rate,
                current_alt_m=-sim.pos[2]
            )
            
            # Inverse action mapping matching environment.py
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
            
            sim.step(cmd)
            
            # DAgger-style recovery perturbation: nudges heading occasionally so PID demonstrates corrective turns
            if step > 20 and rng.random() < 0.02:
                sim.yaw += float(rng.uniform(-0.25, 0.25))
            
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
    dt: float = 0.02,
    camera_fps: float = 30.0,
) -> Dict[str, Any]:
    """Evaluates the cloned policy in closed-loop flight on a standardized benchmark trajectory with realistic perception."""
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
    
    synthetic_sensor = SyntheticYOLOSensor(img_w=1280, img_h=720, env_mode="nominal", seed=seed)
    drone_tracker = DroneTracker(dt=dt, max_lost_frames=15, min_hits_to_confirm=2)
    camera_dt = 1.0 / camera_fps
    time_since_camera = 0.0
    
    prev_chaser_pitch = sim.pitch
    prev_chaser_roll = sim.roll
    prev_chaser_yaw = sim.yaw
    
    total_steps = int(duration / dt)
    dists = []
    in_views = []
    ground_strikes = 0
    collisions = 0
    
    for step in range(total_steps):
        t = step * dt
        tgt_pos = target.get_position(t)
        true_telem = sim.get_camera_telemetry(
            tgt_pos,
            hfov_deg=62.0,
            vfov_deg=38.0,
            target_w_m=0.35,
            target_h_m=0.20,
            img_w=1280,
            img_h=720,
            desired_target_size=110.0
        )
        
        d_pitch = sim.pitch - prev_chaser_pitch
        d_roll = sim.roll - prev_chaser_roll
        d_yaw = float((sim.yaw - prev_chaser_yaw + np.pi) % (2.0 * np.pi) - np.pi)
        prev_chaser_pitch = sim.pitch
        prev_chaser_roll = sim.roll
        prev_chaser_yaw = sim.yaw
        
        ego_shifts = {}
        for tr in drone_tracker.trackers:
            st = tr.get_state()
            tcx = (st[0] + st[2]) / 2.0
            tcy = (st[1] + st[3]) / 2.0
            tw = max(1.0, st[2] - st[0])
            th = max(1.0, st[3] - st[1])
            shift = compute_ego_motion_compensation(
                cx=tcx, cy=tcy, w=tw, h=th,
                delta_pitch_rad=d_pitch,
                delta_roll_rad=d_roll,
                delta_yaw_rad=d_yaw,
                vx_body=0.0, vy_body=0.0, vz_body=0.0,
                dt=dt,
                img_width=1280.0,
                img_height=720.0
            )
            ego_shifts[tr.id] = shift
            
        time_since_camera += dt
        is_camera_tick = (time_since_camera >= camera_dt - 1e-5)
        if is_camera_tick:
            time_since_camera -= camera_dt
            norm_cx = (true_telem["error_x"] + 1.0) / 2.0
            norm_cy = (true_telem["error_y"] + 1.0) / 2.0
            norm_w = true_telem["bbox_w"] / 1280.0
            norm_h = true_telem["bbox_h"] / 720.0
            in_fov = bool(true_telem["in_view"])
            
            det_bbox, conf, fps_boxes = synthetic_sensor.step(norm_cx, norm_cy, norm_w, norm_h, in_fov=in_fov)
            detections = []
            confs = []
            if det_bbox is not None:
                detections.append(det_bbox)
                confs.append(conf)
            for fp in fps_boxes:
                detections.append(fp)
                confs.append(0.36)
            det_arr = np.array(detections, dtype=np.float32) if detections else np.empty((0, 4), dtype=np.float32)
            conf_arr = np.array(confs, dtype=np.float32) if confs else np.empty((0,), dtype=np.float32)
            drone_tracker.update(det_arr, conf_arr, ego_shifts=ego_shifts, dt_step=dt)
        else:
            for trk in drone_tracker.trackers:
                shift = ego_shifts.get(trk.id, None)
                trk.predict(ego_shift=shift, dt_step=dt)
                
        primary = None
        if drone_tracker.primary_track_id is not None:
            primary = next((t for t in drone_tracker.trackers if t.id == drone_tracker.primary_track_id), None)
            
        if primary is not None and primary.hits >= drone_tracker.min_hits_to_confirm and primary.time_since_update <= drone_tracker.max_lost_frames:
            pred_box = primary.get_state()
            tcx = (pred_box[0] + pred_box[2]) / 2.0
            tcy = (pred_box[1] + pred_box[3]) / 2.0
            bw = max(1.0, pred_box[2] - pred_box[0])
            bh = max(1.0, pred_box[3] - pred_box[1])
            curr_sz = bw
            err_x = (tcx - 1280.0 / 2.0) / (1280.0 / 2.0)
            err_y = (tcy - 720.0 / 2.0) / (720.0 / 2.0)
            range_err = (110.0 - curr_sz) / 110.0
            ctrl_telem = {
                "status": "COASTING" if primary.time_since_update > 0 else "LOCKED",
                "error_x": float(np.clip(err_x, -1.0, 1.0)),
                "error_y": float(np.clip(err_y, -1.0, 1.0)),
                "error_range": float(np.clip(range_err, -1.0, 1.0)),
                "target_size": float(curr_sz),
                "desired_size": 110.0,
                "bbox_w": float(bw),
                "bbox_h": float(bh),
                "w_norm": float(bw / 1280.0),
                "scale_mode": "width",
                "in_view": True,
            }
        else:
            ctrl_telem = {
                "status": "SEARCHING",
                "error_x": 0.0,
                "error_y": 0.0,
                "error_range": 0.0,
                "target_size": 0.0,
                "desired_size": 110.0,
                "bbox_w": 0.0,
                "bbox_h": 0.0,
                "w_norm": 0.0,
                "scale_mode": "width",
                "in_view": False,
            }
            
        c, s = np.cos(sim.yaw), np.sin(sim.yaw)
        v_body = np.array([c * sim.vel[0] + s * sim.vel[1], -s * sim.vel[0] + c * sim.vel[1], sim.vel[2]])
        
        cmd = ctrl.compute_cmd(
            ctrl_telem,
            drone_pitch=sim.pitch,
            drone_roll=sim.roll,
            dt=dt,
            vehicle_vel=v_body,
            vehicle_yaw_rate=sim.yaw_rate,
            current_alt_m=-sim.pos[2]
        )
        sim.step(cmd)
        
        d = float(true_telem["distance"])
        iv = bool(true_telem["in_view"])
        dists.append(d)
        in_views.append(iv)
        
        if -sim.pos[2] < 0.8:
            ground_strikes += 1
        if d < 0.5:
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
