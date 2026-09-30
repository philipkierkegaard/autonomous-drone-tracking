#!/usr/bin/env python3
"""
Gymnasium Drone Pursuit & Interception Environment.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Encapsulates FastPixhawkQuadSim and StochasticTargetTrajectory into a standard
Gymnasium interface supporting:
1. 70/30 Training Distribution:
   - 70% Canonical Full Flights (Tail-chase, Crossing, Head-on, Oblique) from cold hover.
   - 30% In Medias Res (FOV Border Escape, Danger-Close, High-Speed Fly-By, Blind Recovery).
2. Physical Flight Profile Sampling (Cruising 30%, Evasive 45%, Hyper-Evasive 25%).
3. Frustum Slack Cone & Lead-Pursuit Reward Shaping:
   - Range closure, standoff basket (+1.0 bonus for 5m-7m in FOV), velocity matching,
     soft FOV boundary repeller (|e| > 0.70), and anti-jerk smoothness penalty.
4. Perception Domain Randomization (Sim-to-Real):
   - Centroid noise, bounding box scale jitter, and random frame dropouts.
5. Aerospace Stopping Criteria:
   - Collision breach (d < 2.5m), ground strike (z < 1.0m AGL), lost-target timeout (2.0s).
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.simulation import FastPixhawkQuadSim
from control.trajectory import StochasticTargetTrajectory, PROFILES, FlightProfile
from control.dataset_generator import EncounterType, PursuitDatasetGenerator
from control.spatial import unproject_camera_to_horizon


class DronePursuitEnv(gym.Env):
    """
    Continuous 50 Hz Visual Servoing Environment for Autonomous Drone Interception.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        dt: float = 0.02,
        max_duration: float = 25.0,
        standoff_nominal_m: float = 6.0,
        standoff_min_m: float = 5.0,
        standoff_max_m: float = 7.0,
        collision_dist_m: float = 1.2,
        min_altitude_agl_m: float = 1.0,
        safe_altitude_cushion_m: float = 2.0,
        lost_target_timeout_s: float = 3.5,
        in_medias_res_ratio: float = 0.30,
        domain_randomization: bool = True,
        frame_drop_rate: float = 0.05,
        img_w: int = 640,
        img_h: int = 480,
        hfov_deg: float = 60.0,
        vfov_deg: float = 45.0,
    ):
        super().__init__()

        self.dt = dt
        self.max_duration = max_duration
        self.max_steps = int(round(max_duration / dt))
        self.standoff_nominal = standoff_nominal_m
        self.standoff_min = standoff_min_m
        self.standoff_max = standoff_max_m
        self.standoff_sigma = 1.0   # Gaussian standard deviation (m) around rear trail anchor
        self.trail_basket_radius = 1.5  # Radius of rear trail anchor basket for lock dwell (m)
        self.collision_dist = collision_dist_m
        self.min_altitude_agl = min_altitude_agl_m
        self.safe_altitude_cushion = safe_altitude_cushion_m
        self.lost_target_timeout = lost_target_timeout_s
        self.in_medias_res_ratio = in_medias_res_ratio
        self.domain_randomization = domain_randomization
        self.frame_drop_rate = frame_drop_rate
        self.img_w = img_w
        self.img_h = img_h
        self.half_hfov = float(np.tan(np.deg2rad(hfov_deg / 2.0)))
        self.half_vfov = float(np.tan(np.deg2rad(vfov_deg / 2.0)))
        self.w_nominal = 0.0505 # Target bbox width fraction at 6.0m standoff (~32.3 pixels)
        self.max_decel = 2.8    # Calibrated quadrotor braking deceleration (m/s^2)

        # Underlying calibrated physics simulator (50 Hz, 60ms delay, drag, kinodynamics)
        self.sim = FastPixhawkQuadSim(dt=dt)
        self.scenario_gen = PursuitDatasetGenerator()

        # -------------------------------------------------------------
        # Action Space: 4 Continuous Setpoints in Body Frame
        # a0: Forward Velocity Setpoint   [-3.0 m/s, 18.0 m/s]
        # a1: Lateral Velocity Setpoint   [-6.0 m/s, +6.0 m/s]
        # a2: Vertical Velocity Setpoint  [-4.0 m/s (climb), +2.5 m/s (dive)]
        # a3: Yaw Rate Setpoint           [-120 deg/s, +120 deg/s]
        # -------------------------------------------------------------
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # -------------------------------------------------------------
        # Observation Space: 13D Normalized State Vector (Generation 5)
        # [0] ex: Normalized Centroid Horizontal Error   [-1, 1]
        # [1] ey: Normalized Centroid Vertical Error     [-1, 1]
        # [2] scale_err: Standoff Scale Error            [-1, 1] (0 = 6m, >0 far, <0 close)
        # [3] w_norm: Normalized Bounding Box Width      [0, 1]
        # [4] in_fov: Target Visibility Flag             {0.0, 1.0}
        # [5] vx_body: Normalized Forward Velocity       [-1, 1] (norm by 18 m/s)
        # [6] vy_body: Normalized Lateral Velocity       [-1, 1] (norm by 6 m/s)
        # [7] vz_body: Normalized Vertical Velocity      [-1, 1] (norm by 4 m/s)
        # [8] yaw_rate: Normalized Yaw Rate              [-1, 1] (norm by 120 deg/s)
        # [9] alt_norm: Normalized Altitude AGL          [0, 2] (norm by 5.0m)
        # [10] d_ex: Optical Azimuth Rate                [-1, 1] (norm by 2.0 s^-1)
        # [11] d_ey: Optical Elevation Rate              [-1, 1] (norm by 2.0 s^-1)
        # [12] d_w: Bounding Box Expansion Rate          [-1, 1] (norm by 0.20 s^-1)
        # -------------------------------------------------------------
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )

        # Internal state tracking
        self.target_oracle: Optional[StochasticTargetTrajectory] = None
        self.step_count = 0
        self.time_lost = 0.0
        self.last_action = np.zeros(4, dtype=np.float32)
        self.last_seen_ex = 0.0
        self.last_seen_ey = 0.0
        self.last_seen_w = 0.05
        self.last_seen_h = 0.03
        self.cumulative_firing_window_s = 0.0
        self.initial_distance = 15.0
        self.prev_distance_error = 9.0
        self.prev_trail_error = 9.0
        self.last_target_heading = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        # Optical Rate Filtering Memory (Causal 3-step exponential smoothing)
        self.prev_ex = 0.0
        self.prev_ey = 0.0
        self.prev_w = 0.05
        self.filtered_d_ex = 0.0
        self.filtered_d_ey = 0.0
        self.filtered_d_w = 0.0
        self.was_in_view_prev = True

        # Mission Objective: Reach 5.0m - 7.0m basket & maintain continuous lock
        self.dwell_time_required = 3.0
        self.continuous_lock_dwell_s = 0.0
        self.time_to_first_lock: Optional[float] = None
        self.lock_achieved = False
        self.milestone_1s_awarded = False
        self.milestone_2s_awarded = False

        # Flight Dynamics & Actuator Slew-Rate Memory
        self.last_physical_cmd = np.zeros(4, dtype=np.float64)

    def _map_action_to_physical(self, action: np.ndarray) -> np.ndarray:
        """
        Maps normalized actions [-1, 1]^4 to physical velocity setpoints for FastPixhawkQuadSim,
        applying aerospace slew-rate limiting (kinematic acceleration limits) matching the
        physical quadrotor flight envelope to eliminate bang-bang control chatter.
        """
        a = np.clip(action, -1.0, 1.0)
        # a0: [-1, 1] -> [-5.0 m/s (active reverse/braking), +15.0 m/s (high-speed sprint)]
        # Neutral stick (a0 = 0.0) commands 5.0 m/s, aligning closely with cruising target speed.
        target_vx = 5.0 + a[0] * 10.0

        # a1: [-1, 1] -> [-6.0 m/s (strafe left), +6.0 m/s (strafe right)]
        target_vy = a[1] * 6.0

        # a2: [-1, 1] -> [-4.0 m/s (max climb), +2.5 m/s (max descend)] (NED coordinates)
        target_vz = -0.75 + a[2] * 3.25

        # a3: [-1, 1] -> [-120 deg/s, +120 deg/s] yaw rate
        target_yaw_deg = a[3] * 120.0

        # Slew-Rate Limiting across all 4 flight axes (matched to Pixhawk quad envelope)
        # Axis 0: Surge acceleration (6.5 m/s^2) and deceleration (5.0 m/s^2 for agile braking)
        max_accel_x = 6.5 * self.dt
        max_decel_x = 5.0 * self.dt
        if target_vx < self.last_physical_cmd[0]:
            cmd_vx = max(target_vx, self.last_physical_cmd[0] - max_decel_x)
        else:
            cmd_vx = min(target_vx, self.last_physical_cmd[0] + max_accel_x)

        # Axis 1: Lateral sway acceleration (5.0 m/s^2)
        max_accel_y = 5.0 * self.dt
        cmd_vy = float(np.clip(target_vy, self.last_physical_cmd[1] - max_accel_y, self.last_physical_cmd[1] + max_accel_y))

        # Axis 2: Vertical heave acceleration (3.0 m/s^2)
        max_accel_z = 3.0 * self.dt
        cmd_vz = float(np.clip(target_vz, self.last_physical_cmd[2] - max_accel_z, self.last_physical_cmd[2] + max_accel_z))

        # Axis 3: Yaw rate angular acceleration (180 deg/s^2 -> 3.6 deg/s per 0.02s step)
        max_accel_yaw = 180.0 * self.dt
        cmd_yaw_deg = float(np.clip(target_yaw_deg, self.last_physical_cmd[3] - max_accel_yaw, self.last_physical_cmd[3] + max_accel_yaw))

        phys_cmd = np.array([cmd_vx, cmd_vy, cmd_vz, cmd_yaw_deg], dtype=np.float64)
        self.last_physical_cmd = phys_cmd.copy()
        return phys_cmd

    def _sample_scenario_config(self) -> Dict[str, Any]:
        """
        Samples an encounter according to the 70/30 distribution and physical profile weights.
        """
        rng = self.np_random

        # Target Flight Profile Split: 30% Cruising, 45% Evasive, 25% Hyper-Evasive
        profile_choice = rng.choice(["cruising", "evasive", "hyper_evasive"], p=[0.30, 0.45, 0.25])

        # 70% Canonical Full Flights vs. 30% In Medias Res
        is_in_medias_res = rng.random() < self.in_medias_res_ratio

        if not is_in_medias_res:
            # Canonical Full Flight (Cold hover start at nominal altitude)
            encounter = str(rng.choice([
                "tail_chase",
                "crossing_right",
                "crossing_left",
                "head_on",
                "oblique_away",
                "oblique_inward",
            ], p=[0.25, 0.175, 0.175, 0.20, 0.10, 0.10]))

            chaser_pos = np.array([0.0, 0.0, -rng.uniform(2.5, 4.0)], dtype=np.float64)
            chaser_yaw = 0.0
            # Realistic initial forward cruising speed matching target encounter (5.0 to 8.5 m/s)
            chaser_vel = np.array([rng.uniform(5.0, 8.5), 0.0, 0.0], dtype=np.float64)

            # Encounter-adaptive spawn distance:
            # - Away encounters (tail_chase, oblique_away): Target is fleeing forward.
            #   50% close spawn (5.5m - 8.0m) trains basket retention & 3.0s lock dwell.
            #   50% far sprint spawn (10.0m - 16.0m) trains long-range max-throttle gap closure.
            # - Inward / crossing encounters (head_on, crossing, oblique_inward): Target closing or crossing at 11 m/s.
            #   Wide spawn (12.0m - 16.0m) provides physical reaction horizon (>1.0s) and wide FOV arc.
            if encounter in ["tail_chase", "oblique_away"]:
                if rng.random() < 0.50:
                    spawn_dist_range = (5.5, 8.0)
                else:
                    spawn_dist_range = (10.0, 16.0)
            else:
                spawn_dist_range = (12.0, 16.0)

            target_pos, dist, az, el = self.scenario_gen.sample_initial_target_pos(
                chaser_pos=chaser_pos,
                chaser_yaw=chaser_yaw,
                rng=rng,
                distance_range=spawn_dist_range,
                azimuth_deg_range=(-15.0, 15.0),
                elevation_deg_range=(-10.0, 10.0)
            )

            target_heading = self.scenario_gen.compute_encounter_heading(
                chaser_pos=chaser_pos,
                target_pos=target_pos,
                encounter_type=encounter,
                rng=rng
            )

            return {
                "mode": "canonical",
                "profile": profile_choice,
                "chaser_pos": chaser_pos,
                "chaser_yaw": chaser_yaw,
                "chaser_vel": chaser_vel,
                "target_pos": target_pos,
                "target_heading": target_heading,
                "initial_time_lost": 0.0,
            }

        else:
            # 30% In Medias Res: Stressed / Non-Equilibrium Initial Conditions
            sub_mode = rng.choice(["fov_border", "danger_close", "fly_by", "blind_recovery"], p=[0.35, 0.25, 0.25, 0.15])

            chaser_pos = np.array([0.0, 0.0, -rng.uniform(2.0, 5.0)], dtype=np.float64)
            chaser_yaw = rng.uniform(-np.pi, np.pi)

            if sub_mode == "fov_border":
                # Target positioned at extreme FOV edge moving outwards
                edge_az = rng.choice([-24.0, 24.0]) + rng.uniform(-2.0, 2.0)
                edge_el = rng.uniform(-10.0, 10.0)
                dist = rng.uniform(6.0, 12.0)

                target_pos, _, _, _ = self.scenario_gen.sample_initial_target_pos(
                    chaser_pos=chaser_pos, chaser_yaw=chaser_yaw, rng=rng,
                    distance_range=(dist, dist),
                    azimuth_deg_range=(edge_az, edge_az),
                    elevation_deg_range=(edge_el, edge_el)
                )
                # Heading fleeing outwards
                outward_sign = 1.0 if edge_az > 0 else -1.0
                target_heading = float(chaser_yaw + outward_sign * np.deg2rad(90.0) + rng.uniform(-0.2, 0.2))
                chaser_vel = np.array([rng.uniform(2.0, 5.0), 0.0, 0.0])

                return {
                    "mode": "in_medias_res_fov_border",
                    "profile": profile_choice,
                    "chaser_pos": chaser_pos,
                    "chaser_yaw": chaser_yaw,
                    "chaser_vel": chaser_vel,
                    "target_pos": target_pos,
                    "target_heading": target_heading,
                    "initial_time_lost": 0.0,
                }

            elif sub_mode == "danger_close":
                # Overshoot recovery: target is close (4.5m - 6.0m) with closing speed requiring braking
                dist = rng.uniform(4.5, 6.0)
                target_pos, _, _, _ = self.scenario_gen.sample_initial_target_pos(
                    chaser_pos=chaser_pos, chaser_yaw=chaser_yaw, rng=rng,
                    distance_range=(dist, dist),
                    azimuth_deg_range=(-10.0, 10.0),
                    elevation_deg_range=(-8.0, 8.0)
                )
                target_heading = float(chaser_yaw + rng.uniform(-0.3, 0.3))
                # Chaser has forward velocity (testing active deceleration / flare recovery)
                c, s = np.cos(chaser_yaw), np.sin(chaser_yaw)
                v_fwd = rng.uniform(4.0, 7.0)
                chaser_vel = np.array([c * v_fwd, s * v_fwd, 0.0])

                return {
                    "mode": "in_medias_res_danger_close",
                    "profile": profile_choice,
                    "chaser_pos": chaser_pos,
                    "chaser_yaw": chaser_yaw,
                    "chaser_vel": chaser_vel,
                    "target_pos": target_pos,
                    "target_heading": target_heading,
                    "initial_time_lost": 0.0,
                }

            elif sub_mode == "fly_by":
                # Target passes across perpendicular at high speed (10-14 m/s)
                dist = rng.uniform(5.0, 9.0)
                target_pos, _, _, _ = self.scenario_gen.sample_initial_target_pos(
                    chaser_pos=chaser_pos, chaser_yaw=chaser_yaw, rng=rng,
                    distance_range=(dist, dist),
                    azimuth_deg_range=(-15.0, 15.0),
                    elevation_deg_range=(-8.0, 8.0)
                )
                cross_sign = rng.choice([-1.0, 1.0])
                target_heading = float(chaser_yaw + cross_sign * np.pi / 2.0)
                chaser_vel = np.zeros(3)

                return {
                    "mode": "in_medias_res_fly_by",
                    "profile": "hyper_evasive",
                    "chaser_pos": chaser_pos,
                    "chaser_yaw": chaser_yaw,
                    "chaser_vel": chaser_vel,
                    "target_pos": target_pos,
                    "target_heading": target_heading,
                    "initial_time_lost": 0.0,
                }

            else:  # blind_recovery
                # Target just broke FOV boundary (out of sight), lost timer active
                dist = rng.uniform(7.0, 12.0)
                blind_az = rng.choice([-32.0, 32.0])  # Just beyond 30 deg half-HFOV
                target_pos, _, _, _ = self.scenario_gen.sample_initial_target_pos(
                    chaser_pos=chaser_pos, chaser_yaw=chaser_yaw, rng=rng,
                    distance_range=(dist, dist),
                    azimuth_deg_range=(blind_az, blind_az),
                    elevation_deg_range=(-5.0, 5.0)
                )
                target_heading = float(chaser_yaw + rng.uniform(-0.5, 0.5))
                chaser_vel = np.array([rng.uniform(2.0, 4.0), 0.0, 0.0])

                return {
                    "mode": "in_medias_res_blind_recovery",
                    "profile": profile_choice,
                    "chaser_pos": chaser_pos,
                    "chaser_yaw": chaser_yaw,
                    "chaser_vel": chaser_vel,
                    "target_pos": target_pos,
                    "target_heading": target_heading,
                    "initial_time_lost": 0.4,  # Starts with 0.4s already elapsed
                }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        cfg = self._sample_scenario_config()

        # Reset simulator
        self.sim.reset(
            initial_pos=cfg["chaser_pos"],
            initial_yaw=cfg["chaser_yaw"]
        )
        self.sim.vel = cfg["chaser_vel"].copy()

        # Initialize target B-spline oracle
        seed_val = int(self.np_random.integers(0, 1_000_000))
        self.target_oracle = StochasticTargetTrajectory(
            duration=self.max_duration + 5.0,
            profile=cfg["profile"],
            initial_pos=cfg["target_pos"],
            initial_heading_rad=cfg["target_heading"],
            seed=seed_val
        )

        self.step_count = 0
        self.time_lost = cfg["initial_time_lost"]
        self.last_action = np.zeros(4, dtype=np.float32)
        self.last_physical_cmd = np.array([self.sim.vel[0], 0.0, 0.0, 0.0], dtype=np.float64)
        self.cumulative_firing_window_s = 0.0
        self.continuous_lock_dwell_s = 0.0
        self.time_to_first_lock = None
        self.lock_achieved = False
        self.milestone_1s_awarded = False
        self.milestone_2s_awarded = False

        # Project initial target state to camera sensor
        t_sim = 0.0
        target_pos_w, target_vel_w, _ = self.target_oracle.get_state(t_sim)
        ex, ey, in_view, dist, bw, bh = self.sim.project_to_camera(
            target_pos_w, return_bbox=True, img_w=self.img_w, img_h=self.img_h
        )
        self.initial_distance = float(dist)

        # Store last seen observation
        if in_view:
            h_ex, h_ey, _, _ = unproject_camera_to_horizon(
                err_x=float(ex),
                err_y=float(ey),
                drone_pitch=self.sim.pitch,
                drone_roll=self.sim.roll,
                camera_uptilt_rad=self.sim.camera_uptilt,
                half_hfov=self.half_hfov,
                half_vfov=self.half_vfov,
            )
            self.last_seen_ex = float(h_ex)
            self.last_seen_ey = float(h_ey)
            self.last_seen_w = float(bw / self.img_w)
            self.last_seen_h = float(bh / self.img_h)
        else:
            self.last_seen_ex = float(np.clip(ex, -1.0, 1.0))
            self.last_seen_ey = float(np.clip(ey, -1.0, 1.0))
            self.last_seen_w = 0.05
            self.last_seen_h = 0.03

        # Target heading vector in horizontal plane
        self.last_target_heading = np.array([np.cos(cfg["target_heading"]), np.sin(cfg["target_heading"]), 0.0], dtype=np.float64)
        target_speed_xy = float(np.linalg.norm(target_vel_w[:2]))
        if target_speed_xy > 0.3:
            h_xy = target_vel_w[:2] / target_speed_xy
            self.last_target_heading = np.array([h_xy[0], h_xy[1], 0.0], dtype=np.float64)

        trail_pos_w = target_pos_w - self.standoff_nominal * self.last_target_heading
        self.prev_trail_error = float(np.linalg.norm(self.sim.pos - trail_pos_w))
        self.prev_distance_error = abs(self.initial_distance - self.standoff_nominal)

        # Optical Rate Filtering Initialization
        self.prev_ex = float(self.last_seen_ex)
        self.prev_ey = float(self.last_seen_ey)
        self.prev_w = float(self.last_seen_w)
        self.filtered_d_ex = 0.0
        self.filtered_d_ey = 0.0
        self.filtered_d_w = 0.0
        self.was_in_view_prev = bool(in_view)

        obs = self._get_observation(ex, ey, in_view, bw, bh)
        info = {
            "mode": cfg["mode"],
            "profile": cfg["profile"],
            "initial_distance": self.initial_distance,
            "dist_m": self.initial_distance,
            "d_trail_m": self.prev_trail_error,
        }
        return obs, info

    def _get_observation(
        self,
        raw_ex: float,
        raw_ey: float,
        raw_in_view: bool,
        raw_bw: float,
        raw_bh: float
    ) -> np.ndarray:
        """
        Constructs the 13D normalized state vector with SO(3) attitude decoupling,
        filtered optical rates (azimuth rate, elevation rate, bbox expansion rate),
        and optional Sim-to-Real domain randomization.
        """
        in_fov = raw_in_view

        # Sim-to-Real Domain Randomization (Perception dropout and Gaussian noise)
        if self.domain_randomization and in_fov:
            # 1. Random frame drop (dropout)
            if self.np_random.random() < self.frame_drop_rate:
                in_fov = False

        if in_fov:
            # SO(3) Attitude Decoupling: Un-roll & un-pitch camera frame errors to horizon frame
            h_ex, h_ey, _, _ = unproject_camera_to_horizon(
                err_x=float(raw_ex),
                err_y=float(raw_ey),
                drone_pitch=self.sim.pitch,
                drone_roll=self.sim.roll,
                camera_uptilt_rad=self.sim.camera_uptilt,
                half_hfov=self.half_hfov,
                half_vfov=self.half_vfov,
            )
            ex = float(h_ex)
            ey = float(h_ey)
            w_norm = float(raw_bw / self.img_w)
            h_norm = float(raw_bh / self.img_h)

            if self.domain_randomization:
                # 2. Centroid jitter noise on horizon angles
                ex += float(self.np_random.normal(0.0, 0.02))
                ey += float(self.np_random.normal(0.0, 0.02))
                w_norm = max(0.01, w_norm + float(self.np_random.normal(0.0, 0.02 * w_norm)))
                h_norm = max(0.01, h_norm + float(self.np_random.normal(0.0, 0.02 * h_norm)))

            ex = float(np.clip(ex, -1.0, 1.0))
            ey = float(np.clip(ey, -1.0, 1.0))
            self.last_seen_ex = ex
            self.last_seen_ey = ey
            self.last_seen_w = w_norm
            self.last_seen_h = h_norm
            flag = 1.0

            # Causal exponential filter for optical rates (alpha = 0.4 suppresses noise while preserving 1-2 Hz turns)
            if self.was_in_view_prev:
                raw_d_ex = (ex - self.prev_ex) / self.dt
                raw_d_ey = (ey - self.prev_ey) / self.dt
                raw_d_w  = (w_norm - self.prev_w) / self.dt
            else:
                raw_d_ex = 0.0
                raw_d_ey = 0.0
                raw_d_w  = 0.0

            alpha_rate = 0.4
            self.filtered_d_ex = alpha_rate * raw_d_ex + (1.0 - alpha_rate) * self.filtered_d_ex
            self.filtered_d_ey = alpha_rate * raw_d_ey + (1.0 - alpha_rate) * self.filtered_d_ey
            self.filtered_d_w  = alpha_rate * raw_d_w  + (1.0 - alpha_rate) * self.filtered_d_w

            self.prev_ex = ex
            self.prev_ey = ey
            self.prev_w = w_norm
            self.was_in_view_prev = True

        else:
            # Target is lost / occluded: Observation holds last seen track with flag = 0
            ex = self.last_seen_ex
            ey = self.last_seen_ey
            w_norm = self.last_seen_w
            h_norm = self.last_seen_h
            flag = 0.0

            self.filtered_d_ex *= 0.5
            self.filtered_d_ey *= 0.5
            self.filtered_d_w  *= 0.5
            self.was_in_view_prev = False

        # Chaser body-frame velocities
        c, s = np.cos(self.sim.yaw), np.sin(self.sim.yaw)
        vx_body = c * self.sim.vel[0] + s * self.sim.vel[1]
        vy_body = -s * self.sim.vel[0] + c * self.sim.vel[1]
        vz_body = self.sim.vel[2]

        # Normalized Standoff Scale Error [-1, 1]
        # (w_nominal - w_norm) / w_nominal: >0 is too far (advance), 0 is perfect 6m standoff, <0 is too close (brake!)
        scale_err = float(np.clip((self.w_nominal - w_norm) / self.w_nominal, -1.0, 1.0))

        # Normalized altitude AGL (NED: z is negative above ground)
        alt_agl = -self.sim.pos[2]
        alt_norm = float(np.clip(alt_agl / 5.0, 0.0, 2.0))

        # Normalized optical rates: d_ex and d_ey clipped [-1, 1] (norm by 2.0 s^-1), d_w norm by 0.20 s^-1
        norm_d_ex = float(np.clip(self.filtered_d_ex / 2.0, -1.0, 1.0))
        norm_d_ey = float(np.clip(self.filtered_d_ey / 2.0, -1.0, 1.0))
        norm_d_w  = float(np.clip(self.filtered_d_w  / 0.20, -1.0, 1.0))

        obs = np.array([
            ex,
            ey,
            scale_err,
            w_norm,
            flag,
            np.clip(vx_body / 18.0, -1.0, 1.0),
            np.clip(vy_body / 6.0, -1.0, 1.0),
            np.clip(vz_body / 4.0, -1.0, 1.0),
            np.clip(self.sim.yaw_rate / self.sim.max_yaw_rate, -1.0, 1.0),
            alt_norm,
            norm_d_ex,
            norm_d_ey,
            norm_d_w,
        ], dtype=np.float32)

        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        t_sim = self.step_count * self.dt

        # 1. Physics integration in FastPixhawkQuadSim
        phys_action = self._map_action_to_physical(action)
        self.sim.step(phys_action)

        # 2. Query ground truth target state
        target_pos_w, target_vel_w, _ = self.target_oracle.get_state(t_sim)

        # 3. Project to tilted camera sensor
        raw_ex, raw_ey, raw_in_view, dist, raw_bw, raw_bh = self.sim.project_to_camera(
            target_pos_w, return_bbox=True, img_w=self.img_w, img_h=self.img_h
        )

        # Update target heading vector in horizontal plane
        target_speed_xy = float(np.linalg.norm(target_vel_w[:2]))
        if target_speed_xy > 0.3:
            h_xy = target_vel_w[:2] / target_speed_xy
            self.last_target_heading = np.array([h_xy[0], h_xy[1], 0.0], dtype=np.float64)

        # 4. Target Standoff & Formation Geometry
        dist_err = abs(dist - self.standoff_nominal)
        trail_pos_w = target_pos_w - self.standoff_nominal * self.last_target_heading
        d_trail = float(np.linalg.norm(self.sim.pos - trail_pos_w))

        # Check re-acquisition transition
        was_lost_prev = (self.time_lost > 0.0)

        # Update lost target timer
        if not raw_in_view:
            self.time_lost += self.dt
        else:
            self.time_lost = 0.0

        # Construct 13D next observation
        obs = self._get_observation(raw_ex, raw_ey, raw_in_view, raw_bw, raw_bh)

        # Engagement Basket Condition: 5.0m <= dist <= 7.0m AND visible in camera FOV
        # Matches benchmark evaluation standard exactly!
        in_basket = (self.standoff_min <= dist <= self.standoff_max) and raw_in_view
        r_milestone = 0.0
        r_sustain = 0.0

        if in_basket:
            self.cumulative_firing_window_s += self.dt
            self.continuous_lock_dwell_s = min(self.dwell_time_required, self.continuous_lock_dwell_s + self.dt)

            # Milestone 1: 1.0s continuous lock (+25.0)
            if self.continuous_lock_dwell_s >= 1.0 and not self.milestone_1s_awarded:
                self.milestone_1s_awarded = True
                r_milestone += 25.0

            # Milestone 2: 2.0s continuous lock (+50.0)
            if self.continuous_lock_dwell_s >= 2.0 and not self.milestone_2s_awarded:
                self.milestone_2s_awarded = True
                r_milestone += 50.0

            # Goal: 3.0s continuous lock (+150.0 jackpot, plus ongoing sustain bonus)
            if self.continuous_lock_dwell_s >= self.dwell_time_required:
                if not self.lock_achieved:
                    self.lock_achieved = True
                    self.time_to_first_lock = float(max(0.0, self.step_count * self.dt - self.dwell_time_required))
                    r_milestone += 150.0
                # Ongoing formation sustain reward: +5.0/s for holding lock through evasive maneuvers
                r_sustain = 0.1
        else:
            # Grace decay (leaky integrator)
            self.continuous_lock_dwell_s = max(0.0, self.continuous_lock_dwell_s - 1.5 * self.dt)
            if self.continuous_lock_dwell_s < 1.0:
                self.milestone_1s_awarded = False
            if self.continuous_lock_dwell_s < 2.0:
                self.milestone_2s_awarded = False

        # 5. Check Termination & Truncation Criteria
        terminated = False
        truncated = False
        termination_reason = "running"
        terminal_reward = 0.0

        # Successful Target Interception Lock (3.0s continuous lock achieved)
        if self.continuous_lock_dwell_s >= self.dwell_time_required:
            terminated = True
            termination_reason = "lock_dwell_achieved"
            terminal_reward = 150.0

        # Physical Collision Breach (d < 1.2m) — True physical collision radius
        elif dist < self.collision_dist:
            terminated = True
            termination_reason = "collision"
            terminal_reward = -50.0

        # Altitude Floor Breach (z > -1.0m in NED means altitude < 1.0m AGL)
        elif self.sim.pos[2] > -self.min_altitude_agl:
            terminated = True
            termination_reason = "ground_strike"
            terminal_reward = -50.0

        # Target Lost Timeout (out of FOV for > 3.5s)
        elif self.time_lost > self.lost_target_timeout:
            terminated = True
            termination_reason = "lost_target_timeout"
            terminal_reward = -30.0

        # Time Horizon Reached (25.0 seconds = 1250 steps)
        elif self.step_count >= self.max_steps:
            truncated = True
            termination_reason = "max_horizon"

        # -------------------------------------------------------------
        # 6. Generation 5 Calibrated Reward Shaping
        # -------------------------------------------------------------
        # A. Visual Acquisition & Sightline Tracking
        if raw_in_view:
            r_in_view = 1.0
            r_center = 1.5 * max(0.0, 1.0 - (obs[0]**2 + obs[1]**2) / (0.45**2))
        else:
            r_in_view = 0.0
            r_center = 0.0
        r_tracking_base = r_in_view + r_center

        # B. Multi-Scale Continuous Trail Anchor Potential Field
        if raw_in_view:
            # Wide basin (sigma = 3.5m) pulls drone relentlessly toward rear wake anchor
            r_wide = 3.0 * float(np.exp(-(d_trail ** 2) / (2.0 * (3.5 ** 2))))
            # Tight core (sigma = 1.2m) provides steep precision centering at 6.0m rear wake
            r_tight = 6.0 * float(np.exp(-(d_trail ** 2) / (2.0 * (1.2 ** 2))))
            r_trail_potential = r_wide + r_tight

            # Distance Closing Rate Progress toward rear trail anchor
            trail_closing_rate = (self.prev_trail_error - d_trail) / self.dt
            r_progress = 0.5 * float(np.clip(trail_closing_rate, -1.0, 5.0))
        else:
            r_trail_potential = 0.0
            r_progress = 0.0

        r_basket = r_trail_potential + r_progress

        # C. Out-of-FOV Sightline Recovery Incentive
        # Clean potential-invariant penalty when lost: encourages finding target without oscillatory spinning
        r_recovery = 0.0 if raw_in_view else -0.10

        # D. Direct In-Basket Continuous Dwell Holding Reward:
        # Awards +2.0/step (+100.0/s) continuously while inside the exact 5.0m - 7.0m benchmark basket.
        # Creates an immediate step-level incentive to avoid creeping below 5.0m or drifting past 7.0m.
        r_basket_dwell = 2.0 if in_basket else 0.0

        # E. Sprint Time Incentive:
        r_time = 0.0 if self.milestone_1s_awarded else -0.05

        self.prev_trail_error = d_trail
        self.prev_distance_error = dist_err

        # F. Proximity & Altitude Soft Barriers
        # Continuous two-tier proximity repeller:
        # - Tier 1: Soft under-standoff cushion between 3.0m and 5.0m prevents creeping inside the 5.0m basket floor.
        # - Tier 2: Steep quadratic barrier below 3.0m prevents physical collision breach (< 1.2m).
        if dist < 3.0 and dist >= self.collision_dist:
            p_proximity = 2.0 + 4.0 * ((3.0 - dist) / (3.0 - self.collision_dist)) ** 2
        elif dist < 5.0 and dist >= 3.0:
            p_proximity = 2.0 * ((5.0 - dist) / 2.0) ** 2
        else:
            p_proximity = 0.0

        alt_agl = -self.sim.pos[2]
        p_low_alt = 1.5 * max(0.0, self.safe_altitude_cushion - alt_agl)**2

        # G. Flight Dynamics Regularization
        # 1. Action Slew / Jerk Penalty
        action_diff_sq = float(np.sum((action - self.last_action)**2))
        p_jerk = 0.02 * action_diff_sq
        self.last_action = action.copy()

        # 2. Symmetric Low-Gain Control Effort Regularization:
        # Treats acceleration and deceleration symmetrically, allowing the drone to comfortably
        # command a0 = -0.15 (3.5 m/s) on cruising targets without incurring asymmetric braking penalties.
        p_effort = 0.01 * float(action[0]**2) + 0.01 * float(action[2]**2) + 0.02 * float(action[1]**2 + action[3]**2)

        # Step Reward Total
        step_reward = (
            r_tracking_base + r_trail_potential + r_progress
            + r_basket_dwell + r_recovery + r_milestone + r_time
            - p_proximity - p_low_alt - p_jerk - p_effort
            + terminal_reward
        )

        info = {
            "dist_m": float(dist),
            "d_trail_m": float(d_trail),
            "in_view": bool(raw_in_view),
            "in_basket": bool(in_basket),
            "continuous_lock_dwell_s": float(self.continuous_lock_dwell_s),
            "time_to_first_lock_s": float(self.time_to_first_lock) if self.time_to_first_lock is not None else -1.0,
            "interception_success": bool(self.lock_achieved),
            "time_lost_s": float(self.time_lost),
            "cumulative_firing_window_s": float(self.cumulative_firing_window_s),
            "firing_window_pct": float(self.cumulative_firing_window_s / max(0.01, t_sim) * 100.0),
            "p_jerk": float(p_jerk),
            "p_effort": float(p_effort),
            "termination_reason": termination_reason,
            "r_tracking_base": float(r_tracking_base),
            "r_basket": float(r_basket),
            "r_basket_dwell": float(r_basket_dwell),
            "r_trail_potential": float(r_trail_potential),
            "r_progress": float(r_progress),
            "r_recovery": float(r_recovery),
            "r_milestone": float(r_milestone),
            "r_time": float(r_time),
            "p_proximity": float(p_proximity),
            "p_low_alt": float(p_low_alt),
        }

        return obs, float(step_reward), bool(terminated), bool(truncated), info
