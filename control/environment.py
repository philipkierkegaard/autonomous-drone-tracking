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
import json
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
from control.synthetic_sensor import SyntheticYOLOSensor
from perception.pipeline import DroneTracker


class DronePursuitEnv(gym.Env):
    """
    Continuous 50 Hz Visual Servoing Environment for Autonomous Drone Interception.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        dt: float = 0.02,
        max_duration: float = 25.0,
        standoff_nominal_m: float = 3.5,
        standoff_min_m: float = 2.5,
        standoff_max_m: float = 4.5,
        collision_dist_m: float = 1.2,
        min_altitude_agl_m: float = 0.8,
        safe_altitude_cushion_m: float = 1.5,
        lost_target_timeout_s: float = 3.5,
        in_medias_res_ratio: float = 0.30,
        domain_randomization: bool = True,
        frame_drop_rate: float = 0.05,
        img_w: int = 1280,
        img_h: int = 720,
        hfov_deg: float = 62.0,
        vfov_deg: float = 38.0,
        use_empirical_perception: bool = True,
        empirical_profile_path: Optional[Union[str, Path]] = None,
        camera_fps: float = 30.0,
        env_perception_mode: str = "nominal",
        reward_mode: str = "monotone_potential",
        profile_distribution: Optional[Tuple[float, float, float]] = None,
    ):
        super().__init__()

        self.reward_mode = reward_mode
        self.profile_distribution = profile_distribution
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
        self.hfov_deg = hfov_deg
        self.vfov_deg = vfov_deg
        self.half_hfov = float(np.tan(np.deg2rad(hfov_deg / 2.0)))
        self.half_vfov = float(np.tan(np.deg2rad(vfov_deg / 2.0)))
        self.w_nominal = 0.0859 # Target bbox width fraction at 3.5m standoff (110.0 pixels on 1280x720)
        self.max_decel = 4.0

        # Physical Kinematic Flight Envelope Limits (matched identically to KinematicVisualServoController & run_tracker.py)
        self.max_fwd_vel = 15.0         # Forward speed cap: 15.0 m/s (superior sprint closing against evasive targets)
        self.max_rev_vel = 0.5          # Max reverse cushion: 0.5 m/s
        self.max_lat_vel = 4.0          # Max lateral velocity: 4.0 m/s (agile roll-slide cornering)
        self.max_climb_vel = 3.0        # Max climb velocity: 3.0 m/s (NED: -3.0)
        self.max_desc_vel = 2.0         # Max descent velocity: 2.0 m/s (NED: +2.0)
        self.max_yaw_rate_deg = 120.0   # Max yaw rate: 120 deg/s (matches Pixhawk & FastPixhawkQuadSim)
        self.max_accel_xy = 6.0         # Max horizontal accel: 6.0 m/s^2 (agile intercept sprint, ~31.5 deg tilt)
        self.max_decel_xy = 6.0         # Max horizontal decel: 6.0 m/s^2 (calibrated braking)
        self.max_accel_z = 3.0          # Max vertical accel: 3.0 m/s^2
        self.max_accel_yaw = 240.0      # Max yaw accel: 240 deg/s^2 (agile corner tracking)

        # Empirical YOLO perception & Kalman tracking configuration
        self.use_empirical_perception = use_empirical_perception
        self.camera_fps = camera_fps
        self.camera_dt = 1.0 / camera_fps
        self.env_perception_mode = env_perception_mode
        self.empirical_profile_path = empirical_profile_path
        self.time_since_camera = 0.0

        self.coasting_step_limit = int(round(0.50 / self.dt))  # Deployment baseline: 0.50s (25 steps at 50 Hz)

        if self.use_empirical_perception:
            self.synthetic_sensor = SyntheticYOLOSensor(
                profile_path=self.empirical_profile_path,
                img_w=self.img_w,
                img_h=self.img_h,
                env_mode=self.env_perception_mode
            )
            self.tracker = DroneTracker(
                dt=self.dt,
                max_lost_frames=self.coasting_step_limit,
                min_hits_to_confirm=2
            )
        else:
            self.synthetic_sensor = None
            self.tracker = None

        # Underlying calibrated physics simulator matching identical limits
        self.sim = FastPixhawkQuadSim(
            dt=dt,
            max_vel_xy=self.max_fwd_vel,
            max_accel_xy=self.max_accel_xy,
            max_vel_up=self.max_climb_vel,
            max_vel_down=self.max_desc_vel
        )
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
        # Observation Space: 15D Normalized State Vector (Generation 6)
        # [0] ex: Normalized Centroid Horizontal Error   [-1, 1]
        # [1] ey: Normalized Centroid Vertical Error     [-1, 1]
        # [2] scale_err: Standoff Scale Error            [-1, 1] (0 = 3.5m, >0 far, <0 close)
        # [3] w_norm: Normalized Bounding Box Width      [0, 1]
        # [4] in_fov: Target Visibility Flag             {0.0, 1.0}
        # [5] vx_body: Normalized Forward Velocity       [-1, 1] (norm by 12 m/s)
        # [6] vy_body: Normalized Lateral Velocity       [-1, 1] (norm by 2.5 m/s)
        # [7] vz_body: Normalized Vertical Velocity      [-1, 1] (norm by 2.5 m/s)
        # [8] yaw_rate: Normalized Yaw Rate              [-1, 1] (norm by 90 deg/s)
        # [9] alt_norm: Normalized Altitude AGL          [0, 2] (norm by 5.0m)
        # [10] d_ex: Optical Azimuth Rate                [-1, 1] (norm by 2.0 s^-1)
        # [11] d_ey: Optical Elevation Rate              [-1, 1] (norm by 2.0 s^-1)
        # [12] d_w: Bounding Box Expansion Rate          [-1, 1] (norm by 0.20 s^-1)
        # [13] drone_pitch: Normalized Pitch Angle       [-1, 1] (norm by 45 deg = pi/4 rad)
        # [14] drone_roll: Normalized Roll Angle         [-1, 1] (norm by 45 deg = pi/4 rad)
        # -------------------------------------------------------------
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

        # Internal state tracking
        self.target_oracle: Optional[StochasticTargetTrajectory] = None
        self.step_count = 0
        self.time_lost = 0.0
        self.last_action = np.zeros(4, dtype=np.float32)
        self.prev_action_diff = np.zeros(4, dtype=np.float32)
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
        self.prev_camera_time = 0.0
        self.last_camera_status = 0.0

        # Mission Objective: Reach 5.0m - 7.0m basket & maintain continuous lock
        self.dwell_time_required = 3.0
        self.continuous_lock_dwell_s = 0.0
        self.time_to_first_lock: Optional[float] = None
        self.lock_achieved = False
        self.milestone_1s_awarded = False
        self.milestone_2s_awarded = False
        self.first_contact_awarded = False
        self.first_bucket_awarded = False
        self.bucket_entry_time: Optional[float] = None
        self.expressive_cfg = self._load_expressive_cfg()

        # Flight Dynamics & Actuator Slew-Rate Memory
        self.last_physical_cmd = np.zeros(4, dtype=np.float64)

    def _load_expressive_cfg(self) -> Dict[str, Any]:
        """Loads expressive reward shaping parameters from control/current_expressive_reward.json if present."""
        cfg_path = PROJECT_ROOT / "control" / "current_expressive_reward.json"
        defaults = {
            "r_arrival": 140.0,
            "tau_arrival": 3.5,
            "dist_peak": 8.0,
            "dist_sigma": 1.5,
            "dist_pull": 0.35,
            "blast_base": 25.0,
            "dwell_scale": 10.0,
            "p_collision": 0.0,
            "p_ground": -20.0,
            "pn_lead_weight": 3.5,
            "collision_dist": 0.9,
            "sprint_weight": 3.5,
        }
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    user_cfg = json.load(f)
                    defaults.update(user_cfg)
            except Exception:
                pass
        # Dynamically apply collision_dist override if provided
        if "collision_dist" in defaults:
            self.collision_dist = float(defaults["collision_dist"])
        return defaults

    def _map_action_to_physical(self, action: np.ndarray) -> np.ndarray:
        """
        Maps normalized actions [-1, 1]^4 to physical velocity setpoints for FastPixhawkQuadSim,
        strictly matching the physical limits and slew rates of KinematicVisualServoController.
        Neutral stick a = [0, 0, 0, 0] commands station-keeping hover (0 m/s on all axes).
        """
        a = np.clip(action, -1.0, 1.0)

        # a0: Forward velocity setpoint [-1, 1]
        # Neutral stick (a0 = 0) commands cruise formation velocity (3.5 m/s) matching target cruise
        # Positive stick (a0 > 0) commands sprint pursuit up to +self.max_fwd_vel (+12.0 m/s)
        # Negative stick (a0 < 0) commands deceleration down to -self.max_rev_vel (-0.5 m/s)
        if a[0] >= 0:
            target_vx = self.standoff_nominal + a[0] * (self.max_fwd_vel - self.standoff_nominal)
        else:
            target_vx = self.standoff_nominal + a[0] * (self.standoff_nominal + self.max_rev_vel)

        # Forward collision proximity barrier: inhibit excess forward surge if dangerously close (distance < 2.2m)
        if hasattr(self, "last_distance") and self.last_distance < 2.2:
            max_safe_vx = float(max(0.0, (self.last_distance - 1.3) / 0.9 * self.standoff_nominal))
            target_vx = min(target_vx, max_safe_vx)

        # a1: Lateral strafe setpoint [-1, 1] -> [-self.max_lat_vel, +self.max_lat_vel] (-2.5 to +2.5 m/s)
        target_vy = a[1] * self.max_lat_vel

        # a2: Vertical velocity setpoint [-1, 1] (NED: negative is climb, positive is descend)
        # Neutral stick (a2 = 0) commands altitude hold (0.0 m/s)
        # Negative stick (a2 < 0) commands climb up to -self.max_climb_vel (-2.5 m/s)
        # Positive stick (a2 > 0) commands descent up to +self.max_desc_vel (+1.5 m/s)
        if a[2] <= 0:
            target_vz = a[2] * self.max_climb_vel
        else:
            target_vz = a[2] * self.max_desc_vel

        # Ground proximity safety governor: inhibit descent if near ground floor (altitude < 1.4m AGL)
        current_alt_m = -self.sim.pos[2]
        if current_alt_m < 1.4 and target_vz > 0.0:
            target_vz = float(target_vz * max(0.0, (current_alt_m - 0.9) / 0.5))

        # a3: Yaw rate setpoint [-1, 1] -> [-self.max_yaw_rate_deg, +self.max_yaw_rate_deg] (-90 to +90 deg/s)
        target_yaw_deg = a[3] * self.max_yaw_rate_deg

        # Slew-Rate Limiting across all 4 flight axes (matched identically to KinematicVisualServoController)
        max_step_accel_x = self.max_accel_xy * self.dt
        max_step_decel_x = self.max_decel_xy * self.dt
        if target_vx < self.last_physical_cmd[0]:
            cmd_vx = max(target_vx, self.last_physical_cmd[0] - max_step_decel_x)
        else:
            cmd_vx = min(target_vx, self.last_physical_cmd[0] + max_step_accel_x)

        max_step_lat = self.max_accel_xy * self.dt
        cmd_vy = float(np.clip(target_vy, self.last_physical_cmd[1] - max_step_lat, self.last_physical_cmd[1] + max_step_lat))

        max_step_z = self.max_accel_z * self.dt
        cmd_vz = float(np.clip(target_vz, self.last_physical_cmd[2] - max_step_z, self.last_physical_cmd[2] + max_step_z))

        max_step_yaw = self.max_accel_yaw * self.dt
        cmd_yaw_deg = float(np.clip(target_yaw_deg, self.last_physical_cmd[3] - max_step_yaw, self.last_physical_cmd[3] + max_step_yaw))

        phys_cmd = np.array([cmd_vx, cmd_vy, cmd_vz, cmd_yaw_deg], dtype=np.float64)
        self.last_physical_cmd = phys_cmd.copy()
        return phys_cmd

    def _sample_scenario_config(self) -> Dict[str, Any]:
        """
        Samples an encounter according to the 70/30 distribution and physical profile weights.
        """
        rng = self.np_random

        # Target Flight Profile Split (configurable for hyper-evasive focus)
        p_dist = self.profile_distribution if self.profile_distribution is not None else [0.10, 0.45, 0.45]
        profile_choice = rng.choice(["cruising", "evasive", "hyper_evasive"], p=p_dist)

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
            # Realistic initial forward cruising speed matching kinematic limit (3.0 to 4.5 m/s)
            chaser_vel = np.array([rng.uniform(3.0, 4.5), 0.0, 0.0], dtype=np.float64)

            # Continuous spawn distance across the entire operational envelope (3.2m close basket to 12.0m sprint):
            spawn_dist = float(rng.uniform(3.2, 12.0))
            spawn_dist_range = (spawn_dist, spawn_dist)

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
                edge_el = rng.uniform(-8.0, 8.0)
                dist = rng.uniform(4.0, 8.0)

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
                # Overshoot recovery: target is close (2.5m - 3.8m) with closing speed requiring braking
                dist = rng.uniform(2.5, 3.8)
                target_pos, _, _, _ = self.scenario_gen.sample_initial_target_pos(
                    chaser_pos=chaser_pos, chaser_yaw=chaser_yaw, rng=rng,
                    distance_range=(dist, dist),
                    azimuth_deg_range=(-10.0, 10.0),
                    elevation_deg_range=(-6.0, 6.0)
                )
                target_heading = float(chaser_yaw + rng.uniform(-0.3, 0.3))
                # Chaser has forward velocity (testing active deceleration / flare recovery)
                c, s = np.cos(chaser_yaw), np.sin(chaser_yaw)
                v_fwd = rng.uniform(3.5, 6.0)
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
                # Target passes across perpendicular at high speed
                dist = rng.uniform(3.5, 6.5)
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
                blind_az = rng.choice([-33.0, 33.0])  # Just beyond 31.0 deg half-HFOV
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

        if options and "scenario" in options:
            scen = options["scenario"]
            if hasattr(scen, "profile"):
                c_yaw = float(scen.initial_chaser_yaw)
                c_vel = getattr(scen, "initial_chaser_vel", None)
                if c_vel is None:
                    c, s = np.cos(c_yaw), np.sin(c_yaw)
                    cruise_v = min(4.0, self.max_fwd_vel)
                    c_vel = np.array([c * cruise_v, s * cruise_v, 0.0], dtype=np.float64)
                else:
                    c_vel = np.array(c_vel, dtype=np.float64)

                cfg = {
                    "mode": "scenario",
                    "profile": scen.profile,
                    "chaser_pos": np.array(scen.initial_chaser_pos, dtype=np.float64),
                    "chaser_yaw": c_yaw,
                    "chaser_vel": c_vel,
                    "target_pos": np.array(scen.initial_target_pos, dtype=np.float64),
                    "target_heading": float(scen.initial_target_heading_rad),
                    "initial_time_lost": 0.0,
                }
                seed_val = int(scen.seed)
            elif isinstance(scen, dict):
                c_yaw = float(scen.get("chaser_yaw", scen.get("initial_chaser_yaw", 0.0)))
                c_vel = scen.get("chaser_vel", scen.get("initial_chaser_vel", None))
                if c_vel is None:
                    c, s = np.cos(c_yaw), np.sin(c_yaw)
                    cruise_v = min(4.0, self.max_fwd_vel)
                    c_vel = np.array([c * cruise_v, s * cruise_v, 0.0], dtype=np.float64)
                else:
                    c_vel = np.array(c_vel, dtype=np.float64)

                c_pos = scen.get("chaser_pos", scen.get("initial_chaser_pos", [0.0, 0.0, -2.5]))
                t_pos = scen.get("target_pos", scen.get("initial_target_pos", [12.0, 0.0, -2.5]))
                t_head = float(scen.get("target_heading", scen.get("initial_target_heading_rad", 0.0)))
                time_lost_init = float(scen.get("initial_time_lost", 0.0))

                cfg = {
                    "mode": scen.get("mode", "scenario"),
                    "profile": scen.get("profile", "evasive"),
                    "chaser_pos": np.array(c_pos, dtype=np.float64),
                    "chaser_yaw": c_yaw,
                    "chaser_vel": c_vel,
                    "target_pos": np.array(t_pos, dtype=np.float64),
                    "target_heading": t_head,
                    "initial_time_lost": time_lost_init,
                }
                seed_val = int(scen.get("seed", int(self.np_random.integers(0, 1_000_000))))
            else:
                cfg = self._sample_scenario_config()
                seed_val = int(self.np_random.integers(0, 1_000_000))
        else:
            cfg = self._sample_scenario_config()
            seed_val = int(self.np_random.integers(0, 1_000_000))

        # Reset simulator with initial cruise velocity and steady-state attitude trim
        self.sim.reset(
            initial_pos=cfg["chaser_pos"],
            initial_yaw=cfg["chaser_yaw"],
            initial_vel=cfg["chaser_vel"]
        )

        # Initialize target B-spline oracle
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
        self.prev_action_diff = np.zeros(4, dtype=np.float32)

        c_yaw, s_yaw = np.cos(cfg["chaser_yaw"]), np.sin(cfg["chaser_yaw"])
        v_fwd = c_yaw * cfg["chaser_vel"][0] + s_yaw * cfg["chaser_vel"][1]
        v_right = -s_yaw * cfg["chaser_vel"][0] + c_yaw * cfg["chaser_vel"][1]
        self.last_physical_cmd = np.array([v_fwd, v_right, cfg["chaser_vel"][2], 0.0], dtype=np.float64)
        self.cumulative_firing_window_s = 0.0
        self.continuous_lock_dwell_s = 0.0
        self.time_to_first_lock = None
        self.lock_achieved = False
        self.milestone_1s_awarded = False
        self.milestone_2s_awarded = False
        self.first_contact_awarded = False
        self.first_bucket_awarded = False
        self.bucket_entry_time = None

        # Project initial target state to camera sensor
        t_sim = 0.0
        target_pos_w, target_vel_w, _ = self.target_oracle.get_state(t_sim)
        ex, ey, in_view, dist, bw, bh = self.sim.project_to_camera(
            target_pos_w, return_bbox=True, img_w=self.img_w, img_h=self.img_h,
            hfov_deg=self.hfov_deg, vfov_deg=self.vfov_deg
        )
        self.initial_distance = float(dist)
        self.last_distance = float(dist)

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

        # Initialize or reset empirical perception & Kalman tracker
        if self.use_empirical_perception:
            self.time_since_camera = 0.0
            sensor_seed = int(self.np_random.integers(0, 1_000_000))
            self.synthetic_sensor.reset(seed=sensor_seed)
            self.tracker = DroneTracker(
                dt=self.dt,
                max_lost_frames=self.coasting_step_limit,
                min_hits_to_confirm=2
            )
            # Initial camera measurement at t=0 (30 Hz camera frame)
            if in_view:
                norm_cx = float((ex + 1.0) / 2.0)
                norm_cy = float((ey + 1.0) / 2.0)
                norm_w = float(bw / self.img_w)
                norm_h = float(bh / self.img_h)
                det_bbox, conf, fps = self.synthetic_sensor.step(norm_cx, norm_cy, norm_w, norm_h, in_fov=True)
                if det_bbox is not None:
                    self.tracker.update(np.array([det_bbox], dtype=np.float32), np.array([conf], dtype=np.float32))
                    if len(self.tracker.trackers) > 0:
                        self.tracker.trackers[0].hits = 2
                        self.tracker.primary_track_id = self.tracker.trackers[0].id

            primary = None
            if self.tracker.primary_track_id is not None:
                primary = next((t for t in self.tracker.trackers if t.id == self.tracker.primary_track_id), None)

            self.prev_camera_time = 0.0
            if primary is not None and primary.hits >= self.tracker.min_hits_to_confirm:
                bbox = primary.get_state()
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                bw_est = max(1.0, bbox[2] - bbox[0])
                bh_est = max(1.0, bbox[3] - bbox[1])
                ex_est = float((cx - self.img_w / 2.0) / (self.img_w / 2.0))
                ey_est = float((cy - self.img_h / 2.0) / (self.img_h / 2.0))
                self.last_camera_status = 1.0
                obs = self._get_observation(ex_est, ey_est, True, bw_est, bh_est, force_flag=1.0, is_fresh_camera=True, dt_cam=self.camera_dt)
            else:
                self.last_camera_status = 0.0
                obs = self._get_observation(self.last_seen_ex, self.last_seen_ey, False, self.last_seen_w * self.img_w, self.last_seen_h * self.img_h, force_flag=0.0, is_fresh_camera=True, dt_cam=self.camera_dt)
        else:
            self.prev_camera_time = 0.0
            self.last_camera_status = 1.0 if in_view else 0.0
            obs = self._get_observation(ex, ey, in_view, bw, bh, is_fresh_camera=True, dt_cam=self.camera_dt)

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
        raw_bh: float,
        force_flag: Optional[float] = None,
        is_fresh_camera: bool = True,
        dt_cam: Optional[float] = None
    ) -> np.ndarray:
        """
        Constructs the 13D normalized state vector with SO(3) attitude decoupling,
        filtered optical rates (azimuth rate, elevation rate, bbox expansion rate),
        and optional Sim-to-Real domain randomization.
        """
        in_fov = raw_in_view

        # Legacy domain randomization (only applied if not using empirical YOLO perception)
        if not self.use_empirical_perception:
            if self.domain_randomization and in_fov:
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

            if not self.use_empirical_perception and self.domain_randomization:
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
            flag = force_flag if force_flag is not None else 1.0

            # Causal exponential filter for optical rates (alpha = 0.4 suppresses noise while preserving 1-2 Hz turns)
            # Optical rates are updated only when a fresh camera frame arrives with dt_cam (approx 33 ms).
            # On intermediate 50 Hz steps between camera frames, the filtered rates are held to maintain sim-to-real parity.
            if is_fresh_camera:
                delta_t = dt_cam if dt_cam is not None and dt_cam > 0.0 else self.camera_dt
                if self.was_in_view_prev:
                    raw_d_ex = (ex - self.prev_ex) / delta_t
                    raw_d_ey = (ey - self.prev_ey) / delta_t
                    raw_d_w  = (w_norm - self.prev_w) / delta_t
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
        scale_err = float(np.clip((self.w_nominal - w_norm) / self.w_nominal, -1.0, 1.0))

        # Normalized altitude AGL (NED: z is negative above ground)
        alt_agl = -self.sim.pos[2]
        alt_norm = float(np.clip(alt_agl / 5.0, 0.0, 2.0))

        # Normalized optical rates: d_ex and d_ey clipped [-1, 1] (norm by 2.0 s^-1), d_w norm by 0.20 s^-1
        norm_d_ex = float(np.clip(self.filtered_d_ex / 2.0, -1.0, 1.0))
        norm_d_ey = float(np.clip(self.filtered_d_ey / 2.0, -1.0, 1.0))
        norm_d_w  = float(np.clip(self.filtered_d_w  / 0.20, -1.0, 1.0))

        # Normalized drone attitude pitch and roll (clipped [-1, 1], normalized by 45 deg = pi/4 rad)
        norm_pitch = float(np.clip(self.sim.pitch / (np.pi / 4.0), -1.0, 1.0))
        norm_roll  = float(np.clip(self.sim.roll  / (np.pi / 4.0), -1.0, 1.0))

        obs = np.array([
            ex,
            ey,
            scale_err,
            w_norm,
            flag,
            np.clip(vx_body / self.max_fwd_vel, -1.0, 1.0),
            np.clip(vy_body / self.max_lat_vel, -1.0, 1.0),
            np.clip(vz_body / self.max_climb_vel, -1.0, 1.0),
            np.clip(self.sim.yaw_rate / np.deg2rad(self.max_yaw_rate_deg), -1.0, 1.0),
            alt_norm,
            norm_d_ex,
            norm_d_ey,
            norm_d_w,
            norm_pitch,
            norm_roll,
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
        raw_ex_unclipped, raw_ey_unclipped, raw_in_view, dist, raw_bw, raw_bh = self.sim.project_to_camera(
            target_pos_w, return_bbox=True, return_unclipped=True, img_w=self.img_w, img_h=self.img_h,
            hfov_deg=self.hfov_deg, vfov_deg=self.vfov_deg
        )
        self.last_distance = float(dist)
        raw_ex = float(np.clip(raw_ex_unclipped, -1.0, 1.0))
        raw_ey = float(np.clip(raw_ey_unclipped, -1.0, 1.0))

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

        # Construct 13D next observation via Empirical YOLO Sensor + Kalman Filter
        if self.use_empirical_perception:
            self.time_since_camera += self.dt
            is_camera_tick = (self.time_since_camera >= self.camera_dt - 1e-5)

            if is_camera_tick:
                dt_cam = max(0.010, t_sim - self.prev_camera_time) if self.prev_camera_time > 0.0 else self.camera_dt
                self.prev_camera_time = t_sim
                self.time_since_camera -= self.camera_dt
                norm_cx = float((raw_ex + 1.0) / 2.0)
                norm_cy = float((raw_ey + 1.0) / 2.0)
                norm_w  = float(raw_bw / self.img_w)
                norm_h  = float(raw_bh / self.img_h)

                det_bbox, conf, fps = self.synthetic_sensor.step(
                    norm_cx, norm_cy, norm_w, norm_h, in_fov=raw_in_view
                )

                detections = []
                confs = []
                if det_bbox is not None:
                    detections.append(det_bbox)
                    confs.append(conf)
                for fp in fps:
                    detections.append(fp)
                    confs.append(0.36)

                det_arr = np.array(detections, dtype=np.float32) if detections else np.empty((0, 4), dtype=np.float32)
                conf_arr = np.array(confs, dtype=np.float32) if confs else np.empty((0,), dtype=np.float32)
                self.tracker.update(det_arr, conf_arr, dt_step=self.dt)
            else:
                dt_cam = self.camera_dt
                # Between camera frames (50 Hz control tick between 30 Hz camera updates):
                # Kalman filter performs prediction/dead reckoning without receiving new measurements
                for trk in self.tracker.trackers:
                    trk.predict(dt_step=self.dt)

            # Retrieve state estimate from primary tracker
            primary = None
            if self.tracker.primary_track_id is not None:
                primary = next((t for t in self.tracker.trackers if t.id == self.tracker.primary_track_id), None)

            if primary is not None and primary.hits >= self.tracker.min_hits_to_confirm:
                bbox = primary.get_state()
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                bw_est = max(1.0, bbox[2] - bbox[0])
                bh_est = max(1.0, bbox[3] - bbox[1])
                ex_est = float((cx - self.img_w / 2.0) / (self.img_w / 2.0))
                ey_est = float((cy - self.img_h / 2.0) / (self.img_h / 2.0))

                # Update latched status on fresh camera ticks
                if is_camera_tick:
                    if primary.time_since_update == 0:
                        self.last_camera_status = 1.0  # Fresh YOLO measurement
                    else:
                        self.last_camera_status = 0.5  # Coasting on missed frame

                # Check track expiration against the 0.50s baseline
                if primary.time_since_update > self.coasting_step_limit:
                    flag = 0.0
                else:
                    flag = self.last_camera_status

                obs = self._get_observation(
                    ex_est, ey_est, True, bw_est, bh_est,
                    force_flag=flag, is_fresh_camera=is_camera_tick, dt_cam=dt_cam
                )
            else:
                self.last_camera_status = 0.0
                obs = self._get_observation(
                    self.last_seen_ex, self.last_seen_ey, False,
                    self.last_seen_w * self.img_w, self.last_seen_h * self.img_h,
                    force_flag=0.0, is_fresh_camera=is_camera_tick, dt_cam=dt_cam
                )
        else:
            primary = None
            obs = self._get_observation(raw_ex, raw_ey, raw_in_view, raw_bw, raw_bh, is_fresh_camera=True, dt_cam=self.dt)

        # Engagement Basket Condition: 5.0m <= dist <= 7.0m AND visible in camera FOV AND active tracker lock
        has_tracker_lock = (
            primary is not None
            and primary.hits >= self.tracker.min_hits_to_confirm
            and primary.time_since_update <= self.coasting_step_limit
        ) if self.use_empirical_perception else raw_in_view

        # Engagement Basket Condition: 2.8m <= dist <= 4.2m AND visible in camera FOV AND active tracker lock
        # Alignment quality modulates dwell accumulation rate rather than hard binary cutoff:
        # Near center (|err| < 0.20): quality ~ 1.0 (accumulates dwell at full 1.0s/s)
        # Moderate off-center (|err| ~ 0.50): quality ~ 0.5 (accumulates at 0.5s/s)
        # Frame border (|err| > 0.70): quality = 0.05 (negligible accumulation)
        in_basket = (self.standoff_min <= dist <= self.standoff_max) and raw_in_view and has_tracker_lock
        r_milestone = 0.0
        r_sustain = 0.0

        # Term 2: Smooth exponential early-arrival multiplier (rewards rapid closing sprint; never hard-zeros out)
        time_decay_arrival = float(np.exp(-t_sim / 6.0))

        # Term 2B: Immediate First-Contact Arrival Jackpot
        # Only awarded when entering the basket with controlled relative closing speed (< 3.0 m/s)
        # to prevent rewarding high-speed "bullet ramming" overshoots
        r_first_contact = 0.0
        if in_basket and not self.first_contact_awarded:
            closing_speed = (self.prev_distance_error - dist_err) / self.dt
            if closing_speed < 3.0 and not self.reward_mode.startswith("expressive"):
                self.first_contact_awarded = True
                r_first_contact = 40.0 * time_decay_arrival

        # Term 2C: Expressive Reverse-Exponential Fast-Arrival Jackpot
        r_first_bucket = 0.0
        if in_basket and not self.first_bucket_awarded:
            self.first_bucket_awarded = True
            self.bucket_entry_time = float(t_sim)
            tau = float(self.expressive_cfg.get("tau_arrival", 4.5))
            r_arr = float(self.expressive_cfg.get("r_arrival", 120.0))
            r_first_bucket = r_arr * float(np.exp(-t_sim / max(0.5, tau)))

        if in_basket:
            self.cumulative_firing_window_s += self.dt
            ang_err_cam = float(np.sqrt(raw_ex**2 + raw_ey**2))
            alignment_quality = float(np.clip(1.0 - (ang_err_cam / 0.70)**2, 0.05, 1.0))
            dwell_rate = alignment_quality
            self.continuous_lock_dwell_s = min(self.dwell_time_required, self.continuous_lock_dwell_s + dwell_rate * self.dt)

            # Milestone 1: 1.0s continuous lock
            if self.continuous_lock_dwell_s >= 1.0 and not self.milestone_1s_awarded:
                self.milestone_1s_awarded = True
                r_milestone += 15.0 + 20.0 * time_decay_arrival

            # Milestone 2: 2.0s continuous lock
            if self.continuous_lock_dwell_s >= 2.0 and not self.milestone_2s_awarded:
                self.milestone_2s_awarded = True
                r_milestone += 25.0 + 35.0 * time_decay_arrival

            # Goal: 3.0s continuous lock (interception jackpot)
            if self.continuous_lock_dwell_s >= self.dwell_time_required:
                if not self.lock_achieved:
                    self.lock_achieved = True
                    self.time_to_first_lock = float(max(0.0, self.step_count * self.dt - self.dwell_time_required))
                    r_milestone += 50.0 + 100.0 * time_decay_arrival
                # Ongoing formation sustain reward: +5.0/s for holding lock through evasive maneuvers
                r_sustain = 0.1
        else:
            # Grace decay (soft leaky integrator: decays only 0.5s per second outside basket to tolerate momentary visual dropouts)
            self.continuous_lock_dwell_s = max(0.0, self.continuous_lock_dwell_s - 0.5 * self.dt)
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
            terminal_reward = 80.0 + 120.0 * time_decay_arrival
            if self.reward_mode.startswith("expressive"):
                terminal_reward = 120.0 + 100.0 * float(np.exp(-t_sim / 6.0))

        # Physical Collision Breach (d < 1.2m) — True physical collision radius
        elif dist < self.collision_dist:
            terminated = True
            termination_reason = "collision"
            terminal_reward = 0.0
            if self.reward_mode.startswith("expressive"):
                terminal_reward = float(self.expressive_cfg.get("p_collision", -1.0))

        # Altitude Floor Breach (z > -1.0m in NED means altitude < 1.0m AGL)
        elif self.sim.pos[2] > -self.min_altitude_agl:
            terminated = True
            termination_reason = "ground_strike"
            terminal_reward = 0.0
            if self.reward_mode.startswith("expressive"):
                terminal_reward = float(self.expressive_cfg.get("p_ground", -20.0))

        # Target Lost Timeout (out of FOV for > 3.5s)
        elif self.time_lost > self.lost_target_timeout:
            terminated = True
            termination_reason = "lost_target_timeout"
            terminal_reward = 0.0

        # Time Horizon Reached (25.0 seconds = 1250 steps)
        elif self.step_count >= self.max_steps:
            truncated = True
            termination_reason = "max_horizon"

        # -------------------------------------------------------------
        # 6. Reward Shaping Architectures (Selected by self.reward_mode)
        # -------------------------------------------------------------
        r_lead_vec = 0.0
        closing_speed = (self.prev_distance_error - dist_err) / self.dt
        dwell_progress = min(1.0, self.continuous_lock_dwell_s / self.dwell_time_required)

        if self.reward_mode == "legacy_gaussian":
            # 1. Baseline: Unconditional living floor + narrow Gaussian (sigma=1.2m)
            r_alive = 0.50
            r_dist = 8.0 * float(np.exp(-0.5 * (dist_err / 1.2)**2)) if raw_in_view else 1.0 * float(np.exp(-0.5 * (dist_err / 3.0)**2))
            r_progress = 2.5 * float(np.clip(closing_speed, 0.0, 5.0)) if dist > self.standoff_max else 0.0
            r_basket_dwell = (6.0 + 8.0 * dwell_progress) if in_basket else 0.0

        elif self.reward_mode == "huber_loss":
            # 2. Smooth Huber Potential Field: Quadratic in bucket, constant linear pull outside
            r_alive = 0.50 if (raw_in_view and dist <= 6.0) else 0.0
            delta = 1.0  # Huber transition radius (m)
            dist_err_m = abs(dist - self.standoff_nominal)
            if dist_err_m <= delta:
                r_dist = 8.0 - 2.5 * (dist_err_m**2)
            else:
                r_dist = 8.0 - 2.5 * (delta**2) - 0.60 * (dist_err_m - delta)
            r_progress = 3.5 * float(np.clip(closing_speed, 0.0, 5.0)) if closing_speed > 0 else 1.8 * float(np.clip(closing_speed, -5.0, 0.0))
            r_basket_dwell = (8.0 + 10.0 * dwell_progress) if in_basket else 0.0

        elif self.reward_mode == "lead_pursuit":
            # 3. Lead-Pursuit Guidance: Dynamic vector alignment with target flight path
            r_alive = 0.50 if (raw_in_view and dist <= 6.0) else 0.0
            dist_err_m = abs(dist - self.standoff_nominal)
            r_peak = 8.0 * float(np.exp(-0.5 * (dist_err_m / 1.6)**2))
            r_tail = -0.45 * float(max(0.0, dist - self.standoff_max))
            r_dist = (r_peak + r_tail) if raw_in_view else (1.0 * float(np.exp(-0.5 * (dist_err_m / 3.0)**2)) - 0.25 * float(max(0.0, dist - self.standoff_max)))
            r_progress = 3.5 * float(np.clip(closing_speed, 0.0, 5.0)) if closing_speed > 0 else 2.0 * float(np.clip(closing_speed, -5.0, 0.0))
            
            v_c = self.sim.vel[:2]
            v_t = target_vel_w[:2]
            nc = float(np.linalg.norm(v_c))
            nt = float(np.linalg.norm(v_t))
            if nc > 1.0 and nt > 1.0 and raw_in_view:
                r_lead_vec = 2.5 * float(max(0.0, np.dot(v_c, v_t) / (nc * nt)))
            r_basket_dwell = (8.0 + 12.0 * dwell_progress) if in_basket else 0.0

        elif self.reward_mode == "speed_adaptive":
            # 4. Speed-Adaptive Standoff: Expands 3.5m -> 4.5m on burst maneuvers for aerodynamic safety
            r_alive = 0.50 if (raw_in_view and dist <= 6.5) else 0.0
            target_speed = float(np.linalg.norm(target_vel_w[:2]))
            eff_standoff = self.standoff_nominal + 0.08 * float(max(0.0, target_speed - 4.0))
            eff_err = abs(dist - eff_standoff)
            r_dist = (8.0 * float(np.exp(-0.5 * (eff_err / 1.6)**2)) - 0.40 * float(max(0.0, dist - (eff_standoff + 1.0)))) if raw_in_view else (1.0 * float(np.exp(-0.5 * (eff_err / 3.0)**2)) - 0.20 * float(max(0.0, dist - (eff_standoff + 1.0))))
            r_progress = (3.0 * float(np.clip(closing_speed, 0.0, 5.0)) if closing_speed > 0 else 1.5 * float(np.clip(closing_speed, -5.0, 0.0))) if dist > (eff_standoff + 1.0) else 0.0
            r_basket_dwell = (8.0 + 10.0 * dwell_progress) if in_basket else 0.0

        elif self.reward_mode == "closing_speed_bell":
            # 5. Aggressive Closing Sprint: High differential progress reward
            r_alive = 0.50 if (raw_in_view and dist <= 5.5) else 0.0
            dist_err_m = abs(dist - self.standoff_nominal)
            r_dist = (8.0 * float(np.exp(-0.5 * (dist_err_m / 1.6)**2)) - 0.50 * float(max(0.0, dist - self.standoff_max))) if raw_in_view else -0.30 * float(max(0.0, dist - self.standoff_max))
            if dist > self.standoff_max:
                r_progress = 4.5 * float(np.clip(closing_speed, 0.0, 6.0)) if closing_speed > 0 else 3.0 * float(np.clip(closing_speed, -6.0, 0.0))
            else:
                rel_v = abs(closing_speed)
                r_progress = 2.0 * float(np.exp(-0.5 * (rel_v / 1.0)**2))  # Reward zero relative speed inside bucket
            r_basket_dwell = (8.0 + 10.0 * dwell_progress) if in_basket else 0.0

        elif self.reward_mode == "tight_centering":
            # 6. High-Gain Crosshair Centering: Strict tracking on maneuvers
            r_alive = 0.50 if (raw_in_view and dist <= 6.0) else 0.0
            dist_err_m = abs(dist - self.standoff_nominal)
            r_dist = (8.0 * float(np.exp(-0.5 * (dist_err_m / 1.6)**2)) - 0.40 * float(max(0.0, dist - self.standoff_max))) if raw_in_view else -0.20 * float(max(0.0, dist - self.standoff_max))
            r_progress = 3.0 * float(np.clip(closing_speed, 0.0, 5.0)) if closing_speed > 0 else 1.5 * float(np.clip(closing_speed, -5.0, 0.0))
            r_basket_dwell = (8.0 + 10.0 * dwell_progress) if in_basket else 0.0

        elif self.reward_mode == "optical_expansion_lead":
            # 7. Optical Bounding-Box Expansion Divergence Feedback
            r_alive = 0.50 if (raw_in_view and dist <= 6.0) else 0.0
            dist_err_m = abs(dist - self.standoff_nominal)
            r_dist = (8.0 * float(np.exp(-0.5 * (dist_err_m / 1.6)**2)) - 0.40 * float(max(0.0, dist - self.standoff_max))) if raw_in_view else -0.20 * float(max(0.0, dist - self.standoff_max))
            # Reward positive bounding-box expansion rate (closing in)
            expansion_bonus = 3.0 * float(np.clip(self.filtered_d_w, 0.0, 0.15)) if dist > self.standoff_max else 0.0
            r_progress = (3.0 * float(np.clip(closing_speed, 0.0, 5.0)) if closing_speed > 0 else 1.5 * float(np.clip(closing_speed, -5.0, 0.0))) + expansion_bonus
            r_basket_dwell = (8.0 + 10.0 * dwell_progress) if in_basket else 0.0

        elif self.reward_mode == "cumulative_dwell_jackpot":
            # 8. Quadratic Dwell Acceleration: Massive reward multiplier for unbroken formation lock
            r_alive = 0.50 if (raw_in_view and dist <= 6.0) else 0.0
            dist_err_m = abs(dist - self.standoff_nominal)
            r_dist = (8.0 * float(np.exp(-0.5 * (dist_err_m / 1.6)**2)) - 0.40 * float(max(0.0, dist - self.standoff_max))) if raw_in_view else -0.20 * float(max(0.0, dist - self.standoff_max))
            r_progress = 3.0 * float(np.clip(closing_speed, 0.0, 5.0)) if closing_speed > 0 else 1.5 * float(np.clip(closing_speed, -5.0, 0.0))
            # Non-linear quadratic dwell scaling: +6.0 base + up to +20.0 for 3s hold
            r_basket_dwell = (6.0 + 20.0 * (dwell_progress**2)) if in_basket else 0.0

        elif self.reward_mode == "terminal_bonus_steep":
            # 9. Steep Linear Pull + Frequent High Milestone Rewards
            r_alive = 0.50 if (raw_in_view and dist <= 5.5) else 0.0
            dist_err_m = abs(dist - self.standoff_nominal)
            r_dist = (8.0 * float(np.exp(-0.5 * (dist_err_m / 1.4)**2)) - 0.60 * float(max(0.0, dist - self.standoff_max))) if raw_in_view else -0.30 * float(max(0.0, dist - self.standoff_max))
            r_progress = 4.0 * float(np.clip(closing_speed, 0.0, 5.0)) if closing_speed > 0 else 2.0 * float(np.clip(closing_speed, -5.0, 0.0))
            r_basket_dwell = (10.0 + 15.0 * dwell_progress) if in_basket else 0.0

        elif self.reward_mode == "unified_dogfight_optimal":
            # 10. Full Synthesis: Huber potential + lead pursuit vector alignment + sprint closing + gated floor
            r_alive = 0.50 if (raw_in_view and dist <= 6.0) else 0.0
            delta = 1.0
            dist_err_m = abs(dist - self.standoff_nominal)
            if dist_err_m <= delta:
                r_dist = 8.0 - 2.5 * (dist_err_m**2)
            else:
                r_dist = 8.0 - 2.5 * (delta**2) - 0.50 * (dist_err_m - delta)
            if dist > self.standoff_max:
                r_progress = 4.0 * float(np.clip(closing_speed, 0.0, 5.0)) if closing_speed > 0 else 2.0 * float(np.clip(closing_speed, -5.0, 0.0))
            else:
                r_progress = 2.0 * float(np.exp(-0.5 * (abs(closing_speed) / 1.0)**2))

            v_c = self.sim.vel[:2]
            v_t = target_vel_w[:2]
            nc = float(np.linalg.norm(v_c))
            nt = float(np.linalg.norm(v_t))
            if nc > 1.0 and nt > 1.0 and raw_in_view:
                r_lead_vec = 2.5 * float(max(0.0, np.dot(v_c, v_t) / (nc * nt)))
            r_basket_dwell = (10.0 + 15.0 * dwell_progress) if in_basket else 0.0

        elif self.reward_mode == "trail_brake_corner_governor":
            # 1. Physics Centripetal Governor & Trail-Braking Incentive
            r_alive = 0.50 if (raw_in_view and dist <= 6.0) else 0.0
            dist_err_m = abs(dist - self.standoff_nominal)
            r_dist = (8.0 * float(np.exp(-0.5 * (dist_err_m / 1.6)**2)) - 0.40 * float(max(0.0, dist - self.standoff_max))) if raw_in_view else -0.20 * float(max(0.0, dist - self.standoff_max))
            
            # Gated progress: only reward/penalize closing speed when aligned! If cornering (|ex| > 0.20), braking is NEVER penalized!
            if abs(raw_ex) < 0.20:
                r_progress = 3.5 * float(np.clip(closing_speed, 0.0, 5.0)) if closing_speed > 0 else 1.5 * float(np.clip(closing_speed, -5.0, 0.0))
            else:
                r_progress = 1.0 * float(np.clip(closing_speed, 0.0, 3.0))

            # Active Trail-Brake Bonus: if off-center (|ex| > 0.25) or optical slip (|filtered_d_ex| > 0.35), reward commanded deceleration
            v_fwd = float(self.sim.vel[0] * np.cos(self.sim.yaw) + self.sim.vel[1] * np.sin(self.sim.yaw))
            if raw_in_view and (abs(raw_ex) > 0.25 or abs(self.filtered_d_ex) > 0.35):
                p_skid = 3.0 * float((v_fwd / 12.0) * max(0.0, abs(raw_ex) - 0.15))
                r_brake = 2.5 * float(max(0.0, -action[0])) if v_fwd > 4.5 else 0.0
                r_progress = r_progress - p_skid + r_brake
            
            r_basket_dwell = (8.0 + 12.0 * dwell_progress) if in_basket else 0.0

        elif self.reward_mode == "proportional_nav_apex_cutter":
            # 2. Proportional Navigation Velocity Vector Alignment (Apex Cutting)
            r_alive = 0.50 if (raw_in_view and dist <= 6.0) else 0.0
            dist_err_m = abs(dist - self.standoff_nominal)
            r_dist = (8.0 * float(np.exp(-0.5 * (dist_err_m / 1.6)**2)) - 0.40 * float(max(0.0, dist - self.standoff_max))) if raw_in_view else -0.20 * float(max(0.0, dist - self.standoff_max))
            
            # Lead pursuit velocity alignment
            v_c = self.sim.vel[:2]
            v_t = target_vel_w[:2]
            nc = float(np.linalg.norm(v_c))
            nt = float(np.linalg.norm(v_t))
            if nc > 0.5 and nt > 0.5 and raw_in_view:
                align = float(np.dot(v_c, v_t) / (nc * nt))
                r_lead_vec = 4.0 * float(max(0.0, align))
                if align < 0.70:
                    r_lead_vec -= 2.0 * float(nc / 12.0) * (0.70 - align)
            
            if abs(raw_ex) < 0.20:
                r_progress = 3.0 * float(np.clip(closing_speed, 0.0, 5.0)) if closing_speed > 0 else 1.0 * float(np.clip(closing_speed, -5.0, 0.0))
            else:
                r_progress = 0.0
            r_basket_dwell = (8.0 + 12.0 * dwell_progress) if in_basket else 0.0

        elif self.reward_mode == "retinal_slip_kinetic_barrier":
            # 3. Retinal Slip Optical Governor (Raptor Attack Model)
            r_alive = 0.50 if (raw_in_view and dist <= 6.0) else 0.0
            dist_err_m = abs(dist - self.standoff_nominal)
            r_dist = (8.0 * float(np.exp(-0.5 * (dist_err_m / 1.6)**2)) - 0.40 * float(max(0.0, dist - self.standoff_max))) if raw_in_view else -0.20 * float(max(0.0, dist - self.standoff_max))
            
            slip = float(np.hypot(self.filtered_d_ex, self.filtered_d_ey))
            v_fwd = float(self.sim.vel[0] * np.cos(self.sim.yaw) + self.sim.vel[1] * np.sin(self.sim.yaw))
            
            if slip < 0.20:
                r_progress = 4.0 * float(np.clip(closing_speed, 0.0, 6.0)) if closing_speed > 0 else 1.0 * float(np.clip(closing_speed, -5.0, 0.0))
            else:
                p_slip_kinetic = 3.5 * float((v_fwd / 12.0)**2 * min(2.0, slip / 0.5))
                r_progress = -p_slip_kinetic
            
            r_basket_dwell = (8.0 + 12.0 * dwell_progress) if in_basket else 0.0

        elif self.reward_mode == "curvature_adaptive_energy_matching":
            # 4. Relative Velocity Vector Nulling at Standoff with Curvature Standoff Relaxation
            r_alive = 0.50 if (raw_in_view and dist <= 6.0) else 0.0
            
            slip = float(abs(self.filtered_d_ex))
            eff_standoff = self.standoff_nominal + 1.0 * float(np.clip(slip / 0.8, 0.0, 1.0))
            eff_err = abs(dist - eff_standoff)
            r_dist = (8.0 * float(np.exp(-0.5 * (eff_err / 1.6)**2)) - 0.40 * float(max(0.0, dist - (eff_standoff + 1.0)))) if raw_in_view else -0.20 * float(max(0.0, dist - (eff_standoff + 1.0)))
            
            rel_vel_norm = float(np.linalg.norm(self.sim.vel - target_vel_w))
            if dist < 8.0 and raw_in_view:
                r_vel_match = 3.0 * float(np.exp(-0.5 * (rel_vel_norm / 2.0)**2) / (1.0 + eff_err))
            else:
                r_vel_match = 0.0
                
            r_progress = (3.0 * float(np.clip(closing_speed, 0.0, 5.0)) if dist > 8.0 else 0.0) + r_vel_match
            r_basket_dwell = (8.0 + 12.0 * dwell_progress) if in_basket else 0.0

        elif self.reward_mode == "staged_curriculum_trail_brake":
            # 5. Cornering Speed-Envelope Guidance
            r_alive = 0.50 if (raw_in_view and dist <= 6.0) else 0.0
            dist_err_m = abs(dist - self.standoff_nominal)
            r_dist = (8.0 * float(np.exp(-0.5 * (dist_err_m / 1.6)**2)) - 0.40 * float(max(0.0, dist - self.standoff_max))) if raw_in_view else -0.20 * float(max(0.0, dist - self.standoff_max))
            
            v_optimal_corner = 4.0 + 8.0 * float(np.exp(-0.5 * (abs(raw_ex) / 0.25)**2 - 0.5 * (abs(self.filtered_d_ex) / 0.40)**2))
            v_fwd = float(self.sim.vel[0] * np.cos(self.sim.yaw) + self.sim.vel[1] * np.sin(self.sim.yaw))
            
            r_envelope_match = 3.0 * float(np.exp(-0.5 * ((v_fwd - v_optimal_corner) / 2.0)**2)) if raw_in_view else 0.0
            
            if dist > self.standoff_max and abs(raw_ex) < 0.25:
                r_progress = 2.5 * float(np.clip(closing_speed, 0.0, 5.0))
            else:
                r_progress = 0.0
            r_progress += r_envelope_match
            r_basket_dwell = (8.0 + 12.0 * dwell_progress) if in_basket else 0.0

        elif self.reward_mode == "unified_apex_dogfight_v2":
            # 6. Unified Synthesis: Huber potential + PN Vector Lead + Deceleration on Shearing + Quadratic Dwell Jackpot
            r_alive = 0.50 if (raw_in_view and dist <= 6.0) else 0.0
            delta = 1.0
            dist_err_m = abs(dist - self.standoff_nominal)
            if dist_err_m <= delta:
                r_dist = 8.0 - 2.5 * (dist_err_m**2)
            else:
                r_dist = 8.0 - 2.5 * (delta**2) - 0.45 * (dist_err_m - delta)
            
            v_c = self.sim.vel[:2]
            v_t = target_vel_w[:2]
            nc = float(np.linalg.norm(v_c))
            nt = float(np.linalg.norm(v_t))
            if nc > 0.5 and nt > 0.5 and raw_in_view:
                r_lead_vec = 3.0 * float(max(0.0, np.dot(v_c, v_t) / (nc * nt)))
            
            v_fwd = float(self.sim.vel[0] * np.cos(self.sim.yaw) + self.sim.vel[1] * np.sin(self.sim.yaw))
            if abs(raw_ex) < 0.20 and abs(self.filtered_d_ex) < 0.30:
                r_progress = 3.5 * float(np.clip(closing_speed, 0.0, 5.0)) if dist > self.standoff_max else 2.0 * float(np.exp(-0.5 * (abs(closing_speed) / 1.0)**2))
            else:
                r_brake = 2.0 * float(max(0.0, -action[0])) if v_fwd > 4.5 else 0.0
                p_skid = 2.5 * float((v_fwd / 12.0) * max(0.0, abs(raw_ex) - 0.15))
                r_progress = r_brake - p_skid
                
            r_basket_dwell = (6.0 + 18.0 * (dwell_progress**2)) if in_basket else 0.0

        elif self.reward_mode == "pn_sprint_apex_v3":
            # 1. Proportional Navigation Velocity Vector Alignment + Monotonic Closing Potential
            r_alive = 0.50 if (raw_in_view and dist <= 6.0) else 0.0
            dist_err_m = abs(dist - self.standoff_nominal)
            r_dist = (8.0 * float(np.exp(-0.5 * (dist_err_m / 1.6)**2)) - 0.40 * float(max(0.0, dist - self.standoff_max))) if raw_in_view else -0.20 * float(max(0.0, dist - self.standoff_max))

            v_c = self.sim.vel[:2]
            v_t = target_vel_w[:2]
            nc = float(np.linalg.norm(v_c))
            nt = float(np.linalg.norm(v_t))
            if nc > 0.5 and nt > 0.5 and raw_in_view:
                align = float(np.dot(v_c, v_t) / (nc * nt))
                r_lead_vec = 4.0 * float(max(0.0, align))
                if align < 0.65:
                    r_lead_vec -= 2.0 * float(nc / 12.0) * (0.65 - align)

            # Monotonic distance closing potential when far
            if dist > self.standoff_max:
                r_progress = 4.0 * float(np.clip(closing_speed, -1.0, 6.0))
            else:
                r_progress = 1.0 * float(np.exp(-0.5 * (closing_speed / 1.0)**2)) if abs(raw_ex) < 0.20 else 0.0
            r_basket_dwell = (12.0 + 16.0 * dwell_progress) if in_basket else 0.0

        elif self.reward_mode == "centripetal_load_envelope":
            # 2. Physics-Grounded Centripetal Acceleration & Safe Speed Envelope Matching
            r_alive = 0.50 if (raw_in_view and dist <= 6.0) else 0.0
            dist_err_m = abs(dist - self.standoff_nominal)
            r_dist = (8.0 * float(np.exp(-0.5 * (dist_err_m / 1.6)**2)) - 0.40 * float(max(0.0, dist - self.standoff_max))) if raw_in_view else -0.20 * float(max(0.0, dist - self.standoff_max))

            # Optical bearing rate demand -> required cornering velocity
            omega_turn = float(max(0.08, 2.5 * abs(self.filtered_d_ex)))
            v_safe = float(np.clip(5.5 / omega_turn, 3.2, 12.0))
            v_fwd = float(self.sim.vel[0] * np.cos(self.sim.yaw) + self.sim.vel[1] * np.sin(self.sim.yaw))

            r_envelope = 3.5 * float(np.exp(-0.5 * ((v_fwd - v_safe) / 2.0)**2)) if raw_in_view else 0.0
            if dist > self.standoff_max:
                r_progress = 3.5 * float(np.clip(closing_speed, 0.0, 5.0)) + r_envelope
            else:
                r_progress = r_envelope
            r_basket_dwell = (10.0 + 15.0 * dwell_progress) if in_basket else 0.0

        elif self.reward_mode == "retinal_slip_raptor_v3":
            # 3. Biological Raptor Attack Model: Retinal Slip vs Forward Closing Drive
            r_alive = 0.50 if (raw_in_view and dist <= 6.0) else 0.0
            dist_err_m = abs(dist - self.standoff_nominal)
            r_dist = (8.0 * float(np.exp(-0.5 * (dist_err_m / 1.6)**2)) - 0.40 * float(max(0.0, dist - self.standoff_max))) if raw_in_view else -0.20 * float(max(0.0, dist - self.standoff_max))

            slip = float(np.hypot(self.filtered_d_ex, self.filtered_d_ey))
            v_fwd = float(self.sim.vel[0] * np.cos(self.sim.yaw) + self.sim.vel[1] * np.sin(self.sim.yaw))

            if slip < 0.22:
                r_progress = 4.5 * float(np.clip(closing_speed, 0.0, 6.0)) if dist > self.standoff_max else 1.0
            else:
                p_slip_kinetic = 3.5 * float((v_fwd / 12.0)**2 * min(2.0, slip / 0.40))
                r_progress = -p_slip_kinetic

            v_c = self.sim.vel[:2]
            v_t = target_vel_w[:2]
            nc = float(np.linalg.norm(v_c))
            nt = float(np.linalg.norm(v_t))
            if nc > 0.5 and nt > 0.5 and raw_in_view:
                r_lead_vec = 2.5 * float(max(0.0, np.dot(v_c, v_t) / (nc * nt)))
            r_basket_dwell = (10.0 + 15.0 * dwell_progress) if in_basket else 0.0

        elif self.reward_mode == "huber_standoff_dogfight":
            # 4. Huber Standoff Potential Well + Closing Momentum + Velocity Matching
            r_alive = 0.50 if (raw_in_view and dist <= 6.0) else 0.0
            delta = 1.0
            dist_err_m = abs(dist - self.standoff_nominal)
            if dist_err_m <= delta:
                r_dist = 8.0 - 2.5 * (dist_err_m**2)
            else:
                r_dist = 8.0 - 2.5 * (delta**2) - 0.40 * (dist_err_m - delta)

            if dist > self.standoff_max:
                r_progress = 3.5 * float(np.clip(closing_speed, 0.0, 5.0))
            else:
                r_progress = 0.0

            # 3D relative velocity matching at standoff
            if 2.5 <= dist <= 5.0 and raw_in_view:
                rel_v = float(np.linalg.norm(self.sim.vel - target_vel_w))
                r_vel_match = 3.0 * float(np.exp(-0.5 * (rel_v / 2.0)**2))
                r_progress += r_vel_match

            v_c = self.sim.vel[:2]
            v_t = target_vel_w[:2]
            nc = float(np.linalg.norm(v_c))
            nt = float(np.linalg.norm(v_t))
            if nc > 0.5 and nt > 0.5 and raw_in_view:
                r_lead_vec = 2.5 * float(max(0.0, np.dot(v_c, v_t) / (nc * nt)))
            r_basket_dwell = (8.0 + 20.0 * float(dwell_progress**1.5)) if in_basket else 0.0

        elif self.reward_mode == "aerodynamic_sideslip_governor":
            # 5. Aerodynamic Sideslip Governor with Dynamic Sprint Closing
            r_alive = 0.50 if (raw_in_view and dist <= 6.0) else 0.0
            dist_err_m = abs(dist - self.standoff_nominal)
            r_dist = (8.0 * float(np.exp(-0.5 * (dist_err_m / 1.6)**2)) - 0.40 * float(max(0.0, dist - self.standoff_max))) if raw_in_view else -0.20 * float(max(0.0, dist - self.standoff_max))

            c_yaw, s_yaw = np.cos(self.sim.yaw), np.sin(self.sim.yaw)
            v_fwd = float(c_yaw * self.sim.vel[0] + s_yaw * self.sim.vel[1])
            v_right = float(-s_yaw * self.sim.vel[0] + c_yaw * self.sim.vel[1])
            sideslip_rad = float(np.arctan2(v_right, max(0.5, v_fwd)))

            if abs(raw_ex) < 0.18 and abs(self.filtered_d_ex) < 0.25:
                r_progress = 4.0 * float(np.clip(closing_speed, 0.0, 5.0)) if dist > self.standoff_max else 1.5
            else:
                r_brake = 2.5 * float(max(0.0, -action[0])) if v_fwd > 4.0 else 0.0
                p_skid = 2.5 * float((v_fwd / 12.0) * (np.sin(sideslip_rad)**2 + max(0.0, abs(raw_ex) - 0.15)))
                r_progress = r_brake - p_skid

            v_c = self.sim.vel[:2]
            v_t = target_vel_w[:2]
            nc = float(np.linalg.norm(v_c))
            nt = float(np.linalg.norm(v_t))
            if nc > 0.5 and nt > 0.5 and raw_in_view:
                r_lead_vec = 2.5 * float(max(0.0, np.dot(v_c, v_t) / (nc * nt)))
            r_basket_dwell = (10.0 + 15.0 * dwell_progress) if in_basket else 0.0

        elif self.reward_mode.startswith("expressive") or self.reward_mode == "expressive_iterative":
            # Expressive Goal-Directed: Fast arrival + 3.5m continuous pull + bucket reward blast
            r_alive = 0.50 if (raw_in_view and dist <= 6.0) else 0.0
            dist_err_m = abs(dist - self.standoff_nominal)
            sig = float(self.expressive_cfg.get("dist_sigma", 1.5))
            peak = float(self.expressive_cfg.get("dist_peak", 8.0))
            pull = float(self.expressive_cfg.get("dist_pull", 0.30))

            if raw_in_view:
                r_dist = peak * float(np.exp(-0.5 * (dist_err_m / sig)**2)) - pull * float(max(0.0, dist - self.standoff_max))
            else:
                r_dist = 1.0 * float(np.exp(-0.5 * (dist_err_m / 3.0)**2)) - 0.20 * float(max(0.0, dist - self.standoff_max))

            # Bucket reward blast (base + quadratic dwell acceleration)
            blast_b = float(self.expressive_cfg.get("blast_base", 15.0))
            dwell_s = float(self.expressive_cfg.get("dwell_scale", 6.0))
            r_basket_dwell = (blast_b + dwell_s * float(self.continuous_lock_dwell_s**2)) if in_basket else 0.0

            # Proportional Navigation velocity vector alignment
            pn_w = float(self.expressive_cfg.get("pn_lead_weight", 2.0))
            v_c = self.sim.vel[:2]
            v_t = target_vel_w[:2]
            nc = float(np.linalg.norm(v_c))
            nt = float(np.linalg.norm(v_t))
            if nc > 0.5 and nt > 0.5 and raw_in_view and pn_w > 0:
                r_lead_vec = pn_w * float(max(0.0, np.dot(v_c, v_t) / (nc * nt)))
            else:
                r_lead_vec = 0.0

            # Sprint Closing & Speed Matching Potential
            sprint_w = float(self.expressive_cfg.get("sprint_weight", 3.5))
            if dist > self.standoff_max:
                r_sprint = sprint_w * float(np.clip(closing_speed, -1.0, 6.0))
            else:
                r_sprint = 2.0 * float(np.exp(-0.5 * (closing_speed / 1.0)**2)) if abs(raw_ex) < 0.25 else 0.0

            # Combined Progress: Sprint acceleration outside + arrival jackpot upon entry + speed matching inside
            r_progress = r_first_bucket + r_sprint

        else:
            # Default: "monotone_potential" (Linear-Exponential Hybrid)
            r_alive = 0.50 if (raw_in_view and dist <= 6.0) else 0.0
            dist_err_m = abs(dist - self.standoff_nominal)
            if raw_in_view:
                r_peak = 8.0 * float(np.exp(-0.5 * (dist_err_m / 1.6)**2))
                r_tail = -0.40 * float(max(0.0, dist - self.standoff_max))
                r_dist = r_peak + r_tail
            else:
                r_dist = 1.0 * float(np.exp(-0.5 * (dist_err_m / 3.0)**2)) - 0.20 * float(max(0.0, dist - self.standoff_max))
            if dist > self.standoff_max:
                r_progress = 3.0 * float(np.clip(closing_speed, 0.0, 5.0)) if closing_speed > 0 else 1.5 * float(np.clip(closing_speed, -5.0, 0.0))
            else:
                r_progress = 0.0
            r_basket_dwell = (8.0 + 10.0 * dwell_progress) if in_basket else 0.0

        # Continuous Field-of-View Retention & Centering Guidance
        if raw_in_view:
            ang_err = float(np.sqrt(raw_ex_unclipped**2 + raw_ey_unclipped**2))
            r_fov_base = 2.0 / (1.0 + (ang_err / 0.45)**2)
            dist_fov_scale = float(1.0 / (1.0 + max(0.0, dist - self.standoff_max) / 6.0))
            r_fov = r_fov_base * dist_fov_scale

            # Predictive Lead Tracking Bonus
            yaw_rate_norm = float(abs(self.sim.yaw_rate) / np.deg2rad(self.max_yaw_rate_deg))
            if ang_err <= 0.35 and yaw_rate_norm > 0.20:
                r_lead = 1.0 * yaw_rate_norm
            else:
                r_lead = 0.0
        else:
            r_fov = 0.0
            r_lead = 0.0

        # Directional re-acquisition guidance when sightline is lost
        r_reacq = 0.0 if raw_in_view else 0.2 * float(np.sign(self.last_seen_ex) * action[3])

        # Positive Reward Accumulator
        r_positive = (
            r_alive + r_fov + r_lead + r_lead_vec + r_dist + r_progress
            + r_first_contact + r_basket_dwell + r_reacq + r_milestone + r_sustain
        )

        # -------------------------------------------------------------
        # Multiplicative Quality Factors Q in [0.35, 1.0]
        # (Sub-optimal behaviors scale down positive rewards rather than inflicting negative pain)
        # -------------------------------------------------------------
        # Q1. Visual Alignment Quality (soft frame border dampening)
        if raw_in_view:
            edge_dist = max(0.0, abs(raw_ex) - 0.60) + max(0.0, abs(raw_ey) - 0.60)
            p_edge = float(edge_dist)
            q_fov = float(np.clip(1.0 - 0.5 * edge_dist, 0.40, 1.0))
        else:
            p_edge = 0.0
            q_fov = 0.25  # Meaningfully reduced reward when blind, but non-zero

        # Q2. Flight Smoothness & Jerk Quality
        act_diff = action - self.last_action
        act_jerk = act_diff - self.prev_action_diff
        p_slew = 0.04 * float(np.sum(act_diff**2))
        p_yaw_jerk = 0.20 * float(act_jerk[3]**2)
        p_yaw_slew = 0.15 * float(act_diff[3]**2)
        p_jerk = p_slew + p_yaw_jerk + p_yaw_slew
        q_smooth = float(np.clip(1.0 - 0.20 * p_jerk, 0.40, 1.0))

        self.prev_action_diff = act_diff.copy()
        self.last_action = action.copy()

        # Q3. Altitude Safety Quality (cushion below 1.6m AGL)
        alt_agl = -self.sim.pos[2]
        if alt_agl < 1.6:
            q_alt = float(np.clip(1.0 - 0.35 * (1.6 - alt_agl)**2, 0.35, 1.0))
        else:
            q_alt = 1.0

        # Q4. Deceleration & Optical Pitch Quality (allow up to -18° backward pitch during dynamic braking)
        pitch_deg = float(np.rad2deg(self.sim.pitch))
        if pitch_deg < -18.0:
            q_pitch = float(np.clip(1.0 - 0.02 * (abs(pitch_deg) - 18.0)**2, 0.50, 1.0))
        else:
            q_pitch = 1.0

        # Q5. Approach Glide-Slope Overspeed Quality (allow up to 5.0 m/s closing speed during high-speed sprint closing)
        if self.standoff_nominal < dist <= 9.0:
            closing_speed = (self.prev_distance_error - dist_err) / self.dt
            max_allowed = max(2.5, float(np.sqrt(2.0 * 5.0 * (dist - self.standoff_nominal))))
            if closing_speed > max_allowed:
                excess_closing = closing_speed - max_allowed
                q_speed = float(np.clip(1.0 - 0.10 * excess_closing, 0.60, 1.0))
            else:
                q_speed = 1.0
        else:
            q_speed = 1.0

        # Control effort regularization (mild micro-trim)
        p_effort = 0.02 * float(np.sum(action**2))

        self.prev_trail_error = d_trail
        self.prev_distance_error = dist_err

        # Total Composite Step Reward (Guaranteed strictly non-negative for all non-terminal steps!)
        step_reward = max(0.05, r_positive * q_fov * q_smooth * q_alt * q_pitch * q_speed - p_effort) + terminal_reward

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
            "p_yaw_jerk": float(p_yaw_jerk),
            "p_yaw_slew": float(p_yaw_slew),
            "p_effort": float(p_effort),
            "termination_reason": termination_reason,
            "r_fov": float(r_fov),
            "r_center": 0.0,
            "p_edge": float(p_edge),
            "r_dist": float(r_dist),
            "r_gamma": float(r_dist),
            "r_basket_dwell": float(r_basket_dwell),
            "r_first_contact": float(r_first_contact),
            "p_blind": 0.0 if raw_in_view else 1.0,
            "p_pitch_blind": 1.0 - q_pitch,
            "r_reacq": float(r_reacq),
            "r_milestone": float(r_milestone),
            "p_proximity": 0.0,
            "p_low_alt": 1.0 - q_alt,
            "p_sink": 0.0,
            "p_overspeed": 1.0 - q_speed,
            "q_fov": float(q_fov),
            "q_smooth": float(q_smooth),
            "q_alt": float(q_alt),
            "q_pitch": float(q_pitch),
            "q_speed": float(q_speed),
            "r_lead": float(r_lead),
        }

        return obs, float(step_reward), bool(terminated), bool(truncated), info
