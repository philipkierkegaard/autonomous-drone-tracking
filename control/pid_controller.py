"""
4-DOF Decoupled Vector PID Controller for Autonomous Drone Visual Servoing & Interception.
Designed for direct deployment to airborne companion computers (e.g. Jetson / Raspberry Pi / PX4).

Inputs:
    Vision Telemetry from track_pipeline.py (error_x, error_y, target_size / distance / error_range, status)
    Instantaneous drone pitch and roll angles from Pixhawk IMU (for full SO(3) 3D ray back-projection)

Outputs:
    4D Flight Command Vector: [v_forward (m/s), v_right (m/s), v_down (m/s), yaw_rate (deg/s)]
    Directly corresponds to PX4/MAVSDK VelocityBodyYawspeed offboard control axes.
"""

import sys
import math
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Ensure control_model directory is on path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

try:
    from control.simulation import FastPixhawkQuadSim
    from control.spatial import unproject_camera_to_horizon
except ImportError:
    from simulation import FastPixhawkQuadSim
    from spatial import unproject_camera_to_horizon


# ==============================================================================
# 1. 3D Kinematic Visual Servoing Controller for Direct Airframe Deployment
# ==============================================================================

class KinematicVisualServoController:
    """
    4-DOF Decoupled Kinematic Visual Servoing Controller for Drone Interception & Tracking.
    
    The 4 Controlled Flight Axes:
        Axis 0: Forward Velocity v_x (Surge axis - regulates bounding box size / standoff)
        Axis 1: Lateral Velocity v_y (Sway axis - coordinates roll strafe to center azimuth)
        Axis 2: Vertical Velocity v_z (Heave axis in NED - aligns altitude with pitch feedforward)
        Axis 3: Yaw Turn Rate yawspeed (Yaw axis - aligns drone heading with target)
        
    Output:
        np.ndarray of shape (4,): [v_forward (m/s), v_right (m/s), v_down (m/s), yaw_rate (deg/s)]
        Directly corresponds to PX4/MAVSDK VelocityBodyYawspeed offboard control axes.
    """
    def __init__(
        self,
        camera_uptilt_deg: float = 15.0,
        hfov_deg: float = 62.0,
        vfov_deg: float = 38.0,
        desired_bbox_size: float = 110.0,    # Desired target bounding box width (pixels, 110px corresponds to w_norm=0.0859 on 1280x720)
        desired_standoff_dist: float = 3.5,  # Target standoff distance in meters (3.5m)
        use_bbox_size: bool = True,          # Strictly regulates optical bounding box visual scale
        scale_mode: str = "width",           # Scale definition: 'width' (w_bbox, matches PPO), 'max' (legacy max(w, h)), or 'height'
        deadband_dist: float = 0.35,         # Deadband cushion in meters (+/- 0.35m around standoff)
        # 3D/4D PID Gains: [Forward (X), Lateral (Y), Vertical (Z), Yaw (Psi)]
        kp: np.ndarray = None,
        ki: np.ndarray = None,
        kd: np.ndarray = None,
        # Backward-compatible individual gain parameters
        kp_forward: float = 1.2,
        ki_forward: float = 0.0,
        kd_forward: float = None,
        kp_lat: float = 2.5,
        ki_lat: float = 0.0,
        kd_lat: float = 0.35,
        kp_z: float = None,
        ki_z: float = None,
        kd_z: float = None,
        kp_yaw: float = None,
        ki_yaw: float = None,
        kd_yaw: float = None,
        # Output Saturation Limits per axis: [Forward, Lateral, Vertical, Yaw]
        min_limits: np.ndarray = None,       # Default: [-0.5 m/s fwd (gentle reverse cushion), -4.0 m/s lat, -3.0 m/s climb, -120 deg/s yaw]
        max_limits: np.ndarray = None,       # Default: [15.0 m/s fwd, +4.0 m/s lat, +2.0 m/s desc, +120 deg/s yaw]
        # Integrator Anti-Windup Clamping Bounds
        int_limits: np.ndarray = None,       # Anti-windup clamping bounds for active integrators
        # Acceleration & Slew Rate Limits (Calibrated for ~2 kg quadrotor flight envelope)
        max_accel: float = 6.0,              # m/s^2 forward acceleration (calibrated ~31.5 deg tilt ramp for 2 kg quad)
        max_decel: float = 6.0,              # m/s^2 calibrated braking deceleration (matched identically to RL envelope)
        max_accel_z: float = 3.0,            # m/s^2 vertical climb/descent acceleration limit
        max_accel_yaw: float = 240.0,        # deg/s^2 yaw angular acceleration limit (matched identically to RL envelope)
        filter_tau: float = 0.06,            # Derivative low-pass filter time constant (seconds)
        max_yawspeed_deg: float = None,
        # 4-DOF Lateral Strafe (Roll Tilt) Control
        enable_lateral_strafe: bool = True,
        max_lat_vel: float = 4.0,            # Maximum lateral velocity in m/s (matched identically to RL envelope)
        max_lat_accel: float = 6.0,          # Maximum lateral acceleration in m/s^2 (matched identically to RL envelope)
        centering_damping: float = 0.15,     # Cornering deceleration factor per unit horizontal error
        min_centering_factor: float = 0.35,  # Speed retention floor during sharp turns (allows braking down to 35% speed)
        search_mode: str = "momentum_cruise", # 'momentum_cruise' (preserve pursuit momentum), 'decay' (smooth brake), or 'brake' (immediate hover)
        coast_mode: str = "decay",           # 'decay' (inhibit forward acceleration & smooth decay), 'brake' (immediate hover), or 'follow'
    ):
        self.camera_uptilt = np.deg2rad(camera_uptilt_deg)
        self.half_hfov = np.tan(np.deg2rad(hfov_deg / 2.0))
        self.half_vfov = np.tan(np.deg2rad(vfov_deg / 2.0))
        self.desired_bbox_size = desired_bbox_size
        self.standoff_dist = desired_standoff_dist
        self.use_bbox_size = use_bbox_size
        self.scale_mode = str(scale_mode).lower().strip()
        self.deadband_dist = float(deadband_dist)
        self.kp_forward = float(kp_forward)
        self.centering_damping = float(centering_damping)
        self.min_centering_factor = float(min_centering_factor)
        self.search_mode = str(search_mode).lower().strip()
        self.coast_mode = str(coast_mode).lower().strip()

        self.lost_time = 0.0

        # 4-DOF Gains: [Forward (X), Lateral (Y), Vertical (Z), Yaw (Psi)]
        fwd_kp = float(kp_forward)
        fwd_ki = float(ki_forward)
        fwd_kd = float(kd_forward) if kd_forward is not None else 0.15

        lat_kp = float(kp_lat) if enable_lateral_strafe else 0.0
        lat_ki = float(ki_lat) if enable_lateral_strafe else 0.0
        lat_kd = float(kd_lat) if enable_lateral_strafe else 0.0

        z_kp = float(kp_z) if kp_z is not None else 6.0
        z_ki = float(ki_z) if ki_z is not None else 0.0
        z_kd = float(kd_z) if kd_z is not None else 1.0

        yaw_kp = float(kp_yaw) if kp_yaw is not None else 180.0
        yaw_ki = float(ki_yaw) if ki_yaw is not None else 0.0
        yaw_kd = float(kd_yaw) if kd_yaw is not None else 20.0

        default_kp = np.array([fwd_kp, lat_kp, z_kp, yaw_kp], dtype=np.float64)
        default_ki = np.array([fwd_ki, lat_ki, z_ki, yaw_ki], dtype=np.float64)
        default_kd = np.array([fwd_kd, lat_kd, z_kd, yaw_kd], dtype=np.float64)

        if kp is not None:
            self.kp = np.array(kp if len(kp) == 4 else [kp[0], lat_kp, kp[1], kp[2]], dtype=np.float64)
        else:
            self.kp = default_kp.copy()

        if ki is not None:
            self.ki = np.array(ki if len(ki) == 4 else [ki[0], lat_ki, ki[1], ki[2]], dtype=np.float64)
        else:
            self.ki = default_ki.copy()

        if kd is not None:
            self.kd = np.array(kd if len(kd) == 4 else [kd[0], lat_kd, kd[1], kd[2]], dtype=np.float64)
        else:
            self.kd = default_kd.copy()

        # Override individual gains if provided
        if kp_forward is not None: self.kp[0] = float(kp_forward)
        if ki_forward is not None: self.ki[0] = float(ki_forward)
        if kd_forward is not None: self.kd[0] = float(kd_forward)
        if kp_lat is not None: self.kp[1] = float(kp_lat)
        if ki_lat is not None: self.ki[1] = float(ki_lat)
        if kd_lat is not None: self.kd[1] = float(kd_lat)
        if kp_z is not None:   self.kp[2] = float(kp_z)
        if ki_z is not None:   self.ki[2] = float(ki_z)
        if kd_z is not None:   self.kd[2] = float(kd_z)
        if kp_yaw is not None: self.kp[3] = float(kp_yaw)
        if ki_yaw is not None: self.ki[3] = float(ki_yaw)
        if kd_yaw is not None: self.kd[3] = float(kd_yaw)

        # Output Saturation Limits per axis: [Forward, Lateral, Vertical, Yaw]
        yaw_limit = float(max_yawspeed_deg) if max_yawspeed_deg is not None else 120.0
        lat_limit = float(max_lat_vel)

        if min_limits is not None:
            if len(min_limits) == 4:
                self.min_limits = np.array(min_limits, dtype=np.float64)
            elif len(min_limits) == 3:
                # [vx_min, vz_min, yaw_min]
                vx_min = float(min_limits[0])
                self.min_limits = np.array([vx_min, -lat_limit, min_limits[1], min_limits[2]], dtype=np.float64)
            else:
                self.min_limits = np.array([-0.5, -lat_limit, -3.0, -yaw_limit], dtype=np.float64)
        else:
            self.min_limits = np.array([-0.5, -lat_limit, -3.0, -yaw_limit], dtype=np.float64)

        if max_limits is not None:
            if len(max_limits) == 4:
                self.max_limits = np.array(max_limits, dtype=np.float64)
            elif len(max_limits) == 3:
                self.max_limits = np.array([max_limits[0], lat_limit, max_limits[1], max_limits[2]], dtype=np.float64)
            else:
                self.max_limits = np.array([15.0, lat_limit, 2.0, yaw_limit], dtype=np.float64)
        else:
            self.max_limits = np.array([15.0, lat_limit, 2.0, yaw_limit], dtype=np.float64)

        if int_limits is not None:
            if len(int_limits) == 4:
                self.int_limits = np.array(int_limits, dtype=np.float64)
            elif len(int_limits) == 3:
                self.int_limits = np.array([20.0, 2.0, int_limits[1], int_limits[2]], dtype=np.float64)
            else:
                self.int_limits = np.array([20.0, 2.0, 1.2, 25.0], dtype=np.float64)
        else:
            self.int_limits = np.array([20.0, 2.0, 1.2, 25.0], dtype=np.float64)

        self.filter_tau = filter_tau

        self.max_accel = max_accel
        self.max_decel = max_decel
        self.max_accel_z = float(max_accel_z)
        self.max_accel_yaw = float(max_accel_yaw)
        self.max_lat_vel = lat_limit
        self.max_lat_accel = float(max_lat_accel)
        self.prev_cmd = np.zeros(4, dtype=np.float64)
        self.prev_cmd_vx = 0.0
        self.prev_cmd_vy = 0.0
        self.last_seen_ex = 0.0
        self.last_seen_ey = 0.0
        self.prev_bbox_w = None
        self.filtered_dw = 0.0
        self.last_valid_target_vx = 0.0
        self.coast_time = 0.0

        # Vectorized internal state (shape: (4,))
        self.integral = np.zeros(4, dtype=np.float64)
        self.prev_error = np.zeros(4, dtype=np.float64)
        self.filtered_deriv = np.zeros(4, dtype=np.float64)
        self.initialized = False

    def reset(self):
        """Resets integrator and derivative states."""
        self.integral = np.zeros(4, dtype=np.float64)
        self.prev_error = np.zeros(4, dtype=np.float64)
        self.filtered_deriv = np.zeros(4, dtype=np.float64)
        self.prev_cmd = np.zeros(4, dtype=np.float64)
        self.prev_cmd_vx = 0.0
        self.prev_cmd_vy = 0.0
        self.last_seen_ex = 0.0
        self.last_seen_ey = 0.0
        self.prev_bbox_w = None
        self.filtered_dw = 0.0
        self.last_valid_target_vx = 0.0
        self.coast_time = 0.0
        self.initialized = False
        self.lost_time = 0.0

    def compute_cmd(
        self,
        telemetry: dict,
        drone_pitch: float = 0.0,
        drone_roll: float = 0.0,
        dt: float = 0.02,
        vehicle_vel: Optional[np.ndarray] = None,
        vehicle_yaw_rate: float = 0.0,
        current_alt_m: float = 2.5,
        **kwargs
    ) -> np.ndarray:
        """
        Main interface: Takes vision telemetry dictionary and returns a 3D or 4D command vector.
        Features 3D Ray Back-Projection SO(3) decoupling to cancel out cross-axis roll/pitch crosstalk
        and the +15-degree mechanical mount up-tilt.

        Args:
            telemetry: Dict from track_pipeline.py with 'error_x', 'error_y',
                       'target_size' / 'error_range' / 'distance', 'status'
            drone_pitch: Instantaneous vehicle pitch angle in radians (+ is nose down)
            drone_roll: Instantaneous vehicle roll angle in radians (+ is right wing down)
            dt: Loop period (seconds)

        Returns:
            np.ndarray of shape (4,): [v_forward (m/s), v_right (m/s), v_down (m/s), yaw_rate (deg/s)]
            Directly corresponds to MAVSDK VelocityBodyYawspeed.
        """
        # Strictly sanitize telemetry: KinematicVisualServoController is 100% optical and must never receive ground-truth distance
        telemetry = {k: v for k, v in telemetry.items() if k != "distance"}

        # Seed previous commanded velocity from measured vehicle velocity on first step
        # to ensure bumpless transfer and prevent slew-rate limiter from forcing an artificial brake
        if not self.initialized and vehicle_vel is not None and len(vehicle_vel) >= 3:
            self.prev_cmd[0] = float(vehicle_vel[0])
            self.prev_cmd[1] = float(vehicle_vel[1])
            self.prev_cmd[2] = float(vehicle_vel[2])
            self.prev_cmd_vx = float(vehicle_vel[0])
            self.prev_cmd_vy = float(vehicle_vel[1])

        status = telemetry.get("status", "SEARCHING")
        if status == "SEARCHING":
            self.lost_time += dt
            self.initialized = False  # Reset initialized state to eliminate derivative spike upon re-acquisition
            if self.search_mode == "momentum_cruise":
                # Option C: Momentum Cruise & Visual Search
                # Preserves forward pursuit speed (nominally 4.0 m/s) and continues turning along
                # the last seen azimuth direction for up to 2.5s, preventing sudden freezing/hovering
                # when target transiently drops out during sharp maneuvers.
                self.integral.fill(0.0)
                if self.lost_time < 2.5:
                    # Hold last commanded pursuit speed decaying smoothly without artificial speed floor
                    decay_factor = max(0.0, 1.0 - (self.lost_time / 2.5))
                    cruise_speed = max(0.0, min(self.max_limits[0], self.prev_cmd_vx * decay_factor))
                    cmd_yaw = float(np.clip(self.prev_cmd[3] * decay_factor, -60.0, 60.0))
                else:
                    cruise_speed = max(0.0, self.prev_cmd_vx - 3.0 * dt)
                    cmd_yaw = 0.0
                self.prev_cmd_vx = cruise_speed
                self.prev_cmd[0] = cruise_speed
                self.prev_cmd[1] = 0.0
                self.prev_cmd[3] = cmd_yaw
                return np.array([cruise_speed, 0.0, 0.0, cmd_yaw], dtype=np.float64)
            elif self.search_mode == "decay":
                # Option A: Smooth Forward Speed Decay & Turn Recovery
                # Bleeds off forward momentum at 3.0 m/s^2 while continuing yaw search in the direction
                # where the target was last seen, preventing blind straight-line overshoot.
                self.integral.fill(0.0)
                cmd_vx = max(0.0, self.prev_cmd_vx - 3.0 * dt)
                if self.lost_time < 1.5 and abs(self.last_seen_ex) > 0.05:
                    sweep_dir = 1.0 if self.last_seen_ex > 0.0 else -1.0
                    cmd_yaw = float(sweep_dir * min(60.0, max(25.0, abs(self.prev_cmd[3]))))
                else:
                    cmd_yaw = 0.0
                self.prev_cmd_vx = cmd_vx
                self.prev_cmd[0] = cmd_vx
                self.prev_cmd[1] = 0.0
                self.prev_cmd[3] = cmd_yaw
                return np.array([cmd_vx, 0.0, 0.0, cmd_yaw], dtype=np.float64)
            elif self.search_mode == "brake":
                # Option B: Immediate Active Brake & Station-Keeping Hover
                # Arrests all forward, lateral, vertical, and angular momentum immediately.
                self.integral.fill(0.0)
                self.prev_error.fill(0.0)
                self.filtered_deriv.fill(0.0)
                self.prev_cmd = np.zeros(4, dtype=np.float64)
                self.prev_cmd_vx = 0.0
                self.prev_cmd_vy = 0.0
                return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            else:
                # Legacy multi-phase coast governor (kept for benchmark comparison)
                if self.lost_time < 1.0:
                    cmd_vx = max(0.0, self.prev_cmd_vx - 1.5 * dt)
                    return np.array([cmd_vx, 0.0, 0.0, 0.0], dtype=np.float64)
                elif self.lost_time < 2.0:
                    return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
                else:
                    return np.array([0.0, 0.0, 0.0, 15.0], dtype=np.float64)

        is_coasting = (status == "COASTING")
        if is_coasting and self.coast_mode == "brake":
            # Immediate active brake on coasting: do not pursue unmeasured Kalman predictions
            self.lost_time += dt
            self.initialized = False
            self.integral.fill(0.0)
            self.prev_error.fill(0.0)
            self.filtered_deriv.fill(0.0)
            self.prev_cmd = np.zeros(4, dtype=np.float64)
            self.prev_cmd_vx = 0.0
            self.prev_cmd_vy = 0.0
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Active lock: reset lost timer
        self.lost_time = 0.0

        err_x = float(telemetry.get("error_x", 0.0))
        err_y = float(telemetry.get("error_y", 0.0))

        # Update last-seen memory when target is actively in view
        if telemetry.get("in_view", True):
            self.last_seen_ex = err_x
            self.last_seen_ey = err_y
        elif is_coasting and abs(err_x) < 1e-4 and abs(err_y) < 1e-4:
            # Fall back to last seen optical errors during dropout
            err_x = self.last_seen_ex
            err_y = self.last_seen_ey

        # Non-finite (NaN / Inf) sanity barrier: abort and return safe zero-velocity holding command
        if not (np.isfinite(err_x) and np.isfinite(err_y) and np.isfinite(drone_pitch) and np.isfinite(drone_roll) and np.isfinite(dt)):
            return np.zeros(4, dtype=np.float64)

        # ----------------------------------------------------------------------
        # 1. 3D RAY BACK-PROJECTION (FULL SO(3) DECOUPLING)
        # ----------------------------------------------------------------------
        err_x_horizon, err_y_horizon, azimuth_horizon, elevation_horizon = unproject_camera_to_horizon(
            err_x=err_x,
            err_y=err_y,
            drone_pitch=drone_pitch,
            drone_roll=drone_roll,
            camera_uptilt_rad=self.camera_uptilt,
            half_hfov=self.half_hfov,
            half_vfov=self.half_vfov,
        )

        # 2. Scale / Range Error (Strictly Optical Bounding Box Scale - Zero Ground Truth Distance)
        # Bounding box width/height or error_range is the ONLY information source for range error.
        # Any ground-truth distance passed in telemetry is strictly ignored to prevent unfair advantages.
        if self.use_bbox_size:
            if "error_range" in telemetry:
                # Direct normalized scale error [-1.0 (too close), +1.0 (far)] from track_pipeline.py
                range_err = float(telemetry["error_range"])
            else:
                if self.scale_mode == "width":
                    current_sz = telemetry.get("bbox_w", None)
                    if current_sz is None:
                        if "bbox" in telemetry and len(telemetry["bbox"]) >= 4:
                            current_sz = float(telemetry["bbox"][2] - telemetry["bbox"][0])
                        else:
                            current_sz = telemetry.get("target_size", 0.0)
                elif self.scale_mode == "height":
                    current_sz = telemetry.get("bbox_h", None)
                    if current_sz is None:
                        if "bbox" in telemetry and len(telemetry["bbox"]) >= 4:
                            current_sz = float(telemetry["bbox"][3] - telemetry["bbox"][1])
                        else:
                            current_sz = telemetry.get("target_size", 0.0)
                else:  # "max"
                    current_sz = telemetry.get("target_size", 0.0)
                    if current_sz <= 0.0:
                        bw = telemetry.get("bbox_w", 0.0)
                        bh = telemetry.get("bbox_h", 0.0)
                        current_sz = max(bw, bh)

                if current_sz is not None and current_sz > 0.0:
                    range_err = (self.desired_bbox_size - current_sz) / max(1.0, self.desired_bbox_size)
                else:
                    range_err = 0.0
            range_term = float(np.clip(range_err, -1.0, 1.0))
        else:
            # When use_bbox_size is False, estimate metric distance error purely via pinhole geometry:
            #   d_est = d_nominal * (s_nominal / s_measured)
            # Never reads ground-truth Euclidean distance from simulation.
            if "error_range" in telemetry:
                scale_err = float(telemetry["error_range"])
                denom = max(0.05, 1.0 - scale_err) if scale_err < 1.0 else 0.05
                range_term = self.standoff_dist * (scale_err / denom)
            else:
                current_sz = telemetry.get("bbox_w", telemetry.get("target_size", 0.0))
                if current_sz > 0.0:
                    est_dist = self.standoff_dist * (self.desired_bbox_size / max(5.0, float(current_sz)))
                    range_term = est_dist - self.standoff_dist
                else:
                    range_term = 0.0

        # Optical Looming Target Velocity Feedforward:
        #   v_target_hat = v_drone - d_hat * (dw / w)
        # Directly estimates target forward velocity along viewing axis from optical looming,
        # ensuring exact speed-matching with zero distance error without hardcoding any constant.
        if vehicle_vel is not None and len(vehicle_vel) > 0 and np.isfinite(vehicle_vel[0]):
            v_drone_fwd = float(vehicle_vel[0])
        else:
            v_drone_fwd = float(self.prev_cmd_vx)

        current_w = float(telemetry.get("bbox_w", telemetry.get("target_size", 0.0)))
        if not is_coasting:
            self.coast_time = 0.0
        else:
            self.coast_time += dt

        # Compute optical looming relative velocity whenever bounding box is valid
        if current_w > 1.0 and telemetry.get("in_view", True):
            if self.prev_bbox_w is not None and dt > 1e-4:
                raw_dw = (current_w - self.prev_bbox_w) / dt
                alpha_w = dt / (dt + 0.08)  # 80ms low-pass filter against bounding box pixel jitter
                self.filtered_dw += alpha_w * (raw_dw - self.filtered_dw)
            self.prev_bbox_w = current_w

            # Distance estimate from pinhole geometry: d_hat = d* * (w* / w)
            d_hat = float(np.clip(self.standoff_dist * (self.desired_bbox_size / max(5.0, current_w)), 1.5, 25.0))
            loom_rate = float(self.filtered_dw / max(5.0, current_w))
            d_dot_rel = -d_hat * loom_rate
            est_v_target = float(np.clip(v_drone_fwd + d_dot_rel, 0.0, self.max_limits[0]))
            self.last_valid_target_vx = est_v_target
        else:
            # During dropout / prolonged loss, decay loom derivative and hold last known velocity for 0.5s
            self.filtered_dw *= max(0.0, 1.0 - 2.0 * dt)
            decay = max(0.0, 1.0 - 1.5 * max(0.0, self.coast_time - 0.2))
            est_v_target = float(self.last_valid_target_vx * decay)

        # Target forward velocity feedforward (uses optical looming estimate unless external is provided)
        target_vx_ff = float(telemetry.get("target_vx", telemetry.get("target_vel_x", 0.0)))
        if target_vx_ff <= 0.0:
            target_vx_ff = est_v_target

        # Verify all derived error terms are finite before proceeding
        if not (np.isfinite(range_term) and np.isfinite(target_vx_ff) and np.isfinite(err_x_horizon) and np.isfinite(elevation_horizon) and np.isfinite(azimuth_horizon)):
            return np.zeros(4, dtype=np.float64)

        # 3. 4D Error Vector: [Range Error, Lateral Error (norm), Optical Elevation (rad), Azimuth Horizon (rad)]
        # For the vertical axis (Axis 2): use camera optical elevation error to center the target
        # directly in the camera optical frame (err_y = 0), naturally establishing the ~0.9m
        # altitude lead below the target to compensate for the +15 deg camera mount up-tilt.
        elevation_optical = float(np.arctan2(err_y * self.half_vfov, 1.0))
        error_vec = np.array([range_term, err_x_horizon, elevation_optical, azimuth_horizon], dtype=np.float64)

        # 4. Apply 4-DOF control update (Surge vx, Sway vy, Heave vz, Yaw yawspeed)
        # Hold feedforward across inter-frame Kalman predictions (coast_time < 0.5s)
        return self.update(
            error_vec,
            drone_pitch=drone_pitch,
            dt=dt,
            horiz_err_norm=abs(err_x),
            decoupled=True,
            target_vx_ff=target_vx_ff if self.coast_time < 0.5 else 0.0,
            is_coasting=is_coasting,
            current_alt_m=current_alt_m,
        )

    def update(
        self,
        error_vec: np.ndarray,
        drone_pitch: float = 0.0,
        dt: float = 0.02,
        horiz_err_norm: float = 0.0,
        decoupled: bool = True,
        target_vx_ff: float = 0.0,
        is_coasting: bool = False,
        current_alt_m: Optional[float] = None,
    ) -> np.ndarray:
        """
        Vectorized 4-DOF Kinematic Control Update.
        
        Args:
            error_vec: [range_or_scale_err, lateral_err_norm, elevation_err (rad), azimuth_err (rad)]
                       (accepts 3D [range, elevation, azimuth] for backward compatibility)
            drone_pitch: Instantaneous pitch angle (+ nose down)
            dt: Loop sample step (seconds)
            horiz_err_norm: Optional normalized horizontal error for centering damping
            decoupled: If True, assumes error_vec[2] is already fully decoupled in 3D
            target_vx_ff: Optional target forward velocity feedforward in m/s
            is_coasting: True if visual target is lost and dead-reckoned by Kalman filter
            
        Returns:
            np.ndarray of shape (4,): [v_forward (m/s), v_right (m/s), v_down (m/s), yaw_rate (deg/s)]
        """
        # Clamp dt to prevent numerical spikes from OS scheduling latency or frame drops
        dt = float(np.clip(dt, 1e-4, 0.1))

        raw_e = np.array(error_vec, dtype=np.float64)
        if len(raw_e) == 3:
            # Backward compatibility: expand 3D [range, elev, azim] to 4D [range, lat, elev, azim]
            lat_err = float(np.clip(np.tan(raw_e[2]) / self.half_hfov, -1.0, 1.0))
            e = np.array([raw_e[0], lat_err, raw_e[1], raw_e[2]], dtype=np.float64)
        else:
            e = raw_e.copy()

        # Convert Axis 0 range error into physical metric distance error (meters)
        if self.use_bbox_size:
            scale_err = e[0]  # normalized scale error: (s* - s) / s*
            denom = max(0.05, 1.0 - scale_err) if scale_err < 1.0 else 0.05
            ratio = scale_err / denom
            approx_dist_err = self.standoff_dist * ratio
            dist_signal = approx_dist_err
        else:
            dist_signal = e[0]

        if dist_signal > self.deadband_dist:
            eff_err = dist_signal - self.deadband_dist
        elif dist_signal < -self.deadband_dist:
            eff_err = dist_signal + self.deadband_dist
        else:
            eff_err = 0.0

        # Axis 0 works in metric distance space: e[0] = eff_err (meters)
        e[0] = eff_err

        # ----------------------------------------------------------------------
        # 2. Vectorized Proportional Term
        # ----------------------------------------------------------------------
        p_term = self.kp * e

        # ----------------------------------------------------------------------
        # 3. Vectorized Derivative Term with Low-Pass Filter
        # ----------------------------------------------------------------------
        if not self.initialized:
            self.filtered_deriv = np.zeros(4, dtype=np.float64)
            deriv = np.zeros(4, dtype=np.float64)
            self.initialized = True
        else:
            raw_deriv = (e - self.prev_error) / dt
            alpha = dt / (dt + self.filter_tau)
            self.filtered_deriv += alpha * (raw_deriv - self.filtered_deriv)
            deriv = self.filtered_deriv

        d_term = self.kd * deriv
        self.prev_error = e.copy()

        # ----------------------------------------------------------------------
        # 4. Vectorized Integrator Update with Anti-Windup Clamping
        # ----------------------------------------------------------------------
        # Maintain integrator during active tracking and short 30Hz-to-50Hz inter-frame Kalman predictions
        if not is_coasting or getattr(self, "coast_time", 0.0) < 0.15:
            tentative_out = p_term + self.ki * self.integral + d_term
            saturated_high = (tentative_out >= self.max_limits) & (e > 0)
            saturated_low  = (tentative_out <= self.min_limits) & (e < 0)
            unclamped_mask = ~(saturated_high | saturated_low) & (self.ki > 0)

            self.integral[unclamped_mask] += e[unclamped_mask] * dt
            # Axes with ki == 0 stay cleanly at 0.0
            self.integral[self.ki == 0] = 0.0
            self.integral = np.clip(self.integral, -self.int_limits, self.int_limits)
        else:
            # During prolonged target loss (> 150ms), decay existing integrals gently
            self.integral *= max(0.0, 1.0 - 0.5 * dt)
        i_term = self.ki * self.integral

        # ----------------------------------------------------------------------
        # 5. Compute Saturated Output Vector
        # ----------------------------------------------------------------------
        raw_output = p_term + i_term + d_term

        # Pure Visual Servoing Guidance Law on Axis 0 (Forward Velocity):
        # When no external velocity feedforward is provided, forward throttle is 100% error-driven:
        #   v_fwd = v_target_ff + raw_output[0]
        nominal_cruise_speed = float(target_vx_ff) if target_vx_ff > 0.0 else 0.0
        v_pid = nominal_cruise_speed + raw_output[0]
        raw_output[0] = np.clip(v_pid, self.min_limits[0], self.max_limits[0])

        # Centering damping: gently trim excess surge during sharp lateral maneuvers without killing pursuit momentum
        if raw_output[0] > 0.0:
            centering_factor = max(self.min_centering_factor, 1.0 - horiz_err_norm * self.centering_damping)
            raw_output[0] *= centering_factor

        # COASTING behavior governance:
        # Decays forward velocity smoothly toward 0 without artificial speed floor
        if is_coasting and self.coast_mode == "decay":
            decayed_vx = max(0.0, self.prev_cmd_vx - 1.5 * dt)
            raw_output[0] = min(raw_output[0], decayed_vx)
            raw_output[1] *= 0.75

        # Ground proximity safety barrier: inhibit descent below 1.4m and push up below 1.1m
        if current_alt_m is not None and current_alt_m < 1.4:
            if raw_output[2] > 0.0:
                raw_output[2] = float(raw_output[2] * max(0.0, (current_alt_m - 0.9) / 0.5))
            if current_alt_m < 1.1:
                climb_cmd = -float(np.clip((1.1 - current_alt_m) * 3.0, 0.0, 2.5))
                raw_output[2] = min(raw_output[2], climb_cmd)

        # ----------------------------------------------------------------------
        # 6. Slew-Rate Limiting Across All 4 Flight Axes
        # ----------------------------------------------------------------------
        # Axis 0: Forward surge acceleration & deceleration
        max_decel_step = self.max_decel * dt
        max_accel_step = self.max_accel * dt
        if raw_output[0] < self.prev_cmd[0]:
            raw_output[0] = max(raw_output[0], self.prev_cmd[0] - max_decel_step)
        else:
            raw_output[0] = min(raw_output[0], self.prev_cmd[0] + max_accel_step)

        # Axis 1: Lateral sway acceleration
        max_lat_step = self.max_lat_accel * dt
        if raw_output[1] < self.prev_cmd[1]:
            raw_output[1] = max(raw_output[1], self.prev_cmd[1] - max_lat_step)
        else:
            raw_output[1] = min(raw_output[1], self.prev_cmd[1] + max_lat_step)

        # Axis 2: Vertical heave acceleration
        max_z_step = self.max_accel_z * dt
        if raw_output[2] < self.prev_cmd[2]:
            raw_output[2] = max(raw_output[2], self.prev_cmd[2] - max_z_step)
        else:
            raw_output[2] = min(raw_output[2], self.prev_cmd[2] + max_z_step)

        # Axis 3: Yaw rate angular acceleration
        max_yaw_step = self.max_accel_yaw * dt
        if raw_output[3] < self.prev_cmd[3]:
            raw_output[3] = max(raw_output[3], self.prev_cmd[3] - max_yaw_step)
        else:
            raw_output[3] = min(raw_output[3], self.prev_cmd[3] + max_yaw_step)

        # Final clip against physical envelope limits
        output_vec = np.clip(raw_output, self.min_limits, self.max_limits)
        if not np.all(np.isfinite(output_vec)):
            output_vec = np.zeros(4, dtype=np.float64)
        self.prev_cmd = output_vec.copy()
        self.prev_cmd_vx = output_vec[0]
        self.prev_cmd_vy = output_vec[1]
        return output_vec

    @staticmethod
    def to_velocity_body(cmd: np.ndarray) -> tuple:
        """
        Converts command vector into standard 4-DOF body-frame velocity arguments:
        (forward_m_s, right_m_s, down_m_s, yawspeed_deg_s).
        Directly compatible with ArduPilot MAVLink SET_POSITION_TARGET_LOCAL_NED (#84)
        as well as legacy MAVSDK VelocityBodyYawspeed.
        Accepts both 3D [v_fwd, v_down, yawspeed] and 4D [v_fwd, v_right, v_down, yawspeed].
        """
        if len(cmd) == 4:
            vals = (float(cmd[0]), float(cmd[1]), float(cmd[2]), float(cmd[3]))
        elif len(cmd) == 3:
            vals = (float(cmd[0]), 0.0, float(cmd[1]), float(cmd[2]))
        else:
            raise ValueError(f"Expected 3D or 4D command vector, got length {len(cmd)}")
        if not all(np.isfinite(v) for v in vals):
            return 0.0, 0.0, 0.0, 0.0
        return vals

    # Dual aliases for compatibility across ArduPilot and legacy MAVSDK scripts
    to_autopilot = to_velocity_body
    to_mavsdk = to_velocity_body


# Aliases for backward compatibility with existing simulation scripts
Vector3DPIDController = KinematicVisualServoController
DroneVisualPIDController = KinematicVisualServoController


# ==============================================================================
# 2. Dynamic Closed-Loop Simulation Test & 2D Top-Down Visualization
# ==============================================================================

def run_simulation_and_plot():
    """
    Simulates 20 seconds of autonomous closed-loop pursuit using KinematicVisualServoController.
    Uses pure bounding-box width regulation (desired_bbox_size = 110.0 px, w_norm=0.0859 on 1280x720).
    """
    dt = 0.02
    total_time = 20.0
    num_steps = int(total_time / dt)

    sim = FastPixhawkQuadSim(dt=dt)
    controller = KinematicVisualServoController(
        camera_uptilt_deg=15.0,
        hfov_deg=62.0,
        vfov_deg=38.0,
        desired_bbox_size=110.0,  # 110px width corresponds to w_norm=0.0859 on 1280x720
        desired_standoff_dist=3.5,
        scale_mode="width",
    )

    sim.reset(initial_pos=[0.0, 0.0, 0.0], initial_yaw=0.0)

    # 3D Circular racetrack orbit trajectory
    def get_target_pos(t):
        R = 14.0
        w = 0.25
        x = 18.0 + R * np.cos(w * t)
        y = R * np.sin(w * t)
        z = -1.5 * np.sin(0.3 * t)
        return np.array([x, y, z], dtype=np.float64)

    times = []
    chaser_pos_history = []
    target_pos_history = []
    chaser_yaw_history = []
    dist_history = []
    target_size_history = []
    err_x_history = []
    err_y_history = []
    in_view_history = []
    cmd_3d_history = []

    print(f"--- Running Kinematic Visual Servoing Controller ({total_time:.1f}s at 50 Hz) ---")

    for step in range(num_steps):
        t = step * dt
        target_world = get_target_pos(t)

        # 1. Project target onto camera sensor and formulate telemetry with bounding-box pixels
        telemetry = sim.get_camera_telemetry(
            target_world,
            hfov_deg=62.0,
            vfov_deg=38.0,
            target_w_m=0.35,
            target_h_m=0.20,
            img_w=1280,
            img_h=720,
            desired_target_size=110.0,
            scale_mode="width"
        )

        # True distance for evaluation logging only
        true_dist = float(telemetry.pop("distance", np.linalg.norm(sim.pos - target_world)))

        # 2. Compute 3D Control Vector using bounding box pixel errors (strictly zero ground-truth distance)
        cmd_3d = controller.compute_cmd(telemetry, drone_pitch=sim.pitch, dt=dt)

        # 3. Advance drone simulation
        sim.step(cmd_3d)

        # Log states
        times.append(t)
        chaser_pos_history.append(sim.pos.copy())
        target_pos_history.append(target_world.copy())
        chaser_yaw_history.append(sim.yaw)
        dist_history.append(true_dist)
        target_size_history.append(telemetry["target_size"])
        err_x_history.append(telemetry["error_x"])
        err_y_history.append(telemetry["error_y"])
        in_view_history.append(telemetry["in_view"])
        cmd_3d_history.append(cmd_3d.copy())

    times = np.array(times)
    chaser_pos = np.array(chaser_pos_history)
    target_pos = np.array(target_pos_history)
    cmd_3d_arr = np.array(cmd_3d_history)
    in_view_arr = np.array(in_view_history)

    pct_in_view = 100.0 * np.sum(in_view_arr) / len(in_view_arr)
    print(f"4-DOF Kinematic Visual Servoing Test Complete. Lock Retention: {pct_in_view:.1f}%")
    print(f"Average Tracking Distance: {np.mean(dist_history):.2f}m (Goal: 6.00m)")
    print(f"4D Command Shape: {cmd_3d.shape} -> Vector: [vx={cmd_3d[0]:.2f}, vy={cmd_3d[1]:.2f}, vz={cmd_3d[2]:.2f}, yawspeed={cmd_3d[3]:.1f}]")

    # Render 2D Top-Down Plot
    fig = plt.figure(figsize=(16, 9.5), facecolor="#0e1117")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.7, 1.0], hspace=0.32, wspace=0.28)

    # 1. Main 2D Top-Down Map
    ax_map = fig.add_subplot(gs[0, :2], facecolor="#151821")
    ax_map.grid(True, linestyle="--", alpha=0.28, color="#3e4659")
    ax_map.plot(target_pos[:, 1], target_pos[:, 0], color="#ff4757", linestyle="--", linewidth=2.2, label="Target Drone (Orbit)", zorder=3)
    ax_map.plot(chaser_pos[:, 1], chaser_pos[:, 0], color="#00d2d3", linewidth=2.8, label="Chaser Drone (4-DOF Kinematic)", zorder=4)

    ax_map.scatter(target_pos[0, 1], target_pos[0, 0], color="#ff4757", s=80, marker="o", edgecolors="white", linewidths=1.5, label="Target Start", zorder=6)
    ax_map.scatter(target_pos[-1, 1], target_pos[-1, 0], color="#ff4757", s=100, marker="X", edgecolors="white", linewidths=1.5, label="Target Finish", zorder=6)
    ax_map.scatter(chaser_pos[0, 1], chaser_pos[0, 0], color="#00d2d3", s=80, marker="o", edgecolors="white", linewidths=1.5, label="Chaser Start", zorder=6)
    ax_map.scatter(chaser_pos[-1, 1], chaser_pos[-1, 0], color="#54a0ff", s=100, marker="X", edgecolors="white", linewidths=1.5, label="Chaser Finish", zorder=6)

    # Sample intervals for FOV Wedges
    sample_indices = np.linspace(0, num_steps - 1, 9, dtype=int)
    fov_range = 8.0
    half_hfov = np.deg2rad(62.0 / 2.0)

    for idx in sample_indices:
        cx, cy = chaser_pos[idx, 0], chaser_pos[idx, 1]
        tx, ty = target_pos[idx, 0], target_pos[idx, 1]
        yaw = chaser_yaw_history[idx]
        t_sec = times[idx]

        ax_map.plot([cy, ty], [cx, tx], color="#747d8c", linestyle=":", linewidth=1.0, alpha=0.7, zorder=2)
        left_yaw = yaw - half_hfov
        right_yaw = yaw + half_hfov
        p_left = np.array([cy + fov_range * np.sin(left_yaw), cx + fov_range * np.cos(left_yaw)])
        p_right = np.array([cy + fov_range * np.sin(right_yaw), cx + fov_range * np.cos(right_yaw)])
        p_origin = np.array([cy, cx])

        wedge_poly = Polygon([p_origin, p_left, p_right], closed=True, facecolor="#00d2d3", edgecolor="#00d2d3", alpha=0.12, zorder=1)
        ax_map.add_patch(wedge_poly)
        ax_map.plot([cy, p_left[0]], [cx, p_left[1]], color="#00d2d3", linewidth=0.7, alpha=0.45)
        ax_map.plot([cy, p_right[0]], [cx, p_right[1]], color="#00d2d3", linewidth=0.7, alpha=0.45)

        ax_map.arrow(cy, cx, 1.4 * np.sin(yaw), 1.4 * np.cos(yaw), head_width=0.45, head_length=0.45, fc="#00d2d3", ec="white", zorder=5)
        ax_map.annotate(f"t={t_sec:.1f}s", (cy + 0.35, cx + 0.35), color="#dfe4ea", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#1e272e", edgecolor="#57606f", alpha=0.8))

    ax_map.set_xlabel("East Position Y (meters)", color="#dfe4ea", fontsize=11, labelpad=8)
    ax_map.set_ylabel("North Position X (meters)", color="#dfe4ea", fontsize=11, labelpad=8)
    ax_map.set_title("TOP-DOWN FLIGHT PATH: 4-DOF KINEMATIC CONTROLLER", color="#f1f2f6", fontsize=13, fontweight="bold", pad=12)
    ax_map.tick_params(colors="#a4b0be")
    ax_map.axis("equal")
    ax_map.legend(loc="upper left", facecolor="#1e272e", edgecolor="#57606f", labelcolor="#f1f2f6", fontsize=8.5)

    # 2. Summary Card
    ax_info = fig.add_subplot(gs[0, 2], facecolor="#151821")
    ax_info.axis("off")
    summary_text = (
        "4-DOF KINEMATIC IBVS\n"
        "───────────────────────────────\n"
        "OUTPUT VECTOR (4D):\n"
        "  u[0] = v_forward  (m/s)\n"
        "  u[1] = v_right    (m/s)\n"
        "  u[2] = v_down     (m/s)\n"
        "  u[3] = yaw_rate   (deg/s)\n"
        "───────────────────────────────\n"
        f"• Duration:            {total_time:.1f} s\n"
        f"• Target Maneuver:     3D Circular Orbit\n"
        f"• Target BBox Goal:    35.0 px (~6m)\n"
        f"• Mean BBox Size:      {np.mean(target_size_history):.1f} px\n"
        f"• Mean Distance:       {np.mean(dist_history):.2f} m\n"
        f"• FOV Lock Retention:  {pct_in_view:.1f}%\n"
        f"• Max Speed (vx):      {np.max(cmd_3d_arr[:, 0]):.1f} m/s\n"
        f"• Min Speed (vx brake):{np.min(cmd_3d_arr[:, 0]):.1f} m/s\n"
        f"• Max Strafe (|vy|):   {np.max(np.abs(cmd_3d_arr[:, 1])):.1f} m/s\n"
        f"• Max Climb (vz):      {np.min(cmd_3d_arr[:, 2]):.1f} m/s\n"
        f"• Max Yaw Rate:        {np.max(np.abs(cmd_3d_arr[:, 3])):.1f} °/s\n"
        "───────────────────────────────\n"
        "HARDENING UPGRADES:\n"
        "✓ 4-DOF Vector Control\n"
        "✓ Reverse Braking (-2.0 m/s)\n"
        "✓ 4-Axis Slew Rate Limiting\n"
        "✓ Azimuth Singularity Guard\n"
        "✓ Velocity Feedforward Ready\n"
        "✓ Full SO(3) Attitude Decoupled"
    )
    ax_info.text(0.05, 0.94, summary_text, color="#f1f2f6", fontsize=10, family="monospace",
                 verticalalignment="top", transform=ax_info.transAxes,
                 bbox=dict(boxstyle="round,pad=0.6", facecolor="#1e272e", edgecolor="#00d2d3", linewidth=1.2))

    # 3. Distance & Bounding Box Size Plot
    ax_dist = fig.add_subplot(gs[1, 0], facecolor="#151821")
    ax_dist.grid(True, linestyle="--", alpha=0.28, color="#3e4659")
    line1 = ax_dist.plot(times, dist_history, color="#2ed573", linewidth=2.0, label="Distance (m)")
    ax_dist.axhline(6.0, color="#2ed573", linestyle=":", linewidth=1.2, alpha=0.6)
    ax_dist.set_xlabel("Time (s)", color="#dfe4ea", fontsize=10)
    ax_dist.set_ylabel("Range (m)", color="#2ed573", fontsize=10)
    ax_dist.tick_params(colors="#a4b0be")

    ax_size = ax_dist.twinx()
    line2 = ax_size.plot(times, target_size_history, color="#ffa502", linewidth=1.8, label="BBox Size (px)")
    line3 = [ax_size.axhline(35.0, color="#ffa502", linestyle="--", linewidth=1.4, label="Goal (35 px)")]
    ax_size.set_ylabel("Target Size (pixels)", color="#ffa502", fontsize=10)
    ax_size.tick_params(colors="#a4b0be")
    ax_dist.set_title("Range & BBox Size vs. Time", color="#f1f2f6", fontsize=11, fontweight="bold")

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax_dist.legend(lines, labels, loc="upper right", facecolor="#1e272e", edgecolor="#57606f", labelcolor="#f1f2f6", fontsize=7.8)

    # 4. Normalized Camera Errors
    ax_err = fig.add_subplot(gs[1, 1], facecolor="#151821")
    ax_err.grid(True, linestyle="--", alpha=0.28, color="#3e4659")
    ax_err.plot(times, err_x_history, color="#00d2d3", linewidth=1.8, label="Azimuth e_x")
    ax_err.plot(times, err_y_history, color="#ff6b81", linewidth=1.8, label="Elevation e_y")
    ax_err.axhline(1.0, color="#ff4757", linestyle=":", linewidth=1.0)
    ax_err.axhline(-1.0, color="#ff4757", linestyle=":", linewidth=1.0)
    ax_err.set_ylim(-1.15, 1.15)
    ax_err.set_xlabel("Time (s)", color="#dfe4ea", fontsize=10)
    ax_err.set_ylabel("Error [-1, 1]", color="#dfe4ea", fontsize=10)
    ax_err.set_title("Visual Sensor Tracking Errors", color="#f1f2f6", fontsize=11, fontweight="bold")
    ax_err.tick_params(colors="#a4b0be")
    ax_err.legend(loc="lower right", facecolor="#1e272e", edgecolor="#57606f", labelcolor="#f1f2f6", fontsize=8.5)

    # 5. 4D Control Vector Components Plot
    ax_cmd = fig.add_subplot(gs[1, 2], facecolor="#151821")
    ax_cmd.grid(True, linestyle="--", alpha=0.28, color="#3e4659")
    l_vx = ax_cmd.plot(times, cmd_3d_arr[:, 0], color="#1e90ff", linewidth=1.8, label="u[0]: vx (fwd)")
    l_vy = ax_cmd.plot(times, cmd_3d_arr[:, 1], color="#00d2d3", linewidth=1.4, label="u[1]: vy (strafe)")
    l_vz = ax_cmd.plot(times, cmd_3d_arr[:, 2], color="#eccc68", linewidth=1.6, label="u[2]: vz (climb)")
    ax_cmd.set_xlabel("Time (s)", color="#dfe4ea", fontsize=10)
    ax_cmd.set_ylabel("Linear Velocity (m/s)", color="#dfe4ea", fontsize=10)
    ax_cmd.tick_params(colors="#a4b0be")

    ax_yaw = ax_cmd.twinx()
    l_yaw = ax_yaw.plot(times, cmd_3d_arr[:, 3], color="#ff7675", linewidth=1.4, linestyle="--", label="u[3]: yaw (deg/s)")
    ax_yaw.set_ylabel("Yaw Rate (deg/s)", color="#ff7675", fontsize=10)
    ax_yaw.tick_params(colors="#a4b0be")
    ax_cmd.set_title("4-DOF Flight Velocity Commands", color="#f1f2f6", fontsize=11, fontweight="bold")

    all_cmd_lines = l_vx + l_vy + l_vz + l_yaw
    all_cmd_labels = [l.get_label() for l in all_cmd_lines]
    ax_cmd.legend(all_cmd_lines, all_cmd_labels, loc="upper right", facecolor="#1e272e", edgecolor="#57606f", labelcolor="#f1f2f6", fontsize=7.5)

    plot_path = CURRENT_DIR / "tracking_topdown_2d.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Updated Kinematic Visual Servoing visualization saved to {plot_path}")


if __name__ == "__main__":
    run_simulation_and_plot()
