"""
Spatial 3D Kinematics and SO(3) Attitude Decoupling Utilities.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception.

Implements exact 3D ray back-projection from 2D normalized camera pixel coordinates
to an inertial horizon-stabilized line-of-sight vector, canceling out:
1. Mechanical camera mount up-tilt (+15 deg)
2. Body roll (drone_roll)
3. Body pitch (drone_pitch)
"""

from typing import Tuple, Optional
import numpy as np


def unproject_camera_to_horizon(
    err_x: float,
    err_y: float,
    drone_pitch: float,
    drone_roll: float,
    camera_uptilt_rad: float = np.deg2rad(15.0),
    half_hfov: float = np.tan(np.deg2rad(31.0)),
    half_vfov: float = np.tan(np.deg2rad(19.0)),
) -> Tuple[float, float, float, float]:
    """
    3D Ray Back-Projection (Full SO(3) Attitude Decoupling).
    
    Transforms normalized 2D camera image-plane errors [err_x, err_y] in [-1, 1]
    into an inertial horizon-stabilized reference frame by:
      1. Converting normalized pixels to 3D unit ray in camera frame.
      2. Inverting mechanical camera mount up-tilt (+15 deg).
      3. Inverting instantaneous body roll (drone_roll).
      4. Inverting instantaneous body pitch (drone_pitch).
      
    Args:
        err_x: Normalized horizontal pixel error [-1.0, 1.0] (+ right, - left)
        err_y: Normalized vertical pixel error [-1.0, 1.0] (+ down, - up)
        drone_pitch: Drone pitch angle in radians (+ is nose-down)
        drone_roll: Drone roll angle in radians (+ is right-wing-down)
        camera_uptilt_rad: Mechanical mount pitch angle in radians (default: +15 deg)
        half_hfov: tan(HFOV / 2) (default: tan(31.0 deg) for 62 deg HFOV)
        half_vfov: tan(VFOV / 2) (default: tan(19.0 deg) for 38 deg VFOV)
        
    Returns:
        err_x_horizon: Normalized [-1.0, 1.0] azimuth error in horizon plane
        err_y_horizon: Normalized [-1.0, 1.0] elevation error in horizon plane
        azimuth_rad: Sightline azimuth relative to drone heading in horizon plane (rad)
        elevation_rad: Sightline elevation relative to horizon plane (rad, + down)
    """
    # Non-finite (NaN / Inf) sanity barrier: return boresight zeros if corrupted
    if not (np.isfinite(err_x) and np.isfinite(err_y) and np.isfinite(drone_pitch) and np.isfinite(drone_roll)):
        return 0.0, 0.0, 0.0, 0.0

    # 1. Reconstruct unit sightline ray in Camera optical frame [Right, Down, Forward]
    rx_cam = err_x * half_hfov
    ry_cam = err_y * half_vfov
    rz_cam = 1.0

    # 2. Un-tilt camera mount (+15 deg) to Body frame [Forward, Right, Down]
    cu, su = np.cos(camera_uptilt_rad), np.sin(camera_uptilt_rad)
    rx_body = cu * rz_cam + su * ry_cam   # Body Forward
    ry_body = rx_cam                      # Body Right
    rz_body = cu * ry_cam - su * rz_cam   # Body Down

    # 3. Un-roll body (drone_roll around Body Forward axis)
    cr, sr = np.cos(drone_roll), np.sin(drone_roll)
    ry_unrolled =  cr * ry_body - sr * rz_body
    rz_unrolled =  sr * ry_body + cr * rz_body

    # 4. Un-pitch body (drone_pitch around Horizon Right axis: nose-down is pitch > 0)
    cp, sp = np.cos(drone_pitch), np.sin(drone_pitch)
    rx_horizon =  cp * rx_body - sp * rz_unrolled
    rz_horizon =  sp * rx_body + cp * rz_unrolled
    ry_horizon = ry_unrolled

    # 5. Exact Horizon-Stabilized Angles (rad)
    azimuth_rad = float(np.arctan2(ry_horizon, rx_horizon))
    elevation_rad = float(np.arctan2(rz_horizon, rx_horizon))

    # Guard against target behind camera (> 90 deg off boresight)
    if rx_horizon > 0.05:
        err_x_horizon = float(np.clip((ry_horizon / rx_horizon) / half_hfov, -1.0, 1.0))
        err_y_horizon = float(np.clip((rz_horizon / rx_horizon) / half_vfov, -1.0, 1.0))
    else:
        err_x_horizon = 1.0 if ry_horizon >= 0.0 else -1.0
        err_y_horizon = 1.0 if rz_horizon >= 0.0 else -1.0

    return err_x_horizon, err_y_horizon, azimuth_rad, elevation_rad


def compute_ego_motion_compensation(*args, **kwargs) -> Tuple[float, float, float, float]:
    """
    Deprecated / Decommissioned: Predictive Kalman ego-motion shift compensation has been removed.
    Always returns zero shift (0.0, 0.0, 0.0, 0.0).
    """
    return 0.0, 0.0, 0.0, 0.0


def apply_command_envelope_and_slew(
    target_cmd: np.ndarray,
    prev_cmd: np.ndarray,
    dt: float,
    max_rev_vel: float = 0.5,
    max_fwd_vel: float = 15.0,
    max_lat_vel: float = 4.0,
    max_vel_up: float = 3.0,
    max_vel_down: float = 2.0,
    max_yaw_rate: float = 120.0,
    max_accel_xy: float = 6.0,
    max_decel_xy: float = 2.45,
    max_lat_accel: float = 6.0,
    max_accel_z: float = 3.0,
    max_accel_yaw: float = 240.0,
    current_alt_m: Optional[float] = None,
) -> np.ndarray:
    """
    Standardized Quadrotor Command Envelope and Slew-Rate Limiter.
    Enforces continuous physical velocity limits, ground proximity governors,
    and slew-rate acceleration bounds identically across training and deployment.
    """
    # 1. Envelope clipping
    target_vx = float(np.clip(target_cmd[0], -max_rev_vel, max_fwd_vel))
    target_vy = float(np.clip(target_cmd[1], -max_lat_vel, max_lat_vel))
    target_vz = float(np.clip(target_cmd[2], -max_vel_up, max_vel_down))
    target_yaw = float(np.clip(target_cmd[3], -max_yaw_rate, max_yaw_rate))

    # 2. Ground proximity safety barrier (inhibit descent below 1.4m, climb below 1.1m)
    if current_alt_m is not None and current_alt_m < 1.4:
        if target_vz > 0.0:
            target_vz = float(target_vz * max(0.0, (current_alt_m - 0.9) / 0.5))
        if current_alt_m < 1.1:
            climb_cmd = -float(np.clip((1.1 - current_alt_m) * 3.0, 0.0, max_vel_up))
            target_vz = min(target_vz, climb_cmd)

    # 3. Slew-rate limiting across all 4 flight axes
    # Axis 0: Forward surge acceleration & deceleration
    max_step_accel_x = max_accel_xy * dt
    max_step_decel_x = max_decel_xy * dt
    if target_vx < prev_cmd[0]:
        cmd_vx = max(target_vx, prev_cmd[0] - max_step_decel_x)
    else:
        cmd_vx = min(target_vx, prev_cmd[0] + max_step_accel_x)

    # Axis 1: Lateral sway acceleration
    max_step_lat = max_lat_accel * dt
    cmd_vy = float(np.clip(target_vy, prev_cmd[1] - max_step_lat, prev_cmd[1] + max_step_lat))

    # Axis 2: Vertical heave acceleration
    max_step_z = max_accel_z * dt
    cmd_vz = float(np.clip(target_vz, prev_cmd[2] - max_step_z, prev_cmd[2] + max_step_z))

    # Axis 3: Yaw rate angular acceleration
    max_step_yaw = max_accel_yaw * dt
    cmd_yaw = float(np.clip(target_yaw, prev_cmd[3] - max_step_yaw, prev_cmd[3] + max_step_yaw))

    phys_cmd = np.array([cmd_vx, cmd_vy, cmd_vz, cmd_yaw], dtype=np.float64)
    if not np.all(np.isfinite(phys_cmd)):
        return prev_cmd.copy()
    return phys_cmd


