"""
Spatial 3D Kinematics and SO(3) Attitude Decoupling Utilities.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception.

Implements exact 3D ray back-projection from 2D normalized camera pixel coordinates
to an inertial horizon-stabilized line-of-sight vector, canceling out:
1. Mechanical camera mount up-tilt (+15 deg)
2. Body roll (drone_roll)
3. Body pitch (drone_pitch)
"""

from typing import Tuple
import numpy as np


def unproject_camera_to_horizon(
    err_x: float,
    err_y: float,
    drone_pitch: float,
    drone_roll: float,
    camera_uptilt_rad: float = np.deg2rad(15.0),
    half_hfov: float = np.tan(np.deg2rad(30.0)),
    half_vfov: float = np.tan(np.deg2rad(22.5)),
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
        half_hfov: tan(HFOV / 2) (default: tan(30 deg))
        half_vfov: tan(VFOV / 2) (default: tan(22.5 deg))
        
    Returns:
        err_x_horizon: Normalized [-1.0, 1.0] azimuth error in horizon plane
        err_y_horizon: Normalized [-1.0, 1.0] elevation error in horizon plane
        azimuth_rad: Sightline azimuth relative to drone heading in horizon plane (rad)
        elevation_rad: Sightline elevation relative to horizon plane (rad, + down)
    """
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


def compute_ego_motion_compensation(
    cx: float,
    cy: float,
    w: float,
    h: float,
    delta_pitch_rad: float,
    delta_roll_rad: float,
    delta_yaw_rad: float,
    vx_body: float = 0.0,
    vy_body: float = 0.0,
    vz_body: float = 0.0,
    dt: float = 1.0 / 30.0,
    estimated_distance_m: float = 6.0,
    img_width: float = 640.0,
    img_height: float = 480.0,
    camera_uptilt_rad: float = np.deg2rad(15.0),
    half_hfov: float = np.tan(np.deg2rad(30.0)),
    half_vfov: float = np.tan(np.deg2rad(22.5)),
) -> Tuple[float, float, float, float]:
    """
    Computes image-plane shift [delta_cx, delta_cy, delta_w, delta_h] in pixels
    induced by chaser drone ego-motion (attitude increments + translational velocity) over dt.

    Features:
    1. Full 3D back-projection through mount uptilt (+15 deg) to Body frame.
    2. Subtracts translation displacement (vx_body, vy_body, vz_body) * dt.
    3. Inverts body rotation increments (delta_roll, delta_pitch, delta_yaw).
    4. Projects back through mount uptilt to camera sensor.
    5. Returns exact differential pixel shifts and box scale expansion/contraction.
    """
    # 1. Normalized image error [-1, 1]
    err_x = (cx - img_width / 2.0) / (img_width / 2.0)
    err_y = (cy - img_height / 2.0) / (img_height / 2.0)

    # 2. Camera optical ray [Right, Down, Forward]
    rx_cam = err_x * half_hfov
    ry_cam = err_y * half_vfov

    # Scale to 3D metric position in camera frame
    dist = max(0.5, estimated_distance_m)
    P_cam_x = rx_cam * dist
    P_cam_y = ry_cam * dist
    P_cam_z = dist

    # 3. Transform to Body frame [Forward, Right, Down] with mount uptilt (+15 deg)
    cu, su = np.cos(camera_uptilt_rad), np.sin(camera_uptilt_rad)
    X_body = cu * P_cam_z + su * P_cam_y   # Body Forward
    Y_body = P_cam_x                       # Body Right
    Z_body = cu * P_cam_y - su * P_cam_z   # Body Down

    # 4. Translation step (relative motion: target appears to move opposite drone translation)
    X_body -= vx_body * dt
    Y_body -= vy_body * dt
    Z_body -= vz_body * dt

    # 5. Rotation step: apply incremental body attitude changes
    # Yaw rotation around Body Z (heading clockwise: + yaw)
    if abs(delta_yaw_rad) > 1e-6:
        cyaw, syaw = np.cos(delta_yaw_rad), np.sin(delta_yaw_rad)
        X_body, Y_body = cyaw * X_body + syaw * Y_body, -syaw * X_body + cyaw * Y_body

    # Pitch rotation around Body Y (nose-down is pitch > 0)
    if abs(delta_pitch_rad) > 1e-6:
        cp, sp = np.cos(delta_pitch_rad), np.sin(delta_pitch_rad)
        X_body, Z_body = cp * X_body + sp * Z_body, -sp * X_body + cp * Z_body

    # Roll rotation around Body X (right-wing-down is roll > 0)
    if abs(delta_roll_rad) > 1e-6:
        cr, sr = np.cos(delta_roll_rad), np.sin(delta_roll_rad)
        Y_body, Z_body = cr * Y_body + sr * Z_body, -sr * Y_body + cr * Z_body

    # 6. Transform back from Body to Camera frame
    P_cam_x_new = Y_body
    P_cam_y_new = cu * Z_body + su * X_body
    P_cam_z_new = cu * X_body - su * Z_body

    if P_cam_z_new <= 0.1:  # Target behind camera
        return 0.0, 0.0, 0.0, 0.0

    # 7. Project back to image plane
    new_err_x = (P_cam_x_new / P_cam_z_new) / half_hfov
    new_err_y = (P_cam_y_new / P_cam_z_new) / half_vfov

    new_cx = (img_width / 2.0) + new_err_x * (img_width / 2.0)
    new_cy = (img_height / 2.0) + new_err_y * (img_height / 2.0)

    delta_cx = float(new_cx - cx)
    delta_cy = float(new_cy - cy)

    # 8. Scale change (apparent box size dilation/contraction)
    scale_factor = float(dist / P_cam_z_new)
    delta_w = float(w * (scale_factor - 1.0))
    delta_h = float(h * (scale_factor - 1.0))

    return delta_cx, delta_cy, delta_w, delta_h

