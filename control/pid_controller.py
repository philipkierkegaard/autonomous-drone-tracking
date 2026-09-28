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
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Ensure control_model directory is on path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

try:
    from control.simulation import FastPixhawkQuadSim
except ImportError:
    from simulation import FastPixhawkQuadSim


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
        hfov_deg: float = 60.0,
        vfov_deg: float = 45.0,
        desired_bbox_size: float = 35.0,     # Desired target bounding box size (pixels, 640x480 frame)
        desired_standoff_dist: float = 6.0,  # Backward-compatible distance in meters (when use_bbox_size=False)
        use_bbox_size: bool = True,          # If True, regulates bounding-box size instead of metric distance
        # 3D/4D PID Gains: [Forward (X), Lateral (Y), Vertical (Z), Yaw (Psi)]
        kp: np.ndarray = None,
        ki: np.ndarray = None,
        kd: np.ndarray = None,
        # Backward-compatible individual gain parameters
        kp_yaw: float = None,
        ki_yaw: float = None,
        kd_yaw: float = None,
        kp_z: float = None,
        ki_z: float = None,
        kd_z: float = None,
        # Output Saturation Limits per axis: [Forward, Lateral, Vertical, Yaw]
        min_limits: np.ndarray = None,       # Default: [-2.0 m/s fwd (reverse brake), -4.0 m/s lat, -3.5 m/s climb, -120 deg/s yaw]
        max_limits: np.ndarray = None,       # Default: [18.0 m/s fwd, +4.0 m/s lat, +2.5 m/s desc, +120 deg/s yaw]
        # Integrator Anti-Windup Clamping Bounds
        int_limits: np.ndarray = None,       # Forward and Lateral integrators are 0.0 (kinematic surge + PD sway)
        # Acceleration & Slew Rate Limits (Calibrated for ~2 kg quadrotor flight envelope)
        max_accel: float = 6.5,              # m/s^2 forward acceleration (calibrated ~33.5 deg tilt ramp for 2 kg quad)
        max_decel: float = 2.8,              # m/s^2 calibrated braking (tightens standoff tracking without tilting camera out of FOV)
        max_accel_z: float = 3.0,            # m/s^2 vertical climb/descent acceleration limit
        max_accel_yaw: float = 180.0,        # deg/s^2 yaw angular acceleration limit
        filter_tau: float = 0.06,            # Derivative low-pass filter time constant (seconds)
        max_yawspeed_deg: float = None,
        # 4-DOF Lateral Strafe (Roll Tilt) Control
        enable_lateral_strafe: bool = True,
        kp_lat: float = 2.5,                 # Lateral proportional gain (m/s per unit error_x)
        kd_lat: float = 0.35,                # Lateral derivative damping (counteracts roll-induced visual swings)
        max_lat_vel: float = 4.0,            # Maximum lateral velocity in m/s (enables pacing crossing targets up to 4 m/s)
        max_lat_accel: float = 5.0,          # Maximum lateral acceleration in m/s^2 for agile roll initiation
    ):
        self.camera_uptilt = np.deg2rad(camera_uptilt_deg)
        self.half_hfov = np.tan(np.deg2rad(hfov_deg / 2.0))
        self.half_vfov = np.tan(np.deg2rad(vfov_deg / 2.0))
        self.desired_bbox_size = desired_bbox_size
        self.standoff_dist = desired_standoff_dist
        self.use_bbox_size = use_bbox_size

        self.lost_time = 0.0

        # 4-DOF Gains: [Forward (X), Lateral (Y), Vertical (Z), Yaw (Psi)]
        lat_kp = float(kp_lat) if enable_lateral_strafe else 0.0
        lat_kd = float(kd_lat) if enable_lateral_strafe else 0.0
        # Axis 0 uses kinematic deceleration curve; integrator is 0.0 to prevent phantom windup.
        default_kp = np.array([1.2, lat_kp, 5.0, 180.0], dtype=np.float64)
        default_ki = np.array([0.0, 0.0, 0.5, 5.0], dtype=np.float64)
        default_kd = np.array([0.15, lat_kd, 1.0, 20.0], dtype=np.float64)

        if kp is not None:
            self.kp = np.array(kp if len(kp) == 4 else [kp[0], lat_kp, kp[1], kp[2]], dtype=np.float64)
        else:
            self.kp = default_kp.copy()

        if ki is not None:
            self.ki = np.array(ki if len(ki) == 4 else [ki[0], 0.0, ki[1], ki[2]], dtype=np.float64)
        else:
            self.ki = default_ki.copy()

        if kd is not None:
            self.kd = np.array(kd if len(kd) == 4 else [kd[0], lat_kd, kd[1], kd[2]], dtype=np.float64)
        else:
            self.kd = default_kd.copy()

        # Override individual gains if provided
        if kp_yaw is not None: self.kp[3] = kp_yaw
        if ki_yaw is not None: self.ki[3] = ki_yaw
        if kd_yaw is not None: self.kd[3] = kd_yaw
        if kp_z is not None:   self.kp[2] = kp_z
        if ki_z is not None:   self.ki[2] = ki_z
        if kd_z is not None:   self.kd[2] = kd_z

        # Output Saturation Limits per axis: [Forward, Lateral, Vertical, Yaw]
        yaw_limit = float(max_yawspeed_deg) if max_yawspeed_deg is not None else 120.0
        lat_limit = float(max_lat_vel)

        if min_limits is not None:
            if len(min_limits) == 4:
                self.min_limits = np.array(min_limits, dtype=np.float64)
            elif len(min_limits) == 3:
                # [vx_min, vz_min, yaw_min] -> allow active reverse braking if vx_min == 0.0
                vx_min = float(min_limits[0]) if min_limits[0] < 0.0 else -2.0
                self.min_limits = np.array([vx_min, -lat_limit, min_limits[1], min_limits[2]], dtype=np.float64)
            else:
                self.min_limits = np.array([-2.0, -lat_limit, -3.5, -yaw_limit], dtype=np.float64)
        else:
            self.min_limits = np.array([-2.0, -lat_limit, -3.5, -yaw_limit], dtype=np.float64)

        if max_limits is not None:
            if len(max_limits) == 4:
                self.max_limits = np.array(max_limits, dtype=np.float64)
            elif len(max_limits) == 3:
                self.max_limits = np.array([max_limits[0], lat_limit, max_limits[1], max_limits[2]], dtype=np.float64)
            else:
                self.max_limits = np.array([18.0, lat_limit, 2.5, yaw_limit], dtype=np.float64)
        else:
            self.max_limits = np.array([18.0, lat_limit, 2.5, yaw_limit], dtype=np.float64)

        if int_limits is not None:
            if len(int_limits) == 4:
                self.int_limits = np.array(int_limits, dtype=np.float64)
            elif len(int_limits) == 3:
                self.int_limits = np.array([0.0, 0.0, int_limits[1], int_limits[2]], dtype=np.float64)
            else:
                self.int_limits = np.array([0.0, 0.0, 1.2, 25.0], dtype=np.float64)
        else:
            self.int_limits = np.array([0.0, 0.0, 1.2, 25.0], dtype=np.float64)

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
        self.initialized = False
        self.lost_time = 0.0

    def compute_cmd(
        self,
        telemetry: dict,
        drone_pitch: float = 0.0,
        drone_roll: float = 0.0,
        dt: float = 0.02
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
        status = telemetry.get("status", "SEARCHING")
        # Graceful coasting and multi-phase search governor
        if status == "SEARCHING":
            self.lost_time += dt
            if self.lost_time < 1.0:
                # Phase 1: Coast forward on last known heading with gentle deceleration
                cmd_vx = max(0.0, self.prev_cmd_vx - 1.5 * dt)
                return np.array([cmd_vx, 0.0, 0.0, 0.0], dtype=np.float64)
            elif self.lost_time < 2.0:
                # Phase 2: Gentle hover hold before spinning
                return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            else:
                # Phase 3: Controlled 15 deg/s search scan
                return np.array([0.0, 0.0, 0.0, 15.0], dtype=np.float64)

        # Active lock: reset lost timer
        self.lost_time = 0.0

        err_x = telemetry.get("error_x", 0.0)
        err_y = telemetry.get("error_y", 0.0)

        # ----------------------------------------------------------------------
        # 1. 3D RAY BACK-PROJECTION (FULL SO(3) DECOUPLING)
        # ----------------------------------------------------------------------
        # Reconstruct unit sightline ray in Camera optical frame [Right, Down, Forward]
        rx_cam = err_x * self.half_hfov
        ry_cam = err_y * self.half_vfov
        rz_cam = 1.0

        # Un-tilt camera mount (+15 deg) to Body frame [Forward, Right, Down]
        cu, su = np.cos(self.camera_uptilt), np.sin(self.camera_uptilt)
        rx_body = cu * rz_cam + su * ry_cam   # Body Forward
        ry_body = rx_cam                      # Body Right
        rz_body = cu * ry_cam - su * rz_cam   # Body Down

        # Un-roll body (drone_roll around Body Forward axis)
        cr, sr = np.cos(drone_roll), np.sin(drone_roll)
        ry_unrolled =  cr * ry_body - sr * rz_body
        rz_unrolled =  sr * ry_body + cr * rz_body

        # Un-pitch body (drone_pitch around Horizon Right axis: nose-down is pitch > 0)
        cp, sp = np.cos(drone_pitch), np.sin(drone_pitch)
        rx_horizon =  cp * rx_body - sp * rz_unrolled
        rz_horizon =  sp * rx_body + cp * rz_unrolled
        ry_horizon = ry_unrolled

        # Exact Horizon-Stabilized Error Angles (rad)
        azimuth_horizon = float(np.arctan2(ry_horizon, rx_horizon))      # + right, - left
        elevation_horizon = float(np.arctan2(rz_horizon, rx_horizon))    # + down, - up

        # Guard against +/- 90 deg tangent singularity when target is abeam or behind
        if rx_horizon > 0.05:
            err_x_horizon = float(np.clip((ry_horizon / rx_horizon) / self.half_hfov, -1.0, 1.0))
        else:
            err_x_horizon = 1.0 if ry_horizon >= 0.0 else -1.0

        # 2. Scale / Range Error (Bounding Box Pixels vs Metric Meters)
        if self.use_bbox_size:
            if "error_range" in telemetry:
                # Direct normalized scale error [-1.0 (too close), +1.0 (far)] from track_pipeline.py
                range_err = float(telemetry["error_range"])
            else:
                current_sz = telemetry.get("target_size", 0.0)
                if current_sz <= 0.0:
                    bw = telemetry.get("bbox_w", 0.0)
                    bh = telemetry.get("bbox_h", 0.0)
                    current_sz = max(bw, bh)

                if current_sz > 0.0:
                    range_err = (self.desired_bbox_size - current_sz) / max(1.0, self.desired_bbox_size)
                else:
                    range_err = 0.0
            range_term = float(np.clip(range_err, -1.0, 1.0))
        else:
            if "distance" in telemetry:
                dist = telemetry["distance"]
                dist_err = dist - self.standoff_dist
            else:
                dist_err = telemetry.get("error_range", 0.0) * 10.0
            range_term = dist_err

        # Optional target forward velocity feedforward (e.g. from Kalman filter)
        target_vx_ff = float(telemetry.get("target_vx", telemetry.get("target_vel_x", 0.0)))

        # 3. 4D Error Vector: [Range Error, Lateral Error (norm), Elevation Horizon (rad), Azimuth Horizon (rad)]
        error_vec = np.array([range_term, err_x_horizon, elevation_horizon, azimuth_horizon], dtype=np.float64)

        # 4. Apply 4-DOF control update (Surge vx, Sway vy, Heave vz, Yaw yawspeed)
        return self.update(
            error_vec,
            drone_pitch=drone_pitch,
            dt=dt,
            horiz_err_norm=abs(err_x),
            decoupled=True,
            target_vx_ff=target_vx_ff
        )

    def update(
        self,
        error_vec: np.ndarray,
        drone_pitch: float = 0.0,
        dt: float = 0.02,
        horiz_err_norm: float = 0.0,
        decoupled: bool = True,
        target_vx_ff: float = 0.0
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
        # Exclude Axis 0 (kinematic surge) and Axis 1 (PD sway) from integrator windup
        tentative_out = p_term + self.ki * self.integral + d_term
        saturated_high = (tentative_out >= self.max_limits) & (e > 0)
        saturated_low  = (tentative_out <= self.min_limits) & (e < 0)
        unclamped_mask = ~(saturated_high | saturated_low)

        self.integral[unclamped_mask] += e[unclamped_mask] * dt
        self.integral[0] = 0.0  # Zero phantom windup on Axis 0
        self.integral[1] = 0.0  # Zero phantom windup on Axis 1
        self.integral = np.clip(self.integral, -self.int_limits, self.int_limits)
        i_term = self.ki * self.integral

        # ----------------------------------------------------------------------
        # 5. Compute Saturated Output Vector
        # ----------------------------------------------------------------------
        raw_output = p_term + i_term + d_term

        # Kinematic approach & recession curve on Axis 0 (Forward Speed) with active reverse braking
        if self.use_bbox_size:
            scale_err = e[0]  # normalized scale error: (s* - s) / s*
            denom = max(0.05, 1.0 - scale_err) if scale_err < 1.0 else 0.05
            ratio = scale_err / denom
            approx_dist_err = self.standoff_dist * ratio

            if approx_dist_err >= 0.0:
                v_kinematic = min(self.max_limits[0], np.sqrt(2.0 * self.max_decel * approx_dist_err))
            else:
                # Active reverse braking when target is closer than standoff cushion
                v_kinematic = -min(abs(self.min_limits[0]), np.sqrt(2.0 * self.max_decel * abs(approx_dist_err)))
        else:
            dist_err = e[0]
            if dist_err >= 0.0:
                v_kinematic = min(self.max_limits[0], np.sqrt(2.0 * self.max_decel * dist_err))
            else:
                v_kinematic = -min(abs(self.min_limits[0]), np.sqrt(2.0 * self.max_decel * abs(dist_err)))

        # Blend kinematic velocity with derivative damping and feedforward
        raw_output[0] = v_kinematic + d_term[0] + target_vx_ff

        # Centering damping: ease forward acceleration gently during sharp turns without killing pursuit momentum
        if raw_output[0] > 0.0:
            centering_factor = max(0.65, 1.0 - horiz_err_norm * 0.35)
            raw_output[0] *= centering_factor

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
        self.prev_cmd = output_vec.copy()
        self.prev_cmd_vx = output_vec[0]
        self.prev_cmd_vy = output_vec[1]
        return output_vec

    @staticmethod
    def to_mavsdk(cmd: np.ndarray) -> tuple:
        """
        Converts command vector into the 4 arguments required by
        MAVSDK VelocityBodyYawspeed(forward_m_s, right_m_s, down_m_s, yawspeed_deg_s).
        Accepts both 3D [v_fwd, v_down, yawspeed] and 4D [v_fwd, v_right, v_down, yawspeed].
        """
        if len(cmd) == 4:
            return float(cmd[0]), float(cmd[1]), float(cmd[2]), float(cmd[3])
        elif len(cmd) == 3:
            return float(cmd[0]), 0.0, float(cmd[1]), float(cmd[2])
        raise ValueError(f"Expected 3D or 4D command vector, got length {len(cmd)}")


# Aliases for backward compatibility with existing simulation scripts
Vector3DPIDController = KinematicVisualServoController
DroneVisualPIDController = KinematicVisualServoController


# ==============================================================================
# 2. Dynamic Closed-Loop Simulation Test & 2D Top-Down Visualization
# ==============================================================================

def run_simulation_and_plot():
    """
    Simulates 20 seconds of autonomous closed-loop pursuit using KinematicVisualServoController.
    Uses pure bounding-box pixel regulation (desired_bbox_size = 35 px on 640x480).
    """
    dt = 0.02
    total_time = 20.0
    num_steps = int(total_time / dt)

    sim = FastPixhawkQuadSim(dt=dt)
    controller = KinematicVisualServoController(
        camera_uptilt_deg=15.0,
        hfov_deg=60.0,
        vfov_deg=45.0,
        desired_bbox_size=35.0,  # 35px corresponds to ~6m standoff on 640x480
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
            hfov_deg=60.0,
            vfov_deg=45.0,
            target_w_m=0.35,
            target_h_m=0.20,
            img_w=640,
            img_h=480,
            desired_target_size=35.0
        )

        # 2. Compute 3D Control Vector using bounding box pixel errors
        cmd_3d = controller.compute_cmd(telemetry, drone_pitch=sim.pitch, dt=dt)

        # 3. Advance drone simulation
        sim.step(cmd_3d)

        # Log states
        times.append(t)
        chaser_pos_history.append(sim.pos.copy())
        target_pos_history.append(target_world.copy())
        chaser_yaw_history.append(sim.yaw)
        dist_history.append(telemetry["distance"])
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
    half_hfov = np.deg2rad(60.0 / 2.0)

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
