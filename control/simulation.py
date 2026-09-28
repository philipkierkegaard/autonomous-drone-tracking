import numpy as np
from collections import deque

class FastPixhawkQuadSim:
    def __init__(
        self,
        dt=0.02,
        max_vel_xy=18.0,
        max_accel_xy=6.5,
        max_vel_up=4.0,
        max_vel_down=2.5,
        k_drag_lin=0.055,
        k_drag_quad=0.008,
    ):  # 50 Hz integration
        self.dt = dt
        
        # Closed-loop tracking time constants (PX4 velocity & yaw loops)
        self.tau_v_horiz = 0.32   # Horizontal velocity response (smooth progressive ramp)
        self.tau_v_vert  = 0.22   # Vertical velocity response
        self.tau_yaw     = 0.15   # Yaw rate response
        
        # Pipeline delay: 60 ms (Camera capture + YOLO + UART MAVLink)
        self.delay_steps = max(1, int(round(0.06 / dt)))
        
        # Kinodynamic limits (Calibrated for ~2 kg quadrotor: ~33.5 deg tilt limit, safe within <37.5 deg camera FOV boundary)
        self.max_accel_xy   = max_accel_xy      # m/s^2 (0.66 g, ~33.5 deg tilt)
        self.max_accel_up   = max_vel_up        # m/s^2 (climb accel limit)
        self.max_accel_down = max_vel_down      # m/s^2 (descent accel limit)
        self.max_yaw_rate   = np.deg2rad(120.0) # 120 deg/s
        self.max_yaw_accel  = np.deg2rad(120.0) # 120 deg/s^2 (smooth yaw ramping)
        
        # Velocity hard limits (Calibrated top speed 18 m/s ~ 65 km/h for 2 kg platform)
        self.max_vel_xy     = max_vel_xy        # m/s (MPC_XY_VEL_MAX)
        self.max_vel_up     = max_vel_up        # m/s (MPC_Z_VEL_MAX_UP)
        self.max_vel_down   = max_vel_down      # m/s (MPC_Z_VEL_MAX_DN)

        # Aerodynamic drag model (Rotor inflow/blade flapping drag + fuselage parasite form drag)
        # Calibrated so steady cruising at 15 m/s produces ~15 deg nose-down tilt to match +15 deg mount
        self.k_drag_lin  = k_drag_lin   # Linear rotor drag (s^-1)
        self.k_drag_quad = k_drag_quad  # Quadratic fuselage drag (m^-1)

        # Camera mounting geometry (15 deg mechanical up-tilt relative to drone body)
        self.camera_uptilt  = np.deg2rad(15.0)  # 15 deg up-tilt
        self.cam_pitch      = -self.camera_uptilt

        self.reset()

    def reset(self, initial_pos=None, initial_yaw=0.0):
        self.pos = np.array(initial_pos, dtype=np.float64) if initial_pos is not None else np.zeros(3)
        self.vel = np.zeros(3)
        self.yaw = float(initial_yaw)
        self.yaw_rate = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.cam_pitch = -self.camera_uptilt
        
        # Delay queue initialized to exact delay steps
        self.cmd_buffer = deque([np.zeros(4) for _ in range(self.delay_steps)], 
                                maxlen=self.delay_steps + 1)
        return self._get_obs()

    def step(self, cmd_body_vel_yawspeed):
        """
        Action: [cmd_vx, cmd_vy, cmd_vz, cmd_yaw_rate_deg]
        Matches MAVSDK VelocityBodyYawspeed(forward_m_s, right_m_s, down_m_s, yawspeed_deg_s)
        """
        # 1. Transport & perception delay (accepts both 3D [vx, vz, yawspeed] and 4D [vx, vy, vz, yawspeed])
        raw_cmd = np.array(cmd_body_vel_yawspeed, dtype=np.float64)
        if len(raw_cmd) == 3:
            # Expand 3D vector [v_forward, v_vertical, yawspeed] to 4D [vx, vy=0, vz, yawspeed]
            formatted_cmd = np.array([raw_cmd[0], 0.0, raw_cmd[1], raw_cmd[2]], dtype=np.float64)
        else:
            formatted_cmd = raw_cmd

        self.cmd_buffer.append(formatted_cmd)
        delayed_cmd = self.cmd_buffer.popleft()

        # 2. Setpoint clamping against autopilot limits
        cmd_body_vel = delayed_cmd[:3].copy()
        cmd_body_vel[0] = np.clip(cmd_body_vel[0], -self.max_vel_xy, self.max_vel_xy)
        cmd_body_vel[1] = np.clip(cmd_body_vel[1], -self.max_vel_xy, self.max_vel_xy)
        cmd_body_vel[2] = np.clip(cmd_body_vel[2], -self.max_vel_up, self.max_vel_down) # NED
        cmd_yaw_rate_rad = np.clip(np.deg2rad(delayed_cmd[3]), -self.max_yaw_rate, self.max_yaw_rate)

        # 3. Transform body velocity to world NED using current heading
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        R_body_to_world = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ])
        cmd_world_vel = R_body_to_world @ cmd_body_vel

        # 4. First-order velocity tracking lag with acceleration limits
        accel = np.zeros(3)
        accel[0] = (cmd_world_vel[0] - self.vel[0]) / self.tau_v_horiz
        accel[1] = (cmd_world_vel[1] - self.vel[1]) / self.tau_v_horiz
        accel[2] = (cmd_world_vel[2] - self.vel[2]) / self.tau_v_vert

        # Clip horizontal acceleration magnitude
        horiz_accel_mag = np.linalg.norm(accel[:2])
        if horiz_accel_mag > self.max_accel_xy:
            accel[:2] = (accel[:2] / horiz_accel_mag) * self.max_accel_xy

        # Clip vertical acceleration (NED: Negative is Climb, Positive is Descend)
        accel[2] = np.clip(accel[2], -self.max_accel_up, self.max_accel_down)

        # 5. Approximate realistic body tilt (pitch & roll) from specific force:
        # Rotors must generate thrust for both inertial acceleration (dv/dt)
        # and aerodynamic drag compensation (rotor inflow drag + fuselage parasite form drag).
        a_fwd = c * accel[0] + s * accel[1]
        a_right = -s * accel[0] + c * accel[1]

        # Horizontal velocities in drone body frame
        v_fwd = c * self.vel[0] + s * self.vel[1]
        v_right = -s * self.vel[0] + c * self.vel[1]
        v_horiz_speed = np.hypot(v_fwd, v_right)

        # Aerodynamic drag deceleration (Rotor blade flapping + Fuselage form drag)
        drag_coeff = self.k_drag_lin + self.k_drag_quad * v_horiz_speed
        drag_fwd = drag_coeff * v_fwd
        drag_right = drag_coeff * v_right

        # Total specific acceleration demanded from quadrotor rotors
        total_a_fwd = a_fwd + drag_fwd
        total_a_right = a_right + drag_right

        self.pitch = np.arctan2(total_a_fwd, 9.81)
        self.roll = np.arctan2(-total_a_right, 9.81)

        # Camera optical pitch relative to horizon (accounts for 15 deg mount up-tilt)
        # When drone pitches nose-down (self.pitch > 0), camera pitch becomes level
        self.cam_pitch = self.pitch - self.camera_uptilt

        # 6. Yaw rate tracking response
        yaw_accel = (cmd_yaw_rate_rad - self.yaw_rate) / self.tau_yaw
        yaw_accel = np.clip(yaw_accel, -self.max_yaw_accel, self.max_yaw_accel)

        # 7. Symplectic Euler numerical integration
        self.vel += accel * self.dt
        
        # Enforce velocity limits
        horiz_vel_mag = np.linalg.norm(self.vel[:2])
        if horiz_vel_mag > self.max_vel_xy:
            self.vel[:2] = (self.vel[:2] / horiz_vel_mag) * self.max_vel_xy
        self.vel[2] = np.clip(self.vel[2], -self.max_vel_up, self.max_vel_down)

        self.pos += self.vel * self.dt
        self.yaw_rate = np.clip(self.yaw_rate + yaw_accel * self.dt, -self.max_yaw_rate, self.max_yaw_rate)
        self.yaw = (self.yaw + self.yaw_rate * self.dt + np.pi) % (2.0 * np.pi) - np.pi

        return self._get_obs()
        

    def project_to_camera(
        self,
        target_pos_world,
        hfov_deg: float = 60.0,
        vfov_deg: float = 45.0,
        target_w_m: float = 0.35,
        target_h_m: float = 0.20,
        img_w: int = 640,
        img_h: int = 480,
        return_bbox: bool = False
    ):
        """
        Projects a 3D target in world NED coordinates onto the drone's tilted camera sensor.
        
        Args:
            target_pos_world: [x, y, z] target coordinates in world NED (meters)
            hfov_deg: Horizontal field of view in degrees (default: 60.0)
            vfov_deg: Vertical field of view in degrees (default: 45.0)
            target_w_m: Target physical wingspan/width in meters (default: 0.35m)
            target_h_m: Target physical height in meters (default: 0.20m)
            img_w: Camera sensor resolution width in pixels (default: 640)
            img_h: Camera sensor resolution height in pixels (default: 480)
            return_bbox: If True, returns (err_x, err_y, in_view, dist, bbox_w, bbox_h)
                         If False, returns (err_x, err_y, in_view, dist) for backward compatibility
        
        Returns:
            error_x: [-1.0 (left), +1.0 (right)]
            error_y: [-1.0 (up), +1.0 (down)]
            in_view: bool
            distance: float (meters)
            bbox_w: float (pixels, only if return_bbox=True)
            bbox_h: float (pixels, only if return_bbox=True)
        """
        # Relative vector in world NED
        rel_world = np.array(target_pos_world) - self.pos
        dist = np.linalg.norm(rel_world)
        if dist < 1e-3:
            if return_bbox:
                return 0.0, 0.0, True, 0.0, float(img_w), float(img_h)
            return 0.0, 0.0, True, 0.0

        # Rotate to body frame: Yaw -> Pitch -> Roll
        # NED: X is North/Forward, Y is East/Right, Z is Down
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        R_yaw = np.array([[cy, sy, 0.0], [-sy, cy, 0.0], [0.0, 0.0, 1.0]])
        rel_yaw = R_yaw @ rel_world

        # Pitch: self.pitch > 0 is nose-DOWN (accelerating forward).
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        R_pitch = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        
        # Roll: self.roll > 0 is right wing down.
        cr, sr = np.cos(self.roll), np.sin(self.roll)
        R_roll = np.array([[1.0, 0.0, 0.0], [0.0, cr, sr], [0.0, -sr, cr]])
        
        rel_body = R_roll @ (R_pitch @ rel_yaw)  # [forward, right, down]

        # Camera frame: [right, down, optical_axis_forward]
        cu, su = np.cos(self.camera_uptilt), np.sin(self.camera_uptilt)
        x_cam = rel_body[1]                             # Right
        y_cam = cu * rel_body[2] + su * rel_body[0]     # Down
        z_cam = cu * rel_body[0] - su * rel_body[2]     # Forward optical axis

        if z_cam <= 0.1:
            sign_x = 1.0 if x_cam >= 0.0 else -1.0
            if return_bbox:
                return sign_x, 0.0, False, float(dist), 0.0, 0.0
            return sign_x, 0.0, False, float(dist)  # Behind camera: steer towards direction of target

        # Perspective projection to normalized [-1, 1] errors
        half_hfov = np.tan(np.deg2rad(hfov_deg / 2.0))
        half_vfov = np.tan(np.deg2rad(vfov_deg / 2.0))

        err_x = (x_cam / z_cam) / half_hfov
        err_y = (y_cam / z_cam) / half_vfov

        in_view = (abs(err_x) <= 1.0) and (abs(err_y) <= 1.0)

        # Pinhole perspective projection for bounding box dimensions in pixels
        fx = (img_w / 2.0) / half_hfov
        fy = (img_h / 2.0) / half_vfov
        bbox_w = (target_w_m / z_cam) * fx
        bbox_h = (target_h_m / z_cam) * fy

        if not in_view:
            bbox_w = 0.0
            bbox_h = 0.0

        if return_bbox:
            return float(np.clip(err_x, -1.0, 1.0)), float(np.clip(err_y, -1.0, 1.0)), in_view, float(dist), float(bbox_w), float(bbox_h)
        return float(np.clip(err_x, -1.0, 1.0)), float(np.clip(err_y, -1.0, 1.0)), in_view, float(dist)

    def get_camera_telemetry(
        self,
        target_pos_world,
        hfov_deg: float = 60.0,
        vfov_deg: float = 45.0,
        target_w_m: float = 0.35,
        target_h_m: float = 0.20,
        img_w: int = 640,
        img_h: int = 480,
        desired_target_size: float = 35.0
    ) -> dict:
        """
        Synthesizes visual servoing flight telemetry matching the exact schema of track_pipeline.py.
        Provides normalized errors, bounding box pixel dimensions, and range error.
        """
        err_x, err_y, in_view, dist, bw, bh = self.project_to_camera(
            target_pos_world,
            hfov_deg=hfov_deg,
            vfov_deg=vfov_deg,
            target_w_m=target_w_m,
            target_h_m=target_h_m,
            img_w=img_w,
            img_h=img_h,
            return_bbox=True
        )
        current_sz = max(bw, bh)
        range_err = (desired_target_size - current_sz) / max(1.0, desired_target_size)

        # Bounding box coordinates on sensor [x1, y1, x2, y2]
        cx = (err_x + 1.0) * (img_w / 2.0)
        cy = (err_y + 1.0) * (img_h / 2.0)
        x1 = max(0.0, cx - bw / 2.0)
        y1 = max(0.0, cy - bh / 2.0)
        x2 = min(float(img_w), cx + bw / 2.0)
        y2 = min(float(img_h), cy + bh / 2.0)

        return {
            "status": "LOCKED" if in_view else "COASTING",
            "error_x": err_x,
            "error_y": err_y,
            "error_range": float(np.clip(range_err, -1.0, 1.0)),
            "target_size": float(current_sz),
            "desired_size": float(desired_target_size),
            "bbox_w": float(bw),
            "bbox_h": float(bh),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "distance": float(dist),  # Ground-truth metric distance for benchmarking logs
            "in_view": in_view,
        }

    def _get_obs(self):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        R_world_to_body = np.array([
            [ c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0]
        ])
        body_vel = R_world_to_body @ self.vel
        
        # Returns [x, y, z, vx_body, vy_body, vz_body, sin_yaw, cos_yaw, yaw_rate, pitch, roll, cam_pitch]
        return np.concatenate([
            self.pos,
            body_vel,
            [np.sin(self.yaw), np.cos(self.yaw), self.yaw_rate, self.pitch, self.roll, self.cam_pitch]
        ])


