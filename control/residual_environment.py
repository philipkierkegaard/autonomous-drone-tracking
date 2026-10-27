#!/usr/bin/env python3
"""
Residual Drone Pursuit Environment (Gymnasium).
Combines a high-frequency Kinematic Visual Servoing PID base controller
with a deep recurrent RL residual policy:
    u_total = clip(u_PID + Delta_u_RL)

When Delta_u_RL = 0, the policy acts identically to the tuned kinematic PID
baseline (63.0% 1.0s catch rate on 300-scenario suite).
The RL policy learns non-linear anticipatory leads, cross-axis pitch-tilt compensation,
and aggressive decel-to-turn maneuvers on hyper-evasive targets.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np
from gymnasium import spaces

from control.environment import DronePursuitEnv
from benchmarks.control.benchmark_controller import create_pid_controller


class ResidualDronePursuitEnv(DronePursuitEnv):
    """
    Gymnasium environment for Residual Reinforcement Learning on drone tracking.
    Action space: 4D continuous residual adjustment [-1.0, 1.0]^4.
    Physical scaling:
      - Axis 0 (Surge):  Delta_vx in [-3.0, +3.0] m/s
      - Axis 1 (Sway):   Delta_vy in [-2.0, +2.0] m/s
      - Axis 2 (Heave):  Delta_vz in [-1.0, +1.0] m/s
      - Axis 3 (Yaw):    Delta_yaw_rate in [-30.0, +30.0] deg/s
    """

    def __init__(
        self,
        *args,
        pid_gains_path: str = "control/weights/tuned_pid_gains.json",
        residual_scales: Optional[Sequence[float]] = (3.0, 2.0, 1.0, 30.0),
        reward_mode: str = "theory_fast_catch_compounding_dwell",
        desired_bbox_size: float = 110.0,
        **kwargs
    ):
        super().__init__(*args, reward_mode=reward_mode, **kwargs)

        # Base classical controller
        resolved_gains = Path(pid_gains_path)
        if not resolved_gains.is_absolute():
            resolved_gains = Path(__file__).resolve().parent.parent / resolved_gains

        self.pid = create_pid_controller(
            gains_path=str(resolved_gains),
            standoff_nominal=self.standoff_nominal,
            desired_bbox_size=desired_bbox_size
        )
        self.desired_bbox_size = float(desired_bbox_size)
        self.residual_scales = np.array(
            residual_scales if residual_scales is not None else [3.0, 2.0, 1.0, 30.0],
            dtype=np.float64
        )

        self.last_ctrl_telem: Dict[str, Any] = {
            "status": "SEARCHING",
            "error_x": 0.0,
            "error_y": 0.0,
            "error_range": 0.0,
            "target_size": self.desired_bbox_size,
            "desired_size": self.desired_bbox_size,
            "bbox_w": self.desired_bbox_size,
            "bbox_h": 40.0,
            "w_norm": float(self.desired_bbox_size / self.img_w),
            "scale_mode": "width",
            "in_view": False,
        }
        self.last_pid_cmd = np.zeros(4, dtype=np.float64)
        self.last_residual_cmd = np.zeros(4, dtype=np.float64)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        if hasattr(self.pid, "reset"):
            self.pid.reset()
        self._update_ctrl_telem()
        return obs, info

    def _update_ctrl_telem(self) -> None:
        """Synchronizes controller telemetry with current perception state."""
        if self.use_empirical_perception and self.tracker is not None:
            primary = None
            if self.tracker.primary_track_id is not None:
                primary = next((t for t in self.tracker.trackers if t.id == self.tracker.primary_track_id), None)

            if (
                primary is not None
                and primary.hits >= self.tracker.min_hits_to_confirm
                and primary.time_since_update <= self.coasting_step_limit
            ):
                bbox = primary.get_state()
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                bw = max(1.0, bbox[2] - bbox[0])
                bh = max(1.0, bbox[3] - bbox[1])
                ex = float((cx - self.img_w / 2.0) / (self.img_w / 2.0))
                ey = float((cy - self.img_h / 2.0) / (self.img_h / 2.0))
                range_err = (self.desired_bbox_size - bw) / self.desired_bbox_size

                self.last_ctrl_telem = {
                    "status": "COASTING" if primary.time_since_update > 0 else "LOCKED",
                    "error_x": float(np.clip(ex, -1.0, 1.0)),
                    "error_y": float(np.clip(ey, -1.0, 1.0)),
                    "error_range": float(np.clip(range_err, -1.0, 1.0)),
                    "target_size": float(bw),
                    "desired_size": self.desired_bbox_size,
                    "bbox_w": float(bw),
                    "bbox_h": float(bh),
                    "w_norm": float(bw / self.img_w),
                    "scale_mode": "width",
                    "in_view": (primary.time_since_update == 0),
                }
            else:
                self.last_ctrl_telem = {
                    "status": "SEARCHING",
                    "error_x": 0.0,
                    "error_y": 0.0,
                    "error_range": 0.0,
                    "target_size": 0.0,
                    "desired_size": self.desired_bbox_size,
                    "bbox_w": 0.0,
                    "bbox_h": 0.0,
                    "w_norm": 0.0,
                    "scale_mode": "width",
                    "in_view": False,
                }
        else:
            t_sim = self.step_count * self.dt
            if self.target_oracle is not None:
                target_pos_w, _, _ = self.target_oracle.get_state(t_sim)
                raw_ex_unclipped, raw_ey_unclipped, raw_in_view, dist, raw_bw, raw_bh = self.sim.project_to_camera(
                    target_pos_w,
                    return_bbox=True,
                    return_unclipped=True,
                    img_w=self.img_w,
                    img_h=self.img_h,
                    hfov_deg=self.hfov_deg,
                    vfov_deg=self.vfov_deg
                )
                if raw_in_view:
                    self.last_ctrl_telem = {
                        "status": "LOCKED",
                        "error_x": float(np.clip(raw_ex_unclipped, -1.0, 1.0)),
                        "error_y": float(np.clip(raw_ey_unclipped, -1.0, 1.0)),
                        "error_range": float(np.clip((self.desired_bbox_size - raw_bw) / self.desired_bbox_size, -1.0, 1.0)),
                        "target_size": float(raw_bw),
                        "desired_size": self.desired_bbox_size,
                        "bbox_w": float(raw_bw),
                        "bbox_h": float(raw_bh),
                        "w_norm": float(raw_bw / self.img_w),
                        "scale_mode": "width",
                        "in_view": True,
                    }
                else:
                    self.last_ctrl_telem = {
                        "status": "SEARCHING",
                        "error_x": 0.0,
                        "error_y": 0.0,
                        "error_range": 0.0,
                        "target_size": 0.0,
                        "desired_size": self.desired_bbox_size,
                        "bbox_w": 0.0,
                        "bbox_h": 0.0,
                        "w_norm": 0.0,
                        "scale_mode": "width",
                        "in_view": False,
                    }

    def _map_action_to_physical(self, action: np.ndarray) -> np.ndarray:
        """
        Combines PID baseline setpoint with bounded RL residual.
        u_total = clip(u_PID + Delta_u_RL)
        """
        # 1. Body velocity transformation for PID
        c, s = np.cos(self.sim.yaw), np.sin(self.sim.yaw)
        vx_b = c * self.sim.vel[0] + s * self.sim.vel[1]
        vy_b = -s * self.sim.vel[0] + c * self.sim.vel[1]
        vz_b = self.sim.vel[2]
        v_body = np.array([vx_b, vy_b, vz_b], dtype=np.float64)

        # 2. Query PID setpoint
        pid_cmd = self.pid.compute_cmd(
            self.last_ctrl_telem,
            drone_pitch=self.sim.pitch,
            drone_roll=self.sim.roll,
            dt=self.dt,
            vehicle_vel=v_body,
            vehicle_yaw_rate=self.sim.yaw_rate,
            current_alt_m=-self.sim.pos[2]
        )
        self.last_pid_cmd = pid_cmd.copy()

        # 3. Add bounded RL residual
        a = np.clip(action, -1.0, 1.0)
        if np.all(np.abs(a) < 1e-5):
            self.last_residual_cmd.fill(0.0)
            self.last_physical_cmd = pid_cmd.copy()
            return pid_cmd

        residual = a * self.residual_scales
        self.last_residual_cmd = residual.copy()
        total_cmd = pid_cmd + residual

        # 4. Clip within quadrotor envelope limits
        target_vx = float(np.clip(total_cmd[0], -self.max_rev_vel, self.max_fwd_vel))
        target_vy = float(np.clip(total_cmd[1], -self.max_lat_vel, self.max_lat_vel))
        target_vz = float(np.clip(total_cmd[2], -self.max_climb_vel, self.max_desc_vel))
        target_yaw_deg = float(np.clip(total_cmd[3], -self.max_yaw_rate_deg, self.max_yaw_rate_deg))

        # 5. Ground proximity safety governor
        current_alt_m = -self.sim.pos[2]
        if current_alt_m < 1.4 and target_vz > 0.0:
            target_vz = float(target_vz * max(0.0, (current_alt_m - 0.9) / 0.5))

        # 6. Slew-Rate Limiting across all axes
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

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        self._update_ctrl_telem()
        info["pid_cmd"] = self.last_pid_cmd.copy()
        info["residual_cmd"] = self.last_residual_cmd.copy()
        return obs, reward, terminated, truncated, info
