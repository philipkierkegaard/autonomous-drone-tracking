#!/usr/bin/env python3
"""
Recurrent Learned Visual Servoing & Predictive Guidance Controller.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Wraps a trained RecurrentPPO policy (Actor-Critic with LSTM/GRU latent state memory)
as a drop-in replacement for KinematicVisualServoController in:
1. run_tracker.py (Onboard Jetson Orin Nano flight loop).
2. benchmarks/control/benchmark_controller.py (Closed-loop Monte Carlo benchmarks).

Inputs:
    telemetry: Dict with 'error_x', 'error_y', 'bbox_w', 'bbox_h', 'status' / 'in_view'
    vehicle telemetry: drone_pitch, drone_roll, body velocities, yaw rate
Outputs:
    4D Command Vector: [v_forward (m/s), v_right (m/s), v_down (m/s), yaw_rate (deg/s)]
    Directly compatible with MAVSDK VelocityBodyYawspeed.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.spatial import unproject_camera_to_horizon


class StandaloneRecurrentPolicy(torch.nn.Module):
    """
    Standalone PyTorch implementation of the Recurrent Actor policy.
    Completely independent of stable-baselines3, cloudpickle, and Python/NumPy version mismatches.
    """
    def __init__(self, obs_dim: int = 10, hidden_dim: int = 128, act_dim: int = 4):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.act_dim = act_dim
        self.lstm = torch.nn.LSTM(obs_dim, hidden_dim, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh()
        )
        self.action_net = torch.nn.Linear(hidden_dim, act_dim)

    def forward(self, obs: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        out, next_hidden = self.lstm(obs, hidden)
        feat = self.mlp(out)
        action = self.action_net(feat)
        return action, next_hidden

    def predict(
        self,
        obs: np.ndarray,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        with torch.no_grad():
            dev = next(self.parameters()).device
            if not isinstance(obs, torch.Tensor):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev)
            else:
                obs_t = obs.float().to(dev)

            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0).unsqueeze(0)
            elif obs_t.ndim == 2:
                obs_t = obs_t.unsqueeze(1)

            if episode_start is not None and np.any(episode_start):
                state = None

            act_t, next_state = self(obs_t, state)
            action = act_t.cpu().squeeze(1).numpy()
            if action.shape[0] == 1 and obs.ndim == 1:
                return action[0], next_state
            return action, next_state


class RecurrentVisualServoController:
    """
    Learned Recurrent Guidance Controller for Autonomous Drone Interception.
    Implements lead-pursuit visual servoing with internal hidden state memory.
    """
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        img_w: int = 1280,
        img_h: int = 720,
        hfov_deg: float = 62.0,
        vfov_deg: float = 38.0,
        camera_uptilt_deg: float = 15.0,
        max_vel_xy: float = 15.0,
        max_rev_vel: float = 0.5,
        max_lat_vel: float = 4.0,
        max_vel_down: float = 2.0,
        max_vel_up: float = 3.0,
        max_yaw_rate: float = 120.0,
        max_accel_xy: float = 6.0,
        max_decel_xy: float = 6.0,
        max_accel_z: float = 3.0,
        max_accel_yaw: float = 240.0,
        deterministic: bool = True,
        w_nominal: float = 0.0859,
    ):
        self.img_w = img_w
        self.img_h = img_h
        self.camera_uptilt = float(np.deg2rad(camera_uptilt_deg))
        self.half_hfov = float(np.tan(np.deg2rad(hfov_deg / 2.0)))
        self.half_vfov = float(np.tan(np.deg2rad(vfov_deg / 2.0)))
        self.w_nominal = w_nominal  # Target bbox width fraction at 3.5m standoff (110.0 pixels on 1280x720)
        self.max_fwd_vel = max_vel_xy
        self.max_rev_vel = max_rev_vel
        self.max_lat_vel = max_lat_vel
        self.max_vel_down = max_vel_down
        self.max_vel_up = max_vel_up
        self.max_yaw_rate = max_yaw_rate
        self.max_accel_xy = max_accel_xy
        self.max_decel_xy = max_decel_xy
        self.max_accel_z = max_accel_z
        self.max_accel_yaw = max_accel_yaw
        self.deterministic = deterministic

        self.model = None
        self.obs_dim = 15
        self.hidden_state = None
        self.episode_start = np.array([True], dtype=bool)

        # Internal state history
        self.last_seen_ex = 0.0
        self.last_seen_ey = 0.0
        self.last_seen_w = self.w_nominal
        self.last_seen_h = 0.03
        self.prev_cmd = np.zeros(4, dtype=np.float64)

        # Causal optical rate filtering (for Gen 5 13D observation space)
        self.prev_ex = 0.0
        self.prev_ey = 0.0
        self.prev_w = self.w_nominal
        self.filtered_d_ex = 0.0
        self.filtered_d_ey = 0.0
        self.filtered_d_w = 0.0
        self.was_in_view_prev = False
        self.prev_frame_idx = None
        self.prev_frame_timestamp = None

        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path: Union[str, Path]):
        """Loads a trained RecurrentPPO policy from disk (supporting standalone PyTorch weights or SB3 zip)."""
        path = Path(model_path)
        if not path.exists():
            if path.with_suffix(".zip").exists():
                path = path.with_suffix(".zip")
            elif path.with_suffix(".pt").exists():
                path = path.with_suffix(".pt")
            else:
                print(f"[WARN] Recurrent model not found at {path}, controller uninitialized.")
                return

        # 1. First check if standalone PyTorch actor weights are available (immune to SB3 / NumPy issues)
        weights_candidates = [
            path if path.suffix == ".pt" else None,
            path.parent / "actor_weights.pt",
            path.parent / "policy_state_dict.pth",
        ]
        standalone_weights = next((p for p in weights_candidates if p and p.exists()), None)

        if standalone_weights is not None:
            print(f"[RECURRENT_CONTROLLER] Loading standalone PyTorch actor from: {standalone_weights}")
            try:
                state_dict = torch.load(str(standalone_weights), map_location="cpu")
                # Support both StandaloneRecurrentPolicy and SB3 actor state dicts
                if "lstm_actor.weight_ih_l0" in state_dict:
                    state_dict = {
                        "lstm.weight_ih_l0": state_dict["lstm_actor.weight_ih_l0"],
                        "lstm.weight_hh_l0": state_dict["lstm_actor.weight_hh_l0"],
                        "lstm.bias_ih_l0": state_dict["lstm_actor.bias_ih_l0"],
                        "lstm.bias_hh_l0": state_dict["lstm_actor.bias_hh_l0"],
                        "mlp.0.weight": state_dict["mlp_extractor.policy_net.0.weight"],
                        "mlp.0.bias": state_dict["mlp_extractor.policy_net.0.bias"],
                        "action_net.weight": state_dict["action_net.weight"],
                        "action_net.bias": state_dict["action_net.bias"],
                    }
                if "lstm.weight_ih_l0" in state_dict:
                    obs_dim = state_dict["lstm.weight_ih_l0"].shape[1]
                    hidden_dim = state_dict["lstm.weight_ih_l0"].shape[0] // 4
                    act_dim = state_dict["action_net.weight"].shape[0]
                    actor = StandaloneRecurrentPolicy(obs_dim=obs_dim, hidden_dim=hidden_dim, act_dim=act_dim)
                    actor.load_state_dict(state_dict)
                    actor.eval()
                    self.model = actor
                    self.obs_dim = obs_dim
                    print(f"[RECURRENT_CONTROLLER] Successfully loaded StandaloneRecurrentPolicy (obs_dim={self.obs_dim}, hidden={hidden_dim})")
                    self.reset()
                    return
            except Exception as e:
                print(f"[WARN] Failed loading standalone weights ({e}), falling back to SB3 loader...")

        # 2. Backwards compatibility fix for models pickled with NumPy 2.x on Python 3.11+
        import sys
        try:
            import numpy as np
            if not hasattr(np, "_core") and hasattr(np, "core"):
                import numpy.core as _np_core
                sys.modules["numpy._core"] = _np_core
                if hasattr(_np_core, "numeric"):
                    sys.modules["numpy._core.numeric"] = _np_core.numeric
                if hasattr(_np_core, "multiarray"):
                    sys.modules["numpy._core.multiarray"] = _np_core.multiarray
                if hasattr(_np_core, "umath"):
                    sys.modules["numpy._core.umath"] = _np_core.umath
        except Exception:
            pass

        from sb3_contrib import RecurrentPPO
        print(f"[RECURRENT_CONTROLLER] Loading policy checkpoint from: {path}")
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.2,
        }
        try:
            self.model = RecurrentPPO.load(str(path), device="cpu", custom_objects=custom_objects)
        except Exception as e:
            print(f"[WARN] First load attempt failed ({e}), retrying default RecurrentPPO.load...")
            self.model = RecurrentPPO.load(str(path), device="cpu")

        if hasattr(self.model, "observation_space") and hasattr(self.model.observation_space, "shape"):
            self.obs_dim = int(self.model.observation_space.shape[0])
        print(f"[RECURRENT_CONTROLLER] Loaded SB3 policy checkpoint (obs_dim={self.obs_dim}) from: {path}")
        self.reset()

    def reset(self):
        """Resets the recurrent hidden state at the start of a new flight or target acquisition."""
        self.hidden_state = None
        self.episode_start = np.array([True], dtype=bool)
        self.last_seen_ex = 0.0
        self.last_seen_ey = 0.0
        self.last_seen_w = self.w_nominal
        self.last_seen_h = 0.03
        self.prev_cmd = np.zeros(4, dtype=np.float64)
        self.prev_ex = 0.0
        self.prev_ey = 0.0
        self.prev_w = self.w_nominal
        self.filtered_d_ex = 0.0
        self.filtered_d_ey = 0.0
        self.filtered_d_w = 0.0
        self.was_in_view_prev = False
        self.prev_frame_idx = None
        self.prev_frame_timestamp = None
        self.last_frame_time = None

    def compute_cmd(
        self,
        telemetry: Dict[str, Any],
        drone_pitch: float = 0.0,
        drone_roll: float = 0.0,
        dt: float = 0.02,
        vehicle_vel: Optional[np.ndarray] = None,
        vehicle_yaw_rate: float = 0.0,
        current_alt_m: float = 2.5
    ) -> np.ndarray:
        """
        Drop-in replacement for KinematicVisualServoController.compute_cmd.

        Args:
            telemetry: Dict containing:
                - 'error_x': Normalized [-1, 1] horizontal error
                - 'error_y': Normalized [-1, 1] vertical error
                - 'bbox_w': Bounding box width (pixels)
                - 'bbox_h': Bounding box height (pixels)
                - 'status': 'TRACKING', 'LOCKED', 'COASTING', or 'SEARCHING'
            drone_pitch: Instantaneous pitch angle (+ is nose down)
            drone_roll: Instantaneous roll angle (+ is right wing down)
            dt: Loop timestep (default: 0.02s = 50 Hz)
            vehicle_vel: Optional [vx, vy, vz] in body frame
            vehicle_yaw_rate: Optional yaw rate in rad/s
            current_alt_m: Current vehicle altitude AGL in meters

        Returns:
            np.ndarray of shape (4,): [v_forward (m/s), v_right (m/s), v_down (m/s), yaw_rate (deg/s)]
        """
        # Strictly sanitize telemetry: RecurrentVisualServoController is 100% optical and must never receive ground-truth distance
        telemetry = {k: v for k, v in telemetry.items() if k != "distance"}

        status = telemetry.get("status", "SEARCHING")
        if status in ("TRACKING", "LOCKED"):
            flag = 1.0
            is_visible = True
        elif status == "COASTING":
            flag = 0.5
            is_visible = True
        else:
            flag = 0.0
            is_visible = False

        raw_ex = telemetry.get("error_x", 0.0)
        raw_ey = telemetry.get("error_y", 0.0)
        bbox_w = telemetry.get("bbox_w", telemetry.get("target_size", 35.0))
        bbox_h = telemetry.get("bbox_h", 20.0)

        current_frame_idx = telemetry.get("frame_idx", None)
        frame_timestamp = telemetry.get("timestamp", None)
        is_fresh_frame = (current_frame_idx is None or current_frame_idx != self.prev_frame_idx)

        if is_visible:
            # SO(3) Attitude Decoupling: Un-roll & un-pitch camera frame errors to horizon frame
            h_ex, h_ey, _, _ = unproject_camera_to_horizon(
                err_x=float(raw_ex),
                err_y=float(raw_ey),
                drone_pitch=float(drone_pitch),
                drone_roll=float(drone_roll),
                camera_uptilt_rad=self.camera_uptilt,
                half_hfov=self.half_hfov,
                half_vfov=self.half_vfov,
            )
            ex = float(np.clip(h_ex, -1.0, 1.0))
            ey = float(np.clip(h_ey, -1.0, 1.0))
            w_norm = float(np.clip(bbox_w / self.img_w, 0.01, 1.0))
            h_norm = float(np.clip(bbox_h / self.img_h, 0.01, 1.0))
            self.last_seen_ex = ex
            self.last_seen_ey = ey
            self.last_seen_w = w_norm
            self.last_seen_h = h_norm

            if self.obs_dim >= 13:
                if is_fresh_frame:
                    if self.was_in_view_prev:
                        # Determine true camera interval dt_cam from consecutive frame timestamps
                        if frame_timestamp is not None and getattr(self, "prev_frame_timestamp", None) is not None:
                            raw_dt_cam = float(frame_timestamp - self.prev_frame_timestamp)
                            dt_cam = raw_dt_cam if 0.005 <= raw_dt_cam <= 0.150 else (1.0 / 30.0)
                        else:
                            dt_cam = 1.0 / 30.0

                        raw_d_ex = (ex - self.prev_ex) / dt_cam
                        raw_d_ey = (ey - self.prev_ey) / dt_cam
                        raw_d_w  = (w_norm - self.prev_w) / dt_cam
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
                    self.prev_frame_idx = current_frame_idx
                    self.prev_frame_timestamp = frame_timestamp

                self.was_in_view_prev = True
        else:
            ex = self.last_seen_ex
            ey = self.last_seen_ey
            w_norm = self.last_seen_w
            h_norm = self.last_seen_h

            if self.obs_dim >= 13:
                self.filtered_d_ex *= 0.5
                self.filtered_d_ey *= 0.5
                self.filtered_d_w  *= 0.5
                self.was_in_view_prev = False

        # Construct observation vector with robust fallback for missing/stale telemetry
        if vehicle_vel is not None and np.all(np.isfinite(vehicle_vel)):
            v_body = np.asarray(vehicle_vel, dtype=np.float32)
        else:
            v_body = np.array([self.prev_cmd[0], self.prev_cmd[1], self.prev_cmd[2]], dtype=np.float32)

        yaw_rate_val = float(vehicle_yaw_rate) if (vehicle_yaw_rate is not None and np.isfinite(vehicle_yaw_rate)) else float(np.deg2rad(self.prev_cmd[3]))
        alt_m = float(current_alt_m) if (current_alt_m is not None and np.isfinite(current_alt_m)) else 2.5
        alt_norm = float(np.clip(alt_m / 5.0, 0.0, 2.0))
        scale_err = float(np.clip((self.w_nominal - w_norm) / self.w_nominal, -1.0, 1.0))

        base_obs = [
            ex,
            ey,
            scale_err,
            w_norm,
            flag,
            float(np.clip(v_body[0] / self.max_fwd_vel, -1.0, 1.0)),
            float(np.clip(v_body[1] / self.max_lat_vel, -1.0, 1.0)),
            float(np.clip(v_body[2] / self.max_vel_up, -1.0, 1.0)),
            float(np.clip(yaw_rate_val / np.deg2rad(self.max_yaw_rate), -1.0, 1.0)),
            alt_norm,
        ]

        if self.obs_dim >= 13:
            norm_d_ex = float(np.clip(self.filtered_d_ex / 2.0, -1.0, 1.0))
            norm_d_ey = float(np.clip(self.filtered_d_ey / 2.0, -1.0, 1.0))
            norm_d_w  = float(np.clip(self.filtered_d_w  / 0.20, -1.0, 1.0))
            base_obs.extend([norm_d_ex, norm_d_ey, norm_d_w])

        if self.obs_dim >= 15:
            norm_pitch = float(np.clip(drone_pitch / (np.pi / 4.0), -1.0, 1.0))
            norm_roll  = float(np.clip(drone_roll  / (np.pi / 4.0), -1.0, 1.0))
            base_obs.extend([norm_pitch, norm_roll])

        obs = np.array(base_obs, dtype=np.float32)

        if self.model is None:
            # Fallback zero-control if no policy loaded
            return np.zeros(4, dtype=np.float64)

        # Recurrent inference step
        action, self.hidden_state = self.model.predict(
            obs,
            state=self.hidden_state,
            episode_start=self.episode_start,
            deterministic=self.deterministic
        )
        self.episode_start = np.array([False], dtype=bool)

        # Unpack continuous action [-1, 1]^4 into physical velocity setpoints
        # Strictly matching the physical limits and cruise-centered mapping of environment.py
        a = np.clip(action, -1.0, 1.0)
        standoff_nom = 3.5
        if a[0] >= 0:
            target_vx = float(standoff_nom + a[0] * (self.max_fwd_vel - standoff_nom))
        else:
            target_vx = float(standoff_nom + a[0] * (standoff_nom + self.max_rev_vel))

        # Forward velocity target
        target_vy = float(a[1] * self.max_lat_vel)

        # a2 in NED: negative is climb, positive is descend (100% learned from policy)
        if a[2] <= 0:
            target_vz = float(a[2] * self.max_vel_up)
        else:
            target_vz = float(a[2] * self.max_vel_down)

        target_yaw = float(a[3] * self.max_yaw_rate)

        # Physical Slew-Rate Limiting across all 4 flight axes (matched identically to KinematicVisualServoController)
        max_step_accel_x = self.max_accel_xy * dt
        max_step_decel_x = self.max_decel_xy * dt
        if target_vx < self.prev_cmd[0]:
            cmd_vx = max(target_vx, self.prev_cmd[0] - max_step_decel_x)
        else:
            cmd_vx = min(target_vx, self.prev_cmd[0] + max_step_accel_x)

        max_step_lat = self.max_accel_xy * dt
        cmd_vy = float(np.clip(target_vy, self.prev_cmd[1] - max_step_lat, self.prev_cmd[1] + max_step_lat))

        max_step_z = self.max_accel_z * dt
        cmd_vz = float(np.clip(target_vz, self.prev_cmd[2] - max_step_z, self.prev_cmd[2] + max_step_z))

        max_step_yaw = self.max_accel_yaw * dt
        cmd_yaw = float(np.clip(target_yaw, self.prev_cmd[3] - max_step_yaw, self.prev_cmd[3] + max_step_yaw))

        cmd = np.array([cmd_vx, cmd_vy, cmd_vz, cmd_yaw], dtype=np.float64)
        self.prev_cmd = cmd.copy()
        return cmd

    @staticmethod
    def to_velocity_body(cmd: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Unpacks the 4D command vector into 4-DOF body velocity arguments:
        (forward_m_s, right_m_s, down_m_s, yawspeed_deg_s).
        Directly compatible with ArduPilot MAVLink SET_POSITION_TARGET_LOCAL_NED (#84)
        as well as legacy MAVSDK VelocityBodyYawspeed.
        """
        return float(cmd[0]), float(cmd[1]), float(cmd[2]), float(cmd[3])

    # Dual aliases for compatibility across ArduPilot and legacy MAVSDK scripts
    to_autopilot = to_velocity_body
    to_mavsdk = to_velocity_body
