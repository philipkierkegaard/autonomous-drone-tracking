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
            if not isinstance(obs, torch.Tensor):
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
            else:
                obs_t = obs.float()

            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0).unsqueeze(0)
            elif obs_t.ndim == 2:
                obs_t = obs_t.unsqueeze(1)

            if episode_start is not None and np.any(episode_start):
                state = None

            act_t, next_state = self(obs_t, state)
            action = act_t.squeeze(1).numpy()
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
        img_w: int = 640,
        img_h: int = 480,
        hfov_deg: float = 60.0,
        vfov_deg: float = 45.0,
        camera_uptilt_deg: float = 15.0,
        max_vel_xy: float = 18.0,
        max_vel_down: float = 2.5,
        max_vel_up: float = 4.0,
        max_yaw_rate: float = 120.0,
        deterministic: bool = True,
        w_nominal: float = 0.0505,
    ):
        self.img_w = img_w
        self.img_h = img_h
        self.camera_uptilt = float(np.deg2rad(camera_uptilt_deg))
        self.half_hfov = float(np.tan(np.deg2rad(hfov_deg / 2.0)))
        self.half_vfov = float(np.tan(np.deg2rad(vfov_deg / 2.0)))
        self.w_nominal = w_nominal  # Target bbox width fraction at 6.0m standoff (~32.3 pixels)
        self.max_vel_xy = max_vel_xy
        self.max_vel_down = max_vel_down
        self.max_vel_up = max_vel_up
        self.max_yaw_rate = max_yaw_rate
        self.deterministic = deterministic

        self.model = None
        self.obs_dim = 10
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
                - 'status': 'TRACKING', 'LOCKED', or 'SEARCHING'
            drone_pitch: Instantaneous pitch angle (+ is nose down)
            drone_roll: Instantaneous roll angle (+ is right wing down)
            dt: Loop timestep (default: 0.02s = 50 Hz)
            vehicle_vel: Optional [vx, vy, vz] in body frame
            vehicle_yaw_rate: Optional yaw rate in rad/s
            current_alt_m: Current vehicle altitude AGL in meters

        Returns:
            np.ndarray of shape (4,): [v_forward (m/s), v_right (m/s), v_down (m/s), yaw_rate (deg/s)]
        """
        status = telemetry.get("status", "SEARCHING")
        if "in_view" in telemetry:
            is_visible = bool(telemetry["in_view"])
        else:
            is_visible = status in ("TRACKING", "LOCKED")

        raw_ex = telemetry.get("error_x", 0.0)
        raw_ey = telemetry.get("error_y", 0.0)
        bbox_w = telemetry.get("bbox_w", telemetry.get("target_size", 35.0))
        bbox_h = telemetry.get("bbox_h", 20.0)

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
            flag = 1.0
            self.last_seen_ex = ex
            self.last_seen_ey = ey
            self.last_seen_w = w_norm
            self.last_seen_h = h_norm

            if self.obs_dim >= 13:
                if self.was_in_view_prev:
                    raw_d_ex = (ex - self.prev_ex) / dt
                    raw_d_ey = (ey - self.prev_ey) / dt
                    raw_d_w  = (w_norm - self.prev_w) / dt
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
            ex = self.last_seen_ex
            ey = self.last_seen_ey
            w_norm = self.last_seen_w
            h_norm = self.last_seen_h
            flag = 0.0

            if self.obs_dim >= 13:
                self.filtered_d_ex *= 0.5
                self.filtered_d_ey *= 0.5
                self.filtered_d_w  *= 0.5
                self.was_in_view_prev = False

        # Construct observation vector
        v_body = vehicle_vel if vehicle_vel is not None else np.zeros(3)
        alt_norm = float(np.clip(current_alt_m / 5.0, 0.0, 2.0))
        scale_err = float(np.clip((self.w_nominal - w_norm) / self.w_nominal, -1.0, 1.0))

        base_obs = [
            ex,
            ey,
            scale_err,
            w_norm,
            flag,
            float(np.clip(v_body[0] / self.max_vel_xy, -1.0, 1.0)),
            float(np.clip(v_body[1] / 6.0, -1.0, 1.0)),
            float(np.clip(v_body[2] / self.max_vel_up, -1.0, 1.0)),
            float(np.clip(vehicle_yaw_rate / np.deg2rad(self.max_yaw_rate), -1.0, 1.0)),
            alt_norm,
        ]

        if self.obs_dim >= 13:
            norm_d_ex = float(np.clip(self.filtered_d_ex / 2.0, -1.0, 1.0))
            norm_d_ey = float(np.clip(self.filtered_d_ey / 2.0, -1.0, 1.0))
            norm_d_w  = float(np.clip(self.filtered_d_w  / 0.20, -1.0, 1.0))
            base_obs.extend([norm_d_ex, norm_d_ey, norm_d_w])

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
        a = np.clip(action, -1.0, 1.0)
        target_vx = float(5.0 + a[0] * 10.0)        # [-5.0 m/s, 15.0 m/s]
        target_vy = float(a[1] * 6.0)                # [-6.0 m/s, +6.0 m/s]
        target_vz = float(-0.75 + a[2] * 3.25)       # [-4.0 m/s, +2.5 m/s]
        target_yaw = float(a[3] * self.max_yaw_rate) # [-120 deg/s, +120 deg/s]

        # Physical Slew-Rate Limiting across all 4 flight axes (matched to Pixhawk quad envelope)
        max_accel_x = 6.5 * dt
        max_decel_x = 5.0 * dt
        if target_vx < self.prev_cmd[0]:
            cmd_vx = max(target_vx, self.prev_cmd[0] - max_decel_x)
        else:
            cmd_vx = min(target_vx, self.prev_cmd[0] + max_accel_x)

        max_accel_y = 5.0 * dt
        cmd_vy = float(np.clip(target_vy, self.prev_cmd[1] - max_accel_y, self.prev_cmd[1] + max_accel_y))

        max_accel_z = 3.0 * dt
        cmd_vz = float(np.clip(target_vz, self.prev_cmd[2] - max_accel_z, self.prev_cmd[2] + max_accel_z))

        max_accel_yaw = 180.0 * dt
        cmd_yaw = float(np.clip(target_yaw, self.prev_cmd[3] - max_accel_yaw, self.prev_cmd[3] + max_accel_yaw))

        cmd = np.array([cmd_vx, cmd_vy, cmd_vz, cmd_yaw], dtype=np.float64)
        self.prev_cmd = cmd.copy()
        return cmd

    @staticmethod
    def to_mavsdk(cmd: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Unpacks the 4D command vector into MAVSDK VelocityBodyYawspeed arguments:
        (forward_m_s, right_m_s, down_m_s, yawspeed_deg_s).
        """
        return float(cmd[0]), float(cmd[1]), float(cmd[2]), float(cmd[3])
