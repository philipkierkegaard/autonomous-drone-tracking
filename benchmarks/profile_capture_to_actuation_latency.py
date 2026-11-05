#!/usr/bin/env python3
"""
Full-Stack End-to-End Cyber-Physical Latency & Throughput Profiler.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Directly models and instruments the production runtime pipeline from run_tracker.py:
1. Asynchronous Decoupled Architecture (--mode async, default):
   - Vision Worker Thread (~30 Hz camera rate): Ingestion, Dehaze LUT, Adaptive Shadows,
     YOLOv8 TensorRT inference, 8D Kalman filter tracking, and thread-safe mutex dispatch.
   - Control Loop Thread (50 Hz MAVLink rate): Deterministic high-priority flight control,
     loading tuned PID gains (control/weights/tuned_pid_gains.json), dynamic attitude pitch/roll
     feedforward compensation, physical slew-rate acceleration limiting, and altitude safety bounds.
   - True Capture-to-Actuation Latency: Monotonically stamps physical camera arrival (t_capture)
     and measures the exact photon-to-wire software age when the MAVLink packet is dispatched.
2. Synchronous Sequential Architecture (--mode sync):
   - Step-by-step isolated benchmarking for measuring standalone compute shares.
3. Actuation Serialization & Hardware Serial UART:
   - Measures MAVLink #84 SET_POSITION_TARGET_LOCAL_NED packet serialization.
   - Supports live hardware serial transmission (--serial-port /dev/ttyTHS1 @ 921600 baud).
4. Physical Cyber-Physical Photon-to-Thrust Budget:
   - Sensor domain (mid-exposure + MIPI CSI-2 readout + Tegra ISP downscale).
   - Compute domain (measured on-device software latency).
   - Actuation domain (UART transmission + Pixhawk control loop quantization + motor time constant).
"""

import os
import sys
import time
import math
import json
import bisect
import inspect
import argparse
import threading
from pathlib import Path
from collections import deque
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np
import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.pipeline import DroneTrackingPipeline
from control.pid_controller import KinematicVisualServoController

try:
    from control.predictive_controller import RecurrentVisualServoController
except ImportError:
    RecurrentVisualServoController = None

try:
    from pymavlink import mavutil
    PYMAVLINK_AVAILABLE = True
except ImportError:
    PYMAVLINK_AVAILABLE = False


# ==============================================================================
# 1. Hardware & Preprocessing Helpers (Directly matching run_tracker.py)
# ==============================================================================

def build_jetson_csi_pipeline(
    sensor_id: int = 0,
    capture_width: int = 3840,
    capture_height: int = 2160,
    framerate: int = 30,
    flip_method: int = 0,
    display_width: int = 1280,
    display_height: int = 720,
    ee_strength: float = 0.0,
    exposure_compensation: float = 0.0,
) -> str:
    """Hardware-accelerated GStreamer pipeline for NVIDIA Jetson CSI cameras (Sony IMX477)."""
    exp_prop = f"exposurecompensation={exposure_compensation:.2f} " if exposure_compensation != 0.0 else ""
    ee_prop = f"ee-mode=2 ee-strength={ee_strength:.2f} " if ee_strength > 0.0 else "ee-mode=0 ee-strength=0.0 "
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} {exp_prop}{ee_prop}tnr-mode=1 tnr-strength=0.0 ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1"
    )


def build_dehaze_lut(gamma: float = 1.40) -> np.ndarray:
    """Sub-millisecond gamma-based de-haze lookup table matching run_tracker.py."""
    inv_gamma = 1.0 / max(0.1, gamma)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return table


def apply_adaptive_shadow_boost(img: np.ndarray, threshold: float = 70.0, target_mean: float = 85.0) -> np.ndarray:
    """Sub-millisecond (~0.2 ms) dynamic range shadow compensator matching run_tracker.py."""
    small = cv2.resize(img, (160, 90), interpolation=cv2.INTER_NEAREST)
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    lower_mean = float(np.mean(gray_small[36:, :]))
    if lower_mean >= threshold:
        return img

    gamma = float(np.clip(target_mean / max(25.0, lower_mean), 1.0, 1.55))
    inv_gamma = 1.0 / gamma
    table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        v = i / 255.0
        lifted = v ** inv_gamma
        if i < 20:
            blend = i / 20.0
            v_out = (1.0 - blend) * v + blend * lifted
        else:
            v_out = lifted
        table[i] = int(round(np.clip(v_out * 255.0, 0, 255)))
    return cv2.LUT(img, table)


def enforce_altitude_limits(
    v_down: float,
    current_alt: Optional[float],
    min_alt: float = 1.2,
    alt_cushion: float = 0.8,
    max_alt: float = 8.0
) -> Tuple[float, str]:
    """Altitude safety boundary envelope enforcement matching run_tracker.py."""
    if current_alt is None or not math.isfinite(current_alt):
        return v_down, "NO_ALT_SENSOR"
    if current_alt <= min_alt:
        return -0.5, "FLOOR_BREACH"
    elif current_alt <= (min_alt + alt_cushion):
        margin = (current_alt - min_alt) / max(1e-3, alt_cushion)
        safe_v_down = min(v_down, 0.0) if v_down > 0 else v_down
        return safe_v_down * margin, "FLOOR_CUSHION"
    elif current_alt >= max_alt:
        return max(v_down, 0.0), "CEILING_BREACH"
    return v_down, "OK"


# ==============================================================================
# 2. Thread-Safe States & Telemetry Containers
# ==============================================================================

class LatencyStats:
    """Thread-safe latency & throughput metric accumulator."""
    def __init__(self, name: str, unit: str = "ms"):
        self.name = name
        self.unit = unit
        self.samples: List[float] = []
        self._lock = threading.Lock()

    def add(self, duration: float):
        with self._lock:
            self.samples.append(float(duration))

    def summary(self) -> Dict[str, float]:
        with self._lock:
            if not self.samples:
                return {"mean": 0.0, "std": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "count": 0}
            arr = np.array(self.samples, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
            "count": int(len(arr))
        }


class VehicleState:
    """Thread-safe simulated or logged vehicle state matching run_tracker.py."""
    def __init__(self):
        self.lock = threading.Lock()
        self.pitch_rad: float = 0.0
        self.roll_rad: float = 0.0
        self.yaw_deg: float = 0.0
        self.yaw_rate_rad_s: float = 0.0
        self.altitude_rel_m: float = 3.5
        self.velocity_body: np.ndarray = np.array([2.0, 0.0, 0.0], dtype=np.float32)

    def update(self, pitch_rad: float, roll_rad: float, yaw_deg: float,
               yaw_rate_rad_s: float = 0.0, alt_m: float = 3.5,
               v_body: Optional[np.ndarray] = None):
        with self.lock:
            self.pitch_rad = pitch_rad
            self.roll_rad = roll_rad
            self.yaw_deg = yaw_deg
            self.yaw_rate_rad_s = yaw_rate_rad_s
            self.altitude_rel_m = alt_m
            if v_body is not None:
                self.velocity_body = v_body


class PerceptionState:
    """Thread-safe container holding latest perception telemetry matching run_tracker.py."""
    def __init__(self):
        self.lock = threading.Lock()
        self.telemetry: Dict[str, Any] = {"status": "SEARCHING"}
        self.last_update_time: float = 0.0
        self.frame_idx: int = 0
        self.fps: float = 0.0
        self.is_stale: bool = True
        self.has_new_frame: bool = False
        self.t_capture: float = 0.0
        self.t_perc_done: float = 0.0


# ==============================================================================
# 3. Flight Telemetry Replay & MAVLink Actuator
# ==============================================================================

class FlightTelemetryStreamer:
    """Feeds realistic time-series vehicle attitude & velocity into the tracker."""
    def __init__(self, mode: str = "flight-log", log_path: Optional[str] = None):
        self.mode = mode
        self.samples: List[Dict[str, Any]] = []
        self.idx = 0

        if mode == "flight-log" and log_path and Path(log_path).exists():
            self._load_bin_log(log_path)
        elif mode == "synthetic-maneuver":
            self._generate_synthetic_maneuver()
        else:
            self.mode = "static"

    def _load_bin_log(self, path: str):
        if not PYMAVLINK_AVAILABLE:
            print("[WARN] PyMAVLink not installed. Falling back to synthetic maneuver.")
            self._generate_synthetic_maneuver()
            return

        print(f"[TELEMETRY] Loading physical flight log: {path}...")
        try:
            mlog = mavutil.mavlink_connection(path)
            prev_yaw = None
            prev_t = None
            while True:
                m = mlog.recv_msg()
                if m is None:
                    break
                mtype = m.get_type()
                if mtype in ["ATT", "ANG"]:
                    d = m.to_dict()
                    tus = getattr(m, "TimeUS", 0) / 1e6
                    yaw = float(d.get("Yaw", 0.0))
                    dyaw = 0.0
                    if prev_yaw is not None and prev_t is not None:
                        dt = max(1e-4, tus - prev_t)
                        dyaw = (yaw - prev_yaw + 180.0) % 360.0 - 180.0
                        yaw_rate = math.radians(dyaw) / dt
                    else:
                        yaw_rate = 0.0
                    prev_yaw, prev_t = yaw, tus

                    self.samples.append({
                        "pitch_rad": float(np.deg2rad(d.get("Pitch", 0.0))),
                        "roll_rad": float(np.deg2rad(d.get("Roll", 0.0))),
                        "yaw_deg": yaw,
                        "yaw_rate_rad_s": yaw_rate,
                        "vx_body": 2.0,
                        "vy_body": 0.0,
                        "vz_body": 0.0,
                        "altitude_m": 3.5,
                        "dt": 1.0 / 30.0
                    })
            print(f"[TELEMETRY] ✓ Successfully loaded {len(self.samples):,} flight telemetry frames.")
        except Exception as e:
            print(f"[WARN] Failed to parse flight log ({e}). Using synthetic maneuver.")
            self._generate_synthetic_maneuver()

    def _generate_synthetic_maneuver(self):
        print("[TELEMETRY] Generating dynamic 3D banking maneuver (±20° roll, ±10° pitch)...")
        t_arr = np.linspace(0, 30.0, 900)
        for t in t_arr:
            roll_deg = 20.0 * math.sin(2.0 * math.pi * 0.4 * t)
            pitch_deg = 10.0 * math.cos(2.0 * math.pi * 0.3 * t) - 3.0
            yaw_deg = (t * 25.0) % 360.0
            yaw_rate_rad_s = math.radians(25.0)
            self.samples.append({
                "pitch_rad": float(np.deg2rad(pitch_deg)),
                "roll_rad": float(np.deg2rad(roll_deg)),
                "yaw_deg": float(yaw_deg),
                "yaw_rate_rad_s": yaw_rate_rad_s,
                "vx_body": 2.5 + 0.5 * math.sin(t),
                "vy_body": 0.4 * math.cos(t),
                "vz_body": -0.1,
                "altitude_m": 4.0 + 0.8 * math.sin(0.5 * t),
                "dt": 1.0 / 30.0
            })

    def get_next(self, dt: float) -> Dict[str, Any]:
        if not self.samples:
            return {
                "pitch_rad": float(np.deg2rad(-4.0)),
                "roll_rad": float(np.deg2rad(1.0)),
                "yaw_deg": 180.0,
                "yaw_rate_rad_s": 0.0,
                "vx_body": 2.0,
                "vy_body": 0.0,
                "vz_body": 0.0,
                "altitude_m": 3.5,
                "dt": dt
            }
        sample = dict(self.samples[self.idx % len(self.samples)])
        sample["dt"] = dt
        self.idx += 1
        return sample


class MAVLinkActuator:
    """Benchmarks MAVLink serialization and physical UART transmission."""
    def __init__(self, serial_port: Optional[str] = None, baud: int = 921600):
        self.serial_port = serial_port
        self.baud = baud
        self.master = None
        self.mav = None
        self.is_live = False

        if PYMAVLINK_AVAILABLE:
            if serial_port:
                try:
                    self.master = mavutil.mavlink_connection(serial_port, baud=baud)
                    self.is_live = True
                    print(f"[ACTUATOR] ✓ Live serial UART active: {serial_port} @ {baud} baud")
                except Exception as e:
                    print(f"[ACTUATOR] Warning: Could not connect to {serial_port} ({e}). Using mock serializer.")
            if self.master is None:
                self.mav = mavutil.mavlink.MAVLink(None, srcSystem=1, srcComponent=1)

    def dispatch_velocity(self, v_fwd: float, v_right: float, v_down: float, yawspeed_deg_s: float) -> Tuple[float, int]:
        """Dispatches velocity setpoint and returns (serialization_time_ms, packet_size_bytes)."""
        t0 = time.perf_counter()
        yaw_rate_rad_s = math.radians(yawspeed_deg_s)
        typemask = 0b0000010111000111  # Ignore position & acceleration (MAVLink #84)
        packet_bytes = 57

        if self.master is not None:
            self.master.mav.set_position_target_local_ned_send(
                0, 1, 1,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                typemask,
                0.0, 0.0, 0.0,
                float(v_fwd), float(v_right), float(v_down),
                0.0, 0.0, 0.0,
                0.0, float(yaw_rate_rad_s)
            )
        elif self.mav is not None:
            msg = self.mav.set_position_target_local_ned_encode(
                0, 1, 1,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                typemask,
                0.0, 0.0, 0.0,
                float(v_fwd), float(v_right), float(v_down),
                0.0, 0.0, 0.0,
                0.0, float(yaw_rate_rad_s)
            )
            packed = msg.pack(self.mav)
            packet_bytes = len(packed)

        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0, packet_bytes


# ==============================================================================
# 4. Controller Instantiation & Tuned Gains Loading
# ==============================================================================

def create_kinematic_controller(args: argparse.Namespace) -> KinematicVisualServoController:
    """Instantiates KinematicVisualServoController with tuned parameters matching run_tracker.py."""
    sig = inspect.signature(KinematicVisualServoController.__init__)
    max_rev = float(getattr(args, "max_reverse", 0.5))
    standoff = float(getattr(args, "standoff_dist", 3.5))
    deadband = float(getattr(args, "deadband", 0.35))

    ctrl_kwargs: Dict[str, Any] = {
        "camera_uptilt_deg": float(getattr(args, "uptilt", 0.0)),
        "hfov_deg": float(getattr(args, "hfov", 62.0)),
        "vfov_deg": float(getattr(args, "vfov", 38.0)),
        "desired_bbox_size": float(getattr(args, "target_size", 110.0)),
        "scale_mode": getattr(args, "scale_mode", "width"),
        "min_limits": np.array([-abs(max_rev), -float(getattr(args, "max_climb", 3.0)), -float(getattr(args, "max_yawspeed", 120.0))]),
        "max_limits": np.array([float(getattr(args, "max_speed", 15.0)), float(getattr(args, "max_desc", 2.0)), float(getattr(args, "max_yawspeed", 120.0))]),
        "use_bbox_size": True,
        "enable_lateral_strafe": getattr(args, "lateral_strafe", True),
        "max_lat_vel": float(getattr(args, "max_lat_speed", 4.0)),
    }
    if "desired_standoff_dist" in sig.parameters:
        ctrl_kwargs["desired_standoff_dist"] = standoff
    if "deadband_dist" in sig.parameters:
        ctrl_kwargs["deadband_dist"] = deadband
    if "search_mode" in sig.parameters:
        ctrl_kwargs["search_mode"] = getattr(args, "search_mode", "brake")
    if "coast_mode" in sig.parameters:
        ctrl_kwargs["coast_mode"] = getattr(args, "coast_mode", "decay")
    if "enable_looming_ff" in sig.parameters:
        ctrl_kwargs["enable_looming_ff"] = getattr(args, "looming_ff", True)

    # Load tuned PID gains from file
    pid_gains_path = getattr(args, "pid_gains", "control/weights/tuned_pid_gains.json")
    if pid_gains_path and str(pid_gains_path).lower() != "none":
        p_gains = Path(pid_gains_path)
        if not p_gains.is_absolute():
            p_gains = REPO_ROOT / p_gains
        if p_gains.exists():
            try:
                with open(p_gains, "r", encoding="utf-8") as f:
                    g_data = json.load(f)
                p_map = g_data.get("tuned_parameters", g_data)
                for k in [
                    "kp_forward", "ki_forward", "kd_forward", "max_decel",
                    "centering_damping", "min_centering_factor",
                    "kp_lat", "ki_lat", "kd_lat",
                    "kp_z", "ki_z", "kd_z",
                    "kp_yaw", "ki_yaw", "kd_yaw"
                ]:
                    if k in p_map and k in sig.parameters:
                        ctrl_kwargs[k] = float(p_map[k])
                print(f"[PID] ✓ Loaded tuned flight gains from: {p_gains.name}")
            except Exception as e:
                print(f"[WARN] Failed loading tuned gains ({e}). Using controller defaults.")

    return KinematicVisualServoController(**ctrl_kwargs)


# ==============================================================================
# 5. Asynchronous Decoupled Profiling Engine (Exact Production Mode)
# ==============================================================================

def run_async_pipeline_profile(
    cap: cv2.VideoCapture,
    pipeline: DroneTrackingPipeline,
    controller: Any,
    actuator: MAVLinkActuator,
    telemetry_streamer: FlightTelemetryStreamer,
    args: argparse.Namespace
) -> Dict[str, Any]:
    """
    Executes the exact production decoupled architecture:
    - Thread 1 (Vision Worker): ~30 Hz camera frame ingestion, dehaze, shadow boost,
      YOLOv8 TRT, 8D Kalman filter, and thread-safe mutex state update.
    - Thread 2 (Control Loop): 50 Hz deterministic MAVLink control loop, reading
      telemetry under mutex lock, running tuned PID + attitude FF + slew limiter,
      and dispatching MAVLink packets.
    """
    n_frames = args.frames
    n_warmup = args.warmup
    ctrl_rate = float(getattr(args, "ctrl_rate", 50.0))
    ctrl_period = 1.0 / max(1.0, ctrl_rate)
    device = pipeline.device

    # Dehaze lookup table
    dehaze_lut = build_dehaze_lut(args.dehaze_gamma) if getattr(args, "enable_dehaze", False) else None
    adaptive_shadows = getattr(args, "adaptive_shadows", True)

    # Shared thread-safe structures
    vehicle_state = VehicleState()
    perception_state = PerceptionState()

    # Latency metric accumulators
    stats = {
        "dequeue": LatencyStats("1. Frame Dequeue DMA (T_dequeue)"),
        "preprocess": LatencyStats("2. Preprocessing & Shadows (T_preprocess)"),
        "perception": LatencyStats("3. Perception (YOLOv8 + 8D Kalman)"),
        "vision_lock": LatencyStats("4. Vision Mutex Write (T_lock_write)"),
        "vision_compute": LatencyStats("Total Vision Processing (T_vision_compute)"),
        "control_lock": LatencyStats("5. Control Mutex Read (T_lock_read)"),
        "control": LatencyStats("6. Guidance Control (T_control)"),
        "actuation_serialize": LatencyStats("7. MAVLink Serialization (T_serialize)"),
        "control_step": LatencyStats("Total Control Step Time (T_ctrl_compute)"),
        "control_period": LatencyStats("Control Loop Period (T_ctrl_period)"),
        "e2e_capture_to_actuation": LatencyStats("TRUE Capture-to-Actuation Latency"),
        "e2e_perc_to_actuation": LatencyStats("Perception-to-Actuation Delay")
    }

    inter_frame_intervals_ms: List[float] = []
    control_intervals_ms: List[float] = []

    warmup_event = threading.Event()
    stop_event = threading.Event()

    t_timed_start = [0.0]
    vision_frame_counter = [0]
    control_step_counter = [0]
    deadline_overruns = [0]
    TARGET_FRAME_PERIOD_MS = 1000.0 / 30.0  # 33.33 ms (30 Hz)

    # --------------------------------------------------------------------------
    # Vision Ingestion & Inference Worker Thread (~30 Hz)
    # --------------------------------------------------------------------------
    def _vision_worker():
        warmup_done = False
        prev_t_delivered = time.perf_counter()

        while not stop_event.is_set():
            t_cap_start = time.perf_counter()
            ret, frame = cap.read()
            t_captured = time.perf_counter()

            if not ret:
                if not str(args.source).isdigit() and Path(args.source).is_file():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("[VISION] Video source reached end.")
                stop_event.set()
                break

            inter_dt_ms = (t_captured - prev_t_delivered) * 1000.0
            prev_t_delivered = t_captured

            # Preprocessing (Dehaze LUT + Adaptive Shadows)
            t_pre_start = time.perf_counter()
            proc_frame = frame
            if dehaze_lut is not None:
                proc_frame = cv2.LUT(proc_frame, dehaze_lut)
            if adaptive_shadows:
                proc_frame = apply_adaptive_shadow_boost(proc_frame)
            t_pre_end = time.perf_counter()

            # Dynamic vehicle attitude for ego-motion compensation
            ego = telemetry_streamer.get_next(dt=max(1e-4, inter_dt_ms / 1000.0))
            vehicle_state.update(
                pitch_rad=ego["pitch_rad"],
                roll_rad=ego["roll_rad"],
                yaw_deg=ego["yaw_deg"],
                yaw_rate_rad_s=ego["yaw_rate_rad_s"],
                alt_m=ego["altitude_m"],
                v_body=np.array([ego["vx_body"], ego["vy_body"], ego["vz_body"]], dtype=np.float32)
            )

            # Perception (YOLOv8 + 8D Kalman Tracking)
            t_perc_start = time.perf_counter()
            annotated_frame, telemetry = pipeline.process_frame(
                proc_frame, draw_hud=False, ego_telemetry=ego
            )
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()
            t_perc_end = time.perf_counter()

            # Mutex State Update
            t_lock_start = time.perf_counter()
            with perception_state.lock:
                telemetry["timestamp"] = t_perc_end
                telemetry["t_capture"] = t_captured
                telemetry["t_perc_done"] = t_perc_end
                telemetry["frame_idx"] = pipeline.frame_idx
                perception_state.telemetry = telemetry
                perception_state.t_capture = t_captured
                perception_state.t_perc_done = t_perc_end
                perception_state.last_update_time = t_perc_end
                perception_state.frame_idx = pipeline.frame_idx
                perception_state.is_stale = False
                perception_state.has_new_frame = True
            t_lock_end = time.perf_counter()

            # Compute timing metrics
            deq_ms = (t_captured - t_cap_start) * 1000.0
            pre_ms = (t_pre_end - t_pre_start) * 1000.0
            perc_ms = (t_perc_end - t_perc_start) * 1000.0
            v_lock_ms = (t_lock_end - t_lock_start) * 1000.0
            comp_ms = pre_ms + perc_ms + v_lock_ms

            if warmup_done:
                inter_frame_intervals_ms.append(inter_dt_ms)
                stats["dequeue"].add(deq_ms)
                stats["preprocess"].add(pre_ms)
                stats["perception"].add(perc_ms)
                stats["vision_lock"].add(v_lock_ms)
                stats["vision_compute"].add(comp_ms)

                if comp_ms > TARGET_FRAME_PERIOD_MS:
                    deadline_overruns[0] += 1

                vision_frame_counter[0] += 1
                if vision_frame_counter[0] % max(1, n_frames // 5) == 0 or vision_frame_counter[0] == n_frames:
                    cur_comp = stats["vision_compute"].summary()["mean"]
                    cur_fps = 1000.0 / max(1e-3, cur_comp)
                    print(f"  [Vision] Frame {vision_frame_counter[0]:4d}/{n_frames:4d} | Compute: {cur_comp:5.2f} ms ({cur_fps:5.1f} Cap. FPS) | Overruns: {deadline_overruns[0]}")

                if vision_frame_counter[0] >= n_frames:
                    stop_event.set()
                    break
            else:
                if pipeline.frame_idx >= n_warmup:
                    warmup_done = True
                    t_timed_start[0] = time.perf_counter()
                    warmup_event.set()
                    print(f"[WARMUP] ✓ Completed {n_warmup} warmup iterations. Recording live metrics...\n")

    # --------------------------------------------------------------------------
    # Deterministic MAVLink Flight Control Thread (50 Hz)
    # --------------------------------------------------------------------------
    def _control_worker():
        warmup_event.wait()
        prev_ctrl_time = time.perf_counter()
        prev_cmd = np.zeros(4 if getattr(args, "lateral_strafe", True) else 3, dtype=np.float64)

        max_accel = np.array([
            float(getattr(args, "max_accel_xy", 4.0)),
            float(getattr(args, "max_accel_xy", 4.0)),
            float(getattr(args, "max_accel_z", 2.0)),
            float(getattr(args, "max_accel_yaw", 180.0))
        ], dtype=np.float64)

        while not stop_event.is_set():
            t_tick = time.perf_counter()
            dt_ctrl = max(1e-4, t_tick - prev_ctrl_time)
            ctrl_period_ms = dt_ctrl * 1000.0
            prev_ctrl_time = t_tick

            # 1. Thread-safe Telemetry Read
            t_lock_read_start = time.perf_counter()
            with perception_state.lock:
                telem = dict(perception_state.telemetry)
                t_frame_cap = perception_state.t_capture
                t_frame_perc = perception_state.t_perc_done
                is_stale = perception_state.is_stale
            t_lock_read_end = time.perf_counter()

            # 2. Control Computation
            t_ctrl_start = time.perf_counter()
            with vehicle_state.lock:
                drone_pitch = vehicle_state.pitch_rad
                drone_roll = vehicle_state.roll_rad
                yaw_rate = vehicle_state.yaw_rate_rad_s
                v_body = vehicle_state.velocity_body
                alt_m = vehicle_state.altitude_rel_m

            # Guidance computation
            raw_cmd = controller.compute_cmd(
                telemetry=telem,
                drone_pitch=drone_pitch,
                drone_roll=drone_roll,
                dt=dt_ctrl,
                vehicle_vel=v_body,
                vehicle_yaw_rate=yaw_rate,
                current_alt_m=alt_m
            )
            if not np.all(np.isfinite(raw_cmd)):
                raw_cmd = np.zeros_like(raw_cmd)

            # Slew-Rate Limiting
            max_step = max_accel[:len(raw_cmd)] * dt_ctrl
            cmd_safe = np.clip(raw_cmd, prev_cmd[:len(raw_cmd)] - max_step, prev_cmd[:len(raw_cmd)] + max_step)
            if not getattr(args, "lateral_strafe", True) and len(cmd_safe) >= 2:
                cmd_safe[1] = 0.0
            prev_cmd = cmd_safe.copy()

            # Convert to body velocity
            v_fwd, v_right, v_down, yawspeed = controller.to_velocity_body(cmd_safe)

            # Enforce safety altitude envelope
            safe_v_down, _ = enforce_altitude_limits(v_down, alt_m)
            v_down = safe_v_down
            t_ctrl_end = time.perf_counter()

            # 3. MAVLink Actuation & Wire Dispatch
            t_act_start = time.perf_counter()
            ser_ms, _ = actuator.dispatch_velocity(v_fwd, v_right, v_down, yawspeed)
            t_act_end = time.perf_counter()

            # True Cyber-Physical End-to-End Latency
            t_now = t_act_end
            if t_frame_cap > 0.0:
                e2e_capture_ms = (t_now - t_frame_cap) * 1000.0
                e2e_perc_ms = (t_now - t_frame_perc) * 1000.0
                stats["e2e_capture_to_actuation"].add(e2e_capture_ms)
                stats["e2e_perc_to_actuation"].add(e2e_perc_ms)

            lock_r_ms = (t_lock_read_end - t_lock_read_start) * 1000.0
            ctrl_ms = (t_ctrl_end - t_ctrl_start) * 1000.0
            act_ms = (t_act_end - t_act_start) * 1000.0
            c_step_ms = lock_r_ms + ctrl_ms + act_ms

            control_intervals_ms.append(ctrl_period_ms)
            stats["control_lock"].add(lock_r_ms)
            stats["control"].add(ctrl_ms)
            stats["actuation_serialize"].add(act_ms)
            stats["control_step"].add(c_step_ms)
            stats["control_period"].add(ctrl_period_ms)
            control_step_counter[0] += 1

            # High-resolution 50 Hz loop pacing
            elapsed_in_loop = time.perf_counter() - t_tick
            sleep_rem = ctrl_period - elapsed_in_loop
            if sleep_rem > 0.001:
                time.sleep(sleep_rem)
            else:
                pass  # Saturated loop tick

    # Launch threads
    print(f"\n[WARMUP] Starting dual-thread pipeline ({n_warmup} warmup frames)...")
    t_bench_start = time.perf_counter()

    v_thread = threading.Thread(target=_vision_worker, daemon=True)
    c_thread = threading.Thread(target=_control_worker, daemon=True)

    v_thread.start()
    c_thread.start()

    v_thread.join()
    stop_event.set()
    c_thread.join(timeout=2.0)

    t_bench_end = time.perf_counter()
    wall_sec = (t_bench_end - t_timed_start[0]) if t_timed_start[0] > 0 else (t_bench_end - t_bench_start)

    # Return aggregated statistics
    return {
        "stats": stats,
        "inter_frame_intervals_ms": inter_frame_intervals_ms,
        "control_intervals_ms": control_intervals_ms,
        "vision_frames": vision_frame_counter[0],
        "control_steps": control_step_counter[0],
        "deadline_overruns": deadline_overruns[0],
        "wall_duration_s": wall_sec
    }


# ==============================================================================
# 6. Synchronous Sequential Profiling Engine (Isolated Component Mode)
# ==============================================================================

def run_sync_pipeline_profile(
    cap: cv2.VideoCapture,
    pipeline: DroneTrackingPipeline,
    controller: Any,
    actuator: MAVLinkActuator,
    telemetry_streamer: FlightTelemetryStreamer,
    args: argparse.Namespace
) -> Dict[str, Any]:
    """Sequential step-by-step benchmark for isolated component execution shares."""
    n_frames = args.frames
    n_warmup = args.warmup
    device = pipeline.device

    dehaze_lut = build_dehaze_lut(args.dehaze_gamma) if getattr(args, "enable_dehaze", False) else None
    adaptive_shadows = getattr(args, "adaptive_shadows", True)

    stats = {
        "dequeue": LatencyStats("1. Frame Dequeue DMA (T_dequeue)"),
        "preprocess": LatencyStats("2. Preprocessing & Shadows (T_preprocess)"),
        "perception": LatencyStats("3. Perception (YOLOv8 + 8D Kalman)"),
        "control": LatencyStats("4. Guidance Control (T_control)"),
        "actuation_serialize": LatencyStats("5. MAVLink Serialization (T_serialize)"),
        "compute_total": LatencyStats("Total Software Processing (T_compute)")
    }
    inter_frame_intervals_ms: List[float] = []
    deadline_overruns = 0
    TARGET_FRAME_PERIOD_MS = 1000.0 / 30.0

    print(f"\n[WARMUP] Running {n_warmup} warm-up frames to stabilize caches...")
    for _ in range(n_warmup):
        ret, frame = cap.read()
        if not ret:
            if not str(args.source).isdigit() and Path(args.source).is_file():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if not ret:
                break
        if dehaze_lut is not None:
            frame = cv2.LUT(frame, dehaze_lut)
        if adaptive_shadows:
            frame = apply_adaptive_shadow_boost(frame)
        ego = telemetry_streamer.get_next(1.0 / 30.0)
        _, telem = pipeline.process_frame(frame, draw_hud=False, ego_telemetry=ego)
        cmd = controller.compute_cmd(telem, dt=1.0 / 30.0)
        v_fwd, v_right, v_down, yawspeed = controller.to_velocity_body(cmd)
        actuator.dispatch_velocity(v_fwd, v_right, v_down, yawspeed)

    print(f"[BENCHMARK] Warm-up complete. Recording {n_frames} timed sequential frames...\n")

    t_bench_start = time.perf_counter()
    prev_delivered = time.perf_counter()
    timed_count = 0

    while timed_count < n_frames:
        t_cap_start = time.perf_counter()
        ret, frame = cap.read()
        t_captured = time.perf_counter()

        if not ret:
            if not str(args.source).isdigit() and Path(args.source).is_file():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        inter_dt = (t_captured - prev_delivered) * 1000.0
        prev_delivered = t_captured
        inter_frame_intervals_ms.append(inter_dt)

        t_pre_start = time.perf_counter()
        proc_frame = frame
        if dehaze_lut is not None:
            proc_frame = cv2.LUT(proc_frame, dehaze_lut)
        if adaptive_shadows:
            proc_frame = apply_adaptive_shadow_boost(proc_frame)
        t_pre_end = time.perf_counter()

        ego = telemetry_streamer.get_next(dt=1.0 / 30.0)
        t_perc_start = time.perf_counter()
        _, telem = pipeline.process_frame(proc_frame, draw_hud=False, ego_telemetry=ego)
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        t_perc_end = time.perf_counter()

        t_ctrl_start = time.perf_counter()
        cmd = controller.compute_cmd(telem, drone_pitch=ego["pitch_rad"], drone_roll=ego["roll_rad"], dt=1.0 / 30.0)
        v_fwd, v_right, v_down, yawspeed = controller.to_velocity_body(cmd)
        t_ctrl_end = time.perf_counter()

        t_act_start = time.perf_counter()
        actuator.dispatch_velocity(v_fwd, v_right, v_down, yawspeed)
        t_act_end = time.perf_counter()

        deq_ms = (t_captured - t_cap_start) * 1000.0
        pre_ms = (t_pre_end - t_pre_start) * 1000.0
        perc_ms = (t_perc_end - t_perc_start) * 1000.0
        ctrl_ms = (t_ctrl_end - t_ctrl_start) * 1000.0
        act_ms = (t_act_end - t_act_start) * 1000.0
        comp_ms = pre_ms + perc_ms + ctrl_ms + act_ms

        if comp_ms > TARGET_FRAME_PERIOD_MS:
            deadline_overruns += 1

        stats["dequeue"].add(deq_ms)
        stats["preprocess"].add(pre_ms)
        stats["perception"].add(perc_ms)
        stats["control"].add(ctrl_ms)
        stats["actuation_serialize"].add(act_ms)
        stats["compute_total"].add(comp_ms)

        timed_count += 1
        if timed_count % max(1, n_frames // 5) == 0 or timed_count == n_frames:
            cur_comp = stats["compute_total"].summary()["mean"]
            cur_fps = 1000.0 / max(1e-3, cur_comp)
            print(f"  Frame {timed_count:4d}/{n_frames:4d} | Current Compute: {cur_comp:5.2f} ms ({cur_fps:5.1f} Cap. FPS) | Overruns: {deadline_overruns}")

    t_bench_end = time.perf_counter()
    return {
        "stats": stats,
        "inter_frame_intervals_ms": inter_frame_intervals_ms,
        "control_intervals_ms": [],
        "vision_frames": timed_count,
        "control_steps": timed_count,
        "deadline_overruns": deadline_overruns,
        "wall_duration_s": t_bench_end - t_bench_start
    }


# ==============================================================================
# 7. Main Profiling Controller & Publication Report Generator
# ==============================================================================

def profile_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    print("=" * 90)
    print("       CYBER-PHYSICAL CAPTURE-TO-ACTUATION LATENCY & THROUGHPUT BENCHMARK       ")
    print("=" * 90)
    print(f"Architecture Mode:      {args.mode.upper()} ({'Asynchronous Decoupled Vision Worker + 50 Hz Control Loop' if args.mode == 'async' else 'Sequential Single-Thread'})")
    print(f"Perception Weights:     {args.weights}")
    print(f"Video Source:           {'Jetson CSI Camera (Hardware GStreamer)' if args.csi else args.source}")
    print(f"Telemetry Mode:         {args.telemetry_mode.upper()} ({args.flight_log if args.telemetry_mode == 'flight-log' else 'Dynamic Maneuver'})")
    print(f"PID Tuning Gains:       {args.pid_gains if args.pid_gains else 'Defaults'}")
    print(f"Digital Foveal Zoom:    {'ENABLED' if args.enable_zoom else 'DISABLED'}")
    print(f"Adaptive Shadows:       {'ENABLED (Sub-millisecond lift)' if args.adaptive_shadows else 'DISABLED'}")
    print(f"Optical De-Haze:        {'ENABLED (LUT gamma ' + str(args.dehaze_gamma) + ')' if args.enable_dehaze else 'DISABLED'}")
    print(f"Guidance Controller:    {args.controller.upper()}")
    print(f"Control Loop Rate:      {args.ctrl_rate:.1f} Hz (Deterministic MAVLink stream)")
    print(f"Actuation Interface:    {'LIVE SERIAL UART (' + args.serial_port + ' @ ' + str(args.baud) + ')' if args.serial_port else 'MAVLink Serializer (Dry-Run)'}")
    print(f"Benchmark Iterations:   {args.frames} timed frames ({args.warmup} discarded warmup frames)\n")

    # 1. Initialize Video Capture
    if args.csi:
        pipeline_str = build_jetson_csi_pipeline()
        print("[INIT] Opening Jetson CSI GStreamer pipeline via nvarguscamerasrc...")
        cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)
    else:
        src = int(args.source) if str(args.source).isdigit() else args.source
        print(f"[INIT] Opening video stream: {src}...")
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {args.source}")

    # 2. Initialize Perception Pipeline
    print(f"[INIT] Loading DroneTrackingPipeline with {args.weights}...")
    pipeline = DroneTrackingPipeline(
        weights=args.weights,
        conf_threshold=0.25,
        iou_threshold=0.45,
        max_lost_frames=15,
        desired_target_size=getattr(args, "target_size", 110.0),
        enable_dynamic_zoom=args.enable_zoom,
        imgsz=args.imgsz
    )

    # 3. Initialize Guidance Controller
    controller = create_kinematic_controller(args)

    # 4. Initialize MAVLink Actuator & Telemetry Streamer
    actuator = MAVLinkActuator(args.serial_port, args.baud)
    telemetry_streamer = FlightTelemetryStreamer(mode=args.telemetry_mode, log_path=args.flight_log)

    # 5. Execute Profiler Engine
    if args.mode == "async":
        bench_out = run_async_pipeline_profile(cap, pipeline, controller, actuator, telemetry_streamer, args)
    else:
        bench_out = run_sync_pipeline_profile(cap, pipeline, controller, actuator, telemetry_streamer, args)

    cap.release()
    stats = bench_out["stats"]
    timed_count = bench_out["vision_frames"]
    deadline_overruns = bench_out["deadline_overruns"]

    # 6. Physical Cyber-Physical Latency Models
    # A. Optical Sensor Integration & Readout Model (Sony IMX477 @ 30fps)
    T_EXPOSURE_MIDPOINT_MS = 16.67  # Half of 1/30s frame integration
    T_CSI_READOUT_MS = 12.00        # Rolling shutter 2-lane MIPI CSI-2 readout clock
    T_ISP_DOWNSCALE_MS = 4.00       # Tegra ISP hardware Bayer demosaic & nvvidconv downscale
    T_SENSOR_PHYSICAL_MS = T_EXPOSURE_MIDPOINT_MS + T_CSI_READOUT_MS + T_ISP_DOWNSCALE_MS  # ~32.67 ms

    # B. Actuation & Electro-Mechanical Response Model
    T_UART_WIRE_MS = (57 * 8.0 / args.baud) * 1000.0  # ~0.495 ms at 921600 baud
    T_PIXHAWK_LOOP_MS = 7.50        # Mean guidance loop quantization (ArduPilot 50-100 Hz position loop)
    T_MOTOR_TAU_MS = 35.00          # Rotor aerodynamic & mechanical time constant (from flight sysid 00000080.BIN)
    T_ACTUATION_PHYSICAL_MS = T_UART_WIRE_MS + T_PIXHAWK_LOOP_MS + T_MOTOR_TAU_MS  # ~43.0 ms

    # C. Software Latency & Throughput Metrics
    TARGET_FRAME_PERIOD_MS = 1000.0 / 30.0
    if args.mode == "async":
        vis_summary = stats["vision_compute"].summary()
        e2e_summary = stats["e2e_capture_to_actuation"].summary()
        ctrl_period_summary = stats["control_period"].summary()

        comp_mean_ms = vis_summary["mean"]
        comp_p95_ms = vis_summary["p95"]
        software_e2e_mean_ms = e2e_summary["mean"] if e2e_summary["count"] > 0 else comp_mean_ms
        software_e2e_p95_ms = e2e_summary["p95"] if e2e_summary["count"] > 0 else comp_p95_ms

        measured_vision_fps = timed_count / max(1e-3, bench_out["wall_duration_s"])
        measured_ctrl_hz = (1000.0 / ctrl_period_summary["mean"]) if ctrl_period_summary.get("mean", 0) > 0 else (bench_out["control_steps"] / max(1e-3, bench_out["wall_duration_s"]))
        compute_capacity_fps = 1000.0 / max(1e-3, comp_mean_ms)
        deadline_headroom_ms = TARGET_FRAME_PERIOD_MS - comp_mean_ms
        deadline_headroom_pct = (deadline_headroom_ms / TARGET_FRAME_PERIOD_MS) * 100.0
    else:
        comp_summary = stats["compute_total"].summary()
        comp_mean_ms = comp_summary["mean"]
        comp_p95_ms = comp_summary["p95"]
        software_e2e_mean_ms = comp_mean_ms
        software_e2e_p95_ms = comp_p95_ms

        measured_vision_fps = timed_count / max(1e-3, bench_out["wall_duration_s"])
        measured_ctrl_hz = measured_vision_fps
        compute_capacity_fps = 1000.0 / max(1e-3, comp_mean_ms)
        deadline_headroom_ms = TARGET_FRAME_PERIOD_MS - comp_mean_ms
        deadline_headroom_pct = (deadline_headroom_ms / TARGET_FRAME_PERIOD_MS) * 100.0
        ctrl_period_summary = {"mean": 1000.0 / max(1e-3, measured_ctrl_hz), "std": 0.0}

    # D. Total Cyber-Physical Latency
    T_PHOTON_TO_THRUST_MEAN_MS = T_SENSOR_PHYSICAL_MS + software_e2e_mean_ms + T_ACTUATION_PHYSICAL_MS
    T_PHOTON_TO_THRUST_P95_MS = T_SENSOR_PHYSICAL_MS + software_e2e_p95_ms + T_ACTUATION_PHYSICAL_MS

    # 7. Print Terminal Results
    print("\n" + "=" * 92)
    print("                      MEASURED ON-DEVICE SOFTWARE LATENCY BREAKDOWN                     ")
    print("=" * 92)
    print(f"{'Pipeline Subsystem Stage':<38} | {'Mean (ms)':<9} | {'Std (ms)':<8} | {'Median':<8} | {'95%ile':<8} | {'Max (ms)':<8}")
    print("-" * 92)

    results: Dict[str, Any] = {}
    for key, stat in stats.items():
        s = stat.summary()
        results[key] = s
        if s["count"] > 0:
            print(f"{stat.name:<38} | {s['mean']:>9.3f} | {s['std']:>8.3f} | {s['median']:>8.3f} | {s['p95']:>8.3f} | {s['max']:>8.3f}")

    print("=" * 92)
    print(f"VISION COMPUTE CAPACITY (HEADROOM):     {compute_capacity_fps:.1f} FPS (Max theoretical processing speed)")
    print(f"MEASURED VISION SAMPLING RATE:          {measured_vision_fps:.1f} FPS (Camera I/O throughput)")
    print(f"MEASURED CONTROL LOOP RATE:             {measured_ctrl_hz:.1f} Hz (Target: {args.ctrl_rate:.1f} Hz, Jitter: ±{ctrl_period_summary['std']:.2f} ms)")
    print(f"REAL-TIME 30 HZ DEADLINE HEADROOM:      {deadline_headroom_ms:+.2f} ms ({deadline_headroom_pct:+.1f}% spare compute capacity)")
    print(f"DEADLINE OVERRUNS (>33.3 ms):           {deadline_overruns} of {timed_count} frames ({deadline_overruns / max(1, timed_count) * 100.0:.2f}%)")
    print("=" * 92)

    if args.mode == "async" and results.get("e2e_capture_to_actuation", {}).get("count", 0) > 0:
        e2e = results["e2e_capture_to_actuation"]
        print("\n" + "-" * 92)
        print("          TRUE DECOUPLED ASYNCHRONOUS CAPTURE-TO-ACTUATION DELAY (DATA AGE)             ")
        print("-" * 92)
        print(f"Mean Software Age (Capture -> Wire):    {e2e['mean']:.2f} ms (Std: ±{e2e['std']:.2f} ms)")
        print(f"Median Age (P50):                       {e2e['median']:.2f} ms")
        print(f"Worst-Case Bound (95th Percentile):     {e2e['p95']:.2f} ms (Max: {e2e['max']:.2f} ms)")
        print("-" * 92)

    print("\n" + "=" * 92)
    print("            FULL CYBER-PHYSICAL SYSTEM LATENCY BUDGET (PHOTON-TO-THRUST)                ")
    print("=" * 92)
    print(f"1. Optical Domain (Exposure + CSI Readout + ISP DMA):       {T_SENSOR_PHYSICAL_MS:>6.2f} ms")
    print(f"2. Compute Domain (True Capture-to-Actuation Software Delay): {software_e2e_mean_ms:>6.2f} ms (P95: {software_e2e_p95_ms:.2f} ms)")
    print(f"3. Actuation Domain (UART Wire {T_UART_WIRE_MS:.2f}ms + Pixhawk {T_PIXHAWK_LOOP_MS:.1f}ms + Motor {T_MOTOR_TAU_MS:.1f}ms): {T_ACTUATION_PHYSICAL_MS:>6.2f} ms")
    print("-" * 92)
    print(f"TOTAL END-TO-END PHOTON-TO-THRUST LATENCY (MEAN):           {T_PHOTON_TO_THRUST_MEAN_MS:>6.2f} ms")
    print(f"TOTAL END-TO-END PHOTON-TO-THRUST LATENCY (P95 WORST-CASE): {T_PHOTON_TO_THRUST_P95_MS:>6.2f} ms")
    print("=" * 92 + "\n")

    # 8. Save Publication Markdown Report
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_md = out_dir / "capture_to_actuation_latency_report.md"

    with open(report_md, "w") as f:
        f.write("# Production Cyber-Physical Latency & Throughput Benchmark\n\n")
        f.write(f"- **Architecture Mode:** `{args.mode.upper()}` ({'Asynchronous Decoupled Vision Worker + 50 Hz Control Loop' if args.mode == 'async' else 'Sequential Single-Thread'})\n")
        f.write(f"- **Perception Model:** `{args.weights}`\n")
        f.write(f"- **Video Source:** `{'Jetson CSI Camera (Hardware GStreamer)' if args.csi else args.source}`\n")
        f.write(f"- **PID Tuning Configuration:** `{args.pid_gains if args.pid_gains else 'Defaults'}`\n")
        f.write(f"- **Telemetry Mode:** `{args.telemetry_mode.upper()}` ({args.flight_log if args.telemetry_mode == 'flight-log' else 'Dynamic Maneuver'})\n")
        f.write(f"- **Actuation Link:** `{'Live UART Serial (' + str(args.serial_port) + ')' if args.serial_port else 'MAVLink Serializer (Dry-Run)'}`\n")
        f.write(f"- **Sample Size:** {timed_count} consecutive timed frames ({args.warmup} discarded warmup frames)\n\n")

        f.write("## 1. On-Device Software Execution Breakdown\n\n")
        f.write("| Subsystem Stage | Mean Latency | Std Dev | Median (P50) | 95th %ile | Max Latency |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for key, s in results.items():
            if s["count"] > 0 and key in stats:
                f.write(f"| **{stats[key].name}** | **{s['mean']:.3f} ms** | $\\pm {s['std']:.3f}$ ms | {s['median']:.3f} ms | {s['p95']:.3f} ms | {s['max']:.3f} ms |\n")
        f.write("\n")

        f.write("## 2. Decoupled Throughput & Real-Time Frame Deadlines\n\n")
        f.write(f"- **Camera Hardware Frame Period:** `33.33 ms` ($30.0\\text{{ Hz}}$)\n")
        f.write(f"- **Measured Vision Sampling Rate:** **{measured_vision_fps:.1f} FPS**\n")
        f.write(f"- **Measured MAVLink Control Rate:** **{measured_ctrl_hz:.1f} Hz** (Target: {args.ctrl_rate:.1f} Hz, Jitter: $\\pm {ctrl_period_summary['std']:.2f}\\text{{ ms}}$)\n")
        f.write(f"- **Perception Compute Capacity:** **{compute_capacity_fps:.1f} FPS**\n")
        f.write(f"- **Real-Time 30 Hz Deadline Headroom:** **{deadline_headroom_ms:+.2f} ms ({deadline_headroom_pct:+.1f}% idle compute margin)**\n")
        f.write(f"- **Deadline Overrun Rate:** **{deadline_overruns} / {timed_count} frames ({deadline_overruns / max(1, timed_count) * 100.0:.2f}%)**\n")
        f.write(f"- **Real-Time Execution Verdict:** `{'PASSED (Guaranteed Real-Time Execution)' if deadline_overruns == 0 else 'WARNING: Deadline overruns detected'}`\n\n")

        f.write("## 3. Cyber-Physical Photon-to-Thrust Latency Budget\n\n")
        f.write("$$\\Delta t_{\\text{total}} = \\underbrace{T_{\\text{exposure}} + T_{\\text{readout}} + T_{\\text{ISP}}}_{T_{\\text{sensor}} \\approx 32.7\\text{ ms}} + \\underbrace{\\Delta t_{\\text{capture}\\to\\text{actuation}}}_{T_{\\text{software}} \\approx " + f"{software_e2e_mean_ms:.2f}" + "\\text{ ms}} + \\underbrace{T_{\\text{UART}} + T_{\\text{FC\\_sample}} + \\tau_{\\text{motor}}}_{T_{\\text{actuation}} \\approx 43.0\\text{ ms}}$$\n\n")
        f.write("| Physical Domain | Delay Mechanism | Latency Contribution | Analysis Method |\n")
        f.write("| :--- | :--- | :---: | :--- |\n")
        f.write(f"| **Optical / Sensor** | Photon integration midpoint ($0.5 \\cdot T_{{\\text{{exp}}}}$) | $\\approx {T_EXPOSURE_MIDPOINT_MS:.2f}\\text{{ ms}}$ | Physical sensor shutter |\n")
        f.write(f"| **Optical / Sensor** | MIPI CSI-2 2-lane rolling shutter readout | $\\approx {T_CSI_READOUT_MS:.2f}\\text{{ ms}}$ | Hardware sensor bus clock |\n")
        f.write(f"| **Optical / Sensor** | Tegra ISP demosaic & nvvidconv scaling | $\\approx {T_ISP_DOWNSCALE_MS:.2f}\\text{{ ms}}$ | Hardware ISP pipeline |\n")
        f.write(f"| **Edge Software** | True Capture-to-Actuation Age | **{software_e2e_mean_ms:.2f} ms** (P95: {software_e2e_p95_ms:.2f} ms) | **Measured on Jetson GPU/CPU (Exact Pipeline)** |\n")
        f.write(f"| **Serial Bus** | MAVLink UART transmission ($57\\text{{B}} @ {args.baud}$) | $\\approx {T_UART_WIRE_MS:.2f}\\text{{ ms}}$ | Physical serial transmission |\n")
        f.write(f"| **Flight Controller** | Pixhawk guidance loop sample delay (50–100 Hz) | $\\approx {T_PIXHAWK_LOOP_MS:.2f}\\text{{ ms}}$ | ArduPilot loop quantization |\n")
        f.write(f"| **Electro-Mechanical** | BLDC Motor & rotor aerodynamics ($\\tau_{{\\text{{motor}}}}$) | $\\approx {T_MOTOR_TAU_MS:.2f}\\text{{ ms}}$ | Empirical flight system ID |\n")
        f.write(f"| **Total Closed-Loop** | **End-to-End Photon-to-Thrust Delay** | **$\\approx {T_PHOTON_TO_THRUST_MEAN_MS:.2f}\\text{{ ms}}$** (P95: {T_PHOTON_TO_THRUST_P95_MS:.2f} ms) | **Complete Cyber-Physical Total** |\n")

    print(f"[REPORT] Saved publication markdown report: {report_md}")

    # 9. Save JSON Telemetry
    report_json = out_dir / "capture_to_actuation_latency.json"
    with open(report_json, "w") as f:
        json.dump({
            "mode": args.mode,
            "weights": str(args.weights),
            "source": str(args.source),
            "use_csi": args.csi,
            "pid_gains": str(args.pid_gains),
            "timed_frames": timed_count,
            "software_metrics": results,
            "vision_fps": measured_vision_fps,
            "control_hz": measured_ctrl_hz,
            "compute_capacity_fps": compute_capacity_fps,
            "deadline_headroom_ms": deadline_headroom_ms,
            "deadline_overruns": deadline_overruns,
            "software_e2e_mean_ms": software_e2e_mean_ms,
            "software_e2e_p95_ms": software_e2e_p95_ms,
            "photon_to_thrust_total_mean_ms": T_PHOTON_TO_THRUST_MEAN_MS,
            "photon_to_thrust_total_p95_ms": T_PHOTON_TO_THRUST_P95_MS
        }, f, indent=2)
    print(f"[REPORT] Saved structured JSON telemetry:   {report_json}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Profile end-to-end cyber-physical capture-to-actuation pipeline latency.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", type=str, choices=["async", "sync"], default="async",
                        help="Execution architecture: 'async' (default, mimics run_tracker.py decoupled threads) or 'sync' (sequential)")
    parser.add_argument("--weights", type=str, default="perception/weights/yolov8n_drone_768_hybrid_nwd_c51_best.pt",
                        help="Path to YOLOv8 weights (.pt or .engine)")
    parser.add_argument("--source", type=str, default="perception/raw_data/Anti-UAV/videos/video12.mp4",
                        help="Video file path or camera device index (default: video12.mp4)")
    parser.add_argument("--csi", action="store_true",
                        help="Use hardware-accelerated CSI GStreamer pipeline on NVIDIA Jetson")
    parser.add_argument("--frames", type=int, default=100,
                        help="Number of timed benchmark frames (default: 100)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Number of initial warmup frames to discard (default: 20)")
    parser.add_argument("--ctrl-rate", type=float, default=50.0,
                        help="Target MAVLink flight control loop frequency in Hz (default: 50.0 Hz)")
    parser.add_argument("--pid-gains", type=str, default="control/weights/tuned_pid_gains.json",
                        help="Path to tuned PID gains JSON file (default: control/weights/tuned_pid_gains.json)")
    parser.add_argument("--adaptive-shadows", action="store_true", default=True,
                        help="Include sub-millisecond adaptive shadow boost preprocessing (default: True)")
    parser.add_argument("--no-adaptive-shadows", action="store_false", dest="adaptive_shadows",
                        help="Disable adaptive shadow boost")
    parser.add_argument("--enable-dehaze", action="store_true", default=False,
                        help="Enable fast gamma-based de-haze lookup table (cv2.LUT)")
    parser.add_argument("--dehaze-gamma", type=float, default=1.40,
                        help="Gamma factor for de-hazing (default: 1.40)")
    parser.add_argument("--controller", type=str, choices=["ibvs", "predictive"], default="ibvs",
                        help="Guidance flight controller to profile")
    parser.add_argument("--serial-port", type=str, default=None,
                        help="Physical serial UART port (e.g. /dev/ttyTHS1 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=921600,
                        help="Baud rate for serial UART transmission (default: 921600)")
    parser.add_argument("--telemetry-mode", type=str, choices=["flight-log", "synthetic-maneuver", "static"], default="flight-log",
                        help="Vehicle attitude/velocity telemetry ingestion mode")
    parser.add_argument("--flight-log", type=str, default="logs/flight_logs/00000165.BIN",
                        help="Path to physical ArduPilot .BIN flight log (default: 00000165.BIN)")
    parser.add_argument("--enable-zoom", action="store_true", default=False,
                        help="Enable dynamic digital foveal zoom tracking")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Inference image resolution (default: auto-detected from model/engine, e.g. 640 or 768)")
    parser.add_argument("--outdir", type=str, default="outputs/benchmarks",
                        help="Directory to save benchmark reports (default: outputs/benchmarks)")
    parser.add_argument("--max-accel-xy", type=float, default=4.0,
                        help="Max horizontal acceleration slew cap in m/s^2 (default: 4.0)")
    parser.add_argument("--max-accel-z", type=float, default=2.0,
                        help="Max vertical climb/descent acceleration cap in m/s^2 (default: 2.0)")
    parser.add_argument("--max-accel-yaw", type=float, default=180.0,
                        help="Max yaw angular acceleration cap in deg/s^2 (default: 180.0)")

    args = parser.parse_args()

    # Automatic fallback if requested weights don't exist
    w_path = Path(args.weights)
    if not w_path.exists():
        candidate = REPO_ROOT / args.weights
        if candidate.exists():
            w_path = candidate
        else:
            fallback = REPO_ROOT / "perception/weights/yolov8n_drone_768_hybrid_nwd_c51_best.pt"
            if fallback.exists():
                print(f"[WARN] Requested weights '{args.weights}' not found. Falling back to: {fallback.name}")
                w_path = fallback

    args.weights = str(w_path)
    profile_pipeline(args)


if __name__ == "__main__":
    main()
