#!/usr/bin/env python3
"""
Autonomous Drone Pursuit Flight Bridge Node.
Interfaces the YOLOv8 + Kalman perception pipeline with the Kinematic IBVS Controller
and dispatches real-time velocity setpoints to Pixhawk Autopilot via MAVSDK / MAVLink.

Features:
- Dual execution: Live flight offboard control or Benchtop Dry-Run (desktop test mode)
- Dynamic attitude streaming: Ingests real-time EKF2 pitch for camera de-rotation
- Metric-free visual servoing: Regulates 2D bounding box pixel size directly
- Flight safety: Graceful failsafe handling, velocity slew-rate limiting, and clean emergency disengage
"""

import os
import sys
import time
import argparse
import asyncio
import threading
from pathlib import Path
from collections import deque
from typing import Optional, Dict, Any, Tuple, Union
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np


def build_jetson_csi_pipeline(
    sensor_id: int = 0,
    capture_width: int = 3840,
    capture_height: int = 2160,
    framerate: int = 30,
    flip_method: int = 0,
    display_width: int = 1280,
    display_height: int = 720,
    ee_strength: float = 1.0,
) -> str:
    """
    Builds hardware-accelerated GStreamer pipeline for NVIDIA Jetson CSI cameras.
    Locks into the sensor's exact native hardware mode (3840x2160 @ 30fps or 1920x1080 @ 60fps),
    activates Tegra ISP Edge Enhancement (sharpening), and supersamples down to display resolution
    using nvvidconv for pin-sharp clarity.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ee-mode=2 ee-strength={ee_strength} tnr-mode=1 tnr-strength=0.0 ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink drop=true sync=false"
    )


def is_zero_yuv_green_frame(frame: np.ndarray) -> bool:
    """
    Detects if a frame is an unpopulated/errored zero-byte YUV buffer (RGB: ~0, ~141, ~0).
    On Jetson Linux for Tegra, reading a CSI camera via plain V4L2 without ISP or a saturated
    USB bus returns zeroed memory, which decodes to solid green.
    """
    if frame is None or frame.size == 0:
        return True
    b_max = int(frame[..., 0].max())
    r_max = int(frame[..., 2].max())
    g_mean = float(frame[..., 1].mean())
    return (b_max == 0 and r_max == 0 and 120 <= g_mean <= 160)


# ==============================================================================
# Module Path Resolution (Allows clean imports outside subdirectories)
# ==============================================================================
ROOT_DIR = Path(__file__).resolve().parent
CONTROL_DIR = ROOT_DIR / "control"
DETECTION_DIR = ROOT_DIR / "perception"

for p in (ROOT_DIR, CONTROL_DIR, DETECTION_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from perception.pipeline import DroneTrackingPipeline
from control.pid_controller import KinematicVisualServoController

# Conditional MAVSDK Import (enables desktop testing without MAVSDK installed or on linker failure)
try:
    from mavsdk import System
    from mavsdk.offboard import VelocityBodyYawspeed, OffboardError
    MAVSDK_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Could not initialize MAVSDK ({e}). Falling back to DRY-RUN mode.")
    MAVSDK_AVAILABLE = False


# ==============================================================================
# Telemetry & Vehicle State Container
# ==============================================================================
class VehicleState:
    """Thread-safe container for asynchronous Pixhawk telemetry."""
    def __init__(self):
        self.pitch_rad: float = 0.0
        self.roll_rad: float = 0.0
        self.yaw_deg: float = 0.0
        self.vx_m_s: float = 0.0               # North (NED) velocity from EKF2
        self.vy_m_s: float = 0.0               # East (NED) velocity from EKF2
        self.vz_m_s: float = 0.0               # Down (NED) vertical velocity (baro+accel)
        self.has_gps_vel: bool = False         # True when horizontal EKF velocity is actively aiding
        self.altitude_rel_m: float = 0.0       # Relative to takeoff/home (EKF2)
        self.altitude_amsl_m: float = 0.0      # Absolute altitude above MSL
        self.distance_sensor_m: Optional[float] = None  # Downward lidar/sonar rangefinder
        self.is_connected: bool = False
        self.is_armed: bool = False
        self.in_air: bool = False
        self.flight_mode: str = "DISCONNECTED"
        self.battery_pct: float = 100.0
        self.alt_safety_status: str = "OK"     # OK, FLOOR_CUSHION, FLOOR_LIMIT, FLOOR_BREACH, etc.


class PerceptionState:
    """Thread-safe container holding latest perception telemetry, frames, and timings."""
    def __init__(self):
        self.lock = threading.Lock()
        self.telemetry: Dict[str, Any] = {"status": "SEARCHING"}
        self.latest_frame: Optional[np.ndarray] = None
        self.last_update_time: float = 0.0
        self.frame_idx: int = 0
        self.fps: float = 0.0
        self.is_stale: bool = True
        self.has_new_frame: bool = False


# ==============================================================================
# Master Autonomous Pursuit Bridge
# ==============================================================================
class AutonomousTrackerNode:
    """
    Coordinates Camera Ingestion, YOLO/Kalman Perception, Kinematic IBVS Control,
    and MAVSDK Pixhawk Communication.
    """
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.vehicle_state = VehicleState()
        self.running = False
        self.drone: Optional[Any] = None

        # 1. Initialize Kinematic Visual Servoing Controller
        print("\n[INIT] Initializing Kinematic Visual Servoing Controller...")
        self.controller = KinematicVisualServoController(
            camera_uptilt_deg=args.uptilt,
            hfov_deg=args.hfov,
            vfov_deg=args.vfov,
            desired_bbox_size=args.target_size,
            min_limits=np.array([0.0, -args.max_climb, -args.max_yawspeed]),
            max_limits=np.array([args.max_speed, args.max_desc, args.max_yawspeed]),
            use_bbox_size=True,
            enable_lateral_strafe=args.lateral_strafe,
            max_lat_vel=args.max_lat_speed
        )

        # 2. Initialize Perception Pipeline (YOLOv8 + 8D Kalman Filter)
        print(f"[INIT] Initializing Perception Pipeline with model: {args.weights}")
        self.pipeline = DroneTrackingPipeline(
            weights=args.weights,
            conf_threshold=args.conf,
            iou_threshold=0.45,
            max_lost_frames=args.max_lost_frames,
            desired_target_size=args.target_size,
            enable_dynamic_zoom=False  # Full sensor frame for flight control
        )

        # 3. Decoupled Threading & State Containers
        self.perception_state = PerceptionState()
        self.latest_cmd_safe = np.zeros(4, dtype=np.float64)
        self.vision_thread: Optional[threading.Thread] = None

        # 4. Video Recording Setup (HUD & Bounding Boxes)
        self.record_path: Optional[Path] = None
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.record_fps: float = float(getattr(args, "record_fps", 30.0))
        if getattr(args, "record", None):
            if args.record == "auto":
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                rec_dir = ROOT_DIR / "outputs" / "recordings"
                rec_dir.mkdir(parents=True, exist_ok=True)
                self.record_path = rec_dir / f"flight_tracking_{timestamp}.mp4"
            else:
                self.record_path = Path(args.record)
                self.record_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[RECORD] Target video recording destination: {self.record_path}")

        # 5. Live MJPEG Web Streamer Setup (Browser / SSH Port Forwarding)
        self.stream_server: Optional[ThreadingHTTPServer] = None
        self.latest_jpeg: Optional[bytes] = None
        self.jpeg_lock = threading.Lock()
        if getattr(args, "stream", False):
            self._start_stream_server(int(getattr(args, "stream_port", 8080)))

        if self.args.dry_run:
            self.vehicle_state.altitude_rel_m = float(args.sim_alt)

        # 6. Target Motion Trajectory Tail (Breadcrumb ribbon)
        self.target_trail: deque = deque(maxlen=24)
        self.prev_trail_id: Optional[int] = None

    def _start_stream_server(self, port: int):
        """Starts a background HTTP MJPEG stream server for live browser viewing over SSH or USB-C."""
        node = self

        class MJPEGHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress noisy HTTP request logging in terminal

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Autonomous Drone Pursuit - Live Stream</title>
    <style>
        body {{
            background: #0f141c;
            color: #ecf0f1;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }}
        .header {{
            margin-bottom: 12px;
            text-align: center;
        }}
        h1 {{
            margin: 0 0 4px 0;
            font-size: 20px;
            letter-spacing: 1px;
            color: #00d2d3;
        }}
        .status {{
            font-size: 13px;
            color: #a4b0be;
        }}
        .video-box {{
            box-shadow: 0 8px 32px rgba(0, 210, 211, 0.15);
            border: 2px solid #2f3542;
            border-radius: 8px;
            overflow: hidden;
            background: #000;
            max-width: 95vw;
            max-height: 85vh;
        }}
        img {{
            display: block;
            max-width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>AUTONOMOUS DRONE TRACKING & SERVOING</h1>
        <div class="status">&#9679; LIVE HUD & INFERENCE STREAM (PORT {port})</div>
    </div>
    <div class="video-box">
        <img src="/stream" alt="Live Companion Stream" />
    </div>
</body>
</html>"""
                    self.wfile.write(html.encode("utf-8"))

                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
                    self.end_headers()
                    try:
                        while node.running:
                            with node.jpeg_lock:
                                jpeg_bytes = node.latest_jpeg
                            if jpeg_bytes is not None:
                                self.wfile.write(b"--FRAME\r\n")
                                self.send_header("Content-Type", "image/jpeg")
                                self.send_header("Content-Length", str(len(jpeg_bytes)))
                                self.end_headers()
                                self.wfile.write(jpeg_bytes)
                                self.wfile.write(b"\r\n")
                            time.sleep(0.033)  # ~30 FPS broadcast cadence
                    except (ConnectionResetError, BrokenPipeError):
                        pass
                else:
                    self.send_error(404)

        class ReusableThreadingHTTPServer(ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        try:
            self.stream_server = ReusableThreadingHTTPServer(("0.0.0.0", port), MJPEGHandler)
            stream_thread = threading.Thread(target=self.stream_server.serve_forever, daemon=True)
            stream_thread.start()
            print(f"[STREAM] Live MJPEG web stream active on port {port}:")
            print(f"         • Over Wi-Fi Hotspot: http://10.42.0.1:{port}/")
            print(f"         • Over USB-C Cable:   http://192.168.55.1:{port}/")
        except Exception as e:
            print(f"[WARN] Failed to start HTTP stream server on port {port}: {e}")


    async def telemetry_listener(self):
        """Asynchronously streams telemetry from Pixhawk EKF2 over MAVLink."""
        if not self.drone:
            return

        async def watch_attitude():
            try:
                async for att in self.drone.telemetry.attitude_euler():
                    if not self.running:
                        break
                    # PX4 right-hand aerospace convention:
                    #   Pitch > 0 is nose UP.
                    # Controller convention:
                    #   Pitch > 0 is nose DOWN (forward acceleration).
                    # We invert the sign:
                    self.vehicle_state.pitch_rad = -float(np.deg2rad(att.pitch_deg))
                    self.vehicle_state.roll_rad = float(np.deg2rad(att.roll_deg))
                    self.vehicle_state.yaw_deg = float(att.yaw_deg)
            except Exception as e:
                print(f"[WARN] Attitude telemetry stream interrupted: {e}")

        async def watch_flight_mode():
            try:
                async for mode in self.drone.telemetry.flight_mode():
                    if not self.running:
                        break
            except Exception as e:
                print(f"[WARN] Flight mode stream interrupted: {e}")

        async def watch_position():
            try:
                async for pos in self.drone.telemetry.position():
                    if not self.running:
                        break
                    self.vehicle_state.altitude_rel_m = float(pos.relative_altitude_m)
                    self.vehicle_state.altitude_amsl_m = float(pos.absolute_altitude_m)
            except Exception as e:
                print(f"[WARN] Position telemetry stream interrupted: {e}")

        async def watch_distance_sensor():
            try:
                async for dist in self.drone.telemetry.distance_sensor():
                    if not self.running:
                        break
                    self.vehicle_state.distance_sensor_m = float(dist.current_distance_m)
            except Exception:
                pass

        async def watch_velocity_ned():
            try:
                async for vel in self.drone.telemetry.velocity_ned():
                    if not self.running:
                        break
                    vn = float(vel.north_m_s)
                    ve = float(vel.east_m_s)
                    vd = float(vel.down_m_s)
                    self.vehicle_state.vx_m_s = vn
                    self.vehicle_state.vy_m_s = ve
                    self.vehicle_state.vz_m_s = vd
                    # Active horizontal aiding is verified if non-zero velocity/variance is reported
                    if abs(vn) > 1e-4 or abs(ve) > 1e-4:
                        self.vehicle_state.has_gps_vel = True
            except Exception as e:
                print(f"[WARN] Velocity NED stream interrupted: {e}")

        async def watch_in_air():
            try:
                async for in_air in self.drone.telemetry.in_air():
                    if not self.running:
                        break
                    self.vehicle_state.in_air = bool(in_air)
            except Exception as e:
                print(f"[WARN] In-air telemetry stream interrupted: {e}")

        await asyncio.gather(
            watch_attitude(),
            watch_flight_mode(),
            watch_position(),
            watch_distance_sensor(),
            watch_velocity_ned(),
            watch_in_air()
        )

    def get_current_altitude(self) -> float:
        """
        Returns the most reliable current vehicle altitude Above Ground Level (AGL) in meters.
        Prefers downward distance sensor (LIDAR/sonar) if healthy, falls back to EKF2 relative altitude.
        """
        if (self.vehicle_state.distance_sensor_m is not None and 
            0.05 < self.vehicle_state.distance_sensor_m < 50.0):
            return self.vehicle_state.distance_sensor_m
        return self.vehicle_state.altitude_rel_m

    def enforce_altitude_limits(self, v_down: float, current_alt: float) -> Tuple[float, str]:
        """
        Enforces altitude floor (ground collision protection) and ceiling (airspace limit).

        Coordinate convention (PX4 NED Body Frame):
          v_down > 0 is DESCENT (moving downward towards the earth)
          v_down < 0 is CLIMB (moving upward into the sky)

        Args:
            v_down: Commanded vertical velocity in m/s (+ down, - up)
            current_alt: Current altitude AGL in meters

        Returns:
            Tuple of (safe_v_down: float, safety_status: str)
        """
        min_alt = self.args.min_alt
        max_alt = self.args.max_alt
        cushion = max(0.1, self.args.alt_cushion)

        # -------------------------------------------------------------
        # 1. ALTITUDE FLOOR (Ground Collision Avoidance)
        # -------------------------------------------------------------
        if current_alt <= min_alt:
            # Below or at floor: Descent is strictly forbidden
            if current_alt < (min_alt - 0.2):
                # Critical breach: Force active emergency climb (negative down)
                safe_v_down = -min(self.args.max_climb, 0.8)
                return safe_v_down, "FLOOR_BREACH"
            else:
                # At floor boundary: Only upward climb is permitted
                safe_v_down = min(0.0, v_down)
                return safe_v_down, "FLOOR_LIMIT"

        elif current_alt < (min_alt + cushion):
            # Inside floor cushion: Proportional descent damping
            if v_down > 0.0:  # Vehicle attempting to descend
                margin_ratio = (current_alt - min_alt) / cushion
                # Smooth quadratic deceleration cushion
                safe_v_down = v_down * (margin_ratio ** 1.5)
                return safe_v_down, "FLOOR_CUSHION"

        # -------------------------------------------------------------
        # 2. ALTITUDE CEILING (Flyaway / Airspace Limit Protection)
        # -------------------------------------------------------------
        if current_alt >= max_alt:
            # Above or at ceiling: Climb is strictly forbidden
            if current_alt > (max_alt + 0.5):
                # Critical breach: Force gentle descent (positive down)
                safe_v_down = min(self.args.max_desc, 0.5)
                return safe_v_down, "CEIL_BREACH"
            else:
                # At ceiling boundary: Only downward descent is permitted
                safe_v_down = max(0.0, v_down)
                return safe_v_down, "CEIL_LIMIT"

        elif current_alt > (max_alt - cushion):
            # Inside ceiling cushion: Proportional climb damping
            if v_down < 0.0:  # Vehicle attempting to climb
                margin_ratio = (max_alt - current_alt) / cushion
                safe_v_down = v_down * (margin_ratio ** 1.5)
                return safe_v_down, "CEIL_CUSHION"

        return v_down, "OK"

    def draw_flight_hud(
        self,
        frame: np.ndarray,
        telemetry: Dict[str, Any],
        cmd_3d: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """Overlays comprehensive real-time flight control gauges on the camera feed."""
        h, w = frame.shape[:2]
        hud = frame.copy()

        # Commanded velocities (supports both 3D and 4D)
        v_fwd = cmd_3d[0]
        v_right = cmd_3d[1] if len(cmd_3d) == 4 else 0.0
        v_down = cmd_3d[2] if len(cmd_3d) == 4 else cmd_3d[1]  # NED (+ is down, - is up)
        yawspeed = cmd_3d[3] if len(cmd_3d) == 4 else cmd_3d[2]
        status = telemetry.get("status", "SEARCHING")
        current_alt = self.get_current_altitude()
        alt_status = self.vehicle_state.alt_safety_status

        # Top Control Banner
        banner_color = (25, 28, 36)
        cv2.rectangle(hud, (0, 0), (w, 55), banner_color, -1)
        cv2.line(hud, (0, 55), (w, 55), (0, 210, 211), 1)

        # Status badge color
        badge_color = (46, 213, 115) if status == "LOCKED" else ((255, 165, 2) if status == "COASTING" else (255, 71, 87))
        cv2.rectangle(hud, (12, 10), (120, 44), badge_color, -1)
        cv2.putText(hud, status, (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

        # Target Size, Pitch Info & Altitude
        curr_sz = telemetry.get("target_size", 0.0)
        goal_sz = self.args.target_size
        pitch_deg = -np.rad2deg(self.vehicle_state.pitch_rad)
        cam_fps = 1.0 / max(1e-4, dt)
        info_str = f"SZ: {curr_sz:3.0f}/{goal_sz:.0f}px | PITCH: {pitch_deg:+4.1f}* | ALT: {current_alt:4.1f}m | CAM: {cam_fps:4.1f}fps | MAV: {self.args.ctrl_rate:.0f}Hz"
        cv2.putText(hud, info_str, (128, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

        # MAVLink link badge (top right)
        if self.args.dry_run and self.vehicle_state.is_connected:
            link_str = "PX4: DRY-RUN (TEL)"
            link_color = (46, 213, 115)  # Green: Live IMU telemetry streaming!
        elif self.args.dry_run or not self.vehicle_state.is_connected:
            link_str = "PX4: DRY-RUN"
            link_color = (120, 120, 120)  # Gray: Standalone desktop dry-run
        else:
            link_str = f"PX4: {self.vehicle_state.flight_mode}"
            link_color = (46, 213, 115)
        cv2.putText(hud, link_str, (w - 180, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48, link_color, 1, cv2.LINE_AA)

        # Altitude Limit Warning Alert Banner (if floor/ceiling active)
        alert_y = 60
        if alt_status != "OK":
            alert_color = (0, 0, 230) if "BREACH" in alt_status else (0, 165, 255)
            alert_text = f"ALT SAFEGUARD: {alt_status} (FLR: {self.args.min_alt:.1f}m | CEIL: {self.args.max_alt:.1f}m)"
            box_w = 420
            cv2.rectangle(hud, (w // 2 - box_w // 2, alert_y), (w // 2 + box_w // 2, alert_y + 30), (20, 20, 20), -1)
            cv2.rectangle(hud, (w // 2 - box_w // 2, alert_y), (w // 2 + box_w // 2, alert_y + 30), alert_color, 2)
            cv2.putText(hud, alert_text, (w // 2 - box_w // 2 + 12, alert_y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.44, alert_color, 2, cv2.LINE_AA)
            alert_y += 35

        # Perception Watchdog Warning Banner
        if self.perception_state.is_stale and self.perception_state.frame_idx > 0:
            wd_text = "WATCHDOG: VISION STALE (SAFE STATION-KEEPING HOVER)"
            box_w = 460
            cv2.rectangle(hud, (w // 2 - box_w // 2, alert_y), (w // 2 + box_w // 2, alert_y + 30), (20, 20, 20), -1)
            cv2.rectangle(hud, (w // 2 - box_w // 2, alert_y), (w // 2 + box_w // 2, alert_y + 30), (0, 165, 255), 2)
            cv2.putText(hud, wd_text, (w // 2 - box_w // 2 + 10, alert_y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 2, cv2.LINE_AA)

        # Bottom Command Gauge Panel
        cv2.rectangle(hud, (0, h - 45), (w, h), banner_color, -1)
        cv2.line(hud, (0, h - 45), (w, h - 45), (0, 210, 211), 1)

        if abs(v_right) > 0.05:
            cmd_text = f"CMD -> FWD: {v_fwd:4.1f} | LAT: {v_right:+4.1f} | VERT: {-v_down:+4.2f} m/s | YAW: {yawspeed:+5.1f} */s | GUARD: {alt_status}"
        else:
            cmd_text = f"CMD -> FWD: {v_fwd:4.1f} m/s | VERT: {-v_down:+4.2f} m/s | YAW: {yawspeed:+5.1f} */s | GUARD: {alt_status}"
        cv2.putText(hud, cmd_text, (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 210, 211), 2, cv2.LINE_AA)

        # Alpha blend background banners with frame
        out = cv2.addWeighted(hud, 0.88, frame, 0.12, 0)

        # Center Gimbal Optical Boresight Crosshair
        cx, cy = w // 2, h // 2
        reticle_color = (180, 180, 180)
        cv2.circle(out, (cx, cy), 16, reticle_color, 1, cv2.LINE_AA)
        cv2.line(out, (cx - 28, cy), (cx - 18, cy), reticle_color, 1, cv2.LINE_AA)
        cv2.line(out, (cx + 18, cy), (cx + 28, cy), reticle_color, 1, cv2.LINE_AA)
        cv2.line(out, (cx, cy - 28), (cx, cy - 18), reticle_color, 1, cv2.LINE_AA)
        cv2.line(out, (cx, cy + 18), (cx, cy + 28), reticle_color, 1, cv2.LINE_AA)

        # Target Bounding Box, Tactical Brackets, Motion Tail, & Tracking Lead Line
        bbox = telemetry.get("bbox")
        if bbox is not None:
            bx1, by1, bx2, by2 = [int(v) for v in bbox]
            tcx, tcy = (bx1 + bx2) // 2, (by1 + by2) // 2
            trk_id = telemetry.get("target_id", 1)
            box_color = (46, 213, 115) if status == "LOCKED" else (255, 165, 2)

            # Update motion trajectory breadcrumb tail
            if self.prev_trail_id != trk_id:
                self.target_trail.clear()
                self.prev_trail_id = trk_id
            self.target_trail.append((tcx, tcy))

            # Draw dynamic motion tail with gradient alpha and fading line thickness
            n_pts = len(self.target_trail)
            if n_pts > 1:
                pts_list = list(self.target_trail)
                for i in range(n_pts - 1):
                    # Progress from oldest (0.15) to newest (1.0)
                    alpha = (i + 1) / n_pts
                    # Gradient color matching target status: fade from subtle to bright
                    r = int(box_color[0] * (0.15 + 0.85 * alpha))
                    g = int(box_color[1] * (0.15 + 0.85 * alpha))
                    b = int(box_color[2] * (0.15 + 0.85 * alpha))
                    seg_color = (r, g, b)
                    seg_thick = max(1, int(round(1.0 + 2.0 * alpha)))
                    cv2.line(out, pts_list[i], pts_list[i + 1], seg_color, seg_thick, cv2.LINE_AA)
                    # Micro breadcrumb dot along key waypoints
                    if i % 4 == 0 or i == 0:
                        dot_radius = max(1, int(round(1.0 + 1.5 * alpha)))
                        cv2.circle(out, pts_list[i], dot_radius, seg_color, -1, cv2.LINE_AA)

            # Tactical corner brackets around target
            c_len = max(8, min(24, int(abs(bx2 - bx1) * 0.25)))
            thick = 2
            # Top-Left
            cv2.line(out, (bx1, by1), (bx1 + c_len, by1), box_color, thick, cv2.LINE_AA)
            cv2.line(out, (bx1, by1), (bx1, by1 + c_len), box_color, thick, cv2.LINE_AA)
            # Top-Right
            cv2.line(out, (bx2, by1), (bx2 - c_len, by1), box_color, thick, cv2.LINE_AA)
            cv2.line(out, (bx2, by1), (bx2, by1 + c_len), box_color, thick, cv2.LINE_AA)
            # Bottom-Left
            cv2.line(out, (bx1, by2), (bx1 + c_len, by2), box_color, thick, cv2.LINE_AA)
            cv2.line(out, (bx1, by2), (bx1, by2 - c_len), box_color, thick, cv2.LINE_AA)
            # Bottom-Right
            cv2.line(out, (bx2, by2), (bx2 - c_len, by2), box_color, thick, cv2.LINE_AA)
            cv2.line(out, (bx2, by2), (bx2, by2 - c_len), box_color, thick, cv2.LINE_AA)

            # Center target reticle dot & lead line to optical boresight
            cv2.circle(out, (tcx, tcy), 4, box_color, -1, cv2.LINE_AA)
            cv2.line(out, (cx, cy), (tcx, tcy), (0, 210, 211), 1, cv2.LINE_AA)

            # Target Lock Badge
            tag_str = f"TARGET #{trk_id} [{curr_sz:.0f}px]"
            badge_w = len(tag_str) * 8 + 12
            cv2.rectangle(out, (bx1, max(10, by1 - 22)), (bx1 + badge_w, max(10, by1)), (20, 20, 20), -1)
            cv2.rectangle(out, (bx1, max(10, by1 - 22)), (bx1 + badge_w, max(10, by1)), box_color, 1)
            cv2.putText(out, tag_str, (bx1 + 5, max(24, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1, cv2.LINE_AA)
        else:
            if len(self.target_trail) > 0:
                self.target_trail.popleft()  # Gracefully decay tail when target is lost
            cv2.putText(out, "[SEARCHING FOR TARGET]", (cx - 90, cy + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)

        return out

    def _open_capture_device(self) -> Optional[cv2.VideoCapture]:
        """
        Robust camera initialization for Jetson Orin Nano, Linux, and macOS.
        Automatically handles:
        - NVIDIA Jetson CSI ribbon cameras (nvarguscamerasrc) to prevent green-screen YUV zero-buffers
        - USB Webcams via V4L2 with MJPG hardware compression (prevents 4K USB bus congestion)
        - Automatic green-screen detection and seamless fallback to Jetson hardware ISP
        - Custom GStreamer pipelines or video files
        """
        source_str = str(self.args.source).strip()
        cam_w = getattr(self.args, "cam_width", 1280)
        cam_h = getattr(self.args, "cam_height", 720)
        cam_fps = int(getattr(self.args, "cam_fps", 30))
        cam_flip = getattr(self.args, "cam_flip", 0)

        # 1. Explicit GStreamer pipeline string
        if "!" in source_str or "nvarguscamerasrc" in source_str:
            print("[VISION] Opening custom GStreamer pipeline...")
            cap = cv2.VideoCapture(source_str, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                return cap
            print("[WARN] Custom GStreamer pipeline failed to open.")

        # Determine exact native hardware mode for Jetson camera sensor:
        # Hardware Mode 0: 3840x2160 @ 30fps (full native sensor supersampling)
        # Hardware Mode 1: 1920x1080 @ 60fps (high-speed tracking)
        if cam_fps >= 50:
            sensor_cap_w, sensor_cap_h, sensor_fps = 1920, 1080, 60
        else:
            sensor_cap_w, sensor_cap_h, sensor_fps = 3840, 2160, 30

        # 2. CSI Camera explicitly requested (--csi or --source csi / csi:0)
        use_csi = getattr(self.args, "csi", False) or source_str.lower().startswith("csi")
        if use_csi:
            sensor_id = 0
            if ":" in source_str:
                try:
                    sensor_id = int(source_str.split(":")[-1])
                except ValueError:
                    sensor_id = 0
            pipe = build_jetson_csi_pipeline(
                sensor_id=sensor_id,
                capture_width=sensor_cap_w,
                capture_height=sensor_cap_h,
                framerate=sensor_fps,
                flip_method=cam_flip,
                display_width=cam_w,
                display_height=cam_h,
            )
            print(f"[VISION] Opening Jetson CSI camera (sensor-id={sensor_id}, Mode: {sensor_cap_w}x{sensor_cap_h}@{sensor_fps}fps) via nvarguscamerasrc...")
            cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    return cap
                cap.release()
            print("[WARN] Failed to open nvarguscamerasrc pipeline.")
            print("       Tip: If camera is locked, run: 'sudo systemctl restart nvargus-daemon'")

        # 3. Numeric source index (webcam or /dev/videoX)
        if source_str.isdigit():
            src_idx = int(source_str)

            # A. On Linux: Try V4L2 with MJPG first (standard for USB webcams to avoid uncompressed YUYV bus congestion)
            if sys.platform.startswith("linux"):
                cap = cv2.VideoCapture(src_idx, cv2.CAP_V4L2)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
                    cap.set(cv2.CAP_PROP_FPS, cam_fps)
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and not is_zero_yuv_green_frame(test_frame):
                        print(f"[VISION] Initialized USB camera on /dev/video{src_idx} via V4L2 (MJPG {cam_w}x{cam_h}).")
                        return cap
                    cap.release()

            # B. Standard VideoCapture with Green-Screen Auto-Recovery
            cap = cv2.VideoCapture(src_idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    if is_zero_yuv_green_frame(test_frame):
                        print(f"\n[NOTICE] Detected solid green frame on /dev/video{src_idx} (uninitialized V4L2 DMA buffer).")
                        print("[NOTICE] CSI ribbon camera detected without ISP! Automatically switching to Jetson hardware ISP (nvarguscamerasrc)...")
                        cap.release()
                        pipe = build_jetson_csi_pipeline(
                            sensor_id=src_idx,
                            capture_width=sensor_cap_w,
                            capture_height=sensor_cap_h,
                            framerate=sensor_fps,
                            flip_method=cam_flip,
                            display_width=cam_w,
                            display_height=cam_h,
                        )
                        csi_cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
                        if csi_cap.isOpened():
                            ret_csi, test_csi = csi_cap.read()
                            if ret_csi and test_csi is not None and not is_zero_yuv_green_frame(test_csi):
                                print(f"[VISION] Successfully recovered Jetson CSI camera via nvarguscamerasrc ({sensor_cap_w}x{sensor_cap_h})!")
                                return csi_cap
                            csi_cap.release()
                        print("[WARN] nvarguscamerasrc auto-recovery failed. Re-opening standard capture.")
                        cap = cv2.VideoCapture(src_idx)
                    return cap

        # 4. Fallback: Video file path
        return cv2.VideoCapture(source_str)

    def _vision_worker(self):
        """
        Independent vision ingestion & inference thread.
        Runs at the camera's native framerate (or YOLO throughput) without blocking MAVLink control.
        """
        cap = self._open_capture_device()
        if cap is None or not cap.isOpened():
            print(f"\n[ERROR] Failed to open video source: {self.args.source}")
            self.running = False
            return

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[VISION] Ingestion worker active ({actual_w}x{actual_h})")

        prev_time = time.perf_counter()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                if not str(self.args.source).isdigit() and Path(self.args.source).is_file():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("[VISION] Video source reached end or disconnected.")
                self.running = False
                break

            now = time.perf_counter()
            raw_dt = now - prev_time
            prev_time = now
            fps = 1.0 / max(1e-4, raw_dt)

            # Compute ego-motion telemetry (Dual Mode: GPS EKF velocity or Option A commanded dead-reckoning)
            if self.vehicle_state.has_gps_vel:
                # Rotate NED velocities into Body frame using drone heading
                yaw_rad = np.deg2rad(self.vehicle_state.yaw_deg)
                vn = self.vehicle_state.vx_m_s
                ve = self.vehicle_state.vy_m_s
                vx_body = float(np.cos(yaw_rad) * vn + np.sin(yaw_rad) * ve)
                vy_body = float(-np.sin(yaw_rad) * vn + np.cos(yaw_rad) * ve)
                vz_body = float(self.vehicle_state.vz_m_s)
            else:
                # Option A: Dead-reckoning translation proxy from latest safe command
                if hasattr(self, "latest_cmd_safe") and self.latest_cmd_safe is not None:
                    vx_body = float(self.latest_cmd_safe[0])
                    vy_body = float(self.latest_cmd_safe[1]) if len(self.latest_cmd_safe) >= 4 else 0.0
                    cmd_down = float(self.latest_cmd_safe[2]) if len(self.latest_cmd_safe) >= 4 else float(self.latest_cmd_safe[1])
                    vz_body = float(self.vehicle_state.vz_m_s) if abs(self.vehicle_state.vz_m_s) > 1e-3 else cmd_down
                else:
                    vx_body, vy_body, vz_body = 0.0, 0.0, 0.0

            ego_telemetry = {
                "pitch_rad": float(self.vehicle_state.pitch_rad),
                "roll_rad": float(self.vehicle_state.roll_rad),
                "yaw_deg": float(self.vehicle_state.yaw_deg),
                "vx_body": vx_body,
                "vy_body": vy_body,
                "vz_body": vz_body,
                "altitude_m": float(self.get_current_altitude()),
                "dt": float(raw_dt)
            }

            # Ingest YOLOv8 + Kalman Tracking (with active ego-motion compensation)
            annotated_frame, telemetry = self.pipeline.process_frame(
                frame, draw_hud=False, ego_telemetry=ego_telemetry
            )

            with self.perception_state.lock:
                self.perception_state.telemetry = telemetry
                self.perception_state.latest_frame = annotated_frame
                self.perception_state.last_update_time = time.perf_counter()
                self.perception_state.frame_idx = self.pipeline.frame_idx
                self.perception_state.fps = fps
                self.perception_state.is_stale = False
                self.perception_state.has_new_frame = True

            if self.args.max_frames > 0 and self.pipeline.frame_idx >= self.args.max_frames:
                print(f"[LIMIT] Reached max requested frames ({self.args.max_frames}).")
                self.running = False
                break

        cap.release()
        print("[VISION] Ingestion worker terminated cleanly.")

    async def control_loop(self):
        """
        High-priority deterministic MAVLink flight control loop (default 50 Hz).
        Runs continuously, decoupled from vision, ensuring PX4 offboard heartbeat never drops.
        """
        ctrl_period = 1.0 / max(1.0, self.args.ctrl_rate)
        prev_time = time.perf_counter()
        prev_cmd_3d = np.zeros(3, dtype=np.float64)

        print(f"[CONTROL] Deterministic MAVLink control loop active at {self.args.ctrl_rate:.0f} Hz.")

        while self.running:
            loop_start = time.perf_counter()
            now = loop_start
            dt = float(np.clip(now - prev_time, 1e-4, 0.10))
            prev_time = now

            # 1. Read latest telemetry under lock
            with self.perception_state.lock:
                telemetry = dict(self.perception_state.telemetry)
                last_vision_time = self.perception_state.last_update_time
                time_since_vision = (now - last_vision_time) if last_vision_time > 0 else 999.0

            # 2. Perception Deadman Watchdog Check
            # If no vision update for > watchdog_timeout, declare STALE and force safe holding hover
            if time_since_vision > self.args.watchdog_timeout:
                telemetry["status"] = "SEARCHING"
                with self.perception_state.lock:
                    self.perception_state.is_stale = True

            # 3. Dynamic attitude compensation from vehicle IMU (Pitch and Roll)
            current_pitch = self.vehicle_state.pitch_rad
            current_roll = self.vehicle_state.roll_rad

            # 4. Compute Velocity Command via Kinematic Visual Servoing (3D or 4D)
            raw_cmd = self.controller.compute_cmd(
                telemetry=telemetry,
                drone_pitch=current_pitch,
                drone_roll=current_roll,
                dt=dt
            )

            # 5. Slew-Rate Limiting (Acceleration Limiting for smooth control signals)
            if len(raw_cmd) == 4:
                max_accel = np.array([
                    self.args.max_accel_xy,
                    self.args.max_accel_xy,
                    self.args.max_accel_z,
                    self.args.max_accel_yaw
                ], dtype=np.float64)
            else:
                max_accel = np.array([
                    self.args.max_accel_xy,
                    self.args.max_accel_z,
                    self.args.max_accel_yaw
                ], dtype=np.float64)

            if len(prev_cmd_3d) != len(raw_cmd):
                prev_cmd_3d = np.zeros_like(raw_cmd)

            max_step = max_accel * dt
            cmd_safe = np.clip(raw_cmd, prev_cmd_3d - max_step, prev_cmd_3d + max_step)
            prev_cmd_3d = cmd_safe.copy()

            # 6. Convert to PX4 MAVSDK Body Frame (forward, right, down, yawspeed)
            v_fwd, v_right, v_down, yawspeed = self.controller.to_mavsdk(cmd_safe)

            # 7. Altitude Floor & Ceiling Safety Envelope Protection
            current_alt = self.get_current_altitude()
            safe_v_down, alt_status = self.enforce_altitude_limits(v_down, current_alt)
            self.vehicle_state.alt_safety_status = alt_status
            v_down = safe_v_down

            # In dry-run mode, simulate altitude dynamics
            if self.args.dry_run:
                self.vehicle_state.altitude_rel_m = max(
                    0.0, self.vehicle_state.altitude_rel_m - (v_down * dt)
                )

            # 8. Dispatch steady velocity setpoint to Pixhawk Autopilot
            if self.drone and not self.args.dry_run and self.vehicle_state.is_connected:
                try:
                    await self.drone.offboard.set_velocity_body(
                        VelocityBodyYawspeed(
                            forward_m_s=v_fwd,
                            right_m_s=v_right,
                            down_m_s=v_down,
                            yawspeed_deg_s=yawspeed
                        )
                    )
                except OffboardError:
                    # Offboard mode dropped (e.g. pilot manual RC takeover)
                    pass

            # 9. Store latest safe command for HUD rendering & diagnostics
            self.latest_cmd_safe = np.array([v_fwd, v_right, v_down, yawspeed], dtype=np.float64)

            # 10. Precise sleep to maintain deterministic frequency
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.001, ctrl_period - elapsed)
            await asyncio.sleep(sleep_time)

    async def display_loop(self):
        """
        GUI rendering, user input listener, and video recording loop (or periodic headless telemetry console logger).
        Runs cooperatively on the main asyncio thread.
        """
        last_log_time = time.perf_counter()

        while self.running:
            new_frame_available = False
            with self.perception_state.lock:
                if self.perception_state.has_new_frame:
                    frame = self.perception_state.latest_frame
                    telemetry = dict(self.perception_state.telemetry)
                    fps = self.perception_state.fps
                    is_stale = self.perception_state.is_stale
                    self.perception_state.has_new_frame = False
                    new_frame_available = True
                else:
                    frame = self.perception_state.latest_frame
                    telemetry = dict(self.perception_state.telemetry)
                    fps = self.perception_state.fps
                    is_stale = self.perception_state.is_stale

            if frame is not None:
                if is_stale:
                    telemetry["status"] = "STALE (WD)"

                cmd_safe = getattr(self, "latest_cmd_safe", np.zeros(3))
                display_frame = self.draw_flight_hud(frame, telemetry, cmd_safe, 1.0 / max(1.0, fps))

                # Live Web Stream: broadcast latest annotated HUD frame to browser / SSH port forward
                if self.stream_server is not None and new_frame_available:
                    ret_enc, buf = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret_enc:
                        with self.jpeg_lock:
                            self.latest_jpeg = buf.tobytes()

                # Video Recording: write each uniquely processed frame with HUD and bounding box
                if self.record_path is not None and new_frame_available:
                    if self.video_writer is None:
                        h, w = display_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        self.video_writer = cv2.VideoWriter(str(self.record_path), fourcc, self.record_fps, (w, h))
                        print(f"[RECORD] Video writer initialized: {self.record_path} ({w}x{h} @ {self.record_fps:.1f} FPS)")
                    self.video_writer.write(display_frame)

                if not self.args.headless:
                    cv2.imshow("Autonomous Drone Tracker - Companion Node", display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[USER] Quit key pressed.")
                        self.running = False
                        break

            if self.args.headless:
                now = time.perf_counter()
                if now - last_log_time >= 0.5:  # 2 Hz clean telemetry log
                    last_log_time = now
                    with self.perception_state.lock:
                        status = self.perception_state.telemetry.get("status", "SEARCHING")
                        f_idx = self.perception_state.frame_idx
                        v_fps = self.perception_state.fps
                        sz = self.perception_state.telemetry.get("target_size", 0.0)
                        is_stale = self.perception_state.is_stale

                    if is_stale and f_idx > 0:
                        status = "STALE_WD"

                    current_alt = self.get_current_altitude()
                    alt_status = self.vehicle_state.alt_safety_status
                    cmd = getattr(self, "latest_cmd_safe", np.zeros(3))

                    print(f"Frame {f_idx:5d} ({v_fps:4.1f}fps) | "
                          f"Status: {status:8s} | "
                          f"Alt: {current_alt:4.1f}m ({alt_status:13s}) | "
                          f"Size: {sz:4.0f}px | "
                          f"Cmd: [vx={cmd[0]:4.1f}, vz={cmd[1]:+4.2f}, yaw={cmd[2]:+5.1f}]")

            await asyncio.sleep(0.01)  # ~100 Hz responsive check

    async def run(self):
        """Entry point that coordinates the decoupled threads and asyncio tasks."""
        self.running = True

        # Check MAVSDK availability & connection mode
        telemetry_task = None
        if self.args.connection and MAVSDK_AVAILABLE:
            mode_desc = "LISTEN-ONLY DRY-RUN (Telemetry active, motor commands disabled)" if self.args.dry_run else "LIVE AUTOPILOT CONTROL"
            print(f"[MAVLINK] Connecting to Pixhawk via {self.args.connection} [{mode_desc}]...")
            self.drone = System()
            try:
                await self.drone.connect(system_address=self.args.connection)
                print("[MAVLINK] Waiting for autopilot heartbeat...")
                async for state in self.drone.core.connection_state():
                    if state.is_connected:
                        print("[MAVLINK] ✓ Autopilot connected!")
                        self.vehicle_state.is_connected = True
                        break
                telemetry_task = asyncio.create_task(self.telemetry_listener())
            except Exception as e:
                print(f"[ERROR] Could not connect to Pixhawk: {e}")
                print("[FALLBACK] Switching to standalone Dry-Run mode.\n")
                self.args.dry_run = True
        else:
            if not self.args.dry_run:
                if not MAVSDK_AVAILABLE:
                    print("\n[NOTICE] 'mavsdk' is not installed in the current Python environment.")
                elif not self.args.connection:
                    print("\n[NOTICE] No MAVLink connection string provided via --connection.")
                print("[NOTICE] Running in standalone DRY-RUN Mode (Desktop Benchtop Test).\n")
                self.args.dry_run = True

        # Startup banner
        print("\n=======================================================")
        print("    AUTONOMOUS DRONE TRACKING & SERVOING RUNNING       ")
        print("=======================================================")
        print(f"• Flight Mode:        {'DRY-RUN (Desktop Test)' if self.args.dry_run else 'LIVE MAVSDK AUTOPILOT'}")
        print(f"• MAVLink Loop Rate:  {self.args.ctrl_rate:.0f} Hz (Decoupled Deterministic Stream)")
        print(f"• Target Pixel Size:  {self.args.target_size:.0f} px")
        print(f"• Altitude Floor:     {self.args.min_alt:.1f} m AGL (Cushion: {self.args.alt_cushion:.1f} m)")
        print(f"• Altitude Ceiling:   {self.args.max_alt:.1f} m AGL")
        print(f"• Watchdog Timeout:   {self.args.watchdog_timeout * 1000:.0f} ms")
        print(f"• Acceleration Caps:  XY={self.args.max_accel_xy:.1f} m/s^2 | Z={self.args.max_accel_z:.1f} m/s^2 | Yaw={self.args.max_accel_yaw:.0f} deg/s^2")
        print(f"• Forward Speed Cap:  {self.args.max_speed:.1f} m/s")
        print(f"• Press 'q' in video window or Ctrl+C to terminate.")
        print("=======================================================\n")

        # 1. Start Vision Worker Thread (Independent OS Thread)
        self.vision_thread = threading.Thread(target=self._vision_worker, name="VisionWorker", daemon=True)
        self.vision_thread.start()

        # 2. Launch Asyncio Control & Display Tasks
        control_task = asyncio.create_task(self.control_loop())
        display_task = asyncio.create_task(self.display_loop())

        try:
            # Run tasks concurrently until user termination or error
            done, pending = await asyncio.wait(
                [control_task, display_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                if not t.cancelled() and t.exception():
                    print(f"\n[ERROR] Task crashed with exception: {t.exception()}")
                    import traceback
                    traceback.print_exception(type(t.exception()), t.exception(), t.exception().__traceback__)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n[STOP] Termination signal received.")
        finally:
            self.running = False

            # Cancel remaining asyncio tasks
            control_task.cancel()
            display_task.cancel()
            if telemetry_task:
                telemetry_task.cancel()

            # Wait for vision thread to cleanly close camera
            if self.vision_thread and self.vision_thread.is_alive():
                self.vision_thread.join(timeout=1.5)

            if not self.args.headless:
                cv2.destroyAllWindows()

            # Flush and release video recording
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                print(f"[RECORD] ✓ Tracked video successfully saved to: {self.record_path}")

            # Shutdown live MJPEG stream server
            if self.stream_server is not None:
                try:
                    self.stream_server.shutdown()
                    self.stream_server.server_close()
                except Exception:
                    pass
                self.stream_server = None

            # Send safe zero-velocity holding command to Pixhawk before exit
            if self.drone and not self.args.dry_run and self.vehicle_state.is_connected:
                print("[SAFETY] Sending zero velocity holding command to Pixhawk...")
                try:
                    await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                except Exception:
                    pass

            print("[SHUTDOWN] Autonomous Tracker Node stopped cleanly.")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous Drone Pursuit Bridge Node: Connects Vision Pipeline with Pixhawk Controller."
    )
    # Vision & Camera Options
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: camera index (e.g. '0'), video file path, or GStreamer string."
    )
    default_weights = (
        str(DETECTION_DIR / "weights/yolov8n_v3_best.pt")
        if (DETECTION_DIR / "weights/yolov8n_v3_best.pt").exists()
        else str(DETECTION_DIR / "runs/detect/yolov8n_drone/weights/best.pt")
    )
    parser.add_argument(
        "--weights", type=str,
        default=default_weights,
        help=f"Path to trained YOLOv8 drone detector weights (default: {default_weights})."
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="YOLO detection confidence threshold (default: 0.35)."
    )
    parser.add_argument(
        "--target-size", type=float, default=35.0,
        help="Desired bounding box size in pixels on 640x480 frame (default: 35 px for ~6m standoff)."
    )
    parser.add_argument(
        "--max-lost-frames", type=int, default=15,
        help="Consecutive missed frames before Kalman track transitions from COASTING to SEARCHING."
    )
    parser.add_argument(
        "--record", nargs="?", const="auto", default=None,
        help="Record video with bounding boxes and flight HUD. Pass flag alone ('--record') to auto-generate timestamped MP4 in outputs/recordings/, or provide a path ('--record my_flight.mp4')."
    )
    parser.add_argument(
        "--record-fps", type=float, default=30.0,
        help="Recording framerate in FPS (default: 30.0)."
    )
    parser.add_argument(
        "--csi", action="store_true",
        help="Use NVIDIA Jetson hardware ISP pipeline (nvarguscamerasrc) for CSI ribbon cameras (e.g. Raspberry Pi HQ / IMX477 / IMX219)."
    )
    parser.add_argument(
        "--cam-flip", type=int, default=0,
        help="CSI camera rotation/flip method: 0=none, 2=rotate 180 degrees (for upside-down mounting)."
    )
    parser.add_argument(
        "--cam-width", type=int, default=1280,
        help="Camera ingestion width (default: 1280 for fast 30+ FPS inference; scales 4K down in hardware)."
    )
    parser.add_argument(
        "--cam-height", type=int, default=720,
        help="Camera ingestion height (default: 720)."
    )
    parser.add_argument(
        "--cam-fps", type=int, default=30,
        help="Camera framerate (default: 30)."
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Enable live HTTP MJPEG stream (viewable in Mac browser at http://192.168.55.1:8080 or http://localhost:8080 via SSH port forward)."
    )
    parser.add_argument(
        "--stream-port", type=int, default=8080,
        help="HTTP port for live MJPEG stream (default: 8080)."
    )

    # Controller & Flight Tuning Options
    parser.add_argument(
        "--max-speed", type=float, default=6.0,
        help="Maximum forward velocity limit (m/s)."
    )
    parser.add_argument(
        "--max-climb", type=float, default=2.5,
        help="Maximum climb velocity limit (m/s)."
    )
    parser.add_argument(
        "--max-desc", type=float, default=1.5,
        help="Maximum descent velocity limit (m/s)."
    )
    parser.add_argument(
        "--max-yawspeed", type=float, default=90.0,
        help="Maximum yaw turn rate limit (deg/s)."
    )
    parser.add_argument(
        "--lateral-strafe", action="store_true", default=True,
        help="Enable 4-DOF lateral strafe roll tilt (default: True)."
    )
    parser.add_argument(
        "--no-lateral-strafe", action="store_false", dest="lateral_strafe",
        help="Disable lateral strafe (enforces 3-DOF coordinated-turn yaw-only mode)."
    )
    parser.add_argument(
        "--max-lat-speed", type=float, default=2.5,
        help="Maximum lateral velocity limit in m/s for roll tilt (default: 2.5 m/s)."
    )
    parser.add_argument(
        "--uptilt", type=float, default=15.0,
        help="Mechanical camera mount up-tilt angle in degrees."
    )
    parser.add_argument(
        "--hfov", type=float, default=60.0,
        help="Horizontal field of view in degrees."
    )
    parser.add_argument(
        "--vfov", type=float, default=45.0,
        help="Vertical field of view in degrees."
    )

    # Altitude Floor and Ceiling Safety Limits
    parser.add_argument(
        "--min-alt", type=float, default=2.0,
        help="Hard altitude floor in meters AGL (prevents ground collision)."
    )
    parser.add_argument(
        "--max-alt", type=float, default=30.0,
        help="Hard altitude ceiling in meters AGL (prevents flyaway/excessive climb)."
    )
    parser.add_argument(
        "--alt-cushion", type=float, default=1.0,
        help="Proportional braking cushion zone (meters) near floor and ceiling boundaries."
    )
    parser.add_argument(
        "--sim-alt", type=float, default=5.0,
        help="Simulated initial altitude in meters AGL for desktop dry-run testing."
    )

    # MAVLink Timing & Slew-Rate Limiting Options
    parser.add_argument(
        "--ctrl-rate", type=float, default=50.0,
        help="Deterministic MAVLink setpoint stream rate in Hz (default: 50 Hz)."
    )
    parser.add_argument(
        "--watchdog-timeout", type=float, default=0.40,
        help="Deadman watchdog timeout in seconds before vision is declared stale (default: 0.40s)."
    )
    parser.add_argument(
        "--max-accel-xy", type=float, default=4.0,
        help="Maximum forward acceleration limit in m/s^2 for smooth slew-rate limiting (default: 4.0)."
    )
    parser.add_argument(
        "--max-accel-z", type=float, default=3.0,
        help="Maximum vertical acceleration limit in m/s^2 for smooth slew-rate limiting (default: 3.0)."
    )
    parser.add_argument(
        "--max-accel-yaw", type=float, default=180.0,
        help="Maximum yaw angular acceleration limit in deg/s^2 (default: 180.0)."
    )

    # MAVLink & Hardware Connection Options
    parser.add_argument(
        "--connection", type=str, default="",
        help="MAVLink connection URI, e.g. 'serial:///dev/ttyTHS1:921600' (Jetson) or 'udp://:14540' (SITL)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Force Dry-Run mode (runs vision & control math without sending commands to hardware)."
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without GUI display window (recommended for companion compute background service)."
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Optional limit on number of frames to process before exiting (0 = infinite)."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    node = AutonomousTrackerNode(args)
    asyncio.run(node.run())
