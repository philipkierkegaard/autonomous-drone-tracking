#!/usr/bin/env python3
"""
ArduPilot MAVLink Autopilot Bridge.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Provides high-rate, low-latency, bidirectional telemetry ingestion and
velocity guidance dispatch for ArduPilot (ArduCopter V4.x) flight controllers.

Key Architecture:
- Uses native PyMAVLink (zero external C++ daemons or gRPC servers).
- Streams 4-DOF body-frame velocity setpoints via MAVLink SET_POSITION_TARGET_LOCAL_NED (#84)
  in GUIDED mode.
- Monitors live HEARTBEAT for pilot manual RC takeover (LOITER / ALT_HOLD / STABILIZE).
- Asynchronously ingests high-rate attitude (50 Hz), velocity (30 Hz), altitude (20 Hz),
  and distance sensor (20 Hz) telemetry with thread-safe exposure-matched ring buffers.
"""

from __future__ import annotations

import sys
import time
import math
import bisect
import threading
from collections import deque
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

import numpy as np

try:
    from pymavlink import mavutil
    PYMAVLINK_AVAILABLE = True
except ImportError:
    PYMAVLINK_AVAILABLE = False


# MAVLink Set Position Target Local NED typemasks:
# We ignore position and acceleration, commanding only body velocity (vx, vy, vz) and yaw rate.
# Bits: 0b0000010111000111 = 0x5C7 = 1479
TYPEMASK_VELOCITY_YAWRATE = 0b0000010111000111


class ArduPilotBridge:
    """
    Robust native PyMAVLink interface for ArduCopter companion computer autonomy.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        baud: int = 921600,
        source_system: int = 255,      # Companion computer default MAVLink system ID
        source_component: int = 190,   # MAV_COMP_ID_ONBOARD_COMPUTER
        heartbeat_timeout_s: float = 3.0,
        vehicle_state: Optional[Any] = None,
    ):
        self.connection_string = connection_string
        self.baud = baud
        self.source_system = source_system
        self.source_component = source_component
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.vehicle_state = vehicle_state

        self.master: Optional[Any] = None
        self.target_system = 1
        self.target_component = 1

        self.is_connected: bool = False
        self.is_armed: bool = False
        self.flight_mode: str = "DISCONNECTED"
        self.last_heartbeat_time: float = 0.0

        # High-Rate Telemetry States
        self.pitch_rad: float = 0.0        # Controller convention: Pitch > 0 nose down
        self.roll_rad: float = 0.0         # Roll > 0 right wing down
        self.yaw_deg: float = 0.0          # Compass heading [0, 360)
        self.yaw_rad: float = 0.0
        self.roll_rate_rad_s: float = 0.0
        self.pitch_rate_rad_s: float = 0.0
        self.yaw_rate_rad_s: float = 0.0

        # Kinematic Local Velocities (NED)
        self.vx_m_s: float = 0.0           # North velocity
        self.vy_m_s: float = 0.0           # East velocity
        self.vz_m_s: float = 0.0           # Down velocity (positive descending)
        self.has_local_pos: bool = False

        # Altitude
        self.altitude_rel_m: float = 0.0   # Relative altitude above home/takeoff
        self.altitude_amsl_m: float = 0.0  # Above Mean Sea Level
        self.distance_sensor_m: Optional[float] = None

        # Thread-safe ring buffer for camera exposure & hardware latency matching
        self.attitude_history = deque(maxlen=300)
        self.lock = threading.Lock()

        # Background thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None

    @staticmethod
    def normalize_connection_string(conn_str: str, default_baud: int = 921600) -> Tuple[str, int]:
        """
        Normalizes MAVLink connection URIs across PyMAVLink and legacy MAVSDK conventions:
        - 'serial:///dev/ttyTHS0:921600' -> ('/dev/ttyTHS0', 921600)
        - '/dev/ttyTHS0' -> ('/dev/ttyTHS0', default_baud)
        - 'udp://:14540' -> ('udpin:0.0.0.0:14540', default_baud)
        - 'udp://0.0.0.0:14550' -> ('udpin:0.0.0.0:14550', default_baud)
        - 'udp:127.0.0.1:14550' -> ('udpin:0.0.0.0:14550', default_baud)
        - 'udpin:0.0.0.0:14550' -> ('udpin:0.0.0.0:14550', default_baud)
        - 'tcp:127.0.0.1:5760' -> ('tcp:127.0.0.1:5760', default_baud)
        """
        s = conn_str.strip()
        baud = default_baud

        # Strip serial:// prefix
        if s.startswith("serial://"):
            s = s[len("serial://"):]
            if ":" in s:
                parts = s.rsplit(":", 1)
                s = parts[0]
                try:
                    baud = int(parts[1])
                except ValueError:
                    pass
            return s, baud

        # Convert udp:// format
        if s.startswith("udp://"):
            rest = s[len("udp://"):]
            if rest.startswith(":"):
                port = rest[1:]
                return f"udpin:0.0.0.0:{port}", baud
            elif ":" in rest:
                host, port = rest.split(":", 1)
                # For companion listening to Pixhawk broadcast, udpin is standard
                if host in ("0.0.0.0", "", "localhost", "127.0.0.1"):
                    return f"udpin:0.0.0.0:{port}", baud
                return f"udpin:{host}:{port}", baud

        # Bare port e.g. ":14550" or "14550"
        if s.startswith(":") and s[1:].isdigit():
            return f"udpin:0.0.0.0:{s[1:]}", baud

        # Serial port with embedded baud e.g. /dev/ttyTHS0:921600
        if (s.startswith("/dev/") or s.startswith("COM")) and ":" in s:
            parts = s.rsplit(":", 1)
            s = parts[0]
            try:
                baud = int(parts[1])
            except ValueError:
                pass

        return s, baud

    def connect(self, timeout_s: float = 10.0) -> bool:
        """
        Connects to the ArduPilot Pixhawk flight controller and waits for the first heartbeat.
        """
        if not PYMAVLINK_AVAILABLE:
            print("[ARDUPILOT] Error: 'pymavlink' is not installed in the current environment.")
            return False

        if not self.connection_string:
            print("[ARDUPILOT] No connection string provided. Autopilot bridge inactive.")
            return False

        clean_conn, baud = self.normalize_connection_string(self.connection_string, self.baud)
        self.baud = baud
        print(f"[ARDUPILOT] Opening MAVLink connection to {clean_conn} (baud={self.baud})...")
        try:
            if clean_conn.startswith(("/dev/", "COM")):
                self.master = mavutil.mavlink_connection(
                    clean_conn,
                    baud=self.baud,
                    source_system=self.source_system,
                    source_component=self.source_component,
                    autoreconnect=True,
                )
            else:
                self.master = mavutil.mavlink_connection(
                    clean_conn,
                    source_system=self.source_system,
                    source_component=self.source_component,
                    autoreconnect=True,
                )
        except Exception as e:
            print(f"[ARDUPILOT] Connection failed: {e}")
            return False

        # Wait for initial heartbeat
        print(f"[ARDUPILOT] Waiting up to {timeout_s:.1f}s for vehicle heartbeat...")
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            msg = self.master.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
            if msg is not None:
                self.target_system = self.master.target_system
                self.target_component = self.master.target_component
                self.last_heartbeat_time = time.time()
                self.is_connected = True
                self._update_heartbeat(msg)
                print(f"[ARDUPILOT] ✓ Heartbeat received from System {self.target_system}, Component {self.target_component}!")
                print(f"[ARDUPILOT] Autopilot Type: {msg.autopilot} | Vehicle Type: {msg.type} | Mode: {self.flight_mode}")
                break

        if not self.is_connected:
            print("[ARDUPILOT] Timed out waiting for heartbeat. Check physical serial wiring or UDP port.")
            return False

        # Configure vehicle message intervals for high-rate visual servoing
        self._request_high_rate_streams()

        # Start background receive and heartbeat dispatch threads
        self._running = True
        self._thread = threading.Thread(target=self._telemetry_worker, daemon=True, name="ArduPilotTelem")
        self._thread.start()

        self._heartbeat_thread = threading.Thread(target=self._heartbeat_sender, daemon=True, name="ArduPilotHb")
        self._heartbeat_thread.start()

        return True

    def _request_high_rate_streams(self):
        """Requests 50 Hz attitude, 30 Hz velocity, and 20 Hz altitude from ArduCopter."""
        if not self.master:
            return

        print("[ARDUPILOT] Requesting high-rate MAVLink streams (Attitude: 50Hz, Vel: 30Hz, Pos: 20Hz)...")
        streams = [
            (mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 50.0),            # 50 Hz attitude
            (mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 30.0),   # 30 Hz local velocity
            (mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 20.0),  # 20 Hz altitude
            (mavutil.mavlink.MAVLINK_MSG_ID_DISTANCE_SENSOR, 20.0),      # 20 Hz rangefinder
        ]
        for msg_id, rate_hz in streams:
            interval_us = int(1e6 / rate_hz)
            self.master.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                msg_id,
                interval_us,
                0, 0, 0, 0, 0
            )

    def _heartbeat_sender(self):
        """Dispatches companion computer heartbeat at 2 Hz to maintain MAVLink link health."""
        while self._running:
            try:
                if self.master:
                    self.master.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                        0, 0,
                        mavutil.mavlink.MAV_STATE_ACTIVE
                    )
            except Exception:
                pass
            time.sleep(0.5)

    def _update_heartbeat(self, msg):
        """Parses flight mode and armed state from HEARTBEAT message."""
        with self.lock:
            self.is_armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            # ArduCopter mode string resolution via pymavlink
            mode_str = self.master.flightmode if hasattr(self.master, "flightmode") else None
            if not mode_str:
                custom_mode = msg.custom_mode
                mode_str = mavutil.mode_mapping_acm.get(custom_mode, f"MODE_{custom_mode}")
            self.flight_mode = str(mode_str).upper()

        if self.vehicle_state is not None:
            self.vehicle_state.is_connected = True
            self.vehicle_state.is_armed = self.is_armed
            self.vehicle_state.flight_mode = self.flight_mode

    def _telemetry_worker(self):
        """High-frequency non-blocking background thread reading MAVLink telemetry."""
        while self._running:
            try:
                msg = self.master.recv_match(blocking=False)
                if msg is None:
                    time.sleep(0.002)
                    continue

                msg_type = msg.get_type()

                if msg_type == "HEARTBEAT":
                    self.last_heartbeat_time = time.time()
                    self._update_heartbeat(msg)

                elif msg_type == "ATTITUDE":
                    # Standard Aerospace MAVLink convention:
                    #   Pitch > 0 is nose UP, Roll > 0 is right wing DOWN.
                    # Controller visual-servoing convention:
                    #   Pitch > 0 is nose DOWN (forward acceleration thrust vector).
                    t_now = time.perf_counter()
                    p_rad = -float(msg.pitch)
                    r_rad = float(msg.roll)
                    y_rad = float(msg.yaw)
                    y_deg = float((math.degrees(y_rad) + 360.0) % 360.0)

                    with self.lock:
                        self.pitch_rad = p_rad
                        self.roll_rad = r_rad
                        self.yaw_deg = y_deg
                        self.yaw_rad = y_rad
                        self.roll_rate_rad_s = float(msg.rollspeed)
                        self.pitch_rate_rad_s = -float(msg.pitchspeed)
                        self.yaw_rate_rad_s = float(msg.yawspeed)
                        self.attitude_history.append((t_now, p_rad, r_rad, y_deg))

                    if self.vehicle_state is not None:
                        self.vehicle_state.record_attitude(t_now, p_rad, r_rad, y_deg)

                elif msg_type == "LOCAL_POSITION_NED":
                    vn = float(msg.vx)
                    ve = float(msg.vy)
                    vd = float(msg.vz)
                    with self.lock:
                        self.vx_m_s = vn
                        self.vy_m_s = ve
                        self.vz_m_s = vd
                        self.has_local_pos = True

                    if self.vehicle_state is not None:
                        self.vehicle_state.vx_m_s = vn
                        self.vehicle_state.vy_m_s = ve
                        self.vehicle_state.vz_m_s = vd
                        if abs(vn) > 1e-4 or abs(ve) > 1e-4:
                            self.vehicle_state.has_gps_vel = True

                elif msg_type == "GLOBAL_POSITION_INT":
                    rel_alt = float(msg.relative_alt) / 1000.0
                    amsl_alt = float(msg.alt) / 1000.0
                    with self.lock:
                        self.altitude_rel_m = rel_alt
                        self.altitude_amsl_m = amsl_alt

                    if self.vehicle_state is not None:
                        self.vehicle_state.altitude_rel_m = rel_alt
                        self.vehicle_state.altitude_amsl_m = amsl_alt

                elif msg_type == "DISTANCE_SENSOR":
                    dist_m = float(msg.current_distance) / 100.0
                    with self.lock:
                        self.distance_sensor_m = dist_m

                    if self.vehicle_state is not None:
                        self.vehicle_state.distance_sensor_m = dist_m

            except Exception as e:
                time.sleep(0.01)

    # ==========================================================================
    # Flight Mode Management & Pilot Handshake
    # ==========================================================================

    @property
    def is_guided(self) -> bool:
        """Returns True if ArduCopter is actively in GUIDED mode."""
        with self.lock:
            # Check link liveness
            if time.time() - self.last_heartbeat_time > self.heartbeat_timeout_s:
                return False
            return self.flight_mode == "GUIDED"

    def set_mode(self, mode_name: str) -> bool:
        """Commands ArduCopter into the specified flight mode (e.g. GUIDED, LOITER, BRAKE)."""
        if not self.master:
            return False

        clean_name = mode_name.upper().strip()
        mode_map = self.master.mode_mapping()
        if not mode_map or clean_name not in mode_map:
            print(f"[ARDUPILOT] Unknown mode '{clean_name}'. Available modes: {list(mode_map.keys()) if mode_map else 'None'}")
            return False

        mode_id = mode_map[clean_name]
        print(f"[ARDUPILOT] Commanding flight mode switch -> {clean_name} (ID: {mode_id})...")
        self.master.mav.command_long_send(
            self.target_system,
            self.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
            0, 0, 0, 0, 0
        )
        return True

    def set_mode_guided(self) -> bool:
        """Transitions into autonomous GUIDED pursuit mode."""
        return self.set_mode("GUIDED")

    def set_mode_loiter(self) -> bool:
        """Transitions into pilot manual LOITER hover holding mode."""
        return self.set_mode("LOITER")

    def set_mode_brake(self) -> bool:
        """Transitions into immediate BRAKE mode."""
        return self.set_mode("BRAKE")

    # ==========================================================================
    # High-Rate Velocity Guidance Dispatch
    # ==========================================================================

    def send_velocity_body(
        self,
        v_fwd: float,
        v_right: float,
        v_down: float,
        yawspeed_deg_s: float,
    ) -> bool:
        """
        Dispatches 4-DOF body-frame velocity setpoint to ArduPilot via SET_POSITION_TARGET_LOCAL_NED (#84).

        Args:
            v_fwd: Forward body velocity setpoint in m/s (+ forward, - backward)
            v_right: Lateral strafe body velocity setpoint in m/s (+ right, - left)
            v_down: Vertical body velocity setpoint in m/s (+ descend, - climb)
            yawspeed_deg_s: Turn rate setpoint in deg/s (+ clockwise / nose-right)
        """
        if not self.master or not self.is_connected:
            return False

        # Inhibit setpoint streaming if pilot has manually switched modes away from GUIDED
        if not self.is_guided:
            return False

        yaw_rate_rad_s = math.radians(yawspeed_deg_s)

        self.master.mav.set_position_target_local_ned_send(
            0,                                      # time_boot_ms (0 = auto)
            self.target_system,
            self.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,     # Coordinate frame: Aircraft Body-NED
            TYPEMASK_VELOCITY_YAWRATE,              # Ignore positions & accels, control vx,vy,vz,yaw_rate
            0.0, 0.0, 0.0,                          # Positions (ignored)
            float(v_fwd),                           # vx (forward) in m/s
            float(v_right),                         # vy (right) in m/s
            float(v_down),                          # vz (down) in m/s
            0.0, 0.0, 0.0,                          # Accelerations (ignored)
            0.0,                                    # Yaw angle (ignored)
            float(yaw_rate_rad_s)                   # Yaw rate in rad/s
        )
        return True

    def send_hold_hover(self) -> bool:
        """Dispatches zero-velocity braking command to arrest all motion."""
        return self.send_velocity_body(0.0, 0.0, 0.0, 0.0)

    # ==========================================================================
    # Attitude History Interpolation
    # ==========================================================================

    def get_interpolated_attitude(self, target_t: float) -> Tuple[float, float, float]:
        """
        Retrieves drone orientation (pitch_rad, roll_rad, yaw_deg) matching camera exposure time target_t.
        """
        with self.lock:
            if not self.attitude_history:
                return self.pitch_rad, self.roll_rad, self.yaw_deg
            if target_t >= self.attitude_history[-1][0]:
                return self.attitude_history[-1][1], self.attitude_history[-1][2], self.attitude_history[-1][3]
            if target_t <= self.attitude_history[0][0]:
                return self.attitude_history[0][1], self.attitude_history[0][2], self.attitude_history[0][3]
            hist = list(self.attitude_history)

        idx = bisect.bisect_right(hist, (target_t, float('inf')))
        t0, p0, r0, y0 = hist[idx - 1]
        t1, p1, r1, y1 = hist[idx]
        dt = max(1e-6, t1 - t0)
        alpha = float(np.clip((target_t - t0) / dt, 0.0, 1.0))

        # Handle yaw wrapping
        dyaw = (y1 - y0 + 180.0) % 360.0 - 180.0
        interp_yaw = y0 + alpha * dyaw
        interp_pitch = p0 + alpha * (p1 - p0)
        interp_roll = r0 + alpha * (r1 - r0)
        return float(interp_pitch), float(interp_roll), float(interp_yaw)

    def close(self):
        """Gracefully shuts down bridge, sends zero-velocity command, and terminates connection."""
        print("[ARDUPILOT] Closing autopilot bridge...")
        self._running = False
        if self.master and self.is_connected:
            try:
                # Send zero-velocity brake and hand back to LOITER
                self.send_hold_hover()
                self.set_mode_loiter()
            except Exception:
                pass
            try:
                self.master.close()
            except Exception:
                pass
        self.is_connected = False
        if self.vehicle_state is not None:
            self.vehicle_state.is_connected = False
        print("[ARDUPILOT] Bridge closed safely.")
