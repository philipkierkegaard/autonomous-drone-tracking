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

# ArduPilot Recognized Safe Fallback Modes
# Any pilot manual flight mode or autonomous failsafe recovery mode:
SAFE_FALLBACK_MODES = frozenset({
    "LOITER",
    "BRAKE",
    "ALT_HOLD",
    "POSHOLD",
    "RTL",
    "LAND",
    "SMART_RTL",
    "STABILIZE",
    "AUTO",
    "QLOITER",
    "QLAND",
    "QRTL",
})


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
        dry_run: bool = False,
        auto_reengage: bool = False,   # If False, pilot override latches until explicit authorize_autonomy()
    ):
        self.connection_string = connection_string
        self.baud = baud
        self.source_system = source_system
        self.source_component = source_component
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.vehicle_state = vehicle_state
        self.dry_run = dry_run
        self.auto_reengage = auto_reengage

        self.master: Optional[Any] = None
        self.target_system = 1
        self.target_component = 1

        self.is_connected: bool = False
        self.is_armed: bool = False
        self.flight_mode: str = "DISCONNECTED"
        self.last_heartbeat_time: float = 0.0

        # Pilot authority & autonomy authorization latch
        self.pilot_override_active: bool = False
        self.autonomous_authorized: bool = False
        self.last_guided_time: float = 0.0

        # ArduPilot failsafe parameters & diagnostic state
        self.fs_gcs_enable: Optional[int] = None
        self.guid_timeout: Optional[float] = None
        self.fs_options: Optional[int] = None
        self.parameters: Dict[str, float] = {}
        self.last_status_text: str = ""

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

        # Query key failsafe parameters to verify failsafe readiness
        self._request_failsafe_parameters()

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

    def _request_failsafe_parameters(self):
        """Requests key failsafe parameters from ArduCopter to verify onboard failsafe configuration."""
        if not self.master:
            return

        params_to_query = ["FS_GCS_ENABLE", "GUID_TIMEOUT", "FS_OPTIONS"]
        for p in params_to_query:
            try:
                if hasattr(self.master, "param_fetch_one"):
                    self.master.param_fetch_one(p)
                elif hasattr(self.master, "mav") and hasattr(self.master.mav, "param_request_read_send"):
                    self.master.mav.param_request_read_send(
                        self.target_system,
                        self.target_component,
                        p.encode("ascii"),
                        -1
                    )
            except Exception:
                pass

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
        """Parses flight mode and armed state from HEARTBEAT message and detects pilot override or link interruption."""
        now = time.time()
        with self.lock:
            prev_mode = self.flight_mode
            time_since_last_hb = (now - self.last_heartbeat_time) if self.last_heartbeat_time > 0 else 0.0
            had_link_loss = (self.last_heartbeat_time > 0 and time_since_last_hb > self.heartbeat_timeout_s)

            # Priority 4: If connection is recovering from a heartbeat blackout, revoke autonomy
            if had_link_loss:
                if self.autonomous_authorized:
                    self.autonomous_authorized = False
                    self.pilot_override_active = True
                    print(f"\n[ARDUPILOT] ⚠ Connection recovered after {time_since_last_hb:.1f}s interruption! Autonomy revoked. Explicit reauthorization required.")

            self.last_heartbeat_time = now
            self.is_armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            # ArduCopter mode string resolution via pymavlink
            mode_str = self.master.flightmode if hasattr(self.master, "flightmode") else None
            if not mode_str:
                custom_mode = msg.custom_mode
                mode_str = mavutil.mode_mapping_acm.get(custom_mode, f"MODE_{custom_mode}")
            new_mode = str(mode_str).upper()
            self.flight_mode = new_mode

            # Pilot takeover & mode transition state machine:
            if prev_mode == "GUIDED" and new_mode != "GUIDED":
                # Pilot manually changed mode away from GUIDED (e.g. LOITER, ALT_HOLD, STABILIZE, RTL)
                self.pilot_override_active = True
                self.autonomous_authorized = False
                print(f"\n[ARDUPILOT] Pilot RC manual takeover detected -> {new_mode}! Autonomy disengaged.")
            elif prev_mode != "GUIDED" and new_mode == "GUIDED":
                self.last_guided_time = now
                if self.auto_reengage and not had_link_loss:
                    self.autonomous_authorized = True
                    self.pilot_override_active = False
                    print("[ARDUPILOT] Vehicle entered GUIDED mode. Autonomy automatically re-engaged.")
                elif not self.pilot_override_active and not had_link_loss:
                    self.autonomous_authorized = True
                    print("[ARDUPILOT] Vehicle entered GUIDED mode. Autonomy authorized.")
                else:
                    # Latched override or link loss: vehicle entered GUIDED, but requires explicit reauthorization
                    self.autonomous_authorized = False
                    print("[ARDUPILOT] Vehicle in GUIDED mode, but override/failsafe is latched. Explicit reauthorization required.")

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

                elif msg_type == "PARAM_VALUE":
                    param_id = msg.param_id
                    if isinstance(param_id, bytes):
                        param_id = param_id.decode("ascii", errors="ignore").rstrip("\x00")
                    val = float(msg.param_value)
                    with self.lock:
                        self.parameters[param_id] = val
                        if param_id == "FS_GCS_ENABLE":
                            self.fs_gcs_enable = int(val)
                        elif param_id == "GUID_TIMEOUT":
                            self.guid_timeout = val
                        elif param_id == "FS_OPTIONS":
                            self.fs_options = int(val)

                elif msg_type == "STATUSTEXT":
                    text = msg.text
                    if isinstance(text, bytes):
                        text = text.decode("ascii", errors="ignore").rstrip("\x00")
                    with self.lock:
                        self.last_status_text = text
                    lower_text = text.lower()
                    if any(kw in lower_text for kw in ("failsafe", "flight mode", "error", "warning", "panic", "crash")):
                        print(f"[ARDUPILOT] Autopilot notice: {text}")

            except Exception as e:
                time.sleep(0.01)

    # ==========================================================================
    # Flight Mode Management & Pilot Handshake
    # ==========================================================================

    @property
    def is_guided(self) -> bool:
        """Returns True if ArduCopter is actively in GUIDED mode."""
        with self.lock:
            if time.time() - self.last_heartbeat_time > self.heartbeat_timeout_s:
                return False
            return self.flight_mode == "GUIDED"

    @property
    def is_guided_and_authorized(self) -> bool:
        """
        Returns True if vehicle is actively in GUIDED mode AND autonomy is authorized.
        If pilot manual takeover or link loss occurred, remains False until explicitly authorized.
        """
        with self.lock:
            # Check link liveness (Priority 4: actively revoke authorization on link interruption)
            if time.time() - self.last_heartbeat_time > self.heartbeat_timeout_s:
                if self.autonomous_authorized:
                    self.autonomous_authorized = False
                    self.pilot_override_active = True
                    print(f"\n[ARDUPILOT] ⚠ Heartbeat timeout (> {self.heartbeat_timeout_s:.1f}s). Autonomy revoked due to link interruption.")
                return False

            if self.flight_mode != "GUIDED":
                return False

            return self.autonomous_authorized

    def authorize_autonomy(self, controller: Optional[Any] = None) -> bool:
        """
        Explicitly re-authorizes autonomous velocity commands following pilot manual takeover or link recovery.
        Enforces strict safety preconditions (Priority 3):
        1. Vehicle must be actively connected and link healthy.
        2. Vehicle must be reported in GUIDED mode.
        3. Flushes controller history to prevent derivative kicks or velocity discontinuities.
        """
        with self.lock:
            # Check link liveness
            if time.time() - self.last_heartbeat_time > self.heartbeat_timeout_s:
                print(f"[ARDUPILOT] ⚠ Cannot authorize autonomy: MAVLink heartbeat timed out (> {self.heartbeat_timeout_s:.1f}s).")
                return False

            if self.flight_mode != "GUIDED":
                print(f"[ARDUPILOT] ⚠ Cannot authorize autonomy: vehicle is in '{self.flight_mode}', not GUIDED mode.")
                return False

            self.pilot_override_active = False
            self.autonomous_authorized = True
            print("[ARDUPILOT] ✓ Autonomous velocity commands explicitly re-authorized by operator.")

        if controller is not None and hasattr(controller, "reset"):
            controller.reset()
            print("[ARDUPILOT] ✓ Controller history flushed clean (integrators, derivatives, and commands reset).")

        return True

    def set_mode(
        self,
        mode_name: str,
        timeout_s: float = 3.0,
        abort_on_override: bool = False,
    ) -> bool:
        """
        Commands ArduCopter into the specified flight mode (e.g. GUIDED, LOITER, BRAKE)
        and verifies execution against incoming telemetry.
        Respects dry_run isolation (does not transmit flight-altering packets).

        Args:
            mode_name: Desired flight mode string.
            timeout_s: Maximum seconds to wait for heartbeat confirmation.
            abort_on_override: If True, aborts the mode change if pilot takeover is detected,
                              preserving pilot manual authority without transmitting conflicting commands.
        """
        clean_name = mode_name.upper().strip()

        if self.dry_run:
            print(f"[ARDUPILOT] [DRY-RUN] Simulating flight mode switch -> {clean_name} (no packet transmitted).")
            with self.lock:
                self.flight_mode = clean_name
                if clean_name == "GUIDED":
                    self.autonomous_authorized = True
                    self.pilot_override_active = False
            if self.vehicle_state is not None:
                self.vehicle_state.flight_mode = clean_name
            return True

        if not self.master:
            return False

        # Pre-dispatch pilot override check (eliminates race where pilot switches modes right before dispatch)
        with self.lock:
            if abort_on_override and self.pilot_override_active:
                print(f"[ARDUPILOT] Mode switch to {clean_name} aborted before dispatch: Pilot override active ({self.flight_mode}). Handover to pilot preserved.")
                return True
            if self.flight_mode == clean_name:
                print(f"[ARDUPILOT] Vehicle is already in mode '{clean_name}'. No command needed.")
                return True

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

        # Mode-change verification: wait for incoming telemetry heartbeat to confirm transition
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            with self.lock:
                # Concurrency guard: If pilot overrides during pending transition, immediately abort
                if abort_on_override and self.pilot_override_active:
                    print(f"[ARDUPILOT] Mode switch to {clean_name} aborted: Pilot manual override detected ({self.flight_mode}). Handover to pilot preserved.")
                    return True
                if self.flight_mode == clean_name:
                    print(f"[ARDUPILOT] ✓ Mode switch to {clean_name} confirmed by vehicle heartbeat ({time.time() - t0:.2f}s).")
                    return True
            time.sleep(0.05)

        print(f"[ARDUPILOT] ⚠ Mode switch to {clean_name} timed out after {timeout_s:.1f}s (Current mode: {self.flight_mode}).")
        return False

    def set_mode_guided(self, timeout_s: float = 3.0) -> bool:
        """Transitions into autonomous GUIDED pursuit mode with execution verification."""
        return self.set_mode("GUIDED", timeout_s=timeout_s, abort_on_override=False)

    def set_mode_loiter(self, timeout_s: float = 3.0, abort_on_override: bool = False) -> bool:
        """Transitions into pilot manual LOITER hover holding mode with execution verification."""
        return self.set_mode("LOITER", timeout_s=timeout_s, abort_on_override=abort_on_override)

    def set_mode_brake(self, timeout_s: float = 3.0, abort_on_override: bool = False) -> bool:
        """Transitions into immediate BRAKE mode with execution verification."""
        return self.set_mode("BRAKE", timeout_s=timeout_s, abort_on_override=abort_on_override)

    def fetch_parameter(self, param_id: str, timeout_s: float = 1.0) -> Optional[float]:
        """Queries and waits for a specific ArduPilot parameter over MAVLink."""
        clean_id = param_id.strip().upper()
        with self.lock:
            if clean_id in self.parameters:
                return self.parameters[clean_id]

        if not self.master or self.dry_run:
            return None

        try:
            if hasattr(self.master, "param_fetch_one"):
                self.master.param_fetch_one(clean_id)
            elif hasattr(self.master, "mav") and hasattr(self.master.mav, "param_request_read_send"):
                self.master.mav.param_request_read_send(
                    self.target_system,
                    self.target_component,
                    clean_id.encode("ascii"),
                    -1
                )
        except Exception:
            return None

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            with self.lock:
                if clean_id in self.parameters:
                    return self.parameters[clean_id]
            time.sleep(0.05)
        return None

    def verify_failsafe_outcome(self) -> Tuple[bool, str]:
        """
        Verifies the actual vehicle flight mode outcome against recognized safe fallback
        modes and ArduPilot's configured failsafe behavior (FS_GCS_ENABLE, GUID_TIMEOUT).

        Returns:
            (is_safe, details_str):
                is_safe: True if vehicle is confirmed in a safe fallback mode (LOITER, BRAKE, ALT_HOLD, etc.)
                details_str: Human-readable diagnostic description of failsafe readiness.
        """
        with self.lock:
            current_mode = self.flight_mode
            gcs_fs = self.fs_gcs_enable
            guid_to = self.guid_timeout

        if current_mode in SAFE_FALLBACK_MODES:
            return True, f"Vehicle verified in safe fallback mode '{current_mode}'."

        if current_mode == "GUIDED":
            fs_details = []
            has_configured_failsafe = False

            if guid_to is not None and guid_to > 0:
                fs_details.append(f"GUID_TIMEOUT={guid_to:.1f}s (guided timeout will brake vehicle)")
                has_configured_failsafe = True
            elif guid_to is None:
                fs_details.append("GUID_TIMEOUT unqueried (ArduCopter default is 3.0s -> BRAKE)")
                has_configured_failsafe = True
            else:
                fs_details.append("GUID_TIMEOUT disabled (0.0s)")

            gcs_actions = {
                1: "RTL",
                2: "Continue in Auto",
                3: "SmartRTL/RTL",
                4: "SmartRTL/Land",
                5: "Land"
            }
            if gcs_fs is not None and gcs_fs in gcs_actions:
                fs_details.append(f"FS_GCS_ENABLE={gcs_fs} ({gcs_actions[gcs_fs]} on heartbeat loss)")
                has_configured_failsafe = True
            elif gcs_fs == 0:
                fs_details.append("FS_GCS_ENABLE=0 (GCS link-loss failsafe disabled)")
            else:
                fs_details.append("FS_GCS_ENABLE unqueried")

            details_msg = "; ".join(fs_details)
            if has_configured_failsafe:
                return False, f"Vehicle remains in GUIDED mode. Primary handover failed, but ArduPilot configured failsafe is armed: {details_msg}."
            else:
                return False, f"CRITICAL: Vehicle remains in GUIDED mode and NO active failsafe was verified ({details_msg})! Immediate manual pilot takeover required!"

        return False, f"Vehicle in non-standard mode '{current_mode}'."

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
        Enforces strict dry-run isolation and pilot authority gating.

        Args:
            v_fwd: Forward body velocity setpoint in m/s (+ forward, - backward)
            v_right: Lateral strafe body velocity setpoint in m/s (+ right, - left)
            v_down: Vertical body velocity setpoint in m/s (+ descend, - climb)
            yawspeed_deg_s: Turn rate setpoint in deg/s (+ clockwise / nose-right)
        """
        if self.dry_run:
            return False

        if not self.master or not self.is_connected:
            return False

        # Inhibit setpoint streaming if pilot has manually switched modes or autonomy not authorized
        if not self.is_guided_and_authorized:
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

    def close(self) -> bool:
        """
        Gracefully shuts down bridge, strictly preserving pilot manual flight mode and enforcing dry-run isolation.
        Keeps telemetry active during the handover handshake until verified or timed out.
        If primary handover (LOITER) times out or fails, attempts emergency fallback (BRAKE) and verifies
        the actual aircraft outcome against ArduPilot's configured failsafe behavior.
        Returns True if shutdown and any required handover completed cleanly, False if handover failed or timed out.
        """
        print("[ARDUPILOT] Closing autopilot bridge...")
        handover_success = True

        # Keep self._running = True during handover so telemetry worker processes confirmation heartbeats!
        if self.master and self.is_connected:
            if not self.dry_run:
                with self.lock:
                    current_mode = self.flight_mode
                    pilot_override = self.pilot_override_active
                    owns_autonomy = (current_mode == "GUIDED" and self.autonomous_authorized and not pilot_override)

                # Only initiate autonomous handover if application actively owned control
                if owns_autonomy:
                    print("[ARDUPILOT] Application holds active GUIDED autonomy. Arresting velocity and transitioning to LOITER...")
                    try:
                        # Re-check pilot override before sending hold hover to eliminate race condition
                        with self.lock:
                            pilot_now = self.pilot_override_active or self.flight_mode != "GUIDED"

                        if pilot_now:
                            print(f"[ARDUPILOT] Pilot manual override detected ({self.flight_mode}). Handover to pilot preserved.")
                            handover_success = True
                        else:
                            self.send_hold_hover()
                            handover_success = self.set_mode_loiter(timeout_s=2.0, abort_on_override=True)

                            # If pilot took over during set_mode_loiter wait
                            with self.lock:
                                if self.pilot_override_active:
                                    print(f"[ARDUPILOT] Pilot manual takeover detected during handover ({self.flight_mode}). Pilot in command.")
                                    handover_success = True

                            # If primary LOITER handover failed or timed out:
                            if not handover_success:
                                print(f"[ARDUPILOT] ⚠ Primary handover to LOITER failed or timed out! Current mode: '{self.flight_mode}'.")

                                # Attempt emergency fallback mode (BRAKE) if still in GUIDED
                                with self.lock:
                                    still_guided = (self.flight_mode == "GUIDED" and not self.pilot_override_active)

                                if still_guided:
                                    print("[ARDUPILOT] Attempting emergency fallback transition to BRAKE mode...")
                                    brake_ok = self.set_mode_brake(timeout_s=1.0, abort_on_override=True)
                                    if brake_ok:
                                        print(f"[ARDUPILOT] ✓ Emergency fallback successful: Vehicle entered BRAKE mode.")
                                    else:
                                        print(f"[ARDUPILOT] ⚠ Emergency fallback to BRAKE also failed or timed out! Current mode: '{self.flight_mode}'.")

                                # Verify actual outcome against ArduPilot's configured failsafe behavior
                                is_safe, failsafe_desc = self.verify_failsafe_outcome()
                                print(f"[ARDUPILOT] Failsafe verification: {failsafe_desc}")

                                if is_safe:
                                    handover_success = True
                                else:
                                    print(f"[ARDUPILOT] ⚠ WARNING: Severing MAVLink connection while vehicle mode is '{self.flight_mode}'.")
                    except Exception as e:
                        print(f"[ARDUPILOT] ⚠ Shutdown command error: {e}")
                        handover_success = False
                else:
                    if current_mode == "GUIDED":
                        print("[ARDUPILOT] Vehicle is in GUIDED mode, but application autonomy is not authorized (standby/pilot control). Preserving vehicle state without forcing LOITER.")
                    else:
                        print(f"[ARDUPILOT] Pilot manual mode '{current_mode}' active at shutdown. Preserving pilot authority (no LOITER switch commanded).")
            else:
                print("[ARDUPILOT] [DRY-RUN] Shutdown: No flight commands transmitted.")

        # Stop background threads AFTER handover sequence completes
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.3)
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=0.3)

        if self.master:
            try:
                self.master.close()
            except Exception:
                pass

        self.is_connected = False
        if self.vehicle_state is not None:
            self.vehicle_state.is_connected = False

        with self.lock:
            final_mode = self.flight_mode

        if handover_success:
            print(f"[ARDUPILOT] Bridge closed safely (final mode: '{final_mode}').")
        else:
            print(f"[ARDUPILOT] ⚠ Bridge closed with handover WARNING: vehicle was not confirmed in a safe fallback mode (current mode: '{final_mode}').")

        return handover_success
