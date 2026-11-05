#!/usr/bin/env python3
"""
ArduPilot Arming Diagnostic Tool for NVIDIA Jetson & Companion Computers.
DTU Bachelor Thesis — Autonomous Drone Pursuit System.

Connects to the Pixhawk / ArduPilot flight controller via UART (/dev/ttyTHS0,
/dev/ttyTHS1), USB (/dev/ttyACM0, /dev/ttyUSB0), or UDP, queries pre-arm checks,
captures real-time STATUSTEXT messages, audits sensor health bitmasks, EKF
variances, GPS fix, RC throttle, and provides clear, actionable instructions
explaining why the drone refuses to arm.

Usage:
    # Auto-detects connected port and runs diagnostic audit:
    python scripts/diagnose_arming.py

    # Explicit UART connection on Jetson:
    python scripts/diagnose_arming.py --connect /dev/ttyTHS0:921600

    # Attempt a live arm command to capture exact autopilot denial reason:
    python scripts/diagnose_arming.py --arm

    # Continuous live monitoring loop (watch pre-arm checks clear in real-time):
    python scripts/diagnose_arming.py --monitor
"""

from __future__ import annotations

import os
import sys
import time
import glob
import argparse
from typing import Optional, Dict, Any, List, Tuple

try:
    from pymavlink import mavutil
    PYMAVLINK_AVAILABLE = True
except ImportError:
    PYMAVLINK_AVAILABLE = False


# ANSI Terminal Colors
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    @classmethod
    def strip_if_no_color(cls):
        """Disable color codes if running in non-interactive environment."""
        if not sys.stdout.isatty():
            for attr in dir(cls):
                if not attr.startswith("__") and isinstance(getattr(cls, attr), str):
                    setattr(cls, attr, "")


# ArduCopter Flight Modes
ARDUCOPTER_MODES = {
    0: "STABILIZE",
    1: "ACRO",
    2: "ALT_HOLD",
    3: "AUTO",
    4: "GUIDED",
    5: "LOITER",
    6: "RTL",
    7: "CIRCLE",
    9: "LAND",
    11: "DRIFT",
    13: "SPORT",
    14: "FLIP",
    15: "AUTOTUNE",
    16: "POSHOLD",
    17: "BRAKE",
    18: "THROW",
    19: "AVOID_ADSB",
    20: "GUIDED_NOGPS",
    21: "SMART_RTL",
    22: "FLOWHOLD",
    23: "FOLLOW",
    24: "ZIGZAG",
    25: "SYSTEMID",
    26: "AUTOROTATE",
    27: "AUTO_RTL",
}

# Flight modes requiring valid 3D GPS fix to arm
GPS_REQUIRED_MODES = frozenset({"GUIDED", "LOITER", "POSHOLD", "AUTO", "RTL", "SMART_RTL", "FOLLOW"})

# GPS Fix Type Names
GPS_FIX_TYPES = {
    0: "0 (No GPS Connected)",
    1: "1 (No Fix / Searching)",
    2: "2 (2D Fix)",
    3: "3 (3D Fix)",
    4: "4 (DGPS / SBAS)",
    5: "5 (RTK Float)",
    6: "6 (RTK Fixed)",
}

# MAVLink Sensor Bitmask Definitions (MAV_SYS_STATUS_SENSOR)
SENSOR_DEFINITIONS = [
    (1 << 0, "3D Gyroscope 1", "IMU (Rate Gyro)"),
    (1 << 1, "3D Accelerometer 1", "IMU (Accelerometer)"),
    (1 << 2, "3D Magnetometer 1", "Compass"),
    (1 << 3, "Absolute Pressure", "Barometer (Altitude)"),
    (1 << 4, "Differential Pressure", "Airspeed Sensor"),
    (1 << 5, "GPS Receiver", "Global Positioning"),
    (1 << 6, "Optical Flow", "Downward Vision"),
    (1 << 7, "Computer Vision Position", "VIO / External Odometry"),
    (1 << 8, "Distance Sensor", "Lidar / Sonar Rangefinder"),
    (1 << 10, "3D Angular Rate Control", "Rate Controller"),
    (1 << 11, "Attitude Stabilization", "Attitude Controller"),
    (1 << 12, "Yaw Position", "Yaw Alignment"),
    (1 << 13, "Z / Altitude Control", "Altitude Controller"),
    (1 << 14, "X/Y Position Control", "Horizontal Position Controller"),
    (1 << 15, "Motor Outputs / ESCs", "Motor Actuation"),
    (1 << 16, "RC Receiver", "Radio Control Signal"),
    (1 << 17, "3D Gyroscope 2", "Secondary Gyro"),
    (1 << 18, "3D Accelerometer 2", "Secondary Accel"),
    (1 << 19, "3D Magnetometer 2", "Secondary Compass"),
    (1 << 20, "Geofence", "Failsafe Geofence"),
    (1 << 21, "AHRS Subsystem", "Attitude & Heading Filter"),
    (1 << 22, "Terrain Database", "Terrain / Elevation Model"),
    (1 << 23, "Battery Monitor", "Power Module"),
    (1 << 24, "Proximity Sensor", "Obstacle Avoidance"),
]

# EKF Status Flags
EKF_FLAG_NAMES = [
    (1 << 0, "Attitude Estimate"),
    (1 << 1, "Velocity Horiz Estimate"),
    (1 << 2, "Velocity Vert Estimate"),
    (1 << 3, "Relative Horiz Pos"),
    (1 << 4, "Absolute Horiz Pos (GPS/Origin)"),
    (1 << 5, "Absolute Vert Pos"),
    (1 << 6, "Terrain Alt (AGL)"),
]

# Severity Label Mapping
SEVERITY_NAMES = {
    0: ("EMERGENCY", C.RED),
    1: ("ALERT", C.RED),
    2: ("CRITICAL", C.RED),
    3: ("ERROR", C.RED),
    4: ("WARNING", C.YELLOW),
    5: ("NOTICE", C.CYAN),
    6: ("INFO", C.WHITE),
    7: ("DEBUG", C.DIM),
}


def scan_candidate_ports() -> List[Tuple[str, int, str]]:
    """Scans and prioritizes possible MAVLink connection endpoints on Jetson/Linux."""
    candidates = []

    # 1. Jetson hardware UARTs
    jetson_uarts = [
        ("/dev/ttyTHS0", 921600, "Jetson 40-pin UART (ttyTHS0)"),
        ("/dev/ttyTHS1", 921600, "Jetson M.2 / Secondary UART (ttyTHS1)"),
        ("/dev/ttyTHS2", 921600, "Jetson UART 2 (ttyTHS2)"),
    ]
    for p, b, desc in jetson_uarts:
        if os.path.exists(p):
            candidates.append((p, b, desc))

    # 2. USB CDC ACM devices (Pixhawk directly connected via USB cable)
    for p in sorted(glob.glob("/dev/ttyACM*")):
        candidates.append((p, 115200, f"Pixhawk Native USB ({os.path.basename(p)})"))

    # 3. USB-to-UART FTDI adapters
    for p in sorted(glob.glob("/dev/ttyUSB*")):
        candidates.append((p, 921600, f"USB-Serial FTDI ({os.path.basename(p)})"))

    # 4. Standard local MAVLink UDP forwarder (MAVProxy, SITL, or GCS bridge)
    candidates.append(("udpin:0.0.0.0:14550", 115200, "MAVLink UDP Broadcast (14550)"))
    candidates.append(("udp:127.0.0.1:14550", 115200, "MAVLink UDP Localhost (14550)"))

    return candidates


def normalize_connection(conn_str: str, default_baud: int = 921600) -> Tuple[str, int]:
    """Parses port and baud from string e.g. /dev/ttyTHS0:921600."""
    s = conn_str.strip()
    baud = default_baud
    if s.startswith("serial://"):
        s = s[len("serial://"):]
    if ":" in s and not s.startswith("udp"):
        parts = s.rsplit(":", 1)
        s = parts[0]
        try:
            baud = int(parts[1])
        except ValueError:
            pass
    return s, baud


class ArmingDiagnostic:
    def __init__(self, connection_string: Optional[str] = None, baud: int = 921600, timeout_s: float = 6.0):
        self.requested_conn = connection_string
        self.default_baud = baud
        self.timeout_s = timeout_s
        self.master: Optional[Any] = None

        # Telemetry State Cache
        self.heartbeat_received = False
        self.target_system = 1
        self.target_component = 1
        self.autopilot_type = "Unknown"
        self.vehicle_type = "Unknown"
        self.flight_mode = "UNKNOWN"
        self.is_armed = False
        self.safety_switch_pressed = None

        # Captured Messages
        self.statustext_log: List[Tuple[float, int, str]] = []
        self.prearm_errors: List[str] = []
        self.arm_ack_result: Optional[int] = None
        self.arm_ack_text: Optional[str] = None

        # Sensor Health
        self.sensors_present = 0
        self.sensors_enabled = 0
        self.sensors_health = 0

        # Subsystems
        self.gps_raw: Optional[Dict[str, Any]] = None
        self.ekf_status: Optional[Dict[str, Any]] = None
        self.battery: Optional[Dict[str, Any]] = None
        self.rc_channels: Optional[Dict[str, Any]] = None
        self.parameters: Dict[str, float] = {}

    def connect(self) -> bool:
        """Connects to ArduPilot via specified URI or automatic candidate discovery."""
        if not PYMAVLINK_AVAILABLE:
            print(f"{C.RED}[ERROR] 'pymavlink' is not installed in the active Python environment.{C.RESET}")
            print(f"        Please run:  pip install pymavlink")
            return False

        if self.requested_conn:
            port, baud = normalize_connection(self.requested_conn, self.default_baud)
            return self._try_connect(port, baud, "Specified Port")

        print(f"{C.BOLD}{C.CYAN}[AUTO-DETECT] Scanning available Jetson ports and MAVLink endpoints...{C.RESET}")
        candidates = scan_candidate_ports()
        if not candidates:
            print(f"{C.RED}[ERROR] No serial devices or MAVLink endpoints found in /dev/.{C.RESET}")
            return False

        for port, baud, desc in candidates:
            # Check Linux file permissions on serial device
            if port.startswith("/dev/"):
                if not os.access(port, os.R_OK | os.W_OK):
                    print(f"  {C.YELLOW}⚠ Warning: Permission denied accessing {port}.{C.RESET}")
                    print(f"    Run: sudo usermod -aG dialout $USER  (then re-login), or: sudo chmod 666 {port}")
                    continue

            print(f"  --> Probing {C.BOLD}{port}{C.RESET} ({desc} @ {baud} baud)...", end="", flush=True)
            if self._try_connect(port, baud, desc, quick_probe=True):
                print(f" {C.GREEN}{C.BOLD}[CONNECTED]{C.RESET}")
                return True
            else:
                print(f" {C.DIM}[No Heartbeat]{C.RESET}")

        print(f"\n{C.RED}[ERROR] Could not connect to ArduPilot on any detected port.{C.RESET}")
        print("        Check physical cable wiring, TX/RX swap, Pixhawk power, or specify port:")
        print("        python scripts/diagnose_arming.py --connect /dev/ttyTHS0:921600")
        return False

    def _try_connect(self, port: str, baud: int, desc: str, quick_probe: bool = False) -> bool:
        try:
            if port.startswith("/dev/"):
                master = mavutil.mavlink_connection(
                    port,
                    baud=baud,
                    source_system=255,
                    source_component=190,
                    autoreconnect=False
                )
            else:
                master = mavutil.mavlink_connection(
                    port,
                    source_system=255,
                    source_component=190,
                    autoreconnect=False
                )
        except Exception as e:
            if not quick_probe:
                print(f"{C.RED}[ERROR] Failed to open {port}: {e}{C.RESET}")
            return False

        probe_time = 2.0 if quick_probe else self.timeout_s
        msg = master.wait_heartbeat(timeout=probe_time)
        if msg is not None:
            self.master = master
            self.target_system = master.target_system
            self.target_component = master.target_component
            self.heartbeat_received = True
            self._update_heartbeat(msg)
            return True

        master.close()
        return False

    def _update_heartbeat(self, msg: Any):
        self.is_armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        custom_mode = msg.custom_mode
        self.flight_mode = ARDUCOPTER_MODES.get(custom_mode, f"MODE_{custom_mode}")
        if msg.autopilot == mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA:
            self.autopilot_type = "ArduPilot (ArduCopter)"
        else:
            self.autopilot_type = f"Autopilot ID {msg.autopilot}"

        if msg.type == mavutil.mavlink.MAV_TYPE_QUADROTOR:
            self.vehicle_type = "Quadcopter"
        elif msg.type == mavutil.mavlink.MAV_TYPE_HEXAROTOR:
            self.vehicle_type = "Hexacopter"
        else:
            self.vehicle_type = f"Vehicle Type {msg.type}"

    def trigger_prearm_checks(self):
        """Dispatches MAV_CMD_RUN_PREARM_CHECKS (401) to force ArduPilot to broadcast prearm status."""
        if not self.master:
            return
        try:
            self.master.mav.command_long_send(
                self.target_system,
                self.target_component,
                401,  # MAV_CMD_RUN_PREARM_CHECKS
                0,
                0, 0, 0, 0, 0, 0, 0
            )
        except Exception:
            pass

    def attempt_arm(self):
        """Sends MAV_CMD_COMPONENT_ARM_DISARM with normal checks to capture exact rejection ACK & error."""
        if not self.master:
            return
        print(f"\n{C.BOLD}{C.YELLOW}[ACTION] Sending MAVLink Arming Request (MAV_CMD_COMPONENT_ARM_DISARM)...{C.RESET}")
        try:
            self.master.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1.0,  # 1.0 = Arm
                0.0,  # 0.0 = Standard pre-arm enforcement
                0, 0, 0, 0, 0
            )
        except Exception as e:
            print(f"{C.RED}[ERROR] Failed to send arm command: {e}{C.RESET}")

    def request_key_parameters(self):
        """Queries critical arming and failsafe parameters."""
        if not self.master:
            return
        params = ["ARMING_CHECK", "BRD_SAFETYENABLE", "BRD_SAFETY_MASK", "FS_THR_ENABLE", "FS_BATT_ENABLE", "FS_BATT_VOLT"]
        for p in params:
            try:
                self.master.mav.param_request_read_send(
                    self.target_system,
                    self.target_component,
                    p.encode("ascii"),
                    -1
                )
            except Exception:
                pass

    def collect_telemetry(self, duration_s: float = 3.5):
        """Gathers messages from ArduPilot to populate diagnostics."""
        if not self.master:
            return

        t_end = time.time() + duration_s
        while time.time() < t_end:
            msg = self.master.recv_msg()
            if msg is None:
                time.sleep(0.01)
                continue

            mtype = msg.get_type()
            now = time.time()

            if mtype == "HEARTBEAT":
                self._update_heartbeat(msg)

            elif mtype == "STATUSTEXT":
                text = msg.text
                if isinstance(text, bytes):
                    text = text.decode("ascii", errors="ignore").rstrip("\x00")
                sev = getattr(msg, "severity", 6)
                self.statustext_log.append((now, sev, text))
                if "prearm" in text.lower() or "arm" in text.lower() or "failsafe" in text.lower():
                    if text not in self.prearm_errors:
                        self.prearm_errors.append(text)

            elif mtype == "COMMAND_ACK":
                cmd = getattr(msg, "command", None)
                if cmd == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                    self.arm_ack_result = msg.result
                    ack_map = {
                        0: "ACCEPTED (Vehicle Armed)",
                        1: "TEMPORARILY_REJECTED (Pre-arm checks failed or gyro settling)",
                        2: "DENIED (Arming disallowed in current state/mode)",
                        3: "UNSUPPORTED",
                        4: "FAILED",
                    }
                    self.arm_ack_text = ack_map.get(msg.result, f"Result Code {msg.result}")

            elif mtype == "SYS_STATUS":
                self.sensors_present = msg.onboard_control_sensors_present
                self.sensors_enabled = msg.onboard_control_sensors_enabled
                self.sensors_health = msg.onboard_control_sensors_health
                volt = msg.voltage_battery / 1000.0 if msg.voltage_battery != 65535 else None
                curr = msg.current_battery / 100.0 if msg.current_battery != -1 else None
                pct = msg.battery_remaining if msg.battery_remaining != -1 else None
                self.battery = {"voltage_v": volt, "current_a": curr, "remaining_pct": pct}

            elif mtype == "GPS_RAW_INT":
                self.gps_raw = {
                    "fix_type": msg.fix_type,
                    "fix_name": GPS_FIX_TYPES.get(msg.fix_type, f"Fix {msg.fix_type}"),
                    "satellites": msg.satellites_visible,
                    "hdop": msg.eph / 100.0 if msg.eph != 65535 else 99.9,
                    "vdop": msg.epv / 100.0 if msg.epv != 65535 else 99.9,
                    "alt_m": msg.alt / 1000.0,
                }

            elif mtype == "EKF_STATUS_REPORT":
                self.ekf_status = {
                    "flags": msg.flags,
                    "velocity_variance": msg.velocity_variance,
                    "pos_horiz_variance": msg.pos_horiz_variance,
                    "pos_vert_variance": msg.pos_vert_variance,
                    "compass_variance": msg.compass_variance,
                    "terrain_alt_variance": msg.terrain_alt_variance,
                }

            elif mtype == "RC_CHANNELS":
                self.rc_channels = {
                    "roll": getattr(msg, "chan1_raw", 0),
                    "pitch": getattr(msg, "chan2_raw", 0),
                    "throttle": getattr(msg, "chan3_raw", 0),
                    "yaw": getattr(msg, "chan4_raw", 0),
                    "mode_sw": getattr(msg, "chan5_raw", 0),
                    "rssi": getattr(msg, "rssi", 0),
                }

            elif mtype == "PARAM_VALUE":
                param_id = msg.param_id
                if isinstance(param_id, bytes):
                    param_id = param_id.decode("ascii", errors="ignore").rstrip("\x00")
                self.parameters[param_id] = msg.param_value

    def print_diagnostic_report(self) -> List[str]:
        """Prints the full structured diagnostic report and returns list of actionable recommendations."""
        recommendations = []

        print(f"\n{C.BOLD}{'=' * 95}{C.RESET}")
        print(f"{C.BOLD}{C.CYAN}  ARDUPILOT COMPANION COMPUTER ARMING DIAGNOSTIC REPORT{C.RESET}")
        print(f"{C.BOLD}{'=' * 95}{C.RESET}")

        # Section 1: Vehicle & Flight Mode Overview
        print(f"\n{C.BOLD}1. VEHICLE & FLIGHT MODE STATUS{C.RESET}")
        print(f"  Autopilot Type:      {C.BOLD}{self.autopilot_type}{C.RESET}")
        print(f"  Vehicle Frame:       {self.vehicle_type}")
        print(f"  Target System/Comp:  ID {self.target_system} / Comp {self.target_component}")

        arm_str = f"{C.GREEN}{C.BOLD}[ARMED]{C.RESET}" if self.is_armed else f"{C.YELLOW}{C.BOLD}[DISARMED]{C.RESET}"
        print(f"  Armed State:         {arm_str}")

        mode_color = C.GREEN if self.flight_mode in ("STABILIZE", "ALT_HOLD", "LOITER", "GUIDED") else C.YELLOW
        print(f"  Active Flight Mode:  {mode_color}{C.BOLD}{self.flight_mode}{C.RESET}")

        if self.flight_mode in GPS_REQUIRED_MODES:
            print(f"  Mode Requirement:    {C.YELLOW}⚠ Mode '{self.flight_mode}' strictly requires 3D GPS fix and healthy EKF.{C.RESET}")
            if not self.gps_raw or self.gps_raw["fix_type"] < 3:
                recommendations.append(
                    f"Flight mode is set to {C.BOLD}{self.flight_mode}{C.RESET}, which cannot arm without 3D GPS fix.\n"
                    f"     --> If testing on the bench / indoors, switch transmitter to {C.BOLD}STABILIZE{C.RESET} or {C.BOLD}ALT_HOLD{C.RESET}."
                )
        else:
            print(f"  Mode Requirement:    {C.GREEN}✓ Mode '{self.flight_mode}' allows bench arming without GPS.{C.RESET}")

        # Section 2: Arming Attempt Result (if requested)
        if self.arm_ack_result is not None:
            print(f"\n{C.BOLD}2. ARMING COMMAND RESPONSE (MAVLink ACK){C.RESET}")
            if self.arm_ack_result == 0:
                print(f"  Result:              {C.GREEN}{C.BOLD}✓ {self.arm_ack_text}{C.RESET}")
            else:
                print(f"  Result:              {C.RED}{C.BOLD}✗ {self.arm_ack_text}{C.RESET}")
                recommendations.append(f"Autopilot rejected arm command: {self.arm_ack_text}")

        # Section 3: Captured Pre-Arm & Failsafe Status Messages
        print(f"\n{C.BOLD}3. REAL-TIME AUTOPILOT NOTICES & PRE-ARM MESSAGES{C.RESET}")
        if self.prearm_errors:
            for pe in self.prearm_errors:
                print(f"  {C.RED}{C.BOLD}✗ {pe}{C.RESET}")
                recommendations.append(f"Autopilot PreArm check failed: {pe}")
        else:
            print(f"  {C.GREEN}✓ No explicit 'PreArm' rejection strings broadcast in the last window.{C.RESET}")

        # Print all recent STATUSTEXT log entries
        if self.statustext_log:
            print(f"\n  Recent Autopilot Console Log (Last {len(self.statustext_log)} messages):")
            for ts, sev, text in self.statustext_log[-8:]:
                sev_name, s_color = SEVERITY_NAMES.get(sev, ("LOG", C.WHITE))
                print(f"    {C.DIM}{time.strftime('%H:%M:%S', time.localtime(ts))}{C.RESET} [{s_color}{sev_name:8s}{C.RESET}] {text}")

        # Section 4: Sensor Health Bitmask (SYS_STATUS)
        print(f"\n{C.BOLD}4. SENSOR & SUBSYSTEM HEALTH BITMASK (SYS_STATUS){C.RESET}")
        print(f"  {'Subsystem Name':35s} | {'Present':7s} | {'Enabled':7s} | {'Health Status':15s}")
        print(f"  {'-' * 35}-|-{'-' * 7}-|-{'-' * 7}-|-{'-' * 15}")

        sensors_failing = []
        for mask, name, cat in SENSOR_DEFINITIONS:
            is_present = bool(self.sensors_present & mask)
            is_enabled = bool(self.sensors_enabled & mask)
            is_healthy = bool(self.sensors_health & mask)

            if not is_present:
                continue

            pres_str = f"{C.GREEN}Yes{C.RESET}" if is_present else f"{C.DIM}No{C.RESET}"
            enab_str = f"{C.GREEN}Yes{C.RESET}" if is_enabled else f"{C.DIM}No{C.RESET}"

            if is_enabled and not is_healthy:
                health_str = f"{C.RED}{C.BOLD}[UNHEALTHY]{C.RESET}"
                sensors_failing.append(name)
            elif is_healthy:
                health_str = f"{C.GREEN}[OK]{C.RESET}"
            else:
                health_str = f"{C.DIM}[Disabled]{C.RESET}"

            print(f"  {name:35s} | {pres_str:16s} | {enab_str:16s} | {health_str}")

        if sensors_failing:
            for sf in sensors_failing:
                recommendations.append(f"Sensor marked UNHEALTHY in SYS_STATUS: {sf}")

        # Section 5: GPS Receiver Diagnostics
        print(f"\n{C.BOLD}5. GPS & NAVIGATION LOCK STATUS{C.RESET}")
        if self.gps_raw:
            fix_type = self.gps_raw["fix_type"]
            sats = self.gps_raw["satellites"]
            hdop = self.gps_raw["hdop"]

            fix_color = C.GREEN if fix_type >= 3 else (C.YELLOW if fix_type == 2 else C.RED)
            hdop_color = C.GREEN if hdop < 2.0 else (C.YELLOW if hdop < 2.5 else C.RED)
            sat_color = C.GREEN if sats >= 10 else (C.YELLOW if sats >= 6 else C.RED)

            print(f"  Fix Type:            {fix_color}{C.BOLD}{self.gps_raw['fix_name']}{C.RESET}")
            print(f"  Satellites Visible:  {sat_color}{C.BOLD}{sats} sats{C.RESET} (minimum for 3D navigation is typically 8–10)")
            print(f"  Horizontal Dilution: {hdop_color}{C.BOLD}HDOP = {hdop:.2f}{C.RESET} (ArduPilot pre-arm requires HDOP < 2.0)")
            print(f"  Vertical Dilution:   VDOP = {self.gps_raw['vdop']:.2f}")

            if fix_type < 3 and self.flight_mode in GPS_REQUIRED_MODES:
                recommendations.append(
                    f"Insufficient GPS fix ({self.gps_raw['fix_name']}, HDOP {hdop:.1f}) for autonomous mode '{self.flight_mode}'.\n"
                    f"     --> Move drone to open sky or wait for satellite lock."
                )
            elif hdop >= 2.0 and fix_type >= 3:
                recommendations.append(f"GPS HDOP is {hdop:.2f} (above 2.0 threshold). Wait for higher satellite count before arming.")
        else:
            print(f"  {C.YELLOW}⚠ No GPS_RAW_INT telemetry received.{C.RESET}")

        # Section 6: Extended Kalman Filter (EKF) Health
        print(f"\n{C.BOLD}6. EKF ESTIMATION & INNOVATION VARIANCES{C.RESET}")
        if self.ekf_status:
            vel_v = self.ekf_status["velocity_variance"]
            pos_h = self.ekf_status["pos_horiz_variance"]
            pos_v = self.ekf_status["pos_vert_variance"]
            cmp_v = self.ekf_status["compass_variance"]
            flags = self.ekf_status["flags"]

            def var_str(v):
                if v < 0.5:
                    return f"{C.GREEN}{v:.2f}{C.RESET}"
                elif v < 1.0:
                    return f"{C.YELLOW}{C.BOLD}{v:.2f} (Elevated){C.RESET}"
                else:
                    return f"{C.RED}{C.BOLD}{v:.2f} [FAIL > 1.0]{C.RESET}"

            print(f"  Velocity Variance:   {var_str(vel_v)} (needs < 1.0)")
            print(f"  Horiz Pos Variance:  {var_str(pos_h)} (needs < 1.0)")
            print(f"  Vert Pos Variance:   {var_str(pos_v)} (needs < 1.0)")
            print(f"  Compass Variance:    {var_str(cmp_v)} (needs < 1.0)")

            if cmp_v >= 1.0:
                recommendations.append(
                    f"EKF Compass Variance is high ({cmp_v:.2f} >= 1.0).\n"
                    f"     --> Calibrate compass in Mission Planner / QGC, or move drone away from metal structures."
                )
            if vel_v >= 1.0 or pos_h >= 1.0:
                recommendations.append(
                    f"EKF Position/Velocity variance is high (Vel: {vel_v:.2f}, Pos: {pos_h:.2f}).\n"
                    f"     --> Keep drone stationary while gyros/EKF align."
                )

            # Check individual health flags
            unhealthy_ekf_flags = []
            for mask, fname in EKF_FLAG_NAMES:
                if not (flags & mask):
                    unhealthy_ekf_flags.append(fname)
            if unhealthy_ekf_flags:
                print(f"  {C.RED}⚠ Inactive EKF Flags: {', '.join(unhealthy_ekf_flags)}{C.RESET}")
        else:
            print(f"  {C.YELLOW}⚠ No EKF_STATUS_REPORT telemetry received.{C.RESET}")

        # Section 7: Battery & Power Management
        print(f"\n{C.BOLD}7. POWER MODULE & BATTERY STATUS{C.RESET}")
        if self.battery and self.battery["voltage_v"] is not None:
            v = self.battery["voltage_v"]
            curr = self.battery["current_a"] or 0.0
            pct = self.battery["remaining_pct"]

            # Estimate cell count
            cells = max(1, round(v / 3.85))
            v_per_cell = v / cells
            v_color = C.GREEN if v_per_cell >= 3.65 else (C.YELLOW if v_per_cell >= 3.5 else C.RED)

            print(f"  Total Voltage:       {v_color}{C.BOLD}{v:.2f} V{C.RESET} (~{cells}S pack, {v_per_cell:.2f} V/cell)")
            print(f"  Current Draw:        {curr:.2f} A")
            if pct is not None:
                pct_color = C.GREEN if pct > 25 else C.RED
                print(f"  Battery Remaining:   {pct_color}{pct} %{C.RESET}")

            if v_per_cell < 3.5:
                recommendations.append(f"Battery voltage is critically low ({v:.2f}V, {v_per_cell:.2f}V/cell). Recharge battery.")
        else:
            print(f"  {C.YELLOW}⚠ No battery voltage telemetry received.{C.RESET}")

        # Section 8: RC Transmitter Link & Throttle Position
        print(f"\n{C.BOLD}8. RC TRANSMITTER & THROTTLE CHANNEL{C.RESET}")
        if self.rc_channels:
            thr = self.rc_channels["throttle"]
            rssi = self.rc_channels["rssi"]

            thr_color = C.GREEN if thr < 1100 else C.RED
            print(f"  Channel 3 (Throttle):{thr_color}{C.BOLD}{thr} µs{C.RESET} (Must be at zero / stick down < 1050–1100 µs to arm)")
            print(f"  RC Signal (RSSI):    {rssi}")

            if thr >= 1100:
                recommendations.append(
                    f"RC Throttle stick is elevated ({thr} µs >= 1100 µs).\n"
                    f"     --> Pull transmitter throttle stick all the way down to minimum before arming."
                )
            if thr == 0:
                recommendations.append(
                    f"RC Channel 3 is 0 µs (No RC signal received).\n"
                    f"     --> Power on RC transmitter and ensure receiver is bound, or check FS_THR_ENABLE."
                )
        else:
            print(f"  {C.YELLOW}⚠ No RC_CHANNELS telemetry received. Transmitter may be turned off.{C.RESET}")

        # Section 9: Critical Safety Parameters
        print(f"\n{C.BOLD}9. CRITICAL SAFETY & HARDWARE SWITCH PARAMETERS{C.RESET}")
        safety_en = self.parameters.get("BRD_SAFETYENABLE")
        if safety_en is not None:
            if int(safety_en) == 1:
                print(f"  Safety Switch:       {C.YELLOW}{C.BOLD}BRD_SAFETYENABLE = 1 (Hardware Safety Switch Required){C.RESET}")
                recommendations.append(
                    f"Hardware safety switch is enabled (BRD_SAFETYENABLE = 1).\n"
                    f"     --> Press and HOLD physical safety button on GPS puck/drone for 2 seconds until red LED is SOLID."
                )
            else:
                print(f"  Safety Switch:       {C.GREEN}BRD_SAFETYENABLE = 0 (Bypassed / Not Enforced){C.RESET}")

        arm_check = self.parameters.get("ARMING_CHECK")
        if arm_check is not None:
            print(f"  Arming Check Mask:   ARMING_CHECK = {int(arm_check)} (1 = All Checks Enforced)")

        # Section 10: Actionable Recommendations Summary
        print(f"\n{C.BOLD}{'=' * 95}{C.RESET}")
        print(f"{C.BOLD}{C.MAGENTA}  ACTIONABLE FIXES TO RESOLVE ARMING ISSUE:{C.RESET}")
        print(f"{C.BOLD}{'=' * 95}{C.RESET}")

        if not recommendations:
            if self.is_armed:
                print(f"  {C.GREEN}{C.BOLD}✓ VEHICLE IS ALREADY ARMED! Motors are live.{C.RESET}")
            else:
                print(f"  {C.GREEN}{C.BOLD}✓ No hardware or sensor pre-arm faults detected.{C.RESET}")
                print(f"    Vehicle should be ready to arm via transmitter or companion command.")
        else:
            # Deduplicate recommendations preserving order
            seen = set()
            idx = 1
            for rec in recommendations:
                clean_rec = rec.strip()
                if clean_rec not in seen:
                    seen.add(clean_rec)
                    print(f"  {C.BOLD}[{idx:02d}]{C.RESET} {rec}")
                    idx += 1

        print(f"{C.BOLD}{'=' * 95}{C.RESET}\n")
        return recommendations


def main():
    C.strip_if_no_color()

    parser = argparse.ArgumentParser(
        description="ArduPilot Jetson Arming Diagnostic Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/diagnose_arming.py
  python scripts/diagnose_arming.py --connect /dev/ttyTHS0:921600
  python scripts/diagnose_arming.py --arm
  python scripts/diagnose_arming.py --monitor
        """
    )
    parser.add_argument("-c", "--connect", type=str, default=None,
                        help="MAVLink connection URI (e.g. /dev/ttyTHS0:921600, /dev/ttyACM0, udp:127.0.0.1:14550)")
    parser.add_argument("-b", "--baud", type=int, default=921600,
                        help="Default baud rate for UART (default: 921600)")
    parser.add_argument("-a", "--arm", action="store_true", default=False,
                        help="Attempt a live arm command to capture exact autopilot denial ACK")
    parser.add_argument("-m", "--monitor", action="store_true", default=False,
                        help="Run continuous live monitoring loop to watch pre-arm checks clear")
    parser.add_argument("-t", "--timeout", type=float, default=5.0,
                        help="Timeout waiting for vehicle heartbeat in seconds (default: 5.0)")

    args = parser.parse_args()

    diag = ArmingDiagnostic(
        connection_string=args.connect,
        baud=args.baud,
        timeout_s=args.timeout
    )

    if not diag.connect():
        sys.exit(1)

    # Initial query
    diag.trigger_prearm_checks()
    diag.request_key_parameters()

    if args.arm:
        # Give 0.5s for initial messages then send arm
        diag.collect_telemetry(0.5)
        diag.attempt_arm()
        diag.collect_telemetry(3.0)
    else:
        # Standard diagnostic collection
        print(f"{C.BOLD}[AUDIT] Ingesting flight controller telemetry and pre-arm checks...{C.RESET}")
        diag.collect_telemetry(3.0)

    # Print main report
    diag.print_diagnostic_report()

    # Optional continuous monitoring loop
    if args.monitor:
        print(f"{C.CYAN}{C.BOLD}[MONITOR MODE] Streaming live PreArm messages and arming state (Ctrl+C to exit)...{C.RESET}\n")
        try:
            prev_mode = diag.flight_mode
            prev_armed = diag.is_armed
            while True:
                msg = diag.master.recv_match(blocking=True, timeout=1.0)
                if msg is not None:
                    mtype = msg.get_type()
                    if mtype == "STATUSTEXT":
                        text = msg.text
                        if isinstance(text, bytes):
                            text = text.decode("ascii", errors="ignore").rstrip("\x00")
                        sev = getattr(msg, "severity", 6)
                        sev_name, s_color = SEVERITY_NAMES.get(sev, ("LOG", C.WHITE))
                        t_str = time.strftime("%H:%M:%S")
                        print(f"[{t_str}] [{s_color}{sev_name:8s}{C.RESET}] {text}")

                    elif mtype == "HEARTBEAT":
                        diag._update_heartbeat(msg)
                        if diag.is_armed != prev_armed:
                            status_str = f"{C.GREEN}{C.BOLD}ARMED{C.RESET}" if diag.is_armed else f"{C.YELLOW}DISARMED{C.RESET}"
                            print(f"\n>>> [STATE CHANGE] Vehicle Armed Status: {status_str}")
                            prev_armed = diag.is_armed
                        if diag.flight_mode != prev_mode:
                            print(f">>> [MODE CHANGE] Flight Mode is now: {C.BOLD}{diag.flight_mode}{C.RESET}")
                            prev_mode = diag.flight_mode
        except KeyboardInterrupt:
            print(f"\n{C.CYAN}[MONITOR] Exiting.{C.RESET}")

    sys.exit(0)


if __name__ == "__main__":
    main()

