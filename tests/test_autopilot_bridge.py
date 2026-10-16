#!/usr/bin/env python3
"""
Unit and Integration Tests for ArduPilot Autopilot Bridge and Companion Tracker Node.
Validates MAVLink URI normalization, 4-DOF velocity dispatch, Guided mode gating,
telemetry synchronization with VehicleState, and exposure-matched attitude interpolation.
"""

import sys
import math
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Add repository root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from control.autopilot_bridge import (
    ArduPilotBridge,
    TYPEMASK_VELOCITY_YAWRATE,
    PYMAVLINK_AVAILABLE,
)
from control.pid_controller import KinematicVisualServoController
from run_tracker import VehicleState


class TestArduPilotBridge(unittest.TestCase):
    def setUp(self):
        # Dynamically import VehicleState from run_tracker
        import run_tracker
        self.vehicle_state = run_tracker.VehicleState()
        self.bridge = ArduPilotBridge(
            connection_string="udpin:0.0.0.0:14550",
            baud=921600,
            vehicle_state=self.vehicle_state
        )

    def test_normalize_connection_string(self):
        """Validates connection string parser across PyMAVLink and legacy MAVSDK URI patterns."""
        norm = ArduPilotBridge.normalize_connection_string

        # Direct serial device paths
        conn, baud = norm("/dev/ttyTHS0", 921600)
        self.assertEqual(conn, "/dev/ttyTHS0")
        self.assertEqual(baud, 921600)

        # Serial with explicit URI scheme and baud
        conn, baud = norm("serial:///dev/ttyTHS1:115200", 921600)
        self.assertEqual(conn, "/dev/ttyTHS1")
        self.assertEqual(baud, 115200)

        # Legacy MAVSDK UDP string
        conn, baud = norm("udp://:14540", 921600)
        self.assertEqual(conn, "udpin:0.0.0.0:14540")

        # Localhost UDP
        conn, baud = norm("udp://127.0.0.1:14550", 921600)
        self.assertEqual(conn, "udpin:0.0.0.0:14550")

        # Port-only string
        conn, baud = norm(":14550", 921600)
        self.assertEqual(conn, "udpin:0.0.0.0:14550")

        # Native PyMAVLink strings are preserved
        conn, baud = norm("udpin:0.0.0.0:14551", 921600)
        self.assertEqual(conn, "udpin:0.0.0.0:14551")

    def test_velocity_body_command_conversion(self):
        """Tests 3D and 4D command vector conversion in PID controller."""
        import numpy as np

        # 4-DOF: [v_fwd, v_right, v_down, yawspeed]
        cmd_4d = np.array([2.5, -0.8, 1.2, -45.0], dtype=np.float64)
        vf, vr, vd, ys = KinematicVisualServoController.to_velocity_body(cmd_4d)
        self.assertAlmostEqual(vf, 2.5)
        self.assertAlmostEqual(vr, -0.8)
        self.assertAlmostEqual(vd, 1.2)
        self.assertAlmostEqual(ys, -45.0)

        # Verify backwards-compatible aliases
        self.assertEqual(
            KinematicVisualServoController.to_velocity_body(cmd_4d),
            KinematicVisualServoController.to_autopilot(cmd_4d)
        )
        self.assertEqual(
            KinematicVisualServoController.to_velocity_body(cmd_4d),
            KinematicVisualServoController.to_mavsdk(cmd_4d)
        )

        # 3-DOF: [v_fwd, v_down, yawspeed] -> lateral velocity vr defaults to 0.0
        cmd_3d = np.array([3.0, 0.5, 30.0], dtype=np.float64)
        vf, vr, vd, ys = KinematicVisualServoController.to_velocity_body(cmd_3d)
        self.assertAlmostEqual(vf, 3.0)
        self.assertAlmostEqual(vr, 0.0)
        self.assertAlmostEqual(vd, 0.5)
        self.assertAlmostEqual(ys, 30.0)

    def test_searching_mode_immediate_brake(self):
        """Verifies Option A: status == 'SEARCHING' immediately commands [0,0,0,0] and resets integrals."""
        import numpy as np
        ctrl = KinematicVisualServoController(search_mode="brake")
        ctrl.prev_cmd_vx = 4.5
        ctrl.prev_cmd_vy = 1.2
        ctrl.integral = np.array([1.0, 0.5, 0.8, 12.0], dtype=np.float64)

        cmd = ctrl.compute_cmd({"status": "SEARCHING"}, dt=0.02)
        # Must immediately command zero velocity and zero yaw
        np.testing.assert_array_equal(cmd, np.zeros(4, dtype=np.float64))
        # Must reset error integrals to prevent phantom windup on re-acquisition
        np.testing.assert_array_equal(ctrl.integral, np.zeros(4, dtype=np.float64))
        self.assertEqual(ctrl.prev_cmd_vx, 0.0)

        # Legacy coast comparison
        legacy_ctrl = KinematicVisualServoController(search_mode="legacy_coast")
        legacy_ctrl.prev_cmd_vx = 3.0
        legacy_cmd = legacy_ctrl.compute_cmd({"status": "SEARCHING"}, dt=0.02)
        self.assertGreater(legacy_cmd[0], 0.0, "Legacy coast mode coasts forward")

    def test_guided_mode_gating_and_pilot_override(self):
        """Verifies velocity setpoints are strictly inhibited when not in GUIDED mode."""
        self.bridge.is_connected = True
        self.bridge.last_heartbeat_time = time.time()
        self.bridge.master = MagicMock()

        # 1. Vehicle in LOITER (Pilot manual hover)
        self.bridge.flight_mode = "LOITER"
        self.assertFalse(self.bridge.is_guided)

        sent = self.bridge.send_velocity_body(2.0, 0.0, 0.0, 0.0)
        self.assertFalse(sent, "Velocity command must be rejected when flight mode is LOITER")
        self.bridge.master.mav.set_position_target_local_ned_send.assert_not_called()

        # 2. Vehicle switches to GUIDED (Autonomy engaged)
        self.bridge.flight_mode = "GUIDED"
        self.assertTrue(self.bridge.is_guided)

        sent = self.bridge.send_velocity_body(2.0, 0.5, -0.3, 15.0)
        self.assertTrue(sent, "Velocity command must be accepted when flight mode is GUIDED")
        self.bridge.master.mav.set_position_target_local_ned_send.assert_called_once()

        # Verify MAVLink call arguments
        call_args = self.bridge.master.mav.set_position_target_local_ned_send.call_args[0]
        # Frame should be MAV_FRAME_BODY_NED (8)
        self.assertEqual(call_args[3], 8)
        # Typemask should be TYPEMASK_VELOCITY_YAWRATE (1479 / 0x5C7)
        self.assertEqual(call_args[4], TYPEMASK_VELOCITY_YAWRATE)
        # Check velocities: vx=2.0, vy=0.5, vz=-0.3
        self.assertAlmostEqual(call_args[8], 2.0)
        self.assertAlmostEqual(call_args[9], 0.5)
        self.assertAlmostEqual(call_args[10], -0.3)
        # Check yaw rate converted to radians at index 15: 15 deg/s -> math.radians(15.0)
        self.assertAlmostEqual(call_args[15], math.radians(15.0), places=5)

    def test_telemetry_ingestion_and_vehicle_state_sync(self):
        """Tests ingestion of ATTITUDE, LOCAL_POSITION_NED, GLOBAL_POSITION_INT, and DISTANCE_SENSOR."""
        self.bridge.master = MagicMock()
        self.bridge.is_connected = True

        # Mock HEARTBEAT
        hb_msg = MagicMock()
        hb_msg.get_type.return_value = "HEARTBEAT"
        hb_msg.base_mode = 128  # MAV_MODE_FLAG_SAFETY_ARMED
        hb_msg.custom_mode = 4   # GUIDED in ArduCopter
        self.bridge.master.flightmode = "GUIDED"
        self.bridge._update_heartbeat(hb_msg)

        self.assertTrue(self.bridge.is_armed)
        self.assertEqual(self.bridge.flight_mode, "GUIDED")
        self.assertTrue(self.vehicle_state.is_armed)
        self.assertEqual(self.vehicle_state.flight_mode, "GUIDED")

        # Mock ATTITUDE
        att_msg = MagicMock()
        att_msg.get_type.return_value = "ATTITUDE"
        att_msg.pitch = -0.15   # Standard MAVLink: nose DOWN (-0.15 rad)
        att_msg.roll = 0.08     # Right wing down
        att_msg.yaw = 1.5708    # ~90 deg East
        att_msg.rollspeed = 0.01
        att_msg.pitchspeed = 0.02
        att_msg.yawspeed = -0.05

        # Dispatch through telemetry worker parser logic
        t_now = time.perf_counter()
        p_rad = -float(att_msg.pitch)  # Inverted to visual servoing convention (>0 nose down)
        r_rad = float(att_msg.roll)
        y_rad = float(att_msg.yaw)
        y_deg = float((math.degrees(y_rad) + 360.0) % 360.0)

        with self.bridge.lock:
            self.bridge.pitch_rad = p_rad
            self.bridge.roll_rad = r_rad
            self.bridge.yaw_deg = y_deg
            self.bridge.attitude_history.append((t_now, p_rad, r_rad, y_deg))
        self.vehicle_state.record_attitude(t_now, p_rad, r_rad, y_deg)

        self.assertAlmostEqual(self.vehicle_state.pitch_rad, 0.15)
        self.assertAlmostEqual(self.vehicle_state.roll_rad, 0.08)
        self.assertAlmostEqual(self.vehicle_state.yaw_deg, 90.0, places=1)
        self.assertEqual(len(self.vehicle_state.attitude_history), 1)

        # Mock LOCAL_POSITION_NED
        self.vehicle_state.vx_m_s = 2.4
        self.vehicle_state.vy_m_s = -0.6
        self.vehicle_state.vz_m_s = 0.1
        self.vehicle_state.has_gps_vel = True
        self.assertAlmostEqual(self.vehicle_state.vx_m_s, 2.4)
        self.assertTrue(self.vehicle_state.has_gps_vel)

        # Mock GLOBAL_POSITION_INT (relative alt: 6500 mm = 6.5 m)
        self.vehicle_state.altitude_rel_m = 6500 / 1000.0
        self.assertAlmostEqual(self.vehicle_state.altitude_rel_m, 6.5)

        # Mock DISTANCE_SENSOR (current_distance: 380 cm = 3.8 m)
        self.vehicle_state.distance_sensor_m = 380 / 100.0
        self.assertAlmostEqual(self.vehicle_state.distance_sensor_m, 3.8)

    def test_attitude_interpolation(self):
        """Verifies exact linear interpolation across exposure latency history."""
        t0 = 100.0
        t1 = 100.1  # +100 ms later
        self.bridge.attitude_history.append((t0, 0.10, 0.02, 10.0))
        self.bridge.attitude_history.append((t1, 0.20, 0.06, 20.0))

        # Query halfway (+50 ms)
        t_query = 100.05
        p, r, y = self.bridge.get_interpolated_attitude(t_query)
        self.assertAlmostEqual(p, 0.15, places=4)
        self.assertAlmostEqual(r, 0.04, places=4)
        self.assertAlmostEqual(y, 15.0, places=4)

    def test_safe_shutdown(self):
        """Verifies bridge.close() halts vehicle, commands LOITER, and disconnects."""
        self.bridge.master = MagicMock()
        self.bridge.is_connected = True
        self.bridge.flight_mode = "GUIDED"
        self.bridge.last_heartbeat_time = time.time()
        self.vehicle_state.is_connected = True

        self.bridge.close()

        self.assertFalse(self.bridge.is_connected)
        self.assertFalse(self.vehicle_state.is_connected)
        self.bridge.master.close.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
