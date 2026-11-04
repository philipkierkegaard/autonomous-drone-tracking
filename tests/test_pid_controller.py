"""
Unit tests for KinematicVisualServoController and vision-synced derivative tracking.
"""

import unittest
import numpy as np
from control.pid_controller import KinematicVisualServoController


class TestKinematicVisualServoController(unittest.TestCase):
    def setUp(self):
        self.controller = KinematicVisualServoController(
            desired_bbox_size=48.0,
            desired_standoff_dist=4.0,
            enable_lateral_strafe=True,
            kp_forward=1.2,
            kp_lat=2.2,
            ki_lat=0.35,
            kd_lat=0.27,
            kp_z=3.0,
            kp_yaw=200.0,
        )

    def test_no_lateral_strafe_enforcement(self):
        """Verifies that setting enable_lateral_strafe=False strictly zeroes all lateral actions."""
        ctrl_no_lat = KinematicVisualServoController(
            desired_bbox_size=48.0,
            enable_lateral_strafe=False,
            kp_forward=1.2,
            kp_lat=2.204,
            ki_lat=0.354,
            kd_lat=0.273,
            kp_z=3.0,
            kp_yaw=200.0,
        )
        # 1. Gains must be strictly zeroed
        self.assertFalse(ctrl_no_lat.enable_lateral_strafe)
        self.assertEqual(ctrl_no_lat.kp[1], 0.0)
        self.assertEqual(ctrl_no_lat.ki[1], 0.0)
        self.assertEqual(ctrl_no_lat.kd[1], 0.0)

        # 2. Large horizontal error must produce exactly 0.0 lateral velocity
        telemetry = {
            "status": "LOCKED",
            "in_view": True,
            "error_x": 0.85,  # Strong horizontal error
            "error_y": 0.0,
            "bbox_w": 48.0,
            "target_size": 48.0,
            "frame_idx": 1,
            "timestamp": 1.0,
        }
        cmd = ctrl_no_lat.compute_cmd(telemetry, dt=0.02)
        self.assertEqual(cmd[1], 0.0)

        # 3. Conversion to body velocity must yield v_right == 0.0
        v_fwd, v_right, v_down, yawspeed = ctrl_no_lat.to_velocity_body(cmd)
        self.assertEqual(v_right, 0.0)

    def test_vision_synced_derivative_holding(self):
        """Verifies that intermediate 50 Hz control ticks on cached 30 Hz vision frames hold derivative without decaying."""
        ctrl = KinematicVisualServoController(
            desired_bbox_size=48.0,
            kd_yaw=30.0,
            filter_tau=0.06,
        )

        # Tick 1: Frame 0 arrives at t=1.000s
        telem_0 = {
            "status": "LOCKED",
            "in_view": True,
            "error_x": 0.20,
            "error_y": 0.0,
            "bbox_w": 48.0,
            "target_size": 48.0,
            "frame_idx": 100,
            "timestamp": 1.000,
        }
        cmd1 = ctrl.compute_cmd(telem_0, dt=0.02)

        # Tick 2: Frame 1 arrives at t=1.033s (delta_x = 0.10 over 33.3 ms)
        telem_1 = dict(telem_0)
        telem_1.update({
            "error_x": 0.30,
            "frame_idx": 101,
            "timestamp": 1.0333,
        })
        cmd2 = ctrl.compute_cmd(telem_1, dt=0.02)
        filtered_d_after_new_frame = ctrl.filtered_deriv.copy()
        # The azimuth derivative should be non-zero
        self.assertGreater(abs(filtered_d_after_new_frame[3]), 1e-3)

        # Tick 3: Intermediate 50 Hz tick (t=1.053s), NO new frame (frame_idx still 101)
        telem_cached = dict(telem_1)  # same frame_idx 101, same timestamp
        cmd3 = ctrl.compute_cmd(telem_cached, dt=0.02)

        # Derivative MUST BE HELD smoothly across the cached frame (not zero-pulsed or decayed!)
        np.testing.assert_allclose(ctrl.filtered_deriv, filtered_d_after_new_frame, rtol=1e-5)

    def test_coasting_integrity(self):
        """Verifies that during coasting, in_view is False, integrators freeze/decay, and optical looming decays."""
        ctrl = KinematicVisualServoController(
            desired_bbox_size=48.0,
            coast_mode="decay",
            ki_forward=0.5,
            enable_looming_ff=True,
        )

        # 1. Active lock accumulates integrator
        telem_lock = {
            "status": "LOCKED",
            "in_view": True,
            "is_coasting": False,
            "error_x": 0.0,
            "error_y": 0.0,
            "bbox_w": 40.0,
            "target_size": 40.0,
            "frame_idx": 1,
            "timestamp": 1.0,
        }
        for _ in range(5):
            ctrl.compute_cmd(telem_lock, dt=0.02)

        integral_before_coast = ctrl.integral[0]
        self.assertGreater(integral_before_coast, 0.0)

        # 2. Transition to COASTING
        telem_coast = {
            "status": "COASTING",
            "in_view": False,
            "is_coasting": True,
            "error_x": 0.5,  # Ghost prediction
            "error_y": 0.2,
            "bbox_w": 35.0,
            "target_size": 35.0,
            "frame_idx": 2,
            "timestamp": 1.033,
        }
        ctrl.compute_cmd(telem_coast, dt=0.02)

        # Integrator MUST NOT wind up further during coasting
        self.assertLessEqual(ctrl.integral[0], integral_before_coast)

        # 3. If coast_mode == "brake", output is immediately zero
        ctrl_brake = KinematicVisualServoController(coast_mode="brake")
        cmd_brake = ctrl_brake.compute_cmd(telem_coast, dt=0.02)
        np.testing.assert_allclose(cmd_brake, np.zeros(4))

    def test_bumpless_reacquisition_no_derivative_kick(self):
        """Verifies that re-acquiring a target after loss/prolonged coasting does not kick derivatives or looming."""
        ctrl = KinematicVisualServoController(
            desired_bbox_size=48.0,
            kd_yaw=30.0,
            enable_looming_ff=True,
        )

        # 1. Active lock on left side of FOV
        telem_init = {
            "status": "LOCKED",
            "in_view": True,
            "error_x": -0.40,
            "error_y": 0.0,
            "bbox_w": 64.0,
            "target_size": 64.0,
            "frame_idx": 10,
            "timestamp": 1.0,
        }
        ctrl.compute_cmd(telem_init, dt=0.02)

        # 2. Prolonged loss / search (> 100ms)
        telem_search = {
            "status": "SEARCHING",
            "in_view": False,
            "error_x": 0.0,
            "error_y": 0.0,
            "bbox_w": 0.0,
            "target_size": 0.0,
            "frame_idx": 15,
            "timestamp": 1.5,
        }
        for _ in range(5):
            ctrl.compute_cmd(telem_search, dt=0.02)

        # 3. Sudden re-acquisition on the OPPOSITE side of FOV with different distance
        telem_reacq = {
            "status": "LOCKED",
            "in_view": True,
            "error_x": +0.50,  # Jumped from -0.40 to +0.50!
            "error_y": 0.0,
            "bbox_w": 32.0,   # Shrunk from 64px to 32px!
            "target_size": 32.0,
            "frame_idx": 20,
            "timestamp": 2.0,
        }
        cmd_reacq = ctrl.compute_cmd(telem_reacq, dt=0.02)

        # On the exact frame of re-acquisition:
        # Derivative term MUST BE ZERO (bumpless transfer, no derivative kick!)
        self.assertEqual(ctrl.filtered_deriv[3], 0.0)
        self.assertEqual(ctrl.filtered_dw, 0.0)

        # 4. Next frame (continuous tracking established on newly acquired target)
        telem_next = {
            "status": "LOCKED",
            "in_view": True,
            "error_x": +0.52,  # Smooth physical motion (delta_x = +0.02)
            "error_y": 0.0,
            "bbox_w": 32.0,
            "target_size": 32.0,
            "frame_idx": 21,
            "timestamp": 2.033,
        }
        cmd_next = ctrl.compute_cmd(telem_next, dt=0.02)

        # Derivative term should now be active, measuring only the fresh physical delta (+0.02 over 33ms)
        self.assertGreater(abs(ctrl.filtered_deriv[3]), 1e-4)
        # But NOT a crazy spike
        self.assertLess(abs(ctrl.filtered_deriv[3]), 5.0)


if __name__ == "__main__":
    unittest.main()

