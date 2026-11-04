#!/usr/bin/env python3
"""
Unit Tests for ArduPilot Arming Diagnostic Tool (scripts/diagnose_arming.py).
Validates URI normalization, port scanning, telemetry ingestion, bitmask decoding,
and actionable troubleshooting recommendation generation.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.diagnose_arming import (
    ArmingDiagnostic,
    normalize_connection,
    scan_candidate_ports,
    GPS_REQUIRED_MODES,
)


class TestDiagnoseArming(unittest.TestCase):
    def test_normalize_connection(self):
        """Verifies port and baud parsing across different formats."""
        p, b = normalize_connection("/dev/ttyTHS0:921600")
        self.assertEqual(p, "/dev/ttyTHS0")
        self.assertEqual(b, 921600)

        p, b = normalize_connection("serial:///dev/ttyTHS1:115200")
        self.assertEqual(p, "/dev/ttyTHS1")
        self.assertEqual(b, 115200)

        p, b = normalize_connection("/dev/ttyACM0", default_baud=115200)
        self.assertEqual(p, "/dev/ttyACM0")
        self.assertEqual(b, 115200)

        p, b = normalize_connection("udpin:0.0.0.0:14550", default_baud=115200)
        self.assertEqual(p, "udpin:0.0.0.0:14550")
        self.assertEqual(b, 115200)

    def test_scan_candidate_ports_includes_udp(self):
        """Ensures default UDP endpoints are included in candidate list."""
        candidates = scan_candidate_ports()
        endpoints = [c[0] for c in candidates]
        self.assertIn("udpin:0.0.0.0:14550", endpoints)
        self.assertIn("udp:127.0.0.1:14550", endpoints)

    def test_diagnostic_report_detects_indoor_guided_mode(self):
        """Verifies warning when vehicle is in GUIDED mode without 3D GPS fix."""
        diag = ArmingDiagnostic()
        diag.flight_mode = "GUIDED"
        diag.is_armed = False
        diag.gps_raw = {
            "fix_type": 1,
            "fix_name": "1 (No Fix / Searching)",
            "satellites": 0,
            "hdop": 99.9,
            "vdop": 99.9,
            "alt_m": 0.0,
        }
        recs = diag.print_diagnostic_report()
        combined = " ".join(recs)
        self.assertIn("STABILIZE", combined)
        self.assertIn("GUIDED", combined)

    def test_diagnostic_report_detects_elevated_throttle(self):
        """Verifies warning when throttle stick is not down at zero."""
        diag = ArmingDiagnostic()
        diag.rc_channels = {"throttle": 1500, "rssi": 200, "roll": 1500, "pitch": 1500, "yaw": 1500}
        recs = diag.print_diagnostic_report()
        combined = " ".join(recs)
        self.assertIn("Throttle", combined)
        self.assertIn("1500", combined)

    def test_diagnostic_report_detects_high_compass_variance(self):
        """Verifies warning when EKF compass variance exceeds 1.0."""
        diag = ArmingDiagnostic()
        diag.ekf_status = {
            "flags": 0,
            "velocity_variance": 0.1,
            "pos_horiz_variance": 0.1,
            "pos_vert_variance": 0.1,
            "compass_variance": 1.45,
            "terrain_alt_variance": 0.0,
        }
        recs = diag.print_diagnostic_report()
        combined = " ".join(recs)
        self.assertIn("Compass Variance", combined)
        self.assertIn("Calibrate compass", combined)

    def test_diagnostic_report_detects_safety_switch(self):
        """Verifies recommendation when hardware safety switch is enabled."""
        diag = ArmingDiagnostic()
        diag.parameters = {"BRD_SAFETYENABLE": 1.0}
        recs = diag.print_diagnostic_report()
        combined = " ".join(recs)
        self.assertIn("safety switch", combined.lower())
        self.assertIn("BRD_SAFETYENABLE", combined)

    def test_diagnostic_report_detects_unhealthy_sensors(self):
        """Verifies sensor health decoding from SYS_STATUS."""
        diag = ArmingDiagnostic()
        # Gyro present and enabled, but unhealthy (bit 0 not in health)
        diag.sensors_present = 1 << 0
        diag.sensors_enabled = 1 << 0
        diag.sensors_health = 0
        recs = diag.print_diagnostic_report()
        combined = " ".join(recs)
        self.assertIn("3D Gyroscope 1", combined)


if __name__ == "__main__":
    unittest.main()
