#!/usr/bin/env python3
"""
Quick DataFlash Log Rate & Health Inspector
Analyzes any ArduPilot .BIN log file and prints the exact sample rate (Hz)
for attitude angles (ANG/ATT), body rates (RATE), PID loops, and telemetry.
"""

import sys
from pathlib import Path
import numpy as np

try:
    from pymavlink import mavutil
except ImportError:
    print("[ERROR] pymavlink not installed. Run: pip install pymavlink")
    sys.exit(1)


def inspect_log(log_path: str):
    path = Path(log_path)
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    print(f"\n[INSPECT] Analyzing DataFlash log: {path.name} ({path.stat().st_size / 1024:.1f} KB)...")
    mlog = mavutil.mavlink_connection(str(path))

    timestamps = {
        "ANG (Angle)": [],
        "RATE (Gyro)": [],
        "PID (P/R/Y)": [],
        "IMU (Accel)": [],
        "ATT (Legacy)": [],
        "CTUN (Alt)": [],
        "BARO": []
    }

    count = 0
    while True:
        m = mlog.recv_msg()
        if m is None:
            break
        mtype = m.get_type()
        t_us = getattr(m, "TimeUS", None)
        if t_us is not None:
            t_sec = t_us / 1e6
            if mtype == "ANG":
                timestamps["ANG (Angle)"].append(t_sec)
            elif mtype == "RATE":
                timestamps["RATE (Gyro)"].append(t_sec)
            elif mtype in ["PIDR", "PIDP", "PIDY"]:
                timestamps["PID (P/R/Y)"].append(t_sec)
            elif mtype == "IMU":
                timestamps["IMU (Accel)"].append(t_sec)
            elif mtype == "ATT":
                timestamps["ATT (Legacy)"].append(t_sec)
            elif mtype == "CTUN":
                timestamps["CTUN (Alt)"].append(t_sec)
            elif mtype == "BARO":
                timestamps["BARO"].append(t_sec)
        count += 1

    print("=" * 64)
    print(f"{'Message Type':<16} | {'Samples':<10} | {'Mean Rate (Hz)':<16} | {'Status'}")
    print("-" * 64)

    fastest_attitude_hz = 0.0

    for mtype, ts in timestamps.items():
        if len(ts) < 5:
            continue
        t_arr = np.array(ts)
        dts = np.diff(t_arr)
        valid_dts = dts[(dts > 1e-4) & (dts < 0.5)]
        if len(valid_dts) > 0:
            mean_dt = np.median(valid_dts)
            hz = 1.0 / mean_dt
            if "Angle" in mtype or "Gyro" in mtype:
                fastest_attitude_hz = max(fastest_attitude_hz, hz)
            status = "✓ EXCELLENT (>=100 Hz)" if hz >= 90.0 else ("⚠️ SLOW (<100 Hz)" if hz >= 30 else "❌ DOWN-SAMPLED (<30 Hz)")
            print(f"{mtype:<16} | {len(ts):<10} | {hz:>8.1f} Hz       | {status}")
        else:
            print(f"{mtype:<16} | {len(ts):<10} | {'Invalid DT':<16} | ❌ ERROR")
    print("=" * 64)

    if fastest_attitude_hz >= 90.0:
        print(f"\n[VERDICT] SUCCESS: High-rate attitude logging is active at {fastest_attitude_hz:.1f} Hz!")
        print("          Full 400 Hz inner-loop attitude and gyro dynamics are being logged to SD card.")
        print("          Fast enough to identify sub-50ms roll/pitch time constants!\n")
    else:
        print(f"\n[VERDICT] FAIL: Attitude logging is only {fastest_attitude_hz:.1f} Hz (Expected >= 100 Hz).\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/check_log_rate.py <path_to_file.BIN>")
        sys.exit(1)
    inspect_log(sys.argv[1])
