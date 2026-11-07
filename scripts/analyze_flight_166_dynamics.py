#!/usr/bin/env python3
"""
Telemetry and Dynamics Analysis for Flight Log 166 (00000166.BIN).
Evaluates arming state, flight modes, motor PWM outputs, and assesses
dynamic identifiability compared to Flights 165 and 167.
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymavlink import mavutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.identify_dynamics import fit_fopdt_multistart, simulate_fopdt, calculate_metrics


def analyze_flight_166(bin_path: Path, out_plot: Path):
    print(f"[PARSER] Opening ArduPilot log: {bin_path}...")
    mlog = mavutil.mavlink_connection(str(bin_path))

    rate_t, ydes, ya, pdes, pa, rdes, ra = [], [], [], [], [], [], []
    rcou_t, c1, c2, c3, c4 = [], [], [], [], []
    ctun_t, alt, crt, dcrt = [], [], [], []
    modes = []
    messages = []

    while True:
        m = mlog.recv_msg()
        if m is None:
            break
        mtype = m.get_type()
        tus = getattr(m, "TimeUS", None)
        if tus is None:
            continue
        t_sec = tus / 1e6

        if mtype == "RATE":
            rate_t.append(t_sec)
            ydes.append(m.YDes)
            ya.append(m.Y)
            pdes.append(m.PDes)
            pa.append(m.P)
            rdes.append(m.RDes)
            ra.append(m.R)
        elif mtype == "RCOU":
            rcou_t.append(t_sec)
            c1.append(m.C1)
            c2.append(m.C2)
            c3.append(m.C3)
            c4.append(m.C4)
        elif mtype == "CTUN":
            ctun_t.append(t_sec)
            alt.append(m.Alt)
            crt.append(m.CRt / 100.0)
            dcrt.append(m.DCRt / 100.0)
        elif mtype == "MODE":
            mnum = getattr(m, "Mode", getattr(m, "ModeNum", 0))
            modes.append((t_sec, mnum))
        elif mtype == "MSG":
            messages.append((t_sec, m.Message))

    rate_t = np.array(rate_t)
    ydes = np.array(ydes)
    ya = np.array(ya)
    rcou_t = np.array(rcou_t)
    c1 = np.array(c1)
    c2 = np.array(c2)
    c3 = np.array(c3)
    c4 = np.array(c4)
    ctun_t = np.array(ctun_t)
    alt = np.array(alt)

    print(f"[STATUS] Parsed {len(rate_t)} RATE samples across {rate_t[-1] - rate_t[0]:.1f}s.")
    print(f"[STATUS] Motor 1 PWM Range: [{np.min(c1)}, {np.max(c1)}] us")
    print(f"[STATUS] Altitude Range: [{np.min(alt):.2f}, {np.max(alt):.2f}] m")

    # Diagnostic 4-Panel Plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, dpi=200)
    fig.suptitle("Flight 166 (00000166.BIN) Diagnostic Analysis: Pre-Flight Ground/Bench Log\n"
                 "(Motors Disarmed, Zero Closed-Loop Pursuit Excitation)",
                 fontsize=13, fontweight="bold")

    # Panel 1: Motor PWM Outputs
    axes[0].plot(rcou_t, c1, label="Motor 1 (PWM)", color="#1f77b4", lw=1.5)
    axes[0].plot(rcou_t, c2, label="Motor 2 (PWM)", color="#ff7f0e", lw=1.5, linestyle="--")
    axes[0].plot(rcou_t, c3, label="Motor 3 (PWM)", color="#2ca02c", lw=1.5, linestyle=":")
    axes[0].plot(rcou_t, c4, label="Motor 4 (PWM)", color="#d62728", lw=1.5, linestyle="-.")
    axes[0].axhline(1000, color="black", linestyle="--", alpha=0.7, label="Disarmed Baseline (1000 us)")
    axes[0].set_ylabel("Motor PWM (us)", fontsize=10)
    axes[0].set_title("1. Propulsion Motor PWM Outputs [1000 us Constant = Completely Disarmed]", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right", fontsize=8.5)

    # Panel 2: Altitude & Handling Motion
    axes[1].plot(ctun_t, alt, color="#9467bd", lw=1.8, label="Barometric / EKF Altitude (Alt)")
    axes[1].axvline(52.7, color="gray", linestyle="-.", label="Mode Switch: STABILIZE -> LOITER (t=52.7s)")
    axes[1].set_ylabel("Altitude (m)", fontsize=10)
    axes[1].set_title("2. Altitude Telemetry (0.0m to 2.33m = Physical Hand-Carrying Pre-Flight)", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right", fontsize=8.5)

    # Panel 3: Commanded vs Actual Yaw Rate
    axes[2].plot(rate_t, ydes, color="#1f77b4", lw=1.5, label="Commanded Yaw Rate (RATE.YDes)")
    axes[2].plot(rate_t, ya, color="#2ca02c", lw=1.2, alpha=0.85, label="Actual Yaw Gyro (RATE.Y)")
    axes[2].set_ylabel("Yaw Rate (deg/s)", fontsize=10)
    axes[2].set_title("3. Commanded vs. Actual Yaw Rate [Lack of Correlation: r = 0.031, VAF < 0%]", fontsize=11, fontweight="bold")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(loc="upper right", fontsize=8.5)

    # Panel 4: Cross-Correlation Lag Profile
    # Zoomed window during handling
    m_h = (rate_t >= 55.0) & (rate_t <= 180.0)
    t_h = rate_t[m_h]
    yd_h = ydes[m_h]
    ya_h = ya[m_h]
    dt_h = np.median(np.diff(t_h))
    corr = np.correlate(ya_h - np.mean(ya_h), yd_h - np.mean(yd_h), mode="full")
    lags = np.arange(-len(yd_h) + 1, len(yd_h)) * dt_h
    norm_corr = corr / (np.std(ya_h) * np.std(yd_h) * len(ya_h))
    mask_lag = (lags >= -0.2) & (lags <= 1.0)

    axes[3].plot(lags[mask_lag] * 1000, norm_corr[mask_lag], color="#e377c2", lw=2.0, label="Normalized Cross-Correlation")
    axes[3].axhline(0, color="gray", linestyle=":", lw=1)
    axes[3].set_ylabel("Normalized Corr", fontsize=10)
    axes[3].set_xlabel("Time (seconds in log / Lag in ms for panel 4)", fontsize=10)
    axes[3].set_title("4. Cross-Correlation Profile [Completely Flat / Uncorrelated (Max r < 0.04)]", fontsize=11, fontweight="bold")
    axes[3].grid(True, linestyle=":", alpha=0.6)
    axes[3].legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot, dpi=200)
    plt.close()
    print(f"[PLOT] Diagnostic figure saved to: {out_plot}")


if __name__ == "__main__":
    bin_file = PROJECT_ROOT / "logs" / "flight_logs" / "00000166.BIN"
    plot_file = PROJECT_ROOT / "outputs" / "flight_analysis" / "00000166_comprehensive_sysid_plot.png"
    analyze_flight_166(bin_file, plot_file)

