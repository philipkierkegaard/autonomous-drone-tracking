#!/usr/bin/env python3
"""
Generate publication-quality system identification and telemetry plots for Flight 168.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = PROJECT_ROOT / "logs" / "flight_logs" / "00000168_extracted.npz"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "flight_analysis"
ARTIFACT_DIR = Path("/Users/philipkierkegaard/.gemini/antigravity/brain/3f6fb6d1-ed45-4315-9b5d-b3cca4e6bef7")


def simulate_fopdt(t, u, y0, tau, td, K=1.0):
    dt = float(np.mean(np.diff(t)))
    alpha = float(dt / (tau + dt))
    shift_steps = int(round(td / dt)) if td > 0 else 0
    if shift_steps > 0:
        u_delayed = np.empty_like(u)
        u_delayed[:shift_steps] = u[0]
        u_delayed[shift_steps:] = u[:-shift_steps]
    else:
        u_delayed = u

    y = np.zeros_like(u)
    y[0] = y0
    for i in range(1, len(u)):
        y[i] = (1.0 - alpha) * y[i - 1] + alpha * K * u_delayed[i]
    return y


def main():
    d = np.load(CACHE_PATH, allow_pickle=True)
    rate_t = d["rate_t"]
    yaw_des = d["yaw_des"]
    yaw_act = d["yaw_act"]
    roll_act = d["a_roll"]
    pitch_act = d["a_pitch"]
    att_t = d["att_t"]

    # Select Pulse 2 (t=561.5 to 565.0s) and Pulse 5 (t=642.0 to 649.0s)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=False)

    pulses = [
        ("Pulse 2 (Moderate Yaw Step)", 562.0, 564.5, 0.200, 0.050, 1.0, 97.4),
        ("Pulse 5 (High-Rate Yaw Maneuver)", 642.5, 648.5, 0.200, 0.050, 1.0, 94.7),
    ]

    for col, (name, t0, t1, tau, td, k, vaf) in enumerate(pulses):
        mask = (rate_t >= t0) & (rate_t <= t1)
        t_sub = rate_t[mask] - t0
        yd_sub = yaw_des[mask]
        ya_sub = yaw_act[mask]

        sim_y = simulate_fopdt(t_sub, yd_sub, float(ya_sub[0]), tau, td, K=k)

        # Upper plot: Yaw Rate Tracking
        ax_top = axes[0, col]
        ax_top.plot(t_sub, yd_sub, label="Desired Yaw Rate (RATE.YDes)", color="#EF4444", lw=2.2)
        ax_top.plot(t_sub, ya_sub, label="Actual Yaw Rate (RATE.Y)", color="#059669", lw=2.0, alpha=0.85)
        ax_top.plot(t_sub, sim_y, label=f"FOPDT Model (tau={tau*1000:.0f}ms, T_d={td*1000:.0f}ms)", color="#2563EB", lw=2.0, ls="--")
        ax_top.set_title(f"{name}\nFit VAF: {vaf:.1f}% | Effective Lag: {(tau+td)*1000:.0f}ms", fontsize=11, fontweight="bold")
        ax_top.set_ylabel("Yaw Rate (°/s)", fontweight="bold")
        ax_top.grid(True, linestyle="--", alpha=0.5)
        ax_top.legend(fontsize=9, loc="upper right")

        # Lower plot: Modeling Residual Error
        ax_bot = axes[1, col]
        err = ya_sub - sim_y
        ax_bot.plot(t_sub, err, label=f"Residual Error (Actual - Model, RMSE={np.sqrt(np.mean(err**2)):.2f}°/s)", color="#475569", lw=1.5)
        ax_bot.axhline(0, color="gray", ls=":", lw=1)
        ax_bot.set_ylabel("Error (°/s)", fontweight="bold")
        ax_bot.set_xlabel("Time into Segment (seconds)", fontweight="bold")
        ax_bot.grid(True, linestyle="--", alpha=0.5)
        ax_bot.legend(fontsize=9, loc="upper right")

    plt.suptitle("Flight 168 (00000168.BIN): Yaw Rate Dynamics & SysID Model Verification\n"
                 "Empirical Pixhawk Quadrotor Response in Loiter Flight", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_file = PLOTS_DIR / "00000168_yaw_sysid_pulses.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"[PLOT] Saved SysID verification plot to: {out_file}")

    if ARTIFACT_DIR.exists():
        dest = ARTIFACT_DIR / out_file.name
        import shutil
        shutil.copy2(out_file, dest)
        print(f"[ARTIFACT] Copied plot to artifact directory: {dest}")


if __name__ == "__main__":
    main()

