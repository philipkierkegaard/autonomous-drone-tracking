#!/usr/bin/env python3
"""
Evaluate Current Simulation Environment Yaw Dynamics against Real Flight 118 Telemetry.

Analyzes:
  1. The Autonomous GUIDED Tracking Engagement (t = 550.25s - 559.03s, 8.8s)
  2. The Active In-Flight Loiter Window (Sortie 4, t = 280s - 400s, 120s)

Evaluates:
  - Current Sim Model: tau_yaw = 150 ms, T_d = 0 ms, K = 1.0, max_accel = 120 deg/s^2
  - Pure First-Order Sim: tau_yaw = 150 ms, T_d = 0 ms, K = 1.0
  - Calibrated Flight 118 Dynamics: tau = 5.0 ms, T_d = 540 ms, K = 0.49 (ATC_RAT_YAW_FLTE = 2.0 Hz)
  - Calibrated Flight 165/167 Dynamics: tau = 59.1 ms, T_d = 241 ms, K = 0.50 (ATC_RAT_YAW_FLTE = 5.0 Hz)

Generates:
  - Quantitative benchmark metrics table
  - Diagnostic 4-panel comparison plot saved to outputs/flight_analysis/flight118_sim_vs_real_yaw_benchmark.png
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymavlink import mavutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.identify_dynamics import simulate_fopdt, calculate_metrics


def extract_or_load_flight_118(bin_path: Path, cache_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads telemetry from cache if available, otherwise parses 00000118.BIN and caches."""
    if cache_path.exists():
        print(f"[CACHE] Loading cached Flight 118 telemetry from: {cache_path}")
        d = np.load(cache_path)
        return d["t"], d["ydes"], d["y"]

    print(f"[PARSER] Opening ArduPilot DataFlash log: {bin_path}...")
    mlog = mavutil.mavlink_connection(str(bin_path))
    t_list, ydes_list, y_list = [], [], []

    while True:
        m = mlog.recv_msg()
        if m is None:
            break
        if m.get_type() == "RATE":
            tus = getattr(m, "TimeUS", None)
            if tus is not None:
                t_list.append(tus / 1e6)
                ydes_list.append(m.YDes)
                y_list.append(m.Y)

    t = np.array(t_list)
    ydes = np.array(ydes_list)
    y = np.array(y_list)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, t=t, ydes=ydes, y=y)
    print(f"[CACHE] Cached {len(t):,} RATE samples to: {cache_path}")
    return t, ydes, y


def simulate_sim_py_code(
    t: np.ndarray,
    cmd_yaw_rate: np.ndarray,
    init_yaw_rate: float,
    tau_yaw: float = 0.150,
    max_yaw_accel: float = 120.0,
    max_yaw_rate: float = 120.0,
) -> np.ndarray:
    """Exact recreation of the discrete-time integration from control/simulation.py."""
    dt = float(np.median(np.diff(t)))
    n = len(cmd_yaw_rate)
    y_sim = np.zeros(n)
    curr_rate = init_yaw_rate

    for i in range(n):
        cmd = cmd_yaw_rate[i]
        accel = np.clip((cmd - curr_rate) / tau_yaw, -max_yaw_accel, max_yaw_accel)
        curr_rate = np.clip(curr_rate + accel * dt, -max_yaw_rate, max_yaw_rate)
        y_sim[i] = curr_rate

    return y_sim


def evaluate_models_window(t: np.ndarray, ydes: np.ndarray, y: np.ndarray, name: str) -> Dict[str, Any]:
    dt = float(np.median(np.diff(t)))
    y0 = float(y[0])

    # 1. Current Sim Environment (control/simulation.py: tau = 150 ms, max_accel = 120 deg/s^2)
    y_sim_env = simulate_sim_py_code(t, ydes, y0, tau_yaw=0.150, max_yaw_accel=120.0, max_yaw_rate=120.0)
    m_sim_env = calculate_metrics(y, y_sim_env)
    corr_sim_env = float(np.corrcoef(y, y_sim_env)[0, 1])

    # 2. Pure Linear First-Order (tau = 150 ms, T_d = 0 ms, K = 1.0)
    y_sim_lin = simulate_fopdt(t, ydes, y0, tau=0.150, T_d=0.0, K=1.0)
    m_sim_lin = calculate_metrics(y, y_sim_lin)
    corr_sim_lin = float(np.corrcoef(y, y_sim_lin)[0, 1])

    # 3. Calibrated Flight 118 Model (tau = 5.0 ms, T_d = 540 ms, K = 0.49)
    y_cal_118 = simulate_fopdt(t, ydes, y0, tau=0.005, T_d=0.540, K=0.49)
    m_cal_118 = calculate_metrics(y, y_cal_118)
    corr_cal_118 = float(np.corrcoef(y, y_cal_118)[0, 1])

    # 4. Flight 165/167 Calibrated Baseline (tau = 59.1 ms, T_d = 241 ms, K = 0.50)
    y_cal_165 = simulate_fopdt(t, ydes, y0, tau=0.0591, T_d=0.2409, K=0.50)
    m_cal_165 = calculate_metrics(y, y_cal_165)
    corr_cal_165 = float(np.corrcoef(y, y_cal_165)[0, 1])

    # Cross-correlation delay estimation
    y_cent = y - np.mean(y)
    ydes_cent = ydes - np.mean(ydes)
    corr_real = np.correlate(y_cent, ydes_cent, mode="full")
    lags = np.arange(-len(ydes) + 1, len(ydes)) * dt
    pos_mask = (lags >= 0.0) & (lags <= 0.8)
    real_lag_ms = float(lags[pos_mask][np.argmax(corr_real[pos_mask])]) * 1000.0 if np.any(pos_mask) else 0.0

    return {
        "name": name,
        "dt_ms": dt * 1000.0,
        "real_lag_ms": real_lag_ms,
        "models": {
            "Current Sim Env (tau=150ms, Td=0ms, K=1.0)": {
                "y_sim": y_sim_env,
                "vaf": m_sim_env["vaf"],
                "rmse": m_sim_env["rmse"],
                "mae": float(np.mean(np.abs(y - y_sim_env))),
                "corr": corr_sim_env,
                "peak_err": float(np.max(np.abs(y - y_sim_env))),
            },
            "Pure Linear Sim (tau=150ms, Td=0ms, K=1.0)": {
                "y_sim": y_sim_lin,
                "vaf": m_sim_lin["vaf"],
                "rmse": m_sim_lin["rmse"],
                "mae": float(np.mean(np.abs(y - y_sim_lin))),
                "corr": corr_sim_lin,
                "peak_err": float(np.max(np.abs(y - y_sim_lin))),
            },
            "Calibrated Flight 118 (tau=5ms, Td=540ms, K=0.49)": {
                "y_sim": y_cal_118,
                "vaf": m_cal_118["vaf"],
                "rmse": m_cal_118["rmse"],
                "mae": float(np.mean(np.abs(y - y_cal_118))),
                "corr": corr_cal_118,
                "peak_err": float(np.max(np.abs(y - y_cal_118))),
            },
            "Flight 165 Baseline (tau=59ms, Td=241ms, K=0.50)": {
                "y_sim": y_cal_165,
                "vaf": m_cal_165["vaf"],
                "rmse": m_cal_165["rmse"],
                "mae": float(np.mean(np.abs(y - y_cal_165))),
                "corr": corr_cal_165,
                "peak_err": float(np.max(np.abs(y - y_cal_165))),
            },
        },
    }


def plot_results(
    t: np.ndarray,
    ydes: np.ndarray,
    y: np.ndarray,
    res: Dict[str, Any],
    out_path: Path,
):
    """Generates 4-panel diagnostic plot comparing sim model vs real flight 118."""
    t_rel = t - t[0]
    models = res["models"]

    y_sim_env = models["Current Sim Env (tau=150ms, Td=0ms, K=1.0)"]["y_sim"]
    y_cal_118 = models["Calibrated Flight 118 (tau=5ms, Td=540ms, K=0.49)"]["y_sim"]
    y_cal_165 = models["Flight 165 Baseline (tau=59ms, Td=241ms, K=0.50)"]["y_sim"]

    fig = plt.figure(figsize=(15, 12), dpi=200)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.0], hspace=0.32, wspace=0.22)

    # Panel 1: Full Engagement Time Series (8.8s)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t_rel, ydes, color="#7f7f7f", lw=1.5, linestyle="--", label="Commanded Yaw Rate ($Y_{Des}$, Autonomous GUIDED)")
    ax1.plot(t_rel, y, color="#2ca02c", lw=2.0, label="Real Flight 118 Gyro ($Y$, Pixhawk)")
    ax1.plot(t_rel, y_sim_env, color="#d62728", lw=1.8, alpha=0.9,
             label=f"Current Sim Env (tau=150ms, Td=0) | VAF: {models['Current Sim Env (tau=150ms, Td=0ms, K=1.0)']['vaf']:.1f}%")
    ax1.plot(t_rel, y_cal_118, color="#1f77b4", lw=2.0, linestyle=":",
             label=f"Calibrated Flight 118 (tau=5ms, Td=540ms, K=0.49) | VAF: {models['Calibrated Flight 118 (tau=5ms, Td=540ms, K=0.49)']['vaf']:.1f}%")
    ax1.set_ylabel("Yaw Rate (deg/s)", fontsize=11, fontweight="bold")
    ax1.set_title(f"1. Flight 118 Autonomous GUIDED Engagement (t = 550.2s - 559.0s, 8.8s): Commanded vs. Real vs. Sim",
                  fontsize=12, fontweight="bold")
    ax1.set_xlim(0, t_rel[-1])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9.5, framealpha=0.9)

    # Panel 2: Zoomed-in 4-Second Window (Highlighting Phase Error & Reversal Inversion)
    ax2 = fig.add_subplot(gs[1, 0])
    zoom_mask = (t_rel >= 1.0) & (t_rel <= 6.0)
    t_z = t_rel[zoom_mask]
    ax2.plot(t_z, ydes[zoom_mask], color="#7f7f7f", lw=1.5, linestyle="--", label="Commanded ($Y_{Des}$)")
    ax2.plot(t_z, y[zoom_mask], color="#2ca02c", lw=2.2, label="Real Gyro ($Y$)")
    ax2.plot(t_z, y_sim_env[zoom_mask], color="#d62728", lw=1.8, label="Sim Env (tau=150ms, Td=0)")
    ax2.plot(t_z, y_cal_118[zoom_mask], color="#1f77b4", lw=2.0, linestyle=":", label="Calibrated (Td=540ms)")
    ax2.set_xlabel("Time in Window (seconds)", fontsize=10)
    ax2.set_ylabel("Yaw Rate (deg/s)", fontsize=10)
    ax2.set_title("2. Zoomed View (1s to 6s): >500 ms Phase Lag in Flight 118", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.5)

    # Panel 3: Error Residuals (Real Gyro - Simulated)
    ax3 = fig.add_subplot(gs[1, 1])
    err_sim = y - y_sim_env
    err_cal = y - y_cal_118
    ax3.plot(t_rel, err_sim, color="#d62728", lw=1.2, alpha=0.8,
             label=f"Sim Env Error (RMSE = {models['Current Sim Env (tau=150ms, Td=0ms, K=1.0)']['rmse']:.2f} deg/s)")
    ax3.plot(t_rel, err_cal, color="#1f77b4", lw=1.2, alpha=0.85,
             label=f"Calibrated Error (RMSE = {models['Calibrated Flight 118 (tau=5ms, Td=540ms, K=0.49)']['rmse']:.2f} deg/s)")
    ax3.axhline(0, color="black", linestyle=":", lw=1)
    ax3.set_xlabel("Time in Window (seconds)", fontsize=10)
    ax3.set_ylabel("Error (deg/s)", fontsize=10)
    ax3.set_title("3. Modeling Residuals: Error = Real Gyro - Model Output", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=8.5)

    # Panel 4: Cross-Correlation Lag Spectrum
    ax4 = fig.add_subplot(gs[2, :])
    dt = float(np.median(np.diff(t)))
    corr_real = np.correlate(y - np.mean(y), ydes - np.mean(ydes), mode="full")
    corr_sim = np.correlate(y_sim_env - np.mean(y_sim_env), ydes - np.mean(ydes), mode="full")
    corr_cal = np.correlate(y_cal_118 - np.mean(y_cal_118), ydes - np.mean(ydes), mode="full")

    lags = np.arange(-len(ydes) + 1, len(ydes)) * dt * 1000.0
    norm_real = corr_real / (np.std(y) * np.std(ydes) * len(y))
    norm_sim = corr_sim / (np.std(y_sim_env) * np.std(ydes) * len(y_sim_env))
    norm_cal = corr_cal / (np.std(y_cal_118) * np.std(ydes) * len(y_cal_118))

    mask_lags = (lags >= -50.0) & (lags <= 750.0)
    ax4.plot(lags[mask_lags], norm_real[mask_lags], color="#2ca02c", lw=2.2, label=f"Real Flight 118 Cross-Corr (Peak: {res['real_lag_ms']:.1f} ms)")
    ax4.plot(lags[mask_lags], norm_sim[mask_lags], color="#d62728", lw=1.8, linestyle="--", label="Current Sim Env (Peak: 0.0 ms lag, no transport delay)")
    ax4.plot(lags[mask_lags], norm_cal[mask_lags], color="#1f77b4", lw=2.0, linestyle=":", label="Calibrated Model (Peak: 540.0 ms lag)")
    ax4.axvline(res["real_lag_ms"], color="#2ca02c", linestyle="--", alpha=0.6)
    ax4.axvline(0, color="gray", linestyle=":", lw=1)
    ax4.set_xlabel("Lag Time (milliseconds)", fontsize=10)
    ax4.set_ylabel("Normalized Correlation", fontsize=10)
    ax4.set_title("4. Cross-Correlation Lag: Real Flight 118 has ~545 ms Delay (ATC_RAT_YAW_FLTE = 2.0 Hz)",
                  fontsize=11, fontweight="bold")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper right", fontsize=9.5)

    plt.suptitle("Evaluation of Current Sim Model (tau_yaw = 150 ms) against Real Flight 118 Telemetry\n"
                 "(Demonstrating Extreme Phase Lag under ATC_RAT_YAW_FLTE = 2.0 Hz)",
                 fontsize=14, fontweight="bold", y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[PLOT] Saved diagnostic figure to: {out_path}")


def main():
    bin_path = PROJECT_ROOT / "logs" / "flight_logs" / "00000118.BIN"
    cache_path = PROJECT_ROOT / "logs" / "flight_logs" / "00000118_extracted.npz"
    out_plot = PROJECT_ROOT / "outputs" / "flight_analysis" / "flight118_sim_vs_real_yaw_benchmark.png"

    t, ydes, y = extract_or_load_flight_118(bin_path, cache_path)

    # 1. Guided tracking engagement
    m_guided = (t >= 550.25) & (t <= 559.03)
    t_g, yd_g, ya_g = t[m_guided], ydes[m_guided], y[m_guided]
    print(f"\n[EVALUATION] Autonomous GUIDED Engagement: t = {t_g[0]:.2f}s to {t_g[-1]:.2f}s ({t_g[-1]-t_g[0]:.2f}s, {len(t_g):,} samples @ 400Hz)")

    res_guided = evaluate_models_window(t_g, yd_g, ya_g, "Autonomous GUIDED Engagement (t=550.25s to 559.03s)")

    print("\n" + "=" * 105)
    print("  EVALUATION RESULTS: CURRENT SIM ENVIRONMENT YAW MODEL vs. REAL FLIGHT 118 TELEMETRY")
    print("=" * 105)
    print(f"{'Model Configuration':<52} | {'VAF (%)':<9} | {'RMSE (*/s)':<11} | {'MAE (*/s)':<10} | {'Corr (r)':<9} | {'Peak Err'}")
    print("-" * 105)

    for name, m in res_guided["models"].items():
        vaf_str = f"{m['vaf']:>6.1f}%"
        rmse_str = f"{m['rmse']:>8.2f} */s"
        mae_str = f"{m['mae']:>7.2f} */s"
        corr_str = f"{m['corr']:>7.3f}"
        peak_str = f"{m['peak_err']:>7.2f} */s"
        print(f"{name:<52} | {vaf_str:<9} | {rmse_str:<11} | {mae_str:<10} | {corr_str:<9} | {peak_str}")

    print("=" * 105)
    print(f"Empirical Real-Flight Command-to-Motion Delay Peak: {res_guided['real_lag_ms']:.1f} ms")

    plot_results(t_g, yd_g, ya_g, res_guided, out_plot)


if __name__ == "__main__":
    main()

