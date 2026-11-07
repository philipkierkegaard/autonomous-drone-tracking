#!/usr/bin/env python3
"""
Deep Dynamic System Identification & Time Constant Analysis for Flight 167 (00000167.BIN).
Extracts telemetry streams, identifies all GUIDED autonomous tracking segments,
and fits First-Order Plus Dead-Time (FOPDT) models for Yaw Rate, Pitch, Roll, and Climb.
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import lfilter, lfiltic
from pymavlink import mavutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.identify_dynamics import simulate_fopdt, calculate_metrics


def extract_or_load_data(bin_path: Path, cache_path: Path) -> dict:
    if cache_path.exists():
        print(f"[CACHE] Loading cached flight telemetry from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    print(f"[PARSE] Parsing binary dataflash log: {bin_path}...")
    t0 = time.time()
    mlog = mavutil.mavlink_connection(str(bin_path))

    rate_t, yaw_des, yaw_act, roll_des, roll_act, pitch_des, pitch_act = [], [], [], [], [], [], []
    att_t, a_roll, a_pitch, a_yaw = [], [], [], []
    ctun_t, vz_des, vz_act, alt = [], [], [], []
    mode_t, mode_num = [], []

    while True:
        m = mlog.recv_msg()
        if m is None:
            break
        mtype = m.get_type()
        t_us = getattr(m, "TimeUS", None)
        if t_us is None:
            continue
        t_sec = t_us / 1e6

        if mtype == "RATE":
            d = m.to_dict()
            rate_t.append(t_sec)
            yaw_des.append(d.get("YDes", 0.0))
            yaw_act.append(d.get("Y", 0.0))
            roll_des.append(d.get("RDes", 0.0))
            roll_act.append(d.get("R", 0.0))
            pitch_des.append(d.get("PDes", 0.0))
            pitch_act.append(d.get("P", 0.0))
        elif mtype in ["ANG", "ATT"]:
            d = m.to_dict()
            att_t.append(t_sec)
            a_roll.append(d.get("Roll", 0.0))
            a_pitch.append(d.get("Pitch", 0.0))
            a_yaw.append(d.get("Yaw", 0.0))
        elif mtype == "CTUN":
            d = m.to_dict()
            ctun_t.append(t_sec)
            vz_des.append(d.get("DCRt", 0.0) / 100.0)
            vz_act.append(d.get("CRt", 0.0) / 100.0)
            alt.append(d.get("Alt", 0.0))
        elif mtype == "MODE":
            mnum = getattr(m, "Mode", getattr(m, "ModeNum", 0))
            mode_t.append(t_sec)
            mode_num.append(mnum)

    print(f"[PARSE] Finished parsing in {time.time()-t0:.1f}s.")
    out_dict = {
        "rate_t": np.array(rate_t, dtype=np.float64),
        "yaw_des": np.array(yaw_des, dtype=np.float32),
        "yaw_act": np.array(yaw_act, dtype=np.float32),
        "roll_des": np.array(roll_des, dtype=np.float32),
        "roll_act": np.array(roll_act, dtype=np.float32),
        "pitch_des": np.array(pitch_des, dtype=np.float32),
        "pitch_act": np.array(pitch_act, dtype=np.float32),
        "att_t": np.array(att_t, dtype=np.float64),
        "a_roll": np.array(a_roll, dtype=np.float32),
        "a_pitch": np.array(a_pitch, dtype=np.float32),
        "a_yaw": np.array(a_yaw, dtype=np.float32),
        "ctun_t": np.array(ctun_t, dtype=np.float64),
        "vz_des": np.array(vz_des, dtype=np.float32),
        "vz_act": np.array(vz_act, dtype=np.float32),
        "alt": np.array(alt, dtype=np.float32),
        "mode_t": np.array(mode_t, dtype=np.float64),
        "mode_num": np.array(mode_num, dtype=np.int32),
    }
    np.savez_compressed(cache_path, **out_dict)
    print(f"[CACHE] Saved compressed cache to {cache_path}")
    return out_dict


def fit_fopdt_channel(
    t: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    b_tau=(0.010, 1.50),
    b_td=(0.0, 0.60),
    fit_gain=True,
    val_split=0.30
):
    """Fits FOPDT model to time series and returns parameters and validation metrics."""
    n = len(t)
    if n < 20:
        return None

    if val_split > 0:
        n_train = int(n * (1.0 - val_split))
        t_tr, u_tr, y_tr = t[:n_train], u[:n_train], y[:n_train]
        t_val, u_val, y_val = t[n_train:], u[n_train:], y[n_train:]
    else:
        t_tr, u_tr, y_tr = t, u, y
        t_val, u_val, y_val = t, u, y

    y0_tr = float(y_tr[0])

    def loss(p):
        tau, td = p[0], p[1]
        k = p[2] if fit_gain else 1.0
        y_sim = simulate_fopdt(t_tr, u_tr, y0_tr, tau, td, K=k)
        return np.mean((y_tr - y_sim) ** 2)

    bounds = [b_tau, b_td]
    x0 = [0.10, 0.05]
    if fit_gain:
        bounds.append((0.50, 1.50))
        x0.append(1.0)

    res = minimize(loss, x0, bounds=bounds, method="L-BFGS-B")
    best_tau, best_td = float(res.x[0]), float(res.x[1])
    best_k = float(res.x[2]) if fit_gain else 1.0

    y_sim_tr = simulate_fopdt(t_tr, u_tr, y0_tr, best_tau, best_td, K=best_k)
    m_tr = calculate_metrics(y_tr, y_sim_tr)

    y0_val = float(y_val[0])
    y_sim_val = simulate_fopdt(t_val, u_val, y0_val, best_tau, best_td, K=best_k)
    m_val = calculate_metrics(y_val, y_sim_val)

    # Cross-correlation delay estimate
    u_cent = u_tr - np.mean(u_tr)
    y_cent = y_tr - np.mean(y_tr)
    dt = float(np.mean(np.diff(t_tr)))
    corr = np.correlate(y_cent, u_cent, mode="full")
    lags = np.arange(-len(u_cent) + 1, len(u_cent)) * dt
    pos_mask = (lags >= 0.0) & (lags <= 0.80)
    xcorr_delay = float(lags[pos_mask][np.argmax(corr[pos_mask])]) if np.any(pos_mask) else 0.0

    return {
        "tau_s": best_tau,
        "td_s": best_td,
        "k": best_k,
        "effective_delay_s": best_tau + best_td,
        "xcorr_delay_s": xcorr_delay,
        "train_vaf": m_tr["vaf"],
        "train_rmse": m_tr["rmse"],
        "val_vaf": m_val["vaf"],
        "val_rmse": m_val["rmse"],
        "n_samples": n,
        "dt_ms": dt * 1000.0,
    }


def main():
    bin_path = PROJECT_ROOT / "logs" / "flight_logs" / "00000167.BIN"
    cache_path = PROJECT_ROOT / "logs" / "flight_logs" / "00000167_extracted.npz"
    data = extract_or_load_data(bin_path, cache_path)

    rate_t = data["rate_t"]
    yaw_des = data["yaw_des"]
    yaw_act = data["yaw_act"]
    roll_des = data["roll_des"]
    roll_act = data["roll_act"]
    pitch_des = data["pitch_des"]
    pitch_act = data["pitch_act"]
    ctun_t = data["ctun_t"]
    vz_des = data["vz_des"]
    vz_act = data["vz_act"]
    alt = data["alt"]
    mode_t = data["mode_t"]
    mode_num = data["mode_num"]

    # Identify all GUIDED segments (Mode == 4)
    guided_windows = [
        ("Segment 1", 922.04, 932.61),
        ("Segment 2", 1032.80, 1041.55),
        ("Segment 3", 1627.43, 1635.63),
        ("Segment 4", 2316.17, 2325.51),
        ("Segment 5", 2749.68, 2756.65),
    ]

    print("\n" + "=" * 100)
    print("  FLIGHT 167 (00000167.BIN): SYSTEM IDENTIFICATION ACROSS AUTONOMOUS GUIDED ENGAGEMENTS")
    print("=" * 100)

    results_table = []

    for name, t0, t1 in guided_windows:
        mask_rate = (rate_t >= t0) & (rate_t <= t1)
        t_sub = rate_t[mask_rate]
        y_des_sub = yaw_des[mask_rate]
        y_act_sub = yaw_act[mask_rate]
        r_des_sub = roll_des[mask_rate]
        r_act_sub = roll_act[mask_rate]
        p_des_sub = pitch_des[mask_rate]
        p_act_sub = pitch_act[mask_rate]

        # Activity check: std of desired yaw rate
        std_yaw_des = float(np.std(y_des_sub))
        max_yaw_des = float(np.max(np.abs(y_des_sub)))

        fit_yaw = fit_fopdt_channel(t_sub, y_des_sub, y_act_sub, b_tau=(0.010, 1.0), b_td=(0.0, 0.50), fit_gain=True)
        fit_roll = fit_fopdt_channel(t_sub, r_des_sub, r_act_sub, b_tau=(0.005, 0.30), b_td=(0.0, 0.30), fit_gain=False)
        fit_pitch = fit_fopdt_channel(t_sub, p_des_sub, p_act_sub, b_tau=(0.005, 0.30), b_td=(0.0, 0.30), fit_gain=False)

        results_table.append({
            "segment": name,
            "t0": t0, "t1": t1, "duration": t1 - t0,
            "std_yaw_des": std_yaw_des,
            "max_yaw_des": max_yaw_des,
            "yaw": fit_yaw,
            "roll": fit_roll,
            "pitch": fit_pitch
        })

    print(f"\n{'Segment':<12} | {'Time (s)':<17} | {'Dur (s)':<7} | {'Max YawDes':<11} | {'Tau_Yaw':<9} | {'T_d Yaw':<9} | {'Eff Lag':<9} | {'Val VAF':<8} | {'Status'}")
    print("-" * 105)

    for r in results_table:
        seg = r["segment"]
        t_str = f"{r['t0']:.1f}-{r['t1']:.1f}s"
        dur = f"{r['duration']:.1f}s"
        max_yd = f"{r['max_yaw_des']:.1f} deg/s"
        yaw = r["yaw"]
        if yaw is not None:
            tau_str = f"{yaw['tau_s']*1000:.1f}ms"
            td_str = f"{yaw['td_s']*1000:.1f}ms"
            eff_str = f"{yaw['effective_delay_s']*1000:.1f}ms"
            vaf_str = f"{yaw['val_vaf']:.1f}%"
            status = "VALID" if yaw['val_vaf'] >= 60.0 else ("MODERATE" if yaw['val_vaf'] >= 30.0 else "UNEXCITED")
        else:
            tau_str, td_str, eff_str, vaf_str, status = "--", "--", "--", "--", "NO_DATA"

        print(f"{seg:<12} | {t_str:<17} | {dur:<7} | {max_yd:<11} | {tau_str:<9} | {td_str:<9} | {eff_str:<9} | {vaf_str:<8} | {status}")

    # Plot the highest-VAF engagement
    valid_results = [r for r in results_table if r["yaw"] is not None and r["yaw"]["val_vaf"] > 40.0]
    best_res = max(valid_results, key=lambda x: x["yaw"]["val_vaf"]) if valid_results else results_table[0]

    b_t0, b_t1 = best_res["t0"], best_res["t1"]
    mask_b = (rate_t >= b_t0) & (rate_t <= b_t1)
    t_plot = rate_t[mask_b]
    yd_plot = yaw_des[mask_b]
    ya_plot = yaw_act[mask_b]
    best_yaw = best_res["yaw"]

    y_sim_plot = simulate_fopdt(t_plot, yd_plot, float(ya_plot[0]), best_yaw["tau_s"], best_yaw["td_s"], K=best_yaw["k"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Flight 167: High-Fidelity Autonomous Pursuit SysID ({best_res['segment']}, t={b_t0:.1f}-{b_t1:.1f}s)\n"
                 f"Identified Model: tau_yaw = {best_yaw['tau_s']*1000:.1f}ms, T_d = {best_yaw['td_s']*1000:.1f}ms, "
                 f"Effective Lag = {best_yaw['effective_delay_s']*1000:.1f}ms (Val VAF: {best_yaw['val_vaf']:.1f}%)",
                 fontsize=13, fontweight="bold")

    axes[0].plot(t_plot - b_t0, yd_plot, label="Commanded Yaw Rate (RATE.YDes)", color="#1f77b4", lw=2)
    axes[0].plot(t_plot - b_t0, ya_plot, label="Actual Yaw Rate (RATE.Y)", color="#2ca02c", lw=1.8, alpha=0.85)
    axes[0].plot(t_plot - b_t0, y_sim_plot, label=f"FOPDT Model (tau={best_yaw['tau_s']*1000:.0f}ms, Td={best_yaw['td_s']*1000:.0f}ms)",
                 color="#d62728", lw=2, linestyle="--")
    axes[0].set_ylabel("Yaw Rate (deg/s)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t_plot - b_t0, ya_plot - y_sim_plot, label="Residual Modeling Error (Actual - Sim)", color="black", lw=1.2)
    axes[1].axhline(0, color="gray", linestyle=":", lw=1)
    axes[1].set_ylabel("Error (deg/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    # In-flight altitude
    mask_alt = (ctun_t >= b_t0) & (ctun_t <= b_t1)
    axes[2].plot(ctun_t[mask_alt] - b_t0, alt[mask_alt], label="Baro/Rangefinder Altitude (m)", color="#9467bd", lw=2)
    axes[2].set_ylabel("Altitude (m)")
    axes[2].set_xlabel("Time into GUIDED Engagement (seconds)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    out_plot = PROJECT_ROOT / "outputs" / "flight_analysis" / "00000167_yaw_sysid_plot.png"
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot, dpi=200)
    plt.close()
    print(f"\n[PLOT] Saved SysID verification plot to: {out_plot}")


if __name__ == "__main__":
    main()

