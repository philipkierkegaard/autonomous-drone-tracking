#!/usr/bin/env python3
"""
Comprehensive 6-Axis System Identification and Latency Analysis Script.
Analyzes ArduPilot .BIN flight logs across:
  1. Pitch Attitude (deg)
  2. Roll Attitude (deg)
  3. Yaw Rate (deg/s) — using YDes / Y from RATE messages
  4. Vertical Climb / Descent Velocity (m/s) — from CTUN
  5. North Horizontal Velocity (m/s) — from PSCN (TVN / DVN vs VN)
  6. East Horizontal Velocity (m/s) — from PSCE (TVE / DVE vs VE)

Features:
- Native sampling rate preserved per signal (no cross-signal oversampling).
- Multi-start global/local optimization over (tau, T_d, K) with coarse delay grid search.
- Identifiability analysis: detects boundary hits and flat cost profiles.
- Chronological Train / Validation split to evaluate generalization on unseen maneuvers.
- Correct ArduPilot target labels (no misleading "stick" labels in autonomous mode).
- Generates publication-ready 6-axis and dedicated velocity & yaw plots.
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import lfilter, lfiltic
from pymavlink import mavutil


def load_flight_log(log_path: str) -> Dict[str, pd.DataFrame]:
    """
    Parses ArduPilot DataFlash .BIN log and extracts telemetry streams.
    Returns dictionary of DataFrames for attitude, rate, altitude, and horizontal velocity.
    """
    print(f"Reading log: {log_path}...")
    mlog = mavutil.mavlink_connection(log_path)

    att_data = []
    rate_data = []
    ctun_data = []
    pscn_data = []
    psce_data = []
    mode_data = []

    while True:
        m = mlog.recv_msg()
        if m is None:
            break
        mtype = m.get_type()
        t_us = getattr(m, 'TimeUS', None)
        if t_us is None:
            continue
        t_sec = t_us / 1e6

        if mtype in ['ANG', 'ATT']:
            d = m.to_dict()
            att_data.append({
                'msg_type': mtype,
                'time': t_sec,
                'des_roll': d.get('DesRoll', 0.0),
                'roll': d.get('Roll', 0.0),
                'des_pitch': d.get('DesPitch', 0.0),
                'pitch': d.get('Pitch', 0.0),
                'des_yaw': d.get('DesYaw', 0.0),
                'yaw': d.get('Yaw', 0.0)
            })
        elif mtype == 'RATE':
            d = m.to_dict()
            rate_data.append({
                'time': t_sec,
                # Yaw Rate fields: YDes and Y
                'yaw_rate_des': d.get('YDes', 0.0),
                'yaw_rate_act': d.get('Y', 0.0),
                # Roll Rate fields: RDes and R
                'roll_rate_des': d.get('RDes', 0.0),
                'roll_rate_act': d.get('R', 0.0),
                # Pitch Rate fields: PDes and P
                'pitch_rate_des': d.get('PDes', 0.0),
                'pitch_rate_act': d.get('P', 0.0),
            })
        elif mtype == 'CTUN':
            d = m.to_dict()
            # DCRt and CRt are in cm/s -> convert to m/s
            ctun_data.append({
                'time': t_sec,
                'vz_des': d.get('DCRt', 0.0) / 100.0,
                'vz_act': d.get('CRt', 0.0) / 100.0,
                'alt': d.get('Alt', 0.0)
            })
        elif mtype == 'PSCN':
            d = m.to_dict()
            pscn_data.append({
                'time': t_sec,
                'tvn': d.get('TVN', 0.0),  # Target Velocity North (companion / guidance)
                'dvn': d.get('DVN', 0.0),  # Desired Velocity North (ArduPilot shaped)
                'vn': d.get('VN', 0.0),    # Actual Velocity North (EKF)
            })
        elif mtype == 'PSCE':
            d = m.to_dict()
            psce_data.append({
                'time': t_sec,
                'tve': d.get('TVE', 0.0),  # Target Velocity East (companion / guidance)
                'dve': d.get('DVE', 0.0),  # Desired Velocity East (ArduPilot shaped)
                've': d.get('VE', 0.0),    # Actual Velocity East (EKF)
            })
        elif mtype == 'MODE':
            mode_num = getattr(m, 'Mode', getattr(m, 'ModeNum', 0))
            mode_data.append({'time': t_sec, 'mode': mode_num})

    df_att_raw = pd.DataFrame(att_data)
    if not df_att_raw.empty and 'ANG' in df_att_raw['msg_type'].values:
        # Prioritize 400 Hz ANG messages over legacy down-sampled 10 Hz ATT
        df_att = df_att_raw[df_att_raw['msg_type'] == 'ANG'].drop(columns=['msg_type']).reset_index(drop=True)
    elif not df_att_raw.empty:
        df_att = df_att_raw.drop(columns=['msg_type']).reset_index(drop=True)
    else:
        df_att = pd.DataFrame()

    return {
        'att': df_att,
        'rate': pd.DataFrame(rate_data),
        'ctun': pd.DataFrame(ctun_data),
        'pscn': pd.DataFrame(pscn_data),
        'psce': pd.DataFrame(psce_data),
        'mode': pd.DataFrame(mode_data),
    }


def simulate_fopdt(
    t: np.ndarray,
    u_target: np.ndarray,
    y_init: float,
    tau: float,
    T_d: float,
    K: float = 1.0
) -> np.ndarray:
    """
    Simulates First-Order Plus Dead-Time (FOPDT) response:
      tau * dy/dt + y(t) = K * u(t - T_d)
    Uses exact discrete IIR filter with linear interpolation for fractional delay.
    """
    if len(t) < 2:
        return np.full_like(u_target, y_init)

    dt = np.mean(np.diff(t))
    # Continuous fractional delay via linear interpolation
    u_delayed = np.interp(t - T_d, t, u_target, left=u_target[0]) * K

    alpha = np.exp(-dt / max(tau, 1e-4))
    b = [1.0 - alpha]
    a = [1.0, -alpha]
    zi = lfiltic(b, a, y=[y_init])
    y_filt, _ = lfilter(b, a, u_delayed[:-1], zi=zi)

    y_sim = np.empty_like(u_target)
    y_sim[0] = y_init
    y_sim[1:] = y_filt
    return y_sim


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates RMSE, MAE, Pearson r, and Variance Accounted For (VAF)."""
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    var_true = float(np.var(y_true))
    vaf = float((1.0 - np.var(err) / (var_true + 1e-8)) * 100.0)

    if np.std(y_true) > 1e-4 and np.std(y_pred) > 1e-4:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = 1.0 if np.allclose(y_true, y_pred, atol=0.05) else 0.0

    return {'rmse': rmse, 'mae': mae, 'vaf': vaf, 'corr': corr}


def prepare_native_signal(
    df: pd.DataFrame,
    t_col: str,
    u_col: str,
    y_col: str,
    t_start: float,
    t_end: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Slices signal within [t_start, t_end] and builds a uniform grid matching
    the channel's native logging rate (no cross-channel upsampling).
    """
    sub = df[(df[t_col] >= t_start) & (df[t_col] <= t_end)].copy().reset_index(drop=True)
    if len(sub) < 5:
        raise ValueError(f"Insufficient telemetry samples ({len(sub)}) in window [{t_start}, {t_end}]")

    dt = float(np.median(sub[t_col].diff().dropna()))
    if np.isnan(dt) or dt <= 0:
        dt = 0.01

    t_unif = np.arange(sub[t_col].min(), sub[t_col].max(), dt)
    u_unif = np.interp(t_unif, sub[t_col], sub[u_col])
    y_unif = np.interp(t_unif, sub[t_col], sub[y_col])
    sample_hz = 1.0 / dt
    return t_unif, u_unif, y_unif, sample_hz


def fit_fopdt_multistart(
    t: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    bounds_tau: Tuple[float, float] = (0.005, 1.5),
    bounds_td: Tuple[float, float] = (0.0, 0.40),
    fit_gain: bool = False,
    bounds_k: Tuple[float, float] = (0.2, 1.8),
    n_td_grid: int = 40,
    n_tau_seeds: int = 5
) -> Dict[str, Any]:
    """
    Performs robust multi-start optimization:
    1. Evaluates coarse grid over (T_d, tau) to escape local minima.
    2. Polishes best candidate using L-BFGS-B bounded optimization.
    3. Runs identifiability check to detect boundary-hitting or unidentifiable parameters.
    """
    td_grid = np.linspace(bounds_td[0], bounds_td[1], n_td_grid)
    tau_seeds = np.logspace(np.log10(bounds_tau[0]), np.log10(bounds_tau[1]), n_tau_seeds)

    if fit_gain:
        k_seeds = [0.5, 1.0, 1.5]
        def cost(p):
            tau, td, k = p
            y_sim = simulate_fopdt(t, u, y[0], tau, td, K=k)
            return float(np.mean((y_sim - y) ** 2))
        
        best_cost = float('inf')
        best_cand = (tau_seeds[len(tau_seeds)//2], td_grid[len(td_grid)//2], 1.0)
        for td_c in td_grid:
            for tau_c in tau_seeds:
                for k_c in k_seeds:
                    c = cost([tau_c, td_c, k_c])
                    if c < best_cost:
                        best_cost = c
                        best_cand = (tau_c, td_c, k_c)

        bounds = [bounds_tau, bounds_td, bounds_k]
        res = minimize(cost, x0=list(best_cand), bounds=bounds, method='L-BFGS-B')
        best_tau, best_td, best_k = res.x
    else:
        best_k = 1.0
        def cost(p):
            tau, td = p
            y_sim = simulate_fopdt(t, u, y[0], tau, td, K=1.0)
            return float(np.mean((y_sim - y) ** 2))

        best_cost = float('inf')
        best_cand = (tau_seeds[len(tau_seeds)//2], td_grid[len(td_grid)//2])
        for td_c in td_grid:
            for tau_c in tau_seeds:
                c = cost([tau_c, td_c])
                if c < best_cost:
                    best_cost = c
                    best_cand = (tau_c, td_c)

        bounds = [bounds_tau, bounds_td]
        res = minimize(cost, x0=list(best_cand), bounds=bounds, method='L-BFGS-B')
        best_tau, best_td = res.x

    # --- Identifiability Checks ---
    at_lower_td = (best_td <= bounds_td[0] + 0.002)
    at_upper_td = (best_td >= bounds_td[1] - 0.005)
    at_lower_tau = (best_tau <= bounds_tau[0] + 0.001)
    at_upper_tau = (best_tau >= bounds_tau[1] - 0.01)

    # Delay sensitivity check: cost variation across +/- 20 ms
    step_td = 0.02
    if fit_gain:
        cost_m = cost([best_tau, max(bounds_td[0], best_td - step_td), best_k])
        cost_p = cost([best_tau, min(bounds_td[1], best_td + step_td), best_k])
    else:
        cost_m = cost([best_tau, max(bounds_td[0], best_td - step_td)])
        cost_p = cost([best_tau, min(bounds_td[1], best_td + step_td)])

    rel_sensitivity = max(abs(cost_m - res.fun), abs(cost_p - res.fun)) / (res.fun + 1e-8)

    status = "CALIBRATED"
    notes = []
    if at_upper_td:
        status = "AT_UPPER_BOUND"
        notes.append(f"T_d hit upper bound ({bounds_td[1]*1000:.0f} ms)")
    elif at_lower_td:
        status = "AT_LOWER_BOUND"
        notes.append("T_d hit 0 ms")
    elif rel_sensitivity < 0.005:
        status = "UNIDENTIFIABLE"
        notes.append("Delay profile flat (low sensitivity)")

    if at_lower_tau:
        notes.append(f"tau hit lower bound ({bounds_tau[0]*1000:.1f} ms)")
    elif at_upper_tau:
        notes.append(f"tau hit upper bound ({bounds_tau[1]*1000:.1f} ms)")

    return {
        'tau': float(best_tau),
        'td': float(best_td),
        'k': float(best_k),
        'status': status,
        'notes': "; ".join(notes) if notes else "Identifiable global minimum",
        'identifiable': (status == "CALIBRATED"),
        'rmse_train': float(np.sqrt(res.fun)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="6-Axis FOPDT System Identification and Latency Analysis from ArduPilot .BIN log"
    )
    parser.add_argument("log", nargs="?", default="logs/flight_logs/00000165.BIN", help="Path to .BIN flight log")
    parser.add_argument("--t-start", type=float, default=None, help="Start timestamp in seconds")
    parser.add_argument("--t-end", type=float, default=None, help="End timestamp in seconds")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.30,
        help="Fraction of time window reserved for out-of-sample validation (default: 0.30)"
    )
    parser.add_argument(
        "--vel-target",
        choices=["target", "desired"],
        default="target",
        help="Horizontal velocity reference: 'target' (TVN/TVE from companion) or 'desired' (DVN/DVE from ArduPilot)"
    )
    parser.add_argument(
        "--no-yaw-gain",
        dest="fit_yaw_gain",
        action="store_false",
        default=True,
        help="Fix steady-state gain K=1.0 for yaw rate (default: fits gain K)"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="logs/flight_logs",
        help="Directory to save output identification plots"
    )
    args = parser.parse_args()

    log_file = args.log
    if not os.path.exists(log_file):
        print(f"[ERROR] Flight log not found: {log_file}")
        sys.exit(1)

    dfs = load_flight_log(log_file)
    df_att = dfs['att']
    df_rate = dfs['rate']
    df_ctun = dfs['ctun']
    df_pscn = dfs['pscn']
    df_psce = dfs['psce']

    if df_att.empty:
        print("[ERROR] No attitude telemetry found in log file.")
        sys.exit(1)

    # Window selection
    if args.t_start is not None and args.t_end is not None:
        t_start = float(args.t_start)
        t_end = float(args.t_end)
    elif "00000165.BIN" in str(log_file):
        # Default to longest autonomous guided engagement
        t_start, t_end = 302.07, 340.42
    elif "00000135.BIN" in str(log_file):
        t_start, t_end = 357.54, 413.89
    elif "00000080.BIN" in str(log_file):
        t_start, t_end = 227.0, 317.0
    else:
        t_start = df_att['time'].min() + 2.0
        t_end = df_att['time'].max() - 1.0

    print(f"\nAnalyzing time window: t = {t_start:.2f}s to {t_end:.2f}s (Duration: {t_end - t_start:.2f}s)")
    if args.val_split > 0:
        val_duration = (t_end - t_start) * args.val_split
        train_duration = (t_end - t_start) * (1.0 - args.val_split)
        print(f"Split: Fit/Train = {train_duration:.2f}s (70%) | Validation = {val_duration:.2f}s (30%)")
    else:
        print("Split: Full window used for fit (no validation split)")

    vel_n_col = 'tvn' if args.vel_target == 'target' else 'dvn'
    vel_e_col = 'tve' if args.vel_target == 'target' else 'dve'
    vel_label_type = "Target" if args.vel_target == 'target' else "Desired"

    channel_specs = [
        {
            'name': 'Pitch Attitude',
            'axis_key': 'pitch',
            'df': df_att,
            'tc': 'time', 'uc': 'des_pitch', 'yc': 'pitch',
            'unit': 'deg',
            'b_tau': (0.005, 0.80), 'b_td': (0.0, 0.50),
            'fit_gain': False,
            'u_label': 'ArduPilot target Pitch',
            'y_label': 'Actual Pitch (EKF)'
        },
        {
            'name': 'Roll Attitude',
            'axis_key': 'roll',
            'df': df_att,
            'tc': 'time', 'uc': 'des_roll', 'yc': 'roll',
            'unit': 'deg',
            'b_tau': (0.005, 0.80), 'b_td': (0.0, 0.50),
            'fit_gain': False,
            'u_label': 'ArduPilot target Roll',
            'y_label': 'Actual Roll (EKF)'
        },
        {
            'name': 'Yaw Rate',
            'axis_key': 'yaw_rate',
            'df': df_rate,
            'tc': 'time', 'uc': 'yaw_rate_des', 'yc': 'yaw_rate_act',
            'unit': 'deg/s',
            'b_tau': (0.005, 1.50), 'b_td': (0.0, 0.60),
            'fit_gain': args.fit_yaw_gain,
            'u_label': 'ArduPilot target Yaw Rate',
            'y_label': 'Actual Yaw Rate (Gyro)'
        },
        {
            'name': 'Climb Rate',
            'axis_key': 'vz',
            'df': df_ctun,
            'tc': 'time', 'uc': 'vz_des', 'yc': 'vz_act',
            'unit': 'm/s',
            'b_tau': (0.02, 1.50), 'b_td': (0.0, 0.40),
            'fit_gain': False,
            'u_label': 'ArduPilot target Climb Rate',
            'y_label': 'Actual Climb Rate (Baro/EKF)'
        },
        {
            'name': 'North Velocity',
            'axis_key': 'vn',
            'df': df_pscn,
            'tc': 'time', 'uc': vel_n_col, 'yc': 'vn',
            'unit': 'm/s',
            'b_tau': (0.05, 2.50), 'b_td': (0.0, 0.60),
            'fit_gain': False,
            'u_label': f'ArduPilot {vel_label_type} North Velocity',
            'y_label': 'Actual North Velocity (EKF)'
        },
        {
            'name': 'East Velocity',
            'axis_key': 've',
            'df': df_psce,
            'tc': 'time', 'uc': vel_e_col, 'yc': 've',
            'unit': 'm/s',
            'b_tau': (0.05, 2.50), 'b_td': (0.0, 0.60),
            'fit_gain': False,
            'u_label': f'ArduPilot {vel_label_type} East Velocity',
            'y_label': 'Actual East Velocity (EKF)'
        },
    ]

    results = []
    plot_data = []

    print("\n" + "=" * 90)
    print(f"       COMPLETE 6-AXIS SYSTEM IDENTIFICATION ({Path(log_file).stem.upper()})")
    print("=" * 90)
    print(f"{'CHANNEL':18s} | {'RATE':6s} | {'TAU':8s} | {'T_D':8s} | {'VAL RMSE':10s} | {'VAL VAF':8s} | {'STATUS':12s} | NOTES")
    print("-" * 90)

    for spec in channel_specs:
        df_sig = spec['df']
        if df_sig.empty:
            print(f"{spec['name']:18s} | [NO DATA IN LOG]")
            continue

        try:
            t_u, u_u, y_u, sample_hz = prepare_native_signal(
                df_sig, spec['tc'], spec['uc'], spec['yc'], t_start, t_end
            )
        except Exception as e:
            print(f"{spec['name']:18s} | [EXTRACTION ERROR: {e}]")
            continue

        # Train / Validation Split
        if args.val_split > 0 and len(t_u) > 10:
            split_idx = int(len(t_u) * (1.0 - args.val_split))
            t_train, u_train, y_train = t_u[:split_idx], u_u[:split_idx], y_u[:split_idx]
            t_val, u_val, y_val = t_u[split_idx:], u_u[split_idx:], y_u[split_idx:]
            t_split_sec = float(t_train[-1])
        else:
            t_train, u_train, y_train = t_u, u_u, y_u
            t_val, u_val, y_val = t_u, u_u, y_u
            t_split_sec = None

        fit_res = fit_fopdt_multistart(
            t_train, u_train, y_train,
            bounds_tau=spec['b_tau'],
            bounds_td=spec['b_td'],
            fit_gain=spec['fit_gain']
        )

        tau = fit_res['tau']
        td = fit_res['td']
        k = fit_res['k']

        # Training simulation & evaluation
        y_sim_train = simulate_fopdt(t_train, u_train, y_train[0], tau, td, K=k)
        metrics_train = calculate_metrics(y_train, y_sim_train)

        # Validation simulation & evaluation
        y_sim_val = simulate_fopdt(t_val, u_val, y_val[0], tau, td, K=k)
        metrics_val = calculate_metrics(y_val, y_sim_val)

        # Full window simulation for plotting
        y_sim_full = simulate_fopdt(t_u, u_u, y_u[0], tau, td, K=k)

        res_item = {
            'spec': spec,
            'rate_hz': sample_hz,
            'tau': tau,
            'td': td,
            'k': k,
            'status': fit_res['status'],
            'notes': fit_res['notes'],
            'identifiable': fit_res['identifiable'],
            'train': metrics_train,
            'val': metrics_val,
        }
        results.append(res_item)

        plot_data.append({
            'spec': spec,
            't': t_u,
            'u': u_u,
            'y': y_u,
            'y_sim': y_sim_full,
            't_split': t_split_sec,
            'tau': tau,
            'td': td,
            'k': k,
            'metrics_val': metrics_val,
            'metrics_train': metrics_train,
            'status': fit_res['status'],
        })

        unit = spec['unit']
        val_rmse_str = f"{metrics_val['rmse']:5.2f} {unit}"
        val_vaf_str = f"{metrics_val['vaf']:5.1f}%"
        status_disp = fit_res['status']
        if status_disp != "CALIBRATED":
            status_disp = f"*{status_disp}"

        print(
            f"{spec['name']:18s} | {sample_hz:4.0f}Hz | {tau*1000:6.1f}ms | {td*1000:6.1f}ms | "
            f"{val_rmse_str:10s} | {val_vaf_str:8s} | {status_disp:12s} | "
            f"Train VAF: {metrics_train['vaf']:4.1f}% [{fit_res['notes']}]"
        )

    print("=" * 90)
    print("* Parameters flagged with an asterisk hit search boundaries or have flat delay sensitivity.")
    print("  Do not report unidentifiable parameters as calibrated dynamic models.")

    # -------------------------------------------------------------------------
    # Plots Generation
    # -------------------------------------------------------------------------
    stem = Path(log_file).stem
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path("docs/reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Complete 6-Axis Plot
    n_plots = len(plot_data)
    if n_plots > 0:
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2.5 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]

        for i, p in enumerate(plot_data):
            ax = axes[i]
            sp = p['spec']
            t_rel = p['t'] - t_start

            ax.plot(t_rel, p['u'], 'k--', linewidth=1.5, label=sp['u_label'])
            ax.plot(t_rel, p['y'], 'C0-', linewidth=1.8, label=sp['y_label'])
            ax.plot(
                t_rel, p['y_sim'], 'C3:', linewidth=2.2,
                label=f"Model (tau={p['tau']*1000:.0f}ms, T_d={p['td']*1000:.0f}ms)"
            )

            if p['t_split'] is not None:
                split_rel = p['t_split'] - t_start
                ax.axvline(split_rel, color='gray', linestyle='-.', alpha=0.8, linewidth=1.2)
                if i == 0:
                    ax.text(split_rel * 0.5, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] != 0 else 0.5,
                            "FIT / TRAIN (70%)", color='dimgray', fontsize=9, fontweight='bold', ha='center')
                    ax.text(split_rel + (t_rel[-1] - split_rel) * 0.5,
                            ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] != 0 else 0.5,
                            "VALIDATION (30%)", color='indigo', fontsize=9, fontweight='bold', ha='center')

            title_str = (
                f"{i+1}. {sp['name']} Dynamics "
                f"[tau={p['tau']*1000:.1f}ms, T_d={p['td']*1000:.1f}ms | "
                f"Val RMSE={p['metrics_val']['rmse']:.2f}{sp['unit']}, Val VAF={p['metrics_val']['vaf']:.1f}%] "
                f"[{p['status']}]"
            )
            ax.set_title(title_str, fontsize=10, fontweight='bold')
            ax.set_ylabel(f"{sp['name'].split()[0]} ({sp['unit']})", fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

        axes[-1].set_xlabel(f"Time in Window (seconds relative to t = {t_start:.1f}s)", fontsize=10)
        plt.tight_layout()
        six_axis_path = out_dir / f"{stem}_6axis_sysid_plot.png"
        fig.savefig(six_axis_path, dpi=160)
        plt.close(fig)
        print(f"\nSaved 6-Axis SysID plot to: {six_axis_path}")

    # 2. Dedicated Standalone Horizontal Velocity Tracking Plot (North & East)
    vel_plots = [p for p in plot_data if p['spec']['axis_key'] in ['vn', 've']]
    if len(vel_plots) == 2:
        fig_vel, (ax_n, ax_e) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
        for ax, p, title_prefix in zip([ax_n, ax_e], vel_plots, ["North Velocity", "East Velocity"]):
            t_rel = p['t'] - t_start
            ax.plot(t_rel, p['u'], 'k--', linewidth=1.6, label=p['spec']['u_label'])
            ax.plot(t_rel, p['y'], 'C0-', linewidth=2.0, label=p['spec']['y_label'])
            ax.plot(
                t_rel, p['y_sim'], 'C3:', linewidth=2.2,
                label=f"FOPDT Fit (tau={p['tau']*1000:.0f}ms, T_d={p['td']*1000:.0f}ms)"
            )
            if p['t_split'] is not None:
                split_rel = p['t_split'] - t_start
                ax.axvline(split_rel, color='gray', linestyle='-.', alpha=0.8, linewidth=1.2)
            ax.set_title(
                f"{title_prefix} Response & Delay [tau={p['tau']*1000:.1f}ms, T_d={p['td']*1000:.1f}ms | "
                f"Val RMSE={p['metrics_val']['rmse']:.2f} m/s, Val VAF={p['metrics_val']['vaf']:.1f}%]",
                fontsize=11, fontweight='bold'
            )
            ax.set_ylabel("Velocity (m/s)", fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='upper right', fontsize=9)

        ax_e.set_xlabel(f"Time in Window (seconds relative to t = {t_start:.1f}s)", fontsize=10)
        plt.tight_layout()
        vel_out = fig_dir / f"{stem}_horizontal_vel_sysid_plot.png"
        fig_vel.savefig(vel_out, dpi=200)
        plt.close(fig_vel)
        print(f"Saved Horizontal Velocity SysID plot to: {vel_out}")

    # 3. Dedicated Standalone Yaw Rate Plot
    yaw_plots = [p for p in plot_data if p['spec']['axis_key'] == 'yaw_rate']
    if yaw_plots:
        p_yaw = yaw_plots[0]
        fig_yaw, ay = plt.subplots(figsize=(11, 4.5))
        t_rel = p_yaw['t'] - t_start
        ay.plot(t_rel, p_yaw['u'], 'k--', linewidth=1.6, label=p_yaw['spec']['u_label'])
        ay.plot(t_rel, p_yaw['y'], 'C2-', linewidth=2.0, label=p_yaw['spec']['y_label'])
        ay.plot(
            t_rel, p_yaw['y_sim'], 'C3:', linewidth=2.2,
            label=f"Model Fit (tau={p_yaw['tau']*1000:.1f}ms, T_d={p_yaw['td']*1000:.1f}ms)"
        )
        if p_yaw['t_split'] is not None:
            split_rel = p_yaw['t_split'] - t_start
            ay.axvline(split_rel, color='gray', linestyle='-.', alpha=0.8, linewidth=1.2)
            ay.text(split_rel * 0.5, ay.get_ylim()[1] * 0.88 if ay.get_ylim()[1] != 0 else 10,
                    "TRAIN (70%)", color='dimgray', fontsize=9, fontweight='bold', ha='center')
            ay.text(split_rel + (t_rel[-1] - split_rel) * 0.5,
                    ay.get_ylim()[1] * 0.88 if ay.get_ylim()[1] != 0 else 10,
                    "VALIDATION (30%)", color='indigo', fontsize=9, fontweight='bold', ha='center')

        ay.set_title(
            f"Yaw Rate Dynamics (YDes / Y) [tau={p_yaw['tau']*1000:.1f}ms, T_d={p_yaw['td']*1000:.1f}ms | "
            f"Val RMSE={p_yaw['metrics_val']['rmse']:.2f}°/s, Val VAF={p_yaw['metrics_val']['vaf']:.1f}%] "
            f"[{p_yaw['status']}]",
            fontsize=11, fontweight='bold'
        )
        ay.set_xlabel(f"Time in Window (seconds relative to t = {t_start:.1f}s)", fontsize=10)
        ay.set_ylabel("Yaw Rate (°/s)", fontsize=10)
        ay.grid(True, linestyle=':', alpha=0.6)
        ay.legend(loc='upper right', fontsize=9)
        plt.tight_layout()
        yaw_out = fig_dir / f"{stem}_yaw_sysid_plot.png"
        fig_yaw.savefig(yaw_out, dpi=200)
        plt.close(fig_yaw)
        print(f"Saved Yaw Rate SysID plot to: {yaw_out}")


if __name__ == "__main__":
    main()
