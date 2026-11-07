#!/usr/bin/env python3
"""
5-Step Flight Dynamics Identification & Simulation Baseline Benchmark.
Analyzes Flight Log 00000120.BIN following the user methodology:

1. Establish Baseline:
   Evaluate original simulation parameters from control/simulation.py:
     - tau_h = 0.32 s, T_d = 0.0 s (North & East velocity)
     - tau_v = 0.22 s, T_d = 0.0 s (Vertical climb velocity)
     - tau_yaw = 0.15 s, T_d = 0.0 s (Yaw rate)
     - tau_att = 0.035 s, T_d = 0.0 s (Pitch & Roll attitude)
   Report baseline Train and Validation RMSE and VAF.

2. Optimize Parameters:
   Grid-search + local optimization over (tau, T_d) on Flight 120 training data.

3. Evaluate Identifiability (Delay Profiling):
   For fixed delays T_d in [0, 500 ms] (step 10 ms), re-optimize tau*(T_d) and record RMSE*(T_d).
   Plot RMSE*(T_d) and tau*(T_d). Determine the flat valley (within 2% of optimal RMSE)
   to quantify parameter coupling / delay uncertainty Delta T_d.

4. Validate on Unseen Data:
   - Within-flight out-of-sample validation on Flight 120 (70% train / 30% val).
   - Cross-flight out-of-sample validation on Flight 118 and Flight 135.
   - Input history buffer preserved across all evaluations to eliminate initial transients.

5. Compare Models:
   Compile comparison table (Baseline vs. Fitted), produce delay profile plots,
   and document explicit limitation regarding ArduPilot internal targets vs companion MAVLink commands.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
from scipy.signal import lfilter, lfiltic

# Import data loading routines from identify_dynamics
from control.identify_dynamics import load_flight_log, prepare_native_signal, calculate_metrics


def simulate_fopdt_buffered(
    t_full: np.ndarray,
    u_full: np.ndarray,
    eval_mask: np.ndarray,
    tau: float,
    T_d: float,
    K: float = 1.0,
    y_init: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates FOPDT continuous response over full signal preserving input history,
    then slices output strictly on eval_mask. Eliminates truncation transients.
    """
    dt = np.mean(np.diff(t_full))
    u_delayed = np.interp(t_full - T_d, t_full, u_full, left=u_full[0]) * K
    alpha = np.exp(-dt / max(tau, 1e-4))
    b = [1.0 - alpha]
    a = [1.0, -alpha]
    
    init_val = y_init if y_init is not None else u_delayed[0]
    zi = lfiltic(b, a, y=[init_val])
    y_filt, _ = lfilter(b, a, u_delayed[:-1], zi=zi)
    
    y_full = np.empty_like(u_full)
    y_full[0] = init_val
    y_full[1:] = y_filt
    
    return y_full, y_full[eval_mask]


def compute_delay_profile(
    t_full: np.ndarray,
    u_full: np.ndarray,
    y_full: np.ndarray,
    train_mask: np.ndarray,
    bounds_tau: Tuple[float, float] = (0.005, 1.50),
    td_grid: np.ndarray = np.linspace(0.0, 0.50, 51),
    K: float = 1.0
) -> Dict[str, Any]:
    """
    Step 3: Sweeps fixed delays T_d and optimizes tau*(T_d) at each point.
    Returns delay grid, optimal tau profile, and RMSE profile.
    """
    y_train = y_full[train_mask]
    y_init = y_full[0]
    
    rmse_profile = []
    tau_profile = []
    
    for td in td_grid:
        def obj(tau_val):
            _, y_sim_tr = simulate_fopdt_buffered(t_full, u_full, train_mask, tau_val, td, K=K, y_init=y_init)
            return float(np.mean((y_train - y_sim_tr) ** 2))
        
        res = minimize_scalar(obj, bounds=bounds_tau, method='bounded')
        opt_tau = float(res.x)
        opt_rmse = float(np.sqrt(res.fun))
        
        tau_profile.append(opt_tau)
        rmse_profile.append(opt_rmse)
        
    rmse_arr = np.array(rmse_profile)
    tau_arr = np.array(tau_profile)
    
    best_idx = int(np.argmin(rmse_arr))
    best_td = float(td_grid[best_idx])
    best_tau = float(tau_arr[best_idx])
    best_rmse = float(rmse_arr[best_idx])
    
    # Flat valley detection: all delays where RMSE is within 2% of minimum
    threshold = best_rmse * 1.02
    valley_mask = rmse_arr <= threshold
    valley_tds = td_grid[valley_mask]
    td_min_valley = float(np.min(valley_tds))
    td_max_valley = float(np.max(valley_tds))
    delta_td = td_max_valley - td_min_valley
    
    # If valley spans > 100 ms, conclude coupled / unidentifiable
    is_separable = (delta_td < 0.10)
    
    return {
        'td_grid': td_grid,
        'rmse_profile': rmse_arr,
        'tau_profile': tau_arr,
        'best_td': best_td,
        'best_tau': best_tau,
        'best_rmse': best_rmse,
        'threshold_2pct': threshold,
        'valley_td_min': td_min_valley,
        'valley_td_max': td_max_valley,
        'delta_td': delta_td,
        'is_separable': is_separable,
    }


def main():
    print("=" * 90)
    print(" 5-STEP DYNAMICS IDENTIFICATION & SIMULATION BENCHMARK (FLIGHT 120)")
    print("=" * 90)

    # 1. Load Primary Flight 120
    log_120_path = "logs/flight_logs/00000120.BIN"
    dfs_120 = load_flight_log(log_120_path)
    
    # 2. Load Cross-Flight Validation Data (Flight 118 and Flight 135)
    log_118_path = "logs/flight_logs/00000118.BIN"
    dfs_118 = load_flight_log(log_118_path)
    
    log_135_path = "logs/flight_logs/00000135.BIN"
    dfs_135 = load_flight_log(log_135_path)

    # Airborne window for Flight 120: t = 188.0s to 328.0s (140.0s)
    t_start_120, t_end_120 = 188.0, 328.0
    val_split = 0.30

    # Primary channels to evaluate with original simulation parameters
    axes = [
        {
            'name': 'North Velocity (Vn)',
            'key': 'vn',
            'df': dfs_120['pscn'], 'tc': 'time', 'uc': 'tvn', 'yc': 'vn',
            'df_118': dfs_118['pscn'], 't_118': (275.0, 405.0),
            'df_135': dfs_135['pscn'], 't_135': (357.5, 413.9),
            'unit': 'm/s',
            'tau_sim': 0.32,  # Original simulation baseline
            'b_tau': (0.05, 2.50),
            'K': 1.0,
        },
        {
            'name': 'East Velocity (Ve)',
            'key': 've',
            'df': dfs_120['psce'], 'tc': 'time', 'uc': 'tve', 'yc': 've',
            'df_118': dfs_118['psce'], 't_118': (275.0, 405.0),
            'df_135': dfs_135['psce'], 't_135': (357.5, 413.9),
            'unit': 'm/s',
            'tau_sim': 0.32,  # Original simulation baseline
            'b_tau': (0.05, 2.50),
            'K': 1.0,
        },
        {
            'name': 'Climb Velocity (Vz)',
            'key': 'vz',
            'df': dfs_120['ctun'], 'tc': 'time', 'uc': 'vz_des', 'yc': 'vz_act',
            'df_118': dfs_118['ctun'], 't_118': (275.0, 405.0),
            'df_135': dfs_135['ctun'], 't_135': (357.5, 413.9),
            'unit': 'm/s',
            'tau_sim': 0.22,  # Original simulation baseline
            'b_tau': (0.02, 1.50),
            'K': 1.0,
        },
        {
            'name': 'Yaw Rate (r)',
            'key': 'yaw_rate',
            'df': dfs_120['rate'], 'tc': 'time', 'uc': 'yaw_rate_des', 'yc': 'yaw_rate_act',
            'df_118': dfs_118['rate'], 't_118': (275.0, 405.0),
            'df_135': dfs_135['rate'], 't_135': (357.5, 413.9),
            'unit': 'deg/s',
            'tau_sim': 0.15,  # Original simulation baseline
            'b_tau': (0.005, 1.50),
            'K': 1.0,
        },
        {
            'name': 'Pitch Attitude',
            'key': 'pitch',
            'df': dfs_120['att'], 'tc': 'time', 'uc': 'des_pitch', 'yc': 'pitch',
            'df_118': dfs_118['att'], 't_118': (275.0, 405.0),
            'df_135': dfs_135['att'], 't_135': (357.5, 413.9),
            'unit': 'deg',
            'tau_sim': 0.035, # Calibrated in simulation.py
            'b_tau': (0.005, 0.80),
            'K': 1.0,
        },
        {
            'name': 'Roll Attitude',
            'key': 'roll',
            'df': dfs_120['att'], 'tc': 'time', 'uc': 'des_roll', 'yc': 'roll',
            'df_118': dfs_118['att'], 't_118': (275.0, 405.0),
            'df_135': dfs_135['att'], 't_135': (357.5, 413.9),
            'unit': 'deg',
            'tau_sim': 0.035, # Calibrated in simulation.py
            'b_tau': (0.005, 0.80),
            'K': 1.0,
        },
    ]

    report_rows = []
    profile_results = []
    comparison_plot_data = []

    td_grid_scan = np.linspace(0.0, 0.50, 51)  # 0 to 500 ms in 10 ms steps

    for axis in axes:
        print(f"\nProcessing {axis['name']}...")
        # 1. Native signal preparation on Flight 120
        t_full, u_full, y_full, hz = prepare_native_signal(
            axis['df'], axis['tc'], axis['uc'], axis['yc'], t_start_120, t_end_120
        )
        n_pts = len(t_full)
        split_idx = int(n_pts * (1.0 - val_split))
        train_mask = np.zeros(n_pts, dtype=bool)
        train_mask[:split_idx] = True
        val_mask = ~train_mask
        
        y_train = y_full[train_mask]
        y_val = y_full[val_mask]

        # ---------------------------------------------------------------------
        # Step 1: Establish Baseline Model (Simulation Parameters: tau_sim, T_d = 0)
        # ---------------------------------------------------------------------
        tau_sim = axis['tau_sim']
        _, y_sim_base_tr = simulate_fopdt_buffered(t_full, u_full, train_mask, tau_sim, 0.0, K=axis['K'], y_init=y_full[0])
        _, y_sim_base_va = simulate_fopdt_buffered(t_full, u_full, val_mask, tau_sim, 0.0, K=axis['K'], y_init=y_full[0])
        y_sim_base_full, _ = simulate_fopdt_buffered(t_full, u_full, np.ones(n_pts, dtype=bool), tau_sim, 0.0, K=axis['K'], y_init=y_full[0])

        base_metrics_tr = calculate_metrics(y_train, y_sim_base_tr)
        base_metrics_va = calculate_metrics(y_val, y_sim_base_va)

        # ---------------------------------------------------------------------
        # Step 2: Optimize Parameters (Grid-search + Local Optimization)
        # ---------------------------------------------------------------------
        prof = compute_delay_profile(
            t_full, u_full, y_full, train_mask,
            bounds_tau=axis['b_tau'], td_grid=td_grid_scan, K=axis['K']
        )
        tau_opt = prof['best_tau']
        td_opt = prof['best_td']

        # Fine-tune local optimization around the best grid point
        def local_obj(p):
            tau_v, td_v = p
            _, y_pred = simulate_fopdt_buffered(t_full, u_full, train_mask, tau_v, td_v, K=axis['K'], y_init=y_full[0])
            return float(np.mean((y_train - y_pred) ** 2))
        
        res_loc = minimize(
            local_obj, x0=[tau_opt, td_opt],
            bounds=[axis['b_tau'], (0.0, 0.50)], method='L-BFGS-B'
        )
        tau_opt, td_opt = float(res_loc.x[0]), float(res_loc.x[1])

        # Evaluate Fitted Model on Train & Val (Preserving input history)
        _, y_fit_tr = simulate_fopdt_buffered(t_full, u_full, train_mask, tau_opt, td_opt, K=axis['K'], y_init=y_full[0])
        _, y_fit_va = simulate_fopdt_buffered(t_full, u_full, val_mask, tau_opt, td_opt, K=axis['K'], y_init=y_full[0])
        y_fit_full, _ = simulate_fopdt_buffered(t_full, u_full, np.ones(n_pts, dtype=bool), tau_opt, td_opt, K=axis['K'], y_init=y_full[0])

        fit_metrics_tr = calculate_metrics(y_train, y_fit_tr)
        fit_metrics_va = calculate_metrics(y_val, y_fit_va)

        # ---------------------------------------------------------------------
        # Step 4: Cross-Flight Validation (Flight 118 and Flight 135)
        # ---------------------------------------------------------------------
        # A. Flight 118
        t_118, u_118, y_118, _ = prepare_native_signal(
            axis['df_118'], axis['tc'], axis['uc'], axis['yc'], axis['t_118'][0], axis['t_118'][1]
        )
        mask_118 = np.ones(len(t_118), dtype=bool)
        _, y_base_118 = simulate_fopdt_buffered(t_118, u_118, mask_118, tau_sim, 0.0, K=axis['K'], y_init=y_118[0])
        _, y_fit_118 = simulate_fopdt_buffered(t_118, u_118, mask_118, tau_opt, td_opt, K=axis['K'], y_init=y_118[0])
        metrics_118_base = calculate_metrics(y_118, y_base_118)
        metrics_118_fit = calculate_metrics(y_118, y_fit_118)

        # B. Flight 135
        t_135, u_135, y_135, _ = prepare_native_signal(
            axis['df_135'], axis['tc'], axis['uc'], axis['yc'], axis['t_135'][0], axis['t_135'][1]
        )
        mask_135 = np.ones(len(t_135), dtype=bool)
        _, y_base_135 = simulate_fopdt_buffered(t_135, u_135, mask_135, tau_sim, 0.0, K=axis['K'], y_init=y_135[0])
        _, y_fit_135 = simulate_fopdt_buffered(t_135, u_135, mask_135, tau_opt, td_opt, K=axis['K'], y_init=y_135[0])
        metrics_135_base = calculate_metrics(y_135, y_base_135)
        metrics_135_fit = calculate_metrics(y_135, y_fit_135)

        # Save results for comparison table
        report_rows.append({
            'name': axis['name'],
            'unit': axis['unit'],
            # Baseline parameters
            'tau_sim': tau_sim,
            'base_tr_rmse': base_metrics_tr['rmse'],
            'base_tr_vaf': base_metrics_tr['vaf'],
            'base_va_rmse': base_metrics_va['rmse'],
            'base_va_vaf': base_metrics_va['vaf'],
            'base_118_vaf': metrics_118_base['vaf'],
            'base_135_vaf': metrics_135_base['vaf'],
            # Fitted parameters
            'tau_opt': tau_opt,
            'td_opt': td_opt,
            'fit_tr_rmse': fit_metrics_tr['rmse'],
            'fit_tr_vaf': fit_metrics_tr['vaf'],
            'fit_va_rmse': fit_metrics_va['rmse'],
            'fit_va_vaf': fit_metrics_va['vaf'],
            'fit_118_vaf': metrics_118_fit['vaf'],
            'fit_135_vaf': metrics_135_fit['vaf'],
            # Identifiability
            'delta_td': prof['delta_td'],
            'valley_min': prof['valley_td_min'],
            'valley_max': prof['valley_td_max'],
            'is_separable': prof['is_separable'],
        })

        profile_results.append({
            'axis': axis,
            'prof': prof,
            'tau_sim': tau_sim,
        })

        comparison_plot_data.append({
            'axis': axis,
            't': t_full - t_start_120,
            'u': u_full,
            'y': y_full,
            'y_base': y_sim_base_full,
            'y_fit': y_fit_full,
            't_split': (t_full[split_idx] - t_start_120),
            'tau_sim': tau_sim,
            'tau_opt': tau_opt,
            'td_opt': td_opt,
            'base_va_vaf': base_metrics_va['vaf'],
            'fit_va_vaf': fit_metrics_va['vaf'],
        })

    # -------------------------------------------------------------------------
    # Generate Output Plots in logs/flight_logs/plots/
    # -------------------------------------------------------------------------
    plot_dir = Path("logs/flight_logs/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    report_fig_dir = Path("docs/reports/figures")
    report_fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: 6-Panel Delay Identifiability Profiles
    fig_prof, axes_prof = plt.subplots(3, 2, figsize=(15, 12))
    axes_prof = axes_prof.flatten()

    for idx, pitem in enumerate(profile_results):
        ax = axes_prof[idx]
        prof = pitem['prof']
        ax2 = ax.twinx()
        
        tds_ms = prof['td_grid'] * 1000.0
        rmse = prof['rmse_profile']
        taus_ms = prof['tau_profile'] * 1000.0
        
        # Plot RMSE vs Delay
        l1 = ax.plot(tds_ms, rmse, 'C0-', linewidth=2.2, label="RMSE*(T_d)")
        # Plot Tau vs Delay (tradeoff)
        l2 = ax2.plot(tds_ms, taus_ms, 'C1--', linewidth=1.8, label="Optimal tau*(T_d)")
        
        # Mark minimum
        best_td_ms = prof['best_td'] * 1000.0
        best_tau_ms = prof['best_tau'] * 1000.0
        ax.scatter([best_td_ms], [prof['best_rmse']], color='red', s=80, zorder=5, label=f"Optimum (Td={best_td_ms:.0f}ms)")
        
        # Highlight 2% flat uncertainty valley
        ax.axvspan(prof['valley_td_min'] * 1000.0, prof['valley_td_max'] * 1000.0, color='gray', alpha=0.18,
                   label=f"Uncertainty Band (ΔTd={prof['delta_td']*1000:.0f}ms)")
        
        # Mark original simulation tau
        sim_tau_ms = pitem['tau_sim'] * 1000.0
        ax2.axhline(sim_tau_ms, color='green', linestyle=':', linewidth=1.5, label=f"Sim Baseline tau={sim_tau_ms:.0f}ms")

        ax.set_title(
            f"{pitem['axis']['name']}: Delay Identifiability Profile\n"
            f"Optimum Td={best_td_ms:.0f}ms, tau={best_tau_ms:.0f}ms | Uncertainty ΔTd={prof['delta_td']*1000:.0f}ms "
            f"({'Coupled' if not prof['is_separable'] else 'Identifiable'})",
            fontsize=10, fontweight='bold'
        )
        ax.set_xlabel("Fixed Delay T_d (ms)", fontsize=9)
        ax.set_ylabel(f"RMSE ({pitem['axis']['unit']})", color='C0', fontsize=9)
        ax2.set_ylabel("Optimal tau (ms)", color='C1', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', fontsize=8, framealpha=0.85)

    plt.tight_layout()
    prof_out_path = plot_dir / "flight_120_delay_identifiability_profiles.png"
    fig_prof.savefig(prof_out_path, dpi=180)
    fig_prof.savefig(report_fig_dir / "flight_120_delay_identifiability_profiles.png", dpi=180)
    plt.close(fig_prof)
    print(f"\nSaved Delay Identifiability Profiles to: {prof_out_path}")

    # Figure 2: Time-Domain Overlay Comparison (Measured vs Baseline Sim vs Fitted Model)
    fig_cmp, axes_cmp = plt.subplots(6, 1, figsize=(15, 14), sharex=True)
    for idx, citem in enumerate(comparison_plot_data):
        ax = axes_cmp[idx]
        ax.plot(citem['t'], citem['u'], 'k--', linewidth=1.2, alpha=0.7, label="ArduPilot Target")
        ax.plot(citem['t'], citem['y'], 'C0-', linewidth=1.8, label="Recorded Flight Response")
        ax.plot(citem['t'], citem['y_base'], 'C2-.', linewidth=1.8, label=f"Baseline Sim (tau={citem['tau_sim']*1000:.0f}ms, Td=0ms) [Val VAF: {citem['base_va_vaf']:.1f}%]")
        ax.plot(citem['t'], citem['y_fit'], 'C3:', linewidth=2.2, label=f"Fitted Model (tau={citem['tau_opt']*1000:.0f}ms, Td={citem['td_opt']*1000:.0f}ms) [Val VAF: {citem['fit_va_vaf']:.1f}%]")

        # Mark train/val split
        ax.axvline(citem['t_split'], color='dimgray', linestyle='--', linewidth=1.2)
        if idx == 0:
            ax.text(citem['t_split'] * 0.5, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] != 0 else 0.5,
                    "TRAIN FIT (70%)", color='dimgray', fontsize=9, fontweight='bold', ha='center')
            ax.text(citem['t_split'] + (citem['t'][-1] - citem['t_split']) * 0.5,
                    ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] != 0 else 0.5,
                    "OUT-OF-SAMPLE VAL (30%)", color='indigo', fontsize=9, fontweight='bold', ha='center')

        ax.set_title(f"{citem['axis']['name']}: Baseline Simulation vs Fitted FOPDT Response", fontsize=10, fontweight='bold')
        ax.set_ylabel(f"{citem['axis']['key']} ({citem['axis']['unit']})", fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    axes_cmp[-1].set_xlabel("Time in Window (seconds relative to t=188.0s)", fontsize=10)
    plt.tight_layout()
    cmp_out_path = plot_dir / "flight_120_baseline_vs_fitted_comparison.png"
    fig_cmp.savefig(cmp_out_path, dpi=180)
    fig_cmp.savefig(report_fig_dir / "flight_120_baseline_vs_fitted_comparison.png", dpi=180)
    plt.close(fig_cmp)
    print(f"Saved Time-Domain Comparison to: {cmp_out_path}")

    # -------------------------------------------------------------------------
    # Print Synthesis Comparison Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("                    FLIGHT 120: BASELINE SIMULATION VS. FITTED MODEL COMPARISON")
    print("=" * 110)
    header = (
        f"{'CHANNEL':20s} | {'SIM BASELINE':18s} | {'VAL VAF':8s} | "
        f"{'FITTED MODEL':22s} | {'VAL VAF':8s} | {'DELTA VAF':9s} | {'UNCERTAINTY ΔTd':15s} | STATUS"
    )
    print(header)
    print("-" * 110)

    for r in report_rows:
        base_desc = f"tau={r['tau_sim']*1000:3.0f}ms, Td=0ms"
        fit_desc = f"tau={r['tau_opt']*1000:3.0f}ms, Td={r['td_opt']*1000:3.0f}ms"
        delta_vaf = r['fit_va_vaf'] - r['base_va_vaf']
        uncert_str = f"[{r['valley_min']*1000:.0f}, {r['valley_max']*1000:.0f}] ms (Δ={r['delta_td']*1000:.0f}ms)"
        sep_str = "Identifiable" if r['is_separable'] else "Coupled/Flat"
        
        print(
            f"{r['name']:20s} | {base_desc:18s} | {r['base_va_vaf']:7.1f}% | "
            f"{fit_desc:22s} | {r['fit_va_vaf']:7.1f}% | {delta_vaf:+8.1f}% | {uncert_str:15s} | {sep_str}"
        )

    print("=" * 110)

    # Return structured report data for markdown artifact generation
    return report_rows, profile_results


if __name__ == "__main__":
    main()
