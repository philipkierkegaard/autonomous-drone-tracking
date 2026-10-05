"""
Comprehensive 4-Axis System Identification and Latency Analysis Script.
Analyzes ArduPilot .BIN flight logs across:
  1. Pitch Attitude (deg)
  2. Roll Attitude (deg)
  3. Yaw Rate (deg/s)
  4. Vertical Climb / Descent Velocity (m/s)

Extracts empirical time constants (tau) and actuation delay (T_d).
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pymavlink import mavutil

def load_flight_log(log_path: str):
    print(f"Reading log: {log_path}...")
    mlog = mavutil.mavlink_connection(log_path)
    
    att_data = []
    rate_data = []
    ctun_data = []
    
    while True:
        m = mlog.recv_msg()
        if m is None:
            break
        mtype = m.get_type()
        t_us = getattr(m, 'TimeUS', None)
        if t_us is None:
            continue
        t_sec = t_us / 1e6
        
        if mtype == 'ATT':
            d = m.to_dict()
            att_data.append({
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
                'r_des': d.get('RDes', 0.0),
                'r_act': d.get('R', 0.0)
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
            
    df_att = pd.DataFrame(att_data)
    df_rate = pd.DataFrame(rate_data)
    df_ctun = pd.DataFrame(ctun_data)
    return df_att, df_rate, df_ctun


def simulate_fopdt(t, u_target, y_init, tau, T_d):
    dt = np.mean(np.diff(t))
    shift_steps = int(round(T_d / dt))
    if shift_steps > 0:
        u_delayed = np.zeros_like(u_target)
        u_delayed[shift_steps:] = u_target[:-shift_steps]
        u_delayed[:shift_steps] = u_target[0]
    else:
        u_delayed = u_target
        
    y_sim = np.zeros_like(u_target)
    y_sim[0] = y_init
    for i in range(len(t) - 1):
        dy = (u_delayed[i] - y_sim[i]) / max(tau, 0.005)
        y_sim[i+1] = y_sim[i] + dy * dt
    return y_sim


def fit_first_order_delay(t, u_target, y_actual, bounds=((0.01, 0.60), (0.00, 0.15))):
    def cost(params):
        tau, T_d = params
        y_sim = simulate_fopdt(t, u_target, y_actual[0], tau, T_d)
        return np.mean((y_sim - y_actual) ** 2)

    res = minimize(cost, x0=[0.05, 0.04], bounds=bounds, method='L-BFGS-B')
    best_tau, best_td = res.x
    y_sim = simulate_fopdt(t, u_target, y_actual[0], best_tau, best_td)
    rmse = np.sqrt(res.fun)
    return best_tau, best_td, y_sim, rmse


def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    var_true = np.var(y_true)
    vaf = (1.0 - np.var(y_true - y_pred) / (var_true + 1e-8)) * 100.0
    return rmse, vaf


def main():
    log_file = "logs/flight_logs/flight_00000073.BIN"
    df_att, df_rate, df_ctun = load_flight_log(log_file)
    
    # Filter armed flight period (127s to 434s)
    mask_att = (df_att['time'] >= 127.0) & (df_att['time'] <= 434.0)
    df_att = df_att[mask_att].reset_index(drop=True)
    
    mask_rate = (df_rate['time'] >= 127.0) & (df_rate['time'] <= 434.0)
    df_rate = df_rate[mask_rate].reset_index(drop=True)
    
    mask_ctun = (df_ctun['time'] >= 127.0) & (df_ctun['time'] <= 434.0)
    df_ctun = df_ctun[mask_ctun].reset_index(drop=True)
    
    # Uniform 50 Hz interpolation
    t_min = max(df_att['time'].min(), df_rate['time'].min(), df_ctun['time'].min())
    t_max = min(df_att['time'].max(), df_rate['time'].max(), df_ctun['time'].max())
    t_uniform = np.arange(t_min, t_max, 0.02)
    
    # 1. Pitch
    u_pitch = np.interp(t_uniform, df_att['time'], df_att['des_pitch'])
    y_pitch = np.interp(t_uniform, df_att['time'], df_att['pitch'])
    tau_p, td_p, y_sim_p, rmse_p = fit_first_order_delay(t_uniform, u_pitch, y_pitch)
    _, vaf_p = calculate_metrics(y_pitch, y_sim_p)

    # 2. Roll
    u_roll = np.interp(t_uniform, df_att['time'], df_att['des_roll'])
    y_roll = np.interp(t_uniform, df_att['time'], df_att['roll'])
    tau_r, td_r, y_sim_r, rmse_r = fit_first_order_delay(t_uniform, u_roll, y_roll)
    _, vaf_r = calculate_metrics(y_roll, y_sim_r)

    # 3. Yaw Rate
    u_yaw = np.interp(t_uniform, df_rate['time'], df_rate['r_des'])
    y_yaw = np.interp(t_uniform, df_rate['time'], df_rate['r_act'])
    tau_y, td_y, y_sim_y, rmse_y = fit_first_order_delay(t_uniform, u_yaw, y_yaw)
    _, vaf_y = calculate_metrics(y_yaw, y_sim_y)

    # 4. Vertical Climb / Descent Rate (Vz)
    u_vz = np.interp(t_uniform, df_ctun['time'], df_ctun['vz_des'])
    y_vz = np.interp(t_uniform, df_ctun['time'], df_ctun['vz_act'])
    tau_z, td_z, y_sim_z, rmse_z = fit_first_order_delay(t_uniform, u_vz, y_vz, bounds=((0.05, 0.80), (0.00, 0.15)))
    _, vaf_z = calculate_metrics(y_vz, y_sim_z)

    print("\n" + "="*70)
    print("        COMPLETE 4-AXIS SYSTEM IDENTIFICATION (FLIGHT 73)")
    print("="*70)
    print(f"1. PITCH:      tau = {tau_p*1000:5.1f} ms, T_d = {td_p*1000:4.1f} ms | RMSE = {rmse_p:5.2f}°    | VAF = {vaf_p:5.1f}%")
    print(f"2. ROLL:       tau = {tau_r*1000:5.1f} ms, T_d = {td_r*1000:4.1f} ms | RMSE = {rmse_r:5.2f}°    | VAF = {vaf_r:5.1f}%")
    print(f"3. YAW RATE:   tau = {tau_y*1000:5.1f} ms, T_d = {td_y*1000:4.1f} ms | RMSE = {rmse_y:5.2f}°/s  | VAF = {vaf_y:5.1f}%")
    print(f"4. CLIMB (Vz): tau = {tau_z*1000:5.1f} ms, T_d = {td_z*1000:4.1f} ms | RMSE = {rmse_z:5.2f} m/s  | VAF = {vaf_z:5.1f}%")
    print("="*70)

    # Generate Comprehensive 4-Subplot Figure (Active 20-second Window with all 4 axes active)
    win_start = 138.0
    win_mask = (t_uniform >= win_start) & (t_uniform < win_start + 20.0)
    tw = t_uniform[win_mask] - win_start

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

    # 1. Pitch
    axes[0].plot(tw, u_pitch[win_mask], 'k--', label='Desired Pitch (Stick)', linewidth=1.6)
    axes[0].plot(tw, y_pitch[win_mask], 'C0-', label='Actual Pitch (EKF)', linewidth=2.0)
    axes[0].plot(tw, y_sim_p[win_mask], 'C3:', label=f'Model Fit (tau={tau_p*1000:.0f}ms, T_d={td_p*1000:.0f}ms)', linewidth=2.2)
    axes[0].set_title(f"1. Pitch Dynamics (tau = {tau_p*1000:.1f} ms, Delay T_d = {td_p*1000:.1f} ms | RMSE = {rmse_p:.2f}°)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Pitch (deg)", fontsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper right', fontsize=9)

    # 2. Roll
    axes[1].plot(tw, u_roll[win_mask], 'k--', label='Desired Roll (Stick)', linewidth=1.6)
    axes[1].plot(tw, y_roll[win_mask], 'C1-', label='Actual Roll (EKF)', linewidth=2.0)
    axes[1].plot(tw, y_sim_r[win_mask], 'C3:', label=f'Model Fit (tau={tau_r*1000:.0f}ms, T_d={td_r*1000:.0f}ms)', linewidth=2.2)
    axes[1].set_title(f"2. Roll Dynamics (tau = {tau_r*1000:.1f} ms, Delay T_d = {td_r*1000:.1f} ms | RMSE = {rmse_r:.2f}°)", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Roll (deg)", fontsize=10)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='upper right', fontsize=9)

    # 3. Yaw Rate
    axes[2].plot(tw, u_yaw[win_mask], 'k--', label='Desired Yaw Rate (Stick)', linewidth=1.6)
    axes[2].plot(tw, y_yaw[win_mask], 'C2-', label='Actual Yaw Rate (Gyro)', linewidth=2.0)
    axes[2].plot(tw, y_sim_y[win_mask], 'C3:', label=f'Model Fit (tau={tau_y*1000:.0f}ms, T_d={td_y*1000:.0f}ms)', linewidth=2.2)
    axes[2].set_title(f"3. Yaw Rate Dynamics (tau = {tau_y*1000:.1f} ms, Delay T_d = {td_y*1000:.1f} ms | RMSE = {rmse_y:.2f}°/s)", fontsize=11, fontweight='bold')
    axes[2].set_ylabel("Yaw Rate (°/s)", fontsize=10)
    axes[2].grid(True, linestyle=':', alpha=0.6)
    axes[2].legend(loc='upper right', fontsize=9)

    # 4. Vertical Climb Rate (Vz)
    axes[3].plot(tw, u_vz[win_mask], 'k--', label='Desired Climb Rate (Throttle Stick)', linewidth=1.6)
    axes[3].plot(tw, y_vz[win_mask], 'C4-', label='Actual Climb Rate (Baro/EKF)', linewidth=2.0)
    axes[3].plot(tw, y_sim_z[win_mask], 'C3:', label=f'Model Fit (tau={tau_z*1000:.0f}ms, T_d={td_z*1000:.0f}ms)', linewidth=2.2)
    axes[3].set_title(f"4. Vertical Climb Rate Dynamics (tau = {tau_z*1000:.1f} ms, Delay T_d = {td_z*1000:.1f} ms | RMSE = {rmse_z:.2f} m/s)", fontsize=11, fontweight='bold')
    axes[3].set_xlabel("Time in Window (seconds)", fontsize=10)
    axes[3].set_ylabel("Climb Rate (m/s)", fontsize=10)
    axes[3].grid(True, linestyle=':', alpha=0.6)
    axes[3].legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    out_file = "logs/flight_logs/flight_4axis_sysid_plot.png"
    plt.savefig(out_file, dpi=150)
    print(f"\nSaved 4-Axis SysID plot to: {out_file}")

if __name__ == "__main__":
    main()
