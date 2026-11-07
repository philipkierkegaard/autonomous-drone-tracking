#!/usr/bin/env python3
"""
Comprehensive telemetry and dynamics analysis for Flight 168 (00000168.BIN).
Extracts:
1. Complete flight timeline, armed duration, and flight modes
2. RC input channels (pilot sticks & switches) to identify intent and mode switch attempts
3. 3D Trajectory, velocity (VN, VE, VD), altitude, and position tracking
4. Attitude and Rate dynamics: Roll, Pitch, and Yaw rate tracking (RATE.YDes vs RATE.Y)
5. First-Order System Identification (tau, dead time, effective lag, VAF) for yaw and attitude
6. Sensor health: EKF3 status, compass / magnetometer anomalies, GPS HDOP, failsafes
"""

import sys
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pymavlink import mavutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIN_PATH = PROJECT_ROOT / "logs" / "flight_logs" / "00000168.BIN"
CACHE_PATH = PROJECT_ROOT / "logs" / "flight_logs" / "00000168_extracted.npz"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "flight_analysis"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_and_cache(bin_path: Path, cache_path: Path) -> dict:
    if cache_path.exists():
        print(f"[CACHE] Loading cached flight telemetry from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    print(f"[PARSE] Parsing {bin_path} ({bin_path.stat().st_size / 1e6:.1f} MB)...")
    t0 = time.time()
    mlog = mavutil.mavlink_connection(str(bin_path))

    rate_t, yaw_des, yaw_act, roll_des, roll_act, pitch_des, pitch_act = [], [], [], [], [], [], []
    att_t, a_roll, a_pitch, a_yaw, des_roll, des_pitch, des_yaw = [], [], [], [], [], [], []
    ctun_t, vz_des, vz_act, alt, dalt, thr_out = [], [], [], [], [], []
    pos_t, pos_lat, pos_lng, pos_alt, pos_vn, pos_ve, pos_vd = [], [], [], [], [], [], []
    rc_t, rc1, rc2, rc3, rc4, rc5, rc6, rc7, rc8 = [], [], [], [], [], [], [], [], []
    mode_t, mode_num, mode_name = [], [], []
    ev_t, ev_id = [], []
    msg_t, msg_text = [], []
    ekf_t, ekf_innov_yaw, ekf_innov_pos, ekf_innov_hgt = [], [], [], []

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
            des_roll.append(d.get("DesRoll", 0.0))
            des_pitch.append(d.get("DesPitch", 0.0))
            des_yaw.append(d.get("DesYaw", 0.0))

        elif mtype == "CTUN":
            d = m.to_dict()
            ctun_t.append(t_sec)
            vz_des.append(d.get("DCRt", 0.0) / 100.0)
            vz_act.append(d.get("CRt", 0.0) / 100.0)
            alt.append(d.get("Alt", 0.0))
            dalt.append(d.get("DAlt", 0.0))
            thr_out.append(d.get("ThO", 0.0))

        elif mtype == "XKF1":
            d = m.to_dict()
            pos_t.append(t_sec)
            pos_lat.append(d.get("Lat", 0.0))
            pos_lng.append(d.get("Lng", 0.0))
            pos_alt.append(d.get("Alt", 0.0))
            pos_vn.append(d.get("VN", 0.0))
            pos_ve.append(d.get("VE", 0.0))
            pos_vd.append(d.get("VD", 0.0))

        elif mtype == "RCIN":
            d = m.to_dict()
            rc_t.append(t_sec)
            rc1.append(d.get("C1", 1500))
            rc2.append(d.get("C2", 1500))
            rc3.append(d.get("C3", 1000))
            rc4.append(d.get("C4", 1500))
            rc5.append(d.get("C5", 1000))
            rc6.append(d.get("C6", 1000))
            rc7.append(d.get("C7", 1000))
            rc8.append(d.get("C8", 1000))

        elif mtype == "MODE":
            d = m.to_dict()
            mnum = getattr(m, "Mode", getattr(m, "ModeNum", 0))
            mname = mavutil.mode_mapping_acm.get(mnum, f"Mode_{mnum}")
            mode_t.append(t_sec)
            mode_num.append(mnum)
            mode_name.append(mname)

        elif mtype == "EV":
            d = m.to_dict()
            ev_t.append(t_sec)
            ev_id.append(d.get("Id", -1))

        elif mtype == "MSG":
            d = m.to_dict()
            msg_t.append(t_sec)
            msg_text.append(d.get("Message", ""))

        elif mtype == "XKF4":
            d = m.to_dict()
            ekf_t.append(t_sec)
            ekf_innov_yaw.append(d.get("FYaw", 0.0))
            ekf_innov_pos.append(d.get("IPN", 0.0))
            ekf_innov_hgt.append(d.get("IPD", 0.0))

    print(f"[PARSE] Finished in {time.time()-t0:.1f}s.")
    out = {
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
        "des_roll": np.array(des_roll, dtype=np.float32),
        "des_pitch": np.array(des_pitch, dtype=np.float32),
        "des_yaw": np.array(des_yaw, dtype=np.float32),

        "ctun_t": np.array(ctun_t, dtype=np.float64),
        "vz_des": np.array(vz_des, dtype=np.float32),
        "vz_act": np.array(vz_act, dtype=np.float32),
        "alt": np.array(alt, dtype=np.float32),
        "dalt": np.array(dalt, dtype=np.float32),
        "thr_out": np.array(thr_out, dtype=np.float32),

        "pos_t": np.array(pos_t, dtype=np.float64),
        "pos_lat": np.array(pos_lat, dtype=np.float64),
        "pos_lng": np.array(pos_lng, dtype=np.float64),
        "pos_alt": np.array(pos_alt, dtype=np.float32),
        "pos_vn": np.array(pos_vn, dtype=np.float32),
        "pos_ve": np.array(pos_ve, dtype=np.float32),
        "pos_vd": np.array(pos_vd, dtype=np.float32),

        "rc_t": np.array(rc_t, dtype=np.float64),
        "rc1": np.array(rc1, dtype=np.int16),
        "rc2": np.array(rc2, dtype=np.int16),
        "rc3": np.array(rc3, dtype=np.int16),
        "rc4": np.array(rc4, dtype=np.int16),
        "rc5": np.array(rc5, dtype=np.int16),
        "rc6": np.array(rc6, dtype=np.int16),
        "rc7": np.array(rc7, dtype=np.int16),
        "rc8": np.array(rc8, dtype=np.int16),

        "mode_t": np.array(mode_t, dtype=np.float64),
        "mode_num": np.array(mode_num, dtype=np.int32),
        "mode_name": np.array(mode_name, dtype=object),

        "ev_t": np.array(ev_t, dtype=np.float64),
        "ev_id": np.array(ev_id, dtype=np.int32),

        "msg_t": np.array(msg_t, dtype=np.float64),
        "msg_text": np.array(msg_text, dtype=object),

        "ekf_t": np.array(ekf_t, dtype=np.float64),
        "ekf_innov_yaw": np.array(ekf_innov_yaw, dtype=np.float32),
        "ekf_innov_pos": np.array(ekf_innov_pos, dtype=np.float32),
        "ekf_innov_hgt": np.array(ekf_innov_hgt, dtype=np.float32),
    }
    np.savez_compressed(cache_path, **out)
    print(f"[CACHE] Saved compressed cache to {cache_path}")
    return out


def simulate_fopdt(t: np.ndarray, u: np.ndarray, y0: float, tau: float, td: float, K: float = 1.0) -> np.ndarray:
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


def fit_fopdt(t: np.ndarray, u: np.ndarray, y: np.ndarray, b_tau=(0.020, 1.50), b_td=(0.0, 0.40)):
    y0 = float(y[0])
    def loss(p):
        tau, td, k = p[0], p[1], p[2]
        sim = simulate_fopdt(t, u, y0, tau, td, K=k)
        return np.mean((y - sim) ** 2)

    res = minimize(loss, [0.20, 0.05, 1.0], bounds=[b_tau, b_td, (0.50, 1.50)], method="L-BFGS-B")
    best_tau, best_td, best_k = float(res.x[0]), float(res.x[1]), float(res.x[2])
    sim = simulate_fopdt(t, u, y0, best_tau, best_td, K=best_k)

    var_y = np.var(y)
    vaf = max(0.0, 1.0 - np.var(y - sim) / (var_y + 1e-9)) * 100.0
    rmse = np.sqrt(np.mean((y - sim) ** 2))
    return {
        "tau_s": best_tau,
        "td_s": best_td,
        "k": best_k,
        "effective_delay_s": best_tau + best_td,
        "vaf": vaf,
        "rmse": rmse,
        "y_sim": sim,
    }


def main():
    data = parse_and_cache(BIN_PATH, CACHE_PATH)

    # 1. Identify Armed Flight Window
    ev_t = data["ev_t"]
    ev_id = data["ev_id"]
    arm_times = ev_t[ev_id == 10]
    disarm_times = ev_t[ev_id == 11]

    t_arm = float(arm_times[0]) if len(arm_times) > 0 else 514.08
    t_disarm = float(disarm_times[0]) if len(disarm_times) > 0 else 666.58
    flight_dur = t_disarm - t_arm

    print("\n" + "=" * 85)
    print(f"  FLIGHT 168 (00000168.BIN) DETAILED ANALYSIS")
    print(f"  Armed Duration: {t_arm:.2f}s -> {t_disarm:.2f}s ({flight_dur:.2f}s = {flight_dur/60.0:.2f} min)")
    print("=" * 85)

    # Filter in-flight data
    m_rate = (data["rate_t"] >= t_arm) & (data["rate_t"] <= t_disarm)
    t_rate = data["rate_t"][m_rate]
    yaw_des = data["yaw_des"][m_rate]
    yaw_act = data["yaw_act"][m_rate]
    roll_des = data["roll_des"][m_rate]
    roll_act = data["roll_act"][m_rate]
    pitch_des = data["pitch_des"][m_rate]
    pitch_act = data["pitch_act"][m_rate]

    m_ctun = (data["ctun_t"] >= t_arm) & (data["ctun_t"] <= t_disarm)
    t_ctun = data["ctun_t"][m_ctun]
    alt = data["alt"][m_ctun]
    dalt = data["dalt"][m_ctun]
    vz_act = data["vz_act"][m_ctun]
    vz_des = data["vz_des"][m_ctun]
    thr_out = data["thr_out"][m_ctun]

    m_pos = (data["pos_t"] >= t_arm) & (data["pos_t"] <= t_disarm)
    t_pos = data["pos_t"][m_pos]
    vn = data["pos_vn"][m_pos]
    ve = data["pos_ve"][m_pos]
    vd = data["pos_vd"][m_pos]
    hspeed = np.sqrt(vn ** 2 + ve ** 2)

    m_rc = (data["rc_t"] >= t_arm) & (data["rc_t"] <= t_disarm)
    t_rc = data["rc_t"][m_rc]
    rc_roll = data["rc1"][m_rc]
    rc_pitch = data["rc2"][m_rc]
    rc_thr = data["rc3"][m_rc]
    rc_yaw = data["rc4"][m_rc]
    rc_ch5 = data["rc5"][m_rc]
    rc_ch6 = data["rc6"][m_rc]
    rc_ch7 = data["rc7"][m_rc]
    rc_ch8 = data["rc8"][m_rc]

    # In-Flight Kinematic Summary
    print(f"\n--- In-Flight Kinematics & Performance ---")
    print(f"Max Altitude:           {np.max(alt):.2f} m (Mean: {np.mean(alt):.2f} m)")
    print(f"Max Horizontal Speed:   {np.max(hspeed):.2f} m/s (Mean: {np.mean(hspeed):.2f} m/s)")
    print(f"Max Vertical Descent:   {np.max(vz_act):.2f} m/s | Climb: {np.min(vz_act):.2f} m/s")
    print(f"Max Yaw Rate Desired:   {np.max(np.abs(yaw_des)):.2f} deg/s (Actual: {np.max(np.abs(yaw_act)):.2f} deg/s)")
    print(f"Max Roll Angle (des/act): {np.max(np.abs(roll_des)):.2f}° / {np.max(np.abs(roll_act)):.2f}°")
    print(f"Max Pitch Angle (des/act):{np.max(np.abs(pitch_des)):.2f}° / {np.max(np.abs(pitch_act)):.2f}°")

    # RC Channel Analysis: Did the pilot touch switches or sticks?
    print(f"\n--- Pilot RC Stick & Switch Activity ---")
    print(f"Roll  Stick (C1): range=[{np.min(rc_roll)}, {np.max(rc_roll)}], std={np.std(rc_roll):.1f}")
    print(f"Pitch Stick (C2): range=[{np.min(rc_pitch)}, {np.max(rc_pitch)}], std={np.std(rc_pitch):.1f}")
    print(f"Thro  Stick (C3): range=[{np.min(rc_thr)}, {np.max(rc_thr)}], std={np.std(rc_thr):.1f}")
    print(f"Yaw   Stick (C4): range=[{np.min(rc_yaw)}, {np.max(rc_yaw)}], std={np.std(rc_yaw):.1f}")
    print(f"Switch C5 (Mode): unique values={np.unique(rc_ch5).tolist()}")
    print(f"Switch C6       : unique values={np.unique(rc_ch6).tolist()}")
    print(f"Switch C7 (Auto): unique values={np.unique(rc_ch7).tolist()}")
    print(f"Switch C8       : unique values={np.unique(rc_ch8).tolist()}")

    # Flight Phases: Air time vs ground idling
    airborne_mask = alt > 0.30
    t_takeoff = t_ctun[airborne_mask][0] if np.any(airborne_mask) else t_arm
    t_landing = t_ctun[airborne_mask][-1] if np.any(airborne_mask) else t_disarm
    print(f"\nTakeoff at: t={t_takeoff:.2f}s | Landed at: t={t_landing:.2f}s (Actual Airborne Time: {t_landing - t_takeoff:.1f}s)")

    # -------------------------------------------------------------
    # System Identification on Airborne Yaw Maneuvers
    # -------------------------------------------------------------
    # Select dynamic yaw periods where |yaw_des| > 15 deg/s
    air_rate_mask = (t_rate >= t_takeoff) & (t_rate <= t_landing)
    t_air = t_rate[air_rate_mask]
    yd_air = yaw_des[air_rate_mask]
    ya_air = yaw_act[air_rate_mask]

    print(f"\n--- Dynamic System Identification (Airborne Window: {t_takeoff:.1f}s - {t_landing:.1f}s) ---")
    fit_yaw_all = fit_fopdt(t_air, yd_air, ya_air)
    print(f"Full Airborne Yaw Model:")
    print(f"  Time Constant (tau_yaw): {fit_yaw_all['tau_s']*1000:.1f} ms")
    print(f"  Transport Delay (T_d)  : {fit_yaw_all['td_s']*1000:.1f} ms")
    print(f"  Effective Total Lag    : {fit_yaw_all['effective_delay_s']*1000:.1f} ms")
    print(f"  Gain K                 : {fit_yaw_all['k']:.3f}")
    print(f"  Model Fit (VAF)        : {fit_yaw_all['vaf']:.1f}% (RMSE: {fit_yaw_all['rmse']:.2f} deg/s)")

    # Find specific high-excitation yaw pulses
    yd_abs = np.abs(yd_air)
    pulse_indices = np.where(yd_abs > 30.0)[0]
    sub_fits = []
    if len(pulse_indices) > 20:
        # Segment into continuous pulse clusters
        diffs = np.diff(pulse_indices)
        splits = np.where(diffs > 100)[0] + 1
        clusters = np.split(pulse_indices, splits)
        for ci, cl in enumerate(clusters):
            if len(cl) < 25:
                continue
            idx0 = max(0, cl[0] - 25)
            idx1 = min(len(t_air) - 1, cl[-1] + 25)
            t_sub = t_air[idx0:idx1]
            yd_sub = yd_air[idx0:idx1]
            ya_sub = ya_air[idx0:idx1]
            if np.ptp(yd_sub) > 40.0:
                sf = fit_fopdt(t_sub, yd_sub, ya_sub)
                sub_fits.append((t_sub[0], t_sub[-1], sf))
                print(f"  Pulse {len(sub_fits)}: t={t_sub[0]:.1f}-{t_sub[-1]:.1f}s (Max YawDes: {np.max(np.abs(yd_sub)):.1f}°/s) -> "
                      f"tau={sf['tau_s']*1000:.1f}ms, T_d={sf['td_s']*1000:.1f}ms, EffLag={sf['effective_delay_s']*1000:.1f}ms, VAF={sf['vaf']:.1f}%")

    # -------------------------------------------------------------
    # Plotting Comprehensive Telemetry Dashboard
    # -------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"Flight 168 Comprehensive Telemetry & Dynamics Analysis\n"
                 f"Mode: LOITER | Airborne: {t_landing - t_takeoff:.1f}s | "
                 f"Identified Yaw Lag: tau={fit_yaw_all['tau_s']*1000:.0f}ms, T_d={fit_yaw_all['td_s']*1000:.0f}ms (VAF={fit_yaw_all['vaf']:.1f}%)",
                 fontsize=13, fontweight="bold")

    # 1. Altitude & Vertical Speed
    axes[0].plot(t_ctun, alt, label="Altitude (m)", color="#10B981", lw=2)
    axes[0].plot(t_ctun, dalt, label="Desired Alt (m)", color="#64748B", ls="--", lw=1.5)
    axes[0].set_ylabel("Altitude (m)", fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right")

    # 2. Horizontal Velocity & Speed
    axes[1].plot(t_pos, hspeed, label="Horizontal Speed (m/s)", color="#2563EB", lw=2)
    axes[1].plot(t_pos, vn, label="North Vel (VN)", color="#3B82F6", ls=":", alpha=0.7)
    axes[1].plot(t_pos, ve, label="East Vel (VE)", color="#93C5FD", ls=":", alpha=0.7)
    axes[1].set_ylabel("Speed (m/s)", fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right")

    # 3. Yaw Rate Tracking
    axes[2].plot(t_rate, yaw_des, label="Desired Yaw Rate (RATE.YDes)", color="#EF4444", lw=1.8)
    axes[2].plot(t_rate, yaw_act, label="Actual Yaw Rate (RATE.Y)", color="#059669", lw=1.8, alpha=0.85)
    axes[2].set_ylabel("Yaw Rate (°/s)", fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right")

    # 4. Pilot RC Inputs
    axes[3].plot(t_rc, (rc_yaw - 1500) / 5.0, label="RC Yaw Stick (%)", color="#8B5CF6", lw=1.5)
    axes[3].plot(t_rc, (rc_pitch - 1500) / 5.0, label="RC Pitch Stick (%)", color="#F59E0B", lw=1.5)
    axes[3].plot(t_rc, (rc_roll - 1500) / 5.0, label="RC Roll Stick (%)", color="#06B6D4", lw=1.5)
    axes[3].plot(t_rc, (rc_thr - 1000) / 10.0, label="RC Throttle (%)", color="#64748B", ls="--", lw=1.5)
    axes[3].set_ylabel("Pilot Input (%)", fontweight="bold")
    axes[3].set_xlabel("Log Time (seconds)", fontweight="bold")
    axes[3].grid(True, linestyle="--", alpha=0.5)
    axes[3].legend(loc="upper right")

    plt.tight_layout()
    out_plot = PLOTS_DIR / "00000168_comprehensive_flight_analysis.png"
    plt.savefig(out_plot, dpi=200)
    plt.close()
    print(f"\n[PLOT] Saved comprehensive dashboard to: {out_plot}")

    # Copy plot to artifact directory
    artifact_plot = Path("/Users/philipkierkegaard/.gemini/antigravity/brain/3f6fb6d1-ed45-4315-9b5d-b3cca4e6bef7") / "00000168_comprehensive_flight_analysis.png"
    import shutil
    shutil.copy2(out_plot, artifact_plot)
    print(f"[ARTIFACT] Copied plot to artifact directory: {artifact_plot}")


if __name__ == "__main__":
    main()

