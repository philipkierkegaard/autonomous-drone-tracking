#!/usr/bin/env python3
"""
ArduPilot Flight Log Velocity Tracking Plotter (North & East Velocity).
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Parses ArduPilot DataFlash binary log files (.BIN), extracts commanded
(desired) velocity setpoints from the position/speed controller (PSCN, PSCE),
compares them against real EKF estimated velocities, highlights autonomous
GUIDED mode engagement intervals vs manual LOITER flight, and produces
publication-quality plots and tracking error statistics.

Usage:
    # Plot default flight log 00000165.BIN (armed flight window):
    python scripts/plot_flight_velocity_tracking.py

    # Explicit path or shorthand:
    python scripts/plot_flight_velocity_tracking.py --log logs/flight_logs/00000165.BIN
    python scripts/plot_flight_velocity_tracking.py --log 165

    # Zoom strictly into the autonomous GUIDED pursuit engagements:
    python scripts/plot_flight_velocity_tracking.py --window guided

    # Plot the full log duration (including pre-arm and post-landing):
    python scripts/plot_flight_velocity_tracking.py --window all
"""

from __future__ import annotations

import os
import sys
import argparse
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

# Use Agg backend for headless / server / desktop compatibility
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from pymavlink import mavutil
    PYMAVLINK_AVAILABLE = True
except ImportError:
    PYMAVLINK_AVAILABLE = False


# ArduCopter Flight Mode Mapping
ARDUCOPTER_MODES = {
    0: "STABILIZE",
    1: "ACRO",
    2: "ALT_HOLD",
    3: "AUTO",
    4: "GUIDED",
    5: "LOITER",
    6: "RTL",
    7: "CIRCLE",
    9: "LAND",
    11: "DRIFT",
    13: "SPORT",
    14: "FLIP",
    15: "AUTOTUNE",
    16: "POSHOLD",
    17: "BRAKE",
    18: "THROW",
    19: "AVOID_ADSB",
    20: "GUIDED_NOGPS",
    21: "SMART_RTL",
    22: "FLOWHOLD",
    23: "FOLLOW",
    24: "ZIGZAG",
    25: "SYSTEMID",
    26: "AUTOROTATE",
    27: "AUTO_RTL",
}


def resolve_log_path(log_arg: Optional[str]) -> Path:
    """Resolves log path handling shortcuts like '165', '0..0165.BIN', or full paths."""
    if not log_arg:
        # Default to 00000165.BIN if present
        default_p = Path("logs/flight_logs/00000165.BIN")
        if default_p.exists():
            return default_p
        # Otherwise find most recent .BIN in logs/flight_logs
        bins = sorted(glob.glob("logs/flight_logs/*.BIN"), key=os.path.getmtime, reverse=True)
        if bins:
            return Path(bins[0])
        raise FileNotFoundError("No .BIN flight log found in logs/flight_logs/")

    p = Path(log_arg)
    if p.exists():
        return p

    # Check inside logs/flight_logs/
    candidates = [
        Path(f"logs/flight_logs/{log_arg}"),
        Path(f"logs/flight_logs/{log_arg}.BIN"),
        Path(f"logs/flight_logs/00000{log_arg}.BIN"),
    ]
    # Clean up patterns like '0..0165.BIN' -> search for '*165.BIN'
    if "165" in str(log_arg):
        candidates.extend([Path(f) for f in glob.glob("logs/flight_logs/*165.BIN")])

    for c in candidates:
        if c.exists():
            return c

    # Fuzzy match in logs/flight_logs
    matched = glob.glob(f"logs/flight_logs/*{log_arg}*")
    if matched:
        return Path(matched[0])

    raise FileNotFoundError(f"Could not locate flight log: '{log_arg}'")


def parse_flight_log(log_path: Path) -> Dict[str, Any]:
    """Parses binary ArduPilot DataFlash log and extracts velocity and state streams."""
    if not PYMAVLINK_AVAILABLE:
        raise ImportError("pymavlink is required to parse .BIN logs. Run: pip install pymavlink")

    print(f"[PARSER] Opening ArduPilot log: {log_path.name} ({log_path.stat().st_size / (1024*1024):.1f} MB)...")
    mlog = mavutil.mavlink_connection(str(log_path))

    # Output storage
    pscn_t, pscn_dvn, pscn_tvn, pscn_vn = [], [], [], []
    psce_t, psce_dve, psce_tve, psce_ve = [], [], [], []
    mode_changes = []  # (t_sec, mode_name)
    arm_events = []    # (t_sec, 'ARM' or 'DISARM')

    first_t = None
    last_t = None

    while True:
        msg = mlog.recv_msg()
        if msg is None:
            break

        mtype = msg.get_type()
        tus = getattr(msg, "TimeUS", None)
        if tus is not None:
            t_sec = tus / 1e6
            if first_t is None:
                first_t = t_sec
            last_t = t_sec

            if mtype == "MODE":
                m_num = getattr(msg, "Mode", None)
                if m_num is None:
                    m_num = getattr(msg, "ModeNum", 0)
                m_name = ARDUCOPTER_MODES.get(m_num, f"MODE_{m_num}")
                mode_changes.append((t_sec, m_name))

            elif mtype == "EV":
                ev_id = getattr(msg, "Id", None)
                if ev_id == 10:
                    arm_events.append((t_sec, "ARM"))
                elif ev_id == 11:
                    arm_events.append((t_sec, "DISARM"))

            elif mtype == "PSCN":
                pscn_t.append(t_sec)
                pscn_dvn.append(getattr(msg, "DVN", 0.0))
                pscn_tvn.append(getattr(msg, "TVN", 0.0))
                pscn_vn.append(getattr(msg, "VN", 0.0))

            elif mtype == "PSCE":
                psce_t.append(t_sec)
                psce_dve.append(getattr(msg, "DVE", 0.0))
                psce_tve.append(getattr(msg, "TVE", 0.0))
                psce_ve.append(getattr(msg, "VE", 0.0))

    data = {
        "log_name": log_path.name,
        "first_t": first_t or 0.0,
        "last_t": last_t or 0.0,
        "total_duration_s": (last_t - first_t) if (last_t and first_t) else 0.0,
        "modes": mode_changes,
        "arms": arm_events,
        "north": {
            "t": np.array(pscn_t),
            "des": np.array(pscn_dvn),
            "target": np.array(pscn_tvn),
            "est": np.array(pscn_vn),
        },
        "east": {
            "t": np.array(psce_t),
            "des": np.array(psce_dve),
            "target": np.array(psce_tve),
            "est": np.array(psce_ve),
        },
    }

    print(f"[PARSER] ✓ Extracted {len(pscn_t):,} North and {len(psce_t):,} East velocity telemetry points.")
    return data


def compute_mode_intervals(data: Dict[str, Any], t_min: float, t_max: float) -> List[Tuple[float, float, str]]:
    """Builds contiguous time intervals for each flight mode inside [t_min, t_max]."""
    raw_modes = data["modes"]
    if not raw_modes:
        return [(t_min, t_max, "UNKNOWN")]

    intervals = []
    # Sort by time
    sorted_modes = sorted(raw_modes, key=lambda x: x[0])

    for i, (t_start, m_name) in enumerate(sorted_modes):
        t_end = sorted_modes[i + 1][0] if i + 1 < len(sorted_modes) else data["last_t"]
        # Clip to window
        seg_start = max(t_min, t_start)
        seg_end = min(t_max, t_end)
        if seg_end > seg_start:
            intervals.append((seg_start, seg_end, m_name))

    return intervals


def compute_tracking_metrics(t_arr: np.ndarray, des: np.ndarray, est: np.ndarray, t_start: float, t_end: float) -> Dict[str, float]:
    """Computes RMSE, Pearson correlation, and peak velocities in a time window."""
    mask = (t_arr >= t_start) & (t_arr <= t_end)
    if np.sum(mask) < 2:
        return {"rmse": 0.0, "corr": 0.0, "max_des": 0.0, "max_est": 0.0, "mae": 0.0}

    d_sub = des[mask]
    e_sub = est[mask]
    err = d_sub - e_sub

    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    max_d = float(np.max(np.abs(d_sub)))
    max_e = float(np.max(np.abs(e_sub)))

    # Correlation
    if np.std(d_sub) > 1e-4 and np.std(e_sub) > 1e-4:
        corr = float(np.corrcoef(d_sub, e_sub)[0, 1])
    else:
        corr = 1.0 if np.allclose(d_sub, e_sub, atol=0.05) else 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "corr": corr,
        "max_des": max_d,
        "max_est": max_e,
        "count": int(np.sum(mask)),
    }


def add_flight_mode_shading(ax: plt.Axes, intervals: List[Tuple[float, float, str]], y_min: float, y_max: float):
    """Adds background color bands for flight modes."""
    guided_added = False
    loiter_added = False

    for t_s, t_e, mode in intervals:
        if mode == "GUIDED":
            ax.axvspan(t_s, t_e, color="#2ecc71", alpha=0.18, label="GUIDED (Autonomous Tracking)" if not guided_added else None)
            guided_added = True
        elif mode == "LOITER":
            ax.axvspan(t_s, t_e, color="#3498db", alpha=0.10, label="LOITER (Manual Flight)" if not loiter_added else None)
            loiter_added = True
        elif mode in ("ALT_HOLD", "STABILIZE"):
            ax.axvspan(t_s, t_e, color="#f39c12", alpha=0.10)


def plot_combined_velocity_tracking(data: Dict[str, Any], t_min: float, t_max: float, out_path: Path,
                                    relative_time: bool = True, subtitle_extra: str = "",
                                    use_target: bool = True):
    """Generates a clean, uncluttered 2-panel figure showing Target vs Actual velocity."""
    t_n = data["north"]["t"]
    ref_n = data["north"]["target"] if use_target else data["north"]["des"]
    est_n = data["north"]["est"]

    t_e = data["east"]["t"]
    ref_e = data["east"]["target"] if use_target else data["east"]["des"]
    est_e = data["east"]["est"]

    mask_n = (t_n >= t_min) & (t_n <= t_max)
    mask_e = (t_e >= t_min) & (t_e <= t_max)

    t_plot_n = (t_n[mask_n] - t_min) if relative_time else t_n[mask_n]
    t_plot_e = (t_e[mask_e] - t_min) if relative_time else t_e[mask_e]

    fig, (ax_n, ax_e) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, dpi=300)
    plt.subplots_adjust(hspace=0.25)

    c_ref = "#d9381e"  # Crimson dashed line
    c_act = "#1d3557"  # Navy solid line
    ref_label = "Target" if use_target else "Desired"

    # -------------------------------------------------------------------------
    # Panel 1: North Velocity
    # -------------------------------------------------------------------------
    ax_n.plot(t_plot_n, ref_n[mask_n], color=c_ref, linestyle="--", linewidth=2.0, label=ref_label)
    ax_n.plot(t_plot_n, est_n[mask_n], color=c_act, linestyle="-", linewidth=1.8, label="Estimated")
    ax_n.set_title("North Velocity", fontsize=13, fontweight="bold", pad=8)
    ax_n.set_ylabel("Velocity (m/s)", fontsize=11)
    ax_n.grid(True, linestyle="--", alpha=0.45)
    ax_n.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax_n.legend(loc="upper right", fontsize=10, framealpha=0.9)

    y_lim_n = max(1.2, float(np.max(np.abs([ref_n[mask_n], est_n[mask_n]])) * 1.25) if np.any(mask_n) else 1.5)
    ax_n.set_ylim(-y_lim_n, y_lim_n)

    # -------------------------------------------------------------------------
    # Panel 2: East Velocity
    # -------------------------------------------------------------------------
    ax_e.plot(t_plot_e, ref_e[mask_e], color=c_ref, linestyle="--", linewidth=2.0, label=ref_label)
    ax_e.plot(t_plot_e, est_e[mask_e], color=c_act, linestyle="-", linewidth=1.8, label="Estimated")
    ax_e.set_title("East Velocity", fontsize=13, fontweight="bold", pad=8)
    ax_e.set_xlabel("Time since GUIDED activation (s)", fontsize=11)
    ax_e.set_ylabel("Velocity (m/s)", fontsize=11)
    ax_e.grid(True, linestyle="--", alpha=0.45)
    ax_e.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax_e.legend(loc="upper right", fontsize=10, framealpha=0.9)

    y_lim_e = max(1.2, float(np.max(np.abs([ref_e[mask_e], est_e[mask_e]])) * 1.25) if np.any(mask_e) else 1.5)
    ax_e.set_ylim(-y_lim_e, y_lim_e)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] ✓ Saved clean combined 2-panel velocity figure: {out_path}")


def plot_single_axis(t_arr: np.ndarray, ref: np.ndarray, est: np.ndarray, axis_name: str,
                      intervals: List[Tuple[float, float, str]], t_min: float, t_max: float,
                      log_name: str, out_path: Path, relative_time: bool = True,
                      subtitle_extra: str = "", ref_label: str = "Target"):
    """Generates a clean, minimalist standalone plot: Target (or Desired) vs Actual velocity only."""
    mask = (t_arr >= t_min) & (t_arr <= t_max)
    t_plot = (t_arr[mask] - t_min) if relative_time else t_arr[mask]

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)

    c_ref = "#d9381e"  # Crimson dashed line for Target/Desired
    c_act = "#1d3557"  # Deep Navy solid line for Actual

    ax.plot(t_plot, ref[mask], color=c_ref, linestyle="--", linewidth=2.0, label=ref_label)
    ax.plot(t_plot, est[mask], color=c_act, linestyle="-", linewidth=1.8, label="Estimated")

    ax.set_title(f"{axis_name} Velocity", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Velocity (m/s)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

    y_lim = max(1.2, float(np.max(np.abs([ref[mask], est[mask]])) * 1.25) if np.any(mask) else 1.5)
    ax.set_ylim(-y_lim, y_lim)

    ax.legend(loc="upper right", fontsize=11, framealpha=0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] ✓ Saved clean {axis_name} figure: {out_path}")


def print_telemetry_summary(data: Dict[str, Any]):
    """Prints a quantitative performance summary table of flight telemetry to the console."""
    print("\n" + "=" * 90)
    print(f"  ARDUPILOT FLIGHT LOG ANALYSIS: {data['log_name']}")
    print("=" * 90)
    print(f"  Total Log Duration:  {data['total_duration_s']:.1f} s ({data['total_duration_s']/60:.1f} min)")

    if data["arms"]:
        t_arm = data["arms"][0][0]
        t_disarm = data["arms"][-1][0] if len(data["arms"]) > 1 else data["last_t"]
        print(f"  Armed Duration:      {t_disarm - t_arm:.1f} s ([{t_arm:.1f}s -> {t_disarm:.1f}s])")

    print("\n  FLIGHT MODE TRANSITION TIMELINE:")
    for idx, (t_m, mode_name) in enumerate(data["modes"], 1):
        print(f"    [{idx:02d}] t = {t_m:7.2f}s  -->  Flight Mode: {mode_name}")

    # Analyze Guided segments specifically
    guided_segments = []
    modes = data["modes"]
    for i, (t_start, m_name) in enumerate(modes):
        if m_name == "GUIDED":
            t_end = modes[i + 1][0] if i + 1 < len(modes) else data["last_t"]
            guided_segments.append((t_start, t_end))

    if guided_segments:
        print("\n" + "-" * 90)
        print("  AUTONOMOUS GUIDED TRACKING PERFORMANCE BREAKDOWN:")
        print("-" * 90)
        for idx, (g_s, g_e) in enumerate(guided_segments, 1):
            dur = g_e - g_s
            # Target velocity metrics (TVN / TVE vs VN / VE)
            s_nt = compute_tracking_metrics(data["north"]["t"], data["north"]["target"], data["north"]["est"], g_s, g_e)
            s_et = compute_tracking_metrics(data["east"]["t"], data["east"]["target"], data["east"]["est"], g_s, g_e)
            # Desired velocity metrics (DVN / DVE vs VN / VE)
            s_nd = compute_tracking_metrics(data["north"]["t"], data["north"]["des"], data["north"]["est"], g_s, g_e)
            s_ed = compute_tracking_metrics(data["east"]["t"], data["east"]["des"], data["east"]["est"], g_s, g_e)

            print(f"  Engagement #{idx:02d}: t = [{g_s:.1f}s to {g_e:.1f}s] (Duration: {dur:.1f}s, {s_nt['count']} samples @ 10Hz)")
            print(f"    • North Target (TVN): RMSE = {s_nt['rmse']:.3f} m/s | MAE = {s_nt['mae']:.3f} m/s | Corr r = {s_nt['corr']:.3f} | Max Tgt: {s_nt['max_des']:.2f} m/s | Max Real: {s_nt['max_est']:.2f} m/s")
            print(f"    • North Desired(DVN): RMSE = {s_nd['rmse']:.3f} m/s | MAE = {s_nd['mae']:.3f} m/s | Corr r = {s_nd['corr']:.3f} | Max Des: {s_nd['max_des']:.2f} m/s | Max Real: {s_nd['max_est']:.2f} m/s")
            print(f"    • East Target  (TVE): RMSE = {s_et['rmse']:.3f} m/s | MAE = {s_et['mae']:.3f} m/s | Corr r = {s_et['corr']:.3f} | Max Tgt: {s_et['max_des']:.2f} m/s | Max Real: {s_et['max_est']:.2f} m/s")
            print(f"    • East Desired (DVE): RMSE = {s_ed['rmse']:.3f} m/s | MAE = {s_ed['mae']:.3f} m/s | Corr r = {s_ed['corr']:.3f} | Max Des: {s_ed['max_des']:.2f} m/s | Max Real: {s_ed['max_est']:.2f} m/s")

    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Desired vs Real Estimated North and East Velocities from ArduPilot .BIN log",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log", "-l", type=str, default=None,
                        help="Path to .BIN flight log (e.g. logs/flight_logs/00000165.BIN or shorthand '165')")
    parser.add_argument("--engagement", "-e", type=str, default="2",
                        help="Autonomous GUIDED engagement index to plot: '2' (default: second 38.4s pursuit), '1', or 'all'")
    parser.add_argument("--window", "-w", type=str, choices=["engagement", "flight", "guided", "all"], default="engagement",
                        help="Time window to plot: 'engagement' (selected GUIDED engagement, default), 'flight' (armed period), 'guided' (all autonomous segments), 'all' (entire log)")
    parser.add_argument("--relative-time", action="store_true", default=True,
                        help="Plot time relative to engagement start (0 to T seconds, default: True)")
    parser.add_argument("--absolute-time", action="store_false", dest="relative_time",
                        help="Plot raw boot timestamp on x-axis")
    parser.add_argument("--tmin", type=float, default=None,
                        help="Custom start time in seconds (optional override)")
    parser.add_argument("--tmax", type=float, default=None,
                        help="Custom end time in seconds (optional override)")
    parser.add_argument("--outdir", "-o", type=str, default="outputs/flight_analysis",
                        help="Output directory to save plots (default: outputs/flight_analysis)")
    parser.add_argument("--save-individual", action="store_true", default=True,
                        help="Save separate standalone figures for North and East in addition to combined panel")
    parser.add_argument("--use-desired", action="store_true", default=False,
                        help="Plot raw desired velocity (DVN/DVE) instead of rate-limited target velocity (TVN/TVE, default)")

    args = parser.parse_args()

    # 1. Locate and parse log file
    log_path = resolve_log_path(args.log)
    data = parse_flight_log(log_path)

    # 2. Print quantitative telemetry summary
    print_telemetry_summary(data)

    # Identify all guided segments
    guided_segments = []
    modes = data["modes"]
    for i, (t_start, m_name) in enumerate(modes):
        if m_name == "GUIDED":
            t_end = modes[i + 1][0] if i + 1 < len(modes) else data["last_t"]
            guided_segments.append((t_start, t_end))

    # 3. Determine time boundaries for plotting
    is_single_engagement = False
    eng_label = ""
    use_rel_time = args.relative_time

    if args.tmin is not None and args.tmax is not None:
        t_min = args.tmin
        t_max = args.tmax
        use_rel_time = False
    elif args.window == "engagement" or (args.engagement in ("1", "2")):
        try:
            eng_idx = int(args.engagement) - 1
            if 0 <= eng_idx < len(guided_segments):
                t_min, t_max = guided_segments[eng_idx]
                is_single_engagement = True
                eng_label = f"GUIDED Engagement #{eng_idx + 1}"
            else:
                print(f"[WARN] Engagement #{args.engagement} not found (found {len(guided_segments)}). Falling back to full flight.")
                t_min = max(0.0, data["arms"][0][0] - 5.0) if data["arms"] else data["first_t"]
                t_max = min(data["last_t"], data["arms"][-1][0] + 5.0) if data["arms"] else data["last_t"]
                use_rel_time = False
        except ValueError:
            t_min = max(0.0, data["arms"][0][0] - 5.0) if data["arms"] else data["first_t"]
            t_max = min(data["last_t"], data["arms"][-1][0] + 5.0) if data["arms"] else data["last_t"]
            use_rel_time = False
    elif args.window == "flight":
        use_rel_time = False
        if data["arms"]:
            t_min = max(0.0, data["arms"][0][0] - 5.0)
            t_max = min(data["last_t"], data["arms"][-1][0] + 5.0)
        else:
            t_min, t_max = data["first_t"], data["last_t"]
    elif args.window == "guided":
        use_rel_time = False
        if guided_segments:
            t_min = max(0.0, guided_segments[0][0] - 3.0)
            t_max = min(data["last_t"], guided_segments[-1][1] + 3.0)
        else:
            t_min, t_max = data["first_t"], data["last_t"]
    else:  # 'all'
        use_rel_time = False
        t_min, t_max = data["first_t"], data["last_t"]

    out_dir = Path(args.outdir)
    stem = log_path.stem

    if is_single_engagement:
        file_prefix = f"{stem}_guided_engagement_{int(args.engagement)}"
        subtitle = f"Autonomous GUIDED Engagement #{args.engagement} (Duration: {t_max - t_min:.1f}s)"
    else:
        file_prefix = f"{stem}_velocity_tracking"
        subtitle = ""

    intervals = compute_mode_intervals(data, t_min, t_max)
    use_target = not args.use_desired
    ref_label = "Target" if use_target else "Desired"
    ref_key = "target" if use_target else "des"

    # 4. Generate Separate Standalone Plots for North and East
    north_file = out_dir / f"{file_prefix}_north.png"
    plot_single_axis(
        data["north"]["t"], data["north"][ref_key], data["north"]["est"],
        "North", intervals, t_min, t_max, stem, north_file,
        relative_time=use_rel_time, subtitle_extra=subtitle,
        ref_label=ref_label
    )

    east_file = out_dir / f"{file_prefix}_east.png"
    plot_single_axis(
        data["east"]["t"], data["east"][ref_key], data["east"]["est"],
        "East", intervals, t_min, t_max, stem, east_file,
        relative_time=use_rel_time, subtitle_extra=subtitle,
        ref_label=ref_label
    )

    # 5. Generate Combined 2-Panel Figure
    combined_file = out_dir / f"{file_prefix}_combined.png"
    plot_combined_velocity_tracking(
        data, t_min, t_max, combined_file,
        relative_time=use_rel_time, subtitle_extra=subtitle,
        use_target=use_target
    )

    print(f"\n[DONE] All velocity tracking plots generated successfully in: {out_dir}/\n")
    print(f"  • North Velocity Plot:    {north_file}")
    print(f"  • East Velocity Plot:     {east_file}")
    print(f"  • Combined 2-Panel Plot:  {combined_file}")


if __name__ == "__main__":
    main()

