#!/usr/bin/env python3
"""
Inspect and summarize ArduPilot DataFlash binary log 00000168.BIN.
"""

import sys
import time
from pathlib import Path
from collections import Counter
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from pymavlink import mavutil

BIN_PATH = PROJECT_ROOT / "logs" / "flight_logs" / "00000168.BIN"

def main():
    print(f"Opening {BIN_PATH} (size: {BIN_PATH.stat().st_size / (1024*1024):.2f} MB)...")
    t0 = time.time()
    mlog = mavutil.mavlink_connection(str(BIN_PATH))

    modes = []
    messages = []
    arm_events = []
    msg_types = Counter()
    first_t, last_t = None, None

    alt_min, alt_max = float("inf"), float("-inf")
    rate_count = 0
    guided_intervals = []
    current_guided_start = None
    last_mode = None

    while True:
        m = mlog.recv_msg()
        if m is None:
            break
        mtype = m.get_type()
        msg_types[mtype] += 1

        t_us = getattr(m, "TimeUS", None)
        if t_us is not None:
            t_sec = t_us / 1e6
            if first_t is None:
                first_t = t_sec
            last_t = t_sec

            if mtype == "MODE":
                d = m.to_dict()
                mode_name = mavutil.mode_mapping_acm.get(d.get("Mode", -1), f"Mode_{d.get('Mode')}")
                modes.append((t_sec, mode_name, d.get("Mode")))
                if mode_name == "GUIDED" and last_mode != "GUIDED":
                    current_guided_start = t_sec
                elif mode_name != "GUIDED" and last_mode == "GUIDED":
                    if current_guided_start is not None:
                        guided_intervals.append((current_guided_start, t_sec))
                        current_guided_start = None
                last_mode = mode_name

            elif mtype == "MSG":
                d = m.to_dict()
                text = d.get("Message", "")
                messages.append((t_sec, text))

            elif mtype == "EV":
                d = m.to_dict()
                eid = d.get("Id", -1)
                # 10=ARMED, 11=DISARMED in ArduPilot event ids
                arm_events.append((t_sec, eid))

            elif mtype == "CTUN":
                alt = getattr(m, "Alt", None)
                if alt is not None:
                    alt_min = min(alt_min, alt)
                    alt_max = max(alt_max, alt)

            elif mtype == "RATE":
                rate_count += 1

    if current_guided_start is not None and last_t is not None:
        guided_intervals.append((current_guided_start, last_t))

    t_elapsed = time.time() - t0
    dur = (last_t - first_t) if (first_t and last_t) else 0.0

    print(f"\n[DONE] Parsed in {t_elapsed:.1f}s.")
    print(f"Log time range: {first_t:.2f}s to {last_t:.2f}s (Total duration: {dur:.1f}s = {dur/60.0:.2f} min)")
    print(f"Total RATE samples: {rate_count} (~{rate_count/max(dur, 1.0):.1f} Hz)")
    print(f"Altitude range (CTUN.Alt): min={alt_min:.2f}m, max={alt_max:.2f}m")

    print(f"\n--- Top Message Types ---")
    for mt, count in msg_types.most_common(15):
        print(f"  {mt:<12}: {count:,}")

    print(f"\n--- Flight Modes Traversed ({len(modes)} transitions) ---")
    for t_sec, mname, mnum in modes:
        print(f"  t={t_sec:8.2f}s | Mode: {mname:<12} (num={mnum})")

    print(f"\n--- Arming / Disarming Events ({len(arm_events)}) ---")
    for t_sec, eid in arm_events:
        desc = "ARMED" if eid == 10 else ("DISARMED" if eid == 11 else f"Event_{eid}")
        print(f"  t={t_sec:8.2f}s | {desc}")

    print(f"\n--- GUIDED Mode Intervals ({len(guided_intervals)}) ---")
    for i, (g_start, g_end) in enumerate(guided_intervals):
        print(f"  Engagement {i+1}: t={g_start:8.2f}s -> {g_end:8.2f}s (Duration: {g_end - g_start:.2f}s)")

    print(f"\n--- System Status & Info Messages (First 20 & Last 10) ---")
    for t_sec, txt in messages[:15]:
        print(f"  [{t_sec:8.2f}s] {txt}")
    if len(messages) > 25:
        print("  ...")
        for t_sec, txt in messages[-10:]:
            print(f"  [{t_sec:8.2f}s] {txt}")

if __name__ == "__main__":
    main()

