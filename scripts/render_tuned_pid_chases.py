#!/usr/bin/env python3
"""
Renders a batch of chase videos using the Tuned Kinematic PID Controller
across Evasive and Hyper-Evasive profiles.
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"
if not PYTHON_BIN.exists():
    PYTHON_BIN = Path(sys.executable)

RENDER_SCRIPT = PROJECT_ROOT / "control" / "rendering" / "render_simple_chase.py"
OUT_DIR = PROJECT_ROOT / "outputs" / "videos" / "control"
OUT_DIR.mkdir(parents=True, exist_ok=True)

runs = [
    {
        "name": "tuned_pid_evasive_crossing_seed42",
        "profile": "evasive",
        "encounter": "crossing_right",
        "seed": 42,
        "duration": 20.0
    },
    {
        "name": "tuned_pid_evasive_tailchase_seed100",
        "profile": "evasive",
        "encounter": "tail_chase",
        "seed": 100,
        "duration": 20.0
    },
    {
        "name": "tuned_pid_hyper_evasive_seed42",
        "profile": "hyper_evasive",
        "encounter": "tail_chase",
        "seed": 42,
        "duration": 20.0
    },
    {
        "name": "tuned_pid_hyper_evasive_crossing_seed202",
        "profile": "hyper_evasive",
        "encounter": "crossing_left",
        "seed": 202,
        "duration": 20.0
    },
]

print(f"[RUNNER] Launching {len(runs)} chase renders with Tuned PID controller...")

for idx, r in enumerate(runs, 1):
    mp4_path = OUT_DIR / f"{r['name']}.mp4"
    print(f"\n=======================================================")
    print(f"[{idx}/{len(runs)}] Rendering: {r['name']}")
    print(f"  Profile: {r['profile']} | Encounter: {r['encounter']} | Seed: {r['seed']}")
    print(f"  Target: {mp4_path}")
    print(f"=======================================================")

    cmd = [
        str(PYTHON_BIN),
        str(RENDER_SCRIPT),
        "--controller", "pid",
        "--profile", r["profile"],
        "--encounter", r["encounter"],
        "--seed", str(r["seed"]),
        "--duration", str(r["duration"]),
        "--out", str(mp4_path),
        "--gif"
    ]

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        print(f"[ERROR] Failed rendering {r['name']}")
    else:
        print(f"[SUCCESS] Finished {r['name']}.mp4 + .gif")

print("\n[ALL COMPLETE] All chase videos successfully generated.")
