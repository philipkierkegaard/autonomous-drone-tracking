#!/usr/bin/env python3
"""
Batch Renderer: Renders head-to-head dual pursuit videos for all 29 benchmark scenarios
where the Learned Controller (Cloned Expert v2) succeeded and the Kinematic Controller (PID) failed.
Produces individual MP4 videos for each scenario and a concatenated compilation video.
Strictly NO GIFs.
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from multiprocessing import Pool

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.rendering.render_simple_chase import render_dual_chase_video


def render_worker(task: Dict[str, Any]) -> str:
    """Worker function to render a single dual chase scenario."""
    sc_id = task["scenario_id"]
    out_path = Path(task["out_path"])
    
    if out_path.exists() and out_path.stat().st_size > 10000:
        print(f"[SKIP] Video already exists: {out_path.name}")
        return str(out_path)

    t0 = time.perf_counter()
    render_dual_chase_video(
        output_path=out_path,
        duration=task["duration"],
        fps=25,
        profile=task["profile"],
        encounter="custom",
        target_init_pos=task["initial_target_pos"],
        target_init_heading_deg=float(np.rad2deg(task["initial_target_heading_rad"])),
        chaser_init_pos=task["initial_chaser_pos"],
        chaser_init_yaw=float(task["initial_chaser_yaw"]),
        seed=task["seed"],
        model_path=PROJECT_ROOT / "control" / "weights" / "cloned_expert_v2" / "actor_weights.pt",
        output_gif_path=None
    )
    dt = time.perf_counter() - t0
    print(f"[DONE] Rendered {out_path.name} in {dt:.1f}s")
    return str(out_path)


def main():
    bench_dir = PROJECT_ROOT / "outputs" / "benchmarks" / "control"
    rl_file = bench_dir / "benchmark_results_evasive_200_rl_cloned_expert_v2.json"
    pid_file = bench_dir / "benchmark_results_evasive_200_pid_tuned.json"
    suite_file = PROJECT_ROOT / "outputs" / "benchmarks" / "datasets" / "evasive_200_benchmark_suite.json"

    with open(rl_file) as f:
        rl_res = {s["scenario_id"]: s for s in json.load(f)["results"]}
    with open(pid_file) as f:
        pid_res = {s["scenario_id"]: s for s in json.load(f)["results"]}
    with open(suite_file) as f:
        scenarios = {s["scenario_id"]: s for s in json.load(f)["scenarios"]}

    # Identify all scenarios where RL succeeded (success_1s) and PID failed
    diff_scenarios = [
        sc_id for sc_id, r in sorted(rl_res.items())
        if r["success_1s"] and not pid_res.get(sc_id, {}).get("success_1s")
    ]

    print(f"=" * 80)
    print(f"  BATCH RENDERING ALL RL-SUCCESS / PID-FAIL CHASES ({len(diff_scenarios)} SCENARIOS)")
    print(f"=" * 80)

    out_dir = PROJECT_ROOT / "outputs" / "videos" / "control" / "success" / "expertv2" / "dual_200_suite"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for sc_id in diff_scenarios:
        sc = scenarios[sc_id]
        out_mp4 = out_dir / f"dual_{sc_id}.mp4"
        tasks.append({
            "scenario_id": sc_id,
            "profile": sc["profile"],
            "encounter_type": sc["encounter_type"],
            "duration": float(sc.get("duration", 25.0)),
            "seed": int(sc["seed"]),
            "initial_target_pos": sc["initial_target_pos"],
            "initial_target_heading_rad": sc["initial_target_heading_rad"],
            "initial_chaser_pos": sc.get("initial_chaser_pos", [0.0, 0.0, -2.5]),
            "initial_chaser_yaw": sc.get("initial_chaser_yaw", 0.0),
            "out_path": str(out_mp4)
        })

    num_workers = min(6, os.cpu_count() or 4)
    print(f"[INIT] Launching parallel render pool with {num_workers} worker processes...")
    t_start = time.perf_counter()

    with Pool(processes=num_workers) as pool:
        rendered_paths = pool.map(render_worker, tasks)

    total_time = time.perf_counter() - t_start
    print(f"\n[COMPLETE] Successfully rendered all {len(rendered_paths)} videos in {total_time:.1f}s ({total_time/60:.2f} min)!")

    # Concatenate all 29 clips into one single continuous showcase video
    print(f"\n[CONCAT] Generating single compilation video of all {len(rendered_paths)} chases...")
    concat_list_file = out_dir / "concat_list.txt"
    with open(concat_list_file, "w") as f:
        for p in rendered_paths:
            f.write(f"file '{Path(p).resolve()}'\n")

    compilation_mp4 = out_dir / "all_29_rl_successes_compilation.mp4"
    ffmpeg_concat = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list_file),
        "-c", "copy",
        str(compilation_mp4)
    ]
    subprocess.run(ffmpeg_concat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"[SUCCESS] Master compilation video saved to: {compilation_mp4}")

    # Copy to brain artifact directory for embedding in reports
    brain_dir = Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/5967d853-7a7e-4cdd-8792-4a77e6d704a2")
    if brain_dir.exists():
        import shutil
        shutil.copy(compilation_mp4, brain_dir / compilation_mp4.name)
        print(f"[COPIED] Mirrored compilation to artifact directory: {brain_dir / compilation_mp4.name}")


if __name__ == "__main__":
    main()
