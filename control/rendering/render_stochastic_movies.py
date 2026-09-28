#!/usr/bin/env python3
"""
Stochastic Trajectory Pursuit Movie Renderer (High-Speed & Extended Duration Suite).
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Generates extended duration (20-22 seconds) stochastic pursuit movies with
higher accelerations (up to 8.0 m/s^2) and higher speeds (up to 12-14 m/s peak).

Outputs:
- Primary: outputs/videos/control/
- Legacy / Mirror: control/rendering/drone_chase/
- IDE Artifacts: Active conversation artifact directory

Available Movies:
1. chase_animation_stochastic_evasive_20s       (2D Top-Down + FPV + Telemetry HUD, v_mean=4.5 m/s)
2. chase_animation_stochastic_hyper_evasive_20s (2D Top-Down + FPV + Telemetry HUD, v_mean=7.5 m/s, a_max=6.5 m/s²)
3. chase_animation_stochastic_sprint_break_20s  (2D Top-Down + FPV + Telemetry HUD, v_mean=9.0 m/s, a_max=8.0 m/s²)
4. chase_animation_stochastic_aerobatic_22s     (2D Top-Down + FPV + Telemetry HUD, v_mean=6.5 m/s, a_max=5.5 m/s²)
5. chase_3d_stochastic_evasive_20s              (3D Spatial Arena + FPV + Altitude HUD)
6. chase_3d_stochastic_hyper_evasive_20s         (3D Spatial Arena with orbiting camera)
7. chase_3d_stochastic_sprint_break_20s          (3D high-G aerial combat pursuit)
"""

import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from control.rendering.create_chase_movie import generate_chase_movie
from control.rendering.create_3d_chase_movie import generate_3d_chase_movie

# Output directories
OUTPUT_VIDEOS_DIR = PROJECT_ROOT / "outputs" / "videos" / "control"
OUTPUT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

DRONE_CHASE_DIR = CURRENT_DIR / "drone_chase"
DRONE_CHASE_DIR.mkdir(exist_ok=True)

# Active session artifact directory for immediate visualization in IDE
ARTIFACT_DIRS = [
    Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/8ca4ccaf-7f51-44db-868c-e36692cc2e86"),
]

# High-speed, high-acceleration extended duration suite
RUNS_2D: List[Tuple[str, str, float, str]] = [
    ("stochastic_evasive_seed42", "chase_animation_stochastic_evasive_20s", 20.0, "2D Evasive (Seed 42)"),
    ("stochastic_hyper_evasive_seed42", "chase_animation_stochastic_hyper_evasive_20s", 20.0, "2D Hyper-Evasive (Seed 42)"),
    ("stochastic_sprint_break_seed99", "chase_animation_stochastic_sprint_break_20s", 20.0, "2D Sprint-Break (Seed 99)"),
    ("stochastic_aerobatic_seed105", "chase_animation_stochastic_aerobatic_22s", 22.0, "2D Aerobatic (Seed 105)"),
]

RUNS_3D: List[Tuple[str, str, float, str]] = [
    ("stochastic_evasive_seed42", "chase_3d_stochastic_evasive_20s", 20.0, "3D Spatial Evasive (Seed 42)"),
    ("stochastic_hyper_evasive_seed42", "chase_3d_stochastic_hyper_evasive_20s", 20.0, "3D Hyper-Evasive (Seed 42)"),
    ("stochastic_sprint_break_seed99", "chase_3d_stochastic_sprint_break_20s", 20.0, "3D Sprint-Break (Seed 99)"),
]


def copy_to_destinations(src_path: Path):
    """Copies generated video and optional gif to mirror directories."""
    # Copy to legacy drone_chase directory
    legacy_path = DRONE_CHASE_DIR / src_path.name
    if legacy_path.resolve() != src_path.resolve():
        shutil.copy(src_path, legacy_path)

    # Copy to IDE artifact directory for display
    for adir in ARTIFACT_DIRS:
        if adir.exists():
            dest = adir / src_path.name
            if dest.resolve() != src_path.resolve():
                shutil.copy(src_path, dest)


def render_stochastic_movies(
    mode: str = "all",
    profile_filter: Optional[str] = None,
    fps: int = 20,
    duration_override: Optional[float] = None,
    export_gif: bool = False
):
    selected_2d = []
    selected_3d = []

    if mode in ["all", "2d"]:
        selected_2d = [
            r for r in RUNS_2D
            if (profile_filter is None or profile_filter.lower() in r[0].lower() or profile_filter.lower() in r[1].lower())
        ]

    if mode in ["all", "3d"]:
        selected_3d = [
            r for r in RUNS_3D
            if (profile_filter is None or profile_filter.lower() in r[0].lower() or profile_filter.lower() in r[1].lower())
        ]

    total_movies = len(selected_2d) + len(selected_3d)
    if total_movies == 0:
        print(f"[ERROR] No runs matched filter mode='{mode}', profile='{profile_filter}'.")
        print(f"Available profiles: evasive, hyper_evasive, sprint_break, aerobatic")
        return

    print(f"\n{'='*76}")
    print(f"  RENDERING {total_movies} HIGH-SPEED STOCHASTIC PURSUIT MOVIES")
    print(f"  Mode:       {mode.upper()} ({len(selected_2d)} 2D runs, {len(selected_3d)} 3D runs)")
    print(f"  Framerate:  {fps} FPS")
    print(f"  Destination:{OUTPUT_VIDEOS_DIR}")
    print(f"{'='*76}\n")

    rendered = []
    curr_idx = 1

    # 1. Render 2D Cockpit + Top-Down Arena Movies
    for traj_key, out_name, def_duration, desc in selected_2d:
        dur = duration_override if duration_override is not None else def_duration
        print(f"\n>>> [{curr_idx}/{total_movies}] 2D Movie: {out_name} ({desc}, {dur:.1f}s) <<<")
        mp4_path = OUTPUT_VIDEOS_DIR / f"{out_name}.mp4"
        gif_path = OUTPUT_VIDEOS_DIR / f"{out_name}.gif" if export_gif else None

        generate_chase_movie(
            output_mp4_path=mp4_path,
            output_gif_path=gif_path,
            trajectory_name=traj_key,
            total_time_s=dur,
            fps=fps
        )

        copy_to_destinations(mp4_path)
        if gif_path and gif_path.exists():
            copy_to_destinations(gif_path)

        rendered.append((out_name, mp4_path, f"2D Top-Down + FPV ({dur:.1f}s)"))
        curr_idx += 1

    # 2. Render 3D Spatial Perspective Movies
    for traj_key, out_name, def_duration, desc in selected_3d:
        dur = duration_override if duration_override is not None else def_duration
        print(f"\n>>> [{curr_idx}/{total_movies}] 3D Movie: {out_name} ({desc}, {dur:.1f}s) <<<")
        mp4_path = OUTPUT_VIDEOS_DIR / f"{out_name}.mp4"
        gif_path = OUTPUT_VIDEOS_DIR / f"{out_name}.gif" if export_gif else None

        generate_3d_chase_movie(
            output_mp4_path=mp4_path,
            output_gif_path=gif_path,
            trajectory_name=traj_key,
            total_time_s=dur,
            fps=fps
        )

        copy_to_destinations(mp4_path)
        if gif_path and gif_path.exists():
            copy_to_destinations(gif_path)

        rendered.append((out_name, mp4_path, f"3D Perspective ({dur:.1f}s)"))
        curr_idx += 1

    print(f"\n{'='*76}")
    print(f"  ALL {total_movies} MOVIES GENERATED SUCCESSFULLY!")
    print(f"{'='*76}")
    for name, mp4, desc in rendered:
        size_mb = mp4.stat().st_size / (1024 * 1024)
        print(f"  • {name:<46} [{desc}] -> {size_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render high-dynamic stochastic pursuit movies.")
    parser.add_argument("--mode", type=str, choices=["all", "2d", "3d"], default="all",
                        help="Render mode: 2d, 3d, or all (default: all)")
    parser.add_argument("--profile", "--trajectory", type=str, default=None,
                        help="Filter by trajectory name (e.g. evasive, hyper_evasive, sprint_break, aerobatic)")
    parser.add_argument("--fps", type=int, default=20, help="Framerate for export (default: 20)")
    parser.add_argument("--duration", type=float, default=None, help="Override duration in seconds")
    parser.add_argument("--gif", action="store_true", help="Also export animated preview GIFs")
    args = parser.parse_args()

    render_stochastic_movies(
        mode=args.mode,
        profile_filter=args.profile,
        fps=args.fps,
        duration_override=args.duration,
        export_gif=args.gif
    )
