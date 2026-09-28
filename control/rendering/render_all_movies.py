"""
Batch renders 2D chase movies for all benchmark and stochastic flight trajectories:
1. figure8 (3D Aerobatic Figure-8 with alternating roll/climb)
2. slalom (High-Speed Evasive Slalom at 3.5 m/s)
3. break_turn (Tactical Break & Dive: 90° snap-turn with rapid altitude plunge)
4. corkscrew (Expanding 3D Helical Spiral ascent)
5. stochastic_evasive (Markov B-spline evasive random walk, seed 42)
6. stochastic_aerobatic (High-G dynamic aerobatic maneuvers, seed 77)
7. stochastic_cruising (Realistic wind-buffeted transit flight, seed 12)

Ensures native Apple QuickTime compatibility (H.264, yuv420p, +faststart)
and exports animated GIFs for instant previewing in reports and documentation.
"""

import sys
import shutil
import argparse
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from create_chase_movie import generate_chase_movie

ARTIFACT_DIRS = [
    Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/42b621c5-bd5c-42b8-ae83-87208090e0a7"),
    Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/c2b8fddd-4624-46f7-9768-e1974371d76e"),
]

DEFAULT_TRAJECTORIES = [
    "figure8",
    "slalom",
    "break_turn",
    "corkscrew",
    "stochastic_evasive",
    "stochastic_aerobatic",
]


def render_all(trajectories=None, duration_s=16.0, fps=20):
    if trajectories is None:
        trajectories = DEFAULT_TRAJECTORIES

    print(f"\n{'='*72}")
    print(f"  STARTING BATCH CHASE MOVIE GENERATION")
    print(f"  Trajectories to render: {trajectories}")
    print(f"  Duration: {duration_s}s | FPS: {fps}")
    print(f"{'='*72}\n")

    drone_chase_dir = CURRENT_DIR / "drone_chase"
    drone_chase_dir.mkdir(exist_ok=True)

    rendered_files = []

    for idx, traj in enumerate(trajectories, 1):
        print(f"\n>>> [{idx}/{len(trajectories)}] Generating Movie for: [{traj.upper()}] <<<")
        mp4_path = CURRENT_DIR / f"chase_animation_{traj}.mp4"
        gif_path = CURRENT_DIR / f"chase_animation_{traj}.gif"

        generate_chase_movie(
            output_mp4_path=mp4_path,
            output_gif_path=gif_path,
            trajectory_name=traj,
            total_time_s=duration_s,
            fps=fps
        )

        # Copy to drone_chase subdirectory
        shutil.copy(mp4_path, drone_chase_dir / mp4_path.name)
        shutil.copy(gif_path, drone_chase_dir / gif_path.name)

        # Copy to conversation artifact directories
        for adir in ARTIFACT_DIRS:
            if adir.exists():
                shutil.copy(mp4_path, adir / mp4_path.name)
                shutil.copy(gif_path, adir / gif_path.name)

        rendered_files.append((traj, mp4_path, gif_path))
        print(f"  [OK] Finished {traj.upper()}: MP4 & GIF saved.")

    print(f"\n{'='*72}")
    print(f"  ALL {len(trajectories)} TRAJECTORY MOVIES SUCCESSFULLY GENERATED!")
    print(f"{'='*72}")
    for traj, mp4, gif in rendered_files:
        print(f"  • {traj:<22} -> {mp4.name} & {gif.name}")
    print(f"\nStored in:\n  - {CURRENT_DIR}\n  - {drone_chase_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch render drone chase movies.")
    parser.add_argument("--trajectories", nargs="+", default=DEFAULT_TRAJECTORIES, help="List of trajectories to render")
    parser.add_argument("--duration", type=float, default=14.0, help="Simulation duration in seconds")
    parser.add_argument("--fps", type=int, default=20, help="Video frame rate")
    args = parser.parse_args()

    render_all(trajectories=args.trajectories, duration_s=args.duration, fps=args.fps)
