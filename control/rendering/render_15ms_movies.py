"""
High-Speed (15 m/s) Long-Duration Stochastic Drone Pursuit Movie Suite.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Generates high-speed pursuit movies where:
- Both the pursuer and target drone move at ~15 m/s (approx 55 km/h)
- High dynamic accelerations (up to 12 - 15 m/s²)
- Long durations (30.0 - 35.0 seconds, 600 - 700 frames at 20 FPS)
- Saves exclusively MP4 format directly in control_model/drone_chase/
"""

import sys
import shutil
import argparse
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from create_chase_movie import generate_chase_movie
from create_3d_chase_movie import generate_3d_chase_movie

DRONE_CHASE_DIR = CURRENT_DIR / "drone_chase"
DRONE_CHASE_DIR.mkdir(exist_ok=True)

ARTIFACT_DIRS = [
    Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/42b621c5-bd5c-42b8-ae83-87208090e0a7"),
    Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/c2b8fddd-4624-46f7-9768-e1974371d76e"),
]

# 15 m/s (~55 km/h) extended duration suite (30 - 35 seconds)
RUNS_2D = [
    ("stochastic_high_speed_evasive_seed42", "chase_animation_stochastic_15ms_evasive_30s", 30.0),
    ("stochastic_high_speed_transit_seed12", "chase_animation_stochastic_15ms_transit_35s", 35.0),
    ("stochastic_interceptor_combat_seed99", "chase_animation_stochastic_16ms_combat_30s", 30.0),
]

RUNS_3D = [
    ("stochastic_high_speed_evasive_seed42", "chase_3d_stochastic_15ms_evasive_30s", 30.0),
    ("stochastic_interceptor_combat_seed99", "chase_3d_stochastic_16ms_combat_30s", 30.0),
]


def render_all_15ms(fps=20):
    total = len(RUNS_2D) + len(RUNS_3D)
    print(f"\n{'='*74}")
    print(f"  RENDERING {total} HIGH-SPEED (~15 m/s) LONG-DURATION STOCHASTIC MOVIES")
    print(f"  Speed: ~15 m/s (55 km/h) | Acceleration: 2.2 m/s² (pursuer), 5.5-7.5 m/s² (target)")
    print(f"  Durations: 30.0s - 35.0s (600 - 700 frames at 20 FPS)")
    print(f"  Output Format: MP4 only | Destination: {DRONE_CHASE_DIR}")
    print(f"{'='*74}\n")

    rendered = []

    # 1. Render 2D Top-Down + Cockpit HUD Movies
    for idx, (traj_key, out_name, duration) in enumerate(RUNS_2D, 1):
        print(f"\n>>> [{idx}/{total}] 2D Cockpit Movie: {out_name} ({duration}s at ~15 m/s) <<<")
        mp4_path = DRONE_CHASE_DIR / f"{out_name}.mp4"

        generate_chase_movie(
            output_mp4_path=mp4_path,
            output_gif_path=None,
            trajectory_name=traj_key,
            total_time_s=duration,
            fps=fps
        )

        for adir in ARTIFACT_DIRS:
            if adir.exists():
                shutil.copy(mp4_path, adir / mp4_path.name)

        rendered.append((out_name, mp4_path, f"2D Top-Down + HUD ({duration}s, ~15 m/s)"))
        print(f"  [DONE] Saved: {mp4_path.name}")

    # 2. Render 3D Perspective Movies
    start_3d = len(RUNS_2D) + 1
    for idx, (traj_key, out_name, duration) in enumerate(RUNS_3D, start_3d):
        print(f"\n>>> [{idx}/{total}] 3D Spatial Arena: {out_name} ({duration}s at ~15 m/s) <<<")
        mp4_path = DRONE_CHASE_DIR / f"{out_name}.mp4"

        generate_3d_chase_movie(
            output_mp4_path=mp4_path,
            output_gif_path=None,
            trajectory_name=traj_key,
            total_time_s=duration,
            fps=fps
        )

        for adir in ARTIFACT_DIRS:
            if adir.exists():
                shutil.copy(mp4_path, adir / mp4_path.name)

        rendered.append((out_name, mp4_path, f"3D Perspective ({duration}s, ~15 m/s)"))
        print(f"  [DONE] Saved: {mp4_path.name}")

    print(f"\n{'='*74}")
    print(f"  ALL {total} HIGH-SPEED (~15 m/s) MOVIES GENERATED IN {DRONE_CHASE_DIR}!")
    print(f"{'='*74}")
    for name, mp4, desc in rendered:
        print(f"  • {name:<46} [{desc}] -> {mp4.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render ~15 m/s long-duration stochastic movies.")
    parser.add_argument("--fps", type=int, default=20, help="Framerate for export (default: 20)")
    args = parser.parse_args()

    render_all_15ms(fps=args.fps)
