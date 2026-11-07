#!/usr/bin/env python3
"""
Render Video: Steady Circling Target Orbiting Around the Chaser Drone.
Uses the tuned slow PID controller with empirical physical yaw lag (tau_yaw = 770 ms).

The target drone flies in a smooth, continuous circle around the chaser at nominal
standoff distance (R = 3.5m) and steady cruise speed (v = 1.0 m/s), matching the
camera mount's +15-degree mechanical up-tilt geometry.
"""

import sys
import argparse
import shutil
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.control.benchmark_controller import (
    create_pid_controller,
    FastPixhawkQuadSim,
    evaluate_single_run
)
from control.rendering.render_pid_chase_bright import render_pid_bright_video

ARTIFACT_DIR = Path("/Users/philipkierkegaard/.gemini/antigravity/brain/3f6fb6d1-ed45-4315-9b5d-b3cca4e6bef7")


class SteadyCirclingOracle:
    """
    Trajectory oracle producing a smooth, continuous circle around the chaser drone.
    Maintains a steady radius R, elevation lead for the +15 deg camera mount,
    and a smooth C^1 velocity ramp from rest.
    """
    def __init__(
        self,
        radius: float = 3.5,
        cruise_speed: float = 1.0,
        ramp_time: float = 2.0,
        altitude: float = -2.5,
        camera_uptilt_deg: float = 15.0,
        direction: int = 1  # +1 = CCW (drifts right in camera), -1 = CW
    ):
        self.sim: Optional[FastPixhawkQuadSim] = None
        self.radius = float(radius)
        self.cruise_speed = float(cruise_speed)
        self.omega_cruise = float(direction * (cruise_speed / radius))
        self.ramp_time = float(ramp_time)
        self.alt = float(altitude)
        # Vertical elevation lead in NED to place target on camera optical center:
        # Camera is tilted +15 deg up, so target must be R * sin(15 deg) above chaser (-z in NED)
        self.dz_optical = -self.radius * np.sin(np.deg2rad(camera_uptilt_deg))

    def _theta(self, t: float) -> float:
        """Computes current orbit angle with a smooth acceleration ramp from rest."""
        if t <= 0.0:
            return 0.0
        elif t < self.ramp_time:
            a = self.omega_cruise / self.ramp_time
            return 0.5 * a * (t ** 2)
        else:
            theta_ramp = 0.5 * self.omega_cruise * self.ramp_time
            return theta_ramp + self.omega_cruise * (t - self.ramp_time)

    def get_position(self, t: float) -> np.ndarray:
        th = self._theta(t)
        if self.sim is not None:
            c_pos = self.sim.pos
            x = c_pos[0] + self.radius * np.cos(th)
            y = c_pos[1] + self.radius * np.sin(th)
            z = c_pos[2] + self.dz_optical
        else:
            x = self.radius * np.cos(th)
            y = self.radius * np.sin(th)
            z = self.alt + self.dz_optical
        return np.array([x, y, z], dtype=np.float64)

    def get_state(self, t: float):
        pos = self.get_position(t)
        th = self._theta(t)
        omega = (self.omega_cruise / self.ramp_time) * t if t < self.ramp_time else self.omega_cruise
        c_vel = self.sim.vel if self.sim is not None else np.zeros(3)
        vx = c_vel[0] - self.radius * omega * np.sin(th)
        vy = c_vel[1] + self.radius * omega * np.cos(th)
        vz = c_vel[2]
        vel = np.array([vx, vy, vz], dtype=np.float64)
        acc = np.zeros(3, dtype=np.float64)
        return pos, vel, acc


class SteadyCirclingScenario:
    """Scenario descriptor conforming to PursuitScenario interface."""
    def __init__(
        self,
        oracle: SteadyCirclingOracle,
        duration: float = 24.0,
        scenario_id: str = "val_steady_circling_around_chaser"
    ):
        self.scenario_id = scenario_id
        self.profile = "steady_slow"
        self.encounter_type = "steady_circling_orbit"
        self.duration = float(duration)
        self.seed = 42
        self.initial_distance = oracle.radius
        self.initial_azimuth_deg = 0.0
        self.initial_elevation_deg = 0.0
        self.initial_target_pos = [oracle.radius, 0.0, oracle.alt + oracle.dz_optical]
        self.initial_target_heading_rad = np.pi / 2.0
        self.initial_chaser_pos = [0.0, 0.0, oracle.alt]
        self.initial_chaser_yaw = 0.0
        self.initial_chaser_speed = 0.0
        self.oracle = oracle

    def create_target_oracle(self) -> SteadyCirclingOracle:
        return self.oracle


def main():
    parser = argparse.ArgumentParser(description="Render video of steady circling target around chaser.")
    parser.add_argument("--radius", type=float, default=3.5, help="Orbit radius in meters (default: 3.5m)")
    parser.add_argument("--speed", type=float, default=1.0, help="Target cruise speed in m/s (default: 1.0 m/s)")
    parser.add_argument("--duration", type=float, default=24.0, help="Flight duration in seconds (default: 24.0s)")
    parser.add_argument("--fps", type=int, default=25, help="Render FPS (default: 25)")
    parser.add_argument(
        "--pid-gains",
        type=str,
        default=str(PROJECT_ROOT / "control" / "weights" / "tuned_pid_gains_slow_tau770.json"),
        help="Path to PID gains JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "videos" / "control" / "val_steady_circling_around_chaser_slow_tau770.mp4"),
        help="Output video path"
    )
    args = parser.parse_args()

    print(f"[CIRCLING] Initializing steady circling experiment:")
    print(f"           Orbit Radius : {args.radius:.2f} m")
    print(f"           Cruise Speed : {args.speed:.2f} m/s")
    print(f"           Duration     : {args.duration:.1f} s")
    print(f"           PID Gains    : {args.pid_gains}")

    oracle = SteadyCirclingOracle(
        radius=args.radius,
        cruise_speed=args.speed,
        ramp_time=2.0,
        altitude=-2.5,
        camera_uptilt_deg=15.0
    )
    sc = SteadyCirclingScenario(oracle=oracle, duration=args.duration)

    pid_controller = create_pid_controller(
        gains_path=args.pid_gains,
        standoff_nominal=args.radius,
        desired_bbox_size=110.0,
    )

    # Dynamically inject the simulator instance into the oracle so it orbits relative to the chaser
    orig_init = FastPixhawkQuadSim.__init__
    def patched_init(self, *s_args, **s_kwargs):
        orig_init(self, *s_args, **s_kwargs)
        oracle.sim = self

    FastPixhawkQuadSim.__init__ = patched_init

    out_path = Path(args.output)
    render_pid_bright_video(
        scenario=sc,
        pid_controller=pid_controller,
        output_path=out_path,
        fps=args.fps
    )

    FastPixhawkQuadSim.__init__ = orig_init

    # Ensure copied to active artifact directory
    if ARTIFACT_DIR.exists():
        dest = ARTIFACT_DIR / out_path.name
        shutil.copy(out_path, dest)
        print(f"[ARTIFACT] Synced video to artifact directory: {dest}")


if __name__ == "__main__":
    main()

