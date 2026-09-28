"""
Stochastic Drone Trajectory Generator (Black-Box Evaluation Oracle).
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Generates physically feasible, continuous 3D target drone trajectories drawn from
a stochastic Markov Waypoint process smoothed with a C^2 cubic B-spline.

Guarantees:
- Differential Flatness compliance: continuous position p(t), velocity v(t), and acceleration a(t).
- Bounded physical acceleration: a(t) <= a_max (motor thrust limits).
- Closed-form analytical ground truth states at any arbitrary query timestamp t.
- Configurable flight profiles: 'cruising', 'evasive', 'aerobatic', or custom distribution parameters.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Union
import numpy as np
from scipy.interpolate import make_interp_spline


@dataclass
class FlightProfile:
    """Parameters governing the stochastic trajectory distribution."""
    mean_speed: float        # Average forward cruise speed (m/s)
    speed_std: float         # Velocity magnitude variance (m/s)
    turn_rate_std: float     # Heading change variance per second (rad/s)
    climb_rate_std: float    # Vertical climb/dive variance per second (m/s)
    max_accel: float         # Maximum allowable physical acceleration ceiling (m/s^2)
    altitude_bounds: Tuple[float, float]  # NED [max_altitude, min_altitude] (e.g. [-1.0, -6.0])
    waypoint_interval: float # Temporal spacing between random walk anchors (seconds)


PROFILES = {
    "cruising": FlightProfile(
        mean_speed=3.0,
        speed_std=0.5,
        turn_rate_std=0.20,
        climb_rate_std=0.25,
        max_accel=2.2,
        altitude_bounds=(-1.0, -5.0),
        waypoint_interval=3.0,
    ),
    "evasive": FlightProfile(
        mean_speed=4.5,
        speed_std=0.8,
        turn_rate_std=0.45,
        climb_rate_std=0.50,
        max_accel=3.5,

        altitude_bounds=(-0.5, -6.5),
        waypoint_interval=2.0,
    ),
    "aerobatic": FlightProfile(
        mean_speed=10.0,
        speed_std=1.2,
        turn_rate_std=0.75,
        climb_rate_std=0.80,
        max_accel=7.0,
        altitude_bounds=(-0.5, -8.0),
        waypoint_interval=1.5,
    ),
    "hyper_evasive": FlightProfile(
        mean_speed=9.5,
        speed_std=1.5,
        turn_rate_std=0.95,
        climb_rate_std=1.10,
        max_accel=7.5,
        altitude_bounds=(-0.5, -10.0),
        waypoint_interval=2.2,
    ),
    "sprint_break": FlightProfile(
        mean_speed=9.0,
        speed_std=1.8,
        turn_rate_std=1.20,
        climb_rate_std=1.40,
        max_accel=8.0,
        altitude_bounds=(-0.5, -12.0),
        waypoint_interval=1.0,
    ),
    "high_speed_transit": FlightProfile(
        mean_speed=14.5,
        speed_std=1.8,
        turn_rate_std=0.20,
        climb_rate_std=0.70,
        max_accel=5.5,
        altitude_bounds=(-1.0, -25.0),
        waypoint_interval=2.8,
    ),
    "high_speed_evasive": FlightProfile(
        mean_speed=15.5,
        speed_std=2.0,
        turn_rate_std=0.28,
        climb_rate_std=0.90,
        max_accel=6.5,
        altitude_bounds=(-1.0, -28.0),
        waypoint_interval=2.4,
    ),
    "interceptor_combat": FlightProfile(
        mean_speed=16.5,
        speed_std=2.4,
        turn_rate_std=0.35,
        climb_rate_std=1.20,
        max_accel=7.5,
        altitude_bounds=(-1.0, -32.0),
        waypoint_interval=2.0,
    ),
}



class StochasticTargetTrajectory:
    """
    Black-Box Drone Trajectory Oracle.
    
    Usage:
        >>> target = StochasticTargetTrajectory(profile="evasive", duration=30.0, seed=42)
        >>> pos = target.get_position(t=5.2)     # [x, y, z] in NED
        >>> vel = target.get_velocity(t=5.2)     # [vx, vy, vz] in NED
        >>> pos, vel, acc = target.get_state(t=5.2)
    """

    def __init__(
        self,
        duration: float = 30.0,
        profile: Union[str, FlightProfile] = "evasive",
        initial_pos: Optional[np.ndarray] = None,
        initial_heading_rad: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """
        Initializes and synthesizes a smooth stochastic flight path.

        Args:
            duration: Total flight duration in seconds.
            profile: Name of preset ('cruising', 'evasive', 'aerobatic') or custom FlightProfile.
            initial_pos: Starting 3D coordinate in NED [x, y, z] (default: [12.0, 0.0, -2.0]).
            initial_heading_rad: Initial horizontal heading (radians). If None, sampled uniformly.
            seed: Random seed for 100% deterministic reproducibility.
        """
        self.duration = float(duration)
        if isinstance(profile, str):
            if profile not in PROFILES:
                raise ValueError(f"Unknown profile '{profile}'. Choose from {list(PROFILES.keys())}")
            self.p = PROFILES[profile]
            self.profile_name = profile
        else:
            self.p = profile
            self.profile_name = "custom"

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if initial_pos is None:
            self.initial_pos = np.array([12.0, 0.0, -2.0], dtype=np.float64)
        else:
            self.initial_pos = np.array(initial_pos, dtype=np.float64)

        if initial_heading_rad is None:
            self.initial_heading = float(self.rng.uniform(-np.pi / 4, np.pi / 4))  # Gen. forward North
        else:
            self.initial_heading = float(initial_heading_rad)

        # Build the spline representation
        self._build_trajectory()

    def _build_trajectory(self):
        """Synthesizes discrete random walk waypoints and solves the C^2 B-spline."""
        dt_wp = self.p.waypoint_interval
        n_wp = int(np.ceil(self.duration / dt_wp)) + 3
        t_wp = np.linspace(0.0, (n_wp - 1) * dt_wp, n_wp)

        waypoints = np.zeros((n_wp, 3), dtype=np.float64)
        waypoints[0] = self.initial_pos

        heading = self.initial_heading
        z_max, z_min = self.p.altitude_bounds  # Note: z_min is more negative (higher altitude)

        for i in range(1, n_wp):
            dt_step = t_wp[i] - t_wp[i - 1]

            # 1. Sample forward speed from truncated normal distribution
            speed = float(self.rng.normal(self.p.mean_speed, self.p.speed_std))
            speed = np.clip(speed, 1.0, self.p.mean_speed * 1.8)

            # 2. Sample heading angular rate
            d_heading = float(self.rng.normal(0.0, self.p.turn_rate_std * np.sqrt(dt_step)))
            heading += d_heading

            # 3. Horizontal position step
            dx = speed * np.cos(heading) * dt_step
            dy = speed * np.sin(heading) * dt_step

            # 4. Vertical position step with smooth soft-boundary repulsion
            current_z = waypoints[i - 1, 2]
            dz = float(self.rng.normal(0.0, self.p.climb_rate_std * dt_step))

            # Repel downwards if climbing too close to upper bound (z_min)
            if current_z < z_min + 1.0:
                dz += 0.4 * dt_step
            # Repel upwards if diving too close to ground bound (z_max)
            elif current_z > z_max - 1.0:
                dz -= 0.4 * dt_step

            next_pos = waypoints[i - 1] + np.array([dx, dy, dz], dtype=np.float64)
            next_pos[2] = np.clip(next_pos[2], z_min, z_max)
            waypoints[i] = next_pos

        # Fit cubic B-spline (C^2 continuous: position, velocity, and acceleration exist everywhere)
        self._spline_pos = make_interp_spline(t_wp, waypoints, k=3)
        self._spline_vel = self._spline_pos.derivative(nu=1)
        self._spline_acc = self._spline_pos.derivative(nu=2)

        # Enforce physical acceleration ceiling
        t_dense = np.linspace(0.0, self.duration, int(self.duration * 50))
        acc_dense = self._spline_acc(t_dense)
        acc_magnitudes = np.linalg.norm(acc_dense, axis=1)
        peak_accel = float(np.max(acc_magnitudes))

        if peak_accel > self.p.max_accel and peak_accel > 1e-4:
            self._time_warp_scale = np.sqrt(self.p.max_accel / peak_accel)
        else:
            self._time_warp_scale = 1.0

    def get_position(self, t: float) -> np.ndarray:
        """Returns the 3D position [x, y, z] in NED coordinates at time t."""
        t_clamped = np.clip(t * self._time_warp_scale, 0.0, self.duration)
        return np.asarray(self._spline_pos(t_clamped), dtype=np.float64)

    def get_velocity(self, t: float) -> np.ndarray:
        """Returns the 3D velocity [vx, vy, vz] in NED coordinates at time t."""
        t_clamped = np.clip(t * self._time_warp_scale, 0.0, self.duration)
        return np.asarray(self._spline_vel(t_clamped), dtype=np.float64) * self._time_warp_scale

    def get_acceleration(self, t: float) -> np.ndarray:
        """Returns the 3D acceleration [ax, ay, az] in NED coordinates at time t."""
        t_clamped = np.clip(t * self._time_warp_scale, 0.0, self.duration)
        return np.asarray(self._spline_acc(t_clamped), dtype=np.float64) * (self._time_warp_scale ** 2)

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convenience method returning (position, velocity, acceleration)."""
        return self.get_position(t), self.get_velocity(t), self.get_acceleration(t)

    def __call__(self, t: float) -> np.ndarray:
        """Allows treating the instance directly as a function `f(t) -> [x, y, z]`."""
        return self.get_position(t)

    def sample_full_trajectory(self, dt: float = 0.033) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Samples the entire trajectory densely at time step dt (e.g. 30 Hz).
        
        Returns:
            t: (N,) timestamps
            pos: (N, 3) positions
            vel: (N, 3) velocities
            acc: (N, 3) accelerations
        """
        effective_duration = self.duration / self._time_warp_scale
        t_eval = np.arange(0.0, effective_duration, dt)
        pos = np.array([self.get_position(ti) for ti in t_eval])
        vel = np.array([self.get_velocity(ti) for ti in t_eval])
        acc = np.array([self.get_acceleration(ti) for ti in t_eval])
        return t_eval, pos, vel, acc

    def plot_trajectory(self, save_path: str = "stochastic_trajectory.png", show: bool = False):
        """Generates a comprehensive diagnostic figure of the trajectory."""
        import matplotlib.pyplot as plt

        t, pos, vel, acc = self.sample_full_trajectory(dt=0.033)
        speeds = np.linalg.norm(vel, axis=1)
        acc_norms = np.linalg.norm(acc, axis=1)

        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(
            f"Stochastic Target Trajectory (Profile: '{self.profile_name}', Seed: {self.seed})\n"
            f"Differential Flatness Compliant (C^2 Cubic B-Spline)",
            fontsize=14, fontweight="bold"
        )

        # 1. 3D Trajectory Path
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        ax1.plot(pos[:, 1], pos[:, 0], -pos[:, 2], label="Target Flight Path", color="#1f77b4", lw=2.5)
        ax1.scatter([pos[0, 1]], [pos[0, 0]], [-pos[0, 2]], color="green", s=60, label="Spawn Point")
        ax1.scatter([pos[-1, 1]], [pos[-1, 0]], [-pos[-1, 2]], color="red", s=60, label="End Point")
        ax1.set_xlabel("East (m)")
        ax1.set_ylabel("North (m)")
        ax1.set_zlabel("Altitude (-D in m)")
        ax1.set_title("3D Spatial Flight Path")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 2. 2D Top-Down Path (North vs East)
        ax2 = fig.add_subplot(2, 2, 2)
        scatter = ax2.scatter(pos[:, 1], pos[:, 0], c=speeds, cmap="viridis", s=12, alpha=0.8)
        ax2.plot(pos[:, 1], pos[:, 0], color="gray", alpha=0.4, lw=1)
        ax2.plot(pos[0, 1], pos[0, 0], "go", markersize=8, label="Start")
        ax2.plot(pos[-1, 1], pos[-1, 0], "rs", markersize=8, label="End")
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label("Speed (m/s)")
        ax2.set_xlabel("East (m)")
        ax2.set_ylabel("North (m)")
        ax2.set_title("2D Top-Down Projection (Color = Speed)")
        ax2.axis("equal")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best", fontsize=9)

        # 3. Speed & Velocity Components over Time
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(t, speeds, "k-", lw=2, label="Total Speed ||v||")
        ax3.plot(t, vel[:, 0], "r--", lw=1.2, label="v_North")
        ax3.plot(t, vel[:, 1], "g--", lw=1.2, label="v_East")
        ax3.plot(t, -vel[:, 2], "b--", lw=1.2, label="v_Up (-v_z)")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Velocity (m/s)")
        ax3.set_title("Velocity Profiles")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="best", fontsize=9)

        # 4. Acceleration Compliance over Time
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(t, acc_norms, color="#d62728", lw=2, label="Acceleration ||a||")
        ax4.axhline(self.p.max_accel, color="k", linestyle=":", lw=1.5, label=f"Limit a_max ({self.p.max_accel} m/s²)")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Acceleration (m/s²)")
        ax4.set_title(f"Dynamic Feasibility (Peak Accel: {np.max(acc_norms):.2f} m/s²)")
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc="best", fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        print(f"[SUCCESS] Diagnostic plot saved to: {save_path}")
        if show:
            plt.show()
        plt.close()


# ==============================================================================
# Helper factory functions
# ==============================================================================

def make_stochastic_trajectory(
    profile: str = "evasive",
    duration: float = 25.0,
    seed: Optional[int] = None
) -> StochasticTargetTrajectory:
    """Convenience factory returning a ready-to-use black-box trajectory oracle."""
    return StochasticTargetTrajectory(duration=duration, profile=profile, seed=seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test and visualize stochastic drone trajectories.")
    parser.add_argument("--profile", type=str, default="evasive", choices=["cruising", "evasive", "aerobatic"])
    parser.add_argument("--duration", type=float, default=25.0, help="Duration in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="stochastic_trajectory_demo.png", help="Output plot path")
    args = parser.parse_args()

    print(f"[INIT] Synthesizing stochastic target trajectory: profile='{args.profile}', duration={args.duration}s, seed={args.seed}...")
    traj = StochasticTargetTrajectory(duration=args.duration, profile=args.profile, seed=args.seed)
    
    # Test queries
    p0, v0, a0 = traj.get_state(0.0)
    p_mid, v_mid, a_mid = traj.get_state(args.duration / 2.0)
    print(f"  t = 0.0s:  Pos: {np.round(p0, 2)}, Vel: {np.round(v0, 2)}, Acc: {np.round(a0, 2)}")
    print(f"  t = {args.duration / 2:.1f}s: Pos: {np.round(p_mid, 2)}, Vel: {np.round(v_mid, 2)}, Acc: {np.round(a_mid, 2)}")

    traj.plot_trajectory(save_path=args.out)
