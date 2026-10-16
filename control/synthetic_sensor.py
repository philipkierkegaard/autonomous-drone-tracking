#!/usr/bin/env python3
"""
Synthetic YOLO Perception Sensor for Sim-to-Real Drone Tracking.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception.

Implements the empirical YOLOv8 perception error layer calibrated on unseen validation sequences:
1. Target-size conditioned 2-state Markov chain for detection/miss dropout bursts:
   - Stationary probability matches empirical P(detect | scale).
   - Recovery probability p_recover matches empirical burst duration.
2. Scale-dependent 4D Gaussian measurement error [dcx, dcy, dw, dh]:
   - Empirical mean bias mu(scale)
   - Full 4x4 spatial error covariance matrix Sigma(scale)
3. AR(1) temporal colored noise autocorrelation:
   - Preserves inter-frame tracking memory: rho_cx=+0.657, rho_cy=+0.651, rho_w=+0.875, rho_h=+0.866
4. Clutter & False Positive Modeling:
   - Environment-conditioned Poisson false alarm rate (Clean Sky vs Tree Canopy Clutter).
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union

import numpy as np


class SyntheticYOLOSensor:
    """
    Simulated YOLOv8 detector operating on ground-truth pinhole projections.
    Outputs realistic bounding box detections or dropouts at camera framerate (30 Hz).
    """

    def __init__(
        self,
        profile_path: Optional[Union[str, Path]] = None,
        img_w: int = 640,
        img_h: int = 480,
        env_mode: str = "nominal",  # "nominal" (clean sky), "cluttered" (tree canopy), "average", "none"
        seed: Optional[int] = None
    ):
        self.img_w = img_w
        self.img_h = img_h
        self.env_mode = env_mode
        self.rng = np.random.default_rng(seed)

        # Locate and load empirical perception model JSON
        if profile_path is None:
            default_path = Path(__file__).resolve().parent.parent / "perception" / "preparation" / "empirical_perception_model.json"
            if not default_path.exists():
                # Fallback to local path relative to current working directory
                default_path = Path("perception/preparation/empirical_perception_model.json")
            self.profile_path = default_path
        else:
            self.profile_path = Path(profile_path)

        self.load_profile(self.profile_path)
        self.reset()

    def load_profile(self, path: Path):
        """Loads and precomputes empirical statistical parameters from JSON."""
        if not path.exists():
            raise FileNotFoundError(f"Empirical perception model not found at {path.resolve()}")

        with open(path, "r") as f:
            self.profile = json.load(f)

        # 1. Scale Bins: extract boundaries, P(detect), mean residuals, and covariance matrices
        self.scale_bins = {}
        for b_name, b_data in self.profile["scale_bins"].items():
            mu = np.array(b_data["mean_residual"], dtype=np.float64)
            cov = np.array(b_data["cov_matrix_4x4"], dtype=np.float64)
            # Ensure covariance matrix is symmetric positive semi-definite
            cov = (cov + cov.T) / 2.0
            min_eig = np.min(np.linalg.eigvalsh(cov))
            if min_eig < 1e-8:
                cov += np.eye(4) * (1e-8 - min_eig)

            self.scale_bins[b_name] = {
                "p_detect": float(b_data["p_detect"]),
                "mu": mu,
                "cov": cov,
            }

        # 2. Markov Temporal Transition Parameters
        markov_data = self.profile["temporal_markov_model"]
        self.base_p_drop = float(markov_data["p_drop"])
        self.p_recover = float(markov_data["p_recover"])

        # 3. Lag-1 Autocorrelation Coefficients (AR(1) colored noise)
        lag1_data = markov_data["lag1_autocorrelation"]
        rho_cx = float(lag1_data.get("rho_cx", 0.6572))
        rho_cy = float(lag1_data.get("rho_cy", 0.6506))
        rho_w  = float(lag1_data.get("rho_w", 0.8748))
        rho_h  = float(lag1_data.get("rho_h", 0.8661))

        self.phi = np.array([rho_cx, rho_cy, rho_w, rho_h], dtype=np.float64)
        # Innovation gain matrix for AR(1) process: e_t = phi * e_{t-1} + sqrt(1 - phi^2) * eta_t
        self.ar1_gain = np.sqrt(np.maximum(1e-4, 1.0 - self.phi**2))

        # 4. False Alarm (Clutter) Rates (per frame at 30 Hz)
        meta = self.profile["metadata"]
        self.fp_rates = {
            "nominal": float(meta.get("clean_sky_fp_rate_per_frame", 0.0257)),
            "cluttered": float(meta.get("tree_canopy_fp_rate_per_frame", 1.5121)),
            "average": float(meta.get("global_fp_rate_per_frame", 0.3057)),
            "none": 0.0
        }

    def reset(self, seed: Optional[int] = None):
        """Resets the sensor internal Markov state and AR(1) colored residual memory."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Markov state: 1 = Seen (detected), 0 = Dropped (missed)
        self.state = 1
        # Colored noise residual vector [dcx, dcy, dw, dh] in normalized coordinates
        self.residual = np.zeros(4, dtype=np.float64)
        self.step_count = 0
        self.consecutive_misses = 0

    def _get_scale_bin(self, scale_norm: float) -> str:
        """Classifies target scale max(w_norm, h_norm) into empirical bins."""
        if scale_norm < 0.035:
            return "tiny"      # < 22 px on 640px sensor
        elif scale_norm < 0.070:
            return "small"     # 22 - 45 px
        elif scale_norm < 0.120:
            return "medium"    # 45 - 77 px
        else:
            return "large"     # > 77 px

    def step(
        self,
        gt_cx: float,
        gt_cy: float,
        gt_w: float,
        gt_h: float,
        in_fov: bool
    ) -> Tuple[Optional[np.ndarray], float, List[np.ndarray]]:
        """
        Simulates one camera frame observation (30 Hz).

        Args:
            gt_cx: Normalized ground-truth target horizontal center [0, 1]
            gt_cy: Normalized ground-truth target vertical center [0, 1]
            gt_w:  Normalized ground-truth target width [0, 1]
            gt_h:  Normalized ground-truth target height [0, 1]
            in_fov: True if target is physically within the optical frustum

        Returns:
            primary_detection: [x1, y1, x2, y2] pixel coordinates, or None if dropped/out-of-view
            confidence: detection confidence score [0.0, 1.0]
            false_positives: list of [x1, y1, x2, y2] pixel boxes for false clutter detections
        """
        self.step_count += 1
        false_positives = []

        # If target is completely outside physical camera field of view
        if not in_fov or gt_w <= 1e-4 or gt_h <= 1e-4:
            self.state = 0
            self.consecutive_misses += 1
            # Decay colored noise towards zero during extended absence
            self.residual = self.phi * self.residual
            fps = self._generate_false_positives()
            return None, 0.0, fps

        # 1. Target Scale & Scale-Conditioned Markov Transition
        target_scale = max(gt_w, gt_h)
        b_name = self._get_scale_bin(target_scale)
        b_stats = self.scale_bins[b_name]
        p_det = b_stats["p_detect"]

        # Condition p_drop on target size so steady-state detection prob equals P(detect | scale):
        # pi_seen = p_recover / (p_drop(scale) + p_recover) = p_det
        # => p_drop(scale) = p_recover * (1 - p_det) / p_det
        if p_det >= 0.999:
            p_drop_scale = 0.0
        elif p_det <= 0.01:
            p_drop_scale = 0.95
        else:
            p_drop_scale = self.p_recover * (1.0 - p_det) / p_det
            p_drop_scale = float(np.clip(p_drop_scale, 0.0, 0.90))

        # 2-State Markov Chain State Update
        if self.state == 1:
            # Currently seen: check if target is dropped in this frame
            if self.rng.random() < p_drop_scale:
                self.state = 0
                self.consecutive_misses = 1
            else:
                self.state = 1
                self.consecutive_misses = 0
        else:
            # Currently dropped: check if target is recovered in this frame
            if self.rng.random() < self.p_recover:
                self.state = 1
                self.consecutive_misses = 0
            else:
                self.state = 0
                self.consecutive_misses += 1

        # 2. Measurement Perturbation or Dropout Output
        if self.state == 1:
            # Sample spatial error from empirical 4D covariance
            mu = b_stats["mu"]
            cov = b_stats["cov"]
            spatial_noise = self.rng.multivariate_normal(mu, cov)

            # Apply AR(1) colored noise temporal autocorrelation:
            # residual_t = phi * residual_{t-1} + sqrt(1 - phi^2) * spatial_noise
            self.residual = self.phi * self.residual + self.ar1_gain * spatial_noise

            # Apply residual to normalized ground truth
            noisy_cx = gt_cx + self.residual[0]
            noisy_cy = gt_cy + self.residual[1]
            noisy_w  = max(0.005, gt_w + self.residual[2])
            noisy_h  = max(0.005, gt_h + self.residual[3])

            # Convert to pixel bounding box [x1, y1, x2, y2]
            px_cx = noisy_cx * self.img_w
            px_cy = noisy_cy * self.img_h
            px_w  = noisy_w * self.img_w
            px_h  = noisy_h * self.img_h

            x1 = float(np.clip(px_cx - px_w / 2.0, 0.0, self.img_w - 1.0))
            y1 = float(np.clip(px_cy - px_h / 2.0, 0.0, self.img_h - 1.0))
            x2 = float(np.clip(px_cx + px_w / 2.0, 1.0, self.img_w))
            y2 = float(np.clip(px_cy + px_h / 2.0, 1.0, self.img_h))

            # Calibrated confidence score (higher for large targets, lower near detection boundary)
            base_conf = 0.70 + 0.20 * min(1.0, target_scale / 0.10)
            conf = float(np.clip(base_conf + self.rng.normal(0.0, 0.06), 0.25, 0.98))

            primary_detection = np.array([x1, y1, x2, y2], dtype=np.float32)
        else:
            # Missed detection (Kalman tracker must coast)
            primary_detection = None
            conf = 0.0
            # Decay residual
            self.residual = self.phi * self.residual

        # 3. Generate False Positive Clutter Detections
        false_positives = self._generate_false_positives()

        return primary_detection, conf, false_positives

    def _generate_false_positives(self) -> List[np.ndarray]:
        """Generates Poisson-distributed false alarms matching empirical clutter properties."""
        lambda_fp = self.fp_rates.get(self.env_mode, 0.0)
        if lambda_fp <= 0.0:
            return []

        n_fp = self.rng.poisson(lambda_fp)
        fps = []
        for _ in range(n_fp):
            # Empirical false positives: small square patches (~15px) scattered across scene
            fp_w = float(self.rng.uniform(10.0, 22.0))
            fp_h = float(self.rng.uniform(10.0, 22.0))
            fp_x = float(self.rng.uniform(0.0, self.img_w - fp_w))
            fp_y = float(self.rng.uniform(0.0, self.img_h - fp_h))
            fps.append(np.array([fp_x, fp_y, fp_x + fp_w, fp_y + fp_h], dtype=np.float32))

        return fps
