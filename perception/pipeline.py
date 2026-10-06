"""
Autonomous Drone Tracker: Pure Detection + Kalman Filter Pipeline
Combines fine-tuned YOLOv8 drone detector + 8D State Kalman Filter for real-time visual tracking and servoing.

Outputs visual servoing flight telemetry (normalized horizontal/vertical errors, velocity derivatives,
and distance proxies) directly ready for PX4 / MAVLink / ROS 2 offboard flight control loops.
"""

import os
import glob
import time
from collections import deque
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

try:
    from control.spatial import compute_ego_motion_compensation
except ImportError:
    try:
        from spatial import compute_ego_motion_compensation
    except ImportError:
        compute_ego_motion_compensation = None


# ==============================================================================
# 1. 8D State Kalman Filter for Bounding Box Tracking
# ==============================================================================

class KalmanBoxTracker:
    """
    Kalman Filter for tracking bounding boxes of flying drones in 2D image coordinates.
    State vector: [cx, cy, w, h, vx, vy, vw, vh]^T
    Measurement vector: [cx, cy, w, h]^T
    """
    _count = 0

    def __init__(
        self,
        bbox: np.ndarray,
        conf: float = 1.0,
        dt: float = 1.0 / 30.0,
        q_pos: float = 1.0,
        q_vel: float = 30.0,
        r_pos: float = 2.0
    ):
        """
        Initialize tracker from initial bounding box [x1, y1, x2, y2].
        """
        KalmanBoxTracker._count += 1
        self.id = KalmanBoxTracker._count
        self.dt = dt

        # State transition matrix F (constant velocity model)
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = self.dt

        # Measurement matrix H
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

        # Process noise covariance Q (tuned for agile quadcopter dynamics)
        self.Q = np.eye(8, dtype=np.float32)
        self.Q[:2, :2] *= q_pos   # Position noise
        self.Q[2:4, 2:4] *= 2.0   # Dimension noise (scale change when approaching/retreating)
        self.Q[4:6, 4:6] *= q_vel # Velocity noise (maneuver accelerations)
        self.Q[6:8, 6:8] *= 10.0  # Scale rate noise

        # Measurement noise covariance R (YOLO bounding box jitter)
        self.R = np.eye(4, dtype=np.float32)
        self.R[:2, :2] *= r_pos   # Center detection precision
        self.R[2:4, 2:4] *= 4.0   # Width/height precision

        # State covariance matrix P
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0   # High uncertainty on initial velocity

        # Convert [x1, y1, x2, y2] to [cx, cy, w, h]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])

        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32).reshape(8, 1)

        self.conf = conf
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.history = deque(maxlen=40)  # Stores trajectory history of (cx, cy)
        self.history.append((float(cx), float(cy)))

    def predict(self, ego_shift: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
        """
        Advance state forward by dt with optional ego-motion compensation.
        ego_shift: (delta_cx, delta_cy, delta_w, delta_h) shift induced by drone movement
        Returns predicted [x1, y1, x2, y2].
        """
        # x = F * x
        self.x = np.dot(self.F, self.x)

        # Apply deterministic ego-motion control shift
        if ego_shift is not None:
            dcx, dcy, dw, dh = ego_shift
            self.x[0, 0] += dcx
            self.x[1, 0] += dcy
            self.x[2, 0] = max(1.0, self.x[2, 0] + dw)
            self.x[3, 0] = max(1.0, self.x[3, 0] + dh)

        # P = F * P * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        self.age += 1
        self.time_since_update += 1

        cx, cy, w, h = self.x[0, 0], self.x[1, 0], max(1.0, self.x[2, 0]), max(1.0, self.x[3, 0])
        self.history.append((float(cx), float(cy)))
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)

    def update(self, bbox: np.ndarray, conf: float = 1.0):
        """
        Measurement update using detected YOLO box [x1, y1, x2, y2].
        """
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        z = np.array([cx, cy, w, h], dtype=np.float32).reshape(4, 1)

        # Innovation: y = z - H * x
        y = z - np.dot(self.H, self.x)

        # Innovation covariance: S = H * P * H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Kalman gain: K = P * H^T * inv(S)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # State update: x = x + K * y
        self.x = self.x + np.dot(K, y)

        # Covariance update: P = (I - K * H) * P
        I = np.eye(8, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

        self.conf = conf
        self.hits += 1
        self.time_since_update = 0

    def get_state(self) -> np.ndarray:
        """
        Returns current bounding box [x1, y1, x2, y2].
        """
        cx, cy, w, h = self.x[0, 0], self.x[1, 0], max(1.0, self.x[2, 0]), max(1.0, self.x[3, 0])
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)

    def get_velocity(self) -> Tuple[float, float]:
        """
        Returns estimated velocity (vx, vy) in pixels/sec.
        """
        return float(self.x[4, 0] / self.dt), float(self.x[5, 0] / self.dt)


# ==============================================================================
# 2. Association & Tracking Manager
# ==============================================================================

def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Computes IoU cost matrix (1.0 - IoU) between two sets of bounding boxes.
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.empty((len(boxes1), len(boxes2)), dtype=np.float32)

    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_x1 = np.maximum(b1_x1[:, None], b2_x1[None, :])
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1[None, :])
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2[None, :])
    inter_y2 = np.minimum(b1_y2[:, None], b2_y2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area[:, None] + b2_area[None, :] - inter_area

    iou = inter_area / np.maximum(1e-6, union_area)
    return 1.0 - iou  # Cost matrix (lower is better)


class DroneTracker:
    """
    Multi-Track Manager with Kalman Filtering, track lifecycle, and target lock-on.
    """
    def __init__(
        self,
        max_lost_frames: int = 15,
        min_hits_to_confirm: int = 2,
        iou_match_threshold: float = 0.75,  # Max cost (IoU >= 0.25)
        dt: float = 1.0 / 30.0,
        q_pos: float = 1.0,
        q_vel: float = 18.0,
        r_pos: float = 2.0
    ):
        self.max_lost_frames = max_lost_frames
        self.min_hits_to_confirm = min_hits_to_confirm
        self.iou_match_threshold = iou_match_threshold
        self.dt = dt
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r_pos = r_pos
        self.trackers: List[KalmanBoxTracker] = []
        self.primary_track_id: Optional[int] = None

    def update(
        self,
        detections: np.ndarray,
        confs: np.ndarray,
        ego_shifts: Optional[Dict[int, Tuple[float, float, float, float]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Updates the tracker with new detections in the current frame.
        detections: array of [x1, y1, x2, y2]
        confs: array of detection confidences
        ego_shifts: optional dict of {track_id: (dcx, dcy, dw, dh)} for ego-motion compensation
        """
        # 1. Predict new positions for all active trackers (incorporating ego-motion)
        predicted_boxes = []
        for trk in self.trackers:
            shift = ego_shifts.get(trk.id, None) if ego_shifts is not None else None
            predicted_boxes.append(trk.predict(ego_shift=shift))
        predicted_boxes = np.array(predicted_boxes) if len(predicted_boxes) > 0 else np.empty((0, 4))

        # 2. Hungarian matching between predictions and detections
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(self.trackers)))

        if len(predicted_boxes) > 0 and len(detections) > 0:
            cost_matrix = compute_iou_matrix(predicted_boxes, detections)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= self.iou_match_threshold:
                    matched_indices.append((r, c))
                    if c in unmatched_detections:
                        unmatched_detections.remove(c)
                    if r in unmatched_trackers:
                        unmatched_trackers.remove(r)
                else:
                    # Fallback: Euclidean center distance gating for fast agile maneuvers
                    pred_center = (predicted_boxes[r, :2] + predicted_boxes[r, 2:]) / 2.0
                    det_center = (detections[c, :2] + detections[c, 2:]) / 2.0
                    pred_diag = np.linalg.norm(predicted_boxes[r, 2:] - predicted_boxes[r, :2])
                    dist = np.linalg.norm(pred_center - det_center)

                    if dist < max(80.0, 2.5 * pred_diag):
                        matched_indices.append((r, c))
                        if c in unmatched_detections:
                            unmatched_detections.remove(c)
                        if r in unmatched_trackers:
                            unmatched_trackers.remove(r)

        # 3. Update matched trackers
        for trk_idx, det_idx in matched_indices:
            self.trackers[trk_idx].update(detections[det_idx], confs[det_idx])

        # 4. Create new tentative trackers for unmatched detections
        for det_idx in unmatched_detections:
            new_tracker = KalmanBoxTracker(
                detections[det_idx], confs[det_idx], dt=self.dt,
                q_pos=self.q_pos, q_vel=self.q_vel, r_pos=self.r_pos
            )
            self.trackers.append(new_tracker)

        # 5. Clean up expired dead trackers
        surviving_trackers = []
        for trk in self.trackers:
            if trk.time_since_update <= self.max_lost_frames:
                surviving_trackers.append(trk)
            elif self.primary_track_id == trk.id:
                # Primary locked target expired
                self.primary_track_id = None
        self.trackers = surviving_trackers

        # 6. Build active track status records
        active_tracks = []
        for trk in self.trackers:
            bbox = trk.get_state()
            vx, vy = trk.get_velocity()
            is_confirmed = trk.hits >= self.min_hits_to_confirm
            is_coasting = trk.time_since_update > 0

            active_tracks.append({
                "id": trk.id,
                "bbox": bbox,
                "confidence": trk.conf,
                "velocity": (vx, vy),
                "is_confirmed": is_confirmed,
                "is_coasting": is_coasting,
                "time_coasting": trk.time_since_update,
                "trajectory": list(trk.history),
                "hits": trk.hits
            })

        # 7. Maintain or assign primary lock target
        self._update_primary_lock(active_tracks)

        return active_tracks

    def _update_primary_lock(self, active_tracks: List[Dict[str, Any]]):
        """
        Locks onto the primary target (persists active target ID, or acquires highest-confidence target).
        """
        active_ids = [t["id"] for t in active_tracks if t["is_confirmed"]]

        if self.primary_track_id is not None:
            if self.primary_track_id not in active_ids:
                # Target lost
                self.primary_track_id = None

        if self.primary_track_id is None and len(active_ids) > 0:
            # Acquire target with highest confidence / most hits
            best_trk = max(
                [t for t in active_tracks if t["is_confirmed"]],
                key=lambda x: (x["hits"], x["confidence"])
            )
            self.primary_track_id = best_trk["id"]


# ==============================================================================
# 3. End-to-End Pipeline & Aeronautical HUD Renderer
# ==============================================================================

class DroneTrackingPipeline:
    """
    End-to-End Autonomous Drone Tracking Pipeline for Edge Jetson Devices.
    Combines YOLOv8 drone detector + 8D Kalman filter with Dynamic Digital PTZ (Foveal Zoom).
    
    When a target retreats or shrinks, the pipeline dynamically crops native high-resolution 
    sensor pixels around the Kalman-predicted center, enabling 2x-3x higher visual acuity 
    without changing the neural network input size or inducing coordinate jumps.
    """
    def __init__(
        self,
        weights: str = "runs/detect/yolov8n_drone/weights/best.pt",
        conf_threshold: float = 0.20,
        iou_threshold: float = 0.45,
        max_lost_frames: int = 15,
        min_hits_to_confirm: int = 2,
        enable_dynamic_zoom: bool = True,
        max_zoom: float = 2.5,
        desired_target_size: float = 140.0,
        view_mode: str = "zoom_pov",
        q_pos: float = 1.0,
        q_vel: float = 18.0,
        r_pos: float = 2.0,
        enable_kalman: bool = True,
        device: Optional[str] = None
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.enable_kalman = enable_kalman
        self.raw_prev_target = None
        self.raw_trajectory = deque(maxlen=40)

        # Constant Apparent Target Size Regulation parameters
        self.enable_dynamic_zoom = enable_dynamic_zoom if self.enable_kalman else False
        self.max_zoom = max_zoom
        self.desired_target_size = desired_target_size  # Target reference pixel size on screen
        self.view_mode = view_mode  # "zoom_pov" (True Zoom POV) or "wide_pip" (Wide Angle + PiP)
        self.current_zoom = 1.0
        self.target_zoom = 1.0
        self.smooth_crop_center: Optional[np.ndarray] = None
        self.active_crop_box: Optional[Tuple[int, int, int, int]] = None  # (xmin, ymin, xmax, ymax) in global sensor space
        self.zoom_coasting_counter = 0

        # Resolve weights path
        weights_path = Path(weights)
        if not weights_path.exists():
            # Fallback checks
            alternatives = [
                Path("runs/detect/runs/detect/yolov8n_drone-2/weights/best.pt"),
                Path("runs/detect/yolov8n_drone-3/weights/best.pt"),
                Path("yolov8n.pt")
            ]
            for alt in alternatives:
                if alt.exists():
                    weights_path = alt
                    break

        print(f"[INIT] Loading YOLOv8 drone detector: {weights_path}")
        self.model = YOLO(str(weights_path))

        if self.enable_kalman:
            print(f"[INIT] Initializing Kalman Multi-Track Manager (max_lost={max_lost_frames} frames, Q_pos={q_pos}, Q_vel={q_vel}, R_pos={r_pos})...")
        else:
            print("[INIT] Kalman Filter: DISABLED (Pure Raw YOLO Detections Mode)")

        if self.enable_dynamic_zoom:
            print(f"[INIT] Constant Size Regulation PTZ ENABLED (Target Size: {self.desired_target_size:.0f}px, Max Zoom: {self.max_zoom:.1f}x, View: {self.view_mode})")
        else:
            print("[INIT] Dynamic Digital PTZ: DISABLED (Full Frame Mode)")

        self.tracker = DroneTracker(
            max_lost_frames=max_lost_frames,
            min_hits_to_confirm=min_hits_to_confirm,
            q_pos=q_pos,
            q_vel=q_vel,
            r_pos=r_pos
        )

        self.frame_idx = 0
        self.fps_history = deque(maxlen=20)
        self.prev_time = None
        self.prev_ego_telemetry: Optional[Dict[str, float]] = None

    def process_frame(
        self,
        frame: np.ndarray,
        draw_hud: bool = True,
        ego_telemetry: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes a single BGR video frame:
        1. Determines Foveal Digital Zoom based on Constant Apparent Size Regulation
        2. Runs YOLOv8 inference (on high-res crop if zoomed, or full frame)
        3. Maps crop detections back to Global Sensor Coordinates
        4. Updates 8D Kalman filter in continuous global space (with optional ego-motion compensation)
        5. Extracts visual servoing telemetry errors (e_x, e_y, velocities, range/size error)
        6. Renders aeronautical HUD overlay in True Zoom POV or Wide FOV
        """
        t_start = time.perf_counter()
        h, w = frame.shape[:2]
        img_center = (w / 2.0, h / 2.0)

        # ----------------------------------------------------------------------
        # 0. Drone Ego-Motion Compensation (Rotation + Translation Decoupling)
        # ----------------------------------------------------------------------
        ego_shifts: Dict[int, Tuple[float, float, float, float]] = {}
        target_pred_shift = (0.0, 0.0, 0.0, 0.0)

        if ego_telemetry is not None and compute_ego_motion_compensation is not None:
            curr_pitch = float(ego_telemetry.get("pitch_rad", 0.0))
            curr_roll = float(ego_telemetry.get("roll_rad", 0.0))
            curr_yaw_rad = float(np.deg2rad(ego_telemetry.get("yaw_deg", 0.0)))
            vx_body = float(ego_telemetry.get("vx_body", 0.0))
            vy_body = float(ego_telemetry.get("vy_body", 0.0))
            vz_body = float(ego_telemetry.get("vz_body", 0.0))
            ego_dt = float(ego_telemetry.get("dt", self.tracker.dt))
            est_dist = float(ego_telemetry.get("distance_m", ego_telemetry.get("altitude_m", 6.0)))

            if self.prev_ego_telemetry is not None:
                d_pitch = curr_pitch - self.prev_ego_telemetry["pitch_rad"]
                d_roll = curr_roll - self.prev_ego_telemetry["roll_rad"]
                d_yaw = curr_yaw_rad - self.prev_ego_telemetry["yaw_rad"]
                d_yaw = float((d_yaw + np.pi) % (2.0 * np.pi) - np.pi)

                for trk in self.tracker.trackers:
                    box = trk.get_state()
                    tcx = (box[0] + box[2]) / 2.0
                    tcy = (box[1] + box[3]) / 2.0
                    tw = max(1.0, box[2] - box[0])
                    th = max(1.0, box[3] - box[1])

                    shift = compute_ego_motion_compensation(
                        cx=tcx, cy=tcy, w=tw, h=th,
                        delta_pitch_rad=d_pitch,
                        delta_roll_rad=d_roll,
                        delta_yaw_rad=d_yaw,
                        vx_body=vx_body,
                        vy_body=vy_body,
                        vz_body=vz_body,
                        dt=ego_dt,
                        estimated_distance_m=est_dist,
                        img_width=float(w),
                        img_height=float(h)
                    )
                    ego_shifts[trk.id] = shift

                if self.tracker.primary_track_id in ego_shifts:
                    target_pred_shift = ego_shifts[self.tracker.primary_track_id]
                elif len(ego_shifts) > 0:
                    target_pred_shift = next(iter(ego_shifts.values()))

            self.prev_ego_telemetry = {
                "pitch_rad": curr_pitch,
                "roll_rad": curr_roll,
                "yaw_rad": curr_yaw_rad
            }

        # ----------------------------------------------------------------------
        # 1. Active Foveal Zoom Decision (Constant Apparent Target Size Regulation)
        # ----------------------------------------------------------------------
        crop_xmin, crop_ymin = 0, 0
        crop_xmax, crop_ymax = w, h
        target_pred_center = None
        primary_trk = None
        current_target_size = 0.0

        if self.tracker.primary_track_id is not None:
            for trk in self.tracker.trackers:
                if trk.id == self.tracker.primary_track_id:
                    primary_trk = trk
                    pred_box = trk.get_state()
                    tcx = (pred_box[0] + pred_box[2]) / 2.0 + target_pred_shift[0]
                    tcy = (pred_box[1] + pred_box[3]) / 2.0 + target_pred_shift[1]
                    target_pred_center = (tcx, tcy)
                    current_target_size = max(pred_box[2] - pred_box[0], pred_box[3] - pred_box[1])

                    if self.enable_dynamic_zoom:
                        # Closed-Loop Target Framing Regulation:
                        # Automatically adjusts zoom so target stays at desired_target_size (e.g. 140px)
                        zoom_calc = self.desired_target_size / max(10.0, current_target_size)
                        self.target_zoom = float(np.clip(zoom_calc, 1.0, self.max_zoom))
                    break

        # Failsafe: if target is lost/coasting for >= 3 frames while zoomed, snap back to wide FOV
        if primary_trk is None or (primary_trk.time_since_update >= 3):
            self.target_zoom = 1.0
            if self.current_zoom > 1.2:
                self.current_zoom = 1.0  # Instant wide FOV search fallback
            self.smooth_crop_center = None

        # Smooth zoom transition (exponential moving average for cinematic autofocus feel)
        if self.enable_dynamic_zoom:
            self.current_zoom = 0.75 * self.current_zoom + 0.25 * self.target_zoom
            if self.current_zoom < 1.08:
                self.current_zoom = 1.0
        else:
            self.current_zoom = 1.0

        # ----------------------------------------------------------------------
        # 2. Extract Native High-Res Crop & Run YOLOv8
        # ----------------------------------------------------------------------
        is_zoomed = (self.current_zoom > 1.1) and (target_pred_center is not None)

        if is_zoomed:
            # Calculate crop dimensions in native sensor pixels
            crop_w = int(w / self.current_zoom)
            crop_h = int(h / self.current_zoom)

            tcx, tcy = target_pred_center
            if self.smooth_crop_center is None:
                self.smooth_crop_center = np.array([tcx, tcy], dtype=np.float32)
            else:
                # Damped tracking for smooth POV camera motion without jitter
                target_pt = np.array([tcx, tcy], dtype=np.float32)
                self.smooth_crop_center = 0.70 * self.smooth_crop_center + 0.30 * target_pt

                # Keep target safely within central 60% of crop
                margin_x = 0.30 * crop_w
                margin_y = 0.30 * crop_h
                self.smooth_crop_center[0] = float(np.clip(self.smooth_crop_center[0], tcx - margin_x, tcx + margin_x))
                self.smooth_crop_center[1] = float(np.clip(self.smooth_crop_center[1], tcy - margin_y, tcy + margin_y))

            ccx, ccy = self.smooth_crop_center
            crop_xmin = max(0, min(w - crop_w, int(ccx - crop_w / 2.0)))
            crop_ymin = max(0, min(h - crop_h, int(ccy - crop_h / 2.0)))
            crop_xmax = crop_xmin + crop_w
            crop_ymax = crop_ymin + crop_h

            self.active_crop_box = (crop_xmin, crop_ymin, crop_xmax, crop_ymax)
            inference_input = frame[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        else:
            self.active_crop_box = None
            self.smooth_crop_center = None
            inference_input = frame

        # Run YOLOv8 inference (on crop or full frame)
        results = self.model.predict(
            source=inference_input,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        detections = []
        confs = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            conf_vals = results[0].boxes.conf.cpu().numpy()
            for box, c in zip(boxes, conf_vals):
                # Map detection from crop coordinates back to Global Sensor Coordinates
                global_x1 = crop_xmin + box[0]
                global_y1 = crop_ymin + box[1]
                global_x2 = crop_xmin + box[2]
                global_y2 = crop_ymin + box[3]
                detections.append([global_x1, global_y1, global_x2, global_y2])
                confs.append(float(c))

        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 4))
        confs = np.array(confs) if len(confs) > 0 else np.empty((0,))

        # ----------------------------------------------------------------------
        # 3. Update 8D Kalman Filter in Global Sensor Coordinates (or Pure Raw Mode)
        # ----------------------------------------------------------------------
        if self.enable_kalman:
            active_tracks = self.tracker.update(detections, confs, ego_shifts=ego_shifts)
            primary_track = None
            for trk in active_tracks:
                if trk["id"] == self.tracker.primary_track_id:
                    primary_track = trk
                    break
        else:
            active_tracks = []
            primary_track = None
            if len(detections) > 0:
                if self.raw_prev_target is not None:
                    centers = (detections[:, :2] + detections[:, 2:]) / 2.0
                    dists = np.linalg.norm(centers - self.raw_prev_target, axis=1)
                    best_idx = int(np.argmin(dists))
                else:
                    best_idx = int(np.argmax(confs))
                
                best_box = detections[best_idx]
                best_conf = confs[best_idx]
                tcx = (best_box[0] + best_box[2]) / 2.0
                tcy = (best_box[1] + best_box[3]) / 2.0
                
                if self.raw_prev_target is not None:
                    vx = float((tcx - self.raw_prev_target[0]) * 30.0)
                    vy = float((tcy - self.raw_prev_target[1]) * 30.0)
                else:
                    vx, vy = 0.0, 0.0
                
                self.raw_prev_target = np.array([tcx, tcy], dtype=np.float32)
                self.raw_trajectory.append((float(tcx), float(tcy)))
                primary_track = {
                    "id": 1,
                    "bbox": best_box,
                    "confidence": float(best_conf),
                    "velocity": (vx, vy),
                    "is_confirmed": True,
                    "is_coasting": False,
                    "time_coasting": 0,
                    "trajectory": list(self.raw_trajectory),
                    "hits": 1
                }
                active_tracks = [primary_track]
            else:
                self.raw_prev_target = None
                self.raw_trajectory.clear()

        # ----------------------------------------------------------------------
        # 4. Extract Visual Servoing Flight Control Telemetry
        # ----------------------------------------------------------------------
        telemetry = {
            "frame_idx": self.frame_idx,
            "status": "SEARCHING",    # "LOCKED", "COASTING", "SEARCHING"
            "target_id": None,
            "bbox": None,
            "center": None,
            "error_x": 0.0,           # Normalized [-1.0 (left), +1.0 (right)]
            "error_y": 0.0,           # Normalized [-1.0 (up), +1.0 (down)]
            "error_range": 0.0,       # Normalized scale/range error [-1.0 close, +1.0 far/retreating]
            "target_size": 0.0,       # px (raw sensor size)
            "desired_size": float(self.desired_target_size),
            "velocity_x": 0.0,        # px/sec (Kalman derivative)
            "velocity_y": 0.0,        # px/sec (Kalman derivative)
            "area_ratio": 0.0,        # bbox_area / image_area (target distance proxy)
            "zoom_level": float(self.current_zoom),
            "is_zoomed": is_zoomed,
            "view_mode": self.view_mode,
            "active_tracks_count": len(active_tracks),
            "ego_compensated": bool(len(ego_shifts) > 0),
            "ego_shift_x": float(target_pred_shift[0]),
            "ego_shift_y": float(target_pred_shift[1]),
            "fps": 0.0
        }

        if self.enable_kalman:
            primary_track = None
            for trk in active_tracks:
                if trk["id"] == self.tracker.primary_track_id:
                    primary_track = trk
                    break
        else:
            primary_track = active_tracks[0] if len(active_tracks) > 0 else None

        if primary_track is not None:
            bbox = primary_track["bbox"]
            tcx = (bbox[0] + bbox[2]) / 2.0
            tcy = (bbox[1] + bbox[3]) / 2.0
            bw = max(1.0, bbox[2] - bbox[0])
            bh = max(1.0, bbox[3] - bbox[1])
            current_sz = max(bw, bh)

            # Normalized error relative to true optical camera center (invariant to zoom!)
            err_x = (tcx - img_center[0]) / (w / 2.0)
            err_y = (tcy - img_center[1]) / (h / 2.0)
            range_err = (self.desired_target_size - current_sz) / max(1.0, self.desired_target_size)
            area_ratio = (bw * bh) / float(w * h)
            vx, vy = primary_track["velocity"]

            telemetry.update({
                "status": "COASTING" if primary_track["is_coasting"] else "LOCKED",
                "target_id": primary_track["id"],
                "bbox": [float(b) for b in bbox],
                "center": (float(tcx), float(tcy)),
                "error_x": float(np.clip(err_x, -1.0, 1.0)),
                "error_y": float(np.clip(err_y, -1.0, 1.0)),
                "error_range": float(np.clip(range_err, -1.0, 1.0)),
                "target_size": float(current_sz),
                "desired_size": float(self.desired_target_size),
                "velocity_x": vx,
                "velocity_y": vy,
                "area_ratio": float(area_ratio)
            })

        # Calculate FPS
        t_elapsed = time.perf_counter() - t_start
        fps = 1.0 / max(1e-4, t_elapsed)
        self.fps_history.append(fps)
        telemetry["fps"] = float(np.mean(self.fps_history))
        self.frame_idx += 1

        # ----------------------------------------------------------------------
        # 5. Render Aeronautical HUD (True Zoom POV or Wide View)
        # ----------------------------------------------------------------------
        if draw_hud:
            annotated = self._render_hud(frame, active_tracks, telemetry)
        else:
            if self.view_mode == "zoom_pov" and is_zoomed and self.active_crop_box is not None:
                zx1, zy1, zx2, zy2 = self.active_crop_box
                annotated = cv2.resize(frame[zy1:zy2, zx1:zx2], (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                annotated = frame.copy()

        return annotated, telemetry

    def _render_hud(
        self,
        raw_frame: np.ndarray,
        active_tracks: List[Dict[str, Any]],
        telemetry: Dict[str, Any]
    ) -> np.ndarray:
        """
        Renders an aerospace-grade HUD overlay.
        In 'zoom_pov' mode:
          - Output video dynamically crops and scales to the target (True Zoom POV)
          - Corner minimap displays Wide Sky Radar overview with active viewing cone
        In 'wide_pip' mode:
          - Output video remains wide-angle with yellow framing box and corner PiP inset
        """
        h, w = raw_frame.shape[:2]
        cx, cy = int(w / 2), int(h / 2)
        is_zoomed = bool(telemetry.get("is_zoomed") and self.active_crop_box is not None)

        if self.view_mode == "zoom_pov" and is_zoomed:
            zx1, zy1, zx2, zy2 = self.active_crop_box
            crop_roi = raw_frame[zy1:zy2, zx1:zx2]
            frame = cv2.resize(crop_roi, (w, h), interpolation=cv2.INTER_LINEAR)

            scale_x = w / float(max(1, zx2 - zx1))
            scale_y = h / float(max(1, zy2 - zy1))

            def to_screen(gx: float, gy: float) -> Tuple[int, int]:
                sx = (gx - zx1) * scale_x
                sy = (gy - zy1) * scale_y
                return int(round(sx)), int(round(sy))
        else:
            frame = raw_frame.copy()
            def to_screen(gx: float, gy: float) -> Tuple[int, int]:
                return int(round(gx)), int(round(gy))

        # A. Center Gimbal Boresight Reticle
        reticle_color = (180, 180, 180)
        cv2.circle(frame, (cx, cy), 14, reticle_color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx - 24, cy), (cx - 16, cy), reticle_color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + 16, cy), (cx + 24, cy), reticle_color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 24), (cx, cy - 16), reticle_color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + 16), (cx, cy + 24), reticle_color, 1, cv2.LINE_AA)

        # B. Wide Sky Radar Minimap (in zoom_pov mode) OR PiP Thumbnail (in wide_pip mode)
        if is_zoomed and self.active_crop_box is not None:
            zx1, zy1, zx2, zy2 = self.active_crop_box

            if self.view_mode == "zoom_pov":
                # Render Tactical Wide Sky Radar Overview Minimap in top-right
                try:
                    mm_w = 260
                    mm_h = max(100, int(mm_w * h / w))
                    mm_margin = 15

                    mm_raw = cv2.resize(raw_frame, (mm_w, mm_h), interpolation=cv2.INTER_AREA)

                    # Map crop box onto minimap coordinates
                    mx1 = int(zx1 * mm_w / w)
                    my1 = int(zy1 * mm_h / h)
                    mx2 = int(zx2 * mm_w / w)
                    my2 = int(zy2 * mm_h / h)

                    # Draw PTZ viewing cone box on minimap
                    cv2.rectangle(mm_raw, (mx1, my1), (mx2, my2), (0, 255, 255), 2)
                    cv2.circle(mm_raw, ((mx1 + mx2) // 2, (my1 + my2) // 2), 3, (0, 255, 0), -1)

                    mm_x1 = w - mm_w - mm_margin
                    mm_y1 = mm_margin
                    mm_x2 = mm_x1 + mm_w
                    mm_y2 = mm_y1 + mm_h

                    frame[mm_y1:mm_y2, mm_x1:mm_x2] = mm_raw
                    cv2.rectangle(frame, (mm_x1, mm_y1), (mm_x2, mm_y2), (0, 255, 255), 2)

                    # Minimap Header Badge
                    cv2.rectangle(frame, (mm_x1, mm_y1), (mm_x1 + mm_w, mm_y1 + 22), (20, 20, 20), -1)
                    cv2.putText(
                        frame, "WIDE RADAR [1.0x OVERVIEW]",
                        (mm_x1 + 8, mm_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA
                    )
                except Exception:
                    pass

            else:
                # wide_pip mode: Draw yellow crop box and corner PiP on wide frame
                zoom_box_color = (255, 230, 0)
                c_len = max(20, int((zx2 - zx1) * 0.1))
                # Top-Left
                cv2.line(frame, (zx1, zy1), (zx1 + c_len, zy1), zoom_box_color, 2, cv2.LINE_AA)
                cv2.line(frame, (zx1, zy1), (zx1, zy1 + c_len), zoom_box_color, 2, cv2.LINE_AA)
                # Top-Right
                cv2.line(frame, (zx2, zy1), (zx2 - c_len, zy1), zoom_box_color, 2, cv2.LINE_AA)
                cv2.line(frame, (zx2, zy1), (zx2, zy1 + c_len), zoom_box_color, 2, cv2.LINE_AA)
                # Bottom-Left
                cv2.line(frame, (zx1, zy2), (zx1 + c_len, zy2), zoom_box_color, 2, cv2.LINE_AA)
                cv2.line(frame, (zx1, zy2), (zx1, zy2 - c_len), zoom_box_color, 2, cv2.LINE_AA)
                # Bottom-Right
                cv2.line(frame, (zx2, zy2), (zx2 - c_len, zy2), zoom_box_color, 2, cv2.LINE_AA)
                cv2.line(frame, (zx2, zy2), (zx2, zy2 - c_len), zoom_box_color, 2, cv2.LINE_AA)

                ptz_label = f"DIGITAL PTZ FOVEA [{telemetry['zoom_level']:.1f}x]"
                cv2.putText(frame, ptz_label, (zx1 + 8, max(24, zy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, zoom_box_color, 1, cv2.LINE_AA)

                try:
                    pip_w, pip_h = 240, 150
                    pip_margin = 15
                    crop_roi = raw_frame[zy1:zy2, zx1:zx2]
                    if crop_roi.size > 0:
                        pip_resized = cv2.resize(crop_roi, (pip_w, pip_h), interpolation=cv2.INTER_LINEAR)
                        pip_x1 = w - pip_w - pip_margin
                        pip_y1 = pip_margin
                        pip_x2 = pip_x1 + pip_w
                        pip_y2 = pip_y1 + pip_h

                        frame[pip_y1:pip_y2, pip_x1:pip_x2] = pip_resized
                        cv2.rectangle(frame, (pip_x1, pip_y1), (pip_x2, pip_y2), zoom_box_color, 2)
                        cv2.circle(frame, (pip_x1 + pip_w // 2, pip_y1 + pip_h // 2), 6, (0, 255, 0), 1)

                        cv2.rectangle(frame, (pip_x1, pip_y1), (pip_x1 + pip_w, pip_y1 + 22), (20, 20, 20), -1)
                        cv2.putText(
                            frame, f"FOVEAL PTZ [{telemetry['zoom_level']:.1f}x]",
                            (pip_x1 + 8, pip_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA
                        )
                except Exception:
                    pass

        # C. Render Active Tracks
        for trk in active_tracks:
            gx1, gy1, gx2, gy2 = trk["bbox"]
            sx1, sy1 = to_screen(gx1, gy1)
            sx2, sy2 = to_screen(gx2, gy2)
            stcx, stcy = int((sx1 + sx2) / 2), int((sy1 + sy2) / 2)

            is_primary = (trk["id"] == telemetry["target_id"])
            is_coasting = trk["is_coasting"]

            if is_primary:
                if is_coasting:
                    color = (0, 165, 255)  # Amber: Coasting
                    status_lbl = f"LOCK #{trk['id']} [COASTING {trk['time_coasting']}]"
                else:
                    color = (0, 255, 0)    # Neon Green: Active Lock
                    status_lbl = f"LOCKED TARGET #{trk['id']} ({trk['confidence']*100:.0f}%)"
            else:
                color = (255, 200, 0)      # Cyan: Secondary
                status_lbl = f"TRK #{trk['id']}"

            # Flight trajectory trail
            pts = [to_screen(pt[0], pt[1]) for pt in trk["trajectory"]]
            if len(pts) > 1:
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)

            # Targeting corner brackets
            line_len = max(8, min(24, int(abs(sx2 - sx1) * 0.25)))
            thick = 3 if is_primary else 1

            # Top-Left
            cv2.line(frame, (sx1, sy1), (sx1 + line_len, sy1), color, thick, cv2.LINE_AA)
            cv2.line(frame, (sx1, sy1), (sx1, sy1 + line_len), color, thick, cv2.LINE_AA)
            # Top-Right
            cv2.line(frame, (sx2, sy1), (sx2 - line_len, sy1), color, thick, cv2.LINE_AA)
            cv2.line(frame, (sx2, sy1), (sx2 - line_len, sy1), color, thick, cv2.LINE_AA)
            # Bottom-Left
            cv2.line(frame, (sx1, sy2), (sx1 + line_len, sy2), color, thick, cv2.LINE_AA)
            cv2.line(frame, (sx1, sy2), (sx1, sy2 - line_len), color, thick, cv2.LINE_AA)
            # Bottom-Right
            cv2.line(frame, (sx2, sy2), (sx2 - line_len, sy2), color, thick, cv2.LINE_AA)
            cv2.line(frame, (sx2, sy2), (sx2, sy2 - line_len), color, thick, cv2.LINE_AA)

            # Label banner
            cv2.putText(
                frame, status_lbl, (sx1, max(22, sy1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
            )

            # Primary tracking error line & velocity lead
            if is_primary:
                cv2.line(frame, (cx, cy), (stcx, stcy), (0, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(frame, (stcx, stcy), 4, (0, 255, 255), -1, cv2.LINE_AA)

                vx, vy = trk["velocity"]
                tg_cx = (gx1 + gx2) / 2.0
                tg_cy = (gy1 + gy2) / 2.0
                slead_x, slead_y = to_screen(tg_cx + vx * 0.3, tg_cy + vy * 0.3)
                cv2.line(frame, (stcx, stcy), (slead_x, slead_y), (0, 165, 255), 1, cv2.LINE_AA)
                cv2.circle(frame, (slead_x, slead_y), 3, (0, 165, 255), -1, cv2.LINE_AA)

        # D. Flight Telemetry HUD Card (Top-Left Corner)
        card_w, card_h = 340, 160
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (15 + card_w, 15 + card_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (15, 15), (15 + card_w, 15 + card_h), (80, 80, 80), 1)

        status = telemetry["status"]
        if status == "LOCKED":
            badge_color = (0, 255, 0)
        elif status == "COASTING":
            badge_color = (0, 165, 255)
        else:
            badge_color = (0, 0, 255)

        mode_tag = "TRUE POV" if self.view_mode == "zoom_pov" else "PTZ PiP"
        zoom_badge = f"{telemetry['zoom_level']:.1f}x [{mode_tag}]" if telemetry.get("is_zoomed") else "1.0x [WIDE]"
        zoom_badge_color = (0, 255, 255) if telemetry.get("is_zoomed") else (160, 160, 160)

        cv2.putText(frame, f"STATUS: {status}", (30, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, badge_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"TARGET ID: {telemetry['target_id'] or 'NONE'} | SIZE: {telemetry.get('target_size', 0):.0f}/{telemetry.get('desired_size', 140):.0f}px", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"ERR X: {telemetry['error_x']:+.3f} | Y: {telemetry['error_y']:+.3f} | RNG: {telemetry.get('error_range', 0.0):+.2f}", (30, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"VEL X: {telemetry['velocity_x']:+.1f} | VEL Y: {telemetry['velocity_y']:+.1f} px/s", (30, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"ZOOM: {zoom_badge}", (30, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.48, zoom_badge_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {telemetry['fps']:.1f} | TRACKS: {telemetry['active_tracks_count']}", (30, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

        return frame

    def run(
        self,
        source: str,
        output_video_path: str = "outputs/tracked_output.mp4",
        max_frames: Optional[int] = None,
        start_frame: int = 0,
        display: bool = False
    ):
        """
        Runs tracking across a video file (.mp4, .avi), directory of images, or live webcam.
        """
        source_path = Path(source) if not source.isdigit() else source
        os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)

        if str(source).isdigit() or (isinstance(source_path, Path) and source_path.is_file()):
            # Video or Camera capture
            cap_src = int(source) if str(source).isdigit() else str(source_path)
            cap = cv2.VideoCapture(cap_src)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {source}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not str(source).isdigit() else 0
            if start_frame > 0 and total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(start_frame, total_frames - 1))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            process_count = min(total_frames - start_frame, max_frames) if max_frames and total_frames > 0 else (total_frames or max_frames)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

            print(f"\n[STREAM] Processing source: {source}")
            print(f"Resolution: {w}x{h} @ {fps:.1f} FPS | Total Frames: {total_frames or 'Live Stream'} (Start: {start_frame})")

            idx = 0
            while cap.isOpened() and (max_frames is None or idx < max_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, telemetry = self.process_frame(frame, draw_hud=True)
                writer.write(annotated)
                idx += 1

                if idx % 30 == 0 or idx == process_count:
                    print(f"  Frame [{idx}/{process_count or '?'}] | Status: {telemetry['status']} | FPS: {telemetry['fps']:.1f}")

                if display:
                    cv2.imshow("Autonomous Drone Tracker", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            writer.release()
            if display:
                cv2.destroyAllWindows()
            print(f"\n[SUCCESS] Tracked video saved to: {output_video_path}")

        elif isinstance(source_path, Path) and source_path.is_dir():
            # Directory of image frames (e.g. Anti-UAV sequence)
            img_files = sorted(glob.glob(f"{source_path}/*.jpg") + glob.glob(f"{source_path}/*.png"))
            if not img_files:
                raise FileNotFoundError(f"No images found in: {source_path}")

            if start_frame > 0:
                img_files = img_files[start_frame:]
            if max_frames:
                img_files = img_files[:max_frames]

            sample = cv2.imread(img_files[0])
            h, w = sample.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (w, h))

            print(f"\n[SEQUENCE] Tracking across {len(img_files)} frames from: {source_path}")
            for idx, img_p in enumerate(img_files):
                frame = cv2.imread(img_p)
                annotated, telemetry = self.process_frame(frame, draw_hud=True)
                writer.write(annotated)

                if (idx + 1) % 30 == 0 or (idx + 1) == len(img_files):
                    print(f"  Frame [{idx+1}/{len(img_files)}] | Status: {telemetry['status']} | FPS: {telemetry['fps']:.1f}")

                if display:
                    cv2.imshow("Autonomous Drone Tracker", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            writer.release()
            if display:
                cv2.destroyAllWindows()
            print(f"\n[SUCCESS] Tracked video saved to: {output_video_path}")


# ==============================================================================
# 4. Command Line Entry Point
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous Drone Tracking: Pure YOLO + Kalman Filter with Foveal PTZ")
    parser.add_argument("--source", type=str, default="data/tests/drones/drone-flying.mp4", help="Video file (.mp4), folder of frames, or camera index (0)")
    parser.add_argument("--weights", type=str, default="runs/detect/yolov8n_drone/weights/best.pt", help="Path to trained YOLO best.pt weights")
    parser.add_argument("--out", type=str, default="outputs/tracked_kalman_output.mp4", help="Output MP4 video file path")
    parser.add_argument("--conf", type=float, default=0.20, help="Detection confidence threshold")
    parser.add_argument("--max_lost", type=int, default=15, help="Max coasting frames before track is declared lost")
    parser.add_argument("--start-frame", type=int, default=0, help="Starting frame number")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit number of frames to process")
    parser.add_argument("--display", action="store_true", help="Display live tracking window")
    parser.add_argument("--no-zoom", action="store_true", help="Disable Dynamic Digital PTZ (Foveal Zoom)")
    parser.add_argument("--max-zoom", type=float, default=2.5, help="Maximum digital zoom level (e.g. 2.0 to 3.0)")
    parser.add_argument("--target-size", "--desired-size", type=float, default=140.0, help="Target reference pixel size to regulate on screen (e.g. 140 px)")
    parser.add_argument("--view", type=str, default="zoom_pov", choices=["zoom_pov", "wide_pip"], help="Display view: 'zoom_pov' (magnified true camera POV) or 'wide_pip' (wide angle + corner PiP)")
    parser.add_argument("--q-pos", type=float, default=1.0, help="Kalman process position noise (Q_pos)")
    parser.add_argument("--q-vel", type=float, default=18.0, help="Kalman process velocity noise (Q_vel)")
    parser.add_argument("--r-pos", type=float, default=2.0, help="Kalman measurement position noise (R_pos)")
    parser.add_argument("--no-kalman", action="store_true", help="Disable Kalman filtering (Pure Raw YOLO Mode)")
    args = parser.parse_args()

    pipeline = DroneTrackingPipeline(
        weights=args.weights,
        conf_threshold=args.conf,
        max_lost_frames=args.max_lost,
        enable_dynamic_zoom=(not args.no_zoom),
        max_zoom=args.max_zoom,
        desired_target_size=args.target_size,
        view_mode=args.view,
        q_pos=args.q_pos,
        q_vel=args.q_vel,
        r_pos=args.r_pos,
        enable_kalman=(not args.no_kalman)
    )

    pipeline.run(
        source=args.source,
        output_video_path=args.out,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        display=args.display
    )
