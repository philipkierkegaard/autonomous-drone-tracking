#!/usr/bin/env python3
"""
Kalman Filter Calibration & Hyperparameter Tuning on Anti-UAV Validation Sequences.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception.

Calibrates:
1. Empirical Measurement Noise Covariance Matrix R from YOLO residual errors [dcx, dcy, dw, dh]
2. Process Noise Covariance Matrix Q (q_pos, q_vel, q_dim, q_scale_rate) via multi-objective sweep
3. Lifecycle & Association Hyperparameters (conf_thresh, iou_match_threshold, dist_gate_scale, max_lost_frames, min_hits)

Evaluates:
- Center Location Error (CLE) & Precision @ 20px / 50px
- Bounding Box Overlap (Mean IoU, Success @ IoU 0.50)
- Tracking Continuity & Verified Lock Rate
- Spatial Jitter Index (Control Jerk)
- Coasting Recall during detector dropout

Outputs:
- outputs/kalman_tuning/calibrated_kalman_parameters.json
- outputs/kalman_tuning/tuning_sweep_results.json
"""

import os
import sys
import time
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# Add repo root to python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.pipeline import KalmanBoxTracker, DroneTracker, compute_iou_matrix

VAL_VIDEOS = ["video01", "video04", "video13", "video15", "video16", "video17"]
VIDEOS_DIR = REPO_ROOT / "perception" / "raw_data" / "Anti-UAV" / "videos"
GT_DIR = REPO_ROOT / "perception" / "raw_data" / "Anti-UAV" / "Anti-UAV-Tracking-V0GT"
OUTPUT_DIR = REPO_ROOT / "outputs" / "kalman_tuning"
CACHE_FILE = OUTPUT_DIR / "val_detections_cache.pkl"


def load_gt_boxes(gt_path: Path) -> List[Optional[np.ndarray]]:
    """Loads Anti-UAV GT txt file [x, y, w, h] -> [x1, y1, x2, y2]."""
    boxes = []
    if not gt_path.exists():
        return boxes
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            vals = [float(p) for p in parts]
            x, y, w, h = vals[:4]
            if w > 0 and h > 0:
                boxes.append(np.array([x, y, x + w, y + h], dtype=np.float32))
            else:
                boxes.append(None)
    return boxes


def compute_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    boxAArea = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return float(interArea / (boxAArea + boxBArea - interArea))


def extract_or_load_detections(
    weights_path: str,
    device: str = "mps",
    imgsz: int = 768,
    conf_min: float = 0.10,
    force_recompute: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Runs YOLO across all validation videos and caches frame-level detections."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists() and not force_recompute:
        print(f"[CACHE] Loading cached validation detections from: {CACHE_FILE}")
        with open(CACHE_FILE, "rb") as f:
            cached = pickle.load(f)
            if cached.get("weights") == str(weights_path):
                print(f"[CACHE] Successfully loaded {len(cached['data'])} validation sequences from cache.")
                return cached["data"]
            else:
                print(f"[CACHE] Weights mismatch in cache ({cached.get('weights')} vs {weights_path}), recomputing...")

    print("=" * 80)
    print("EXTRACTING DETECTIONS ACROSS ANTI-UAV VALIDATION SEQUENCES")
    print(f"Model: {weights_path} | Imgsz: {imgsz} | Min Conf: {conf_min} | Device: {device}")
    print("=" * 80)

    model = YOLO(weights_path)
    val_data = {}

    for vid_name in VAL_VIDEOS:
        vid_path = VIDEOS_DIR / f"{vid_name}.mp4"
        gt_path = GT_DIR / f"{vid_name}_gt.txt"

        if not vid_path.exists() or not gt_path.exists():
            print(f"[WARN] Skipping {vid_name}: missing video or GT.")
            continue

        gt_boxes = load_gt_boxes(gt_path)
        cap = cv2.VideoCapture(str(vid_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"---> Processing {vid_name} ({total_frames} frames, {w}x{h} @ {fps:.1f} FPS)...")
        frame_detections = []  # list of [(boxes, confs)] per frame
        frame_idx = 0
        t0 = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                source=frame,
                imgsz=imgsz,
                device=device,
                conf=conf_min,
                iou=0.45,
                verbose=False
            )

            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
            else:
                boxes = np.empty((0, 4), dtype=np.float32)
                confs = np.empty((0,), dtype=np.float32)

            frame_detections.append({
                "boxes": boxes,
                "confs": confs
            })

            frame_idx += 1
            if frame_idx % 300 == 0 or frame_idx == total_frames:
                el = time.time() - t0
                print(f"     Frame {frame_idx}/{total_frames} ({frame_idx/max(1e-3, el):.1f} FPS)")

        cap.release()
        val_data[vid_name] = {
            "vid_name": vid_name,
            "total_frames": total_frames,
            "fps": fps,
            "width": w,
            "height": h,
            "gt_boxes": gt_boxes,
            "detections": frame_detections
        }

    # Save to cache
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"weights": str(weights_path), "data": val_data}, f)
    print(f"[CACHE] Saved validation detections cache to {CACHE_FILE}")

    return val_data


def calibrate_measurement_noise(
    val_data: Dict[str, Dict[str, Any]],
    conf_thresh: float = 0.25,
    match_iou_thresh: float = 0.25
) -> Dict[str, Any]:
    """
    Computes empirical measurement noise covariance matrix R = diag(sigma_cx^2, sigma_cy^2, sigma_w^2, sigma_h^2).
    """
    residuals = []  # [dcx, dcy, dw, dh]
    matched_count = 0
    total_gt_count = 0
    total_det_count = 0

    for vid_name, vdata in val_data.items():
        gt_boxes = vdata["gt_boxes"]
        frames_det = vdata["detections"]

        for f_idx, (gt_box, f_det) in enumerate(zip(gt_boxes, frames_det)):
            boxes = f_det["boxes"]
            confs = f_det["confs"]

            # Filter by confidence
            mask = confs >= conf_thresh
            boxes = boxes[mask]
            confs = confs[mask]
            total_det_count += len(boxes)

            if gt_box is not None:
                total_gt_count += 1
                if len(boxes) == 0:
                    continue

                # Match with GT
                ious = [compute_iou(b, gt_box) for b in boxes]
                best_idx = int(np.argmax(ious))
                if ious[best_idx] >= match_iou_thresh:
                    matched_box = boxes[best_idx]
                    matched_count += 1

                    # Compute center and size
                    gt_cx = (gt_box[0] + gt_box[2]) / 2.0
                    gt_cy = (gt_box[1] + gt_box[3]) / 2.0
                    gt_w = gt_box[2] - gt_box[0]
                    gt_h = gt_box[3] - gt_box[1]

                    det_cx = (matched_box[0] + matched_box[2]) / 2.0
                    det_cy = (matched_box[1] + matched_box[3]) / 2.0
                    det_w = matched_box[2] - matched_box[0]
                    det_h = matched_box[3] - matched_box[1]

                    residuals.append([
                        det_cx - gt_cx,
                        det_cy - gt_cy,
                        det_w - gt_w,
                        det_h - gt_h
                    ])

    residuals = np.array(residuals, dtype=np.float32)
    mean_res = np.mean(residuals, axis=0)
    cov_matrix = np.cov(residuals, rowvar=False)
    var_res = np.var(residuals, axis=0)
    std_res = np.std(residuals, axis=0)

    calibration_result = {
        "matched_pairs": matched_count,
        "total_gt": total_gt_count,
        "total_det": total_det_count,
        "match_rate": float(matched_count / max(1, total_gt_count)),
        "mean_residual_px": [float(v) for v in mean_res],
        "std_residual_px": [float(v) for v in std_res],
        "variance_residual_px2": [float(v) for v in var_res],
        "covariance_matrix_4x4": [[float(c) for c in row] for row in cov_matrix],
        "recommended_R_diag": [float(v) for v in var_res],
        "sigma_cx": float(std_res[0]),
        "sigma_cy": float(std_res[1]),
        "sigma_w": float(std_res[2]),
        "sigma_h": float(std_res[3])
    }

    print("\n" + "=" * 80)
    print("CALIBRATED MEASUREMENT NOISE COVARIANCE (R)")
    print("=" * 80)
    print(f"Matched Detections: {matched_count}/{total_gt_count} ({calibration_result['match_rate']*100:.1f}%)")
    print(f"Residual Biases (Mean):   cx: {mean_res[0]:+.2f} px | cy: {mean_res[1]:+.2f} px | w: {mean_res[2]:+.2f} px | h: {mean_res[3]:+.2f} px")
    print(f"Standard Deviations (σ):  cx: {std_res[0]:.2f} px  | cy: {std_res[1]:.2f} px  | w: {std_res[2]:.2f} px  | h: {std_res[3]:.2f} px")
    print(f"Variances (σ² = R_diag):  cx: {var_res[0]:.2f} px² | cy: {var_res[1]:.2f} px² | w: {var_res[2]:.2f} px² | h: {var_res[3]:.2f} px²")
    print("Full 4x4 Covariance Matrix:")
    for row in cov_matrix:
        print("  " + "  ".join(f"{c:8.2f}" for c in row))
    print("=" * 80 + "\n")

    return calibration_result


def simulate_tracker_on_sequence(
    vdata: Dict[str, Any],
    conf_thresh: float,
    tracker_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Runs a simulated DroneTracker across a pre-cached sequence."""
    gt_boxes = vdata["gt_boxes"]
    frames_det = vdata["detections"]

    # Instantiate DroneTracker with candidate parameters
    tracker = DroneTracker(**tracker_kwargs)

    visible_frames = 0
    absent_frames = 0
    locked_frames = 0
    coasting_frames = 0
    lost_frames = 0

    cle_list = []
    iou_list = []
    prec_20_count = 0
    prec_50_count = 0
    succ_50_count = 0
    verified_lock_50_count = 0
    verified_cont_50_count = 0
    false_lock_count = 0
    missed_gt_count = 0
    coasting_maintained_count = 0

    trajectory_pts = []  # list of (f_idx, cx, cy) for jitter computation

    for f_idx, (gt_box, f_det) in enumerate(zip(gt_boxes, frames_det)):
        boxes = f_det["boxes"]
        confs = f_det["confs"]

        # Filter by confidence threshold
        mask = confs >= conf_thresh
        curr_boxes = boxes[mask]
        curr_confs = confs[mask]

        # Update tracker
        active_tracks = tracker.update(curr_boxes, curr_confs)

        primary_track = None
        for trk in active_tracks:
            if trk["id"] == tracker.primary_track_id:
                primary_track = trk
                break

        status = "SEARCHING"
        pred_box = None
        if primary_track is not None:
            pred_box = primary_track["bbox"]
            status = "COASTING" if primary_track["is_coasting"] else "LOCKED"

        if status == "LOCKED":
            locked_frames += 1
        elif status == "COASTING":
            coasting_frames += 1
        else:
            lost_frames += 1

        if gt_box is not None:
            visible_frames += 1
            gt_cx = (gt_box[0] + gt_box[2]) / 2.0
            gt_cy = (gt_box[1] + gt_box[3]) / 2.0

            if pred_box is not None:
                p_cx = (pred_box[0] + pred_box[2]) / 2.0
                p_cy = (pred_box[1] + pred_box[3]) / 2.0

                cle = float(np.sqrt((p_cx - gt_cx) ** 2 + (p_cy - gt_cy) ** 2))
                iou = compute_iou(pred_box, gt_box)

                cle_list.append(cle)
                iou_list.append(iou)

                if cle <= 20.0:
                    prec_20_count += 1
                if cle <= 50.0:
                    prec_50_count += 1
                if iou >= 0.50:
                    succ_50_count += 1

                if status == "LOCKED":
                    if iou >= 0.50:
                        verified_lock_50_count += 1
                    if iou < 0.10:
                        false_lock_count += 1
                if status in ["LOCKED", "COASTING"] and iou >= 0.50:
                    verified_cont_50_count += 1

                trajectory_pts.append((f_idx, p_cx, p_cy))
            else:
                cle_list.append(100.0)  # penalty for loss
                iou_list.append(0.0)

            # Evaluate Coasting Recall when detector missed
            if len(curr_boxes) == 0:
                missed_gt_count += 1
                if pred_box is not None and compute_iou(pred_box, gt_box) >= 0.30:
                    coasting_maintained_count += 1
        else:
            absent_frames += 1
            if status == "LOCKED":
                false_lock_count += 1

    # Compute Spatial Jitter Index (Second-order difference jerk on tracked trajectory)
    jitter_vals = []
    for i in range(1, len(trajectory_pts) - 1):
        f_prev, x_prev, y_prev = trajectory_pts[i - 1]
        f_curr, x_curr, y_curr = trajectory_pts[i]
        f_next, x_next, y_next = trajectory_pts[i + 1]
        # Only compute jitter if consecutive frames
        if f_curr == f_prev + 1 and f_next == f_curr + 1:
            jerk_x = x_next - 2.0 * x_curr + x_prev
            jerk_y = y_next - 2.0 * y_curr + y_prev
            jitter_vals.append(float(np.sqrt(jerk_x ** 2 + jerk_y ** 2)))

    mean_jitter = float(np.mean(jitter_vals)) if len(jitter_vals) > 0 else 0.0

    return {
        "visible_frames": visible_frames,
        "absent_frames": absent_frames,
        "locked_frames": locked_frames,
        "coasting_frames": coasting_frames,
        "lost_frames": lost_frames,
        "prec_20_count": prec_20_count,
        "prec_50_count": prec_50_count,
        "succ_50_count": succ_50_count,
        "verified_lock_50_count": verified_lock_50_count,
        "verified_cont_50_count": verified_cont_50_count,
        "false_lock_count": false_lock_count,
        "missed_gt_count": missed_gt_count,
        "coasting_maintained_count": coasting_maintained_count,
        "cle_list": cle_list,
        "iou_list": iou_list,
        "mean_jitter": mean_jitter
    }


def evaluate_candidate(
    val_data: Dict[str, Dict[str, Any]],
    conf_thresh: float,
    tracker_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluates candidate parameters across all validation sequences and returns aggregated metrics."""
    tot_visible = 0
    tot_prec_20 = 0
    tot_prec_50 = 0
    tot_succ_50 = 0
    tot_verif_lock_50 = 0
    tot_verif_cont_50 = 0
    tot_false_lock = 0
    tot_locked = 0
    tot_missed_gt = 0
    tot_coasting_maintained = 0

    all_cle = []
    all_iou = []
    all_jitters = []

    for vid_name, vdata in val_data.items():
        res = simulate_tracker_on_sequence(vdata, conf_thresh, tracker_kwargs)
        tot_visible += res["visible_frames"]
        tot_prec_20 += res["prec_20_count"]
        tot_prec_50 += res["prec_50_count"]
        tot_succ_50 += res["succ_50_count"]
        tot_verif_lock_50 += res["verified_lock_50_count"]
        tot_verif_cont_50 += res["verified_cont_50_count"]
        tot_false_lock += res["false_lock_count"]
        tot_locked += res["locked_frames"]
        tot_missed_gt += res["missed_gt_count"]
        tot_coasting_maintained += res["coasting_maintained_count"]

        all_cle.extend(res["cle_list"])
        all_iou.extend(res["iou_list"])
        if res["mean_jitter"] > 0:
            all_jitters.append(res["mean_jitter"])

    prec_20 = float(tot_prec_20 / max(1, tot_visible))
    prec_50 = float(tot_prec_50 / max(1, tot_visible))
    succ_50 = float(tot_succ_50 / max(1, tot_visible))
    verif_lock_50 = float(tot_verif_lock_50 / max(1, tot_visible))
    verif_cont_50 = float(tot_verif_cont_50 / max(1, tot_visible))
    false_lock_rate = float(tot_false_lock / max(1, tot_locked))
    coasting_recall = float(tot_coasting_maintained / max(1, tot_missed_gt))
    mean_cle = float(np.mean(all_cle)) if all_cle else 100.0
    mean_iou = float(np.mean(all_iou)) if all_iou else 0.0
    mean_jitter = float(np.mean(all_jitters)) if all_jitters else 0.0

    # Composite objective score:
    # Rewards accuracy (Prec@20, IoU, Continuity) while penalizing CLE, jitter, and false locks
    score = (
        prec_20 * 40.0
        + succ_50 * 30.0
        + verif_cont_50 * 20.0
        + mean_iou * 20.0
        + coasting_recall * 10.0
        - (mean_cle * 0.5)
        - (mean_jitter * 2.0)
        - (false_lock_rate * 50.0)
    )

    return {
        "score": float(score),
        "prec_20": prec_20,
        "prec_50": prec_50,
        "succ_50": succ_50,
        "verif_lock_50": verif_lock_50,
        "verif_cont_50": verif_cont_50,
        "coasting_recall": coasting_recall,
        "mean_cle": mean_cle,
        "mean_iou": mean_iou,
        "mean_jitter": mean_jitter,
        "false_lock_rate": false_lock_rate
    }


def run_kalman_parameter_sweep(
    val_data: Dict[str, Dict[str, Any]],
    calibrated_R_diag: List[float]
) -> Dict[str, Any]:
    """Sweeps Q covariance terms and tracker lifecycle thresholds to find optimal tuning."""
    print("=" * 80)
    print("STARTING MULTI-OBJECTIVE KALMAN HYPERPARAMETER SWEEP ON VALIDATION SET")
    print(f"Calibrated R: diag({calibrated_R_diag[0]:.2f}, {calibrated_R_diag[1]:.2f}, {calibrated_R_diag[2]:.2f}, {calibrated_R_diag[3]:.2f})")
    print("=" * 80)

    # Grid search space
    q_pos_vals = [0.2, 0.5, 1.0, 2.0]
    q_vel_vals = [5.0, 10.0, 18.0, 30.0, 50.0]
    q_dim_vals = [1.0, 2.0, 4.0]
    q_scale_rate_vals = [5.0, 10.0, 20.0]
    conf_thresh_vals = [0.15, 0.20, 0.25]
    max_lost_vals = [10, 15, 20]
    min_hits_vals = [1, 2]

    # Stage 1: Fine-grain sweep over Q (q_pos, q_vel, q_dim, q_scale_rate) with fixed lifecycle
    print("\n[PHASE 1] Sweeping Process Noise Covariance Q terms...")
    best_q_score = -float("inf")
    best_q_params = None
    best_q_metrics = None
    sweep_history = []

    # First, test the untuned baseline configuration for reference
    untuned_kwargs = {
        "q_pos": 1.0,
        "q_vel": 18.0,
        "r_pos": 2.0,
        "q_dim": 2.0,
        "q_scale_rate": 10.0,
        "r_dim": 4.0,
        "max_lost_frames": 15,
        "min_hits_to_confirm": 2,
        "iou_match_threshold": 0.75,
        "dist_gate_scale": 2.5
    }
    baseline_metrics = evaluate_candidate(val_data, conf_thresh=0.20, tracker_kwargs=untuned_kwargs)
    print(f"--> Baseline (Untuned Q, Untuned R):")
    print(f"    Prec@20: {baseline_metrics['prec_20']*100:.2f}% | Mean IoU: {baseline_metrics['mean_iou']:.3f} | CLE: {baseline_metrics['mean_cle']:.2f}px | Jitter: {baseline_metrics['mean_jitter']:.2f}px | Score: {baseline_metrics['score']:.2f}")

    # Next, test calibrated R with default Q
    calibrated_r_kwargs = dict(untuned_kwargs)
    calibrated_r_kwargs["r_cov"] = np.diag(calibrated_R_diag).astype(np.float32)
    cal_r_metrics = evaluate_candidate(val_data, conf_thresh=0.20, tracker_kwargs=calibrated_r_kwargs)
    print(f"--> Calibrated R (with Default Q):")
    print(f"    Prec@20: {cal_r_metrics['prec_20']*100:.2f}% | Mean IoU: {cal_r_metrics['mean_iou']:.3f} | CLE: {cal_r_metrics['mean_cle']:.2f}px | Jitter: {cal_r_metrics['mean_jitter']:.2f}px | Score: {cal_r_metrics['score']:.2f}")

    r_cov_matrix = np.diag(calibrated_R_diag).astype(np.float32)

    total_combos = len(q_pos_vals) * len(q_vel_vals) * len(q_dim_vals) * len(q_scale_rate_vals)
    combo_idx = 0
    t0 = time.time()

    for q_pos in q_pos_vals:
        for q_vel in q_vel_vals:
            for q_dim in q_dim_vals:
                for q_sr in q_scale_rate_vals:
                    combo_idx += 1
                    t_kwargs = {
                        "q_pos": q_pos,
                        "q_vel": q_vel,
                        "q_dim": q_dim,
                        "q_scale_rate": q_sr,
                        "r_cov": r_cov_matrix,
                        "max_lost_frames": 15,
                        "min_hits_to_confirm": 2,
                        "iou_match_threshold": 0.75,
                        "dist_gate_scale": 2.5
                    }
                    metrics = evaluate_candidate(val_data, conf_thresh=0.20, tracker_kwargs=t_kwargs)
                    record = {
                        "q_pos": q_pos,
                        "q_vel": q_vel,
                        "q_dim": q_dim,
                        "q_scale_rate": q_sr,
                        "metrics": metrics
                    }
                    sweep_history.append(record)

                    if metrics["score"] > best_q_score:
                        best_q_score = metrics["score"]
                        best_q_params = (q_pos, q_vel, q_dim, q_sr)
                        best_q_metrics = metrics

    el = time.time() - t0
    print(f"Completed {total_combos} Q configurations in {el:.2f}s ({total_combos/el:.1f} configs/sec).")
    print(f"Best Q Parameters: q_pos={best_q_params[0]}, q_vel={best_q_params[1]}, q_dim={best_q_params[2]}, q_scale_rate={best_q_params[3]}")
    print(f"Best Q Metrics: Prec@20: {best_q_metrics['prec_20']*100:.2f}% | Mean IoU: {best_q_metrics['mean_iou']:.3f} | CLE: {best_q_metrics['mean_cle']:.2f}px | Jitter: {best_q_metrics['mean_jitter']:.2f}px | Score: {best_q_metrics['score']:.2f}")

    # Stage 2: Sweep lifecycle and association thresholds with best Q
    print("\n[PHASE 2] Sweeping Tracker Lifecycle & Association Hyperparameters...")
    best_final_score = -float("inf")
    best_final_config = None
    best_final_metrics = None

    for conf in conf_thresh_vals:
        for max_lost in max_lost_vals:
            for min_hits in min_hits_vals:
                for iou_thresh in [0.70, 0.75, 0.80]:
                    for dist_scale in [2.0, 2.5, 3.0]:
                        t_kwargs = {
                            "q_pos": best_q_params[0],
                            "q_vel": best_q_params[1],
                            "q_dim": best_q_params[2],
                            "q_scale_rate": best_q_params[3],
                            "r_cov": r_cov_matrix,
                            "max_lost_frames": max_lost,
                            "min_hits_to_confirm": min_hits,
                            "iou_match_threshold": iou_thresh,
                            "dist_gate_scale": dist_scale
                        }
                        metrics = evaluate_candidate(val_data, conf_thresh=conf, tracker_kwargs=t_kwargs)

                        if metrics["score"] > best_final_score:
                            best_final_score = metrics["score"]
                            best_final_config = {
                                "conf_threshold": conf,
                                "q_pos": best_q_params[0],
                                "q_vel": best_q_params[1],
                                "q_dim": best_q_params[2],
                                "q_scale_rate": best_q_params[3],
                                "max_lost_frames": max_lost,
                                "min_hits_to_confirm": min_hits,
                                "iou_match_threshold": iou_thresh,
                                "dist_gate_scale": dist_scale,
                                "r_cov_diag": [float(v) for v in calibrated_R_diag]
                            }
                            best_final_metrics = metrics

    print("\n" + "=" * 80)
    print("FINAL TUNING OUTCOME & OPTIMAL HYPERPARAMETERS")
    print("=" * 80)
    print(f"Optimal Conf Threshold:     {best_final_config['conf_threshold']}")
    print(f"Optimal Q pos:              {best_final_config['q_pos']}")
    print(f"Optimal Q vel:              {best_final_config['q_vel']}")
    print(f"Optimal Q dim:              {best_final_config['q_dim']}")
    print(f"Optimal Q scale rate:       {best_final_config['q_scale_rate']}")
    print(f"Optimal Max Lost Frames:    {best_final_config['max_lost_frames']}")
    print(f"Optimal Min Hits to Lock:   {best_final_config['min_hits_to_confirm']}")
    print(f"Optimal IoU Match Cost:     {best_final_config['iou_match_threshold']}")
    print(f"Optimal Dist Gate Scale:    {best_final_config['dist_gate_scale']}")
    print(f"Calibrated R Diag:          {best_final_config['r_cov_diag']}")
    print("-" * 80)
    print(f"Validation Performance Comparison:")
    print(f"  • Untuned Baseline: Prec@20={baseline_metrics['prec_20']*100:.2f}%, IoU={baseline_metrics['mean_iou']:.3f}, CLE={baseline_metrics['mean_cle']:.2f}px, Jitter={baseline_metrics['mean_jitter']:.2f}px, VerifCont={baseline_metrics['verif_cont_50']*100:.2f}%")
    print(f"  • Tuned Kalman:     Prec@20={best_final_metrics['prec_20']*100:.2f}%, IoU={best_final_metrics['mean_iou']:.3f}, CLE={best_final_metrics['mean_cle']:.2f}px, Jitter={best_final_metrics['mean_jitter']:.2f}px, VerifCont={best_final_metrics['verif_cont_50']*100:.2f}%")
    print(f"  • Relative Gain:    Prec@20: {(best_final_metrics['prec_20'] - baseline_metrics['prec_20'])*100:+.2f}%, CLE: {best_final_metrics['mean_cle'] - baseline_metrics['mean_cle']:+.2f}px, Jitter: {(best_final_metrics['mean_jitter'] - baseline_metrics['mean_jitter'])/baseline_metrics['mean_jitter']*100:+.1f}%")
    print("=" * 80)

    tuning_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_weights": "perception/weights/yolov8n_drone_768_hybrid_nwd_c51_best.pt",
        "validation_videos": VAL_VIDEOS,
        "calibrated_R_diag": [float(v) for v in calibrated_R_diag],
        "optimal_parameters": best_final_config,
        "baseline_metrics": baseline_metrics,
        "tuned_metrics": best_final_metrics
    }

    results_file = OUTPUT_DIR / "calibrated_kalman_parameters.json"
    with open(results_file, "w") as f:
        json.dump(tuning_payload, f, indent=2)
    print(f"[SAVED] Tuned parameters written to: {results_file}")

    return tuning_payload


def main():
    parser = argparse.ArgumentParser(description="Calibrate and Tune Kalman Filter on Anti-UAV Validation Split")
    parser.add_argument(
        "--weights",
        type=str,
        default="perception/weights/yolov8n_drone_768_hybrid_nwd_c51_best.pt",
        help="YOLO model checkpoint path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Inference device"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=768,
        help="Input image resolution"
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputing validation detections instead of using cache"
    )
    args = parser.parse_args()

    # Step 1: Extract and cache detections
    val_data = extract_or_load_detections(
        weights_path=args.weights,
        device=args.device,
        imgsz=args.imgsz,
        force_recompute=args.force_recompute
    )

    # Step 2: Calibrate Measurement Noise Covariance Matrix R
    calib = calibrate_measurement_noise(val_data, conf_thresh=0.25, match_iou_thresh=0.25)

    # Step 3: Run hyperparameter sweep
    tuning_result = run_kalman_parameter_sweep(val_data, calib["recommended_R_diag"])


if __name__ == "__main__":
    main()
