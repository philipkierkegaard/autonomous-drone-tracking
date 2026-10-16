"""
Empirical YOLOv8 Detection Error & Temporal Dropout Profiler.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception.

Analyzes the performance of the trained YOLOv8 model on unseen validation sequences to extract:
1. Size-conditioned detection probabilities P(detect | scale)
2. Size-conditioned 4D bounding box localization errors [dcx, dcy, dw, dh]:
   - Mean vector mu in normalized coordinates
   - Full 4x4 covariance matrix Sigma
3. Temporal dropout dynamics:
   - 2-State Markov Chain transition probabilities (p_drop, p_recover)
   - Run-length distribution of consecutive missed frames (burst dropouts)
   - Lag-1 residual temporal autocorrelation (colored noise)
4. False positive rate per frame (lambda_fp) across hard negatives and background frames
5. Per-sequence dataset breakdown (preserving dataset identities)

Outputs:
  - perception/preparation/empirical_perception_model.json
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


def box_iou_xywhn(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Computes pairwise IoU matrix between boxes1 and boxes2.
    boxes: Nx4 array of normalized [cx, cy, w, h].
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.empty((len(boxes1), len(boxes2)), dtype=np.float32)

    # Convert cx, cy, w, h to x1, y1, x2, y2
    b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2.0
    b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2.0
    b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2.0
    b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2.0

    b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2.0
    b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2.0
    b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2.0
    b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2.0

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

    return inter_area / np.maximum(1e-6, union_area)


def load_yolo_labels(label_path: Path) -> np.ndarray:
    """Reads normalized [cx, cy, w, h] from a YOLO txt annotation file."""
    if not label_path.exists():
        return np.empty((0, 4), dtype=np.float32)
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Format: class cx cy w h
                boxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return np.array(boxes, dtype=np.float32) if len(boxes) > 0 else np.empty((0, 4), dtype=np.float32)


def get_scale_bin(scale: float) -> str:
    """Classifies target scale max(w, h) into standardized bins."""
    if scale < 0.035:
        return "tiny"     # < ~22px on 640px sensor
    elif scale < 0.070:
        return "small"    # 22 - 45px
    elif scale < 0.120:
        return "medium"   # 45 - 77px
    else:
        return "large"    # > 77px


def profile_validation_data(
    weights_path: str = "perception/weights/yolov8n_drone_768_hybrid_nwd_c51_best.pt",
    images_dir: str = "perception/datasets/anti_uav_yolo/images/val",
    labels_dir: str = "perception/datasets/anti_uav_yolo/labels/val",
    output_json: str = "perception/preparation/empirical_perception_model.json",
    conf_thresh: float = 0.25,
    match_iou_thresh: float = 0.25,
    batch_size: int = 32,
    device: str = "mps"
):
    print("=" * 78)
    print("EMPIRICAL YOLO PERCEPTION ERROR & DROPOUT PROFILER")
    print("=" * 78)
    print(f"Model Checkpoint: {weights_path}")
    print(f"Validation Set:   {images_dir}")
    print(f"Confidence Gate:  {conf_thresh} | Match IoU Gate: {match_iou_thresh}")
    print(f"Execution Device: {device} | Batch Size: {batch_size}")

    model = YOLO(weights_path)
    img_dir = Path(images_dir)
    lbl_dir = Path(labels_dir)

    all_images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    print(f"Discovered {len(all_images)} validation frames.\n")

    # Group frames by sequence/source
    sequences = defaultdict(list)
    for img_p in all_images:
        name = img_p.stem
        # Extract sequence prefix (e.g. video01_00010 -> video01, hardneg_... -> hardneg)
        parts = name.split("_")
        prefix = parts[0]
        # Try to extract numerical frame index for chronological sorting
        m = re.search(r"\d+", parts[-1])
        idx = int(m.group(0)) if m else len(sequences[prefix])
        sequences[prefix].append((idx, img_p))

    # Sort each sequence chronologically
    for prefix in sequences:
        sequences[prefix].sort(key=lambda x: x[0])

    print("Sequence Breakdown:")
    for seq, items in sorted(sequences.items()):
        print(f"  • {seq:<12}: {len(items):5d} frames")

    # Metrics Accumulators
    # 1. Scale-conditioned statistics: {bin_name: {'total_gt': int, 'detected': int, 'residuals': []}}
    scale_bins = ["tiny", "small", "medium", "large"]
    bin_data = {
        b: {"total_gt": 0, "detected": 0, "residuals": []}
        for b in scale_bins
    }

    # 2. Sequence-level temporal statistics
    seq_stats = {}
    total_false_positives = 0
    total_frames_analyzed = len(all_images)
    
    # Global Pooled Markov transition counts & consecutive pairs
    pooled_n11, pooled_n10, pooled_n01, pooled_n00 = 0, 0, 0, 0
    pooled_bursts = []
    consecutive_res_pairs = []

    print("\nRunning batched YOLO inference across all validation sequences...")

    # Process sequence by sequence to maintain temporal state
    for seq_name, frame_list in sorted(sequences.items()):
        img_paths = [p for _, p in frame_list]
        seq_gt_count = 0
        seq_det_count = 0
        seq_fp_count = 0
        frame_data = {}  # idx -> {"has_gt": bool, "detected": bool, "res": np.ndarray or None}

        # Run batched inference on this sequence
        results = []
        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i : i + batch_size]
            batch_res = model.predict(
                batch_paths,
                imgsz=768,
                conf=conf_thresh,
                iou=0.45,
                device=device,
                verbose=False
            )
            results.extend(batch_res)

        for (idx, img_p), res in zip(frame_list, results):
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            gt_boxes = load_yolo_labels(lbl_p)

            # Extract predictions in normalized [cx, cy, w, h]
            if len(res.boxes) > 0:
                pred_boxes = res.boxes.xywhn.cpu().numpy()
            else:
                pred_boxes = np.empty((0, 4), dtype=np.float32)

            n_gt = len(gt_boxes)
            n_pred = len(pred_boxes)
            seq_gt_count += n_gt

            if n_gt == 0:
                # Pure background frame or target absent
                frame_data[idx] = {"has_gt": False, "detected": False, "res": None}
                seq_fp_count += n_pred
                continue

            # Hungarian matching on IoU cost
            matched = False
            res_vec = None
            if n_pred > 0:
                iou_mat = box_iou_xywhn(gt_boxes, pred_boxes)
                cost_mat = 1.0 - iou_mat
                row_ind, col_ind = linear_sum_assignment(cost_mat)

                for r, c in zip(row_ind, col_ind):
                    if iou_mat[r, c] >= match_iou_thresh:
                        matched = True
                        res_vec = pred_boxes[c] - gt_boxes[r]
                        gt_scale = max(gt_boxes[r, 2], gt_boxes[r, 3])
                        b_name = get_scale_bin(gt_scale)

                        bin_data[b_name]["detected"] += 1
                        bin_data[b_name]["residuals"].append(res_vec)
                        break

            # Scale bin total GT
            gt_scale = max(gt_boxes[0, 2], gt_boxes[0, 3])
            b_name = get_scale_bin(gt_scale)
            bin_data[b_name]["total_gt"] += 1

            if matched:
                seq_det_count += 1
                unmatched_preds = n_pred - 1
                frame_data[idx] = {"has_gt": True, "detected": True, "res": res_vec}
            else:
                unmatched_preds = n_pred
                frame_data[idx] = {"has_gt": True, "detected": False, "res": None}

            seq_fp_count += unmatched_preds

        total_false_positives += seq_fp_count

        # Compute strictly consecutive Markov dynamics
        seq_bursts = []
        curr_miss_len = 0
        n11, n10, n01, n00 = 0, 0, 0, 0
        sorted_indices = sorted(frame_data.keys())

        for i, idx in enumerate(sorted_indices):
            curr = frame_data[idx]
            if not curr["has_gt"]:
                if curr_miss_len > 0:
                    seq_bursts.append(curr_miss_len)
                    curr_miss_len = 0
                continue

            if not curr["detected"]:
                curr_miss_len += 1
            else:
                if curr_miss_len > 0:
                    seq_bursts.append(curr_miss_len)
                    curr_miss_len = 0

            # Transition check: strictly adjacent frame indices with target present in both
            if i > 0:
                prev_idx = sorted_indices[i - 1]
                prev = frame_data[prev_idx]
                if prev_idx == idx - 1 and prev["has_gt"]:
                    if prev["detected"] and curr["detected"]:
                        n11 += 1
                        consecutive_res_pairs.append((prev["res"], curr["res"]))
                    elif prev["detected"] and not curr["detected"]:
                        n10 += 1
                    elif not prev["detected"] and curr["detected"]:
                        n01 += 1
                    elif not prev["detected"] and not curr["detected"]:
                        n00 += 1

        if curr_miss_len > 0:
            seq_bursts.append(curr_miss_len)

        # Transition probabilities for this sequence
        p_drop = float(n10 / max(1, n11 + n10)) if (n11 + n10) > 0 else 0.0
        p_recover = float(n01 / max(1, n01 + n00)) if (n01 + n00) > 0 else 0.0
        mean_burst = float(np.mean(seq_bursts)) if len(seq_bursts) > 0 else 0.0
        max_burst = int(np.max(seq_bursts)) if len(seq_bursts) > 0 else 0

        seq_recall = float(seq_det_count / max(1, seq_gt_count)) * 100.0
        seq_fp_rate = float(seq_fp_count / max(1, len(img_paths)))

        seq_stats[seq_name] = {
            "frames": len(img_paths),
            "ground_truth_targets": seq_gt_count,
            "detected_targets": seq_det_count,
            "recall_pct": round(seq_recall, 2),
            "false_positives": seq_fp_count,
            "fp_rate_per_frame": round(seq_fp_rate, 4),
            "n11": n11,
            "n10": n10,
            "n01": n01,
            "n00": n00,
            "p_drop": round(p_drop, 4),
            "p_recover": round(p_recover, 4),
            "mean_dropout_burst_len": round(mean_burst, 2),
            "max_dropout_burst_len": max_burst,
            "burst_count": len(seq_bursts)
        }

        # Accumulate pooled transitions (excluding pure negative sets without targets)
        if seq_gt_count > 0:
            pooled_n11 += n11
            pooled_n10 += n10
            pooled_n01 += n01
            pooled_n00 += n00
            pooled_bursts.extend(seq_bursts)

    # --------------------------------------------------------------------------
    # Aggregate Scale-Conditioned Covariance Matrices & Detection Probabilities
    # --------------------------------------------------------------------------
    # Compute clean-sky vs tree-canopy false positive rates
    clean_fp = sum(s["false_positives"] for k, s in seq_stats.items() if k != "video16")
    clean_frames = sum(s["frames"] for k, s in seq_stats.items() if k != "video16")
    tree_fp = seq_stats.get("video16", {}).get("false_positives", 0)
    tree_frames = seq_stats.get("video16", {}).get("frames", 1)

    model_profile = {
        "metadata": {
            "model_weights": weights_path,
            "validation_images_count": total_frames_analyzed,
            "confidence_threshold": conf_thresh,
            "match_iou_threshold": match_iou_thresh,
            "global_fp_rate_per_frame": round(total_false_positives / max(1, total_frames_analyzed), 4),
            "clean_sky_fp_rate_per_frame": round(clean_fp / max(1, clean_frames), 4),
            "tree_canopy_fp_rate_per_frame": round(tree_fp / max(1, tree_frames), 4),
        },
        "scale_bins": {},
        "temporal_markov_model": {},
        "sequence_breakdown": seq_stats
    }

    print("\n" + "=" * 78)
    print("SCALE-CONDITIONED DETECTION PROBABILITY & 4D LOCALIZATION NOISE")
    print("=" * 78)
    print(f"{'Scale Bin':<8} | {'Scale Range':<15} | {'GT Count':<8} | {'P(detect)':<10} | {'sigma_cx (norm)':<14} | {'sigma_cy (norm)':<14} | {'sigma_w (norm)'}")
    print("-" * 78)

    scale_ranges = {
        "tiny": "[0.00, 0.035)",
        "small": "[0.035, 0.070)",
        "medium": "[0.070, 0.120)",
        "large": "[0.120, 1.000]"
    }

    for b in scale_bins:
        tot = bin_data[b]["total_gt"]
        det = bin_data[b]["detected"]
        res_list = bin_data[b]["residuals"]
        p_det = float(det / max(1, tot))

        if len(res_list) >= 4:
            res_arr = np.array(res_list, dtype=np.float64)
            mu = np.mean(res_arr, axis=0)
            cov = np.cov(res_arr, rowvar=False)
        else:
            mu = np.zeros(4, dtype=np.float64)
            cov = np.diag([1e-4, 1e-4, 1e-4, 1e-4])

        sig_cx = np.sqrt(cov[0, 0])
        sig_cy = np.sqrt(cov[1, 1])
        sig_w  = np.sqrt(cov[2, 2])
        sig_h  = np.sqrt(cov[3, 3])

        print(f"{b:<8} | {scale_ranges[b]:<15} | {tot:<8d} | {p_det*100:8.2f}%  | {sig_cx:14.5f} | {sig_cy:14.5f} | {sig_w:14.5f}")

        model_profile["scale_bins"][b] = {
            "scale_range": scale_ranges[b],
            "total_gt": tot,
            "detected": det,
            "p_detect": round(p_det, 4),
            "mean_residual": [round(float(v), 6) for v in mu],
            "cov_matrix_4x4": [[round(float(val), 8) for val in row] for row in cov],
            "std_devs": {
                "sigma_cx": round(float(sig_cx), 5),
                "sigma_cy": round(float(sig_cy), 5),
                "sigma_w":  round(float(sig_w), 5),
                "sigma_h":  round(float(sig_h), 5)
            }
        }

    # Maximum Likelihood Estimation of Global Markov Transitions (Pooled Counts)
    global_p_drop = float(pooled_n10 / max(1, pooled_n11 + pooled_n10))
    global_p_rec = float(pooled_n01 / max(1, pooled_n01 + pooled_n00))
    theoretical_mean_burst = float(1.0 / max(1e-6, global_p_rec))
    empirical_mean_burst = float(np.mean(pooled_bursts)) if len(pooled_bursts) > 0 else 0.0

    # Strictly consecutive lag-1 autocorrelation
    if len(consecutive_res_pairs) > 10:
        p_res = np.array([p[0] for p in consecutive_res_pairs], dtype=np.float64)
        c_res = np.array([p[1] for p in consecutive_res_pairs], dtype=np.float64)
        rho_cx_lag1 = float(np.corrcoef(p_res[:, 0], c_res[:, 0])[0, 1])
        rho_cy_lag1 = float(np.corrcoef(p_res[:, 1], c_res[:, 1])[0, 1])
    else:
        rho_cx_lag1, rho_cy_lag1 = 0.0, 0.0

    model_profile["temporal_markov_model"] = {
        "pooled_transitions": {
            "N11": pooled_n11,
            "N10": pooled_n10,
            "N01": pooled_n01,
            "N00": pooled_n00
        },
        "p_drop": round(global_p_drop, 4),
        "p_recover": round(global_p_rec, 4),
        "theoretical_mean_burst_len": round(theoretical_mean_burst, 2),
        "empirical_mean_burst_length_frames": round(empirical_mean_burst, 2),
        "expected_burst_duration_sec_at_30fps": round(empirical_mean_burst / 30.0, 3),
        "lag1_autocorrelation": {
            "rho_cx": round(rho_cx_lag1, 4),
            "rho_cy": round(rho_cy_lag1, 4),
            "consecutive_pairs_evaluated": len(consecutive_res_pairs)
        }
    }

    print("\n" + "=" * 78)
    print("GLOBAL TEMPORAL DROPOUT DYNAMICS (POOLED MARKOV MLE MODEL)")
    print("=" * 78)
    print(f"P(Drop | Seen):             {global_p_drop*100:.2f}% (Pooled N10={pooled_n10}, N11={pooled_n11})")
    print(f"P(Recover | Dropped):       {global_p_rec*100:.2f}% (Pooled N01={pooled_n01}, N00={pooled_n00})")
    print(f"Theoretical Mean Burst:     {theoretical_mean_burst:.2f} frames ({theoretical_mean_burst/30.0*1000:.0f} ms at 30 FPS)")
    print(f"Empirical Mean Burst:       {empirical_mean_burst:.2f} frames ({empirical_mean_burst/30.0*1000:.0f} ms at 30 FPS)")
    print(f"Consecutive Autocorrelation:rho_cx(1) = {rho_cx_lag1:+.3f}, rho_cy(1) = {rho_cy_lag1:+.3f} (N={len(consecutive_res_pairs)} pairs)")
    print(f"Clean-Sky False Alarm Rate: {model_profile['metadata']['clean_sky_fp_rate_per_frame']:.4f} per frame")
    print(f"Tree Canopy False Alarm:    {model_profile['metadata']['tree_canopy_fp_rate_per_frame']:.4f} per frame")
    print(f"Pooled Global FP Rate:      {model_profile['metadata']['global_fp_rate_per_frame']:.4f} per frame")

    # Save to JSON
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model_profile, f, indent=2)

    print(f"\n[SUCCESS] Successfully written empirical perception model to:")
    print(f"          {out_path.resolve()}\n")
    return model_profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile empirical YOLO perception errors on validation sequences.")
    parser.add_argument("--weights", type=str, default="perception/weights/yolov8n_drone_768_hybrid_nwd_c51_best.pt")
    parser.add_argument("--images", type=str, default="perception/datasets/anti_uav_yolo/images/val")
    parser.add_argument("--labels", type=str, default="perception/datasets/anti_uav_yolo/labels/val")
    parser.add_argument("--output", type=str, default="perception/preparation/empirical_perception_model.json")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.25)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    profile_validation_data(
        weights_path=args.weights,
        images_dir=args.images,
        labels_dir=args.labels,
        output_json=args.output,
        conf_thresh=args.conf,
        match_iou_thresh=args.iou,
        batch_size=args.batch,
        device=args.device
    )
