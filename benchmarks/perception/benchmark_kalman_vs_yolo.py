#!/usr/bin/env python3
"""
Comprehensive Benchmark: Raw YOLO vs Kalman Filter on Anti-UAV Test Set.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception.

Benchmarks:
- Raw YOLO (Memoryless Spatial Detector)
- Tuned Kalman Filter (Calibrated R, Tuned Q & Lifecycle Hyperparameters)
- Untuned Baseline Kalman Filter (Reference Comparison)

Model: yolov8n_drone_768_hybrid_nwd_c51_best.pt
Test Sequences: video05, video12, video20 (3,570 frames)

Outputs:
- outputs/benchmarks/perception/drone_tracking/c51_hybrid_test_benchmark/benchmark_results.json
- outputs/benchmarks/perception/drone_tracking/c51_hybrid_test_benchmark/figures/
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# Add repository root to python path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.pipeline import DroneTrackingPipeline

TEST_VIDEOS = ["video05", "video12", "video20"]
VIDEOS_DIR = REPO_ROOT / "perception" / "raw_data" / "Anti-UAV" / "videos"
GT_DIR = REPO_ROOT / "perception" / "raw_data" / "Anti-UAV" / "Anti-UAV-Tracking-V0GT"
OUTPUT_BASE = REPO_ROOT / "outputs" / "benchmarks" / "perception" / "drone_tracking" / "c51_hybrid_test_benchmark"
TUNED_PARAMS_FILE = REPO_ROOT / "outputs" / "kalman_tuning" / "calibrated_kalman_parameters.json"


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


def run_benchmark_on_video(
    vid_name: str,
    cfg_name: str,
    cfg_label: str,
    pipeline_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    vid_path = VIDEOS_DIR / f"{vid_name}.mp4"
    gt_path = GT_DIR / f"{vid_name}_gt.txt"

    gt_boxes = load_gt_boxes(gt_path)
    cap = cv2.VideoCapture(str(vid_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pipeline = DroneTrackingPipeline(**pipeline_kwargs)

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
    succ_30_count = 0
    verified_lock_50_count = 0
    verified_lock_30_count = 0
    verified_cont_50_count = 0
    false_lock_count = 0
    false_alarm_count = 0
    missed_gt_count = 0
    coasting_maintained_count = 0

    trajectory_pts = []  # (f_idx, cx, cy)
    per_frame_records = []

    print(f"\n---> Running [{cfg_label}] on {vid_name} ({total_frames} frames)...")
    start_time = time.time()
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame without drawing HUD overhead
        _, telemetry = pipeline.process_frame(frame, draw_hud=False)

        gt_box = gt_boxes[frame_idx] if frame_idx < len(gt_boxes) else None
        pred_box = telemetry.get("bbox")
        status = telemetry.get("status", "SEARCHING")

        if status == "LOCKED":
            locked_frames += 1
        elif status == "COASTING":
            coasting_frames += 1
        else:
            lost_frames += 1

        cle = 100.0
        iou = 0.0
        p_cx, p_cy = None, None
        gt_cx, gt_cy = None, None

        if gt_box is not None:
            visible_frames += 1
            gt_cx = float((gt_box[0] + gt_box[2]) / 2.0)
            gt_cy = float((gt_box[1] + gt_box[3]) / 2.0)

            if pred_box is not None:
                p_box = np.array(pred_box, dtype=np.float32)
                p_cx = float((p_box[0] + p_box[2]) / 2.0)
                p_cy = float((p_box[1] + p_box[3]) / 2.0)

                cle = float(np.sqrt((p_cx - gt_cx) ** 2 + (p_cy - gt_cy) ** 2))
                iou = compute_iou(p_box, gt_box)

                cle_list.append(cle)
                iou_list.append(iou)

                if cle <= 20.0:
                    prec_20_count += 1
                if cle <= 50.0:
                    prec_50_count += 1
                if iou >= 0.50:
                    succ_50_count += 1
                if iou >= 0.30:
                    succ_30_count += 1

                if status == "LOCKED":
                    if iou >= 0.50:
                        verified_lock_50_count += 1
                    if iou >= 0.30:
                        verified_lock_30_count += 1
                    if iou < 0.10:
                        false_lock_count += 1

                if status in ["LOCKED", "COASTING"] and iou >= 0.50:
                    verified_cont_50_count += 1

                trajectory_pts.append((frame_idx, p_cx, p_cy))
            else:
                cle_list.append(100.0)  # Loss penalty
                iou_list.append(0.0)

            # Check dead-reckoning coasting recall when raw YOLO did not have an active detection
            if status == "COASTING":
                missed_gt_count += 1
                if pred_box is not None and compute_iou(np.array(pred_box, dtype=np.float32), gt_box) >= 0.30:
                    coasting_maintained_count += 1
        else:
            absent_frames += 1
            if pred_box is not None:
                false_alarm_count += 1
            if status == "LOCKED":
                false_lock_count += 1

        per_frame_records.append({
            "frame": frame_idx,
            "status": status,
            "has_gt": gt_box is not None,
            "has_pred": pred_box is not None,
            "cle": cle if gt_box is not None else None,
            "iou": iou if gt_box is not None else None,
            "pred_cx": p_cx,
            "pred_cy": p_cy,
            "gt_cx": gt_cx,
            "gt_cy": gt_cy
        })

        frame_idx += 1
        if frame_idx % 400 == 0 or frame_idx == total_frames:
            el = time.time() - start_time
            print(f"     Frame {frame_idx}/{total_frames} | Prec@20: {prec_20_count/max(1, visible_frames)*100:.1f}% | {frame_idx/max(1e-3, el):.1f} FPS")

    cap.release()
    total_time = time.time() - start_time
    avg_fps = frame_idx / max(1e-3, total_time)

    # Compute Spatial Jitter Index (Second-order temporal jerk)
    jitter_vals = []
    for i in range(1, len(trajectory_pts) - 1):
        f_prev, x_prev, y_prev = trajectory_pts[i - 1]
        f_curr, x_curr, y_curr = trajectory_pts[i]
        f_next, x_next, y_next = trajectory_pts[i + 1]
        if f_curr == f_prev + 1 and f_next == f_curr + 1:
            jerk_x = x_next - 2.0 * x_curr + x_prev
            jerk_y = y_next - 2.0 * y_curr + y_prev
            jitter_vals.append(float(np.sqrt(jerk_x ** 2 + jerk_y ** 2)))

    mean_jitter = float(np.mean(jitter_vals)) if len(jitter_vals) > 0 else 0.0

    # Compute Success Curve AUC (IoU thresholds 0.0 to 1.0)
    iou_arr = np.array(iou_list) if len(iou_list) > 0 else np.array([0.0])
    thresholds = np.linspace(0.0, 1.0, 101)
    success_curve = [float(np.mean(iou_arr >= t)) for t in thresholds]
    iou_auc = float(np.mean(success_curve))

    metrics = {
        "video": vid_name,
        "config": cfg_name,
        "label": cfg_label,
        "total_frames": frame_idx,
        "visible_frames": visible_frames,
        "absent_frames": absent_frames,
        "locked_frames": locked_frames,
        "coasting_frames": coasting_frames,
        "lost_frames": lost_frames,
        "raw_lock_rate": float(locked_frames / max(1, frame_idx)),
        "coasting_rate": float(coasting_frames / max(1, frame_idx)),
        "tracking_continuity": float((locked_frames + coasting_frames) / max(1, frame_idx)),
        "verified_lock_rate_50": float(verified_lock_50_count / max(1, visible_frames)),
        "verified_lock_rate_30": float(verified_lock_30_count / max(1, visible_frames)),
        "verified_continuity_50": float(verified_cont_50_count / max(1, visible_frames)),
        "false_lock_rate": float(false_lock_count / max(1, locked_frames)),
        "precision_20px": float(prec_20_count / max(1, visible_frames)),
        "precision_50px": float(prec_50_count / max(1, visible_frames)),
        "success_rate_50": float(succ_50_count / max(1, visible_frames)),
        "success_rate_30": float(succ_30_count / max(1, visible_frames)),
        "mean_iou": float(np.mean(iou_list)) if iou_list else 0.0,
        "iou_auc": iou_auc,
        "mean_cle": float(np.mean(cle_list)) if cle_list else 100.0,
        "median_cle": float(np.median(cle_list)) if cle_list else 100.0,
        "spatial_jitter_index": mean_jitter,
        "coasting_recall": float(coasting_maintained_count / max(1, missed_gt_count)),
        "false_alarms": false_alarm_count,
        "avg_fps": float(avg_fps),
        "per_frame_records": per_frame_records,
        "success_curve": success_curve
    }

    print(f"  --> Completed {vid_name} [{cfg_name}]:")
    print(f"      Prec@20: {metrics['precision_20px']*100:.2f}% | Mean IoU: {metrics['mean_iou']:.3f} | CLE: {metrics['mean_cle']:.2f}px | Jitter: {mean_jitter:.2f}px | FPS: {avg_fps:.1f}")

    return metrics


def generate_benchmark_figures(results: List[Dict[str, Any]], out_dir: Path):
    """Generates publication-quality comparative plots across test configurations."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    # 1. Spatial Jitter Index Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    videos = TEST_VIDEOS
    configs = sorted(list(set(r["config"] for r in results)))
    colors = {"raw_yolo": "#EF4444", "kalman_baseline": "#3B82F6", "kalman_tuned": "#10B981"}
    labels = {
        "raw_yolo": "Raw YOLO (No Kalman)",
        "kalman_baseline": "YOLO + Baseline Kalman",
        "kalman_tuned": "YOLO + Tuned Kalman"
    }

    x = np.arange(len(videos))
    width = 0.25 if len(configs) == 3 else 0.35

    for idx, cfg in enumerate(configs):
        jitters = [
            next(r["spatial_jitter_index"] for r in results if r["video"] == v and r["config"] == cfg)
            for v in videos
        ]
        offset = (idx - (len(configs) - 1) / 2.0) * width
        bars = ax.bar(x + offset, jitters, width, label=labels.get(cfg, cfg), color=colors.get(cfg, "#6B7280"), alpha=0.9, edgecolor="none")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}px", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, weight="bold")

    ax.set_ylabel("Spatial Jitter Index (px/frame²)", fontsize=11, weight="bold")
    ax.set_title("Set-point Jitter Suppression: Raw YOLO vs Kalman Filter", fontsize=12, weight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}\n({next(r['total_frames'] for r in results if r['video'] == v)} frames)" for v in videos], fontsize=10)
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig_path = out_dir / "jitter_comparison.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"[PLOT] Saved jitter comparison to: {fig_path}")

    # 2. Precision & Success Comparison Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
    for idx, cfg in enumerate(configs):
        p20 = [
            next(r["precision_20px"] * 100 for r in results if r["video"] == v and r["config"] == cfg)
            for v in videos
        ]
        miou = [
            next(r["mean_iou"] * 100 for r in results if r["video"] == v and r["config"] == cfg)
            for v in videos
        ]
        offset = (idx - (len(configs) - 1) / 2.0) * width
        ax1.bar(x + offset, p20, width, label=labels.get(cfg, cfg), color=colors.get(cfg, "#6B7280"), alpha=0.9)
        ax2.bar(x + offset, miou, width, label=labels.get(cfg, cfg), color=colors.get(cfg, "#6B7280"), alpha=0.9)

    ax1.set_ylabel("Precision @ 20px (%)", fontsize=10, weight="bold")
    ax1.set_title("Tracking Accuracy (CLE ≤ 20px)", fontsize=11, weight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(videos)
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=8.5)

    ax2.set_ylabel("Mean IoU (%)", fontsize=10, weight="bold")
    ax2.set_title("Bounding Box Overlap", fontsize=11, weight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(videos)
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig_path2 = out_dir / "precision_and_iou_comparison.png"
    plt.savefig(fig_path2)
    plt.close()
    print(f"[PLOT] Saved precision and IoU comparison to: {fig_path2}")

    # 3. Trajectory Jitter Trace Snippet for video20 (frames 200-350)
    vid_rec = [r for r in results if r["video"] == "video20"]
    if len(vid_rec) >= 2:
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
        start_f, end_f = 200, 350
        for r in vid_rec:
            cfg = r["config"]
            pts = [rec for rec in r["per_frame_records"] if start_f <= rec["frame"] <= end_f and rec["has_pred"]]
            fx = [p["frame"] for p in pts]
            cy = [p["pred_cy"] for p in pts]
            ls = "-" if cfg == "kalman_tuned" else ("--" if cfg == "kalman_baseline" else ":")
            lw = 2.2 if cfg == "kalman_tuned" else 1.5
            ax.plot(fx, cy, label=labels.get(cfg, cfg), color=colors.get(cfg, "black"), linestyle=ls, linewidth=lw)

        # Plot GT
        gt_pts = [rec for rec in vid_rec[0]["per_frame_records"] if start_f <= rec["frame"] <= end_f and rec["has_gt"]]
        ax.plot([p["frame"] for p in gt_pts], [p["gt_cy"] for p in gt_pts], label="Ground Truth", color="#1E293B", linewidth=1.2, alpha=0.6)

        ax.set_xlabel("Frame Index", fontsize=10, weight="bold")
        ax.set_ylabel("Centroid Y Position (px)", fontsize=10, weight="bold")
        ax.set_title("Centroid Y Trajectory Smoothing under High-Frequency Jitter (video20)", fontsize=11, weight="bold")
        ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        fig_path3 = out_dir / "trajectory_smoothing_trace.png"
        plt.savefig(fig_path3)
        plt.close()
        print(f"[PLOT] Saved trajectory trace to: {fig_path3}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Raw YOLO vs Kalman Filter on Anti-UAV Test Set")
    parser.add_argument(
        "--weights",
        type=str,
        default="perception/weights/yolov8n_drone_768_hybrid_nwd_c51_best.pt",
        help="Model weights path"
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
        help="Inference image resolution"
    )
    parser.add_argument(
        "--include-baseline-kalman",
        action="store_true",
        default=True,
        help="Include untuned baseline Kalman filter for 3-way comparative ablation"
    )
    args = parser.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Load tuned parameters if available
    tuned_params = {}
    if TUNED_PARAMS_FILE.exists():
        with open(TUNED_PARAMS_FILE, "r") as f:
            t_data = json.load(f)
            tuned_params = t_data.get("optimal_parameters", {})
            print(f"[TUNING] Loaded calibrated Kalman parameters from: {TUNED_PARAMS_FILE}")
    else:
        print(f"[WARN] Tuned parameters file {TUNED_PARAMS_FILE} not found, using calibrated defaults.")

    conf_thresh = tuned_params.get("conf_threshold", 0.20)
    q_pos = tuned_params.get("q_pos", 1.0)
    q_vel = tuned_params.get("q_vel", 30.0)
    q_dim = tuned_params.get("q_dim", 2.0)
    q_scale_rate = tuned_params.get("q_scale_rate", 10.0)
    max_lost = tuned_params.get("max_lost_frames", 15)
    min_hits = tuned_params.get("min_hits_to_confirm", 2)
    dist_gate = tuned_params.get("dist_gate_scale", 2.5)
    r_diag = tuned_params.get("r_cov_diag", None)
    r_cov = np.diag(r_diag).astype(np.float32) if r_diag is not None else None

    # Define Configurations
    configurations = [
        {
            "name": "raw_yolo",
            "label": "Raw YOLOv8 (No Kalman)",
            "pipeline_kwargs": {
                "weights": args.weights,
                "conf_threshold": conf_thresh,
                "enable_kalman": False,
                "enable_dynamic_zoom": False,
                "imgsz": args.imgsz,
                "device": args.device
            }
        },
        {
            "name": "kalman_tuned",
            "label": "YOLOv8 + Tuned Kalman Filter",
            "pipeline_kwargs": {
                "weights": args.weights,
                "conf_threshold": conf_thresh,
                "enable_kalman": True,
                "enable_dynamic_zoom": False,
                "q_pos": q_pos,
                "q_vel": q_vel,
                "q_dim": q_dim,
                "q_scale_rate": q_scale_rate,
                "max_lost_frames": max_lost,
                "min_hits_to_confirm": min_hits,
                "dist_gate_scale": dist_gate,
                "r_cov": r_cov,
                "imgsz": args.imgsz,
                "device": args.device
            }
        }
    ]

    if args.include_baseline_kalman:
        configurations.insert(1, {
            "name": "kalman_baseline",
            "label": "YOLOv8 + Untuned Baseline Kalman",
            "pipeline_kwargs": {
                "weights": args.weights,
                "conf_threshold": 0.20,
                "enable_kalman": True,
                "enable_dynamic_zoom": False,
                "q_pos": 1.0,
                "q_vel": 18.0,
                "r_pos": 2.0,
                "max_lost_frames": 15,
                "min_hits_to_confirm": 2,
                "imgsz": args.imgsz,
                "device": args.device
            }
        })

    print("=" * 80)
    print("STARTING TEST BENCHMARK: RAW YOLO VS KALMAN FILTER")
    print(f"Model Checkpoint: {args.weights}")
    print(f"Test Videos:      {TEST_VIDEOS} (Total 3,570 frames)")
    print(f"Configurations:   {[c['name'] for c in configurations]}")
    print(f"Device:           {args.device} | Imgsz: {args.imgsz}")
    print("=" * 80)

    all_results = []
    for cfg in configurations:
        for vid in TEST_VIDEOS:
            res = run_benchmark_on_video(vid, cfg["name"], cfg["label"], cfg["pipeline_kwargs"])
            all_results.append(res)

    # Save benchmark results JSON (excluding heavy per-frame records in main summary)
    summary_results = []
    for r in all_results:
        rec_copy = {k: v for k, v in r.items() if k not in ["per_frame_records", "success_curve"]}
        summary_results.append(rec_copy)

    results_file = OUTPUT_BASE / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\n[SAVED] Benchmark summary results saved to: {results_file}")

    # Generate comparative figures
    figures_dir = OUTPUT_BASE / "figures"
    generate_benchmark_figures(all_results, figures_dir)

    # Print Aggregate Comparative Table
    print("\n" + "=" * 105)
    print(f"{'Configuration':<30} | {'Prec@20px':<10} | {'Prec@50px':<10} | {'Mean IoU':<10} | {'Mean CLE':<10} | {'Jitter (px)':<12} | {'Continuity':<10}")
    print("-" * 105)
    for cfg in configurations:
        cfg_name = cfg["name"]
        cfg_res = [r for r in summary_results if r["config"] == cfg_name]
        tot_vis = sum(r["visible_frames"] for r in cfg_res)
        tot_frames = sum(r["total_frames"] for r in cfg_res)
        avg_p20 = sum(r["precision_20px"] * r["visible_frames"] for r in cfg_res) / max(1, tot_vis)
        avg_p50 = sum(r["precision_50px"] * r["visible_frames"] for r in cfg_res) / max(1, tot_vis)
        avg_iou = sum(r["mean_iou"] * r["visible_frames"] for r in cfg_res) / max(1, tot_vis)
        avg_cle = sum(r["mean_cle"] * r["visible_frames"] for r in cfg_res) / max(1, tot_vis)
        avg_jit = np.mean([r["spatial_jitter_index"] for r in cfg_res])
        avg_cont = sum(r["tracking_continuity"] * r["total_frames"] for r in cfg_res) / max(1, tot_frames)

        print(f"{cfg['label']:<30} | {avg_p20*100:8.2f}% | {avg_p50*100:8.2f}% | {avg_iou:8.3f}   | {avg_cle:8.2f}px | {avg_jit:10.2f}px | {avg_cont*100:8.2f}%")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
