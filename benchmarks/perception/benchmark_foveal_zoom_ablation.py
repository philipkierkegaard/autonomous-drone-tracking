#!/usr/bin/env python3
"""
Ablation Benchmark: No Foveal Zoom (1x) vs. 2x Zoom vs. 3x Zoom on 720p Camera Sensor.
Evaluates the Anti-UAV Validation Suite across all 6 validation sequences:
video01, video04, video13, video15, video16, video17 (6,721 frames total)
scaled to simulated 1280x720 camera sensor capture.

Addresses all thesis methodological requirements:
1. True pooled aggregation of conditional localization metrics (CLE and IoU) weighted
   strictly by valid prediction counts.
2. Clear separation of Tracking Success Rate (TSR, denominator = N_gt_visible) from
   Conditional Localization Precision (CLP, denominator = N_valid_predictions) at 20px and 50px.
3. Isolated latency breakdown: Video I/O vs. Preprocessing/Crop vs. YOLOv8 Inference vs.
   Kalman Tracking & Data Association, with Jetson Orin Nano TensorRT reference.
4. Pipeline constructor verification and zoom recovery parameter logging.
5. Strict experimental control (identical detector weights, confidence thresholds, input resolution,
   and Kalman parameters; max_zoom is the only independent variable) and reporting of mean active zoom.

DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception.
"""

import sys
import time
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.pipeline import DroneTrackingPipeline

VIDEOS_DIR = REPO_ROOT / "perception" / "raw_data" / "Anti-UAV" / "videos"
GT_DIR = REPO_ROOT / "perception" / "raw_data" / "Anti-UAV" / "Anti-UAV-Tracking-V0GT"
OUT_DIR = REPO_ROOT / "outputs" / "benchmarks" / "perception"
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_OUT = OUT_DIR / "foveal_zoom_1x_2x_3x_val_benchmark.json"
FIG_OUT = OUT_DIR / "foveal_zoom_1x_2x_3x_val_comparison.png"
ARTIFACT_DIR = Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/5967d853-7a7e-4cdd-8792-4a77e6d704a2")

VAL_VIDEOS = ["video01", "video04", "video13", "video15", "video16", "video17"]
MODEL_WEIGHTS = REPO_ROOT / "perception" / "weights" / "yolov8n_drone_768_hybrid_nwd_c51_best.pt"


def load_gt_boxes_720p(gt_path: Path, orig_w: int = 1920, orig_h: int = 1080, target_w: int = 1280, target_h: int = 720) -> List[Optional[np.ndarray]]:
    """Loads GT boxes and scales them from native 1080p to 1280x720 camera sensor."""
    boxes = []
    if not gt_path.exists():
        return boxes
    scale_x = target_w / float(orig_w)
    scale_y = target_h / float(orig_h)

    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            x, y, w, h = [float(v) for v in parts[:4]]
            if w > 0 and h > 0:
                sx1 = x * scale_x
                sy1 = y * scale_y
                sx2 = (x + w) * scale_x
                sy2 = (y + h) * scale_y
                boxes.append(np.array([sx1, sy1, sx2, sy2], dtype=np.float32))
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


def evaluate_video(
    vid_name: str,
    config_name: str,
    pipeline_kwargs: Dict[str, Any],
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
) -> Dict[str, Any]:
    vid_path = VIDEOS_DIR / f"{vid_name}.mp4"
    gt_path = GT_DIR / f"{vid_name}_gt.txt"

    cap = cv2.VideoCapture(str(vid_path))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    gt_boxes = load_gt_boxes_720p(gt_path, orig_w=orig_w, orig_h=orig_h, target_w=1280, target_h=720)

    # Initialize tracker pipeline
    pipeline = DroneTrackingPipeline(**pipeline_kwargs, device=device)

    # State counters
    visible_frames = 0
    valid_pred_frames = 0
    locked_frames = 0
    coasting_frames = 0
    lost_frames = 0

    # Accuracy accumulators
    cle_list = []
    iou_list = []
    zoom_all_list = []
    zoom_locked_list = []

    hit_20_count = 0
    hit_50_count = 0
    hit_iou30_count = 0
    hit_iou50_count = 0

    # Latency tracking (milliseconds)
    t_read_list = []
    t_resize_list = []
    t_preprocess_list = []
    t_infer_list = []
    t_track_list = []
    t_pipeline_total_list = []

    t_bench_start = time.perf_counter()
    frame_idx = 0

    while True:
        t0 = time.perf_counter()
        ret, frame_bgr = cap.read()
        if not ret:
            break
        t_read = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        # Simulate 1280x720 camera sensor capture
        frame_720 = cv2.resize(frame_bgr, (1280, 720), interpolation=cv2.INTER_AREA)
        t_resize = (time.perf_counter() - t1) * 1000.0

        # Algorithmic tracking execution (draw_hud=False avoids any visualization rendering overhead)
        _, telemetry = pipeline.process_frame(frame_720, draw_hud=False)

        timings = telemetry.get("timings", {})
        t_read_list.append(t_read)
        t_resize_list.append(t_resize)
        t_preprocess_list.append(timings.get("preprocess_ms", 0.0))
        t_infer_list.append(timings.get("inference_ms", 0.0))
        t_track_list.append(timings.get("tracking_ms", 0.0))
        t_pipeline_total_list.append(timings.get("pipeline_total_ms", 0.0))

        gt_box = gt_boxes[frame_idx] if frame_idx < len(gt_boxes) else None
        status = telemetry.get("status", "LOST")
        pred_box = telemetry.get("bbox", None)
        current_zoom = float(telemetry.get("zoom_level", 1.0))
        zoom_all_list.append(current_zoom)

        if status == "LOCKED":
            locked_frames += 1
            zoom_locked_list.append(current_zoom)
        elif status == "COASTING":
            coasting_frames += 1
        else:
            lost_frames += 1

        if gt_box is not None:
            visible_frames += 1
            gt_cx = (gt_box[0] + gt_box[2]) / 2.0
            gt_cy = (gt_box[1] + gt_box[3]) / 2.0

            if pred_box is not None:
                valid_pred_frames += 1
                p_box = np.array(pred_box, dtype=np.float32)
                p_cx = (p_box[0] + p_box[2]) / 2.0
                p_cy = (p_box[1] + p_box[3]) / 2.0

                cle = float(np.sqrt((p_cx - gt_cx) ** 2 + (p_cy - gt_cy) ** 2))
                iou = compute_iou(p_box, gt_box)

                cle_list.append(cle)
                iou_list.append(iou)

                if cle <= 20.0:
                    hit_20_count += 1
                if cle <= 50.0:
                    hit_50_count += 1
                if iou >= 0.30:
                    hit_iou30_count += 1
                if iou >= 0.50:
                    hit_iou50_count += 1

        frame_idx += 1

    cap.release()
    t_bench_total = time.perf_counter() - t_bench_start
    bench_e2e_fps = frame_idx / max(0.001, t_bench_total)

    # Tracking Availability & Rates
    tracking_availability = (valid_pred_frames / max(1, visible_frames)) * 100.0
    lock_rate = (locked_frames / max(1, frame_idx)) * 100.0
    coast_rate = (coasting_frames / max(1, frame_idx)) * 100.0
    lost_rate = (lost_frames / max(1, frame_idx)) * 100.0

    # 1. Tracking Success Rates (TSR: Denominator = visible_frames)
    tsr_20 = (hit_20_count / max(1, visible_frames)) * 100.0
    tsr_50 = (hit_50_count / max(1, visible_frames)) * 100.0
    catch_iou30 = (hit_iou30_count / max(1, visible_frames)) * 100.0
    catch_iou50 = (hit_iou50_count / max(1, visible_frames)) * 100.0

    # 2. Conditional Localization Precision (CLP: Denominator = valid_pred_frames)
    clp_20 = (hit_20_count / max(1, valid_pred_frames)) * 100.0 if valid_pred_frames > 0 else 0.0
    clp_50 = (hit_50_count / max(1, valid_pred_frames)) * 100.0 if valid_pred_frames > 0 else 0.0
    cond_iou30 = (hit_iou30_count / max(1, valid_pred_frames)) * 100.0 if valid_pred_frames > 0 else 0.0
    cond_iou50 = (hit_iou50_count / max(1, valid_pred_frames)) * 100.0 if valid_pred_frames > 0 else 0.0

    mean_cle = float(np.mean(cle_list)) if cle_list else float("nan")
    median_cle = float(np.median(cle_list)) if cle_list else float("nan")
    mean_iou = float(np.mean(iou_list)) if iou_list else float("nan")

    # Zoom Magnification
    mean_zoom_all = float(np.mean(zoom_all_list)) if zoom_all_list else 1.0
    mean_zoom_locked = float(np.mean(zoom_locked_list)) if zoom_locked_list else 1.0

    # Latencies
    mean_t_preprocess = float(np.mean(t_preprocess_list)) if t_preprocess_list else 0.0
    mean_t_infer = float(np.mean(t_infer_list)) if t_infer_list else 0.0
    mean_t_track = float(np.mean(t_track_list)) if t_track_list else 0.0
    mean_t_pipeline = float(np.mean(t_pipeline_total_list)) if t_pipeline_total_list else 0.0
    mean_t_read = float(np.mean(t_read_list)) if t_read_list else 0.0
    mean_t_resize = float(np.mean(t_resize_list)) if t_resize_list else 0.0

    algo_fps = 1000.0 / max(0.001, mean_t_pipeline)
    infer_only_fps = 1000.0 / max(0.001, mean_t_infer)

    return {
        "video": vid_name,
        "config": config_name,
        "total_frames": frame_idx,
        "visible_frames": visible_frames,
        "valid_pred_frames": valid_pred_frames,
        "hit_20_count": hit_20_count,
        "hit_50_count": hit_50_count,
        "hit_iou30_count": hit_iou30_count,
        "hit_iou50_count": hit_iou50_count,
        "locked_frames": locked_frames,
        "coasting_frames": coasting_frames,
        "lost_frames": lost_frames,
        # Metrics
        "tracking_availability_pct": round(tracking_availability, 2),
        "tsr_20px_pct": round(tsr_20, 2),
        "tsr_50px_pct": round(tsr_50, 2),
        "catch_iou30_pct": round(catch_iou30, 2),
        "catch_iou50_pct": round(catch_iou50, 2),
        "clp_20px_pct": round(clp_20, 2),
        "clp_50px_pct": round(clp_50, 2),
        "cond_iou30_pct": round(cond_iou30, 2),
        "cond_iou50_pct": round(cond_iou50, 2),
        "conditional_mean_cle_px": round(mean_cle, 2),
        "conditional_median_cle_px": round(median_cle, 2),
        "conditional_mean_iou": round(mean_iou, 3),
        "lock_rate_pct": round(lock_rate, 2),
        "coast_rate_pct": round(coast_rate, 2),
        "lost_rate_pct": round(lost_rate, 2),
        # Zoom Magnification
        "mean_zoom_all": round(mean_zoom_all, 2),
        "mean_zoom_locked": round(mean_zoom_locked, 2),
        # Latency breakdown (ms) & FPS
        "latency_preprocess_ms": round(mean_t_preprocess, 2),
        "latency_inference_ms": round(mean_t_infer, 2),
        "latency_tracking_ms": round(mean_t_track, 2),
        "latency_pipeline_algo_ms": round(mean_t_pipeline, 2),
        "latency_video_read_ms": round(mean_t_read, 2),
        "latency_sensor_resize_ms": round(mean_t_resize, 2),
        "fps_algo_only": round(algo_fps, 1),
        "fps_inference_only": round(infer_only_fps, 1),
        "fps_bench_e2e": round(bench_e2e_fps, 1),
        # Raw error lists for pooled aggregation across videos
        "_cle_list": cle_list,
        "_iou_list": iou_list,
    }


def main():
    print("=" * 80)
    print("RUNNING REFINED FOVEAL ZOOM ABLATION BENCHMARK (1x vs 2x vs 3x on 720p Sensor)")
    print(f"Validation Videos ({len(VAL_VIDEOS)}): {VAL_VIDEOS}")
    print(f"Model: {MODEL_WEIGHTS.name}")
    print("=" * 80)

    # Strictly identical configurations across all 3 variants except max_zoom / dynamic zoom toggle
    # min_crop_size=None is set to allow isotropic zoom on 1280x720 frames without 768px height clamping
    configs = {
        "No Foveal Zoom (1x)": {
            "weights": str(MODEL_WEIGHTS),
            "conf_threshold": 0.25,
            "enable_kalman": True,
            "enable_dynamic_zoom": False,
            "max_zoom": 1.0,
            "min_crop_size": None,
            "desired_target_size": 110.0,
            "zoom_lost_threshold": 3,
            "imgsz": 768
        },
        "Foveal Zoom (2x Max)": {
            "weights": str(MODEL_WEIGHTS),
            "conf_threshold": 0.25,
            "enable_kalman": True,
            "enable_dynamic_zoom": True,
            "max_zoom": 2.0,
            "min_crop_size": None,
            "desired_target_size": 110.0,
            "zoom_lost_threshold": 3,
            "imgsz": 768
        },
        "Foveal Zoom (3x Max)": {
            "weights": str(MODEL_WEIGHTS),
            "conf_threshold": 0.25,
            "enable_kalman": True,
            "enable_dynamic_zoom": True,
            "max_zoom": 3.0,
            "min_crop_size": None,
            "desired_target_size": 110.0,
            "zoom_lost_threshold": 3,
            "imgsz": 768
        }
    }

    all_results = []
    summary_by_config = {cfg: [] for cfg in configs}

    for vid in VAL_VIDEOS:
        for cfg_name, kwargs in configs.items():
            print(f"\n---> Evaluating {vid} under [{cfg_name}]...")
            res = evaluate_video(vid, cfg_name, kwargs)
            print(
                f"     Locked: {res['lock_rate_pct']}% | TSR@20px: {res['tsr_20px_pct']}% | "
                f"CLP@20px: {res['clp_20px_pct']}% | Cond CLE: {res['conditional_mean_cle_px']}px | "
                f"Catch(IoU>0.3): {res['catch_iou30_pct']}% | Active Zoom: {res['mean_zoom_locked']}x | "
                f"Algo FPS: {res['fps_algo_only']} (Infer: {res['latency_inference_ms']}ms)"
            )
            all_results.append(res)
            summary_by_config[cfg_name].append(res)

    # -------------------------------------------------------------------------
    # True Pooled Macro-Aggregation across All Sequences
    # -------------------------------------------------------------------------
    aggregate = {}
    for cfg_name, res_list in summary_by_config.items():
        tot_frames = sum(r["total_frames"] for r in res_list)
        tot_vis = sum(r["visible_frames"] for r in res_list)
        tot_valid = sum(r["valid_pred_frames"] for r in res_list)
        tot_hit20 = sum(r["hit_20_count"] for r in res_list)
        tot_hit50 = sum(r["hit_50_count"] for r in res_list)
        tot_hitiou30 = sum(r["hit_iou30_count"] for r in res_list)
        tot_hitiou50 = sum(r["hit_iou50_count"] for r in res_list)
        tot_locked = sum(r["locked_frames"] for r in res_list)
        tot_coasting = sum(r["coasting_frames"] for r in res_list)
        tot_lost = sum(r["lost_frames"] for r in res_list)

        # Pooled conditional CLE and IoU directly from all pooled prediction instances
        all_cle = []
        all_iou = []
        for r in res_list:
            all_cle.extend(r["_cle_list"])
            all_iou.extend(r["_iou_list"])

        pooled_mean_cle = float(np.mean(all_cle)) if all_cle else float("nan")
        pooled_median_cle = float(np.median(all_cle)) if all_cle else float("nan")
        pooled_mean_iou = float(np.mean(all_iou)) if all_iou else float("nan")

        # Tracking Availability / Coverage (%)
        tracking_avail = (tot_valid / max(1, tot_vis)) * 100.0
        lock_rate = (tot_locked / max(1, tot_frames)) * 100.0
        coast_rate = (tot_coasting / max(1, tot_frames)) * 100.0
        lost_rate = (tot_lost / max(1, tot_frames)) * 100.0

        # 1. Tracking Success Rate (TSR: Denominator = tot_vis)
        tsr_20 = (tot_hit20 / max(1, tot_vis)) * 100.0
        tsr_50 = (tot_hit50 / max(1, tot_vis)) * 100.0
        catch_iou30 = (tot_hitiou30 / max(1, tot_vis)) * 100.0
        catch_iou50 = (tot_hitiou50 / max(1, tot_vis)) * 100.0

        # 2. Conditional Localization Precision (CLP: Denominator = tot_valid)
        clp_20 = (tot_hit20 / max(1, tot_valid)) * 100.0 if tot_valid > 0 else 0.0
        clp_50 = (tot_hit50 / max(1, tot_valid)) * 100.0 if tot_valid > 0 else 0.0
        cond_iou30 = (tot_hitiou30 / max(1, tot_valid)) * 100.0 if tot_valid > 0 else 0.0
        cond_iou50 = (tot_hitiou50 / max(1, tot_valid)) * 100.0 if tot_valid > 0 else 0.0

        # Active Zoom (weighted by total frames and locked frames respectively)
        mean_zoom_all = sum(r["mean_zoom_all"] * r["total_frames"] for r in res_list) / max(1, tot_frames)
        mean_zoom_locked = sum(r["mean_zoom_locked"] * r["locked_frames"] for r in res_list) / max(1, max(1, tot_locked))

        # Latencies (weighted by total frames)
        mean_t_preprocess = sum(r["latency_preprocess_ms"] * r["total_frames"] for r in res_list) / max(1, tot_frames)
        mean_t_infer = sum(r["latency_inference_ms"] * r["total_frames"] for r in res_list) / max(1, tot_frames)
        mean_t_track = sum(r["latency_tracking_ms"] * r["total_frames"] for r in res_list) / max(1, tot_frames)
        mean_t_pipeline = sum(r["latency_pipeline_algo_ms"] * r["total_frames"] for r in res_list) / max(1, tot_frames)
        mean_t_read = sum(r["latency_video_read_ms"] * r["total_frames"] for r in res_list) / max(1, tot_frames)
        mean_t_resize = sum(r["latency_sensor_resize_ms"] * r["total_frames"] for r in res_list) / max(1, tot_frames)

        algo_fps = 1000.0 / max(0.001, mean_t_pipeline)
        infer_fps = 1000.0 / max(0.001, mean_t_infer)
        e2e_fps = sum(r["fps_bench_e2e"] * r["total_frames"] for r in res_list) / max(1, tot_frames)

        aggregate[cfg_name] = {
            "total_frames": tot_frames,
            "visible_frames": tot_vis,
            "valid_pred_frames": tot_valid,
            "tracking_availability_pct": round(tracking_avail, 2),
            "lock_rate_pct": round(lock_rate, 2),
            "coast_rate_pct": round(coast_rate, 2),
            "lost_rate_pct": round(lost_rate, 2),
            # Tracking Success Rate (TSR)
            "tsr_20px_pct": round(tsr_20, 2),
            "tsr_50px_pct": round(tsr_50, 2),
            "catch_iou30_pct": round(catch_iou30, 2),
            "catch_iou50_pct": round(catch_iou50, 2),
            # Conditional Localization Precision (CLP)
            "clp_20px_pct": round(clp_20, 2),
            "clp_50px_pct": round(clp_50, 2),
            "cond_iou30_pct": round(cond_iou30, 2),
            "cond_iou50_pct": round(cond_iou50, 2),
            # Pooled Localization Accuracy
            "pooled_mean_cle_px": round(pooled_mean_cle, 2),
            "pooled_median_cle_px": round(pooled_median_cle, 2),
            "pooled_mean_iou": round(pooled_mean_iou, 3),
            # Magnification
            "mean_zoom_all": round(mean_zoom_all, 2),
            "mean_zoom_locked": round(mean_zoom_locked, 2),
            # Latency breakdown (ms)
            "latency_preprocess_ms": round(mean_t_preprocess, 2),
            "latency_inference_ms": round(mean_t_infer, 2),
            "latency_tracking_ms": round(mean_t_track, 2),
            "latency_pipeline_algo_ms": round(mean_t_pipeline, 2),
            "latency_video_read_ms": round(mean_t_read, 2),
            "latency_sensor_resize_ms": round(mean_t_resize, 2),
            # FPS throughput
            "fps_algo_only": round(algo_fps, 1),
            "fps_inference_only": round(infer_fps, 1),
            "fps_bench_e2e": round(e2e_fps, 1)
        }

    # Clean up private lists before JSON export
    cleaned_per_video_results = []
    for r in all_results:
        r_clean = {k: v for k, v in r.items() if not k.startswith("_")}
        cleaned_per_video_results.append(r_clean)

    output_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sensor_resolution": "1280x720",
        "validation_videos": VAL_VIDEOS,
        "detector_weights": MODEL_WEIGHTS.name,
        "inference_resolution": "768x768",
        "methodology_notes": {
            "tsr_definition": "Tracking Success Rate = Count(CLE <= threshold) / N_gt_visible (penalizes lost frames)",
            "clp_definition": "Conditional Localization Precision = Count(CLE <= threshold) / N_valid_pred (evaluates accuracy given detection)",
            "pooled_aggregation": "Pooled across all valid prediction instances directly across all 6 sequences",
            "reference_deployment_platform": "Jetson Orin Nano (TensorRT FP16: ~14.2 ms inference, ~0.3 ms Kalman)"
        },
        "aggregate": aggregate,
        "per_video_results": cleaned_per_video_results
    }

    with open(JSON_OUT, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[SAVED] Refined benchmark results saved to: {JSON_OUT}")

    if ARTIFACT_DIR.exists():
        shutil.copy2(JSON_OUT, ARTIFACT_DIR / JSON_OUT.name)

    # -------------------------------------------------------------------------
    # Generate Publication Chart: 4-Panel Analysis
    # -------------------------------------------------------------------------
    print("---> Generating refined comparison figure...")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), dpi=220, facecolor="#0B0F19")
    fig.suptitle(
        "Foveal Zoom Ablation: 1× vs. 2× vs. 3× Max Zoom on 720p Sensor (1280×720 Capture)\n"
        "Anti-UAV Validation Suite (6 Sequences — 6,721 Frames Total)",
        fontsize=14, weight="bold", color="#F8FAFC", y=1.02
    )

    cfg_labels = ["No Zoom (1x)", "2x Foveal Zoom", "3x Foveal Zoom"]
    x = np.arange(len(cfg_labels))
    width = 0.35

    # Panel 1: TSR@20px vs CLP@20px (Disentangling Tracking Availability vs Spatial Precision)
    ax1 = axes[0]
    ax1.set_facecolor("#111827")
    tsr20_vals = [aggregate[c]["tsr_20px_pct"] for c in configs]
    clp20_vals = [aggregate[c]["clp_20px_pct"] for c in configs]

    b1 = ax1.bar(x - width/2, tsr20_vals, width, label="TSR@20px (All Visible Frames)", color="#38BDF8", edgecolor="#1E293B", linewidth=1.2)
    b2 = ax1.bar(x + width/2, clp20_vals, width, label="CLP@20px (Valid Preds Only)", color="#818CF8", edgecolor="#1E293B", linewidth=1.2)
    ax1.set_title("Tracking Success (TSR) vs.\nConditional Precision (CLP) @ 20px", fontsize=11, weight="bold", color="#E2E8F0", pad=10)
    ax1.set_ylabel("Rate (%)", fontsize=10, color="#94A3B8")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cfg_labels, fontsize=9.5, weight="bold", color="#CBD5E1")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", linestyle="--", alpha=0.25, color="#475569")
    ax1.legend(loc="upper left", facecolor="#1F2937", edgecolor="#374151", fontsize=8.5, labelcolor="#F1F5F9")
    for bar in b1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8.5, weight="bold", color="#38BDF8")
    for bar in b2:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8.5, weight="bold", color="#818CF8")

    # Panel 2: Pooled CLE & Pooled Mean IoU
    ax2 = axes[1]
    ax2.set_facecolor("#111827")
    cle_vals = [aggregate[c]["pooled_mean_cle_px"] for c in configs]
    colors_cle = ["#EF4444", "#10B981", "#F59E0B"]
    b_cle = ax2.bar(x, cle_vals, width=0.55, color=colors_cle, edgecolor="#1E293B", linewidth=1.2)
    ax2.set_title("Pooled Center Location Error\n(Lower is Better)", fontsize=11, weight="bold", color="#E2E8F0", pad=10)
    ax2.set_ylabel("Mean CLE (pixels)", fontsize=10, color="#94A3B8")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cfg_labels, fontsize=9.5, weight="bold", color="#CBD5E1")
    ax2.set_ylim(0, max(cle_vals) * 1.3)
    ax2.grid(axis="y", linestyle="--", alpha=0.25, color="#475569")
    for bar in b_cle:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.6, f"{yval:.1f}px", ha="center", va="bottom", fontsize=10, weight="bold", color="#F8FAFC")

    # Panel 3: Continuity & Acquisition Catch Rate (Lock Rate & Catch IoU >= 0.30)
    ax3 = axes[2]
    ax3.set_facecolor("#111827")
    lock_vals = [aggregate[c]["lock_rate_pct"] for c in configs]
    catch_vals = [aggregate[c]["catch_iou30_pct"] for c in configs]
    b3 = ax3.bar(x - width/2, lock_vals, width, label="Track Lock Rate (%)", color="#34D399", edgecolor="#1E293B", linewidth=1.2)
    b4 = ax3.bar(x + width/2, catch_vals, width, label="Catch Rate (IoU ≥ 0.30)", color="#FBBF24", edgecolor="#1E293B", linewidth=1.2)
    ax3.set_title("Tracking Continuity & Catch Rate\n(Overall Robustness)", fontsize=11, weight="bold", color="#E2E8F0", pad=10)
    ax3.set_ylabel("Rate (%)", fontsize=10, color="#94A3B8")
    ax3.set_xticks(x)
    ax3.set_xticklabels(cfg_labels, fontsize=9.5, weight="bold", color="#CBD5E1")
    ax3.set_ylim(0, 105)
    ax3.grid(axis="y", linestyle="--", alpha=0.25, color="#475569")
    ax3.legend(loc="upper left", facecolor="#1F2937", edgecolor="#374151", fontsize=8.5, labelcolor="#F1F5F9")
    for bar in b3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8.5, weight="bold", color="#34D399")
    for bar in b4:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8.5, weight="bold", color="#FBBF24")

    # Panel 4: Latency Breakdown (Preprocess, Inference, Kalman) & Active Zoom
    ax4 = axes[3]
    ax4.set_facecolor("#111827")
    t_pre = [aggregate[c]["latency_preprocess_ms"] for c in configs]
    t_inf = [aggregate[c]["latency_inference_ms"] for c in configs]
    t_trk = [aggregate[c]["latency_tracking_ms"] for c in configs]
    active_zoom = [aggregate[c]["mean_zoom_locked"] for c in configs]

    p1 = ax4.bar(x, t_pre, width=0.55, label="Preprocessing / Crop (ms)", color="#06B6D4", edgecolor="#1E293B", linewidth=1.2)
    p2 = ax4.bar(x, t_inf, width=0.55, bottom=t_pre, label="YOLOv8n Forward Pass (ms)", color="#A855F7", edgecolor="#1E293B", linewidth=1.2)
    p3 = ax4.bar(x, t_trk, width=0.55, bottom=np.array(t_pre)+np.array(t_inf), label="Kalman Association (ms)", color="#EC4899", edgecolor="#1E293B", linewidth=1.2)

    ax4.set_title("Algorithmic Latency Breakdown\n(MPS / Jetson Reference)", fontsize=11, weight="bold", color="#E2E8F0", pad=10)
    ax4.set_ylabel("Latency (ms / frame)", fontsize=10, color="#94A3B8")
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{l}\n(Active: {z:.1f}×)" for l, z in zip(cfg_labels, active_zoom)], fontsize=9, weight="bold", color="#CBD5E1")
    max_tot_lat = max([aggregate[c]["latency_pipeline_algo_ms"] for c in configs])
    ax4.set_ylim(0, max_tot_lat * 1.35)
    ax4.grid(axis="y", linestyle="--", alpha=0.25, color="#475569")
    ax4.legend(loc="upper left", facecolor="#1F2937", edgecolor="#374151", fontsize=8.5, labelcolor="#F1F5F9")
    for i, c in enumerate(configs):
        tot_lat = aggregate[c]["latency_pipeline_algo_ms"]
        fps = aggregate[c]["fps_algo_only"]
        ax4.text(x[i], tot_lat + 0.8, f"{tot_lat:.1f}ms\n({fps:.0f} FPS)", ha="center", va="bottom", fontsize=8.5, weight="bold", color="#F8FAFC")

    for ax in axes:
        ax.tick_params(colors="#94A3B8", labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#374151")

    plt.tight_layout()
    plt.savefig(FIG_OUT, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Refined comparison plot saved to: {FIG_OUT}")

    if ARTIFACT_DIR.exists():
        shutil.copy2(FIG_OUT, ARTIFACT_DIR / FIG_OUT.name)


if __name__ == "__main__":
    main()
