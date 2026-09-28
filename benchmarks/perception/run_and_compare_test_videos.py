#!/usr/bin/env python3
"""
Comprehensive Benchmarking & Video Generation Script
Evaluates YOLOv8n-drone v1 vs v3 with and without Kalman filtering
on the Anti-UAV test set sequential videos (video05, video12, video20).
"""

import os
import sys
import time
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np

# Add object-detection directory to path
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from pipeline import DroneTrackingPipeline
VIDEOS_DIR = BASE_DIR / "data/Anti-UAV/videos"
GT_DIR = BASE_DIR / "data/Anti-UAV/Anti-UAV-Tracking-V0GT"
OUTPUTS_BASE = ROOT_DIR / "outputs/drone_tracking/test_comparison"

V1_WEIGHTS = BASE_DIR / "runs/detect/yolov8n_drone/weights/best.pt"
V3_WEIGHTS = BASE_DIR / "runs/detect/yolov8n_drone_v3/weights/best.pt"

TEST_VIDEOS = ["video05", "video12", "video20"]

CONFIGURATIONS = [
    {
        "name": "v3_kalman",
        "label": "YOLOv8 v3 + Kalman",
        "weights": V3_WEIGHTS,
        "kalman": True,
        "conf": 0.25
    },
    {
        "name": "v3_no_kalman",
        "label": "YOLOv8 v3 (Raw YOLO, No Kalman)",
        "weights": V3_WEIGHTS,
        "kalman": False,
        "conf": 0.25
    },
    {
        "name": "v1_kalman",
        "label": "YOLOv8 v1 + Kalman",
        "weights": V1_WEIGHTS,
        "kalman": True,
        "conf": 0.25
    },
    {
        "name": "v1_no_kalman",
        "label": "YOLOv8 v1 (Raw YOLO, No Kalman)",
        "weights": V1_WEIGHTS,
        "kalman": False,
        "conf": 0.25
    },
]


def load_gt(gt_path: Path) -> List[np.ndarray]:
    """
    Loads Anti-UAV ground truth [x, y, w, h] format.
    Returns list of [x1, y1, x2, y2] or None if target is absent.
    """
    boxes = []
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


def run_single_benchmark(cfg: Dict[str, Any], vid_name: str, force_video: bool = True) -> Dict[str, Any]:
    vid_file = VIDEOS_DIR / f"{vid_name}.mp4"
    gt_file = GT_DIR / f"{vid_name}_gt.txt"
    
    out_dir = OUTPUTS_BASE / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / f"{vid_name}_{cfg['name']}.mp4"
    temp_raw = out_dir / f"temp_raw_{vid_name}_{cfg['name']}.mp4"

    gt_boxes = load_gt(gt_file)
    
    cap = cv2.VideoCapture(str(vid_file))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pipeline = DroneTrackingPipeline(
        weights=str(cfg["weights"]),
        conf_threshold=cfg["conf"],
        enable_dynamic_zoom=False,  # baseline wide full frame
        enable_kalman=cfg["kalman"]
    )

    frame_idx = 0
    start_time = time.time()
    
    # Tracking evaluation accumulators
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
    false_alarm_count = 0

    # Ground-Truth Verified Tracking Accumulators
    verified_lock_50_count = 0      # LOCKED and IoU >= 0.50
    verified_lock_30_count = 0      # LOCKED and IoU >= 0.30
    verified_cont_50_count = 0      # (LOCKED or COASTING) and IoU >= 0.50
    false_lock_count = 0            # LOCKED but target absent or IoU < 0.10

    print(f"\n---> Running [{cfg['label']}] on {vid_name} ({total_frames} frames)...")

    # Encode video if forced, missing, or empty (<1KB)
    needs_video = force_video or (not out_video.exists()) or (out_video.stat().st_size < 1000)
    save_video = needs_video
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_raw), fourcc, fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        annotated, telemetry = pipeline.process_frame(frame, draw_hud=save_video)
        if writer is not None:
            writer.write(annotated)

        gt_box = gt_boxes[frame_idx] if frame_idx < len(gt_boxes) else None
        pred_box = telemetry.get("bbox")
        status = telemetry.get("status", "SEARCHING")

        if status == "LOCKED":
            locked_frames += 1
        elif status == "COASTING":
            coasting_frames += 1
        else:
            lost_frames += 1

        iou = 0.0
        if gt_box is not None:
            visible_frames += 1
            gt_cx = (gt_box[0] + gt_box[2]) / 2.0
            gt_cy = (gt_box[1] + gt_box[3]) / 2.0

            if pred_box is not None:
                p_box = np.array(pred_box, dtype=np.float32)
                p_cx = (p_box[0] + p_box[2]) / 2.0
                p_cy = (p_box[1] + p_box[3]) / 2.0

                cle = float(np.sqrt((p_cx - gt_cx)**2 + (p_cy - gt_cy)**2))
                iou = compute_iou(p_box, gt_box)

                cle_list.append(cle)
                iou_list.append(iou)

                if cle <= 20.0:
                    prec_20_count += 1
                if cle <= 50.0:
                    prec_50_count += 1
                if iou >= 0.50:
                    succ_50_count += 1
                
                # Check ground-truth verified lock
                if status == "LOCKED":
                    if iou >= 0.50:
                        verified_lock_50_count += 1
                    if iou >= 0.30:
                        verified_lock_30_count += 1
                    if iou < 0.10:
                        false_lock_count += 1
                if status in ["LOCKED", "COASTING"] and iou >= 0.50:
                    verified_cont_50_count += 1
            else:
                cle_list.append(100.0)  # penalty
                iou_list.append(0.0)
        else:
            absent_frames += 1
            if pred_box is not None:
                false_alarm_count += 1
            if status == "LOCKED":
                false_lock_count += 1

        frame_idx += 1
        if frame_idx % 300 == 0 or frame_idx == total_frames:
            elapsed = time.time() - start_time
            current_fps = frame_idx / max(1e-3, elapsed)
            print(f"     [{vid_name}] Frame {frame_idx}/{total_frames} | Lock: {locked_frames/frame_idx*100:.1f}% | Verif(0.5): {verified_lock_50_count/max(1, visible_frames)*100:.1f}% | {current_fps:.1f} FPS")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"     Converting to QuickTime-compatible H.264 (yuv420p): {out_video.name}...")
        try:
            cmd = [
                "ffmpeg", "-y", "-i", str(temp_raw),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(out_video)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if temp_raw.exists():
                temp_raw.unlink()
        except Exception as e:
            print(f"     Warning: ffmpeg transcode failed ({e}), saving raw MP4.")
            if temp_raw.exists():
                if out_video.exists():
                    out_video.unlink()
                shutil.move(str(temp_raw), str(out_video))

    total_time = time.time() - start_time
    avg_fps = frame_idx / max(1e-3, total_time)

    metrics = {
        "video": vid_name,
        "config": cfg["name"],
        "label": cfg["label"],
        "weights": str(cfg["weights"].name),
        "kalman": cfg["kalman"],
        "total_frames": frame_idx,
        "visible_frames": visible_frames,
        "absent_frames": absent_frames,
        "locked_frames": locked_frames,
        "coasting_frames": coasting_frames,
        "lost_frames": lost_frames,
        "raw_lock_rate": float(locked_frames / max(1, frame_idx)),
        "verified_lock_rate_50": float(verified_lock_50_count / max(1, visible_frames)),
        "verified_lock_rate_30": float(verified_lock_30_count / max(1, visible_frames)),
        "verified_continuity_50": float(verified_cont_50_count / max(1, visible_frames)),
        "false_lock_rate": float(false_lock_count / max(1, locked_frames)),
        "coasting_rate": float(coasting_frames / max(1, frame_idx)),
        "tracking_continuity": float((locked_frames + coasting_frames) / max(1, frame_idx)),
        "precision_20px": float(prec_20_count / max(1, visible_frames)),
        "precision_50px": float(prec_50_count / max(1, visible_frames)),
        "success_rate_50": float(succ_50_count / max(1, visible_frames)),
        "mean_iou": float(np.mean(iou_list)) if iou_list else 0.0,
        "mean_cle": float(np.mean(cle_list)) if cle_list else 0.0,
        "false_alarms": false_alarm_count,
        "avg_fps": float(avg_fps),
        "video_output": str(out_video)
    }

    print(f"  --> Completed {vid_name} [{cfg['name']}]:")
    print(f"      Raw Lock: {metrics['raw_lock_rate']*100:.1f}% | Verified Lock (IoU>=0.5): {metrics['verified_lock_rate_50']*100:.1f}% | Verified (IoU>=0.3): {metrics['verified_lock_rate_30']*100:.1f}%")
    print(f"      Prec@20px: {metrics['precision_20px']*100:.1f}% | Mean IoU: {metrics['mean_iou']:.3f} | Speed: {avg_fps:.1f} FPS")
    print(f"      Video saved: {out_video.name}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Autonomous Drone Tracking Benchmark & Video Generation")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[c["name"] for c in CONFIGURATIONS],
        choices=[c["name"] for c in CONFIGURATIONS],
        help="Configurations to run (default: all 4)"
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        default=TEST_VIDEOS,
        choices=TEST_VIDEOS,
        help="Test videos to run (default: video05 video12 video20)"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video generation (metrics only)"
    )
    args = parser.parse_args()

    selected_cfgs = [c for c in CONFIGURATIONS if c["name"] in args.configs]

    print("=" * 80)
    print("  AUTONOMOUS DRONE TRACKING BENCHMARK: V1 VS V3 (KALMAN VS RAW YOLO)")
    print(f"  Test Videos: {args.videos}")
    print(f"  Configurations: {[c['name'] for c in selected_cfgs]}")
    print(f"  Generate Videos: {not args.no_video} (QuickTime H.264 yuv420p)")
    print(f"  Outputs Directory: {OUTPUTS_BASE}")
    print("=" * 80)

    results_file = OUTPUTS_BASE / "benchmark_results.json"
    existing_results = []
    if results_file.exists():
        try:
            with open(results_file, "r") as f:
                existing_results = json.load(f)
        except Exception:
            existing_results = []

    results_map = {(r.get("config"), r.get("video")): r for r in existing_results if "config" in r and "video" in r}

    for cfg in selected_cfgs:
        for vid in args.videos:
            res = run_single_benchmark(cfg, vid, force_video=not args.no_video)
            results_map[(res["config"], res["video"])] = res

    all_results = list(results_map.values())
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("  BENCHMARK & VIDEO GENERATION COMPLETED!")
    print(f"  Full JSON results saved to: {results_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
