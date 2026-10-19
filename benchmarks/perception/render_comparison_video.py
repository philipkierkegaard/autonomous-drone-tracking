#!/usr/bin/env python3
"""
Render Synchronized Side-by-Side Video: Raw YOLO vs Tuned Kalman Filter.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception.

Generates a 2-panel synchronized video demonstrating:
- Left: Raw YOLOv8n (Hybrid NWD C51, memoryless single-frame detection)
- Right: YOLOv8n + Tuned 8D Kalman Filter (Recursive Bayesian state estimation, smoothed trajectory, coasting)
"""

import os
import sys
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.pipeline import DroneTrackingPipeline

VIDEOS_DIR = REPO_ROOT / "perception" / "raw_data" / "Anti-UAV" / "videos"
GT_DIR = REPO_ROOT / "perception" / "raw_data" / "Anti-UAV" / "Anti-UAV-Tracking-V0GT"
OUTPUT_DIR = REPO_ROOT / "outputs" / "benchmarks" / "perception" / "drone_tracking" / "c51_hybrid_test_benchmark" / "videos"
TUNED_PARAMS_FILE = REPO_ROOT / "outputs" / "kalman_tuning" / "calibrated_kalman_parameters.json"


def load_gt_boxes(gt_path: Path):
    boxes = []
    if not gt_path.exists():
        return boxes
    with open(gt_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if not p:
                continue
            x, y, w, h = [float(v) for v in p[:4]]
            if w > 0 and h > 0:
                boxes.append(np.array([x, y, x + w, y + h], dtype=np.float32))
            else:
                boxes.append(None)
    return boxes


def render_comparison_video(vid_name: str = "video05", weights_path: str = "perception/weights/yolov8n_drone_768_hybrid_nwd_c51_best.pt"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_raw = OUTPUT_DIR / f"temp_{vid_name}_comparison.mp4"
    final_video = OUTPUT_DIR / f"{vid_name}_raw_vs_kalman_comparison.mp4"

    vid_path = VIDEOS_DIR / f"{vid_name}.mp4"
    gt_path = GT_DIR / f"{vid_name}_gt.txt"
    gt_boxes = load_gt_boxes(gt_path)

    # Load tuned parameters
    with open(TUNED_PARAMS_FILE, "r") as f:
        t_data = json.load(f)
    t_opt = t_data["optimal_parameters"]
    r_cov = np.diag(t_opt["r_cov_diag"]).astype(np.float32)

    # Initialize both pipelines
    pipe_raw = DroneTrackingPipeline(
        weights=weights_path,
        conf_threshold=t_opt["conf_threshold"],
        enable_kalman=False,
        enable_dynamic_zoom=False,
        imgsz=768,
        device="mps" if torch.backends.mps.is_available() else "cpu"
    )

    pipe_kalman = DroneTrackingPipeline(
        weights=weights_path,
        conf_threshold=t_opt["conf_threshold"],
        enable_kalman=True,
        enable_dynamic_zoom=False,
        q_pos=t_opt["q_pos"],
        q_vel=t_opt["q_vel"],
        q_dim=t_opt["q_dim"],
        q_scale_rate=t_opt["q_scale_rate"],
        max_lost_frames=t_opt["max_lost_frames"],
        min_hits_to_confirm=t_opt["min_hits_to_confirm"],
        dist_gate_scale=t_opt["dist_gate_scale"],
        r_cov=r_cov,
        imgsz=768,
        device="mps" if torch.backends.mps.is_available() else "cpu"
    )

    cap = cv2.VideoCapture(str(vid_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Downscale each panel slightly so side-by-side fits 1920x540 or 1280x360
    panel_w = 960
    panel_h = 540
    out_w = panel_w * 2
    out_h = panel_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_raw), fourcc, fps, (out_w, out_h))

    print(f"---> Rendering synchronized side-by-side video for {vid_name} ({total_frames} frames)...")
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process through both pipelines
        ann_raw, telem_raw = pipe_raw.process_frame(frame, draw_hud=True)
        ann_kf, telem_kf = pipe_kalman.process_frame(frame, draw_hud=True)

        gt_box = gt_boxes[frame_idx] if frame_idx < len(gt_boxes) else None

        # Draw Ground Truth on both panels in cyan
        if gt_box is not None:
            gx1, gy1, gx2, gy2 = [int(v) for v in gt_box]
            cv2.rectangle(ann_raw, (gx1, gy1), (gx2, gy2), (255, 230, 0), 1, cv2.LINE_AA)
            cv2.rectangle(ann_kf, (gx1, gy1), (gx2, gy2), (255, 230, 0), 1, cv2.LINE_AA)

        # Resize to panel dimensions
        panel_left = cv2.resize(ann_raw, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        panel_right = cv2.resize(ann_kf, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

        # Header overlays
        # Left Panel Banner
        cv2.rectangle(panel_left, (0, 0), (panel_w, 36), (15, 23, 42), -1)
        cv2.putText(panel_left, "BASELINE: Raw YOLOv8 (No Kalman)", (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (239, 68, 68), 2, cv2.LINE_AA)
        status_l = telem_raw.get("status", "SEARCHING")
        col_l = (46, 213, 115) if status_l == "LOCKED" else (239, 68, 68)
        cv2.putText(panel_left, f"STATUS: {status_l}", (panel_w - 220, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col_l, 2, cv2.LINE_AA)

        # Right Panel Banner
        cv2.rectangle(panel_right, (0, 0), (panel_w, 36), (15, 23, 42), -1)
        cv2.putText(panel_right, "TUNED: YOLOv8 + 8D Kalman State Estimator", (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (16, 185, 129), 2, cv2.LINE_AA)
        status_r = telem_kf.get("status", "SEARCHING")
        col_r = (46, 213, 115) if status_r == "LOCKED" else ((255, 165, 0) if status_r == "COASTING" else (239, 68, 68))
        cv2.putText(panel_right, f"STATUS: {status_r}", (panel_w - 220, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col_r, 2, cv2.LINE_AA)

        # Concatenate horizontally
        combined = np.hstack([panel_left, panel_right])
        # Separator line
        cv2.line(combined, (panel_w, 0), (panel_w, panel_h), (255, 255, 255), 2)

        writer.write(combined)
        frame_idx += 1
        if frame_idx % 100 == 0 or frame_idx == total_frames:
            print(f"     Rendered {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")

    cap.release()
    writer.release()

    # Transcode with ffmpeg to QuickTime H.264
    print(f"     Transcoding to QuickTime-compatible H.264: {final_video.name}...")
    cmd = [
        "ffmpeg", "-y", "-i", str(temp_raw),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(final_video)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if temp_raw.exists():
        temp_raw.unlink()
    print(f"[VIDEO COMPLETED] Saved side-by-side video to: {final_video}")


if __name__ == "__main__":
    render_comparison_video("video05")
