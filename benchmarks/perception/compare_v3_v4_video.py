#!/usr/bin/env python3
"""
Comparative Video Benchmark: YOLOv8n v3 Baseline (640px) vs YOLOv8n v4 Production (768px + NWD)
Evaluates detection persistence, confidence, clutter robustness, and trajectory tail on physical flight footage.
"""

import sys
import time
import argparse
from pathlib import Path
from collections import deque
import cv2
import numpy as np
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
V3_WEIGHTS = REPO_ROOT / "perception" / "weights" / "yolov8n_v3_best.pt"
V4_WEIGHTS = REPO_ROOT / "perception" / "weights" / "yolov8n_drone_v4_continued_best.pt"
OUTPUT_DIR = REPO_ROOT / "outputs" / "evaluation" / "v3_vs_v4_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_hud_panel(frame, model_label, boxes, confs, color_bgr, imgsz, trajectory):
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (18, 22, 28), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    # Status & Header
    has_det = len(boxes) > 0
    status_text = f"LOCKED ({len(boxes)})" if has_det else "SEARCHING"
    status_color = (46, 213, 115) if has_det else (255, 71, 87)

    cv2.putText(annotated, model_label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, f"[{imgsz}px]", (w - 220, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 170, 185), 1, cv2.LINE_AA)
    cv2.putText(annotated, status_text, (w - 125, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

    # Draw Motion Trajectory Tail (Breadcrumb History)
    if len(trajectory) > 1:
        pts = list(trajectory)
        n_pts = len(pts)
        for i in range(1, n_pts):
            # Gradient alpha thickness
            frac = i / float(n_pts)
            thickness = max(1, int(3 * frac))
            # Blend color towards white at head
            tail_color = tuple(int(c * (0.4 + 0.6 * frac)) for c in color_bgr)
            cv2.line(annotated, pts[i - 1], pts[i], tail_color, thickness, cv2.LINE_AA)
            if i % 3 == 0 or i == n_pts - 1:
                cv2.circle(annotated, pts[i], 2, (255, 255, 255), -1, cv2.LINE_AA)

    # Draw Detections
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2, cv2.LINE_AA)

        # Corner accents
        cw, ch = 12, 12
        cv2.line(annotated, (x1, y1), (x1 + cw, y1), (255, 255, 255), 2)
        cv2.line(annotated, (x1, y1), (x1, y1 + ch), (255, 255, 255), 2)
        cv2.line(annotated, (x2, y1), (x2 - cw, y1), (255, 255, 255), 2)
        cv2.line(annotated, (x2, y1), (x2, y1 + ch), (255, 255, 255), 2)
        cv2.line(annotated, (x1, y2), (x1 + cw, y2), (255, 255, 255), 2)
        cv2.line(annotated, (x1, y2), (x1, y2 - ch), (255, 255, 255), 2)
        cv2.line(annotated, (x2, y2), (x2 - cw, y2), (255, 255, 255), 2)
        cv2.line(annotated, (x2, y2), (x2, y2 - ch), (255, 255, 255), 2)

        # Label tag
        lbl = f"DRONE {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color_bgr, -1)
        cv2.putText(annotated, lbl, (x1 + 4, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (15, 15, 15), 1, cv2.LINE_AA)

    return annotated


def run_comparison(video_path: Path, conf_thresh: float = 0.20):
    print("=" * 72)
    print("    DRONE DETECTION COMPARISON: YOLOv8n v3 (640px) vs v4 (768px+NWD)   ")
    print("=" * 72)
    print(f"Target Video: {video_path}")
    print(f"v3 Weights:   {V3_WEIGHTS}")
    print(f"v4 Weights:   {V4_WEIGHTS}")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}\n")

    print("[INIT] Loading YOLOv8n v3 Baseline (640px)...")
    model_v3 = YOLO(str(V3_WEIGHTS))

    print("[INIT] Loading YOLOv8n v4 Production (768px + NWD)...")
    model_v4 = YOLO(str(V4_WEIGHTS))

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[VIDEO] Stream: {w}x{h} @ {fps:.1f} FPS | Total Frames: {total_frames}\n")

    # Output Side-by-Side Video
    sbs_w = w * 2
    sbs_h = h
    stem = video_path.stem.replace("flight_raw_", "").replace("flight_", "")
    out_video_path = OUTPUT_DIR / f"v3_vs_v4_sbs_{stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (sbs_w, sbs_h))

    # Trajectory Tails (History of Center Positions)
    v3_trajectory = deque(maxlen=35)
    v4_trajectory = deque(maxlen=35)
    v3_miss_count = 0
    v4_miss_count = 0

    # Metrics Accumulators
    v3_detected_frames = 0
    v4_detected_frames = 0
    both_detected_frames = 0
    v4_only_frames = 0
    v3_only_frames = 0
    neither_frames = 0

    v3_confs = []
    v4_confs = []

    v3_streak = 0
    v4_streak = 0
    max_v3_streak = 0
    max_v4_streak = 0

    notable_saved = 0
    frame_idx = 0
    t_start_all = time.perf_counter()

    print(f"[BENCHMARK] Processing {total_frames} frames (conf >= {conf_thresh})...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference v3 (640px)
        res_v3 = model_v3.predict(source=frame, imgsz=640, conf=conf_thresh, device=device, verbose=False)
        boxes_v3 = res_v3[0].boxes.xyxy.cpu().numpy() if len(res_v3[0].boxes) > 0 else np.empty((0, 4))
        confs_v3 = res_v3[0].boxes.conf.cpu().numpy() if len(res_v3[0].boxes) > 0 else np.empty((0,))

        # Inference v4 (768px)
        res_v4 = model_v4.predict(source=frame, imgsz=768, conf=conf_thresh, device=device, verbose=False)
        boxes_v4 = res_v4[0].boxes.xyxy.cpu().numpy() if len(res_v4[0].boxes) > 0 else np.empty((0, 4))
        confs_v4 = res_v4[0].boxes.conf.cpu().numpy() if len(res_v4[0].boxes) > 0 else np.empty((0,))

        v3_has = len(boxes_v3) > 0
        v4_has = len(boxes_v4) > 0

        # Update Trajectory Tails
        if v3_has:
            v3_miss_count = 0
            cx = int((boxes_v3[0][0] + boxes_v3[0][2]) / 2.0)
            cy = int((boxes_v3[0][1] + boxes_v3[0][3]) / 2.0)
            v3_trajectory.append((cx, cy))
        else:
            v3_miss_count += 1
            if v3_miss_count > 15:
                v3_trajectory.clear()

        if v4_has:
            v4_miss_count = 0
            cx = int((boxes_v4[0][0] + boxes_v4[0][2]) / 2.0)
            cy = int((boxes_v4[0][1] + boxes_v4[0][3]) / 2.0)
            v4_trajectory.append((cx, cy))
        else:
            v4_miss_count += 1
            if v4_miss_count > 15:
                v4_trajectory.clear()

        # Stats tracking
        if v3_has:
            v3_detected_frames += 1
            v3_confs.append(float(np.max(confs_v3)))
            v3_streak += 1
            max_v3_streak = max(max_v3_streak, v3_streak)
        else:
            v3_streak = 0

        if v4_has:
            v4_detected_frames += 1
            v4_confs.append(float(np.max(confs_v4)))
            v4_streak += 1
            max_v4_streak = max(max_v4_streak, v4_streak)
        else:
            v4_streak = 0

        if v3_has and v4_has:
            both_detected_frames += 1
        elif v4_has and not v3_has:
            v4_only_frames += 1
        elif v3_has and not v4_has:
            v3_only_frames += 1
        else:
            neither_frames += 1

        # Draw Side-by-Side Panels with Trajectory Tails
        panel_v3 = draw_hud_panel(frame, "YOLOv8n v3 (Baseline)", boxes_v3, confs_v3, (0, 210, 211), 640, v3_trajectory)
        panel_v4 = draw_hud_panel(frame, "YOLOv8n v4 (Production + NWD)", boxes_v4, confs_v4, (46, 213, 115), 768, v4_trajectory)

        sbs_frame = np.hstack([panel_v3, panel_v4])
        writer.write(sbs_frame)

        # Save Key Stills for Inspection
        if (v4_has and not v3_has and notable_saved < 5) or (v4_has and len(v4_trajectory) >= 15 and notable_saved < 8):
            still_path = OUTPUT_DIR / f"{stem}_frame_{frame_idx:04d}_sbs_tail.jpg"
            cv2.imwrite(str(still_path), sbs_frame)
            notable_saved += 1

        frame_idx += 1
        if frame_idx % 200 == 0 or frame_idx == total_frames:
            elapsed = time.perf_counter() - t_start_all
            current_fps = frame_idx / max(1e-4, elapsed)
            print(f"  Frame {frame_idx:4d} / {total_frames} | Progress: {frame_idx / total_frames * 100:5.1f}% | "
                  f"v3 Hits: {v3_detected_frames:4d} | v4 Hits: {v4_detected_frames:4d} | Processing Speed: {current_fps:.1f} FPS")

    cap.release()
    writer.release()

    total_time = time.perf_counter() - t_start_all

    pct_v3 = (v3_detected_frames / total_frames) * 100.0
    pct_v4 = (v4_detected_frames / total_frames) * 100.0
    mean_c_v3 = np.mean(v3_confs) if v3_confs else 0.0
    mean_c_v4 = np.mean(v4_confs) if v4_confs else 0.0
    p95_c_v3 = np.percentile(v3_confs, 95) if v3_confs else 0.0
    p95_c_v4 = np.percentile(v4_confs, 95) if v4_confs else 0.0
    p99_c_v3 = np.percentile(v3_confs, 99) if v3_confs else 0.0
    p99_c_v4 = np.percentile(v4_confs, 99) if v4_confs else 0.0

    print("\n" + "=" * 72)
    print("                 FINAL COMPARATIVE BENCHMARK RESULTS                    ")
    print("=" * 72)
    print(f"• Video Analyzed:               {video_path.name}")
    print(f"• Total Analyzed Frames:        {total_frames} ({total_frames / fps:.1f}s)")
    print(f"• Processing Time:              {total_time:.1f}s ({total_frames / total_time:.1f} FPS)\n")
    print(f"• YOLOv8n v3 (640px) Frames:    {v3_detected_frames:4d} / {total_frames} ({pct_v3:.1f}%)")
    print(f"  - Mean Detection Confidence:  {mean_c_v3:.3f}")
    print(f"  - Tail Confidence (P95 / P99):{p95_c_v3:.3f} / {p99_c_v3:.3f}")
    print(f"  - Max Continuous Lock Streak: {max_v3_streak} frames ({max_v3_streak / fps:.2f}s)\n")
    print(f"• YOLOv8n v4 (768px) Frames:    {v4_detected_frames:4d} / {total_frames} ({pct_v4:.1f}%)")
    print(f"  - Mean Detection Confidence:  {mean_c_v4:.3f}")
    print(f"  - Tail Confidence (P95 / P99):{p95_c_v4:.3f} / {p99_c_v4:.3f}")
    print(f"  - Max Continuous Lock Streak: {max_v4_streak} frames ({max_v4_streak / fps:.2f}s)\n")
    print("────────────────────────────────────────────────────────────────────────")
    print(f"• Both Models Locked:           {both_detected_frames:4d} frames ({(both_detected_frames / total_frames) * 100:.1f}%)")
    print(f"• v4 ONLY Locked (v4 Advantage):{v4_only_frames:4d} frames ({(v4_only_frames / total_frames) * 100:.1f}%)")
    print(f"• v3 ONLY Locked:               {v3_only_frames:4d} frames ({(v3_only_frames / total_frames) * 100:.1f}%)")
    print(f"• Neither Model Detected:       {neither_frames:4d} frames ({(neither_frames / total_frames) * 100:.1f}%)")
    print("========================================================================\n")
    print(f"[OUTPUT] ✓ Side-by-side comparison video saved to: {out_video_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare v3 vs v4 detection on flight video")
    default_vid = REPO_ROOT / "outputs" / "recordings" / "flight_raw_20261002_190424.mp4"
    parser.add_argument("--video", type=str, default=str(default_vid), help="Path to input flight video")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    args = parser.parse_args()

    run_comparison(video_path=Path(args.video), conf_thresh=args.conf)


if __name__ == "__main__":
    main()
