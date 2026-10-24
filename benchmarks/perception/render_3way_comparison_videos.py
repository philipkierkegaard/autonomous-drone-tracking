#!/usr/bin/env python3
"""
Render Synchronized 3-Panel Comparison Videos for Anti-UAV Test Set.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception.

Generates synchronized 3-way ablation videos:
- Panel 1 (Left):   Raw YOLOv8n (Memoryless detector baseline)
- Panel 2 (Center): YOLOv8n + Tuned 8D Kalman Filter (Full FOV state estimator)
- Panel 3 (Right):  YOLOv8n + Tuned Kalman + Foveal Zoom (>=768px native crop with PiP)
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.pipeline import DroneTrackingPipeline

VIDEOS_DIR = REPO_ROOT / "perception" / "raw_data" / "Anti-UAV" / "videos"
GT_DIR = REPO_ROOT / "perception" / "raw_data" / "Anti-UAV" / "Anti-UAV-Tracking-V0GT"
OUT_DIR = REPO_ROOT / "outputs" / "videos" / "perception"
TUNED_PARAMS_FILE = REPO_ROOT / "outputs" / "kalman_tuning" / "calibrated_kalman_parameters.json"

ARTIFACT_DIRS = [
    Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/5967d853-7a7e-4cdd-8792-4a77e6d704a2"),
]


def load_gt_boxes(gt_path: Path) -> List[Optional[np.ndarray]]:
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


def render_3way_video(
    vid_name: str,
    weights_path: str = "perception/weights/yolov8n_drone_768_hybrid_nwd_c51_best.pt",
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    generate_gif: bool = True
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_raw = OUT_DIR / f"temp_{vid_name}_3way.mp4"
    final_video = OUT_DIR / f"{vid_name}_3way_comparison.mp4"

    vid_path = VIDEOS_DIR / f"{vid_name}.mp4"
    gt_path = GT_DIR / f"{vid_name}_gt.txt"
    gt_boxes = load_gt_boxes(gt_path)

    # Load tuned parameters
    with open(TUNED_PARAMS_FILE, "r") as f:
        t_data = json.load(f)
    t_opt = t_data["optimal_parameters"]
    r_cov = np.diag(t_opt["r_cov_diag"]).astype(np.float32)

    print(f"\n{'='*78}")
    print(f"RENDERING 3-WAY COMPARISON: {vid_name}")
    print(f"Video File: {vid_path}")
    print(f"Output:     {final_video}")
    print(f"{'='*78}")

    # 1. Raw YOLO Pipeline
    pipe_raw = DroneTrackingPipeline(
        weights=weights_path,
        conf_threshold=t_opt["conf_threshold"],
        enable_kalman=False,
        enable_dynamic_zoom=False,
        imgsz=768,
        device=device
    )

    # 2. Kalman Filter Pipeline (Full FOV)
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
        device=device
    )

    # 3. Kalman Filter + Foveal Zoom Pipeline (>=768px native crop)
    pipe_foveal = DroneTrackingPipeline(
        weights=weights_path,
        conf_threshold=t_opt["conf_threshold"],
        enable_kalman=True,
        enable_dynamic_zoom=True,
        max_zoom=2.5,
        min_crop_size=768,
        zoom_lost_threshold=3,
        desired_target_size=110.0,
        view_mode="wide_pip",
        q_pos=t_opt["q_pos"],
        q_vel=t_opt["q_vel"],
        q_dim=t_opt["q_dim"],
        q_scale_rate=t_opt["q_scale_rate"],
        max_lost_frames=t_opt["max_lost_frames"],
        min_hits_to_confirm=t_opt["min_hits_to_confirm"],
        dist_gate_scale=t_opt["dist_gate_scale"],
        r_cov=r_cov,
        imgsz=768,
        device=device
    )

    cap = cv2.VideoCapture(str(vid_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    panel_w = 640
    panel_h = 360
    header_h = 48
    out_w = panel_w * 3
    out_h = panel_h + header_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_raw), fourcc, fps, (out_w, out_h))

    frame_idx = 0
    raw_drops = 0
    kf_drops = 0
    fov_drops = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process through all three pipelines
        ann_raw, telem_raw = pipe_raw.process_frame(frame.copy(), draw_hud=True)
        ann_kf, telem_kf = pipe_kalman.process_frame(frame.copy(), draw_hud=True)
        ann_fov, telem_fov = pipe_foveal.process_frame(frame.copy(), draw_hud=True)

        gt_box = gt_boxes[frame_idx] if frame_idx < len(gt_boxes) else None

        # Draw Ground Truth on all panels in Cyan (0, 255, 255)
        if gt_box is not None:
            gx1, gy1, gx2, gy2 = [int(v) for v in gt_box]
            cv2.rectangle(ann_raw, (gx1, gy1), (gx2, gy2), (255, 240, 0), 1, cv2.LINE_AA)
            cv2.rectangle(ann_kf, (gx1, gy1), (gx2, gy2), (255, 240, 0), 1, cv2.LINE_AA)
            cv2.rectangle(ann_fov, (gx1, gy1), (gx2, gy2), (255, 240, 0), 1, cv2.LINE_AA)

        # Count dropouts
        if gt_box is not None:
            if not telem_raw.get("bbox"):
                raw_drops += 1
            if telem_kf.get("status") not in ["LOCKED", "COASTING"]:
                kf_drops += 1
            if telem_fov.get("status") not in ["LOCKED", "COASTING"]:
                fov_drops += 1

        # Resize to panel dimensions
        p_raw = cv2.resize(ann_raw, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        p_kf = cv2.resize(ann_kf, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        p_fov = cv2.resize(ann_fov, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

        # Panel Headers (30 px dark banner at top of each panel)
        # Panel 1: Raw YOLO
        cv2.rectangle(p_raw, (0, 0), (panel_w, 32), (15, 23, 42), -1)
        cv2.putText(p_raw, "CONFIG A: Raw YOLOv8 (Memoryless)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (239, 68, 68), 2, cv2.LINE_AA)
        status_l = telem_raw.get("status", "SEARCHING")
        col_l = (46, 213, 115) if status_l == "LOCKED" else (239, 68, 68)
        cv2.putText(p_raw, f"STATUS: {status_l}", (panel_w - 180, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, col_l, 2, cv2.LINE_AA)

        # Panel 2: Kalman Tuned
        cv2.rectangle(p_kf, (0, 0), (panel_w, 32), (15, 23, 42), -1)
        cv2.putText(p_kf, "CONFIG B: YOLOv8 + Tuned Kalman", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (16, 185, 129), 2, cv2.LINE_AA)
        status_m = telem_kf.get("status", "SEARCHING")
        col_m = (46, 213, 115) if status_m == "LOCKED" else ((255, 165, 0) if status_m == "COASTING" else (239, 68, 68))
        cv2.putText(p_kf, f"STATUS: {status_m}", (panel_w - 180, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, col_m, 2, cv2.LINE_AA)

        # Panel 3: Foveal Zoom
        cv2.rectangle(p_fov, (0, 0), (panel_w, 32), (15, 23, 42), -1)
        cv2.putText(p_fov, "CONFIG C: YOLO + Kalman + Foveal Zoom", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (192, 132, 252), 2, cv2.LINE_AA)
        status_r = telem_fov.get("status", "SEARCHING")
        col_r = (46, 213, 115) if status_r == "LOCKED" else ((255, 165, 0) if status_r == "COASTING" else (239, 68, 68))
        z_level = telem_fov.get("zoom_level", 1.0)
        cv2.putText(p_fov, f"PTZ: {z_level:.1f}x", (panel_w - 190, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(p_fov, f"[{status_r}]", (panel_w - 100, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, col_r, 2, cv2.LINE_AA)

        # Combine Panels Horizontally
        combined_panels = np.hstack([p_raw, p_kf, p_fov])

        # Vertical Divider Lines
        cv2.line(combined_panels, (panel_w, 0), (panel_w, panel_h), (255, 255, 255), 2)
        cv2.line(combined_panels, (panel_w * 2, 0), (panel_w * 2, panel_h), (255, 255, 255), 2)

        # Top Master Header (48 px)
        header = np.zeros((header_h, out_w, 3), dtype=np.uint8)
        header[:] = (10, 15, 29)  # Dark slate background

        title_text = f"ANTI-UAV TEST SET ABLATION: {vid_name.upper()} ({w}x{h} at {fps:.0f} FPS)"
        cv2.putText(header, title_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        time_s = frame_idx / fps
        time_text = f"Frame {frame_idx:04d}/{total_frames:04d} ({time_s:5.2f}s)"
        cv2.putText(header, time_text, (out_w // 2 - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 220, 255), 2, cv2.LINE_AA)

        gt_legend = "Cyan Box = Ground Truth"
        cv2.rectangle(header, (out_w - 260, 12), (out_w - 245, 27), (255, 240, 0), 2)
        cv2.putText(header, gt_legend, (out_w - 235, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 240, 0), 1, cv2.LINE_AA)

        full_frame = np.vstack([header, combined_panels])
        writer.write(full_frame)

        frame_idx += 1
        if frame_idx % 150 == 0 or frame_idx == total_frames:
            print(f"     Rendered {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")

    cap.release()
    writer.release()

    # Transcode with ffmpeg
    print(f"     Transcoding {vid_name} to QuickTime H.264...")
    cmd = [
        "ffmpeg", "-y", "-i", str(temp_raw),
        "-c:v", "libx264",
        "-crf", "22",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(final_video)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if temp_raw.exists():
        temp_raw.unlink()

    print(f"[VIDEO COMPLETED] Saved: {final_video}")

    # Copy to Artifact Dirs
    for adir in ARTIFACT_DIRS:
        if adir.exists():
            dest = adir / final_video.name
            import shutil
            shutil.copy2(final_video, dest)

    # Optional GIF generation for interactive preview
    if generate_gif:
        gif_path = OUT_DIR / f"{vid_name}_3way_comparison.gif"
        print(f"     Generating preview GIF: {gif_path.name}...")
        # For long videos, render a key 8-second excerpt at 15 FPS
        # video05: frames 100-340; video12: frames 900-1140 (the cloud dropout region); video20: frames 200-440
        if vid_name == "video05":
            ss, dur = "3.0", "8.0"
        elif vid_name == "video12":
            ss, dur = "30.0", "8.0"  # Cloud occlusion dropout region
        else:
            ss, dur = "7.0", "8.0"   # Rapid maneuver region

        gif_cmd = [
            "ffmpeg", "-y",
            "-ss", ss,
            "-t", dur,
            "-i", str(final_video),
            "-vf", "fps=12,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
            str(gif_path)
        ]
        try:
            subprocess.run(gif_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"[GIF COMPLETED] Saved: {gif_path}")
            for adir in ARTIFACT_DIRS:
                if adir.exists():
                    import shutil
                    shutil.copy2(gif_path, adir / gif_path.name)
        except Exception as e:
            print(f"[WARN] Failed generating GIF for {vid_name}: {e}")

    return final_video


def main():
    parser = argparse.ArgumentParser(description="Render synchronized 3-way ablation comparison videos")
    parser.add_argument("--videos", nargs="+", default=["video05", "video12", "video20"], help="Videos to render")
    parser.add_argument("--weights", type=str, default="perception/weights/yolov8n_drone_768_hybrid_nwd_c51_best.pt")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--no-gif", action="store_true", help="Disable GIF generation")
    args = parser.parse_args()

    for vid in args.videos:
        render_3way_video(
            vid_name=vid,
            weights_path=args.weights,
            device=args.device,
            generate_gif=not args.no_gif
        )

    print("\n[ALL RENDERS FINISHED] All 3-way comparison videos successfully generated in outputs/videos/perception/!")


if __name__ == "__main__":
    main()
