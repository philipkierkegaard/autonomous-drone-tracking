import json
from pathlib import Path
import subprocess
import shutil
import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path("/Users/philipkierkegaard/Development/autonomous-drone-tracking")
MANIFEST_PATH = PROJECT_ROOT / "perception/datasets/usc_drone_tracking_benchmark/benchmark_manifest.json"
MODEL_WEIGHTS = PROJECT_ROOT / "perception/weights/yolov8n_drone_768_hybrid_nwd_c51_best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs/videos"
ARTIFACT_DIR = Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/5967d853-7a7e-4cdd-8792-4a77e6d704a2")

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def load_gt(gt_file):
    boxes = []
    with open(gt_file) as f:
        for line in f:
            parts = [float(p) for p in line.strip().split(",")]
            if len(parts) >= 4 and parts[2] > 0 and parts[3] > 0:
                boxes.append([parts[0], parts[1], parts[0] + parts[2], parts[1] + parts[3]])
            else:
                boxes.append(None)
    return boxes

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_mp4_temp = OUTPUT_DIR / "temp_canopy_full.mp4"
    final_mp4 = OUTPUT_DIR / "usc_yolo_hybrid_768_canopy_full.mp4"
    artifact_mp4 = ARTIFACT_DIR / "usc_yolo_hybrid_768_canopy_full.mp4"

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    seq_name = "Drone_frames_1_canopy"
    s_meta = manifest["sequences"][seq_name]
    img_dir = PROJECT_ROOT / s_meta["image_dir"]
    gt_boxes = load_gt(PROJECT_ROOT / s_meta["ground_truth_file"])

    start_frame = 1
    end_frame = len(gt_boxes)  # 1051 frames (~35.0 seconds at 30 fps)
    total_frames = end_frame - start_frame + 1

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[MODEL] Loading YOLO model: {MODEL_WEIGHTS} (Device: {device})")
    model = YOLO(str(MODEL_WEIGHTS))

    # Read first frame to get dimensions
    first_img = cv2.imread(str(img_dir / f"{start_frame:04d}.jpg"))
    orig_h, orig_w = first_img.shape[:2]

    # Target render resolution: 1280x720 for crisp web playback and fast encoding
    target_w, target_h = 1280, 720
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    writer = cv2.VideoWriter(str(out_mp4_temp), fourcc, fps, (target_w, target_h))

    print(f"[RENDER] Rendering full sequence: {total_frames} frames ({start_frame} -> {end_frame}) at {target_w}x{target_h}...")

    # Color Palette
    COLOR_GT = (50, 220, 50)       # Green
    COLOR_DET_TRUE = (255, 120, 0) # Cyan/Sky Blue in BGR
    COLOR_DET_FP = (30, 30, 255)   # Vivid Red in BGR
    COLOR_BG = (20, 24, 28)        # Dark Slate for HUD panels
    COLOR_BAR = (0, 180, 255)      # Amber/Gold

    running_visible = 0
    running_detected = 0
    running_fp = 0

    storyboard_frames = {
        100: "usc_canopy_storyboard_f0100.jpg",
        300: "usc_canopy_storyboard_f0300.jpg",
        473: "usc_canopy_storyboard_f0473.jpg",
        700: "usc_canopy_storyboard_f0700.jpg",
        950: "usc_canopy_storyboard_f0950.jpg",
    }

    for idx, f_idx in enumerate(range(start_frame, end_frame + 1)):
        fn = img_dir / f"{f_idx:04d}.jpg"
        frame = cv2.imread(str(fn))
        if frame is None:
            continue

        gt = gt_boxes[f_idx - 1]
        if gt is not None:
            running_visible += 1

        # YOLO inference
        preds = model.predict(source=str(fn), imgsz=768, conf=0.15, device=device, verbose=False)[0]
        boxes = preds.boxes.xyxy.cpu().numpy()
        confs = preds.boxes.conf.cpu().numpy()

        # Canvas resize to 1280x720
        canvas = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Scale coordinates
        scaled_gt = None
        if gt is not None:
            scaled_gt = [gt[0] * scale_x, gt[1] * scale_y, gt[2] * scale_x, gt[3] * scale_y]

        scaled_boxes = []
        for b in boxes:
            scaled_boxes.append([b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y])

        # Draw Ground Truth
        if scaled_gt is not None:
            gx1, gy1, gx2, gy2 = [int(v) for v in scaled_gt]
            cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), COLOR_GT, 2)
            cv2.putText(canvas, "GT: Phantom", (gx1, max(15, gy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GT, 1, cv2.LINE_AA)

        # Classify and Draw YOLO detections
        matched_any = False
        highest_conf = 0.0

        for b_idx, (b_orig, b_scaled, conf) in enumerate(zip(boxes, scaled_boxes, confs)):
            highest_conf = max(highest_conf, conf)
            bx1, by1, bx2, by2 = [int(v) for v in b_scaled]

            is_match = False
            if gt is not None:
                iou = compute_iou(b_orig, gt)
                if iou >= 0.25:
                    is_match = True
                    matched_any = True

            if is_match:
                color = COLOR_DET_TRUE
                label = f"YOLO: {conf:.2f}"
                # Crosshair in center
                cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
                cv2.drawMarker(canvas, (cx, cy), color, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
            else:
                color = COLOR_DET_FP
                label = f"YOLO FP: {conf:.2f}"
                running_fp += 1

            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, 2)
            
            # Label background pill
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(canvas, (bx1, by2 + 2), (bx1 + tw + 6, by2 + th + 6), (15, 15, 15), -1)
            cv2.putText(canvas, label, (bx1 + 3, by2 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        if matched_any:
            running_detected += 1

        running_recall = (running_detected / running_visible * 100.0) if running_visible > 0 else 0.0

        # -------------------------------------------------------------
        # 1. Top HUD Telemetry Banner (Semi-transparent Glassmorphism)
        # -------------------------------------------------------------
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (target_w, 48), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.line(canvas, (0, 48), (target_w, 48), (60, 70, 80), 1)

        # Progress bar
        prog_w = int((idx + 1) / total_frames * target_w)
        cv2.line(canvas, (0, 48), (prog_w, 48), COLOR_BAR, 2)

        # HUD Text
        cv2.putText(canvas, "YOLOv8n-Drone (768 Hybrid NWD)  |  USC Real-World Benchmark", (16, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 240, 250), 1, cv2.LINE_AA)
        
        status_str = f"STATUS: LOCK ({highest_conf:.2f})" if matched_any else "STATUS: OCCLUSION / DROPOUT"
        status_col = (50, 220, 50) if matched_any else (40, 100, 255)
        cv2.putText(canvas, status_str, (16, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_col, 1, cv2.LINE_AA)

        frame_info = f"Seq: {seq_name} | Frame: {f_idx:04d}/{end_frame} ({idx/30.0:4.1f}s)"
        cv2.putText(canvas, frame_info, (target_w - 470, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 200), 1, cv2.LINE_AA)

        recall_info = f"Running Recall: {running_recall:5.1f}% | Detections: {len(boxes)}"
        cv2.putText(canvas, recall_info, (target_w - 470, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 200), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 2. Picture-in-Picture (PiP) Inset: 2.2x Zoom on Target
        # -------------------------------------------------------------
        if gt is not None:
            gt_cx = (gt[0] + gt[2]) / 2.0
            gt_cy = (gt[1] + gt[3]) / 2.0
            crop_half_w = 180
            crop_half_h = 135

            cx1 = int(max(0, gt_cx - crop_half_w))
            cy1 = int(max(0, gt_cy - crop_half_h))
            cx2 = int(min(orig_w, gt_cx + crop_half_w))
            cy2 = int(min(orig_h, gt_cy + crop_half_h))

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                pip_w, pip_h = 240, 180
                pip_resized = cv2.resize(crop, (pip_w, pip_h), interpolation=cv2.INTER_LINEAR)

                # PiP placement in top-right below HUD
                px1 = target_w - pip_w - 16
                py1 = 56
                px2 = px1 + pip_w
                py2 = py1 + pip_h

                cv2.rectangle(canvas, (px1 - 2, py1 - 2), (px2 + 2, py2 + 2), (100, 110, 120), 1)
                canvas[py1:py2, px1:px2] = pip_resized
                
                # PiP Label
                cv2.rectangle(canvas, (px1, py1), (px1 + 150, py1 + 18), (20, 24, 28), -1)
                cv2.putText(canvas, "TARGET CLOSE-UP (NATIVE)", (px1 + 4, py1 + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 230, 240), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 3. Legend at bottom-left
        # -------------------------------------------------------------
        leg_overlay = canvas.copy()
        cv2.rectangle(leg_overlay, (16, target_h - 40), (320, target_h - 10), COLOR_BG, -1)
        cv2.addWeighted(leg_overlay, 0.8, canvas, 0.2, 0, canvas)
        
        # GT box legend
        cv2.rectangle(canvas, (24, target_h - 30), (36, target_h - 20), COLOR_GT, 2)
        cv2.putText(canvas, "GT", (42, target_h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)

        # YOLO match legend
        cv2.rectangle(canvas, (80, target_h - 30), (92, target_h - 20), COLOR_DET_TRUE, 2)
        cv2.putText(canvas, "YOLO Match", (98, target_h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)

        # YOLO distractor legend
        cv2.rectangle(canvas, (186, target_h - 30), (198, target_h - 20), COLOR_DET_FP, 2)
        cv2.putText(canvas, "YOLO Distractor", (204, target_h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)

        writer.write(canvas)

        # Save storyboard snapshots
        if f_idx in storyboard_frames:
            s_name = storyboard_frames[f_idx]
            cv2.imwrite(str(ARTIFACT_DIR / s_name), canvas)

        if (idx + 1) % 100 == 0 or idx == total_frames - 1:
            print(f"  Processed {idx + 1}/{total_frames} frames ({running_recall:5.1f}% recall)...")

    writer.release()
    print(f"[CONVERT] Remuxing full MP4 with ffmpeg to H.264 (libx264)...")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(out_mp4_temp),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(final_mp4)
    ]
    subprocess.run(cmd, check=True)

    # Copy to artifact dir
    shutil.copy2(final_mp4, artifact_mp4)
    print(f"[SAVED] Full MP4 video saved to:\n  - {final_mp4}\n  - {artifact_mp4}")

    if out_mp4_temp.exists():
        out_mp4_temp.unlink()

if __name__ == "__main__":
    main()
