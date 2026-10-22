import json
from pathlib import Path
import subprocess
import shutil
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from perception.pipeline import DroneTracker

PROJECT_ROOT = Path("/Users/philipkierkegaard/Development/autonomous-drone-tracking")
MANIFEST_PATH = PROJECT_ROOT / "perception/datasets/usc_drone_tracking_benchmark/benchmark_manifest.json"
MODEL_WEIGHTS = PROJECT_ROOT / "perception/weights/yolov8n_drone_768_hybrid_nwd_c51_best.pt"
CALIB_PARAMS_PATH = PROJECT_ROOT / "outputs/kalman_tuning/calibrated_kalman_parameters.json"
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
    out_mp4_temp = OUTPUT_DIR / "temp_drone8_1000.mp4"
    final_mp4 = OUTPUT_DIR / "usc_drone8_kalman_1000frames.mp4"
    artifact_mp4 = ARTIFACT_DIR / "usc_drone8_kalman_1000frames.mp4"

    # 1. Load Parameters from Calibrated Anti-UAV Run
    with open(CALIB_PARAMS_PATH) as f:
        calib_data = json.load(f)
    opt_params = calib_data.get("optimal_parameters", {})

    conf_thresh = opt_params.get("conf_threshold", 0.15)
    tracker_kwargs = {
        "max_lost_frames": opt_params.get("max_lost_frames", 20),
        "min_hits_to_confirm": opt_params.get("min_hits_to_confirm", 2),
        "iou_match_threshold": opt_params.get("iou_match_threshold", 0.70),
        "dt": 1.0 / 30.0,
        "q_pos": opt_params.get("q_pos", 2.0),
        "q_vel": opt_params.get("q_vel", 50.0),
        "q_dim": opt_params.get("q_dim", 4.0),
        "q_scale_rate": opt_params.get("q_scale_rate", 5.0),
        "dist_gate_scale": opt_params.get("dist_gate_scale", 3.0),
        "r_cov": np.diag(opt_params.get("r_cov_diag", [11.0, 8.4, 128.1, 77.1])).astype(np.float32),
    }

    tracker = DroneTracker(**tracker_kwargs)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    seq_name = "Drone8"
    s_meta = manifest["sequences"][seq_name]
    img_dir = PROJECT_ROOT / s_meta["image_dir"]
    gt_boxes = load_gt(PROJECT_ROOT / s_meta["ground_truth_file"])

    start_frame = 1
    end_frame = 1000  # Exactly 1,000 frames (~33.3 seconds at 30 fps)
    total_frames = end_frame - start_frame + 1

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[MODEL] Loading YOLO model: {MODEL_WEIGHTS} (Device: {device})")
    model = YOLO(str(MODEL_WEIGHTS))

    # Read first frame to get dimensions
    first_img = cv2.imread(str(img_dir / f"{start_frame:04d}.jpg"))
    orig_h, orig_w = first_img.shape[:2]

    # Target render resolution: 1280x720
    target_w, target_h = 1280, 720
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    writer = cv2.VideoWriter(str(out_mp4_temp), fourcc, fps, (target_w, target_h))

    print(f"[RENDER] Rendering Drone8 airborne sequence: {total_frames} frames ({start_frame} -> {end_frame})...")

    # Aesthetics Palette
    COLOR_GT = (50, 220, 50)          # Clean Green
    COLOR_YOLO = (255, 200, 50)       # Cyan/Light Blue for YOLO detections
    COLOR_KF_LOCKED = (0, 215, 255)   # Radiant Gold/Amber for Kalman Locked Track
    COLOR_KF_COAST = (0, 100, 255)    # Bright Coral/Orange for Kalman Coasting
    COLOR_TRAIL = (255, 100, 200)     # Neon Magenta for Trajectory History
    COLOR_BG = (20, 24, 28)           # Dark Slate HUD
    COLOR_BAR = (0, 215, 255)         # Gold accent

    running_visible = 0
    running_prec50 = 0
    running_prec20 = 0
    running_cle_list = []
    trajectory_trail = []

    storyboard_frames = {
        100: "usc_drone8_storyboard_f0100.jpg",
        300: "usc_drone8_storyboard_f0300.jpg",
        500: "usc_drone8_storyboard_f0500.jpg",
        750: "usc_drone8_storyboard_f0750.jpg",
        950: "usc_drone8_storyboard_f0950.jpg",
    }

    for idx, f_idx in enumerate(range(start_frame, end_frame + 1)):
        fn = img_dir / f"{f_idx:04d}.jpg"
        frame = cv2.imread(str(fn))
        if frame is None:
            continue

        gt = gt_boxes[f_idx - 1]

        # 1. YOLO inference
        preds = model.predict(source=str(fn), imgsz=768, conf=conf_thresh, device=device, verbose=False)[0]
        boxes = preds.boxes.xyxy.cpu().numpy()
        confs = preds.boxes.conf.cpu().numpy()

        # 2. Kalman Tracker Update
        active_tracks = tracker.update(boxes, confs)

        primary_track = None
        for trk in active_tracks:
            if trk["id"] == tracker.primary_track_id:
                primary_track = trk
                break

        # Canvas resize to 1280x720
        canvas = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Scale coordinates
        scaled_gt = None
        if gt is not None:
            scaled_gt = [gt[0] * scale_x, gt[1] * scale_y, gt[2] * scale_x, gt[3] * scale_y]

        # 3. Draw Ground Truth Box
        if scaled_gt is not None:
            gx1, gy1, gx2, gy2 = [int(v) for v in scaled_gt]
            cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), COLOR_GT, 1)
            cv2.putText(canvas, "GT: Phantom", (gx1, max(15, gy1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_GT, 1, cv2.LINE_AA)

        # 4. Draw Raw YOLO Detections (Thin Cyan)
        for b, c in zip(boxes, confs):
            bx1, by1, bx2, by2 = int(b[0] * scale_x), int(b[1] * scale_y), int(b[2] * scale_x), int(b[3] * scale_y)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), COLOR_YOLO, 1)
            cv2.putText(canvas, f"YOLO {c:.2f}", (bx1, min(target_h - 6, by2 + 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_YOLO, 1, cv2.LINE_AA)

        # 5. Process Kalman Estimate
        kf_box = None
        kf_is_coasting = False
        kf_vel = (0.0, 0.0)
        kf_cle = 500.0
        kf_status_str = "SEARCHING / NO ACTIVE LOCK"
        kf_status_col = (100, 110, 120)

        if primary_track is not None:
            orig_kf_box = primary_track["bbox"]
            kf_box = [orig_kf_box[0] * scale_x, orig_kf_box[1] * scale_y, orig_kf_box[2] * scale_x, orig_kf_box[3] * scale_y]
            kf_is_coasting = primary_track["is_coasting"]
            kf_vel = primary_track["velocity"]
            time_coast = primary_track.get("time_coasting", 0)

            kx1, ky1, kx2, ky2 = [int(v) for v in kf_box]
            kcx, kcy = (kx1 + kx2) // 2, (ky1 + ky2) // 2
            trajectory_trail.append((kcx, kcy))

            if len(trajectory_trail) > 40:
                trajectory_trail.pop(0)

            # Draw trajectory trail
            for t_i in range(1, len(trajectory_trail)):
                thickness = 1 if t_i < 20 else 2
                pt1 = trajectory_trail[t_i - 1]
                pt2 = trajectory_trail[t_i]
                cv2.line(canvas, pt1, pt2, COLOR_TRAIL, thickness, cv2.LINE_AA)

            if kf_is_coasting:
                kf_color = COLOR_KF_COAST
                kf_status_str = f"STATE: COASTING (Bridge {time_coast}f)"
                kf_status_col = COLOR_KF_COAST
                tag_label = f"KF #{primary_track['id']} [COASTING {time_coast}]"
            else:
                kf_color = COLOR_KF_LOCKED
                kf_status_str = f"STATE: AIRBORNE LOCK (Hits: {primary_track['hits']})"
                kf_status_col = (50, 220, 50)
                tag_label = f"KF #{primary_track['id']} [LOCKED]"

            # Draw tactical Kalman box
            cv2.rectangle(canvas, (kx1, ky1), (kx2, ky2), kf_color, 2)

            c_len = min(14, (kx2 - kx1) // 3, (ky2 - ky1) // 3)
            if c_len > 2:
                cv2.line(canvas, (kx1, ky1), (kx1 + c_len, ky1), (255, 255, 255), 2)
                cv2.line(canvas, (kx1, ky1), (kx1, ky1 + c_len), (255, 255, 255), 2)
                cv2.line(canvas, (kx2, ky1), (kx2 - c_len, ky1), (255, 255, 255), 2)
                cv2.line(canvas, (kx2, ky1), (kx2, ky1 + c_len), (255, 255, 255), 2)
                cv2.line(canvas, (kx1, ky2), (kx1 + c_len, ky2), (255, 255, 255), 2)
                cv2.line(canvas, (kx1, ky2), (kx1, ky2 - c_len), (255, 255, 255), 2)
                cv2.line(canvas, (kx2, ky2), (kx2 - c_len, ky2), (255, 255, 255), 2)
                cv2.line(canvas, (kx2, ky2), (kx2, ky2 - c_len), (255, 255, 255), 2)

            # Reticle
            cv2.drawMarker(canvas, (kcx, kcy), kf_color, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
            cv2.circle(canvas, (kcx, kcy), 8, kf_color, 1, cv2.LINE_AA)

            # Velocity Vector Arrow
            vx_scaled = kf_vel[0] * scale_x * 0.25
            vy_scaled = kf_vel[1] * scale_y * 0.25
            tip_x = int(np.clip(kcx + vx_scaled, 0, target_w - 1))
            tip_y = int(np.clip(kcy + vy_scaled, 0, target_h - 1))
            if np.hypot(vx_scaled, vy_scaled) > 4.0:
                cv2.arrowedLine(canvas, (kcx, kcy), (tip_x, tip_y), (0, 255, 255), 2, tipLength=0.3)

            # Tag
            (tw, th), _ = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(canvas, (kx1, max(0, ky1 - th - 8)), (kx1 + tw + 8, max(th + 8, ky1)), (15, 15, 15), -1)
            cv2.putText(canvas, tag_label, (kx1 + 4, max(th + 2, ky1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, kf_color, 1, cv2.LINE_AA)

            # Evaluate distance to GT
            if gt is not None:
                gt_cx = (gt[0] + gt[2]) / 2.0
                gt_cy = (gt[1] + gt[3]) / 2.0
                kf_orig_cx = (orig_kf_box[0] + orig_kf_box[2]) / 2.0
                kf_orig_cy = (orig_kf_box[1] + orig_kf_box[3]) / 2.0
                kf_cle = float(np.hypot(kf_orig_cx - gt_cx, kf_orig_cy - gt_cy))
        else:
            trajectory_trail.clear()

        # Update metrics
        if gt is not None:
            running_visible += 1
            running_cle_list.append(kf_cle)
            if kf_cle <= 50.0:
                running_prec50 += 1
            if kf_cle <= 20.0:
                running_prec20 += 1

        p50_curr = (running_prec50 / running_visible * 100.0) if running_visible > 0 else 0.0
        p20_curr = (running_prec20 / running_visible * 100.0) if running_visible > 0 else 0.0
        cle_curr = np.mean(running_cle_list) if running_cle_list else 0.0

        # -------------------------------------------------------------
        # 6. Top HUD Telemetry Banner
        # -------------------------------------------------------------
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (target_w, 52), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.line(canvas, (0, 52), (target_w, 52), (60, 70, 80), 1)

        # Progress bar
        prog_w = int((idx + 1) / total_frames * target_w)
        cv2.line(canvas, (0, 52), (prog_w, 52), COLOR_BAR, 2)

        # Left Column: System & Status
        cv2.putText(canvas, "YOLOv8n-768 + 8D Kalman State Estimator  |  Drone8 Airborne Interception", (16, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 240, 250), 1, cv2.LINE_AA)
        cv2.putText(canvas, kf_status_str, (16, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, kf_status_col, 1, cv2.LINE_AA)

        # Middle Column: Velocity & Frame
        vel_str = f"Vel: ({kf_vel[0]:+5.1f}, {kf_vel[1]:+5.1f}) px/s | Fr: {f_idx:04d}/{end_frame} ({idx/30.0:4.1f}s)"
        cv2.putText(canvas, vel_str, (target_w // 2 - 130, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 210, 220), 1, cv2.LINE_AA)
        
        # Right Column: Tracking Metrics
        metric_str = f"Prec@50: {p50_curr:4.1f}% | Prec@20: {p20_curr:4.1f}% | Mean CLE: {cle_curr:4.1f} px"
        cv2.putText(canvas, metric_str, (target_w - 490, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 215, 255), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 7. Picture-in-Picture (PiP) Inset: Gimbal Boresight (Kalman Center)
        # -------------------------------------------------------------
        center_x = orig_kf_box[0] + orig_kf_box[2] if primary_track is not None else (gt[0] + gt[2] if gt else orig_w)
        center_x /= 2.0
        center_y = orig_kf_box[1] + orig_kf_box[3] if primary_track is not None else (gt[1] + gt[3] if gt else orig_h)
        center_y /= 2.0

        crop_half_w = 160
        crop_half_h = 120
        cx1 = int(max(0, center_x - crop_half_w))
        cy1 = int(max(0, center_y - crop_half_h))
        cx2 = int(min(orig_w, center_x + crop_half_w))
        cy2 = int(min(orig_h, center_y + crop_half_h))

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size > 0:
            pip_w, pip_h = 240, 180
            pip_resized = cv2.resize(crop, (pip_w, pip_h), interpolation=cv2.INTER_LINEAR)

            px1 = target_w - pip_w - 16
            py1 = 60
            px2 = px1 + pip_w
            py2 = py1 + pip_h

            cv2.rectangle(canvas, (px1 - 2, py1 - 2), (px2 + 2, py2 + 2), (100, 110, 120), 1)
            canvas[py1:py2, px1:px2] = pip_resized
            
            # PiP Header
            cv2.rectangle(canvas, (px1, py1), (px1 + 175, py1 + 18), (20, 24, 28), -1)
            cv2.putText(canvas, "GIMBAL BORESIGHT (KF ESTIMATE)", (px1 + 4, py1 + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (220, 230, 240), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 8. Legend at bottom-left
        # -------------------------------------------------------------
        leg_overlay = canvas.copy()
        cv2.rectangle(leg_overlay, (16, target_h - 40), (430, target_h - 10), COLOR_BG, -1)
        cv2.addWeighted(leg_overlay, 0.8, canvas, 0.2, 0, canvas)
        
        cv2.rectangle(canvas, (24, target_h - 29), (34, target_h - 21), COLOR_GT, 1)
        cv2.putText(canvas, "GT", (38, target_h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

        cv2.rectangle(canvas, (65, target_h - 29), (75, target_h - 21), COLOR_YOLO, 1)
        cv2.putText(canvas, "YOLO Raw", (80, target_h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

        cv2.rectangle(canvas, (145, target_h - 29), (155, target_h - 21), COLOR_KF_LOCKED, 2)
        cv2.putText(canvas, "KF Locked", (160, target_h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

        cv2.rectangle(canvas, (230, target_h - 29), (240, target_h - 21), COLOR_KF_COAST, 2)
        cv2.putText(canvas, "KF Coasting", (245, target_h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

        cv2.line(canvas, (325, target_h - 25), (345, target_h - 25), COLOR_TRAIL, 2)
        cv2.putText(canvas, "Trajectory", (350, target_h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

        writer.write(canvas)

        if f_idx in storyboard_frames:
            s_name = storyboard_frames[f_idx]
            cv2.imwrite(str(ARTIFACT_DIR / s_name), canvas)

        if (idx + 1) % 100 == 0 or idx == total_frames - 1:
            print(f"  Processed {idx + 1}/{total_frames} frames (Prec@50: {p50_curr:5.1f}%, CLE: {cle_curr:5.1f} px)...")

    writer.release()
    print(f"[CONVERT] Remuxing Drone8 Kalman MP4 with ffmpeg to H.264 (libx264)...")

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
    print(f"[SAVED] Drone8 1000-frame Kalman MP4 video saved to:\n  - {final_mp4}\n  - {artifact_mp4}")

    if out_mp4_temp.exists():
        out_mp4_temp.unlink()

if __name__ == "__main__":
    main()
