"""
Render Video Tracking with YOLOv8n-v4
Processes Anti-UAV video sequences, draws predicted bounding boxes,
HUD telemetry, trajectory trails, and a top-right zoomed picture-in-picture target crop.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def render_tracked_video(
    video_path: str = "perception/raw_data/Anti-UAV/videos/video15.mp4",
    weights_path: str = "perception/weights/yolov8n_drone_v4_continued_best.pt",
    output_path: str = "outputs/video15_tracked_v4.mp4",
    conf_threshold: float = 0.25,
    imgsz: int = 768,
    device: str = "mps"
):
    video_path = Path(video_path).resolve()
    weights_path = Path(weights_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_raw_output = output_path.parent / "temp_raw_video15.mp4"

    print(f"Loading YOLO model from: {weights_path}")
    model = YOLO(str(weights_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video input: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    print(f"Starting tracking inference on device='{device}'...")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(temp_raw_output), fourcc, fps, (width, height))

    trajectory = []
    max_traj_len = 45

    frame_idx = 0
    detections_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Run YOLO inference
        results = model.predict(
            frame,
            imgsz=imgsz,
            conf=conf_threshold,
            device=device,
            verbose=False
        )[0]

        boxes = results.boxes
        target_locked = len(boxes) > 0
        best_box = None
        best_conf = 0.0

        if target_locked:
            detections_count += 1
            # Pick highest confidence detection
            best_idx = int(np.argmax(boxes.conf.cpu().numpy()))
            xyxy = boxes.xyxy[best_idx].cpu().numpy().astype(int)
            best_conf = float(boxes.conf[best_idx].cpu().numpy())
            best_box = xyxy

            x1, y1, x2, y2 = xyxy
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            bw = x2 - x1
            bh = y2 - y1

            trajectory.append((cx, cy))
            if len(trajectory) > max_traj_len:
                trajectory.pop(0)

            # 1. Draw smooth trajectory trail
            for i in range(1, len(trajectory)):
                alpha = i / len(trajectory)
                thickness = max(1, int(3 * alpha))
                color = (int(0 * alpha), int(255 * alpha), int(255 * (1 - alpha)))
                cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 180), thickness)

            # 2. Draw Predicted Bounding Box directly on top of the drone
            # Neon Green Box with subtle border
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Corner brackets for tactical HUD look
            corner_len = max(8, min(bw, bh) // 4)
            # Top-left
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (0, 255, 255), 3)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), (0, 255, 255), 3)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), (0, 255, 255), 3)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), (0, 255, 255), 3)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), (0, 255, 255), 3)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), (0, 255, 255), 3)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (0, 255, 255), 3)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), (0, 255, 255), 3)

            # Center crosshair dot
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

            # Bounding box label pill
            label = f"DRONE {best_conf * 100:.1f}% ({bw}x{bh}px)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)

            # Label badge
            label_y1 = max(10, y1 - th - 8)
            label_y2 = max(th + 8, y1)
            cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 10, label_y2), (0, 0, 0), -1)
            cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 10, label_y2), (0, 255, 0), 1)
            cv2.putText(frame, label, (x1 + 5, label_y2 - 4), font, font_scale, (0, 255, 0), 1, cv2.LINE_AA)

            # 3. Top-Right Picture-in-Picture (PiP) Zoom Crop
            pip_size = 220
            pip_x2 = width - 25
            pip_x1 = pip_x2 - pip_size
            pip_y1 = 25
            pip_y2 = pip_y1 + pip_size

            # Crop around target with padding
            pad = int(max(bw, bh) * 0.8)
            crop_x1 = max(0, cx - pad)
            crop_x2 = min(width, cx + pad)
            crop_y1 = max(0, cy - pad)
            crop_y2 = min(height, cy + pad)

            if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                target_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                target_resized = cv2.resize(target_crop, (pip_size, pip_size), interpolation=cv2.INTER_LINEAR)

                # Draw semi-transparent dark backing
                sub_img = frame[pip_y1:pip_y2, pip_x1:pip_x2]
                pip_canvas = target_resized.copy()
                # PiP border & crosshair
                cv2.rectangle(pip_canvas, (0, 0), (pip_size - 1, pip_size - 1), (0, 255, 0), 2)
                ch_center = pip_size // 2
                cv2.drawMarker(pip_canvas, (ch_center, ch_center), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
                cv2.putText(pip_canvas, f"OPTICAL ZOOM {best_conf*100:.0f}%", (10, 20), font, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

                frame[pip_y1:pip_y2, pip_x1:pip_x2] = pip_canvas

        else:
            # Trajectory decays when target lost
            if len(trajectory) > 0:
                trajectory.pop(0)

        # 4. Heads-Up Display (Top-Left HUD)
        hud_w = 420
        hud_h = 105
        hud_x1 = 25
        hud_y1 = 25
        hud_x2 = hud_x1 + hud_w
        hud_y2 = hud_y1 + hud_h

        # Overlay translucent black background
        overlay = frame.copy()
        cv2.rectangle(overlay, (hud_x1, hud_y1), (hud_x2, hud_y2), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (hud_x1, hud_y1), (hud_x2, hud_y2), (0, 255, 180), 1)

        # HUD Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "C-UAS PURSUIT TRACKER | YOLOv8n-v4", (hud_x1 + 12, hud_y1 + 24), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        
        status_color = (0, 255, 0) if target_locked else (0, 165, 255)
        status_text = "TARGET: LOCKED" if target_locked else "TARGET: SEARCHING..."
        cv2.putText(frame, status_text, (hud_x1 + 12, hud_y1 + 52), font, 0.60, status_color, 2, cv2.LINE_AA)

        info_text = f"Frame: {frame_idx:04d}/{total_frames} | Res: 768px | Loss: CIoU+NWD"
        cv2.putText(frame, info_text, (hud_x1 + 12, hud_y1 + 78), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        out.write(frame)

        if frame_idx % 150 == 0 or frame_idx == total_frames:
            lock_rate = (detections_count / frame_idx) * 100
            print(f"Rendered {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%) | Lock rate: {lock_rate:.1f}%")

    cap.release()
    out.release()

    print(f"\nFinished raw render: {temp_raw_output}")
    print(f"Transcoding to web-compatible H.264 MP4 via ffmpeg...")

    # Fast ffmpeg remux / transcode to H.264 yuv420p for universal browser playback
    ffmpeg_cmd = (
        f"ffmpeg -y -i '{temp_raw_output}' -c:v libx264 -pix_fmt yuv420p -preset fast -crf 22 '{output_path}'"
    )
    res = os.system(ffmpeg_cmd)
    if res == 0 and output_path.exists():
        temp_raw_output.unlink(missing_ok=True)
        print(f"Success! Output ready at: {output_path}")
    else:
        print(f"ffmpeg transcode warning, raw output kept at: {temp_raw_output}")


if __name__ == "__main__":
    render_tracked_video()
