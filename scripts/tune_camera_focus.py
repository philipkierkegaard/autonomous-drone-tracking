#!/usr/bin/env python3
"""
Camera Focus & Sharpness Peaking Tool
Helps manually tune the physical lens focus ring to infinity (peak sharpness).

Features:
- Live real-time Sharpness Score (Laplacian variance of high-frequency edges)
- Center 4x Digital Loupe / Zoom Box to visually inspect distant leaves and drone contours
- High-water mark tracking (records the highest sharpness achieved so you know when you've hit peak)
- Built-in HTTP MJPEG Web Streamer: View live in your Mac browser at http://192.168.55.1:8080
- Configurable Tegra ISP Edge Enhancement (--ee-strength)
"""

import sys
import time
import threading
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Optional
import cv2
import numpy as np

class ReusableThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

class MJPEGStreamer:
    def __init__(self, port: int = 8080):
        self.port = port
        self.latest_jpeg: Optional[bytes] = None
        self.lock = threading.Lock()
        self.running = True
        self.server: Optional[ReusableThreadingHTTPServer] = None

    def update_frame(self, frame: np.ndarray, quality: int = 85):
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ret:
            with self.lock:
                self.latest_jpeg = buf.tobytes()

    def start(self):
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress request spam

            def do_GET(self):
                if self.path in ["/", "/index.html"]:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Camera Focus Peaking Tool</title>
    <style>
        body {{ background: #0f1117; color: #fff; font-family: -apple-system, sans-serif; text-align: center; margin: 0; padding: 20px; }}
        h1 {{ margin: 0 0 10px 0; font-size: 22px; color: #00d2d3; }}
        .hud-img {{ max-width: 95vw; height: auto; border: 2px solid #333; border-radius: 8px; box-shadow: 0 10px 30px rgba(0,0,0,0.7); }}
        .hint {{ margin-top: 15px; font-size: 14px; color: #aaa; max-width: 800px; margin-left: auto; margin-right: auto; line-height: 1.6; }}
        .hint b {{ color: #00d2d3; }}
    </style>
</head>
<body>
    <h1>CAMERA FOCUS PEAKING & SHARPNESS TUNER</h1>
    <img class="hud-img" src="/stream" alt="Focus Stream" />
    <div class="hint">
        &bull; Aim camera 20m+ away at trees/building &bull; Watch the <b>Sharpness score</b> and <b>4x Loupe</b> (top-right)<br/>
        &bull; Rotate <b>NEAR / FAR</b> ring until Sharpness hits its <b>PEAK</b> &bull; If sunny, stop aperture down slightly from <b>OPEN</b> towards <b>CLOSE</b>!
    </div>
</body>
</html>"""
                    self.wfile.write(html.encode("utf-8"))
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
                    self.end_headers()
                    try:
                        while parent.running:
                            with parent.lock:
                                jpeg_bytes = parent.latest_jpeg
                            if jpeg_bytes is not None:
                                self.wfile.write(b"--FRAME\r\n")
                                self.send_header("Content-Type", "image/jpeg")
                                self.send_header("Content-Length", str(len(jpeg_bytes)))
                                self.end_headers()
                                self.wfile.write(jpeg_bytes)
                                self.wfile.write(b"\r\n")
                            time.sleep(0.033)
                    except (ConnectionResetError, BrokenPipeError):
                        pass
                else:
                    self.send_error(404)

        try:
            self.server = ReusableThreadingHTTPServer(("0.0.0.0", self.port), Handler)
            t = threading.Thread(target=self.server.serve_forever, daemon=True)
            t.start()
            print(f"[STREAM] Live Focus Stream active at: http://0.0.0.0:{self.port} (or http://192.168.55.1:{self.port})")
        except Exception as e:
            print(f"[WARN] Failed to start stream server: {e}")

    def stop(self):
        self.running = False
        if self.server:
            self.server.shutdown()
            self.server.server_close()

def parse_args():
    parser = argparse.ArgumentParser(description="Live camera focus peaking & tuning tool")
    parser.add_argument("--source", default="csi", help="Camera source index (e.g. 0) or 'csi' for Jetson CSI")
    parser.add_argument("--width", type=int, default=1280, help="Display width")
    parser.add_argument("--height", type=int, default=720, help="Display height")
    parser.add_argument("--sensor-id", type=int, default=0, help="Jetson CSI sensor ID (default: 0)")
    parser.add_argument("--ee-strength", type=float, default=0.3, help="Tegra ISP Edge Enhancement (0.0=none, 0.3=natural, 1.0=strong)")
    parser.add_argument("--stream", action="store_true", default=True, help="Stream to web browser (default: True)")
    parser.add_argument("--port", type=int, default=8080, help="Web stream HTTP port (default: 8080)")
    parser.add_argument("--no-gui", action="store_true", help="Headless mode (don't attempt cv2.imshow)")
    return parser.parse_args()

def open_camera(args):
    src = str(args.source).strip()
    if src.lower() == "csi":
        ee_prop = f"ee-mode=2 ee-strength={args.ee_strength:.2f} " if args.ee_strength > 0.0 else "ee-mode=0 ee-strength=0.0 "
        pipeline = (
            f"nvarguscamerasrc sensor-id={args.sensor_id} {ee_prop}tnr-mode=1 tnr-strength=0.0 ! "
            f"video/x-raw(memory:NVMM), width=3840, height=2160, format=NV12, framerate=30/1 ! "
            f"nvvidconv ! video/x-raw, width={args.width}, height={args.height}, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false"
        )
        print(f"[FOCUS] Opening Jetson CSI camera (Mode 3840x2160 supersampled, EE={args.ee_strength}):\n  {pipeline}")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        idx = int(src) if src.isdigit() else 0
        print(f"[FOCUS] Opening V4L2 camera on /dev/video{idx}...")
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    return cap

def main():
    args = parse_args()
    cap = open_camera(args)

    if not cap.isOpened():
        print("[ERROR] Could not open camera. If on Jetson, try: sudo systemctl restart nvargus-daemon")
        sys.exit(1)

    streamer = None
    if args.stream:
        streamer = MJPEGStreamer(args.port)
        streamer.start()

    print("\n" + "="*60)
    print(" CAMERA FOCUS PEAKING TOOL READY")
    print(f" Web Browser: http://192.168.55.1:{args.port} or http://localhost:{args.port}")
    print(" 1. Aim the camera at a distant object (trees/roof/building 20m+ away).")
    print(" 2. Rotate the lens NEAR / FAR ring slowly.")
    print(" 3. Watch the 'Sharpness' number and the center 4x zoom loupe.")
    print(" 4. When the score hits its PEAK, tighten the lens lock screw!")
    print("="*60 + "\n")

    max_score = 0.0
    has_display = not args.no_gui

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]

            # 1. Compute Focus Metric: Laplacian Variance on center ROI
            roi_h, roi_w = int(h * 0.40), int(w * 0.40)
            cy, cx = h // 2, w // 2
            center_roi = frame[cy - roi_h // 2:cy + roi_h // 2, cx - roi_w // 2:cx + roi_w // 2]

            gray_roi = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
            if score > max_score:
                max_score = score

            # 2. Extract Center 4x Digital Zoom Loupe (picture-in-picture)
            loupe_h, loupe_w = int(h * 0.125), int(w * 0.125)
            crop_zoom = frame[cy - loupe_h:cy + loupe_h, cx - loupe_w:cx + loupe_w]
            if crop_zoom.size > 0:
                loupe_display = cv2.resize(crop_zoom, (320, 240), interpolation=cv2.INTER_NEAREST)
                frame[20:260, w - 340:w - 20] = loupe_display
                cv2.rectangle(frame, (w - 340, 20), (w - 20, 260), (0, 255, 255), 2)
                cv2.putText(frame, "CENTER 4x ZOOM", (w - 330, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Draw box showing where loupe is looking
            cv2.rectangle(frame, (cx - loupe_w, cy - loupe_h), (cx + loupe_w, cy + loupe_h), (0, 255, 255), 1)

            # 3. Draw On-Screen Display (OSD)
            ratio = min(1.0, score / max(1.0, max_score)) if max_score > 0 else 0.0
            bar_color = (0, 255, 0) if ratio > 0.90 else ((0, 165, 255) if ratio > 0.70 else (0, 255, 255))

            # Header background banner
            cv2.rectangle(frame, (10, 10), (450, 140), (20, 20, 25), -1)
            cv2.rectangle(frame, (10, 10), (450, 140), (100, 100, 100), 1)

            cv2.putText(frame, f"Current Sharpness: {score:6.1f}", (25, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Peak Sharpness:    {max_score:6.1f}", (25, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Progress bar
            bar_x, bar_y, bar_len, bar_h = 25, 105, 380, 16
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len, bar_y + bar_h), (60, 60, 60), -1)
            fill_w = int(bar_len * ratio)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

            if streamer is not None:
                streamer.update_frame(frame)

            if has_display:
                try:
                    cv2.imshow("Camera Focus Peaking Tool", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [27, ord('q')]:
                        break
                    elif key == ord('r'):
                        max_score = 0.0
                except cv2.error:
                    has_display = False  # Headless fallback if no X11 display

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if streamer:
            streamer.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
