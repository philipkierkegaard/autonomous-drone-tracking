#!/usr/bin/env python3
"""
Camera & Vision Pipeline Frequency Profiler
Measures:
1. Sensor Ingestion Rate (Camera FPS)
2. YOLOv8 Inference Latency & Throughput (Inference FPS)
3. VideoWriter Encoding Latency & Bottleneck
4. End-to-End Real-World Frequency vs. 30 FPS target
"""

import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import queue
import threading


class AsyncVideoWriter:
    """Non-blocking background video writer using worker thread and queue."""
    def __init__(self, filename: str, fourcc: int, fps: float, frame_size: tuple, max_queue: int = 120):
        self.filename = filename
        self.writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        self.queue = queue.Queue(maxsize=max_queue)
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.05)
                self.writer.write(frame)
                self.queue.task_done()
            except queue.Empty:
                continue

    def write(self, frame: np.ndarray):
        if not self.running:
            return
        try:
            self.queue.put_nowait(frame.copy())
        except queue.Full:
            try:
                _ = self.queue.get_nowait()
                self.queue.put_nowait(frame.copy())
            except Exception:
                pass

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.writer.release()


def build_csi_pipeline(sensor_id=0, width=1280, height=720, framerate=30, flip=0, mode_1080p=True):
    cap_w = 1920 if mode_1080p else 3840
    cap_h = 1080 if mode_1080p else 2160
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={cap_w}, height={cap_h}, format=NV12, framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false"
    )


def profile_pipeline(args):
    print("\n=======================================================")
    print("   CAMERA & VISION PIPELINE FREQUENCY PROFILER         ")
    print("=======================================================")
    print(f"• Source:        {'CSI Camera' if args.csi else args.source}")
    print(f"• Target Frames: {args.frames}")
    print(f"• Weights:       {args.weights or 'None (Raw Sensor Capture)'}")
    print(f"• Record Video:  {args.record}")
    print("=======================================================\n")

    # 1. Open Capture Source
    if args.csi:
        pipe = build_csi_pipeline(
            sensor_id=int(args.source) if args.source.isdigit() else 0,
            mode_1080p=not getattr(args, "mode_4k", False)
        )
        print(f"[CAPTURE] Opening CSI pipeline: {pipe}")
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    elif args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print(f"[ERROR] Failed to open video source: {args.source}")
        return

    # Try setting 1280x720 @ 30 FPS on standard webcam
    if not args.csi and args.source.isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30.0)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sensor_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[CAPTURE] Stream opened: {w}x{h} (driver reported FPS: {sensor_fps:.1f})")

    # 2. Load YOLO Model if requested
    model = None
    if args.weights:
        print(f"[MODEL] Loading detector: {args.weights}")
        try:
            from ultralytics import YOLO
            model = YOLO(args.weights)
            # Warm up
            dummy = np.zeros((h, w, 3), dtype=np.uint8)
            _ = model.predict(dummy, verbose=False, imgsz=args.imgsz)
            print("[MODEL] ✓ Detector initialized and warmed up.")
        except Exception as e:
            print(f"[MODEL] Failed to load YOLO: {e}")
            return

    # 3. Setup VideoWriter if requested
    writer = None
    writer_raw = None
    if args.record:
        out_dir = PROJECT_ROOT / "outputs" / "recordings" / "profiler"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        rec_path = out_dir / f"profile_tracked_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_cls = AsyncVideoWriter if args.async_write else cv2.VideoWriter
        writer_name = "Threaded Async" if args.async_write else "Synchronous Blocking"
        writer = writer_cls(str(rec_path), fourcc, 30.0, (w, h))
        print(f"[RECORD] Writing test video via {writer_name} writer at 30.0 FPS to: {rec_path}")
        if args.record_both:
            raw_path = out_dir / f"profile_raw_{ts}.mp4"
            writer_raw = writer_cls(str(raw_path), fourcc, 30.0, (w, h))
            print(f"[RECORD] Writing raw test video via {writer_name} writer at 30.0 FPS to: {raw_path}")

    # 4. Benchmarking Loop
    print("\n[RUNNING] Collecting frame timestamps...")
    t_captures = []
    t_inferences = []
    t_encodes = []
    t_loop_intervals = []

    t_start = time.perf_counter()
    prev_loop_time = t_start

    frame_count = 0
    try:
        while frame_count < args.frames:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            t1 = time.perf_counter()
            if not ret or frame is None:
                print(f"[WARN] Frame read failed at index {frame_count}")
                break

            t_captures.append((t1 - t0) * 1000.0)

            # Inference
            t2 = t1
            if model is not None:
                _ = model.predict(frame, verbose=False, imgsz=args.imgsz, conf=0.25)
                t2 = time.perf_counter()
                t_inferences.append((t2 - t1) * 1000.0)

            # Encoding / VideoWriter
            t3 = t2
            if writer is not None:
                writer.write(frame)
                if writer_raw is not None:
                    writer_raw.write(frame)
                t3 = time.perf_counter()
                t_encodes.append((t3 - t2) * 1000.0)

            loop_now = time.perf_counter()
            loop_dt = loop_now - prev_loop_time
            prev_loop_time = loop_now
            t_loop_intervals.append(loop_dt)

            frame_count += 1
            if frame_count % 10 == 0:
                inst_fps = 1.0 / max(1e-4, loop_dt)
                print(f"  Frame {frame_count:3d}/{args.frames} | Inst FPS: {inst_fps:5.1f} | Cap: {t_captures[-1]:4.1f}ms | " +
                      (f"YOLO: {t_inferences[-1]:4.1f}ms | " if model else "") +
                      (f"Enc: {t_encodes[-1]:4.1f}ms" if writer else ""))

    except KeyboardInterrupt:
        print("\n[USER] Interrupted benchmark.")

    t_total = time.perf_counter() - t_start
    cap.release()
    if writer:
        writer.release()
    if writer_raw:
        writer_raw.release()

    # 5. Summary Statistics
    if frame_count < 2:
        print("[ERROR] Insufficient frames collected.")
        return

    # Drop first frame initialization artifact
    intervals = np.array(t_loop_intervals[1:])
    fps_arr = 1.0 / intervals
    mean_fps = float(np.mean(fps_arr))
    median_fps = float(np.median(fps_arr))
    p5_fps = float(np.percentile(fps_arr, 5))
    p95_fps = float(np.percentile(fps_arr, 95))
    speedup_ratio = 30.0 / max(0.1, mean_fps)

    print("\n=======================================================")
    print("               PROFILER RESULTS SUMMARY                ")
    print("=======================================================")
    print(f"• Total Elapsed Wall Time:  {t_total:.2f} s")
    print(f"• Total Processed Frames:   {frame_count}")
    print(f"• Real Frequency (Mean):    {mean_fps:.2f} FPS")
    print(f"• Real Frequency (Median):  {median_fps:.2f} FPS")
    print(f"• 5th-95th Percentile:      {p5_fps:.1f} - {p95_fps:.1f} FPS")
    print(f"• Mean Camera Grab Time:    {np.mean(t_captures):.2f} ms")
    if t_inferences:
        print(f"• Mean YOLO Inference Time: {np.mean(t_inferences):.2f} ms ({1000.0/np.mean(t_inferences):.1f} model FPS)")
    if t_encodes:
        print(f"• Mean Video Encode Time:   {np.mean(t_encodes):.2f} ms")
    print("-------------------------------------------------------")
    print(f"• VideoWriter Container FPS: 30.0 FPS")
    print(f"• Playback Speedup Factor:   {speedup_ratio:.2f}x " +
          ("(REAL TIME MATCH ✓)" if 0.95 <= speedup_ratio <= 1.05 else f"(SPED UP by {(speedup_ratio-1.0)*100:+.0f}%)"))
    if speedup_ratio > 1.10:
        print(f"\n[DIAGNOSIS] Because the system ran at {mean_fps:.1f} FPS while the MP4 container")
        print(f"was saved at 30.0 FPS, a {t_total:.1f}s real-world event will play back in")
        print(f"only {frame_count/30.0:.1f}s, appearing {speedup_ratio:.2f}x FASTER than reality!")
    print("=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Profile Camera & Vision Pipeline Frequency.")
    parser.add_argument("--source", type=str, default="0", help="Camera index or video path.")
    parser.add_argument("--csi", action="store_true", help="Use NVIDIA Jetson CSI pipeline.")
    parser.add_argument("--mode-4k", action="store_true", help="Force 4K sensor mode (default: 1080p high-speed mode).")
    parser.add_argument("--weights", type=str, default=None, help="YOLO model path (optional).")
    parser.add_argument("--imgsz", type=int, default=768, help="YOLO inference image size.")
    parser.add_argument("--record", action="store_true", help="Enable VideoWriter encoding.")
    parser.add_argument("--record-both", action="store_true", help="Test encoding both tracked and raw.")
    parser.add_argument("--async-write", action="store_true", default=True, help="Use non-blocking AsyncVideoWriter (default: True).")
    parser.add_argument("--sync-write", action="store_false", dest="async_write", help="Force synchronous blocking VideoWriter.")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames to profile.")
    args = parser.parse_args()

    profile_pipeline(args)


if __name__ == "__main__":
    main()
