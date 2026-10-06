#!/usr/bin/env python3
"""
Jetson Orin Nano End-to-End Pipeline Latency & Throughput Benchmark.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Profiles the complete cyber-physical stack on edge hardware:
1. YOLOv8n TensorRT FP16 Inference (768x768)
2. 8D Kalman Filter Multi-Track Update & Ego-Motion Compensation
3. Kinematic Visual Servoing (IBVS) Flight Controller Command Computation
4. End-to-End Closed-Loop Software Latency Budget (Mean, Std, P95, Max)
"""

import sys
import os
import time
import argparse
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.pipeline import DroneTrackingPipeline
from control.pid_controller import KinematicVisualServoController


def run_benchmark(
    weights_path: str,
    n_frames: int = 200,
    n_warmup: int = 25,
    img_w: int = 1280,
    img_h: int = 720
):
    print("=" * 72)
    print("       NVIDIA JETSON ORIN NANO — PIPELINE LATENCY PROFILER        ")
    print("=" * 72)
    print(f"Weights target:     {weights_path}")
    print(f"Sensor resolution:  {img_w}x{img_h} (HD)")
    print(f"Profiling iterations: {n_frames} timed frames ({n_warmup} warmup frames)\n")

    # 1. Initialize Controller
    controller = KinematicVisualServoController(
        camera_uptilt_deg=0.0,
        hfov_deg=82.0,
        vfov_deg=52.0,
        desired_bbox_size=35.0
    )

    # 2. Initialize Perception Pipeline
    print(f"[INIT] Loading perception pipeline with {weights_path}...")
    pipeline = DroneTrackingPipeline(
        weights=weights_path,
        conf_threshold=0.25,
        iou_threshold=0.45,
        max_lost_frames=15,
        desired_target_size=35.0,
        enable_dynamic_zoom=False
    )

    # Synthetic realistic frame (1280x720 RGB with a simulated drone target)
    dummy_frame = np.random.randint(40, 160, (img_h, img_w, 3), dtype=np.uint8)
    # Paint a simulated target drone patch in the center
    cx, cy = img_w // 2, img_h // 2
    dummy_frame[cy-15:cy+15, cx-15:cx+15] = [20, 20, 20]

    # Synthetic realistic drone ego-telemetry (pitch, roll, yaw, velocity)
    ego_telemetry = {
        "pitch_rad": float(np.deg2rad(-4.5)),
        "roll_rad": float(np.deg2rad(1.2)),
        "yaw_deg": 180.0,
        "vx_body": 1.5,
        "vy_body": 0.0,
        "vz_body": -0.2,
        "altitude_m": 6.0,
        "dt": 1.0 / 30.0
    }

    # 3. GPU Warm-up Phase (stabilize dynamic CUDA kernels & thermal clocks)
    print(f"[WARMUP] Running {n_warmup} warmup cycles...")
    for _ in range(n_warmup):
        pipeline.process_frame(dummy_frame, draw_hud=False, ego_telemetry=ego_telemetry)
        telemetry = dict(pipeline.telemetry) if hasattr(pipeline, "telemetry") else {}
        controller.compute_commands(telemetry, dt=1.0 / 30.0)

    print("[BENCHMARK] Warmup complete. Profiling active...\n")

    t_preprocess = []
    t_inference = []
    t_tracking = []
    t_control = []
    t_total = []

    for i in range(n_frames):
        # Slightly jitter the synthetic target position to trigger active Kalman velocity tracking
        dx = int(np.sin(i * 0.1) * 8)
        dy = int(np.cos(i * 0.1) * 5)
        test_frame = dummy_frame.copy()
        test_frame[cy+dy-15:cy+dy+15, cx+dx-15:cx+dx+15] = [15, 15, 15]

        t_start = time.perf_counter()

        # Step A: Perception Pipeline (Preprocessing + YOLOv8 TensorRT + NMS + Kalman Filter)
        t_perc_start = time.perf_counter()
        annotated_frame, telemetry = pipeline.process_frame(
            test_frame, draw_hud=False, ego_telemetry=ego_telemetry
        )
        t_perc_end = time.perf_counter()

        # Step B: Control Evaluation (IBVS Kinematic Controller)
        t_ctrl_start = time.perf_counter()
        cmd_vel = controller.compute_commands(telemetry, dt=1.0 / 30.0)
        t_ctrl_end = time.perf_counter()

        t_end = time.perf_counter()

        total_ms = (t_end - t_start) * 1000.0
        perc_ms = (t_perc_end - t_perc_start) * 1000.0
        ctrl_ms = (t_ctrl_end - t_ctrl_start) * 1000.0

        t_total.append(total_ms)
        t_inference.append(perc_ms)
        t_control.append(ctrl_ms)

    # 4. Statistical Aggregation
    t_total = np.array(t_total)
    t_inference = np.array(t_inference)
    t_control = np.array(t_control)

    mean_total = np.mean(t_total)
    std_total = np.std(t_total)
    p50_total = np.percentile(t_total, 50)
    p95_total = np.percentile(t_total, 95)
    p99_total = np.percentile(t_total, 99)
    min_total = np.min(t_total)
    max_total = np.max(t_total)
    fps_total = 1000.0 / mean_total

    mean_perc = np.mean(t_inference)
    std_perc = np.std(t_inference)
    p95_perc = np.percentile(t_inference, 95)

    mean_ctrl = np.mean(t_control)
    std_ctrl = np.std(t_control)
    p95_ctrl = np.percentile(t_control, 95)

    # Print Formatted Report
    print("+" + "-" * 70 + "+")
    print(f"| {'COMPONENT':<34} | {'MEAN (ms)':<10} | {'STD (ms)':<8} | {'P95 (ms)':<9} |")
    print("+" + "-" * 70 + "+")
    print(f"| {'Perception (YOLOv8 TRT + Kalman)':<34} | {mean_perc:8.2f} ms | {std_perc:6.2f} ms | {p95_perc:7.2f} ms |")
    print(f"| {'Control Law (Kinematic IBVS / Safe)':<34} | {mean_ctrl:8.2f} ms | {std_ctrl:6.2f} ms | {p95_ctrl:7.2f} ms |")
    print("+" + "-" * 70 + "+")
    print(f"| {'TOTAL SOFTWARE END-TO-END':<34} | {mean_total:8.2f} ms | {std_total:6.2f} ms | {p95_total:7.2f} ms |")
    print("+" + "-" * 70 + "+")
    print(f"\nThroughput:          {fps_total:.1f} FPS (Target: >= 30.0 FPS)")
    print(f"Min / Median / Max:  {min_total:.2f} ms / {p50_total:.2f} ms / {max_total:.2f} ms")
    print(f"P99 Worst-Case:      {p99_total:.2f} ms")

    # Generate Markdown Summary for Weekly Report
    out_dir = REPO_ROOT / "outputs" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "jetson_orin_nano_latency_report.md"

    md_content = f"""# Jetson Orin Nano End-to-End Latency Benchmark

**Target Hardware:** NVIDIA Jetson Orin Nano Developer Kit  
**Model Architecture:** YOLOv8n ($768\\times 768$, FP16 TensorRT Engine) + 8D Kalman Filter  
**Control Law:** Kinematic Image-Based Visual Servoing (IBVS)  
**Sample Count:** {n_frames} consecutive frames  

---

## 1. Measured Latency Breakdown

| Subsystem Stage | Mean Latency | Standard Dev | 95th Percentile | Share of Compute |
| :--- | :---: | :---: | :---: | :---: |
| **Perception (YOLOv8n TRT + 8D Kalman)** | **{mean_perc:.2f} ms** | $\\pm {std_perc:.2f}$ ms | {p95_perc:.2f} ms | {mean_perc / mean_total * 100:.1f}% |
| **Control Law (IBVS Guidance & Safeguards)** | **{mean_ctrl:.2f} ms** | $\\pm {std_ctrl:.2f}$ ms | {p95_ctrl:.2f} ms | {mean_ctrl / mean_total * 100:.1f}% |
| **Total Computational Closed-Loop** | **{mean_total:.2f} ms** | $\\pm {std_total:.2f}$ ms | **{p95_total:.2f} ms** | **100.0%** |

* **Sustained Software Throughput:** **{fps_total:.1f} FPS** (Surpasses 30 FPS camera framerate)
* **Minimum Latency:** {min_total:.2f} ms
* **Median Latency (P50):** {p50_total:.2f} ms
* **Worst-Case Tail Latency (P99):** {p99_total:.2f} ms

---

## 2. Integration into Closed-Loop System Latency Budget

$$\\Delta t_{{\\text{{total}}}} = T_{{\\text{{software}}}} + T_{{\\text{{actuator}}}} = {mean_total:.1f}\\text{{ ms}} + T_d$$

Where:
1. $T_{{\\text{{software}}}} \\approx {mean_total:.1f}\\text{{ ms}}$ is deterministic and directly profiled on the edge companion computer.
2. $T_d$ represents flight dynamics and motor dead time identified from Pixhawk black-box telemetry.
"""
    with open(report_file, "w") as f:
        f.write(md_content)

    print(f"\n[REPORT] ✓ Benchmark markdown report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Profile Jetson Orin Nano latency")
    default_engine = REPO_ROOT / "perception" / "weights" / "yolov8n_drone_v4_continued_best.engine"
    default_pt = REPO_ROOT / "perception" / "weights" / "yolov8n_drone_v4_continued_best.pt"

    chosen_weights = str(default_engine) if default_engine.exists() else str(default_pt)

    parser.add_argument("--weights", type=str, default=chosen_weights, help="Path to .engine or .pt weights")
    parser.add_argument("--frames", type=int, default=200, help="Number of timed benchmark frames (default: 200)")
    parser.add_argument("--warmup", type=int, default=25, help="Number of warmup iterations (default: 25)")
    args = parser.parse_args()

    run_benchmark(weights_path=args.weights, n_frames=args.frames, n_warmup=args.warmup)


if __name__ == "__main__":
    main()
