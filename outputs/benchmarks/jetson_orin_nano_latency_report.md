# Jetson Orin Nano End-to-End Latency Benchmark

**Target Hardware:** NVIDIA Jetson Orin Nano Developer Kit  
**Model Architecture:** YOLOv8n ($768\times 768$, FP16 TensorRT Engine) `yolov8n_drone_v4_continued_best.engine` + 8D Kalman Filter  
**Control Law:** Kinematic Image-Based Visual Servoing (IBVS)  
**Sample Count:** 200 consecutive frames (25 warmup frames)  
**Camera Resolution:** 1280x720 (HD)  

---

## 1. Measured Latency Breakdown

| Subsystem Stage | Mean Latency | Standard Dev | 95th Percentile | Share of Compute |
| :--- | :---: | :---: | :---: | :---: |
| **Perception (YOLOv8n TRT + 8D Kalman)** | **32.55 ms** | $\pm 0.91$ ms | 33.08 ms | 99.9% |
| **Control Law (IBVS Guidance & Safeguards)** | **0.03 ms** | $\pm 0.01$ ms | 0.04 ms | 0.1% |
| **Total Computational Closed-Loop** | **32.58 ms** | $\pm 0.91$ ms | **33.11 ms** | **100.0%** |

* **Sustained Software Throughput:** **30.7 FPS** (Surpasses 30.0 FPS camera framerate)
* **Minimum Latency:** 28.54 ms
* **Median Latency (P50):** 32.79 ms
* **Worst-Case Tail Latency (P99):** 33.82 ms
* **Maximum Observed Latency:** 36.04 ms
* **Engine Memory Footprint:** 8 MiB model file, +13 MiB GPU execution context

---

## 2. Integration into Closed-Loop System Latency Budget

$$\Delta t_{\text{total}} = T_{\text{software}} + T_{\text{sensor}} + T_{\text{actuator}} = 32.58\text{ ms} + 16.67\text{ ms} + T_d$$

Where:
1. $T_{\text{software}} \approx 32.58\text{ ms}$ is deterministic edge compute on the Jetson Orin Nano.
2. $T_{\text{sensor}} \approx 16.67\text{ ms}$ is half-frame rolling shutter exposure delay at 30 FPS.
3. $T_d \approx 70\text{--}90\text{ ms}$ represents Pixhawk PX4 velocity control loop dynamics and motor lag.
