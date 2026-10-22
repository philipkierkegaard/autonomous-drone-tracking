# Cyber-Physical Edge Perception & Guidance Pipeline

This directory documents the on-device real-time software pipeline and hardware benchmarking for the **NVIDIA Jetson Orin Nano** companion computer.

---

## 1. Documentation Index

* **[JETSON_LATENCY_640_CIOU_BENCHMARK.md](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/docs/pipeline/JETSON_LATENCY_640_CIOU_BENCHMARK.md)**  
  Full benchmark report for the baseline $640\times 640$ CIoU TensorRT engine (`yolov8n_v3_best.engine`).  
  *Execution time: **21.20 ms** compute, **47.2 FPS** capacity, **96.90 ms** photon-to-thrust delay, 0 deadline overruns.*

* **[768×768 Hybrid NWD Benchmark Report](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/outputs/benchmarks/jetson_orin_nano_capture_to_actuation_latency_report.md)**  
  Full benchmark report for the proposed $768\times 768$ Hybrid NWD + C2f-51 streamlined engine (`yolov8n_drone_768_hybrid_nwd_c51_best.engine`).  
  *Execution time: **15.63 ms** compute, **64.0 FPS** capacity, **91.32 ms** photon-to-thrust delay, 0 deadline overruns.*

---

## 2. On-Device Comparative Performance Summary

Both models were profiled over **1,000 continuous hardware CSI frames** on the physical NVIDIA Jetson Orin Nano running the active IBVS guidance loop:

| Metric | 640×640 CIoU Baseline (`v3_best`) | 768×768 Hybrid NWD (`c51_best`) | Comparison / Advantage |
| :--- | :---: | :---: | :--- |
| **Input Grid** | $640 \times 640$ (0.41 MP) | $768 \times 768$ (0.59 MP) | +44% spatial resolution |
| **YOLO + 8D Kalman Execution** | **20.48 ms** | **15.10 ms** | **768 Hybrid is 5.38 ms faster** |
| **Total Host Software Compute ($T_{\text{compute}}$)** | **21.20 ms** | **15.63 ms** | **768 Hybrid saves 5.57 ms** |
| **Theoretical Processing Capacity** | **47.2 FPS** | **64.0 FPS** | **+16.8 FPS throughput headroom** |
| **30 Hz Deadline Margin ($33.33\text{ ms} - T_{\text{compute}}$)** | **+12.13 ms** (36.4%) | **+17.71 ms** (53.1%) | Hard real-time verified on both |
| **Deadline Overruns (>33.3 ms)** | **0 / 1,000 (0.00%)** | **0 / 1,000 (0.00%)** | Zero frame drops |
| **End-to-End Photon-to-Thrust Delay** | **96.90 ms** | **91.32 ms** | **768 Hybrid is 5.58 ms faster** |
| **Target mAP@50 (Test Set)** | 88.4% | **92.2%** | **+3.8% accuracy gain** |
| **Small Target Recall ($<32\text{px}$)** | 71.2% | **86.4%** | **+15.2% recall on distant targets** |

---

## 3. Cyber-Physical Photon-to-Thrust Equation

$$\Delta t_{\text{total}} = \underbrace{T_{\text{exposure}} + T_{\text{readout}} + T_{\text{ISP}}}_{T_{\text{sensor}} = 32.67\text{ ms}} + \underbrace{T_{\text{preprocess}} + T_{\text{yolo}} + T_{\text{kalman}} + T_{\text{control}} + T_{\text{serialize}}}_{T_{\text{compute}} \in [15.6\text{ ms}, 21.2\text{ ms}]} + \underbrace{T_{\text{UART}} + T_{\text{FC\_sample}} + \tau_{\text{motor}}}_{T_{\text{actuation}} = 43.03\text{ ms}}$$

Software compute accounts for only **16%–22%** of total delay; sensor exposure integration ($32.7\text{ ms}$) and physical motor lag ($35.0\text{ ms}$) dominate the loop time constant.
