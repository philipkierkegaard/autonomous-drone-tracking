# Jetson Orin Nano Cyber-Physical Capture-to-Actuation Latency Report

**Model Weights:** `perception/weights/yolov8n_v3_best.engine` (640×640 FP16 TensorRT)  
**Video Source:** Jetson CSI Camera (Hardware GStreamer, Sony IMX477)  
**Telemetry Mode:** `FLIGHT-LOG` (`logs/flight_logs/00000080.BIN`)  
**Flight Controller:** Kinematic Visual Servoing (`ibvs`)  
**Actuation Interface:** MAVLink Serializer (`SET_POSITION_TARGET_LOCAL_NED`)  
**Sample Size:** 1,000 timed frames (50 warmup discarded)  
**Date:** October 2026  

---

## 1. Measured On-Device Software Execution Breakdown

| Subsystem Stage | Mean (ms) | Std (ms) | Median | 95%ile | Max (ms) | Compute Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Frame Dequeue DMA ($T_{\text{dequeue}}$)** | 12.214 | 2.020 | 12.198 | 15.420 | 19.946 | — *(I/O Wait)* |
| **2. Preprocessing & Shadows ($T_{\text{preprocess}}$)** | 0.469 | 0.086 | 0.459 | 0.623 | 0.894 | 2.2% |
| **3. Perception (YOLOv8 TRT + 8D Kalman)** | 20.484 | 0.849 | 20.937 | 21.261 | 23.462 | 96.6% |
| **4. Guidance Control ($T_{\text{control}}$)** | 0.099 | 0.032 | 0.099 | 0.115 | 0.462 | 0.5% |
| **5. MAVLink Serialization ($T_{\text{serialize}}$)** | 0.151 | 0.032 | 0.154 | 0.200 | 0.417 | 0.7% |
| **Total Software Processing ($T_{\text{compute}}$)** | **21.203** | **0.850** | **21.652** | **21.974** | **24.115** | **100.0%** |

---

## 2. Real-Time Schedulability & Throughput

* **Compute Capacity (Headroom):** 47.2 FPS (max processing speed)
* **Actual Wall-Clock Rate:** 29.9 Hz (sensor acquisition, jitter ±1.82 ms)
* **Real-Time 30 Hz Deadline Headroom:** +12.13 ms (+36.4% spare capacity)
* **Deadline Overruns (>33.3 ms):** **0 of 1,000 frames (0.00%)**

---

## 3. Full Cyber-Physical System Latency Budget (Photon-to-Thrust)

$$\Delta t_{\text{total}} = T_{\text{sensor}} (32.67\text{ ms}) + T_{\text{compute}} (21.20\text{ ms}) + T_{\text{actuation}} (43.03\text{ ms}) = \mathbf{96.90\text{ ms}}$$

1. **Optical Domain (Exposure + CSI Readout + ISP DMA):** 32.67 ms
2. **Compute Domain (Host Preprocess + YOLO + IBVS):** 21.20 ms (P95: 21.97 ms)
3. **Actuation Domain (UART Wire 0.53ms + Pixhawk 7.5ms + Motor 35.0ms):** 43.03 ms
* **Total Photon-to-Thrust Latency (Mean):** **96.90 ms**
* **Total Photon-to-Thrust Latency (P95 Worst-Case):** **97.67 ms**
