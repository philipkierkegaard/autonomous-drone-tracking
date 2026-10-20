# Cyber-Physical End-to-End Latency & Throughput Benchmark

- **Device:** NVIDIA Jetson Orin Nano (MAXN Power Mode)
- **Perception Model:** `perception/weights/yolov8n_drone_768_hybrid_nwd_c51_best.engine` (FP16 TensorRT Engine, 768×768, 8 MiB)
- **Video Source:** Sony IMX477 MIPI CSI-2 Camera (`nvarguscamerasrc` GStreamer Hardware Pipeline)
- **Telemetry Mode:** Live Attitude & Velocity Stream
- **Controller:** 4-DOF Kinematic Visual Servoing (IBVS)
- **Actuation Link:** MAVLink `SET_POSITION_TARGET_LOCAL_NED` Serializer
- **Sample Size:** **1,000 consecutive timed frames** (50 discarded warmup frames)

---

## 1. On-Device Software Execution Breakdown

All statistics measured over 1,000 consecutive frames on physical Jetson Orin Nano hardware:

| Subsystem Stage | Mean Latency | Std Dev | Median (P50) | 95th %ile | Max Latency | Compute Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Frame Dequeue DMA (T_dequeue)** | **17.807 ms** | $\pm 1.074$ ms | 17.924 ms | 19.265 ms | 20.730 ms | — *(I/O wait)* |
| **2. Preprocessing & Shadows (T_preprocess)** | **0.330 ms** | $\pm 0.038$ ms | 0.325 ms | 0.371 ms | 0.763 ms | 2.1% |
| **3. Perception (YOLOv8 TRT + 8D Kalman)** | **15.099 ms** | $\pm 0.553$ ms | 14.976 ms | 16.413 ms | 17.993 ms | 96.6% |
| **4. Guidance Control (T_control)** | **0.081 ms** | $\pm 0.043$ ms | 0.072 ms | 0.113 ms | 0.378 ms | 0.5% |
| **5. MAVLink Serialization (T_serialize)** | **0.115 ms** | $\pm 0.018$ ms | 0.116 ms | 0.126 ms | 0.420 ms | 0.8% |
| **Total Software Compute (T_compute)** | **15.625 ms** | $\pm 0.592$ ms | **15.495 ms** | **17.050 ms** | **18.812 ms** | **100.0%** |

---

## 2. Real-Time Frame Deadline & Throughput Verification

- **Camera Hardware Frame Period:** `33.33 ms` ($30.0\text{ Hz}$)
- **Measured Wall-Clock Sampling Rate:** **29.9 Hz** (Frame interval jitter: $\pm 0.88\text{ ms}$)
- **Compute Capacity Headroom:** **64.0 FPS** (Max theoretical GPU processing rate)
- **Deadline Headroom Margin:** **+17.71 ms (+53.1% idle compute margin)**
- **Deadline Overrun Rate:** **0 / 1,000 frames (0.00%)**
- **Real-Time 30 Hz Verdict:** `PASSED (Guaranteed 30 Hz Deterministic Execution)`

---

## 3. Cyber-Physical Photon-to-Thrust Latency Budget

To accurately reflect physical flight dynamics in the thesis, the end-to-end response delay decomposes into three coupled physical domains:

$$\Delta t_{\text{total}} = \underbrace{T_{\text{exposure}} + T_{\text{readout}} + T_{\text{ISP}}}_{T_{\text{sensor}} \approx 32.7\text{ ms}} + \underbrace{T_{\text{pre}} + T_{\text{yolo}} + T_{\text{kalman}} + T_{\text{ctrl}} + T_{\text{ser}}}_{T_{\text{compute}} \approx 15.63\text{ ms}} + \underbrace{T_{\text{UART}} + T_{\text{FC\_sample}} + \tau_{\text{motor}}}_{T_{\text{actuation}} \approx 43.0\text{ ms}}$$

| Physical Domain | Delay Mechanism | Latency Contribution | Analysis Method |
| :--- | :--- | :---: | :--- |
| **Optical / Sensor** | Photon integration midpoint ($0.5 \cdot T_{\text{exp}}$) | $\approx 16.67\text{ ms}$ | Physical sensor shutter |
| **Optical / Sensor** | MIPI CSI-2 2-lane rolling shutter readout | $\approx 12.00\text{ ms}$ | Hardware sensor bus clock |
| **Optical / Sensor** | Tegra ISP demosaic & nvvidconv scaling | $\approx 4.00\text{ ms}$ | Hardware ISP pipeline |
| **Edge Software** | Total On-Device Compute ($T_{\text{compute}}$) | **15.63 ms** (P95: 17.05 ms) | **Measured on Jetson GPU/CPU (N=1,000)** |
| **Serial Bus** | MAVLink UART transmission ($57\text{B} @ 921600$) | $\approx 0.53\text{ ms}$ | Physical serial transmission |
| **Flight Controller** | Pixhawk guidance loop sample delay (50–100 Hz) | $\approx 7.50\text{ ms}$ | ArduPilot loop quantization |
| **Electro-Mechanical** | BLDC Motor & rotor aerodynamics ($\tau_{\text{motor}}$) | $\approx 35.00\text{ ms}$ | Empirical flight system ID |
| **Total Closed-Loop** | **End-to-End Photon-to-Thrust Delay** | **$\approx 91.32\text{ ms}$** (P95: 92.75 ms) | **Complete Cyber-Physical Total** |
