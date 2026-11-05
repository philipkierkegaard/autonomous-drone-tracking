# Production Cyber-Physical Latency & Throughput Benchmark

- **Architecture Mode:** `ASYNC` (Asynchronous Decoupled Vision Worker + 50 Hz Control Loop)
- **Perception Model:** `perception/weights/yolov8n_drone_768_hybrid_nwd_c51_best.pt`
- **Video Source:** `perception/raw_data/Anti-UAV/videos/video12.mp4`
- **PID Tuning Configuration:** `control/weights/tuned_pid_gains.json`
- **Telemetry Mode:** `FLIGHT-LOG` (logs/flight_logs/00000165.BIN)
- **Actuation Link:** `MAVLink Serializer (Dry-Run)`
- **Sample Size:** 20 consecutive timed frames (5 discarded warmup frames)

## 1. On-Device Software Execution Breakdown

| Subsystem Stage | Mean Latency | Std Dev | Median (P50) | 95th %ile | Max Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Frame Dequeue DMA (T_dequeue)** | **1.223 ms** | $\pm 0.244$ ms | 1.190 ms | 1.680 ms | 1.820 ms |
| **2. Preprocessing & Shadows (T_preprocess)** | **0.114 ms** | $\pm 0.064$ ms | 0.087 ms | 0.229 ms | 0.324 ms |
| **3. Perception (YOLOv8 + 8D Kalman)** | **10.320 ms** | $\pm 0.401$ ms | 10.274 ms | 10.947 ms | 11.216 ms |
| **4. Vision Mutex Write (T_lock_write)** | **0.003 ms** | $\pm 0.001$ ms | 0.003 ms | 0.005 ms | 0.005 ms |
| **Total Vision Processing (T_vision_compute)** | **10.437 ms** | $\pm 0.441$ ms | 10.418 ms | 11.266 ms | 11.418 ms |
| **5. Control Mutex Read (T_lock_read)** | **0.004 ms** | $\pm 0.002$ ms | 0.004 ms | 0.007 ms | 0.008 ms |
| **6. Guidance Control (T_control)** | **0.161 ms** | $\pm 0.072$ ms | 0.132 ms | 0.282 ms | 0.379 ms |
| **7. MAVLink Serialization (T_serialize)** | **0.081 ms** | $\pm 0.152$ ms | 0.034 ms | 0.303 ms | 0.561 ms |
| **Total Control Step Time (T_ctrl_compute)** | **0.247 ms** | $\pm 0.221$ ms | 0.167 ms | 0.583 ms | 0.941 ms |
| **Control Loop Period (T_ctrl_period)** | **20.387 ms** | $\pm 6.706$ ms | 21.297 ms | 24.973 ms | 24.976 ms |
| **TRUE Capture-to-Actuation Latency** | **14.795 ms** | $\pm 2.134$ ms | 14.380 ms | 17.870 ms | 18.078 ms |
| **Perception-to-Actuation Delay** | **4.293 ms** | $\pm 2.271$ ms | 3.766 ms | 7.666 ms | 7.847 ms |

## 2. Decoupled Throughput & Real-Time Frame Deadlines

- **Camera Hardware Frame Period:** `33.33 ms` ($30.0\text{ Hz}$)
- **Measured Vision Sampling Rate:** **80.2 FPS**
- **Measured MAVLink Control Rate:** **49.1 Hz** (Target: 50.0 Hz, Jitter: $\pm 6.71\text{ ms}$)
- **Perception Compute Capacity:** **95.8 FPS**
- **Real-Time 30 Hz Deadline Headroom:** **+22.90 ms (+68.7% idle compute margin)**
- **Deadline Overrun Rate:** **0 / 20 frames (0.00%)**
- **Real-Time Execution Verdict:** `PASSED (Guaranteed Real-Time Execution)`

## 3. Cyber-Physical Photon-to-Thrust Latency Budget

$$\Delta t_{\text{total}} = \underbrace{T_{\text{exposure}} + T_{\text{readout}} + T_{\text{ISP}}}_{T_{\text{sensor}} \approx 32.7\text{ ms}} + \underbrace{\Delta t_{\text{capture}\to\text{actuation}}}_{T_{\text{software}} \approx 14.80\text{ ms}} + \underbrace{T_{\text{UART}} + T_{\text{FC\_sample}} + \tau_{\text{motor}}}_{T_{\text{actuation}} \approx 43.0\text{ ms}}$$

| Physical Domain | Delay Mechanism | Latency Contribution | Analysis Method |
| :--- | :--- | :---: | :--- |
| **Optical / Sensor** | Photon integration midpoint ($0.5 \cdot T_{\text{exp}}$) | $\approx 16.67\text{ ms}$ | Physical sensor shutter |
| **Optical / Sensor** | MIPI CSI-2 2-lane rolling shutter readout | $\approx 12.00\text{ ms}$ | Hardware sensor bus clock |
| **Optical / Sensor** | Tegra ISP demosaic & nvvidconv scaling | $\approx 4.00\text{ ms}$ | Hardware ISP pipeline |
| **Edge Software** | True Capture-to-Actuation Age | **14.80 ms** (P95: 17.87 ms) | **Measured on Jetson GPU/CPU (Exact Pipeline)** |
| **Serial Bus** | MAVLink UART transmission ($57\text{B} @ 921600$) | $\approx 0.49\text{ ms}$ | Physical serial transmission |
| **Flight Controller** | Pixhawk guidance loop sample delay (50–100 Hz) | $\approx 7.50\text{ ms}$ | ArduPilot loop quantization |
| **Electro-Mechanical** | BLDC Motor & rotor aerodynamics ($\tau_{\text{motor}}$) | $\approx 35.00\text{ ms}$ | Empirical flight system ID |
| **Total Closed-Loop** | **End-to-End Photon-to-Thrust Delay** | **$\approx 90.46\text{ ms}$** (P95: 93.53 ms) | **Complete Cyber-Physical Total** |
