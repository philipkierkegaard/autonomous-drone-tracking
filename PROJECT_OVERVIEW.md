# DTU Bachelor Project: Autonomous Drone Pursuit & Interception System

## Project Identity & High-Level Scope
* **Institution:** Danmarks Tekniske Universitet (DTU)
* **Type:** Dual-Track Bachelor Thesis Project (9-Week Schedule)
* **Goal:** Deliver a **fully functional autonomous pursuit/interception drone** by **Week 9**.
* **Core Architecture:** Real-time **Detect & Control (Visual Servoing)** pipeline spanning an airborne companion computer (NVIDIA Jetson) and flight controller (Pixhawk/PX4).

---

## Current Baseline & Completed Foundations (Ahead of Schedule)

Critical foundational milestones already verified and operational:
* [x] **Airframe Airworthiness:** The physical drone is built, powered, and verified stable in **manual hover**.
* [x] **Companion Environment:** NVIDIA Jetson Orin Nano flashed and configured with JetPack OS.
* [x] **Camera Pipeline:** Live interface between Jetson and the onboard camera is established and tested.
* [x] **Edge Inference:** A baseline YOLO model has already been successfully run on the Jetson Orin Nano.
* [x] **Perception Dataset:** Anti-UAV dataset collated into a 3-way sequence-stratified split with zero temporal data leakage (17,628 train / 3,606 val / 3,570 test frames).
* [x] **Development Setup:** Isolated Python 3.11 `.venv` created with Ultralytics, PyTorch, and OpenCV.

---

## Hardware & Software Stack

### Hardware Stack (Airframe, Avionics & Compute)
* **Airframe:** Proven multirotor platform, balanced and tuned for stable flight.
* **Companion Computer:** **NVIDIA Jetson Orin Nano** (running JetPack OS, hardware-accelerated TensorRT FP16 inference).
* **Flight Controller:** **Pixhawk** running **PX4 Autopilot**.
* **Inter-System Comms:** Hardware **UART** serial link running **MAVLink / MAVSDK-Python** (< 30 Hz throttled streams).
* **Sensors:** Live onboard camera interfaced to Jetson; Pixhawk IMU/telemetry.
* **Power Distribution:** Onboard regulated BEC power rails with low-ESR capacitors (preventing Jetson brownouts under motor load).
* **Safety Failsafe:** Dedicated hardware RC receiver kill-switch and manual override.

### Software & Simulation Stack
* **Perception Model:** Custom **YOLOv8n** trained on Anti-UAV (+ bird false-alarm mitigation) and exported to **TensorRT FP16**.
* **Control Stack:** 
  1. **Baseline Controller:** 3-axis Line-of-Sight Visual Tracking Controller (Yaw $\dot{\psi}$, Altitude $v_z$, Forward Velocity $v_x$) with temporal state filtering.
  2. **Advanced Policy (Simulation Study):** Adaptive tracking policy (RL / Policy Gradient in simulation).
  3. **Target Recovery:** Lost-target coasting state machine (kinematic velocity holding).
* **Simulation Environment:** Headless **PX4 SITL** + **QGroundControl (QGC)** on Mac with 3D-to-2D pinhole projection camera simulation.

---

## System Architecture: Detect-to-Control Pipeline

```
┌─────────────────────────┐          2D Bounding Box          ┌───────────────────────────┐
│     Live Camera Feed    │ ────►  [xc, yc, w, h, conf] ────► │ Line-of-Sight Controller  │
│ (YOLOv8n / TensorRT)    │                                   │   (Visual PID vs. Policy) │
└─────────────────────────┘                                   └─────────────┬─────────────┘
                                                                            │ MAVSDK Offboard
                                                                            ▼ Velocity Commands [vx, vy, vz, yaw]
┌─────────────────────────┐           UART (MAVLink)          ┌───────────────────────────┐
│   Hover-Proven Airframe │ ◄──────────────────────────────── │      Pixhawk Autopilot    │
│  (Motors / Actuators)   │                                   │         (PX4 Firmware)    │
└─────────────────────────┘                                   └───────────────────────────┘
```

### Detector-to-Control Output Schema:
* **Horizontal error ($e_x = x_c - x_{\text{center}}$):** Governs Yaw rate $\dot{\psi}$ (or lateral velocity $v_y$) to center the target in FOV.
* **Vertical error ($e_y = y_c - y_{\text{center}}$):** Governs Vertical velocity $v_z$ (altitude matching).
* **Bounding box scale ($w$ or $w \times h$):** Proxy for target distance; regulates Forward velocity $v_x$.
* **Temporal State Filter:** Exponential smoothing / low-pass filter on bounding box centroid coordinates to eliminate derivative kick.

---

## 9-Week Master Project Schedule (Phased Capabilities, Milestones, and Mitigations)

| Week | Track A: Perception, Control & Sim | Track B: Avionics, Offboard & Systems | Critical Track Risks & Mitigations |
| :---: | :--- | :--- | :--- |
| **W1** *(Current)* | **Problem Scope & Data Foundations**<br>• [x] Audit and curate target drone detection datasets [M]<br>• [ ] Formalize tracking metrics and data interfaces [L] | **Companion Platform Bring-Up**<br>• [x] Initialize companion compute OS & dev environment [M]<br>• [x] Validate camera pipeline [M]<br>• [ ] Establish telemetry link to flight controller [M] | **[A] Dataset Scale Bias (High):** Distant target drones lack resolution; audit small-box labels & define augmentation needs.<br>**[B] Serial Bus Conflicts (Med):** Port reservation conflicts; isolate primary communication channels before testing. |
| **W2** | **Model Baseline & Target Simulation**<br>• Train baseline tiny-target detection model [M]<br>• Bring up SITL environment & model target drone trajectories [M] | **Edge Inference & Offboard Stack**<br>• Deploy and benchmark hardware-accelerated model on companion board [M]<br>• Establish two-way offboard command dispatch & telemetry ingestion [M] | **[A] Small-Target Recall Gap (Med):** Tiny drone detection drops at range; tune confidence thresholds & feature scales.<br>**[B] Stream Congestion (Med):** High-rate telemetry floods serial buffers; filter down to mission-critical state streams. |
| **W3** | **Simulated Closed-Loop Intercept**<br>• Formulate line-of-sight visual tracking controller in simulation [M]<br>• Evaluate tracking stability against simulated dynamic flight paths [M] | **Benchtop Sensor-to-Control Integration**<br>• Pipe live vision output directly into offboard control loop on bench [M]<br>• Verify manual RC override and failsafe behaviors [H] | **[A] Detection Jitter (Med):** Centroid noise destabilizes velocity setpoints; apply temporal state filtering.<br>**[B] Offboard Timeouts (High):** Vision pipeline bottlenecks drop control packets; decouple vision and control loops asynchronously. |
| **W4** | **Advanced Guidance Policy**<br>• Formulate & evaluate advanced tracking policy in simulation [H]<br>• Benchmark tracking performance against baseline controller [M] | **Airframe Integration & Bench Validation**<br>• Package companion compute, camera, & regulated power onto airframe [M]<br>• Run dynamic bench tracking tests with moving targets across FOV [M] | **[A] Policy Instability (Med):** Advanced policy struggles with edge maneuvers; keep tuned baseline controller as fallback.<br>**[B] System Resource Limits (Med):** Heavy vision & control pipelines compete for resources; profile process loads. |
| **W5** | **Disturbance Testing & Target Loss**<br>• Test controller disturbance rejection against simulated wind gusts [M]<br>• Implement state-machine logic for target re-acquisition & coasting [M] | **Power Integrity & Vibration Checks**<br>• Verify regulated power rails under high motor loads to prevent voltage dips [H]<br>• Inspect camera optical feed under live motor vibration [M] | **[A] Field-of-View Loss (Med):** Sharp maneuvers push target out of view; enforce kinematic coasting states.<br>**[B] Transient Voltage Spikes (High):** Motor load surges risk companion computer resets; verify clean regulated rails & filtering. |
| **W6** | **Sim-to-Real Characterization**<br>• Quantify detector drop-off under outdoor clutter & dynamic backgrounds [M]<br>• Replay real detection tracks through simulation loop [L] | **Safety Failsafes & Captive Testing**<br>• Benchmark end-to-end sensor-to-actuation response time [M]<br>• Perform captive hand-carry tracking checks with active RC failsafes [H] | **[A] Visual Domain Shift (Med):** Outdoor background clutter hurts confidence; fine-tune with flight footage.<br>**[B] Failsafe Delay (High):** Software crash risks losing control; ensure physical RC switch cuts motor outputs instantly. |
| **W7** | **Robustness Limits & Flight Envelopes**<br>• Test tracking limits against simulated frame drops and latency [M]<br>• Define safe velocity, acceleration, and rate limits for flight [L] | **Rig Servoing & Tethered Hover**<br>• Mount airframe on test rig for single-axis servoing and damping [H]<br>• Run tethered hover tests under autonomous offboard control [H] | **[A] Latency Phase Lag (Med):** Pipeline delay causes control overshoot; tune damping and setpoint filtering.<br>**[B] Ground Turbulence (High):** Low-altitude prop wash destabilizes optical tracking; damp vertical control inputs. |
| **W8** | **Telemetry Analysis & Validation**<br>• Align companion logs with autopilot telemetry and compute tracking errors [H]<br>• Validate empirical performance against simulation predictions [M] | **Flight Evaluation Campaign**<br>• Perform autonomous hover & stability checks in offboard mode [M]<br>• Run controlled autonomous tracking passes against target drone [H] | **[A] Log Time Drift (Med):** Clock differences between companion board and flight controller blur log analysis; sync clock references.<br>**[B] Trajectory Drift (High):** Wind gusts destabilize tracking; enforce geofence and prepare for manual RC intervention. |
| **W9** | **Thesis Writing & Wrap-Up**<br>• Write up comparative results, limits, and Sim-to-Real discussion [L]<br>• Finalize and compile thesis manuscript draft [L] | **System Documentation & Archival**<br>• Document system architecture, schematics, and demo modes [L]<br>• Prepare reliable benchtop demo and clean flight media for defense [L] | **[A] Writing Bottlenecks (Low):** Over-polishing plots slows drafting; freeze data figures early to finish text.<br>**[B] Hardware Demonstration Fault (Low):** Last-minute bench wear damages demo setup; lock down physical build once tests finish. |

---

## Immediate Next Actions (Week 1 Closeout)
1. **[Track A] Formalize Tracking Metrics & Data Interfaces:**
   * Write a clean, modular Python interface specifying how bounding box detections `[x_c, y_c, w, h]` are timestamped, filtered, and passed to the controller.
2. **[Track B] Establish Telemetry Link:**
   * Connect Jetson UART to Pixhawk (TELEM2) and run a Python MAVSDK heartbeat/attitude polling script to verify zero serial bus contention.
3. **[Prep W2] Train Baseline YOLOv8n:**
   * Run training on the 3-way Anti-UAV dataset to obtain the baseline detector weights.
