# Classical Kinematic Visual-Servo PID Baseline: System Specification & Locked Parameters

**Document Status:** LOCKED & VALIDATED  
**Applicability:** Autonomous Airborne Drone Pursuit & Visual Servoing  
**Author:** DTU Autonomous Drone Tracking Project  
**Date Locked:** October 26, 2026  

---

## 1. Executive Summary

This document specifies the locked configuration, architecture, and empirical benchmark validation of the **4-DOF Kinematic Visual-Servoing PID Controller**. 

This controller serves as the primary non-learning classical baseline against which all learning-based agents (Behavioral Cloning, Recurrent PPO, and Warm-Started Neural Policies) are rigorously evaluated. It enforces strict physical and operational realism: **zero ground-truth state or distance oracle cheats, full SO(3) attitude decoupling, optical looming velocity feedforward, and tiered fail-safe deceleration on target loss**.

---

## 2. Locked Parameter Manifest

The following parameters are locked in [`control/weights/tuned_pid_gains.json`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/tuned_pid_gains.json):

```json
{
  "status": "LOCKED_VALIDATED_BASELINE",
  "baseline_parameters": {
    "kp_forward": 0.5513988676339123,
    "ki_forward": 0.052667646084163786,
    "kd_forward": 0.3032011569902938,
    "max_decel": 2.450244530033334,
    "centering_damping": 0.31113583820293206,
    "min_centering_factor": 0.6497876245386804,
    "kp_lat": 2.798024066985397,
    "ki_lat": 0.1123530104101074,
    "kd_lat": 0.25266841177910676,
    "kp_z": 3.6434255139643272,
    "ki_z": 0.5301181835351595,
    "kd_z": 0.5481234501400266,
    "kp_yaw": 211.63780915848608,
    "ki_yaw": 17.396459263161336,
    "kd_yaw": 14.182590673527143
  }
}
```

### Physical Scaling & Operational Constraints
* **Nominal Standoff Distance ($d^*$):** $3.5\,\text{m}$ (Engagement capture basket: $[2.5\,\text{m}, 4.5\,\text{m}]$)
* **Desired Bounding Box Width ($w^*$):** $110.0\,\text{px}$ (on $1280 \times 720$ sensor)
* **Maximum Forward Surge Velocity ($v_{x,\max}$):** $12.0\,\text{m/s}$
* **Maximum Lateral Strafe Velocity ($v_{y,\max}$):** $4.0\,\text{m/s}$ (Max lateral acceleration: $6.0\,\text{m/s}^2$)
* **Maximum Heave Rate ($v_{z,\max}$):** $3.0\,\text{m/s}$ ascent, $2.0\,\text{m/s}$ descent
* **Maximum Yaw Rate ($\dot{\psi}_{\max}$):** $120.0^\circ/\text{s}$

---

## 3. Core Architectural Mechanisms

### 3.1 3D Ray Back-Projection (Full SO(3) Attitude Decoupling)
Standard image-plane errors $(e_x, e_y)$ mix drone roll and pitch tilt with true target bearing. The controller back-projects camera optical vectors through the drone's instantaneous rotation matrix:
$$\mathbf{r}_{\text{cam}} = \begin{bmatrix} e_x \tan(\text{HFOV}/2) \\ e_y \tan(\text{VFOV}/2) \\ 1 \end{bmatrix}, \quad \mathbf{r}_{\text{body}} = \mathbf{R}_{\text{cam}}^{\text{body}} \mathbf{r}_{\text{cam}}, \quad \mathbf{r}_{\text{horizon}} = \mathbf{R}_{\text{body}}^{\text{horizon}} \mathbf{r}_{\text{body}}$$
This isolates true horizontal azimuth ($\psi_{\text{err}}$) and elevation without pitch-induced false dive commands.

### 3.2 Continuous Optical Looming Velocity Feedforward
Range error is calculated purely from bounding box scale:
$$e_{\text{range}} = \frac{w^* - w}{w^*}$$
Relative closing velocity $\dot{d}_{\text{rel}}$ is estimated directly from the optical looming derivative:
$$\dot{d}_{\text{rel}} = -\hat{d} \cdot \left(\frac{\dot{w}}{w}\right), \quad \text{where } \hat{d} = d^* \left(\frac{w^*}{w}\right)$$
Target velocity feedforward is synthesized as:
$$\hat{v}_{\text{target}} = v_{\text{drone}} + \dot{d}_{\text{rel}}$$
**Coast Persistence:** Across $30\,\text{Hz}$-to-$50\,\text{Hz}$ inter-frame Kalman predictions (`coast_time < 0.5s`), $\hat{v}_{\text{target}}$ is held smoothly rather than zeroed, eliminating throttle chop.

### 3.3 Tiered Fail-Safe State Governor
* **Tier 1 (Transient Dropouts $\le 300\,\text{ms}$):** Track remains `COASTING`. 8D Kalman filter dead-reckons position and width. Feedforward and integrator maintain cruise trim.
* **Tier 2 (Target Loss $> 300\,\text{ms}$):** Track expires into `SEARCHING`. The controller enters active deceleration (`search_mode="decay"`), bleeding forward speed at $a = -3.0\,\text{m/s}^2$ while sweeping yaw toward the last-known azimuth vector to prevent blind high-speed fly-bys and collisions.

---

## 4. Empirical Benchmark Validation

### 4.1 Comprehensive 300-Trajectory Benchmark Suite
Evaluated under $30\,\text{Hz}$ camera FPS, empirical Jetson Orin YOLO detector noise, $5\%$ random frame drops, and 8D Kalman filter tracking:

| Performance Metric | Tuned PID Baseline | Learned RL Policy (`cloned_expert_v4`) |
| :--- | :---: | :---: |
| **1.0s Catch Rate (%)** | **$63.0\%$** | $3.0\%$ |
| **3.0s Continuous Lock Rate (%)** | **$40.3\%$** | $0.0\%$ |
| **Mean Max Continuous Dwell** | **$2.86\,\text{s}$** | $0.11\,\text{s}$ |
| **Median Max Continuous Dwell** | **$2.30\,\text{s}$** | $0.00\,\text{s}$ |
| **Visual FOV Retention (%)** | **$60.5\%$** | $87.2\%$ |
| **Mean Time to First Lock** | **$5.47\,\text{s}$** | $5.02\,\text{s}$ |
| **Collision Rate (%)** | **$0.33\%$** | $0.0\%$ |

#### Profile Breakdown (100 Scenarios Each)
* **Cruising:** $76.0\%$ 1s Catch | $64.0\%$ 3s Lock | $4.44\,\text{s}$ Mean Dwell | $1.0\%$ Collision
* **Evasive:** $71.0\%$ 1s Catch | $45.0\%$ 3s Lock | $2.98\,\text{s}$ Mean Dwell | $0.0\%$ Collision
* **Hyper-Evasive:** $42.0\%$ 1s Catch | $12.0\%$ 3s Lock | $1.15\,\text{s}$ Mean Dwell | $0.0\%$ Collision

### 4.2 Constant Velocity ($5.0\,\text{m/s}$) Pursuit
* **In-Flight Rendezvous (Nominal $d^* = 3.5\,\text{m}$):**
  * Steady-State Distance: **$4.096\,\text{m}$** (Error: $0.596\,\text{m}$)
  * Steady-State Velocity: **$5.005\,\text{m/s}$** (Target: $5.0\,\text{m/s}$, Error: **$0.005\,\text{m/s}$**)
  * Capture Basket Retention ($[2.5, 4.5]\,\text{m}$): **$85.6\%$**
  * Continuous Dwell: **$10.54\,\text{s}$**

---

## 5. Artifact Reference Map
* **Gains JSON:** [`control/weights/tuned_pid_gains.json`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/tuned_pid_gains.json)
* **Controller Implementation:** [`control/pid_controller.py`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/pid_controller.py)
* **300 Benchmark Comparative Summary:** [`outputs/benchmarks/control/benchmark_300_suite_comparative_summary.json`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/outputs/benchmarks/control/benchmark_300_suite_comparative_summary.json)
* **Constant 5 m/s Evaluation Report:** [`outputs/control/constant_velocity_5ms_metrics.json`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/outputs/control/constant_velocity_5ms_metrics.json)

---

## 6. Derived Neural Policies: Behavioral Cloning Progression

Using this locked classical PID controller as the supervisory expert, recurrent neural network policies were trained via supervised imitation learning across 2,500 trajectories (1,875,000 transitions):

### 6.1 The Perception Domain Gap (`cloned_expert_v5`)
In the initial `cloned_expert_v5` iteration, expert trajectories were generated using idealized $50\,\text{Hz}$ ground-truth camera geometry. When evaluated on the standardized benchmark with realistic $30\,\text{Hz}$ camera sampling, $8\text{D}$ Kalman filtering, empirical noise, and `COASTING` dropouts, the policy suffered catastrophic out-of-distribution drift, hovering outside the basket ($5.90\,\text{m}$) and achieving only a $2.33\%$ catch rate.

### 6.2 Domain-Aligned Perception & Realistic Curriculum (`cloned_expert_v6`)
To eliminate the domain gap, `control/train_imitation.py` was rebuilt with the exact same perception stack used during evaluation:
* **Perception In-the-Loop:** Asynchronous $30\,\text{Hz}$ camera, `SyntheticYOLOSensor` with empirical Jetson noise, $8\text{D}$ Kalman `DroneTracker`, ego-motion compensation, and $5\%$ detection dropouts.
* **Observation Parity:** State observations $s_t \in \mathbb{R}^{15}$ synthesized directly via `RecurrentVisualServoController.build_obs()`.
* **Spawn Geometry Curriculum:** $20\%$ close basket dwell ($2.5\text{--}4.5\,\text{m}$), $20\%$ medium tail chase ($4.5\text{--}8.0\,\text{m}$), $20\%$ long catch-up ($8.0\text{--}18.0\,\text{m}$), $12\%$ crossing right, $12\%$ crossing left, $8\%$ head-on, and $8\%$ oblique break.

---

## 7. Comparative Benchmark: PID Baseline vs. Cloned Policies (300 Scenarios)

Evaluated under $30\,\text{Hz}$ camera FPS, empirical Jetson Orin YOLO noise, $5\%$ frame drops, and 8D Kalman tracking:

| Performance Metric | Tuned PID Baseline | `cloned_expert_v5` (Domain Gap) | `cloned_expert_v6` (Domain-Aligned) |
| :--- | :---: | :---: | :---: |
| **1.0s Catch Rate (%)** | **$63.00\%$** | $2.33\%$ | **$45.00\%$** (+42.67%) |
| **3.0s Continuous Lock Rate (%)** | **$40.33\%$** | $0.00\%$ | **$32.00\%$** (+32.00%) |
| **Mean Max Continuous Dwell** | **$2.86\,\text{s}$** | $0.17\,\text{s}$ | **$2.20\,\text{s}$** (+2.03s) |
| **Visual FOV Retention (%)** | **$60.52\%$** | $40.67\%$ | **$52.13\%$** (+11.46%) |
| **Mean Minimum Distance** | $2.78\,\text{m}$ | $5.90\,\text{m}$ | **$2.79\,\text{m}$** (Basket entry) |
| **Collision Rate (%)** | $0.33\%$ | $0.00\%$ | $2.67\%$ |

### Profile Breakdown (`cloned_expert_v6` vs. PID Baseline)
* **Cruising (100 scenarios):** $64.0\%$ 1s Catch (PID: $76.0\%$) | $55.0\%$ 3s Lock (PID: $64.0\%$) | $3.79\,\text{s}$ Mean Dwell (PID: $4.44\,\text{s}$)
* **Evasive (100 scenarios):** $55.0\%$ 1s Catch (PID: $71.0\%$) | $35.0\%$ 3s Lock (PID: $45.0\%$) | $2.20\,\text{s}$ Mean Dwell (PID: $2.98\,\text{s}$)
* **Hyper-Evasive (100 scenarios):** $16.0\%$ 1s Catch (PID: $42.0\%$) | $6.0\%$ 3s Lock (PID: $12.0\%$) | $0.60\,\text{s}$ Mean Dwell (PID: $1.15\,\text{s}$)
* **Head-On Encounters (60 scenarios):** `cloned_expert_v6` outperformed PID on head-on encounters ($26.7\%$ vs. $21.7\%$ 1s Catch, $27.8\%$ vs. $22.2\%$ FOV retention).

### Artifact References
* **Policy Weights:** [`control/weights/cloned_expert_v6/actor_weights.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/cloned_expert_v6/actor_weights.pt)
* **Model Manifest:** [`control/weights/cloned_expert_v6/model_manifest.json`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/cloned_expert_v6/model_manifest.json)
* **Benchmark Results JSON:** [`outputs/benchmarks/control/benchmark_results_benchmark_300_suite_rl_cloned_expert_v6.json`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/outputs/benchmarks/control/benchmark_results_benchmark_300_suite_rl_cloned_expert_v6.json)

