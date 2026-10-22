# Benchmark Protocol Specification: Autonomous Drone Pursuit & Interception
**DTU Bachelor Thesis — Autonomous Drone Tracking & Interception System**
**Document Version:** 1.0 (Task A Deliverable)
**Author:** Pair Programming Agent & User
**Status:** Approved Specification

---

## 1. Executive Summary & Objective

The objective of this specification is to establish a rigorous, fair, and reproducible evaluation protocol for comparing:
1. **Conventional Kinematic Visual-Servoing (PID)**: 4-DOF velocity setpoint guidance with $SO(3)$ horizon attitude decoupling and lateral strafe.
2. **Learned Recurrent PPO (RL)**: 13D observation space with recurrent LSTM hidden state, optical rate filtering, and direct 4D continuous body velocity control.

To ensure that observed performance differences reflect **guidance and control capability** rather than unequal experimental conditions or information asymmetry, both controllers are evaluated under identical dynamics, optics, perception noise, camera frame rates, initial states, target trajectories, and termination criteria.

---

## 2. Identified Inconsistencies & Resolutions

| Evaluation Dimension | Legacy Inconsistency | Corrected Benchmark Contract |
| :--- | :--- | :--- |
| **Range Perception** | PID controller accessed ground-truth 3D range (`distance = true_telem["distance"]`), while RL relied on 2D bbox scale (`w_norm`). | **Strict Monocular Equivalence**: PID must use `use_bbox_size=True` with `scale_mode="width"`. Range error is derived purely from bounding-box width ($w_{\text{nominal}} = 0.0859$, $110.0\,\text{px}$ on $1280 \times 720$) for both controllers. Neither receives ground truth range. |
| **Camera Frame Rate** | Benchmark ran perception at 50 Hz sim tick, while deployment and RL training used 30 Hz camera ticks with inter-frame zero-order hold. | **Unified 30 Hz Camera Timing**: Camera measurements arrive at 30 Hz (33.3 ms intervals), held over 50 Hz physics sub-steps. |
| **Target Standoff Objective** | Legacy benchmark checked 5.0m–7.0m standoff (35px), while calibrated deployment target is 3.5m standoff (110.0px). | **Calibrated 3.5m Standoff**: Target standoff is strictly 3.5m ($w_{\text{bbox}} = 110.0\,\text{px}$, $w_{\text{norm}} = 0.0859$), with engagement basket $[2.5\,\text{m}, 4.5\,\text{m}]$ and 3.0s continuous lock requirement. |
| **Kinodynamic Limits** | Benchmark used unconstrained $18\,\text{m/s}$ limits in simulator, while physical flight software caps speed at $6.0\,\text{m/s}$. | **Physical Flight Envelope**: Simulator limits locked to $v_{\max} = 6.0\,\text{m/s}$, $v_{\text{lat}} \le 2.5\,\text{m/s}$, $a \le 4.0\,\text{m/s}^2$. |
| **Perception Processing** | Ad-hoc Gaussian noise without track loss handling. | **Standardized 8D Kalman Filter**: Identical `KalmanBoxTracker` state estimation with 15-frame coasting timeout across both controllers. |

---

## 3. Unified Experimental Environment

### 3.1 Vehicle Dynamics (`FastPixhawkQuadSim`)
* **Physics Integration Rate**: 50 Hz ($\Delta t = 0.02\,\text{s}$).
* **Aerodynamic Drag**: Quadratic drag coefficient $C_d = 0.08\,\text{m}^{-1}$.
* **Forward Velocity Limits**: $[-0.5\,\text{m/s}, +6.0\,\text{m/s}]$.
* **Lateral Velocity Limits**: $[-2.5\,\text{m/s}, +2.5\,\text{m/s}]$.
* **Vertical Velocity Limits**: $[-2.5\,\text{m/s} \text{ (climb)}, +1.5\,\text{m/s} \text{ (descend)}]$.
* **Yaw Rate Limit**: $\pm 90.0^\circ/\text{s}$.
* **Translational Acceleration Limits**: Horizontal $\le 4.0\,\text{m/s}^2$, Vertical $\le 3.0\,\text{m/s}^2$.
* **Yaw Angular Acceleration Limit**: $\le 180.0^\circ/\text{s}^2$.

### 3.2 Sensor & Optics Geometry
* **Sensor Resolution**: 1280 x 720 pixels (16:9 widescreen).
* **Horizontal Field of View (HFOV)**: 62.0° (half-HFOV: tan(31.0°) = 0.60086, focal length fx = 1065.1 px).
* **Vertical Field of View (VFOV)**: 38.0° (half-VFOV: tan(19.0°) = 0.34433, focal length fy = 1045.5 px).
* **Standoff Target Bounding Box**: 110.0 x 61.7 pixels at 3.5 m nominal distance ($w_{\text{nominal}} = 110 / 1280 \approx 0.0859$).
* **Camera Mount Uptilt**: +15.0° fixed mechanical pitch-up.
* **Sampling Rate**: 30 Hz (camera dt = 33.3 ms).
* **Sensor Measurement Noise**: AR(1) colored noise matching empirical on-board Jetson detector residuals.
* **Frame Drop Simulation**: Nominal 5%, with stress sweeps at 0%, 10%, and 20%.

### 3.3 Apparent Target Scale Calibration & Controller Consistency
To eliminate systematic evaluation bias between the conventional visual-servoing PID controller and the learned recurrent PPO controller, apparent target scale is standardized:
* **Resolution-Invariant Bounding-Box Width Scaling**:
  $$w_{\text{norm}} = \frac{w_{\text{bbox}}}{W_{\text{image}}}$$
  At $1280 \times 720$ resolution, a reference width of $110.0\,\text{px}$ corresponds to $w_{\text{nominal}} = 110 / 1280 \approx 0.0859375$ (preserving the original relative target scale of $0.0866$ under the updated $62.0^\circ$ HFOV).
* **PID vs. PPO Scale Formulation Unification**:
  * Previously, PID regulated $s_{\text{PID}} = \max(w_{\text{bbox}}, h_{\text{bbox}})$, while PPO regulated normalized width $s_{\text{PPO}} = w_{\text{bbox}} / W_{\text{image}}$.
  * Under bank turns or pitch maneuvers where $h_{\text{bbox}} > w_{\text{bbox}}$, this created an artificial standoff discrepancy between controllers.
  * Both controllers now strictly regulate normalized horizontal bounding-box width (`scale_mode = "width"`), ensuring mathematical equivalence:
    $$\text{range\_err}_{\text{PID}} = \frac{110.0 - w_{\text{bbox}}}{110.0} \equiv \frac{w_{\text{nominal}} - w_{\text{norm}}}{w_{\text{nominal}}} = \text{scale\_err}_{\text{PPO}}$$

---

## 4. Dataset Partitioning & Anti-Leakage Architecture

To guarantee unbiased scientific evaluation, scenarios are strictly segregated into three non-overlapping tiers:

```
+-------------------------------------------------------------------------------+
|  TIER 1: PROCEDURAL TRAINING SET                                              |
|  - Dynamic randomized sampling on reset (seeds < 10,000)                      |
|  - 50% Close / 50% Sprint spawn distributions                                 |
|  - Domain randomization: Jitter, lighting, and frame drops enabled           |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|  TIER 2: VALIDATION SUITE (validation_suite_30.json)                          |
|  - 30 fixed deterministic scenarios (seeds 10,000 - 10,029)                   |
|  - Used solely for checkpoint evaluation and hyperparameter selection         |
|  - Balanced across Cruising, Evasive, and Hyper-Evasive profiles              |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|  TIER 3: FROZEN HELD-OUT TEST SUITE (heldout_test_suite_50.json)              |
|  - 50 isolated scenarios (seeds 50,000 - 50,049)                              |
|  - NEVER exposed during training, reward shaping, or hyperparameter tuning   |
|  - Official benchmark evaluation dataset for final thesis results             |
+-------------------------------------------------------------------------------+
```

### Held-Out Test Suite Profile Distribution
* **Cruising** (15 scenarios): Straight-line and smooth turning formation tracking ($v_{\text{target}} \approx 4.0\,\text{m/s}$).
* **Evasive** (23 scenarios): Pseudo-random sinusoidal and S-turn evasive maneuvers ($a_{\text{target}} \le 2.5\,\text{m/s}^2$).
* **Hyper-Evasive** (12 scenarios): High-frequency jinking and sharp 3D switchbacks ($a_{\text{target}} \le 4.5\,\text{m/s}^2$).
* **Encounter Geometries**: Tail-chase ($0^\circ$), crossing left/right ($\pm 90^\circ$), head-on ($180^\circ$), and oblique angles ($\pm 45^\circ, \pm 135^\circ$).

---

## 5. Unified Benchmark Evaluation Metrics

Metrics are computed deterministically from physical state trajectories and perception outputs, completely independent of RL reward functions:

### 5.1 Primary Interception Performance
1. **Interception Success Rate ($\%$)**:
   $$\text{Success} = \mathbb{I}\left(t_{\text{continuous\_dwell}} \ge 3.0\,\text{s}\right) \times 100\%$$
   The flight is deemed successful if and only if the chaser drone continuously maintains the target inside the $[2.5\,\text{m}, 4.5\,\text{m}]$ basket with active visual lock for $\ge 3.0\,\text{seconds}$.
2. **Time to Interception ($t_{\text{int}}$, seconds)**:
   The elapsed flight time from mission start until the 3.0-second continuous dwell lock is satisfied. If the mission times out without lock, $t_{\text{int}} = \text{TIMEOUT}$ ($25.0\,\text{s}$).

### 5.2 Tracking Accuracy & Weapon Envelope
3. **Standoff Error (MAE, meters)**:
   $$\text{MAE}_{\text{standoff}} = \frac{1}{N_{\text{lock}}} \sum_{t \in \text{lock}} |d_t - 3.5\,\text{m}|$$
   Mean absolute distance error relative to the $3.5\,\text{m}$ nominal standoff during engaged tracking.
4. **Visual FOV Retention Rate ($\%$)**:
   $$\text{FOV}_{\text{retention}} = \frac{1}{T} \int_0^T \mathbb{I}\left(\text{target in FOV}\right) dt \times 100\%$$
5. **Firing Window ($\%$)**:
   $$\text{Firing}_{\text{pct}} = \frac{1}{T} \int_0^T \mathbb{I}\left(2.5\,\text{m} \le d_t \le 4.5\,\text{m} \land \text{in FOV}\right) dt \times 100\%$$

### 5.3 Flight Safety & Survivability
6. **Collision Breach Rate ($\%$)**: Percentage of runs where minimum distance $d_{\min} < 1.2\,\text{m}$ (physical collision radius). Must be $0.0\%$ for aerospace certification.
7. **Ground Strike Rate ($\%$)**: Percentage of runs where altitude falls below $1.0\,\text{m}$ AGL. Must be $0.0\%$.
8. **Lost Target Timeout Rate ($\%$)**: Percentage of runs where the target is lost from camera FOV for $> 3.5\,\text{seconds}$.

### 5.4 Control Smoothness & Actuation Effort
9. **Cumulative Control Effort**:
   $$E_{\text{ctrl}} = \int_0^T \|\mathbf{u}_t\|^2 dt$$
10. **Actuator Slew / Jerk**:
    $$J_{\text{act}} = \frac{1}{N} \sum_{k=1}^N \|\mathbf{u}_k - \mathbf{u}_{k-1}\|^2$$

---

## 6. Execution Command Reference

To execute the standardized benchmark suite against the frozen held-out test dataset with calibrated empirical YOLO noise:

```bash
# 1. Monocular Kinematic Visual Servoing (PID Baseline) with Empirical YOLO Noise
python benchmarks/control/benchmark_controller.py \
  --dataset outputs/benchmarks/datasets/heldout_test_suite_50.json \
  --controller pid \
  --noise-mode empirical \
  --drop-rate 0.05

# 2. Learned Recurrent PPO (RL Policy) with Empirical YOLO Noise
python benchmarks/control/benchmark_controller.py \
  --dataset outputs/benchmarks/datasets/heldout_test_suite_50.json \
  --controller rl \
  --model-path control/weights/recurrent_ppo_v5_clean_accum/best_model/best_model.zip \
  --noise-mode empirical \
  --drop-rate 0.05

# 3. Automated Head-to-Head Comparative Run
python benchmarks/control/benchmark_controller.py \
  --dataset outputs/benchmarks/datasets/heldout_test_suite_50.json \
  --controller both \
  --model-path control/weights/recurrent_ppo_v5_clean_accum/best_model/best_model.zip \
  --noise-mode empirical \
  --drop-rate 0.05
```
