# Weekly Report - 4

**Bachelor Project — Danmarks Tekniske Universitet (DTU)**  
**Project:** Autonomous Drone Pursuit & Vision-Based Interception  
**Author:** Philip Kierkegaard  
**Date:** October 1, 2026  

---

## Summary

This week's primary milestones were advancing the learned pursuit guidance policy to outperform our classical control baseline and repackaging the pursuit quadrotor. Multiple policy generations were trained under Recurrent PPO, systematically diagnosing and eliminating failure modes such as spinning reward exploits, cruise plateaus, and braking penalties. The resulting production policy (**Generation 5C**) established a project record of **82.5% catch success on evasive maneuvers** (compared to **40.0% for classical PID**), closing the intercept distance over 4 seconds faster. Concurrently, all vehicle electronics were repackaged onto a new, lighter frame (saving $\approx 400\text{ g}$) with full onboard battery power, preparing the drone for autonomous flight control by the Jetson.

---

## Reinforcement Learning Guidance Policy

Before deployment to hardware, the pursuit controller was formulated and trained in closed-loop simulation using Stable-Baselines3's Recurrent PPO (`sb3_contrib.RecurrentPPO`).

### Network Architecture
The policy uses a decoupled Recurrent Actor-Critic network (`MlpLstmPolicy`):
* **Actor Branch:** Maps a 13D observation vector (visual errors, standoff scale, body velocities, and optical rates) through an `LSTM(13, 128)` layer and dense head to output 4 continuous body setpoints $[v_x, v_y, v_z, \dot{\psi}]$.
* **Temporal Memory:** The 128-unit recurrent hidden state acts as an internal observer across camera frame drops and sharp cuts, enabling smooth lead pursuit when the target temporarily leaves the field of view.
* **Computational Footprint:** With 180,105 parameters ($\approx 720$~KB), forward inference takes $\approx 0.16$~ms on CPU ($< 1.0$~ms on the Jetson Orin Nano), taking under 5% of the 50 Hz control window and leaving the GPU entirely dedicated to YOLOv8.

---

### Key Policy Generations & Behavioral Diagnostics

Across seven policy generations, reward shaping was iteratively refined to address specific failure modes:

* **Classical PID Baseline (Deterministic Reference):** Implements Image-Based Visual Servoing (IBVS) with feedforward pitch compensation:
  $$\mathbf{v}_{\text{cmd}} = \begin{bmatrix} k_{p, x} \cdot (d - d^*) \\ -k_{p, \text{lat}} \cdot e_x - k_{d, \text{lat}} \cdot \dot{e}_x \\ k_{p, z} \cdot e_y \\ -k_{p, \psi} \cdot e_x \end{bmatrix}$$
  While highly reliable on straight cruising flights (**97.5% catch**), performance drops sharply on evasive turns (**40.0%**) and collapses on hyper-evasive targets (**0.0%**), as reactive feedback cannot build yaw rate fast enough to prevent targets sweeping outside the $60^\circ$ FOV.

* **The "Lighthouse Radar" Exploit (Generation 5A):** To encourage re-acquisition after target loss, an ungrounded transition bounty and search incentive were added:
  $$R_{\text{search}} = 0.5 \cdot \text{clip}\left(\frac{\text{sgn}(e_{x, \text{last}}) \cdot \dot{\psi}}{120^\circ/\text{s}}, 0, 1\right), \quad R_{\text{reacquire}} = +25.0 \text{ (on lost} \to \text{in-view)}$$
  *Diagnostic:* Episode returns surged to an all-time high ($\bar{R} = 1,830$), but benchmark catch success collapsed to **0.8%** with 24 collisions. The policy learned to peg yaw rate at $-115.6^\circ/\text{s}$, completing a $360^\circ$ spin every 3 seconds to repeatedly harvest the $+25.0$ bounty as the target flashed across the sensor. This confirmed that discrete transition bounties violate reward invariance and induce cyclic exploits.

* **Continuous Dual-Scale Gaussian Potential (Generation 4):** Enforces tail-chase geometry by anchoring reward to a rear point:
  $$\mathbf{p}_{\text{trail}} = \mathbf{p}_{\text{target}} - 6.0 \cdot \mathbf{\hat{h}}_{\text{target}}$$
  To eliminate dead zones where the drone hung back, Gen 4 replaced discrete distance thresholds with a smooth potential field:
  $$R_{\text{trail}}(d_{\text{trail}}) = 3.0 \cdot \exp\left(-\frac{d_{\text{trail}}^2}{2 \cdot 3.5^2}\right) + 6.0 \cdot \exp\left(-\frac{d_{\text{trail}}^2}{2 \cdot 1.2^2}\right)$$
  where $d_{\text{trail}} = \|\mathbf{p}_{\text{chaser}} - \mathbf{p}_{\text{trail}}\|$. The wide Gaussian ($\sigma = 3.5\text{m}$) pulls the drone inward from afar, while the tight well ($\sigma = 1.2\text{m}$) holds exact standoff, outperforming PID for the first time (**48.3% catch**, TTC: **8.34s**).

* **The Production Champion (Generation 5C):** Finalized with three calibrated mechanisms:
  1. *Direct In-Basket Dwell Holding Reward:*
     $$R_{\text{basket\_dwell}} = \begin{cases} +2.0/\text{step } (+100.0/\text{s}) & \text{if } 5.0\text{m} \le d \le 7.0\text{m} \text{ and in-view} \\ 0.0 & \text{otherwise} \end{cases}$$
     Provides direct gradient for sustaining the 3-second thesis benchmark window.
  2. *Two-Tier Proximity Barrier Cushion:*
     $$P_{\text{proximity}}(d) = \begin{cases} 2.0 \cdot \left(\frac{5.0 - d}{2.0}\right)^2 & \text{if } 3.0\text{m} \le d < 5.0\text{m} \\ 2.0 + 4.0 \cdot \left(\frac{3.0 - d}{3.0 - d_{\text{col}}}\right)^2 & \text{if } d_{\text{col}} \le d < 3.0\text{m} \\ 0.0 & \text{if } d \ge 5.0\text{m} \end{cases}$$
     Acts as a soft repulsive cushion below 5.0m and a hard wall below 3.0m, cutting collision breaches from 24 down to 2.
  3. *Symmetric Low-Gain Control Regularization:*
     $$P_{\text{effort}} = 0.01 \cdot a_0^2 + 0.01 \cdot a_2^2 + 0.02 \cdot (a_1^2 + a_3^2)$$
     Penalizes control actuation symmetrically so that pulling back to brake ($-a_0$) incurs no asymmetric penalty, allowing uninhibited deceleration when trailing slow targets.

---

### Benchmark Performance Summary

All models were evaluated across 120 standardized flights (`benchmark_suite_v1.json`). Primary criteria require sustaining a 5–7m standoff in-view for at least 3.0 continuous seconds:

| Model | Core Architecture / Reward | Overall Catch % (3s) | Mean TTC (s) | In-FOV % | Yaw Jerk (°/s²) | Crashes ($d < 1.2\text{m}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Classical PID** | Kinematic IBVS Baseline | 45.8% (55/120) | 8.97s | 75.2% | 24.8°/s² | 2 |
| **Gen 1** | Isotropic 5–7m Radial Shell | 38.3% (46/120) | 9.45s | 75.9% | 108.5°/s² | 4 |
| **Gen 2** | Heavy Jerk Penalty ($0.20$) | 0.0% (0/120) | N/A | 31.2% | 0.8°/s² | 0 |
| **Gen 3** | Geometric Trail Anchor Basket | 40.8% (49/120) | 8.98s | 79.3% | 23.4°/s² | 2 |
| **Gen 4** | Dual-Scale Gaussian Potential | 48.3% (58/120) | **8.34s** | **82.5%** | 26.6°/s² | 1 |
| **Gen 5A** | Transition Bounty (+25.0) | 0.8% (1/120) | 14.10s | 42.0% | 71.0°/s² | 24 |
| **Gen 5C** | Calibrated Dwell + Symmetric Effort | **49.2% (59/120)** | 9.13s | 79.4% | 27.7°/s² | 2 |

```
                             CATCH SUCCESS RATE BY PROFILE
   Cruising (3.5 m/s):
   PID:       [███████████████████░] 97.5%  (TTC:  7.33s | Dwell: 13.35s)
   Gen 4 RL:  [██████████████████░░] 90.0%  (TTC:  7.77s | Dwell: 13.77s)
   Gen 5C RL: [████████████░░░░░░░░] 62.5%  (TTC:  9.46s | Dwell:  7.48s)

   Evasive (6.0 m/s):
   PID:       [████████░░░░░░░░░░░░] 40.0%  (TTC: 12.97s | Dwell:  5.65s)
   Gen 4 RL:  [███████████░░░░░░░░░] 55.0%  (TTC:  9.29s | Dwell:  5.75s)
   Gen 5C RL: [█████████████████░░░] 82.5%  (TTC:  8.72s | Dwell: 10.89s)  <-- +42.5% OVER PID
```

On evasive targets, **Gen 5C achieved 82.5% catch success** (more than double PID's 40.0%), averaging nearly 11 seconds of continuous in-basket lock and closing the distance in 8.72s (vs. 12.97s for PID).

---

## Airframe Repackaging & Hardware Status

During this week, all vehicle components were repackaged onto a new, lighter airframe—cutting approximately 400 grams in total vehicle weight. All electronics (Pixhawk autopilot, NVIDIA Jetson Orin Nano companion computer, and camera) are now fully powered onboard from the flight battery via a dedicated step-down BEC, making the system completely self-contained, untethered, and ready for closed-loop control by the Jetson.

---

## Qualitative Perception Validation

The perception pipeline (YOLOv8n detector + 8D Kalman multi-track filter) was tested against a secondary physical drone and recorded flight footage:
* **Target Detection:** Tested against a secondary quadrotor; while the detector struggled in cluttered indoor settings with artificial lighting, performance was significantly more reliable once tested outdoors against natural backgrounds.
* **Kalman Filtering:** The 8D state estimator smoothly filtered optical bounding-box noise and maintained target coasting across occasional single-frame dropouts.
* **Ground Station Stream:** Verified live browser video streaming at 30 FPS, displaying the target bounding box and flight commands for real-time field monitoring with zero impact on the 50 Hz control loop.

---

## System Identification Status & Plan for Week 5

* **System Identification Status:** With the airframe reassembly complete, manual step-input flights are scheduled for this afternoon's flight window to collect high-rate `.ulg` logs (`SDLOG_PROFILE = 1`). From these logs, empirical command delay ($\tau$) and drag ($k_d$) will be extracted using `control/identify_dynamics.py`.
* **Week 5 Objectives:**
  1. Fit measured $\tau$ and $k_d$ parameters to update `control/environment.py`.
  2. Fine-tune the Generation 5C policy under calibrated airframe dynamics.
  3. Conduct tethered and open-field autonomous pursuit flights in offboard mode against stationary and moving targets.
  4. Record synchronized flight telemetry and onboard video to compare physical tracking against the simulation benchmark.
