# Head-to-Head Empirical Benchmark: Classical PID vs. Tail-Chase Recurrent PPO

**Project**: Autonomous Drone Pursuit & Interception System  
**Author**: Philip Kierkegaard — DTU Bachelor Thesis  
**Date**: September 30, 2026  
**Test Suite**: [`outputs/benchmarks/datasets/benchmark_suite_v1.json`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/outputs/benchmarks/datasets/benchmark_suite_v1.json) (120 Standardized Scenarios)  
**Physics**: [`FastPixhawkQuadSim`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/simulation.py) (50 Hz, 60ms delay, +15° camera up-tilt, 25.0s horizon)  
**Target Standoff**: **Exactly 6.0 meters** for both controllers  

---

## 1. Executive Summary

We evaluated both controllers on the exact same **120 standardized test encounters** spanning all combinations of:
- **Flight Profiles**: `cruising`, `evasive`, `hyper_evasive` (40 scenarios each)
- **Encounter Geometries**: `tail_chase`, `crossing_right`, `crossing_left`, `head_on` (30 scenarios each)

### Core Takeaway:
Your intuition was **100% correct**: the Classical PID controller is an exceptionally strong, mature baseline. Rather than one controller dominating the other across the board, the benchmark reveals a classic aerospace trade-off:

1. **Classical PID** delivers **tighter standoff precision on cruising targets** (Standoff RMSE $2.84\text{m}$ vs $5.80\text{m}$), leveraging its analytical kinematic glide-slope ($v_x = \sqrt{2 a_{\text{decel}} \Delta x}$).
2. **Tail-Chase Recurrent PPO (RL)** delivers **superior visual line-of-sight retention on aggressive targets** ($84.3\%$ in-FOV vs PID's $63.4\%$ on Hyper-Evasive).
3. **Flight Smoothness Parity**: Following our jerk regularizer, RL yaw jerk dropped from the unregularized $108.5^\circ/\text{s}^2$ down to **$23.4^\circ/\text{s}^2$**, matching PID's **$23.9^\circ/\text{s}^2$**.

---

## 2. Overall Head-to-Head Benchmark Table (120 Scenarios)

| Performance Metric | Classical PID (6.0m Target) | Tail-Chase RL (6.0m Target) | Difference / Winner |
| :--- | :---: | :---: | :---: |
| **Interception Success Rate (3s lock)** | **$45.8\%$** (55/120) | $40.8\%$ (49/120) | **PID by $+5.0\%$** |
| **In-FOV Tracking Rate (%)** | $75.9\%$ | **$79.3\%$** | **RL holds FOV $+3.4\%$ longer** |
| **Basket Firing Window Duration (s)** | **$6.50\text{s}$** ($26.0\%$) | $5.39\text{s}$ ($21.6\%$) | PID holds basket longer |
| **Standoff RMSE (vs 6.0m target)** | **$3.75\text{m}$** | $8.52\text{m}$ | **PID regulates distance tighter** |
| **Mean Standoff Distance (m)** | $8.78\text{m}$ | $12.70\text{m}$ | PID tracks closer to 6.0m |
| **Mean Minimum Distance (Safety Margin)** | $5.43\text{m}$ | **$5.83\text{m}$** | Comparable safe standoff |
| **Collision Breaches ($d < 1.2\text{m}$)** | $1.7\%$ (2/120) | $1.7\%$ (2/120) | **Identical safety record** |
| **Surge Jerk ($dV_x/dt$)** | **$1.30\text{ m/s}^2$** | $1.64\text{ m/s}^2$ | PID slightly smoother surge |
| **Vertical Jerk ($dV_z/dt$)** | $0.66\text{ m/s}^2$ | **$0.35\text{ m/s}^2$** | **RL is $1.9\times$ smoother in heave** |
| **Yaw Jerk ($d\text{Yaw}/dt$)** | $23.9^\circ/\text{s}^2$ | **$23.4^\circ/\text{s}^2$** | **Virtual tie (Parity achieved!)** |

---

## 3. Breakdown by Target Flight Profile

### A. Cruising Profile (40 Scenarios)
*Linear / curved target flight at steady $\sim 3.5\text{ m/s}$*

| Metric | Classical PID | Tail-Chase RL | Advantage |
| :--- | :---: | :---: | :---: |
| **Success Rate (3s lock dwell)** | **$95.0\%$** (38/40) | $80.0\%$ (32/40) | PID leads on clean lines |
| **In-FOV Tracking Rate** | **$83.6\%$** | $75.3\%$ | PID |
| **Basket Firing Window** | **$13.23\text{s}$** | $11.65\text{s}$ | PID $+1.58\text{s}$ |
| **Standoff RMSE (vs 6.0m)** | **$2.84\text{m}$** | $5.80\text{m}$ | PID tighter standoff |
| **Yaw Jerk** | **$16.2^\circ/\text{s}^2$** | $18.3^\circ/\text{s}^2$ | Both smooth |

---

### B. Evasive Profile (40 Scenarios)
*S-turns, lateral jinking, speed modulations up to $7\text{ m/s}$*

| Metric | Classical PID | Tail-Chase RL | Advantage |
| :--- | :---: | :---: | :---: |
| **Success Rate (3s lock dwell)** | **$42.5\%$** (17/40) | **$42.5\%$** (17/40) | **EXACT TIE** |
| **In-FOV Tracking Rate** | **$80.8\%$** | $78.3\%$ | PID by $+2.5\%$ |
| **Basket Firing Window** | **$5.99\text{s}$** | $4.36\text{s}$ | PID $+1.63\text{s}$ |
| **Standoff RMSE (vs 6.0m)** | **$2.67\text{m}$** | $6.15\text{m}$ | PID tighter standoff |
| **Minimum Distance Margin** | $5.46\text{m}$ | $5.11\text{m}$ | Identical safety |
| **Yaw Jerk** | **$18.3^\circ/\text{s}^2$** | $21.9^\circ/\text{s}^2$ | Both smooth |

---

### C. Hyper-Evasive Profile (40 Scenarios)
*Extreme 3D evasive maneuvers, high angular rates, speeds up to $11\text{ m/s}$*

| Metric | Classical PID | Tail-Chase RL | Advantage |
| :--- | :---: | :---: | :---: |
| **Success Rate (3s lock dwell)** | $0.0\%$ (0/40) | $0.0\%$ (0/40) | Hard limit for 3s lock dwell |
| **In-FOV Tracking Rate** | $63.4\%$ | **$84.3\%$** | **RL leads by $+20.9\%$!** |
| **Yaw Jerk** | $37.3^\circ/\text{s}^2$ | **$30.2^\circ/\text{s}^2$** | **RL is $19\%$ smoother** |
| **Mean Standoff Distance** | $10.76\text{m}$ | $17.98\text{m}$ | RL backs off to keep FOV |

> [!NOTE]
> **Hyper-Evasive Insight**: In hyper-evasive maneuvers, PID attempts to force the drone to 6m, cuts inside too sharply, and loses the target out of the camera's $60^\circ$ viewport ($36.6\%$ lost time). The RL policy's recurrent LSTM predicts the high-speed arc, widens its pursuit radius ($17.98\text{m}$), and maintains camera line-of-sight $84.3\%$ of the time!

---

## 4. Breakdown by Encounter Geometry

| Encounter Geometry | Scenarios | PID Success% | RL Success% | PID In-FOV% | RL In-FOV% | PID Standoff RMSE | RL Standoff RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tail-Chase** | 30 | **$40.0\%$** | **$40.0\%$** | $78.6\%$ | **$87.3\%$** | **$4.21\text{m}$** | $6.91\text{m}$ |
| **Crossing Right** | 30 | **$56.7\%$** | **$56.7\%$** | $78.5\%$ | **$82.4\%$** | **$3.11\text{m}$** | $6.57\text{m}$ |
| **Crossing Left** | 30 | **$46.7\%$** | $36.7\%$ | $78.8\%$ | $78.5\%$ | **$3.50\text{m}$** | $8.17\text{m}$ |
| **Head-On** | 30 | **$40.0\%$** | $30.0\%$ | $67.8\%$ | **$69.1\%$** | **$4.18\text{m}$** | $12.42\text{m}$ |

---

## 5. Academic Defense Takeaways for Your Thesis

1. **Why PID Wins on Distance Precision**:
   - The kinematic visual servo controller computes an analytical deceleration curve $v_x = \sqrt{2 a_{\text{decel}} \Delta x}$. When error drops to zero, the commanded velocity is calculated exactly.
   - For an agent trained purely via trial-and-error reward gradients, matching an analytical square-root curve within a few centimeters requires substantial gradient precision.
2. **Where RL Wins on Temporal Reasoning**:
   - In crossing and hyper-evasive flights, PID suffers from reactive lag. RL's LSTM latent state carries temporal momentum, maintaining line-of-sight ($84.3\%$ vs $63.4\%$) where PID drops the target.
3. **The Power of Slew-Rate Regularization**:
   - Unregularized RL yielded an intolerable $108.5^\circ/\text{s}^2$ yaw chatter.
   - Tail-Chase RL with slew-rate limiting and jerk penalty achieved **$23.4^\circ/\text{s}^2$**, perfectly matching PID's **$23.9^\circ/\text{s}^2$**.
