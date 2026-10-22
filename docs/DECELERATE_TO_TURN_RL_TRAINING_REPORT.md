# Decelerate-to-Turn & Cornering Guidance: 6-Model RL Optimization Sweep

**Project:** Autonomous Drone Pursuit & Interception System  
**Institution:** Technical University of Denmark (DTU)  
**Date:** October 2026  
**Artifact Scope:** Control Pipeline (`control/environment.py`, `control/train_rl.py`, `control/run_decel_turn_sweep.py`)  
**Base Weights:** All models fine-tuned from cloned expert weights ([`control/weights/cloned_expert_v1/actor_weights.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/cloned_expert_v1/actor_weights.pt))  

---

## 1. Executive Summary & Problem Formulation

In high-speed air-to-air drone pursuit, the pursuer faces an aerodynamic and kinematic tradeoff between **straightaway sprint capability** and **cornering turn radius**.

### 1.1 The $v^2$ Turn Radius Dilemma
Centripetal acceleration limits the minimum turn radius of a quadrotor with maximum horizontal acceleration $a_{\text{max}} = 6.0\text{ m/s}^2$:
$$R_{\text{min}} = \frac{v^2}{a_{\text{max}}}$$

- **At Cruising / Cornering Speed ($v = 3.5\text{ m/s}$):** $R_{\text{min}} = \frac{12.25}{6.0} = \mathbf{2.04\text{ m}}$ (tight apex cutting, tracks sudden multi-axis evasive maneuvers).
- **At Full Sprint Speed ($v = 12.0\text{ m/s}$):** $R_{\text{min}} = \frac{144.0}{6.0} = \mathbf{24.0\text{ m}}$ (drifts wide like a car on ice, blowing target completely out of the $62^\circ$ camera cone).

### 1.2 Why the Previous Sweep Failed
The previous 10-candidate overnight sweep attempted to force close following by rewarding forward closing speed and optical expansion unconditionally. The neural network responded by pegging the throttle at $11.3\text{--}12.0\text{ m/s}$ continuously. Whenever the hyper-evasive target pulled a lateral cut, the drone's 24m turn radius caused it to overshoot, blow past the target, and drift $40\text{--}100\text{ m}$ away.

### 1.3 The Goal of This Sweep
To enable the drone to **sprint fast (7–12 m/s) on straightaways to catch accelerating targets, but dynamically and aggressively decelerate ($a_{\text{decel}} = 6.0\text{ m/s}^2$) into corners (down to 2–3 m/s) to negotiate tight turns, before punching the throttle back out of the apex.**

---

## 2. Six Diverse Mathematical Approaches

All 6 candidates were trained for **200,000 timesteps** with Recurrent PPO (LSTM hidden size 128, learning rate $5 \times 10^{-5}$) starting from identical cloned expert weights:

### Run 1: `trail_brake_corner_governor`
- **Concept:** Physics Centripetal Governor + Active Trail-Braking Incentive.
- **Formulation:** 
  - Gated closing progress: When $|e_x| \ge 0.20$, negative closing speed is **never penalized** (decelerating into a corner carries zero penalty).
  - Skid penalty: $p_{\text{skid}} = -3.0 \cdot (v_{\text{fwd}} / 12.0) \cdot \max(0, |e_x| - 0.15)$.
  - Explicit trail-brake bonus: Grants $+2.5 \cdot \max(0, -a_{\text{cmd, fwd}})$ if the network commands negative forward acceleration while $|e_x| > 0.25$ and $v_{\text{fwd}} > 4.5\text{ m/s}$.

### Run 2: `proportional_nav_apex_cutter` (Champion)
- **Concept:** Proportional Navigation Velocity Vector Alignment (Apex Cutting).
- **Formulation:**
  - Rewards alignment between pursuer 2D velocity $\vec{v}_c$ and target velocity $\vec{v}_t$:
    $$r_{\text{pn}} = 4.0 \cdot \max\left(0, \frac{\vec{v}_c \cdot \vec{v}_t}{\|\vec{v}_c\| \|\vec{v}_t\|}\right)$$
  - Misalignment drag: If $\cos \theta_v < 0.70$ (vectors misaligned during target turn), high speed is penalized: $-2.0 \cdot (v_c / 12.0) \cdot (0.70 - \cos \theta_v)$.
  - Velocity vector rotation rate is $\dot{\theta}_v = a_{\text{lat}} / v$. At high speed, the velocity vector turns sluggishly; at low speed, it rotates rapidly. This naturally compels the network to brake in order to swing its velocity vector and cut the apex!

### Run 3: `retinal_slip_kinetic_barrier`
- **Concept:** Biological Raptor Attack Model (Optical Retinal Slip Governor).
- **Formulation:**
  - Measures total retinal slip rate $s = \sqrt{\dot{e}_x^2 + \dot{e}_y^2}$.
  - When $s < 0.20\text{ s}^{-1}$ (target stable on optical axis): Full sprint progress rewarded ($+4.0 \cdot \dot{w}$).
  - When $s \ge 0.30\text{ s}^{-1}$ (target shearing across frame): Imposes a quadratic kinetic energy penalty $\mathcal{P} = -3.5 \cdot (v_{\text{fwd}}/12.0)^2 \cdot (s / 0.5)$.

### Run 4: `curvature_adaptive_energy_matching`
- **Concept:** 3D Relative Velocity Nulling with Dynamic Standoff Expansion.
- **Formulation:**
  - Standoff buffer expands from $3.5\text{ m} \to 4.5\text{ m}$ during high optical rate turns to create turn radius clearance.
  - As distance closes ($d < 8\text{ m}$), directly rewards nulling the 3D relative velocity vector:
    $$r_{\text{vel\_match}} = 3.0 \cdot \frac{\exp\left(-0.5 (\|\vec{v}_c - \vec{v}_t\| / 2.0)^2\right)}{1.0 + |d - d_{\text{standoff}}|}$$

### Run 5: `staged_curriculum_trail_brake`
- **Concept:** Analytical Cornering Speed Envelope Matching with Balanced Curriculum.
- **Formulation:**
  - Computes the physically optimal cornering speed:
    $$v^*(e_x, \dot{e}_x) = 4.0 + 8.0 \cdot \exp\left(-0.5 \left(\frac{e_x}{0.25}\right)^2 - 0.5 \left(\frac{\dot{e}_x}{0.40}\right)^2\right)$$
  - Rewards matching $v^*$ ($12.0\text{ m/s}$ on straights, $4.0\text{ m/s}$ in hairpins).
  - Trained on balanced curriculum: 35% hyper-evasive, 45% evasive, 20% cruising.

### Run 6: `unified_apex_dogfight_v2` (Runner-Up)
- **Concept:** Unified Dogfight Synthesis (Huber + PN Lead + Shearing Brake + Quadratic Dwell Jackpot).
- **Formulation:**
  - Huber potential field ($d_{\text{standoff}} = 3.5\text{ m}$).
  - Lead vector alignment ($+3.0 \cdot \vec{v}_c \cdot \vec{v}_t$).
  - Gated sprint vs. active trail-brake bonus ($+2.0 \cdot \max(0, -a_0)$ on shearing).
  - Quadratic dwell jackpot ($+18.0 \cdot \text{dwell}^2$).

---

## 3. Comprehensive Experimental Leaderboard

All policies were evaluated against the standardized validation suite across both **7 Hyper-Evasive** (`val_24` to `val_30`) and **8 Evasive** (`val_16` to `val_23`) scenarios:

| Rank | Candidate Name | Hyper-Evasive FOV Lock (%) | Hyper Mean Dist (m) | Min Corner Speed (m/s) | Max Straight Speed (m/s) | Evasive FOV Lock (%) | Evasive Mean Dist (m) | Safety (Collisions) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **`run_02_pn_apex_cutter`** | **68.0%** | **18.67 m** | **3.1 m/s** | **7.1 m/s** | **41.3%** | **13.29 m** | 1 (grazing) |
| 🥈 **2** | **`run_06_unified_apex_dogfight_v2`** | **46.5%** | **38.95 m** | **2.0 m/s** | **8.3 m/s** | **54.9%** | **26.92 m** | **0 (Zero)** |
| 🥉 **3** | **`run_05_curriculum_speed_envelope`**| **23.4%** | **49.58 m** | **3.3 m/s** | **6.6 m/s** | **23.2%** | **39.54 m** | **0 (Zero)** |
| 4 | `run_01_trail_brake_governor` | 8.8% | 48.92 m | 2.3 m/s | 4.0 m/s | 9.1% | 38.45 m | 0 (Zero) |
| 5 | `run_03_retinal_slip_barrier` | 3.9% | 49.69 m | 1.9 m/s | 4.0 m/s | 3.5% | 39.31 m | 0 (Zero) |
| 6 | `run_04_curvature_energy_match` | 3.8% | 49.75 m | 1.7 m/s | 4.0 m/s | 3.4% | 39.30 m | 0 (Zero) |

---

## 4. Key Engineering & Physical Insights

### 4.1 Why `run_02_pn_apex_cutter` Achieved a Breakthrough (68.0% FOV Lock)
- **Natural Trail-Braking Emergence:** Unlike Runs 1, 3, and 4 which tried to force deceleration through heavy penalties, Proportional Navigation rewarded **velocity vector alignment**. Because rotating the velocity vector requires centripetal acceleration ($\dot{\theta} = a_{\text{lat}} / v$), the network discovered that slowing down to **$3.1\text{ m/s}$ in turns allowed its velocity vector to swing rapidly inside the target's curve**.
- **Dynamic Speed Modulation:** When the target straightened out, the policy punched the throttle up to **$7.1\text{ m/s}$** to close the gap.
- **Dramatic Separation Improvement:** Mean distance collapsed from **$41.0\text{ m} \to 18.67\text{ m}$**, maintaining visual lock for **$68.0\%$** of the hyper-evasive flight (more than double the previous overnight sweep's best model of $31.0\%$).

### 4.2 Why Runs 1, 3, and 4 Failed (The "Speed Starvation" Trap)
- In Runs 1, 3, and 4, the direct kinetic penalties on forward speed during turns were so severe that the policy learned to cap its maximum forward speed at **$4.0\text{ m/s}$** even on straights (`hyper_max_speed = 4.0 m/s`).
- Because the hyper-evasive target cruises at $6\text{--}10\text{ m/s}$, a drone capped at $4.0\text{ m/s}$ was quickly left behind, trailing at ~49 meters.
- **Rule of Thumb:** Never penalize speed itself; incentivize velocity vector alignment so the network chooses the optimal cornering speed naturally.

### 4.3 `run_06_unified_apex_dogfight_v2` for Zero-Collision High-Speed Chases
- Run 6 achieved the widest dynamic speed envelope: braking down to **$2.0\text{ m/s}$** in tight hairpins and sprinting up to **$8.3\text{ m/s}$** on straightaways.
- It scored the highest FOV retention on standard evasive targets (**$54.9\%$**, mean dist $26.9\text{ m}$) with **zero collisions ($0.0\%$)**.

---

## 5. Artifact Reference & Checkpoints

- **Champion Model (Best Hyper-Evasive Tracking):**
  - Path: [`control/weights/sweep_run_02_pn_apex_cutter/best_model/best_model.zip`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/sweep_run_02_pn_apex_cutter/best_model/best_model.zip)
  - Standalone Actor: [`control/weights/sweep_run_02_pn_apex_cutter/best_model/actor_weights.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/sweep_run_02_pn_apex_cutter/best_model/actor_weights.pt)
- **Runner-Up Model (Best Evasive Tracking & Zero Collision):**
  - Path: [`control/weights/sweep_run_06_unified_apex_dogfight_v2/best_model/best_model.zip`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/sweep_run_06_unified_apex_dogfight_v2/best_model/best_model.zip)
  - Standalone Actor: [`control/weights/sweep_run_06_unified_apex_dogfight_v2/best_model/actor_weights.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/sweep_run_06_unified_apex_dogfight_v2/best_model/actor_weights.pt)
- **Raw Benchmark Data:** [`outputs/benchmarks/control/decel_turn_sweep_results.json`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/outputs/benchmarks/control/decel_turn_sweep_results.json)
