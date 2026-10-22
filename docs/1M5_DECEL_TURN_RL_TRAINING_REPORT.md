# 1.5M Timestep Decelerate-to-Turn & Standoff Capture RL Training Report

**Date:** 2026-10-22 17:08:35
**Pipeline:** RecurrentPPO (MlpLstmPolicy) fine-tuning from `cloned_expert_v1` actor weights
**Timesteps per Candidate:** 1,500,000 steps (~7.5× more than the 200k sweep)
**Parallel Environments:** 8 vectorized workers
**Exploration Variance:** `log_std = -0.8` (std ≈ 0.45, un-clamping throttle dynamics)
**Deceleration Physics:** `q_pitch` relaxed to -18° backward pitch, `q_speed` relaxed to 5.0 m/s closing speed
**Autonomous Policy:** Zero manual regulation / overrides. Pure end-to-end neural actor.
**Total Sweep Runtime:** 137.5 minutes

---

## 1. Candidate Leaderboard (Hyper-Evasive Validation Suite)

| Rank | Candidate Name | Reward Mode | Hyper FOV% | Mean Dist | Turn Speed (Min) | Straight Sprint (Max) | Max Continuous Dwell | Success Rate (≥3.0s Dwell) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `run_04_huber_standoff_dogfight` | `huber_standoff_dogfight` | **32.7%** | 46.08 m | 3.4 m/s | 6.0 m/s | **0.02 s** | **0.0%** |
| **2** | `run_05_aerodynamic_sideslip_governor` | `aerodynamic_sideslip_governor` | **13.5%** | 25.45 m | 2.1 m/s | 12.0 m/s | **0.00 s** | **0.0%** |
| **3** | `run_01_pn_sprint_apex_v3` | `pn_sprint_apex_v3` | **13.2%** | 31.53 m | 0.9 m/s | 10.4 m/s | **0.00 s** | **0.0%** |
| **4** | `run_02_centripetal_load_envelope` | `centripetal_load_envelope` | **9.6%** | 40.18 m | 0.1 m/s | 11.7 m/s | **0.00 s** | **0.0%** |
| **5** | `run_03_retinal_slip_raptor_v3` | `retinal_slip_raptor_v3` | **6.7%** | 54.30 m | 1.6 m/s | 8.3 m/s | **0.00 s** | **0.0%** |

---

## 2. Detailed Candidate Formulations & Findings

### `run_04_huber_standoff_dogfight`
- **Reward Mode:** `huber_standoff_dogfight`
- **Core Hypothesis:** Huber Standoff Potential Well + 3D Relative Velocity Matching at Standoff + Superlinear Dwell Jackpot
- **Hyper-Evasive Optical Retention (FOV%):** 32.7%
- **Hyper-Evasive Mean Distance:** 46.08 m (Min Distance: 3.28 m)
- **Speed Modulation:** 3.4 m/s in turns $\longleftrightarrow$ 6.0 m/s on straights (Mean: 4.2 m/s)
- **Standoff Capture Dwell:** Cumulative: 0.00 s, Peak Continuous Dwell: **0.02 s**
- **Interception Success Rate:** **0.0%**
- **Evasive Benchmark Baseline:** FOV=41.6%, Mean Dist=22.65 m
- **Model Checkpoint:** `/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/sweep_1m5_run_04_huber_standoff_dogfight/best_model/best_model.zip`

### `run_05_aerodynamic_sideslip_governor`
- **Reward Mode:** `aerodynamic_sideslip_governor`
- **Core Hypothesis:** Aerodynamic Sideslip Angle Nulling + Boresight Sprint Drive + Active Trail-Brake Turn Bonus
- **Hyper-Evasive Optical Retention (FOV%):** 13.5%
- **Hyper-Evasive Mean Distance:** 25.45 m (Min Distance: 1.23 m)
- **Speed Modulation:** 2.1 m/s in turns $\longleftrightarrow$ 12.0 m/s on straights (Mean: 9.5 m/s)
- **Standoff Capture Dwell:** Cumulative: 0.00 s, Peak Continuous Dwell: **0.00 s**
- **Interception Success Rate:** **0.0%**
- **Evasive Benchmark Baseline:** FOV=16.5%, Mean Dist=37.41 m
- **Model Checkpoint:** `/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/sweep_1m5_run_05_aerodynamic_sideslip_governor/best_model/best_model.zip`

### `run_01_pn_sprint_apex_v3`
- **Reward Mode:** `pn_sprint_apex_v3`
- **Core Hypothesis:** Proportional Navigation Vector Alignment + Monotonic Closing Potential (Unconstrained Sprint on Straights)
- **Hyper-Evasive Optical Retention (FOV%):** 13.2%
- **Hyper-Evasive Mean Distance:** 31.53 m (Min Distance: 1.65 m)
- **Speed Modulation:** 0.9 m/s in turns $\longleftrightarrow$ 10.4 m/s on straights (Mean: 6.9 m/s)
- **Standoff Capture Dwell:** Cumulative: 0.00 s, Peak Continuous Dwell: **0.00 s**
- **Interception Success Rate:** **0.0%**
- **Evasive Benchmark Baseline:** FOV=12.4%, Mean Dist=22.45 m
- **Model Checkpoint:** `/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/sweep_1m5_run_01_pn_sprint_apex_v3/best_model/best_model.zip`

### `run_02_centripetal_load_envelope`
- **Reward Mode:** `centripetal_load_envelope`
- **Core Hypothesis:** Physics-Grounded Centripetal Safe Speed Envelope Matching (v_safe = a_max / omega_turn)
- **Hyper-Evasive Optical Retention (FOV%):** 9.6%
- **Hyper-Evasive Mean Distance:** 40.18 m (Min Distance: 3.18 m)
- **Speed Modulation:** 0.1 m/s in turns $\longleftrightarrow$ 11.7 m/s on straights (Mean: 6.9 m/s)
- **Standoff Capture Dwell:** Cumulative: 0.00 s, Peak Continuous Dwell: **0.00 s**
- **Interception Success Rate:** **0.0%**
- **Evasive Benchmark Baseline:** FOV=9.2%, Mean Dist=30.21 m
- **Model Checkpoint:** `/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/sweep_1m5_run_02_centripetal_load_envelope/best_model/best_model.zip`

### `run_03_retinal_slip_raptor_v3`
- **Reward Mode:** `retinal_slip_raptor_v3`
- **Core Hypothesis:** Biological Raptor Pursuit: Low Retinal Slip Sprint Drive vs High Slip Kinetic Braking
- **Hyper-Evasive Optical Retention (FOV%):** 6.7%
- **Hyper-Evasive Mean Distance:** 54.30 m (Min Distance: 3.32 m)
- **Speed Modulation:** 1.6 m/s in turns $\longleftrightarrow$ 8.3 m/s on straights (Mean: 4.8 m/s)
- **Standoff Capture Dwell:** Cumulative: 0.00 s, Peak Continuous Dwell: **0.00 s**
- **Interception Success Rate:** **0.0%**
- **Evasive Benchmark Baseline:** FOV=5.5%, Mean Dist=35.95 m
- **Model Checkpoint:** `/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/sweep_1m5_run_03_retinal_slip_raptor_v3/best_model/best_model.zip`

---

## 3. Comparative Summary vs. Baseline and 200k Sweep

| Architecture | Timesteps | Hyper FOV% | Mean Dist | Speed Range | Success Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Classical PID | N/A | 32.5% | 29.8 m | Oscillatory | 0.0% (Crash in val_26) |
| Untrained Cloned Expert | 0 | 44.6% | 23.4 m | Flat (~5.1 m/s) | 0.0% |
| Overnight Sweep Best | 200k | 31.0% | 41.0 m | Flat | 0.0% |
| Decel-to-Turn Run 02 (PN Apex) | 200k | 68.0% | 18.7 m | 3.1 - 7.1 m/s | 0.0% |
| **run_04_huber_standoff_dogfight (Current 1.5M)** | **1.5M** | **32.7%** | **46.08 m** | **3.4 - 6.0 m/s** | **0.0%** |