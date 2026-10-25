# Autonomous Drone Pursuit: Overnight 17-Run Iterative RL Optimization Campaign

**Date:** 2026-10-25 11:47:30  
**System:** DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception  
**Harness:** 2,000,000 timesteps/candidate, Warm-Started from Cloned Expert, 8 Parallel Envs  
**Completed Candidates:** 14 / 17  

---

## 1. Executive Master Leaderboard (Held-Out Test Set 50 — Empirical Jetson Noise + 5% Drops)

| Rank | Run ID | Phase | Reward Architecture | Test Catch% | Test FOV% | Basket Dwell (s) | Min Dist (m) | Collisions | Status |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | [`run_10_tight_centering_border_penalty`](#run_10_tight_centering_border_penalty) | Phase III: Sensor Robustness | `tight_centering` | **0.0%** | **18.9%** | 0.04s | 5.58m | 1 | Completed |
| **2** | [`run_09_optical_expansion_lead`](#run_09_optical_expansion_lead) | Phase III: Sensor Robustness | `optical_expansion_lead` | **0.0%** | **18.5%** | 0.03s | 5.60m | 1 | Completed |
| **3** | [`run_01_monotone_potential_cushion`](#run_01_monotone_potential_cushion) | Phase I: Foundation & Safety | `monotone_potential` | **0.0%** | **17.4%** | 0.06s | 5.80m | 1 | Completed |
| **4** | [`run_11_aerodynamic_sideslip_governor`](#run_11_aerodynamic_sideslip_governor) | Phase III: Sensor Robustness | `aerodynamic_sideslip_governor` | **0.0%** | **17.3%** | 0.03s | 5.33m | 1 | Completed |
| **5** | [`run_02_huber_loss_standoff_well`](#run_02_huber_loss_standoff_well) | Phase I: Foundation & Safety | `huber_loss` | **0.0%** | **17.1%** | 0.02s | 5.42m | 1 | Completed |
| **6** | [`run_08_centripetal_load_envelope`](#run_08_centripetal_load_envelope) | Phase II: Agile Cornering | `centripetal_load_envelope` | **0.0%** | **16.9%** | 0.05s | 5.69m | 2 | Completed |
| **7** | [`run_07_retinal_slip_kinetic_barrier`](#run_07_retinal_slip_kinetic_barrier) | Phase II: Agile Cornering | `retinal_slip_kinetic_barrier` | **0.0%** | **16.1%** | 0.02s | 5.49m | 2 | Completed |
| **8** | [`run_05_trail_brake_corner_governor`](#run_05_trail_brake_corner_governor) | Phase II: Agile Cornering | `trail_brake_corner_governor` | **0.0%** | **12.8%** | 0.01s | 5.67m | 2 | Completed |
| **9** | [`run_14_unified_apex_dogfight_v2`](#run_14_unified_apex_dogfight_v2) | Phase IV: Synthesis Champions | `unified_apex_dogfight_v2` | **0.0%** | **11.2%** | 0.05s | 5.86m | 1 | Completed |
| **10** | [`run_06_proportional_nav_apex_cutter`](#run_06_proportional_nav_apex_cutter) | Phase II: Agile Cornering | `proportional_nav_apex_cutter` | **0.0%** | **6.6%** | 0.02s | 5.91m | 0 | Completed |
| **11** | [`run_03_closing_speed_bell_glide_slope`](#run_03_closing_speed_bell_glide_slope) | Phase I: Foundation & Safety | `closing_speed_bell` | **0.0%** | **4.8%** | 0.02s | 5.82m | 1 | Completed |
| **12** | [`run_13_unified_dogfight_optimal`](#run_13_unified_dogfight_optimal) | Phase IV: Synthesis Champions | `unified_dogfight_optimal` | **0.0%** | **4.8%** | 0.05s | 5.45m | 2 | Completed |
| **13** | [`run_04_cumulative_dwell_jackpot`](#run_04_cumulative_dwell_jackpot) | Phase I: Foundation & Safety | `cumulative_dwell_jackpot` | **0.0%** | **4.2%** | 0.07s | 4.44m | 3 | Completed |
| **14** | [`run_12_staged_curriculum_trail_brake`](#run_12_staged_curriculum_trail_brake) | Phase III: Sensor Robustness | `staged_curriculum_trail_brake` | **0.0%** | **3.9%** | 0.02s | 6.23m | 1 | Completed |

---

## 2. Benchmark Reference Standards (Held-Out Test Set 50)

| Controller Paradigm | Standoff Envelope | Interception Catch% | FOV Retention% | Cruising Catch% | Hyper-Evasive Catch% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Classical Baseline PID** | 3.5m (Empirical Noise) | 8.0% | 17.2% | 6.7% | 0.0% |
| **Retuned PID (Tuned Gains)** | 3.5m (Empirical Noise) | 12.0% | 30.0% | 33.3% | 0.0% |
| **Classical PID (Historical GT)** | 6.0m (Clean Telemetry) | 45.8% (100% Cruise) | 75.2% (90.9% Cruise) | 100.0% | 0.0% |

---

## 3. Iterative Analytical Diagnostics & Evolution Trace

### <a id="run_10_tight_centering_border_penalty"></a> `run_10_tight_centering_border_penalty` (Phase III: Sensor Robustness)
- **Reward Mode:** `tight_centering`
- **Core Hypothesis:** Steep quartic frame border penalties force target into tight 20% central optical cone.
- **Key Design:** High-Gain Boresight Centering with Quartic Frame Border Dampening
- **Training Duration:** 2,000,000 timesteps (~35.9 min)
- **Validation Score:** Task Score = -87.2 | Val FOV = 19.3% | Val Catch = 0.0%
- **Held-Out Test Results (50 Scenarios, Jetson Noise):**
  - **Catch Success Rate (Strict 3.0s lock):** **0.0%**
  - **In-FOV Line-of-Sight Retention:** **18.9%**
  - **Cumulative Firing Window Dwell:** **0.04s**
  - **Mean Minimum Distance:** 5.58m (Collisions: 1)
- **Profile Breakdown (Held-Out Test):**
  - *Cruising:* Catch = 0.0% | FOV = 19.0% | Dwell = 0.10s
  - *Evasive:* Catch = 0.0% | FOV = 19.4% | Dwell = 0.02s
  - *Hyper_evasive:* Catch = 0.0% | FOV = 18.0% | Dwell = 0.00s
- **Diagnostic Failure Mode Root-Cause Analysis:**
  > **Low Catch Success (0.0%):** Although FOV was 18.9%, the continuous 3.0s lock condition was rarely satisfied (dwell: 0.04s). **Severe Field-of-View Loss (18.9%):** Agent regularly loses visual lock. Indicates insufficient cornering deceleration or weak directional re-acquisition. **Profile Breakdown:** Cruising: 0.0% catch / 19.0% FOV | Hyper-Evasive: 0.0% catch / 18.0% FOV. **Hyper-Evasive Deficit:** High-G target jinks exceed the policy's turn rate envelope without aggressive deceleration. **Safety Infraction:** 1 physical collision breaches (d < 1.2m). Closing speed at the basket entry exceeded safe kinematic glide-slope limits.
- **Model Artifact:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_10_tight_centering_border_penalty/best_model/best_model.zip`](/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_10_tight_centering_border_penalty/best_model/best_model.zip)

### <a id="run_09_optical_expansion_lead"></a> `run_09_optical_expansion_lead` (Phase III: Sensor Robustness)
- **Reward Mode:** `optical_expansion_lead`
- **Core Hypothesis:** Bypasses metric range estimation noise by directly regulating image-plane bounding box divergence.
- **Key Design:** Direct Image-Plane Bbox Expansion Rate Feedback (range-sensor invariant)
- **Training Duration:** 2,000,000 timesteps (~38.5 min)
- **Validation Score:** Task Score = -94.0 | Val FOV = 19.1% | Val Catch = 0.0%
- **Held-Out Test Results (50 Scenarios, Jetson Noise):**
  - **Catch Success Rate (Strict 3.0s lock):** **0.0%**
  - **In-FOV Line-of-Sight Retention:** **18.5%**
  - **Cumulative Firing Window Dwell:** **0.03s**
  - **Mean Minimum Distance:** 5.60m (Collisions: 1)
- **Profile Breakdown (Held-Out Test):**
  - *Cruising:* Catch = 0.0% | FOV = 18.8% | Dwell = 0.04s
  - *Evasive:* Catch = 0.0% | FOV = 18.7% | Dwell = 0.03s
  - *Hyper_evasive:* Catch = 0.0% | FOV = 17.6% | Dwell = 0.01s
- **Diagnostic Failure Mode Root-Cause Analysis:**
  > **Low Catch Success (0.0%):** Although FOV was 18.5%, the continuous 3.0s lock condition was rarely satisfied (dwell: 0.03s). **Severe Field-of-View Loss (18.5%):** Agent regularly loses visual lock. Indicates insufficient cornering deceleration or weak directional re-acquisition. **Profile Breakdown:** Cruising: 0.0% catch / 18.8% FOV | Hyper-Evasive: 0.0% catch / 17.6% FOV. **Hyper-Evasive Deficit:** High-G target jinks exceed the policy's turn rate envelope without aggressive deceleration. **Safety Infraction:** 1 physical collision breaches (d < 1.2m). Closing speed at the basket entry exceeded safe kinematic glide-slope limits.
- **Model Artifact:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_09_optical_expansion_lead/best_model/best_model.zip`](/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_09_optical_expansion_lead/best_model/best_model.zip)

### <a id="run_01_monotone_potential_cushion"></a> `run_01_monotone_potential_cushion` (Phase I: Foundation & Safety)
- **Reward Mode:** `monotone_potential`
- **Core Hypothesis:** Eliminates suicidal ground strikes by establishing negative terminal barriers while testing baseline standoff attraction.
- **Key Design:** Linear-Exponential Hybrid Standoff Well with Ground Strike Barrier (-30.0 terminal penalty)
- **Training Duration:** 2,000,000 timesteps (~36.8 min)
- **Validation Score:** Task Score = -91.6 | Val FOV = 19.2% | Val Catch = 0.0%
- **Held-Out Test Results (50 Scenarios, Jetson Noise):**
  - **Catch Success Rate (Strict 3.0s lock):** **0.0%**
  - **In-FOV Line-of-Sight Retention:** **17.4%**
  - **Cumulative Firing Window Dwell:** **0.06s**
  - **Mean Minimum Distance:** 5.80m (Collisions: 1)
- **Profile Breakdown (Held-Out Test):**
  - *Cruising:* Catch = 0.0% | FOV = 17.4% | Dwell = 0.14s
  - *Evasive:* Catch = 0.0% | FOV = 17.0% | Dwell = 0.02s
  - *Hyper_evasive:* Catch = 0.0% | FOV = 18.0% | Dwell = 0.02s
- **Diagnostic Failure Mode Root-Cause Analysis:**
  > **Low Catch Success (0.0%):** Although FOV was 17.4%, the continuous 3.0s lock condition was rarely satisfied (dwell: 0.06s). **Severe Field-of-View Loss (17.4%):** Agent regularly loses visual lock. Indicates insufficient cornering deceleration or weak directional re-acquisition. **Profile Breakdown:** Cruising: 0.0% catch / 17.4% FOV | Hyper-Evasive: 0.0% catch / 18.0% FOV. **Hyper-Evasive Deficit:** High-G target jinks exceed the policy's turn rate envelope without aggressive deceleration. **Safety Infraction:** 1 physical collision breaches (d < 1.2m). Closing speed at the basket entry exceeded safe kinematic glide-slope limits.
- **Model Artifact:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_01_monotone_potential_cushion/best_model/best_model.zip`](/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_01_monotone_potential_cushion/best_model/best_model.zip)

### <a id="run_11_aerodynamic_sideslip_governor"></a> `run_11_aerodynamic_sideslip_governor` (Phase III: Sensor Robustness)
- **Reward Mode:** `aerodynamic_sideslip_governor`
- **Core Hypothesis:** Nulling aerodynamic sideslip maximizes forward thrust efficiency while keeping lateral strafe for micro-adjustments.
- **Key Design:** Sideslip Angle Nulling: aligns thrust vector with camera optical axis during sprint
- **Training Duration:** 2,000,000 timesteps (~38.4 min)
- **Validation Score:** Task Score = -62.4 | Val FOV = 16.0% | Val Catch = 0.0%
- **Held-Out Test Results (50 Scenarios, Jetson Noise):**
  - **Catch Success Rate (Strict 3.0s lock):** **0.0%**
  - **In-FOV Line-of-Sight Retention:** **17.3%**
  - **Cumulative Firing Window Dwell:** **0.03s**
  - **Mean Minimum Distance:** 5.33m (Collisions: 1)
- **Profile Breakdown (Held-Out Test):**
  - *Cruising:* Catch = 0.0% | FOV = 17.2% | Dwell = 0.06s
  - *Evasive:* Catch = 0.0% | FOV = 17.1% | Dwell = 0.03s
  - *Hyper_evasive:* Catch = 0.0% | FOV = 17.8% | Dwell = 0.00s
- **Diagnostic Failure Mode Root-Cause Analysis:**
  > **Low Catch Success (0.0%):** Although FOV was 17.3%, the continuous 3.0s lock condition was rarely satisfied (dwell: 0.03s). **Severe Field-of-View Loss (17.3%):** Agent regularly loses visual lock. Indicates insufficient cornering deceleration or weak directional re-acquisition. **Profile Breakdown:** Cruising: 0.0% catch / 17.2% FOV | Hyper-Evasive: 0.0% catch / 17.8% FOV. **Hyper-Evasive Deficit:** High-G target jinks exceed the policy's turn rate envelope without aggressive deceleration. **Safety Infraction:** 1 physical collision breaches (d < 1.2m). Closing speed at the basket entry exceeded safe kinematic glide-slope limits.
- **Model Artifact:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_11_aerodynamic_sideslip_governor/best_model/best_model.zip`](/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_11_aerodynamic_sideslip_governor/best_model/best_model.zip)

### <a id="run_02_huber_loss_standoff_well"></a> `run_02_huber_loss_standoff_well` (Phase I: Foundation & Safety)
- **Reward Mode:** `huber_loss`
- **Core Hypothesis:** Linear gradients at long range prevent 30-50m drift without explosive gradient spikes near the basket.
- **Key Design:** Smooth Huber Loss Potential Field (delta=1.0m, constant linear pull outside 4.5m)
- **Training Duration:** 2,000,000 timesteps (~36.9 min)
- **Validation Score:** Task Score = -44.3 | Val FOV = 17.4% | Val Catch = 0.0%
- **Held-Out Test Results (50 Scenarios, Jetson Noise):**
  - **Catch Success Rate (Strict 3.0s lock):** **0.0%**
  - **In-FOV Line-of-Sight Retention:** **17.1%**
  - **Cumulative Firing Window Dwell:** **0.02s**
  - **Mean Minimum Distance:** 5.42m (Collisions: 1)
- **Profile Breakdown (Held-Out Test):**
  - *Cruising:* Catch = 0.0% | FOV = 17.7% | Dwell = 0.06s
  - *Evasive:* Catch = 0.0% | FOV = 17.3% | Dwell = 0.01s
  - *Hyper_evasive:* Catch = 0.0% | FOV = 16.1% | Dwell = 0.01s
- **Diagnostic Failure Mode Root-Cause Analysis:**
  > **Low Catch Success (0.0%):** Although FOV was 17.1%, the continuous 3.0s lock condition was rarely satisfied (dwell: 0.02s). **Severe Field-of-View Loss (17.1%):** Agent regularly loses visual lock. Indicates insufficient cornering deceleration or weak directional re-acquisition. **Profile Breakdown:** Cruising: 0.0% catch / 17.7% FOV | Hyper-Evasive: 0.0% catch / 16.1% FOV. **Hyper-Evasive Deficit:** High-G target jinks exceed the policy's turn rate envelope without aggressive deceleration. **Safety Infraction:** 1 physical collision breaches (d < 1.2m). Closing speed at the basket entry exceeded safe kinematic glide-slope limits.
- **Model Artifact:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_02_huber_loss_standoff_well/best_model/best_model.zip`](/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_02_huber_loss_standoff_well/best_model/best_model.zip)

### <a id="run_08_centripetal_load_envelope"></a> `run_08_centripetal_load_envelope` (Phase II: Agile Cornering)
- **Reward Mode:** `centripetal_load_envelope`
- **Core Hypothesis:** Centripetal acceleration matching (v_safe = a_max / omega_turn) guarantees the drone stays inside camera turn envelope.
- **Key Design:** Physics Centripetal Load Matching: dynamically constrains surge speed by yaw rate
- **Training Duration:** 2,000,000 timesteps (~38.0 min)
- **Validation Score:** Task Score = -86.3 | Val FOV = 18.8% | Val Catch = 0.0%
- **Held-Out Test Results (50 Scenarios, Jetson Noise):**
  - **Catch Success Rate (Strict 3.0s lock):** **0.0%**
  - **In-FOV Line-of-Sight Retention:** **16.9%**
  - **Cumulative Firing Window Dwell:** **0.05s**
  - **Mean Minimum Distance:** 5.69m (Collisions: 2)
- **Profile Breakdown (Held-Out Test):**
  - *Cruising:* Catch = 0.0% | FOV = 16.1% | Dwell = 0.05s
  - *Evasive:* Catch = 0.0% | FOV = 16.6% | Dwell = 0.06s
  - *Hyper_evasive:* Catch = 0.0% | FOV = 18.5% | Dwell = 0.02s
- **Diagnostic Failure Mode Root-Cause Analysis:**
  > **Low Catch Success (0.0%):** Although FOV was 16.9%, the continuous 3.0s lock condition was rarely satisfied (dwell: 0.05s). **Severe Field-of-View Loss (16.9%):** Agent regularly loses visual lock. Indicates insufficient cornering deceleration or weak directional re-acquisition. **Profile Breakdown:** Cruising: 0.0% catch / 16.1% FOV | Hyper-Evasive: 0.0% catch / 18.5% FOV. **Hyper-Evasive Deficit:** High-G target jinks exceed the policy's turn rate envelope without aggressive deceleration. **Safety Infraction:** 2 physical collision breaches (d < 1.2m). Closing speed at the basket entry exceeded safe kinematic glide-slope limits.
- **Model Artifact:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_08_centripetal_load_envelope/best_model/best_model.zip`](/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_08_centripetal_load_envelope/best_model/best_model.zip)

### <a id="run_07_retinal_slip_kinetic_barrier"></a> `run_07_retinal_slip_kinetic_barrier` (Phase II: Agile Cornering)
- **Reward Mode:** `retinal_slip_kinetic_barrier`
- **Core Hypothesis:** Raptor attack law stabilizes line-of-sight angular rates before authorizing high-speed forward acceleration.
- **Key Design:** Biological Raptor Pursuit: high optical retinal slip triggers kinetic braking
- **Training Duration:** 2,000,000 timesteps (~35.7 min)
- **Validation Score:** Task Score = -80.9 | Val FOV = 21.1% | Val Catch = 0.0%
- **Held-Out Test Results (50 Scenarios, Jetson Noise):**
  - **Catch Success Rate (Strict 3.0s lock):** **0.0%**
  - **In-FOV Line-of-Sight Retention:** **16.1%**
  - **Cumulative Firing Window Dwell:** **0.02s**
  - **Mean Minimum Distance:** 5.49m (Collisions: 2)
- **Profile Breakdown (Held-Out Test):**
  - *Cruising:* Catch = 0.0% | FOV = 17.4% | Dwell = 0.03s
  - *Evasive:* Catch = 0.0% | FOV = 16.1% | Dwell = 0.01s
  - *Hyper_evasive:* Catch = 0.0% | FOV = 14.7% | Dwell = 0.03s
- **Diagnostic Failure Mode Root-Cause Analysis:**
  > **Low Catch Success (0.0%):** Although FOV was 16.1%, the continuous 3.0s lock condition was rarely satisfied (dwell: 0.02s). **Severe Field-of-View Loss (16.1%):** Agent regularly loses visual lock. Indicates insufficient cornering deceleration or weak directional re-acquisition. **Profile Breakdown:** Cruising: 0.0% catch / 17.4% FOV | Hyper-Evasive: 0.0% catch / 14.7% FOV. **Hyper-Evasive Deficit:** High-G target jinks exceed the policy's turn rate envelope without aggressive deceleration. **Safety Infraction:** 2 physical collision breaches (d < 1.2m). Closing speed at the basket entry exceeded safe kinematic glide-slope limits.
- **Model Artifact:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_07_retinal_slip_kinetic_barrier/best_model/best_model.zip`](/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_07_retinal_slip_kinetic_barrier/best_model/best_model.zip)

### <a id="run_05_trail_brake_corner_governor"></a> `run_05_trail_brake_corner_governor` (Phase II: Agile Cornering)
- **Reward Mode:** `trail_brake_corner_governor`
- **Core Hypothesis:** Prevents hyper-evasive blow-by by incentivizing deceleration to 2-3 m/s during hard target jinks.
- **Key Design:** Decelerate-to-Turn Governor: rewards active surge braking when |ex| > 0.25
- **Training Duration:** 2,000,000 timesteps (~38.0 min)
- **Validation Score:** Task Score = -104.4 | Val FOV = 13.0% | Val Catch = 0.0%
- **Held-Out Test Results (50 Scenarios, Jetson Noise):**
  - **Catch Success Rate (Strict 3.0s lock):** **0.0%**
  - **In-FOV Line-of-Sight Retention:** **12.8%**
  - **Cumulative Firing Window Dwell:** **0.01s**
  - **Mean Minimum Distance:** 5.67m (Collisions: 2)
- **Profile Breakdown (Held-Out Test):**
  - *Cruising:* Catch = 0.0% | FOV = 13.2% | Dwell = 0.02s
  - *Evasive:* Catch = 0.0% | FOV = 12.7% | Dwell = 0.00s
  - *Hyper_evasive:* Catch = 0.0% | FOV = 12.7% | Dwell = 0.01s
- **Diagnostic Failure Mode Root-Cause Analysis:**
  > **Low Catch Success (0.0%):** Although FOV was 12.8%, the continuous 3.0s lock condition was rarely satisfied (dwell: 0.01s). **Severe Field-of-View Loss (12.8%):** Agent regularly loses visual lock. Indicates insufficient cornering deceleration or weak directional re-acquisition. **Profile Breakdown:** Cruising: 0.0% catch / 13.2% FOV | Hyper-Evasive: 0.0% catch / 12.7% FOV. **Hyper-Evasive Deficit:** High-G target jinks exceed the policy's turn rate envelope without aggressive deceleration. **Safety Infraction:** 2 physical collision breaches (d < 1.2m). Closing speed at the basket entry exceeded safe kinematic glide-slope limits.
- **Model Artifact:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_05_trail_brake_corner_governor/best_model/best_model.zip`](/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_05_trail_brake_corner_governor/best_model/best_model.zip)

### <a id="run_14_unified_apex_dogfight_v2"></a> `run_14_unified_apex_dogfight_v2` (Phase IV: Synthesis Champions)
- **Reward Mode:** `unified_apex_dogfight_v2`
- **Core Hypothesis:** Refined dogfight formulation combining cornering deceleration with superlinear dwell jackpot bonuses.
- **Key Design:** Enhanced Dogfight Synthesis: Apex Cutting + Kinetic Braking + High Dwell Jackpot
- **Training Duration:** 2,000,000 timesteps (~36.3 min)
- **Validation Score:** Task Score = -62.9 | Val FOV = 19.3% | Val Catch = 0.0%
- **Held-Out Test Results (50 Scenarios, Jetson Noise):**
  - **Catch Success Rate (Strict 3.0s lock):** **0.0%**
  - **In-FOV Line-of-Sight Retention:** **11.2%**
  - **Cumulative Firing Window Dwell:** **0.05s**
  - **Mean Minimum Distance:** 5.86m (Collisions: 1)
- **Profile Breakdown (Held-Out Test):**
  - *Cruising:* Catch = 0.0% | FOV = 9.8% | Dwell = 0.14s
  - *Evasive:* Catch = 0.0% | FOV = 11.2% | Dwell = 0.02s
  - *Hyper_evasive:* Catch = 0.0% | FOV = 12.7% | Dwell = 0.01s
- **Diagnostic Failure Mode Root-Cause Analysis:**
  > **Low Catch Success (0.0%):** Although FOV was 11.2%, the continuous 3.0s lock condition was rarely satisfied (dwell: 0.05s). **Severe Field-of-View Loss (11.2%):** Agent regularly loses visual lock. Indicates insufficient cornering deceleration or weak directional re-acquisition. **Profile Breakdown:** Cruising: 0.0% catch / 9.8% FOV | Hyper-Evasive: 0.0% catch / 12.7% FOV. **Hyper-Evasive Deficit:** High-G target jinks exceed the policy's turn rate envelope without aggressive deceleration. **Safety Infraction:** 1 physical collision breaches (d < 1.2m). Closing speed at the basket entry exceeded safe kinematic glide-slope limits.
- **Model Artifact:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_14_unified_apex_dogfight_v2/best_model/best_model.zip`](/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_14_unified_apex_dogfight_v2/best_model/best_model.zip)

### <a id="run_06_proportional_nav_apex_cutter"></a> `run_06_proportional_nav_apex_cutter` (Phase II: Agile Cornering)
- **Reward Mode:** `proportional_nav_apex_cutter`
- **Core Hypothesis:** Aligns velocity vector with target heading vector to cut the inside corner of evasive weave patterns.
- **Key Design:** Proportional Navigation Velocity Alignment: cuts evasive turn apexes
- **Training Duration:** 2,000,000 timesteps (~36.5 min)
- **Validation Score:** Task Score = -42.8 | Val FOV = 6.3% | Val Catch = 0.0%
- **Held-Out Test Results (50 Scenarios, Jetson Noise):**
  - **Catch Success Rate (Strict 3.0s lock):** **0.0%**
  - **In-FOV Line-of-Sight Retention:** **6.6%**
  - **Cumulative Firing Window Dwell:** **0.02s**
  - **Mean Minimum Distance:** 5.91m (Collisions: 0)
- **Profile Breakdown (Held-Out Test):**
  - *Cruising:* Catch = 0.0% | FOV = 5.0% | Dwell = 0.04s
  - *Evasive:* Catch = 0.0% | FOV = 6.4% | Dwell = 0.01s
  - *Hyper_evasive:* Catch = 0.0% | FOV = 8.9% | Dwell = 0.01s
- **Diagnostic Failure Mode Root-Cause Analysis:**
  > **Low Catch Success (0.0%):** Although FOV was 6.6%, the continuous 3.0s lock condition was rarely satisfied (dwell: 0.02s). **Severe Field-of-View Loss (6.6%):** Agent regularly loses visual lock. Indicates insufficient cornering deceleration or weak directional re-acquisition. **Profile Breakdown:** Cruising: 0.0% catch / 5.0% FOV | Hyper-Evasive: 0.0% catch / 8.9% FOV. **Hyper-Evasive Deficit:** High-G target jinks exceed the policy's turn rate envelope without aggressive deceleration. **Clean Safety Record:** 0 collisions detected across all 50 held-out test encounters.
- **Model Artifact:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_06_proportional_nav_apex_cutter/best_model/best_model.zip`](/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_06_proportional_nav_apex_cutter/best_model/best_model.zip)

### <a id="run_03_closing_speed_bell_glide_slope"></a> `run_03_closing_speed_bell_glide_slope` (Phase I: Foundation & Safety)
- **Reward Mode:** `closing_speed_bell`
- **Core Hypothesis:** Analytical glide-slope matching stops overshoot collisions by shaping closing speed to zero as distance reaches 3.5m.
- **Key Design:** Kinematic Deceleration Glide-Slope + Gaussian Bell on Approach Velocity
- **Training Duration:** 2,000,000 timesteps (~36.0 min)
- **Validation Score:** Task Score = -45.8 | Val FOV = 3.4% | Val Catch = 0.0%
- **Held-Out Test Results (50 Scenarios, Jetson Noise):**
  - **Catch Success Rate (Strict 3.0s lock):** **0.0%**
  - **In-FOV Line-of-Sight Retention:** **4.8%**
  - **Cumulative Firing Window Dwell:** **0.02s**
  - **Mean Minimum Distance:** 5.82m (Collisions: 1)
- **Profile Breakdown (Held-Out Test):**
  - *Cruising:* Catch = 0.0% | FOV = 4.4% | Dwell = 0.03s
  - *Evasive:* Catch = 0.0% | FOV = 5.2% | Dwell = 0.02s
  - *Hyper_evasive:* Catch = 0.0% | FOV = 4.5% | Dwell = 0.02s
- **Diagnostic Failure Mode Root-Cause Analysis:**
  > **Low Catch Success (0.0%):** Although FOV was 4.8%, the continuous 3.0s lock condition was rarely satisfied (dwell: 0.02s). **Severe Field-of-View Loss (4.8%):** Agent regularly loses visual lock. Indicates insufficient cornering deceleration or weak directional re-acquisition. **Profile Breakdown:** Cruising: 0.0% catch / 4.4% FOV | Hyper-Evasive: 0.0% catch / 4.5% FOV. **Hyper-Evasive Deficit:** High-G target jinks exceed the policy's turn rate envelope without aggressive deceleration. **Safety Infraction:** 1 physical collision breaches (d < 1.2m). Closing speed at the basket entry exceeded safe kinematic glide-slope limits.
- **Model Artifact:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_03_closing_speed_bell_glide_slope/best_model/best_model.zip`](/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_03_closing_speed_bell_glide_slope/best_model/best_model.zip)

### <a id="run_13_unified_dogfight_optimal"></a> `run_13_unified_dogfight_optimal` (Phase IV: Synthesis Champions)
- **Reward Mode:** `unified_dogfight_optimal`
- **Core Hypothesis:** Synergistic multi-objective synthesis combining the strongest distance, lead, and speed matching mechanisms.
- **Key Design:** Unified Multi-Objective: Huber Well + PN Lead Vector + Glide-Slope Closing
- **Training Duration:** 2,000,000 timesteps (~35.9 min)
- **Validation Score:** Task Score = -74.7 | Val FOV = 7.1% | Val Catch = 0.0%
- **Held-Out Test Results (50 Scenarios, Jetson Noise):**
  - **Catch Success Rate (Strict 3.0s lock):** **0.0%**
  - **In-FOV Line-of-Sight Retention:** **4.8%**
  - **Cumulative Firing Window Dwell:** **0.05s**
  - **Mean Minimum Distance:** 5.45m (Collisions: 2)
- **Profile Breakdown (Held-Out Test):**
  - *Cruising:* Catch = 0.0% | FOV = 3.6% | Dwell = 0.03s
  - *Evasive:* Catch = 0.0% | FOV = 4.8% | Dwell = 0.04s
  - *Hyper_evasive:* Catch = 0.0% | FOV = 6.3% | Dwell = 0.07s
- **Diagnostic Failure Mode Root-Cause Analysis:**
  > **Low Catch Success (0.0%):** Although FOV was 4.8%, the continuous 3.0s lock condition was rarely satisfied (dwell: 0.05s). **Severe Field-of-View Loss (4.8%):** Agent regularly loses visual lock. Indicates insufficient cornering deceleration or weak directional re-acquisition. **Profile Breakdown:** Cruising: 0.0% catch / 3.6% FOV | Hyper-Evasive: 0.0% catch / 6.3% FOV. **Hyper-Evasive Deficit:** High-G target jinks exceed the policy's turn rate envelope without aggressive deceleration. **Safety Infraction:** 2 physical collision breaches (d < 1.2m). Closing speed at the basket entry exceeded safe kinematic glide-slope limits.
- **Model Artifact:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_13_unified_dogfight_optimal/best_model/best_model.zip`](/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_13_unified_dogfight_optimal/best_model/best_model.zip)

### <a id="run_04_cumulative_dwell_jackpot"></a> `run_04_cumulative_dwell_jackpot` (Phase I: Foundation & Safety)
- **Reward Mode:** `cumulative_dwell_jackpot`
- **Core Hypothesis:** Superlinear dwell acceleration creates strong gradient to hold continuous lock rather than making high-speed passes.
- **Key Design:** Superlinear Dwell Scaling (rewards continuous formation hold over glancing passes)
- **Training Duration:** 2,000,000 timesteps (~35.1 min)
- **Validation Score:** Task Score = -38.1 | Val FOV = 7.1% | Val Catch = 0.0%
- **Held-Out Test Results (50 Scenarios, Jetson Noise):**
  - **Catch Success Rate (Strict 3.0s lock):** **0.0%**
  - **In-FOV Line-of-Sight Retention:** **4.2%**
  - **Cumulative Firing Window Dwell:** **0.07s**
  - **Mean Minimum Distance:** 4.44m (Collisions: 3)
- **Profile Breakdown (Held-Out Test):**
  - *Cruising:* Catch = 0.0% | FOV = 2.2% | Dwell = 0.02s
  - *Evasive:* Catch = 0.0% | FOV = 4.4% | Dwell = 0.10s
  - *Hyper_evasive:* Catch = 0.0% | FOV = 6.2% | Dwell = 0.05s
- **Diagnostic Failure Mode Root-Cause Analysis:**
  > **Low Catch Success (0.0%):** Although FOV was 4.2%, the continuous 3.0s lock condition was rarely satisfied (dwell: 0.07s). **Severe Field-of-View Loss (4.2%):** Agent regularly loses visual lock. Indicates insufficient cornering deceleration or weak directional re-acquisition. **Profile Breakdown:** Cruising: 0.0% catch / 2.2% FOV | Hyper-Evasive: 0.0% catch / 6.2% FOV. **Hyper-Evasive Deficit:** High-G target jinks exceed the policy's turn rate envelope without aggressive deceleration. **Safety Infraction:** 3 physical collision breaches (d < 1.2m). Closing speed at the basket entry exceeded safe kinematic glide-slope limits.
- **Model Artifact:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_04_cumulative_dwell_jackpot/best_model/best_model.zip`](/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_04_cumulative_dwell_jackpot/best_model/best_model.zip)

### <a id="run_12_staged_curriculum_trail_brake"></a> `run_12_staged_curriculum_trail_brake` (Phase III: Sensor Robustness)
- **Reward Mode:** `staged_curriculum_trail_brake`
- **Core Hypothesis:** Biasing training encounter distribution to 55% hyper-evasive and crossing targets forces specialized multi-aspect capability.
- **Key Design:** Trail-Braking + Encounter Curriculum (heavy weighting on crossing & head-on)
- **Training Duration:** 2,000,000 timesteps (~38.3 min)
- **Validation Score:** Task Score = -35.5 | Val FOV = 5.1% | Val Catch = 0.0%
- **Held-Out Test Results (50 Scenarios, Jetson Noise):**
  - **Catch Success Rate (Strict 3.0s lock):** **0.0%**
  - **In-FOV Line-of-Sight Retention:** **3.9%**
  - **Cumulative Firing Window Dwell:** **0.02s**
  - **Mean Minimum Distance:** 6.23m (Collisions: 1)
- **Profile Breakdown (Held-Out Test):**
  - *Cruising:* Catch = 0.0% | FOV = 3.3% | Dwell = 0.05s
  - *Evasive:* Catch = 0.0% | FOV = 4.4% | Dwell = 0.01s
  - *Hyper_evasive:* Catch = 0.0% | FOV = 3.9% | Dwell = 0.02s
- **Diagnostic Failure Mode Root-Cause Analysis:**
  > **Low Catch Success (0.0%):** Although FOV was 3.9%, the continuous 3.0s lock condition was rarely satisfied (dwell: 0.02s). **Severe Field-of-View Loss (3.9%):** Agent regularly loses visual lock. Indicates insufficient cornering deceleration or weak directional re-acquisition. **Profile Breakdown:** Cruising: 0.0% catch / 3.3% FOV | Hyper-Evasive: 0.0% catch / 3.9% FOV. **Hyper-Evasive Deficit:** High-G target jinks exceed the policy's turn rate envelope without aggressive deceleration. **Safety Infraction:** 1 physical collision breaches (d < 1.2m). Closing speed at the basket entry exceeded safe kinematic glide-slope limits.
- **Model Artifact:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_12_staged_curriculum_trail_brake/best_model/best_model.zip`](/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_17_campaign/run_12_staged_curriculum_trail_brake/best_model/best_model.zip)
