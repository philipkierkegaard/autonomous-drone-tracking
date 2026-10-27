# Autonomous Drone Pursuit: Overnight 12-Run Iterative RL Optimization Campaign

**Date:** 2026-10-27 02:56:05  
**System:** DTU Autonomous Drone Pursuit & Interception — Airborne Visual Servoing  
**Status:** Completed 12 / 12 Candidate Runs  
**Baseline Reference (Tuned PID):** 1.0s Catch = **63.0%** | 3.0s Lock = **40.3%** | Mean Dwell = **2.86s** | FOV = **60.5%**  
**Warm-Start Prior (`cloned_expert_v6`):** 1.0s Catch = **45.0%** | 3.0s Lock = **32.0%** | Mean Dwell = **2.20s** | FOV = **52.1%**  

---

## 1. Executive Master Leaderboard (300-Scenario Benchmark Suite)

| Rank | Run ID | Phase | Reward Architecture | 1.0s Catch% | 3.0s Lock% | Mean Dwell (s) | FOV Retention% | Min Dist (m) | Collisions | Status |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | [`run_04_kinematic_glide_slope`](#run_04_kinematic_glide_slope) | Phase II: Relative Velocity Matching | `theory_04_kinematic_glide_slope` | **5.3%** | **0.7%** | 0.23s | 14.0% | 4.44m | 4 | Completed |
| **2** | [`run_07_decel_to_turn_centripetal`](#run_07_decel_to_turn_centripetal) | Phase III: Agile Cornering | `theory_07_decel_to_turn_centripetal` | **1.0%** | **0.0%** | 0.06s | 3.0% | 5.16m | 2 | Completed |
| **3** | [`run_01_potential_dwell_baseline`](#run_01_potential_dwell_baseline) | Phase I: Potential Fields | `theory_01_potential_dwell_baseline` | **0.3%** | **0.0%** | 0.06s | 9.8% | 5.85m | 0 | Completed |
| **4** | [`run_12_adaptive_grand_champion`](#run_12_adaptive_grand_champion) | Phase IV: Synthesis Champions | `theory_12_adaptive_grand_champion` | **0.3%** | **0.0%** | 0.07s | 3.5% | 4.56m | 4 | Completed |
| **5** | [`run_02_asymmetric_sprint_huber`](#run_02_asymmetric_sprint_huber) | Phase I: Potential Fields | `theory_02_asymmetric_sprint_huber` | **0.0%** | **0.0%** | 0.07s | 11.2% | 5.74m | 0 | Completed |
| **6** | [`run_03_optical_looming_expansion`](#run_03_optical_looming_expansion) | Phase I: Potential Fields | `theory_03_optical_looming_expansion` | **0.0%** | **0.0%** | 0.03s | 6.5% | 5.35m | 2 | Completed |
| **7** | [`run_05_relative_velocity_nulling`](#run_05_relative_velocity_nulling) | Phase II: Relative Velocity Matching | `theory_05_relative_velocity_nulling` | **0.0%** | **0.0%** | 0.04s | 1.1% | 6.03m | 1 | Completed |
| **8** | [`run_06_lead_pursuit_vector_alignment`](#run_06_lead_pursuit_vector_alignment) | Phase II: Relative Velocity Matching | `theory_06_lead_pursuit_vector_alignment` | **0.0%** | **0.0%** | 0.06s | 2.4% | 5.83m | 2 | Completed |
| **9** | [`run_08_retinal_slip_raptor`](#run_08_retinal_slip_raptor) | Phase III: Agile Cornering | `theory_08_retinal_slip_raptor` | **0.0%** | **0.0%** | 0.04s | 10.0% | 5.98m | 0 | Completed |
| **10** | [`run_09_curvature_adaptive_basket`](#run_09_curvature_adaptive_basket) | Phase III: Agile Cornering | `theory_09_curvature_adaptive_basket` | **0.0%** | **0.0%** | 0.03s | 2.1% | 6.13m | 0 | Completed |
| **11** | [`run_10_superlinear_dwell_jackpot`](#run_10_superlinear_dwell_jackpot) | Phase IV: Synthesis Champions | `theory_10_superlinear_dwell_jackpot` | **0.0%** | **0.0%** | 0.04s | 7.4% | 5.74m | 0 | Completed |
| **12** | [`run_11_unified_dogfight_synthesis`](#run_11_unified_dogfight_synthesis) | Phase IV: Synthesis Champions | `theory_11_unified_dogfight_synthesis` | **0.0%** | **0.0%** | 0.05s | 10.1% | 5.44m | 1 | Completed |

---

## 2. Iterative Analytical Diagnostics & Evolution Trace

### <a id="run_01_potential_dwell_baseline"></a> `run_01_potential_dwell_baseline` (Phase I: Potential Fields)
- **Architecture / Reward Mode:** `theory_01_potential_dwell_baseline`
- **Hypothesis:** Pure additive potential field (Gaussian LOS + parabolic 2.5-4.5m well + smooth progressive dwell) eliminates multiplicative gradient squashing.
- **Status:** Completed
- **Training Duration:** 1082.2s (~18.0 min)
- **300 Benchmark Overall Results:**
  - **1.0s Catch Rate:** **0.33%** (PID: 63.0%, Cloned v6: 45.0%)
  - **3.0s Continuous Lock Rate:** **0.00%** (PID: 40.3%, Cloned v6: 32.0%)
  - **Mean Max Continuous Dwell:** **0.06s** (PID: 2.86s, Cloned v6: 2.20s)
  - **Visual FOV Retention:** **9.84%**
  - **Mean Minimum Distance:** **5.85m** | **Collisions:** 0
- **Profile Breakdown:**
  - *Cruising (100):* Catch: 1.0% | Dwell: 0.11s | FOV: 8.6%
  - *Evasive (100):* Catch: 0.0% | Dwell: 0.03s | FOV: 8.7%
  - *Hyper-Evasive (100):* Catch: 0.0% | Dwell: 0.04s | FOV: 12.2%
- **Automated Diagnostic Analysis:**
  > Policy collapsed to standoff loitering at 5.85m (FOV: 9.8%, 1s Catch: 0.3%). PPO exploration noise (σ=0.45) caused early visual line-of-sight loss and uncalibrated critic value collapse, washing out the cloned expert prior.
- **Actor Weights:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_01_potential_dwell_baseline/actor_weights.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_01_potential_dwell_baseline/actor_weights.pt)

### <a id="run_02_asymmetric_sprint_huber"></a> `run_02_asymmetric_sprint_huber` (Phase I: Potential Fields)
- **Architecture / Reward Mode:** `theory_02_asymmetric_sprint_huber`
- **Hypothesis:** High linear drag outside basket (-0.75/m) and amplified closing speed reward forces immediate full throttle surge on distant targets.
- **Status:** Completed
- **Training Duration:** 1027.2s (~17.1 min)
- **300 Benchmark Overall Results:**
  - **1.0s Catch Rate:** **0.00%** (PID: 63.0%, Cloned v6: 45.0%)
  - **3.0s Continuous Lock Rate:** **0.00%** (PID: 40.3%, Cloned v6: 32.0%)
  - **Mean Max Continuous Dwell:** **0.07s** (PID: 2.86s, Cloned v6: 2.20s)
  - **Visual FOV Retention:** **11.21%**
  - **Mean Minimum Distance:** **5.74m** | **Collisions:** 0
- **Profile Breakdown:**
  - *Cruising (100):* Catch: 0.0% | Dwell: 0.11s | FOV: 11.7%
  - *Evasive (100):* Catch: 0.0% | Dwell: 0.06s | FOV: 11.4%
  - *Hyper-Evasive (100):* Catch: 0.0% | Dwell: 0.03s | FOV: 10.5%
- **Automated Diagnostic Analysis:**
  > Policy collapsed to standoff loitering at 5.74m (FOV: 11.2%, 1s Catch: 0.0%). PPO exploration noise (σ=0.45) caused early visual line-of-sight loss and uncalibrated critic value collapse, washing out the cloned expert prior.
- **Actor Weights:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_02_asymmetric_sprint_huber/actor_weights.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_02_asymmetric_sprint_huber/actor_weights.pt)

### <a id="run_03_optical_looming_expansion"></a> `run_03_optical_looming_expansion` (Phase I: Potential Fields)
- **Architecture / Reward Mode:** `theory_03_optical_looming_expansion`
- **Hypothesis:** Direct bounding box divergence feedback (filtered_d_w) bypasses Kalman depth estimation lag during initial sprint catch-up.
- **Status:** Completed
- **Training Duration:** 1019.4s (~17.0 min)
- **300 Benchmark Overall Results:**
  - **1.0s Catch Rate:** **0.00%** (PID: 63.0%, Cloned v6: 45.0%)
  - **3.0s Continuous Lock Rate:** **0.00%** (PID: 40.3%, Cloned v6: 32.0%)
  - **Mean Max Continuous Dwell:** **0.03s** (PID: 2.86s, Cloned v6: 2.20s)
  - **Visual FOV Retention:** **6.51%**
  - **Mean Minimum Distance:** **5.35m** | **Collisions:** 2
- **Profile Breakdown:**
  - *Cruising (100):* Catch: 0.0% | Dwell: 0.05s | FOV: 5.4%
  - *Evasive (100):* Catch: 0.0% | Dwell: 0.03s | FOV: 6.2%
  - *Hyper-Evasive (100):* Catch: 0.0% | Dwell: 0.02s | FOV: 8.0%
- **Automated Diagnostic Analysis:**
  > Policy collapsed to standoff loitering at 5.35m (FOV: 6.5%, 1s Catch: 0.0%). PPO exploration noise (σ=0.45) caused early visual line-of-sight loss and uncalibrated critic value collapse, washing out the cloned expert prior.
- **Actor Weights:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_03_optical_looming_expansion/actor_weights.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_03_optical_looming_expansion/actor_weights.pt)

### <a id="run_04_kinematic_glide_slope"></a> `run_04_kinematic_glide_slope` (Phase II: Relative Velocity Matching)
- **Architecture / Reward Mode:** `theory_04_kinematic_glide_slope`
- **Hypothesis:** Bounded parabolic closing velocity envelope (sqrt(2*a*d)) prevents overshooting collisions while allowing 12 m/s sprint at range.
- **Status:** Completed
- **Training Duration:** 1049.6s (~17.5 min)
- **300 Benchmark Overall Results:**
  - **1.0s Catch Rate:** **5.33%** (PID: 63.0%, Cloned v6: 45.0%)
  - **3.0s Continuous Lock Rate:** **0.67%** (PID: 40.3%, Cloned v6: 32.0%)
  - **Mean Max Continuous Dwell:** **0.23s** (PID: 2.86s, Cloned v6: 2.20s)
  - **Visual FOV Retention:** **14.04%**
  - **Mean Minimum Distance:** **4.44m** | **Collisions:** 4
- **Profile Breakdown:**
  - *Cruising (100):* Catch: 11.0% | Dwell: 0.46s | FOV: 13.9%
  - *Evasive (100):* Catch: 4.0% | Dwell: 0.17s | FOV: 14.0%
  - *Hyper-Evasive (100):* Catch: 1.0% | Dwell: 0.05s | FOV: 14.2%
- **Automated Diagnostic Analysis:**
  > Policy collapsed to standoff loitering at 4.44m (FOV: 14.0%, 1s Catch: 5.3%). PPO exploration noise (σ=0.45) caused early visual line-of-sight loss and uncalibrated critic value collapse, washing out the cloned expert prior.
- **Actor Weights:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_04_kinematic_glide_slope/actor_weights.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_04_kinematic_glide_slope/actor_weights.pt)

### <a id="run_05_relative_velocity_nulling"></a> `run_05_relative_velocity_nulling` (Phase II: Relative Velocity Matching)
- **Architecture / Reward Mode:** `theory_05_relative_velocity_nulling`
- **Hypothesis:** Explicit 3D relative velocity matching inside the basket creates tight formation lock, preventing drift during evasive weaves.
- **Status:** Completed
- **Training Duration:** 1020.2s (~17.0 min)
- **300 Benchmark Overall Results:**
  - **1.0s Catch Rate:** **0.00%** (PID: 63.0%, Cloned v6: 45.0%)
  - **3.0s Continuous Lock Rate:** **0.00%** (PID: 40.3%, Cloned v6: 32.0%)
  - **Mean Max Continuous Dwell:** **0.04s** (PID: 2.86s, Cloned v6: 2.20s)
  - **Visual FOV Retention:** **1.10%**
  - **Mean Minimum Distance:** **6.03m** | **Collisions:** 1
- **Profile Breakdown:**
  - *Cruising (100):* Catch: 0.0% | Dwell: 0.04s | FOV: 0.9%
  - *Evasive (100):* Catch: 0.0% | Dwell: 0.05s | FOV: 1.2%
  - *Hyper-Evasive (100):* Catch: 0.0% | Dwell: 0.03s | FOV: 1.2%
- **Automated Diagnostic Analysis:**
  > Policy collapsed to standoff loitering at 6.03m (FOV: 1.1%, 1s Catch: 0.0%). PPO exploration noise (σ=0.45) caused early visual line-of-sight loss and uncalibrated critic value collapse, washing out the cloned expert prior.
- **Actor Weights:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_05_relative_velocity_nulling/actor_weights.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_05_relative_velocity_nulling/actor_weights.pt)

### <a id="run_06_lead_pursuit_vector_alignment"></a> `run_06_lead_pursuit_vector_alignment` (Phase II: Relative Velocity Matching)
- **Architecture / Reward Mode:** `theory_06_lead_pursuit_vector_alignment`
- **Hypothesis:** Aligning velocity vector with target heading cuts turn apexes rather than chasing target exhaust, retaining FOV on sharp cuts.
- **Status:** Completed
- **Training Duration:** 1044.9s (~17.4 min)
- **300 Benchmark Overall Results:**
  - **1.0s Catch Rate:** **0.00%** (PID: 63.0%, Cloned v6: 45.0%)
  - **3.0s Continuous Lock Rate:** **0.00%** (PID: 40.3%, Cloned v6: 32.0%)
  - **Mean Max Continuous Dwell:** **0.06s** (PID: 2.86s, Cloned v6: 2.20s)
  - **Visual FOV Retention:** **2.35%**
  - **Mean Minimum Distance:** **5.83m** | **Collisions:** 2
- **Profile Breakdown:**
  - *Cruising (100):* Catch: 0.0% | Dwell: 0.07s | FOV: 2.2%
  - *Evasive (100):* Catch: 0.0% | Dwell: 0.07s | FOV: 2.5%
  - *Hyper-Evasive (100):* Catch: 0.0% | Dwell: 0.04s | FOV: 2.4%
- **Automated Diagnostic Analysis:**
  > Policy collapsed to standoff loitering at 5.83m (FOV: 2.4%, 1s Catch: 0.0%). PPO exploration noise (σ=0.45) caused early visual line-of-sight loss and uncalibrated critic value collapse, washing out the cloned expert prior.
- **Actor Weights:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_06_lead_pursuit_vector_alignment/actor_weights.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_06_lead_pursuit_vector_alignment/actor_weights.pt)

### <a id="run_07_decel_to_turn_centripetal"></a> `run_07_decel_to_turn_centripetal` (Phase III: Agile Cornering)
- **Architecture / Reward Mode:** `theory_07_decel_to_turn_centripetal`
- **Hypothesis:** Constraining allowable forward speed when bearing error is large (|ex| > 0.20) tightens turn radius and prevents hyper-evasive blow-by.
- **Status:** Completed
- **Training Duration:** 1020.1s (~17.0 min)
- **300 Benchmark Overall Results:**
  - **1.0s Catch Rate:** **1.00%** (PID: 63.0%, Cloned v6: 45.0%)
  - **3.0s Continuous Lock Rate:** **0.00%** (PID: 40.3%, Cloned v6: 32.0%)
  - **Mean Max Continuous Dwell:** **0.06s** (PID: 2.86s, Cloned v6: 2.20s)
  - **Visual FOV Retention:** **2.99%**
  - **Mean Minimum Distance:** **5.16m** | **Collisions:** 2
- **Profile Breakdown:**
  - *Cruising (100):* Catch: 1.0% | Dwell: 0.09s | FOV: 2.6%
  - *Evasive (100):* Catch: 2.0% | Dwell: 0.07s | FOV: 2.7%
  - *Hyper-Evasive (100):* Catch: 0.0% | Dwell: 0.03s | FOV: 3.7%
- **Automated Diagnostic Analysis:**
  > Policy collapsed to standoff loitering at 5.16m (FOV: 3.0%, 1s Catch: 1.0%). PPO exploration noise (σ=0.45) caused early visual line-of-sight loss and uncalibrated critic value collapse, washing out the cloned expert prior.
- **Actor Weights:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_07_decel_to_turn_centripetal/actor_weights.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_07_decel_to_turn_centripetal/actor_weights.pt)

### <a id="run_08_retinal_slip_raptor"></a> `run_08_retinal_slip_raptor` (Phase III: Agile Cornering)
- **Architecture / Reward Mode:** `theory_08_retinal_slip_raptor`
- **Hypothesis:** Penalizing high optical slip rate (LOS angular rate) forces the policy to stabilize boresight before opening forward throttle.
- **Status:** Completed
- **Training Duration:** 1034.0s (~17.2 min)
- **300 Benchmark Overall Results:**
  - **1.0s Catch Rate:** **0.00%** (PID: 63.0%, Cloned v6: 45.0%)
  - **3.0s Continuous Lock Rate:** **0.00%** (PID: 40.3%, Cloned v6: 32.0%)
  - **Mean Max Continuous Dwell:** **0.04s** (PID: 2.86s, Cloned v6: 2.20s)
  - **Visual FOV Retention:** **9.99%**
  - **Mean Minimum Distance:** **5.98m** | **Collisions:** 0
- **Profile Breakdown:**
  - *Cruising (100):* Catch: 0.0% | Dwell: 0.06s | FOV: 7.9%
  - *Evasive (100):* Catch: 0.0% | Dwell: 0.04s | FOV: 9.4%
  - *Hyper-Evasive (100):* Catch: 0.0% | Dwell: 0.03s | FOV: 12.7%
- **Automated Diagnostic Analysis:**
  > Policy collapsed to standoff loitering at 5.98m (FOV: 10.0%, 1s Catch: 0.0%). PPO exploration noise (σ=0.45) caused early visual line-of-sight loss and uncalibrated critic value collapse, washing out the cloned expert prior.
- **Actor Weights:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_08_retinal_slip_raptor/actor_weights.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_08_retinal_slip_raptor/actor_weights.pt)

### <a id="run_09_curvature_adaptive_basket"></a> `run_09_curvature_adaptive_basket` (Phase III: Agile Cornering)
- **Architecture / Reward Mode:** `theory_09_curvature_adaptive_basket`
- **Hypothesis:** Dynamically expanding standoff from 3.5m to 4.2m on violent turns gives wider angular margin to track evasive jinks without losing FOV.
- **Status:** Completed
- **Training Duration:** 1101.3s (~18.4 min)
- **300 Benchmark Overall Results:**
  - **1.0s Catch Rate:** **0.00%** (PID: 63.0%, Cloned v6: 45.0%)
  - **3.0s Continuous Lock Rate:** **0.00%** (PID: 40.3%, Cloned v6: 32.0%)
  - **Mean Max Continuous Dwell:** **0.03s** (PID: 2.86s, Cloned v6: 2.20s)
  - **Visual FOV Retention:** **2.11%**
  - **Mean Minimum Distance:** **6.13m** | **Collisions:** 0
- **Profile Breakdown:**
  - *Cruising (100):* Catch: 0.0% | Dwell: 0.05s | FOV: 2.0%
  - *Evasive (100):* Catch: 0.0% | Dwell: 0.02s | FOV: 1.9%
  - *Hyper-Evasive (100):* Catch: 0.0% | Dwell: 0.03s | FOV: 2.4%
- **Automated Diagnostic Analysis:**
  > Policy collapsed to standoff loitering at 6.13m (FOV: 2.1%, 1s Catch: 0.0%). PPO exploration noise (σ=0.45) caused early visual line-of-sight loss and uncalibrated critic value collapse, washing out the cloned expert prior.
- **Actor Weights:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_09_curvature_adaptive_basket/actor_weights.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_09_curvature_adaptive_basket/actor_weights.pt)

### <a id="run_10_superlinear_dwell_jackpot"></a> `run_10_superlinear_dwell_jackpot` (Phase IV: Synthesis Champions)
- **Architecture / Reward Mode:** `theory_10_superlinear_dwell_jackpot`
- **Hypothesis:** Quadratic dwell progression ((t_dwell / 3.0)^2 * 14.0) heavily penalizes losing lock after 1-2s, locking drone into formation.
- **Status:** Completed
- **Training Duration:** 1019.9s (~17.0 min)
- **300 Benchmark Overall Results:**
  - **1.0s Catch Rate:** **0.00%** (PID: 63.0%, Cloned v6: 45.0%)
  - **3.0s Continuous Lock Rate:** **0.00%** (PID: 40.3%, Cloned v6: 32.0%)
  - **Mean Max Continuous Dwell:** **0.04s** (PID: 2.86s, Cloned v6: 2.20s)
  - **Visual FOV Retention:** **7.42%**
  - **Mean Minimum Distance:** **5.74m** | **Collisions:** 0
- **Profile Breakdown:**
  - *Cruising (100):* Catch: 0.0% | Dwell: 0.06s | FOV: 6.7%
  - *Evasive (100):* Catch: 0.0% | Dwell: 0.04s | FOV: 6.8%
  - *Hyper-Evasive (100):* Catch: 0.0% | Dwell: 0.03s | FOV: 8.7%
- **Automated Diagnostic Analysis:**
  > Policy collapsed to standoff loitering at 5.74m (FOV: 7.4%, 1s Catch: 0.0%). PPO exploration noise (σ=0.45) caused early visual line-of-sight loss and uncalibrated critic value collapse, washing out the cloned expert prior.
- **Actor Weights:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_10_superlinear_dwell_jackpot/actor_weights.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_10_superlinear_dwell_jackpot/actor_weights.pt)

### <a id="run_11_unified_dogfight_synthesis"></a> `run_11_unified_dogfight_synthesis` (Phase IV: Synthesis Champions)
- **Architecture / Reward Mode:** `theory_11_unified_dogfight_synthesis`
- **Hypothesis:** Combines sprint Huber pull + proportional nav lead pursuit + kinematic glide-slope brake + convex dwell.
- **Status:** Completed
- **Training Duration:** 1034.2s (~17.2 min)
- **300 Benchmark Overall Results:**
  - **1.0s Catch Rate:** **0.00%** (PID: 63.0%, Cloned v6: 45.0%)
  - **3.0s Continuous Lock Rate:** **0.00%** (PID: 40.3%, Cloned v6: 32.0%)
  - **Mean Max Continuous Dwell:** **0.05s** (PID: 2.86s, Cloned v6: 2.20s)
  - **Visual FOV Retention:** **10.12%**
  - **Mean Minimum Distance:** **5.44m** | **Collisions:** 1
- **Profile Breakdown:**
  - *Cruising (100):* Catch: 0.0% | Dwell: 0.08s | FOV: 8.7%
  - *Evasive (100):* Catch: 0.0% | Dwell: 0.04s | FOV: 10.2%
  - *Hyper-Evasive (100):* Catch: 0.0% | Dwell: 0.03s | FOV: 11.4%
- **Automated Diagnostic Analysis:**
  > Policy collapsed to standoff loitering at 5.44m (FOV: 10.1%, 1s Catch: 0.0%). PPO exploration noise (σ=0.45) caused early visual line-of-sight loss and uncalibrated critic value collapse, washing out the cloned expert prior.
- **Actor Weights:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_11_unified_dogfight_synthesis/actor_weights.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_11_unified_dogfight_synthesis/actor_weights.pt)

### <a id="run_12_adaptive_grand_champion"></a> `run_12_adaptive_grand_champion` (Phase IV: Synthesis Champions)
- **Architecture / Reward Mode:** `theory_12_adaptive_grand_champion`
- **Hypothesis:** Grand synthesis combining Centripetal Decel-to-Turn + 3D Relative Velocity Nulling + Optical Looming + Superlinear Dwell.
- **Status:** Completed
- **Training Duration:** 1030.2s (~17.2 min)
- **300 Benchmark Overall Results:**
  - **1.0s Catch Rate:** **0.33%** (PID: 63.0%, Cloned v6: 45.0%)
  - **3.0s Continuous Lock Rate:** **0.00%** (PID: 40.3%, Cloned v6: 32.0%)
  - **Mean Max Continuous Dwell:** **0.07s** (PID: 2.86s, Cloned v6: 2.20s)
  - **Visual FOV Retention:** **3.49%**
  - **Mean Minimum Distance:** **4.56m** | **Collisions:** 4
- **Profile Breakdown:**
  - *Cruising (100):* Catch: 0.0% | Dwell: 0.11s | FOV: 3.3%
  - *Evasive (100):* Catch: 1.0% | Dwell: 0.07s | FOV: 3.6%
  - *Hyper-Evasive (100):* Catch: 0.0% | Dwell: 0.03s | FOV: 3.6%
- **Automated Diagnostic Analysis:**
  > Policy collapsed to standoff loitering at 4.56m (FOV: 3.5%, 1s Catch: 0.3%). PPO exploration noise (σ=0.45) caused early visual line-of-sight loss and uncalibrated critic value collapse, washing out the cloned expert prior.
- **Actor Weights:** [`/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_12_adaptive_grand_champion/actor_weights.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_12_campaign/run_12_adaptive_grand_champion/actor_weights.pt)
