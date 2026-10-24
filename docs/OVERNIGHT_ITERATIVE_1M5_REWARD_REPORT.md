# Overnight 15-Run Iterative Reward Shaping Campaign Report

**Date:** 2026-10-23 08:08:34
**Campaign Type:** Autonomous Iterative RL Loop (`/goal`) with Warm-Started RecurrentPPO
**Warm-Start Baseline:** Cloned Expert (`control/weights/cloned_expert_v1/actor_weights.pt`)
**Timesteps per Candidate:** 1,500,000 steps (~27.5 minutes on 8 vectorized environments)
**Total Completed Runs:** 15 / 15
**Total Campaign Runtime:** 6.29 hours (377.2 mins)

### User Reward Objectives & Design Constraints:
1. **Fast-Arrival Jackpot:** Reverse-exponential arrival reward $R_{\text{arrival}} \cdot \exp(-t_{\text{arrival}} / \tau)$ awarded once upon first bucket entry ($2.5\text{ m} \le d \le 4.5\text{ m}$).
2. **Continuous Distance Pull:** Peak reward at $3.5\text{ m}$ pulling chaser smoothly inward.
3. **Reward Blast in Bucket:** Dense rewards ($B_{\text{base}} + S_{\text{dwell}} \cdot t^2$) while inside bucket.
4. **No Cautious Hang-Back:** Collision penalty deliberately kept mild ($P_{\text{collision}} = 0.0$ to $-1.0$) so agent never becomes overly cautious or hesitates to close.
5. **Hold 3 Consecutive Seconds:** Interception lock terminal jackpot ($+120$ to $+220$) awarded at 3.0s unbroken dwell.

---

## 1. Candidate Leaderboard (Standardized 30-Scenario Validation Suite)

| Rank | Run Name | Hyper FOV% | Mean Dist | Min Dist | Turn Speed | Straight Speed | Continuous Dwell | Success Rate (≥3.0s) | Collisions |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `run_14_peak_lock_pursuit` | **5.0%** | 47.16 m | 2.87 m | 0.0 m/s | 4.1 m/s | **0.26 s** | **0.0%** | 0 |
| **2** | `run_01_pure_bucket_blast` | **89.9%** | 51.34 m | 3.05 m | 0.3 m/s | 4.0 m/s | **0.20 s** | **0.0%** | 0 |
| **3** | `run_13_high_velocity_match` | **24.6%** | 44.47 m | 2.62 m | 0.2 m/s | 4.4 m/s | **0.16 s** | **0.0%** | 0 |
| **4** | `run_12_adaptive_refinement_2` | **18.7%** | 46.69 m | 2.99 m | 2.0 m/s | 4.6 m/s | **0.16 s** | **0.0%** | 0 |
| **5** | `run_05_proportional_nav_apex` | **50.7%** | 42.34 m | 3.13 m | 3.5 m/s | 4.7 m/s | **0.14 s** | **0.0%** | 0 |
| **6** | `run_07_crescendo_dwell_escalator` | **17.8%** | 49.74 m | 3.43 m | 2.1 m/s | 4.0 m/s | **0.14 s** | **0.0%** | 0 |
| **7** | `run_03_massive_bucket_magnet` | **14.6%** | 22.40 m | 1.72 m | 4.0 m/s | 6.0 m/s | **0.14 s** | **0.0%** | 0 |
| **8** | `run_04_feather_cushion_tap` | **4.8%** | 45.51 m | 2.01 m | 0.2 m/s | 4.3 m/s | **0.14 s** | **0.0%** | 0 |
| **9** | `run_08_deep_basket_08m_clearance` | **2.1%** | 50.72 m | 2.37 m | 1.7 m/s | 5.2 m/s | **0.14 s** | **0.0%** | 0 |
| **10** | `run_11_adaptive_refinement_1` | **15.6%** | 39.47 m | 2.62 m | 0.1 m/s | 4.0 m/s | **0.12 s** | **0.0%** | 0 |
| **11** | `run_15_grand_champion_convergence` | **17.6%** | 47.33 m | 3.10 m | 1.6 m/s | 7.3 m/s | **0.00 s** | **0.0%** | 0 |
| **12** | `run_09_hyper_evasive_dogfighter` | **16.5%** | 47.00 m | 3.09 m | 1.8 m/s | 9.0 m/s | **0.00 s** | **0.0%** | 0 |
| **13** | `run_10_balanced_elite_synthesis` | **15.5%** | 48.18 m | 2.42 m | 2.2 m/s | 8.1 m/s | **0.00 s** | **0.0%** | 0 |
| **14** | `run_02_turbo_sprint_arrival` | **13.4%** | 45.56 m | 3.23 m | 1.1 m/s | 6.8 m/s | **0.00 s** | **0.0%** | 0 |
| **15** | `run_06_steep_distance_suction` | **3.3%** | 75.91 m | 3.33 m | 3.8 m/s | 11.5 m/s | **0.00 s** | **0.0%** | 0 |

---

## 2. Iteration Chronology & Telemetry Breakdown

### `run_01_pure_bucket_blast` (Iteration 1)
- **Hypothesis:** Baseline Expressive: Zero collision penalty, R_arr=120, tau=4.5, blast_base=15.0, dwell_scale=6.0
- **Reward Parameters:**
  - Arrival Jackpot: $R_0 = 120.0$, $\tau = 4.5\text{ s}$
  - Continuous Pull: $\text{Peak} = 8.0$ at $3.5\text{ m}$, $\text{Pull} = 0.35$
  - Bucket Blast: $\text{Base} = 15.0$, $\text{Dwell Scale} = 6.0$
  - Collision Safety: $P_{\text{collision}} = 0.0$, Clearance Radius $= 1.0\text{ m}$ (Non-punitive; no cautious hang-back)
  - Proportional Nav Lead: $w_{\text{pn}} = 2.0$
- **Validation Performance:**
  - Hyper FOV Retention: **89.9%**
  - Standoff Distance: Mean=51.34m, Min=3.05m
  - Peak Continuous Dwell: **0.20 s** (Success Rate: **0.0%**)
  - Speed Envelope: 0.3 m/s (turns) $\longleftrightarrow$ 4.0 m/s (straights)
  - Collisions: 0
- **Checkpoint Path:** `/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_expressive/run_01_pure_bucket_blast/best_model/actor_weights.pt`

### `run_02_turbo_sprint_arrival` (Iteration 2)
- **Hypothesis:** Fast-Arrival Turbo: tau=3.5, R_arr=160.0 to aggressively penalize sluggish closing times
- **Reward Parameters:**
  - Arrival Jackpot: $R_0 = 184.0$, $\tau = 3.15\text{ s}$
  - Continuous Pull: $\text{Peak} = 8.0$ at $3.5\text{ m}$, $\text{Pull} = 0.6$
  - Bucket Blast: $\text{Base} = 20.0$, $\text{Dwell Scale} = 8.0$
  - Collision Safety: $P_{\text{collision}} = 0.0$, Clearance Radius $= 1.0\text{ m}$ (Non-punitive; no cautious hang-back)
  - Proportional Nav Lead: $w_{\text{pn}} = 2.5$
- **Validation Performance:**
  - Hyper FOV Retention: **13.4%**
  - Standoff Distance: Mean=45.56m, Min=3.23m
  - Peak Continuous Dwell: **0.00 s** (Success Rate: **0.0%**)
  - Speed Envelope: 1.1 m/s (turns) $\longleftrightarrow$ 6.8 m/s (straights)
  - Collisions: 0
- **Checkpoint Path:** `/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_expressive/run_02_turbo_sprint_arrival/best_model/actor_weights.pt`

### `run_03_massive_bucket_magnet` (Iteration 3)
- **Hypothesis:** Bucket Magnet: blast_base=25.0, dwell_scale=10.0 to lock chaser like glue once inside 2.5-4.5m
- **Reward Parameters:**
  - Arrival Jackpot: $R_0 = 149.5$, $\tau = 3.6\text{ s}$
  - Continuous Pull: $\text{Peak} = 8.0$ at $3.5\text{ m}$, $\text{Pull} = 0.44999999999999996$
  - Bucket Blast: $\text{Base} = 30.0$, $\text{Dwell Scale} = 12.0$
  - Collision Safety: $P_{\text{collision}} = 0.0$, Clearance Radius $= 1.0\text{ m}$ (Non-punitive; no cautious hang-back)
  - Proportional Nav Lead: $w_{\text{pn}} = 3.0$
- **Validation Performance:**
  - Hyper FOV Retention: **14.6%**
  - Standoff Distance: Mean=22.40m, Min=1.72m
  - Peak Continuous Dwell: **0.14 s** (Success Rate: **0.0%**)
  - Speed Envelope: 4.0 m/s (turns) $\longleftrightarrow$ 6.0 m/s (straights)
  - Collisions: 0
- **Checkpoint Path:** `/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_expressive/run_03_massive_bucket_magnet/best_model/actor_weights.pt`

### `run_04_feather_cushion_tap` (Iteration 4)
- **Hypothesis:** Feather Cushion: p_collision=-1.0, collision_dist=0.9m to test gentle tap vs overshoot
- **Reward Parameters:**
  - Arrival Jackpot: $R_0 = 161.0$, $\tau = 3.6\text{ s}$
  - Continuous Pull: $\text{Peak} = 8.0$ at $3.5\text{ m}$, $\text{Pull} = 0.5$
  - Bucket Blast: $\text{Base} = 25.0$, $\text{Dwell Scale} = 10.0$
  - Collision Safety: $P_{\text{collision}} = 0.0$, Clearance Radius $= 0.9\text{ m}$ (Non-punitive; no cautious hang-back)
  - Proportional Nav Lead: $w_{\text{pn}} = 2.5$
- **Validation Performance:**
  - Hyper FOV Retention: **4.8%**
  - Standoff Distance: Mean=45.51m, Min=2.01m
  - Peak Continuous Dwell: **0.14 s** (Success Rate: **0.0%**)
  - Speed Envelope: 0.2 m/s (turns) $\longleftrightarrow$ 4.3 m/s (straights)
  - Collisions: 0
- **Checkpoint Path:** `/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_expressive/run_04_feather_cushion_tap/best_model/actor_weights.pt`

### `run_05_proportional_nav_apex` (Iteration 5)
- **Hypothesis:** High PN Lead (w_pn=4.0): Antennas velocity vector to slice turns and maintain lock on evasive bursts
- **Reward Parameters:**
  - Arrival Jackpot: $R_0 = 161.0$, $\tau = 3.6\text{ s}$
  - Continuous Pull: $\text{Peak} = 8.0$ at $3.5\text{ m}$, $\text{Pull} = 0.5$
  - Bucket Blast: $\text{Base} = 25.0$, $\text{Dwell Scale} = 10.0$
  - Collision Safety: $P_{\text{collision}} = 0.0$, Clearance Radius $= 1.0\text{ m}$ (Non-punitive; no cautious hang-back)
  - Proportional Nav Lead: $w_{\text{pn}} = 4.5$
- **Validation Performance:**
  - Hyper FOV Retention: **50.7%**
  - Standoff Distance: Mean=42.34m, Min=3.13m
  - Peak Continuous Dwell: **0.14 s** (Success Rate: **0.0%**)
  - Speed Envelope: 3.5 m/s (turns) $\longleftrightarrow$ 4.7 m/s (straights)
  - Collisions: 0
- **Checkpoint Path:** `/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_expressive/run_05_proportional_nav_apex/best_model/actor_weights.pt`

### `run_06_steep_distance_suction` (Iteration 6)
- **Hypothesis:** Steep Suction: dist_pull=0.65, dist_peak=10.0 to create a steep potential funnel toward 3.5m
- **Reward Parameters:**
  - Arrival Jackpot: $R_0 = 161.0$, $\tau = 3.6\text{ s}$
  - Continuous Pull: $\text{Peak} = 10.0$ at $3.5\text{ m}$, $\text{Pull} = 0.75$
  - Bucket Blast: $\text{Base} = 25.0$, $\text{Dwell Scale} = 10.0$
  - Collision Safety: $P_{\text{collision}} = 0.0$, Clearance Radius $= 1.0\text{ m}$ (Non-punitive; no cautious hang-back)
  - Proportional Nav Lead: $w_{\text{pn}} = 3.0$
- **Validation Performance:**
  - Hyper FOV Retention: **3.3%**
  - Standoff Distance: Mean=75.91m, Min=3.33m
  - Peak Continuous Dwell: **0.00 s** (Success Rate: **0.0%**)
  - Speed Envelope: 3.8 m/s (turns) $\longleftrightarrow$ 11.5 m/s (straights)
  - Collisions: 0
- **Checkpoint Path:** `/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_expressive/run_06_steep_distance_suction/best_model/actor_weights.pt`

### `run_07_crescendo_dwell_escalator` (Iteration 7)
- **Hypothesis:** Crescendo Dwell: blast_base=30.0, dwell_scale=12.0 for exponential reward ramp towards 3.0s mark
- **Reward Parameters:**
  - Arrival Jackpot: $R_0 = 149.5$, $\tau = 3.7800000000000002\text{ s}$
  - Continuous Pull: $\text{Peak} = 8.0$ at $3.5\text{ m}$, $\text{Pull} = 0.5$
  - Bucket Blast: $\text{Base} = 35.0$, $\text{Dwell Scale} = 14.0$
  - Collision Safety: $P_{\text{collision}} = 0.0$, Clearance Radius $= 1.0\text{ m}$ (Non-punitive; no cautious hang-back)
  - Proportional Nav Lead: $w_{\text{pn}} = 3.0$
- **Validation Performance:**
  - Hyper FOV Retention: **17.8%**
  - Standoff Distance: Mean=49.74m, Min=3.43m
  - Peak Continuous Dwell: **0.14 s** (Success Rate: **0.0%**)
  - Speed Envelope: 2.1 m/s (turns) $\longleftrightarrow$ 4.0 m/s (straights)
  - Collisions: 0
- **Checkpoint Path:** `/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_expressive/run_07_crescendo_dwell_escalator/best_model/actor_weights.pt`

### `run_08_deep_basket_08m_clearance` (Iteration 8)
- **Hypothesis:** Deep Clearance: collision_dist=0.8m giving 1.7m deep operating pocket inside bucket
- **Reward Parameters:**
  - Arrival Jackpot: $R_0 = 161.0$, $\tau = 3.42\text{ s}$
  - Continuous Pull: $\text{Peak} = 9.0$ at $3.5\text{ m}$, $\text{Pull} = 0.55$
  - Bucket Blast: $\text{Base} = 27.0$, $\text{Dwell Scale} = 11.0$
  - Collision Safety: $P_{\text{collision}} = 0.0$, Clearance Radius $= 0.8\text{ m}$ (Non-punitive; no cautious hang-back)
  - Proportional Nav Lead: $w_{\text{pn}} = 3.5$
- **Validation Performance:**
  - Hyper FOV Retention: **2.1%**
  - Standoff Distance: Mean=50.72m, Min=2.37m
  - Peak Continuous Dwell: **0.14 s** (Success Rate: **0.0%**)
  - Speed Envelope: 1.7 m/s (turns) $\longleftrightarrow$ 5.2 m/s (straights)
  - Collisions: 0
- **Checkpoint Path:** `/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_expressive/run_08_deep_basket_08m_clearance/best_model/actor_weights.pt`

### `run_09_hyper_evasive_dogfighter` (Iteration 9)
- **Hypothesis:** Hyper-Evasive Master: 55% hyper training, high PN lead + massive dwell bonus
- **Reward Parameters:**
  - Arrival Jackpot: $R_0 = 172.5$, $\tau = 3.24\text{ s}$
  - Continuous Pull: $\text{Peak} = 9.0$ at $3.5\text{ m}$, $\text{Pull} = 0.6$
  - Bucket Blast: $\text{Base} = 30.0$, $\text{Dwell Scale} = 12.0$
  - Collision Safety: $P_{\text{collision}} = 0.0$, Clearance Radius $= 0.9\text{ m}$ (Non-punitive; no cautious hang-back)
  - Proportional Nav Lead: $w_{\text{pn}} = 4.0$
- **Validation Performance:**
  - Hyper FOV Retention: **16.5%**
  - Standoff Distance: Mean=47.00m, Min=3.09m
  - Peak Continuous Dwell: **0.00 s** (Success Rate: **0.0%**)
  - Speed Envelope: 1.8 m/s (turns) $\longleftrightarrow$ 9.0 m/s (straights)
  - Collisions: 0
- **Checkpoint Path:** `/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_expressive/run_09_hyper_evasive_dogfighter/best_model/actor_weights.pt`

### `run_10_balanced_elite_synthesis` (Iteration 10)
- **Hypothesis:** Balanced Elite Synthesis combining top characteristics of runs 1-9
- **Reward Parameters:**
  - Arrival Jackpot: $R_0 = 161.0$, $\tau = 3.42\text{ s}$
  - Continuous Pull: $\text{Peak} = 9.0$ at $3.5\text{ m}$, $\text{Pull} = 0.55$
  - Bucket Blast: $\text{Base} = 29.0$, $\text{Dwell Scale} = 11.0$
  - Collision Safety: $P_{\text{collision}} = 0.0$, Clearance Radius $= 0.9\text{ m}$ (Non-punitive; no cautious hang-back)
  - Proportional Nav Lead: $w_{\text{pn}} = 3.5$
- **Validation Performance:**
  - Hyper FOV Retention: **15.5%**
  - Standoff Distance: Mean=48.18m, Min=2.42m
  - Peak Continuous Dwell: **0.00 s** (Success Rate: **0.0%**)
  - Speed Envelope: 2.2 m/s (turns) $\longleftrightarrow$ 8.1 m/s (straights)
  - Collisions: 0
- **Checkpoint Path:** `/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_expressive/run_10_balanced_elite_synthesis/best_model/actor_weights.pt`

### `run_11_adaptive_refinement_1` (Iteration 11)
- **Hypothesis:** Adaptive tuning based on telemetry from runs 1-10
- **Reward Parameters:**
  - Arrival Jackpot: $R_0 = 172.5$, $\tau = 3.15\text{ s}$
  - Continuous Pull: $\text{Peak} = 9.0$ at $3.5\text{ m}$, $\text{Pull} = 0.6$
  - Bucket Blast: $\text{Base} = 31.0$, $\text{Dwell Scale} = 12.0$
  - Collision Safety: $P_{\text{collision}} = 0.0$, Clearance Radius $= 0.9\text{ m}$ (Non-punitive; no cautious hang-back)
  - Proportional Nav Lead: $w_{\text{pn}} = 4.0$
- **Validation Performance:**
  - Hyper FOV Retention: **15.6%**
  - Standoff Distance: Mean=39.47m, Min=2.62m
  - Peak Continuous Dwell: **0.12 s** (Success Rate: **0.0%**)
  - Speed Envelope: 0.1 m/s (turns) $\longleftrightarrow$ 4.0 m/s (straights)
  - Collisions: 0
- **Checkpoint Path:** `/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_expressive/run_11_adaptive_refinement_1/best_model/actor_weights.pt`

### `run_12_adaptive_refinement_2` (Iteration 12)
- **Hypothesis:** Adaptive tuning focused on maximizing 3.0s continuous lock completion
- **Reward Parameters:**
  - Arrival Jackpot: $R_0 = 172.5$, $\tau = 3.15\text{ s}$
  - Continuous Pull: $\text{Peak} = 10.0$ at $3.5\text{ m}$, $\text{Pull} = 0.65$
  - Bucket Blast: $\text{Base} = 33.0$, $\text{Dwell Scale} = 13.0$
  - Collision Safety: $P_{\text{collision}} = 0.0$, Clearance Radius $= 0.85\text{ m}$ (Non-punitive; no cautious hang-back)
  - Proportional Nav Lead: $w_{\text{pn}} = 4.0$
- **Validation Performance:**
  - Hyper FOV Retention: **18.7%**
  - Standoff Distance: Mean=46.69m, Min=2.99m
  - Peak Continuous Dwell: **0.16 s** (Success Rate: **0.0%**)
  - Speed Envelope: 2.0 m/s (turns) $\longleftrightarrow$ 4.6 m/s (straights)
  - Collisions: 0
- **Checkpoint Path:** `/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_expressive/run_12_adaptive_refinement_2/best_model/actor_weights.pt`

### `run_13_high_velocity_match` (Iteration 13)
- **Hypothesis:** High Velocity Lead & Dynamic Decel Tuning
- **Reward Parameters:**
  - Arrival Jackpot: $R_0 = 184.0$, $\tau = 2.8800000000000003\text{ s}$
  - Continuous Pull: $\text{Peak} = 10.0$ at $3.5\text{ m}$, $\text{Pull} = 0.7$
  - Bucket Blast: $\text{Base} = 35.0$, $\text{Dwell Scale} = 14.0$
  - Collision Safety: $P_{\text{collision}} = 0.0$, Clearance Radius $= 0.85\text{ m}$ (Non-punitive; no cautious hang-back)
  - Proportional Nav Lead: $w_{\text{pn}} = 4.5$
- **Validation Performance:**
  - Hyper FOV Retention: **24.6%**
  - Standoff Distance: Mean=44.47m, Min=2.62m
  - Peak Continuous Dwell: **0.16 s** (Success Rate: **0.0%**)
  - Speed Envelope: 0.2 m/s (turns) $\longleftrightarrow$ 4.4 m/s (straights)
  - Collisions: 0
- **Checkpoint Path:** `/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_expressive/run_13_high_velocity_match/best_model/actor_weights.pt`

### `run_14_peak_lock_pursuit` (Iteration 14)
- **Hypothesis:** Peak formation hold tuning with wide basket dwell acceleration
- **Reward Parameters:**
  - Arrival Jackpot: $R_0 = 172.5$, $\tau = 3.15\text{ s}$
  - Continuous Pull: $\text{Peak} = 9.5$ at $3.5\text{ m}$, $\text{Pull} = 0.65$
  - Bucket Blast: $\text{Base} = 37.0$, $\text{Dwell Scale} = 16.0$
  - Collision Safety: $P_{\text{collision}} = 0.0$, Clearance Radius $= 0.85\text{ m}$ (Non-punitive; no cautious hang-back)
  - Proportional Nav Lead: $w_{\text{pn}} = 4.3$
- **Validation Performance:**
  - Hyper FOV Retention: **5.0%**
  - Standoff Distance: Mean=47.16m, Min=2.87m
  - Peak Continuous Dwell: **0.26 s** (Success Rate: **0.0%**)
  - Speed Envelope: 0.0 m/s (turns) $\longleftrightarrow$ 4.1 m/s (straights)
  - Collisions: 0
- **Checkpoint Path:** `/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_expressive/run_14_peak_lock_pursuit/best_model/actor_weights.pt`

### `run_15_grand_champion_convergence` (Iteration 15)
- **Hypothesis:** Final convergence run unifying all optimal parameters from campaign
- **Reward Parameters:**
  - Arrival Jackpot: $R_0 = 184.0$, $\tau = 3.06\text{ s}$
  - Continuous Pull: $\text{Peak} = 10.0$ at $3.5\text{ m}$, $\text{Pull} = 0.65$
  - Bucket Blast: $\text{Base} = 40.0$, $\text{Dwell Scale} = 17.0$
  - Collision Safety: $P_{\text{collision}} = 0.0$, Clearance Radius $= 0.85\text{ m}$ (Non-punitive; no cautious hang-back)
  - Proportional Nav Lead: $w_{\text{pn}} = 4.5$
- **Validation Performance:**
  - Hyper FOV Retention: **17.6%**
  - Standoff Distance: Mean=47.33m, Min=3.10m
  - Peak Continuous Dwell: **0.00 s** (Success Rate: **0.0%**)
  - Speed Envelope: 1.6 m/s (turns) $\longleftrightarrow$ 7.3 m/s (straights)
  - Collisions: 0
- **Checkpoint Path:** `/Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_expressive/run_15_grand_champion_convergence/best_model/actor_weights.pt`

---

## 3. Morning Analysis & Next Actionable Recommendations

**Best Model to Date:** `run_14_peak_lock_pursuit`
- **Success Rate:** **0.0%**
- **Max Continuous Formation Dwell:** **0.26 s**
- **Mean Distance to Target:** **47.16 m**
- **Hyper-Evasive FOV Retention:** **5.0%**

To deploy or benchmark the best policy interactively:
```bash
.venv/bin/python control/predictive_controller.py --weights /Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/overnight_expressive/run_14_peak_lock_pursuit/best_model/actor_weights.pt
```