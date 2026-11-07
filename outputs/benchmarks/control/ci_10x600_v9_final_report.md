# Controller Comparison with 95% Confidence Intervals (10 x 600 Trajectories: Tuned PID vs Residual Ppo V9 Wide35 Best Model)
**Sample Size**: 10 folds x 600 scenarios = 6,000 trajectories per controller (12,000 total runs)  
**Method**: Paired Student's t 95% Confidence Intervals & Two-Tailed Significance Testing  
**RRL Model**: `best_model.zip` (`residual_ppo_v9_wide35_best_model`)  
**PID Base**: `tuned_pid_gains_pre_attitude_dynamics.json`  

## 1. Overall Aggregate Metrics (6,000 Trajectories)

| Evaluation Metric | Tuned PID Baseline (95% CI) | Residual Ppo V9 Wide35 Best Model (95% CI) | Paired Difference (Δ, 95% CI) | p-value | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2.0s Dwell Catch (%)** | 41.78 [40.18, 43.39] | **54.10 [52.85, 55.35]** | +12.32 [11.10, 13.54] | < 0.001 | **p < 0.001 (***)** |
| **1.0s Dwell Catch (%)** | 51.53 [49.94, 53.12] | **62.45 [61.39, 63.51]** | +10.92 [9.83, 12.00] | < 0.001 | **p < 0.001 (***)** |
| **3.0s Dwell Catch (%)** | 28.98 [27.82, 30.15] | **42.32 [41.33, 43.30]** | +13.33 [12.37, 14.30] | < 0.001 | **p < 0.001 (***)** |
| **Time-to-Success (2.0s Dwell) (s)** | 6.10 [5.87, 6.33] | **8.85 [8.64, 9.07]** | +2.76 [2.40, 3.11] | < 0.001 | **p < 0.001 (***)** |
| **Mean Max Dwell (s)** | 1.91 [1.84, 1.97] | **2.78 [2.72, 2.84]** | +0.87 [0.83, 0.91] | < 0.001 | **p < 0.001 (***)** |
| **FOV Retention (%)** | 52.90 [51.68, 54.12] | **63.22 [62.38, 64.06]** | +10.32 [9.81, 10.83] | < 0.001 | **p < 0.001 (***)** |
| **Yaw Jerk (deg/s^2)** | 37.93 [37.01, 38.84] | **35.76 [35.08, 36.45]** | -2.16 [-2.86, -1.47] | < 0.001 | **p < 0.001 (***)** |
| **Surge Jerk (m/s^2)** | 2.06 [2.02, 2.10] | **2.09 [2.06, 2.11]** | +0.03 [0.01, 0.05] | 0.0084 | **p < 0.01 (**)** |
| **Collision Breaches (%)** | 1.03 [0.70, 1.36] | **1.13 [0.82, 1.45]** | +0.10 [-0.23, 0.43] | 0.5146 | n.s. |

## 2. Profile Breakdown (2.0s Dwell Catch Success Rate)

| Target Profile | Tuned PID Baseline (95% CI) | Residual Ppo V9 Wide35 Best Model (95% CI) | Paired Difference (Δ, 95% CI) | p-value |
| :--- | :---: | :---: | :---: | :---: |
| **Cruising** (2,000 runs) | 64.2% [62.2%, 66.1%] | **73.2% [71.6%, 74.9%]** | +9.1% [7.1%, 11.1%] | < 0.001 |
| **Evasive** (2,000 runs) | 47.1% [44.2%, 50.1%] | **62.8% [60.8%, 64.7%]** | +15.6% [13.4%, 17.8%] | < 0.001 |
| **Hyper_evasive** (2,000 runs) | 14.1% [12.4%, 15.7%] | **26.3% [24.5%, 28.1%]** | +12.2% [10.2%, 14.3%] | < 0.001 |

## 3. Profile Breakdown (Time-to-Success for 2.0s Dwell, including 2s dwell)

| Target Profile | Tuned PID Baseline (95% CI) | Residual Ppo V9 Wide35 Best Model (95% CI) | Paired Difference (Δ, 95% CI) | p-value |
| :--- | :---: | :---: | :---: | :---: |
| **Cruising** | 5.53s [5.15s, 5.91s] | **7.06s [6.84s, 7.28s]** | +1.53s [1.14s, 1.92s] | < 0.001 |
| **Evasive** | 6.36s [6.09s, 6.64s] | **9.23s [8.79s, 9.68s]** | +2.87s [2.36s, 3.38s] | < 0.001 |
| **Hyper_evasive** | 7.72s [6.97s, 8.48s] | **12.92s [12.59s, 13.24s]** | +5.19s [4.26s, 6.12s] | < 0.001 |
