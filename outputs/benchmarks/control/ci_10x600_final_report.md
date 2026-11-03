# Controller Comparison with 95% Confidence Intervals (10 x 600 Trajectories)
**Sample Size**: 10 folds x 600 scenarios = 6,000 trajectories per controller (12,000 total runs)  
**Method**: Paired Student's t 95% Confidence Intervals & Two-Tailed Significance Testing  

## 1. Overall Aggregate Metrics (6,000 Trajectories)

| Evaluation Metric | Tuned PID Baseline (95% CI) | Residual RL v6 (95% CI) | Paired Difference (Δ, 95% CI) | p-value | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2.0s Dwell Catch (%)** | 41.78 [40.18, 43.39] | **44.20 [42.74, 45.66]** | +2.42 [1.05, 3.78] | 0.0030 | **p < 0.01 (**)** |
| **1.0s Dwell Catch (%)** | 51.53 [49.94, 53.12] | **52.40 [50.99, 53.81]** | +0.87 [-0.52, 2.25] | 0.1904 | n.s. |
| **3.0s Dwell Catch (%)** | 28.98 [27.82, 30.15] | **35.47 [33.70, 37.23]** | +6.48 [5.50, 7.46] | < 0.001 | **p < 0.001 (***)** |
| **Time-to-Success (2.0s Dwell) (s)** | 8.10 [7.87, 8.33] | **7.25 [7.08, 7.42]** | -0.84 [-1.11, -0.57] | < 0.001 | **p < 0.001 (***)** |
| **Mean Max Dwell (s)** | 1.91 [1.84, 1.97] | **2.38 [2.32, 2.44]** | +0.47 [0.44, 0.51] | < 0.001 | **p < 0.001 (***)** |
| **FOV Retention (%)** | 52.90 [51.68, 54.12] | **45.72 [44.37, 47.07]** | -7.18 [-7.98, -6.38] | < 0.001 | **p < 0.001 (***)** |
| **Yaw Jerk (deg/s^2)** | 37.93 [37.01, 38.84] | **41.11 [39.95, 42.28]** | +3.19 [2.44, 3.93] | < 0.001 | **p < 0.001 (***)** |
| **Surge Jerk (m/s^2)** | 2.06 [2.02, 2.10] | **1.96 [1.92, 2.00]** | -0.10 [-0.12, -0.07] | < 0.001 | **p < 0.001 (***)** |
| **Collision Breaches (%)** | 1.03 [0.70, 1.36] | **2.18 [1.81, 2.56]** | +1.15 [0.75, 1.55] | < 0.001 | **p < 0.001 (***)** |

## 2. Profile Breakdown (2.0s Dwell Catch Success Rate)

| Target Profile | Tuned PID Baseline (95% CI) | Residual RL v6 (95% CI) | Paired Difference (Δ, 95% CI) | p-value |
| :--- | :---: | :---: | :---: | :---: |
| **Cruising** (2,000 runs) | 64.2% [62.2%, 66.1%] | **63.7% [61.7%, 65.7%]** | -0.5% [-2.8%, 1.9%] | 0.6729 |
| **Evasive** (2,000 runs) | 47.1% [44.2%, 50.1%] | **48.5% [46.0%, 51.1%]** | +1.4% [-0.0%, 2.8%] | 0.0566 |
| **Hyper_evasive** (2,000 runs) | 14.1% [12.4%, 15.7%] | **20.4% [19.1%, 21.6%]** | +6.3% [5.0%, 7.6%] | < 0.001 |

## 3. Profile Breakdown (Time-to-Success for 2.0s Dwell, including 2s dwell)

| Target Profile | Tuned PID Baseline (95% CI) | Residual RL v6 (95% CI) | Paired Difference (Δ, 95% CI) | p-value |
| :--- | :---: | :---: | :---: | :---: |
| **Cruising** | 7.53s [7.15s, 7.91s] | **6.31s [6.11s, 6.51s]** | -1.22s [-1.60s, -0.84s] | < 0.001 |
| **Evasive** | 8.36s [8.09s, 8.64s] | **7.63s [7.24s, 8.03s]** | -0.73s [-1.03s, -0.43s] | < 0.001 |
| **Hyper_evasive** | 9.72s [8.97s, 10.48s] | **9.27s [8.75s, 9.79s]** | -0.45s [-1.41s, 0.51s] | 0.3131 |
