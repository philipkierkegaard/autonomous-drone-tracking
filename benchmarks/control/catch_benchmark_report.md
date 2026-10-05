# Empirical Evaluation: Autonomous Drone Pursuit & Interception Benchmark Suite

**Date:** October 1, 2026  
**Project:** DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System  
**Dataset:** [`outputs/benchmarks/datasets/benchmark_suite_v1.json`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/outputs/benchmarks/datasets/benchmark_suite_v1.json) (120 Standardized Scenarios, 50 Hz, 60ms Latency)  
**Evaluation Script:** [`benchmarks/control/run_catch_benchmark.py`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/benchmarks/control/run_catch_benchmark.py)  
**Raw Results:** [`outputs/benchmarks/control/catch_benchmark_results.json`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/outputs/benchmarks/control/catch_benchmark_results.json)  

---

## 1. Metric Hierarchy & Academic Rationale

In vision-based aerial pursuit, optical field-of-view (In-FOV %) retention is necessary but insufficient: an agile chaser could maintain visual contact while trailing 20 meters away or circling uncontrollably. 

To rigorously quantify tactical mission effectiveness, the evaluation metrics are structured into a strict hierarchy:

### Primary Measures (Tactical Interception & Catching)
1. **Catch Success Rate (%):** The percentage of pursuit sorties where the chaser successfully closes distance to the target envelope ($5.0\text{m} \le d \le 7.0\text{m}$, centered on $6.0\text{m}$ nominal standoff) and **sustains continuous lock in-view for at least 3.0 consecutive seconds**.
2. **Mean Time-to-Catch ($t_{\text{catch}}$, seconds):** The average elapsed flight time required to reach the target standoff envelope that establishes the 3.0-second sustained lock.

### Secondary Measures (Aerospace & Visual Servoing Quality)
- **Visual In-FOV Retention Rate (%):** Percentage of total episode duration where the target is projected within the camera sensor frustum.
- **Standoff Distance RMSE (meters):** Root-mean-square error relative to the desired $6.0\text{m}$ standoff distance.
- **Cumulative Basket Dwell Time ($t_{\text{basket}}$, seconds):** Total non-continuous time spent inside the $5.0\text{m} - 7.0\text{m}$ firing window.
- **Actuator Smoothness / Yaw Jerk ($\dot{\omega}_z$, $^\circ/\text{s}^2$):** Rate of change of yaw rate commands, indicating absence of control chatter.
- **Physical Collision Breaches ($d < 1.2\text{m}$):** Critical flight safety infractions.

---

## 2. Macro Performance Telemetry (120 Standardized Scenarios)

| Controller Architecture | Policy Description | Primary: Catch Success (Strict 3s) | Primary: Mean Time-to-Catch | Secondary: In-FOV Retention | Secondary: Standoff RMSE | Secondary: Basket Dwell | Secondary: Yaw Jerk | Collisions ($d < 1.2\text{m}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical PID Baseline** | Kinematic IBVS (6.0m Goal) | 45.8% (55/120) | 8.97s (±5.78s) | 75.2% | 3.76m | 6.42s | 24.8°/s² | 2 (1.7%) |
| **Generation 3 RL** | Geometric Rear Trail Anchor | 40.8% (49/120) | 8.98s (±4.70s) | 79.3% | 8.52m | 5.39s | **23.4°/s²** | 2 (1.7%) |
| **Generation 4 RL** | Multi-Scale Gaussian Potential | 48.3% (58/120) | **8.34s** (±4.70s) | **82.5%** | 8.72m | **6.58s** | 26.6°/s² | **1 (0.8%)** |
| **Generation 5B RL** | 13D Filtered Optical Rates | 28.3% (34/120) | 10.55s (±5.03s) | 68.2% | 10.09m | 4.49s | 34.6°/s² | **1 (0.8%)** |
| **Generation 5C RL** | Calibrated Dwell + Symmetric | **49.2% (59/120)** | 9.13s (±4.83s) | 79.4% | 8.71m | 6.35s | 27.7°/s² | 2 (1.7%) |

> [!NOTE]
> Under a 1-frame grace tolerance (leaky decay, accounting for momentary camera occultation), the catch rates are: **Classical PID: 46.7%**, **Gen 3: 40.8%**, **Gen 4: 48.3%**, **Gen 5B: 30.0%**, and **Gen 5C: 50.8%**.

---

## 3. Stratified Flight Profile Breakdown (40 Sorties Each)

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

  Hyper-Evasive (8.5 m/s):
  PID:       [░░░░░░░░░░░░░░░░░░░░]  0.0%  (TTC:   N/A  | Dwell:  0.27s)
  Gen 4 RL:  [░░░░░░░░░░░░░░░░░░░░]  0.0%  (TTC:   N/A  | Dwell:  0.23s)
  Gen 5C RL: [█░░░░░░░░░░░░░░░░░░░]  2.5%  (TTC: 14.36s | Dwell:  0.70s)  <-- First successful lock
```

### Detailed Metric Breakdown Table

| Maneuver Profile | Metric Evaluated | Classical PID | Gen 3 RL | Gen 4 RL | Gen 5B RL | Gen 5C RL | Advantage / Analysis |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cruising**<br>*(Gentle arcs, 3.5 m/s)* | **Catch Success %**<br>Mean Time-to-Catch<br>In-FOV Retention<br>Basket Dwell Time | **97.5%**<br>**7.33s**<br>83.8%<br>13.35s | 80.0%<br>7.84s<br>75.3%<br>11.65s | 90.0%<br>7.77s<br>**88.3%**<br>**13.77s** | 17.5%<br>12.91s<br>67.6%<br>4.49s | 62.5%<br>9.46s<br>81.6%<br>7.48s | PID excels at low-speed rectilinear pursuit due to zero non-linear dynamics. |
| **Evasive**<br>*(High-G hooks, 6.0 m/s)* | **Catch Success %**<br>Mean Time-to-Catch<br>In-FOV Retention<br>Basket Dwell Time | 40.0%<br>12.97s<br>78.3%<br>5.65s | 42.5%<br>11.11s<br>78.3%<br>4.36s | 55.0%<br>9.29s<br>**84.9%**<br>5.75s | 60.0%<br>9.10s<br>68.6%<br>7.59s | **82.5%**<br>**8.72s**<br>81.2%<br>**10.89s** | **Gen 5C is overwhelmingly dominant**: +42.5% higher catch rate and 4.25s faster interception than PID. |
| **Hyper-Evasive**<br>*(3D corkscrews, 8.5 m/s)* | **Catch Success %**<br>Mean Time-to-Catch<br>In-FOV Retention<br>Basket Dwell Time | 0.0%<br>N/A<br>63.5%<br>0.27s | 0.0%<br>N/A<br>**84.3%**<br>0.17s | 0.0%<br>N/A<br>74.3%<br>0.23s | **7.5%**<br>16.57s<br>68.4%<br>**1.36s** | 2.5%<br>**14.36s**<br>75.4%<br>0.70s | Only the 13D optical-rate policies (Gen 5B/5C) manage to sustain 3s locks against extreme 8.5 m/s maneuvers. |

---

## 4. Key Academic & Engineering Takeaways

1. **Why In-FOV % Masked Controller Limitations:**
   - Gen 3 RL achieved 79.3% in-FOV retention, but suffered from the "cruise plateau" at 9.7m, yielding only 40.8% catch success.
   - Focusing on **Catch Success Rate** and **Time-to-Catch** unmasks whether the drone actually closed distance and held tactical formation.
2. **Speed of Interception (Time-to-Catch):**
   - Gen 4 RL achieved the fastest overall average Time-to-Catch (**8.34s**) across all successful flights, closing from the initial 12m–15m offset quickly.
   - On aggressive, evasive targets, Gen 5C reached the catch envelope in **8.72s** compared to **12.97s for PID**—a **32.8% reduction in interception time**.
3. **The Evasive Breakthrough:**
   - Classical PID reacted purely to instantaneous pixel errors, overshooting corners due to 60ms camera transport delay.
   - Gen 5C's GRU recurrence with 13D optical rates $[\dot{e}_x, \dot{e}_y, \dot{w}]$ allowed it to predict target curvature, leading to **82.5% Catch Success** and **10.89s of basket dwell** (vs 5.65s for PID).
