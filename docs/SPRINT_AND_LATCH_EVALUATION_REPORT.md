# Sprint & Latch Recurrent Guidance Evaluation Report

**Date:** 2026-10-23 10:48:34
**Pipeline:** RecurrentPPO fine-tuned from `cloned_expert_v1` with Sprint & Latch Reward Shaping
**Training Timesteps:** 1,000,000 steps (18.4 minutes on 8 vectorized environments)
**Reward Mode:** `expressive_iterative` with active differential sprint closing potential
**Model Weights:** [`control/weights/sprint_and_latch_v1/best_model/actor_weights.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/sprint_and_latch_v1/best_model/actor_weights.pt)

---

## 1. Comparative Validation Suite Benchmark

| Architecture | Timesteps | Hyper FOV% | Mean Dist | Closest Approach | Speed Range | Max Continuous Dwell | Success Rate (≥3.0s) | Collisions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Cloned Expert Baseline | 0 | 41.1% | 21.8 m | 1.12 m | 3.4 – 7.9 m/s | 0.26 s | 0.0% | 0 |
| Overnight Run 01 (Pure Bucket Blast) | 1.5M | 89.9% | 51.3 m | 3.05 m | 0.3 – 4.0 m/s | 0.20 s | 0.0% | 0 |
| Overnight Run 03 (Bucket Magnet) | 1.5M | 14.6% | 22.4 m | 1.72 m | 4.0 – 6.0 m/s | 0.14 s | 0.0% | 0 |
| Overnight Run 14 (Peak Lock Pursuit) | 1.5M | 5.0% | 47.2 m | 2.87 m | 0.0 – 4.1 m/s | 0.26 s | 0.0% | 0 |
| **Sprint & Latch (Current)** | **1.0M** | **15.3%** | **50.15 m** | **2.91 m** | **3.1 – 4.5 m/s** | **0.14 s** | **0.0%** | **0** |

---

## 2. Key Synthesis Architectural Elements

1. **Active Closing Sprint Potential ($w_{\text{sprint}} = 3.5$):** Directly rewards closing speed $\dot{d}_{\text{close}}$ when outside the bucket ($d > 4.5\text{m}$), eliminating the low-speed telescope trap.
2. **Reverse-Exponential Arrival Jackpot ($R_0 = 140.0, \tau = 3.5\text{s}$):** Awarded once upon first bucket entry, strongly incentivizing an immediate arrival sprint.
3. **Speed Matching & Braking Inside Bucket:** Transitions from sprint drive into Gaussian velocity matching ($2.0 \cdot \exp(-0.5 (\dot{d}/1.0)^2)$), helping the drone match speed and avoid overshooting.
4. **Dense Escalating Bucket Blast ($B = 25.0, S = 10.0$):** Massive reward acceleration inside the $2.5\text{--}4.5\text{m}$ formation basket.
5. **Zero Collision Penalty ($P_{\text{collision}} = 0.0$):** Zero punishment for drone contact, preventing cautious hang-back behaviour.
6. **Proportional Navigation Lead Guidance ($w_{\text{pn}} = 3.5$):** Anticipates evasive flight paths to slice corner apexes.