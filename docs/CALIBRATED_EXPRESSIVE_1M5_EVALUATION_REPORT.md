# Calibrated Expressive Guidance Evaluation Report (1.5M Steps)

**Date:** 2026-10-25 00:51:58
**Pipeline:** RecurrentPPO fine-tuned from `cloned_expert_v1` with Calibrated Expressive Reward Shaping
**Training Timesteps:** 1,500,000 steps (28.4 minutes on 8 vectorized environments)
**Reward Mode:** `expressive` (0.5s dwell-gated arrival jackpot, 3.5m center Gaussian well, 0.5m collision, FOV loss penalty)
**Model Weights:** [`control/weights/calibrated_expressive_1m5/best_model/actor_weights.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/weights/calibrated_expressive_1m5/best_model/actor_weights.pt)

---

## 1. Comparative Validation Suite Benchmark

| Architecture | Timesteps | Hyper FOV% | Mean Dist | Closest Approach | Speed Range | Max Continuous Dwell | Success Rate (≥3.0s) | Collisions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Cloned Expert Baseline | 0 | 41.1% | 21.8 m | 1.12 m | 3.4 – 7.9 m/s | 0.26 s | 0.0% | 0 |
| Overnight Run 01 (Pure Bucket Blast) | 1.5M | 89.9% | 51.3 m | 3.05 m | 0.3 – 4.0 m/s | 0.20 s | 0.0% | 0 |
| Overnight Run 03 (Bucket Magnet) | 1.5M | 14.6% | 22.4 m | 1.72 m | 4.0 – 6.0 m/s | 0.14 s | 0.0% | 0 |
| Overnight Run 14 (Peak Lock Pursuit) | 1.5M | 5.0% | 47.2 m | 2.87 m | 0.0 – 4.1 m/s | 0.26 s | 0.0% | 0 |
| Sprint & Latch (Ungated) | 1.0M | 15.3% | 50.15 m | 2.91 m | 3.1 – 4.5 m/s | 0.14 s | 0.0% | 0 |
| **Calibrated Expressive (Current)** | **1.5M** | **72.7%** | **15.81 m** | **1.15 m** | **2.9 – 10.2 m/s** | **0.56 s** | **0.0%** | **0** |

---

## 2. Key Synthesis Architectural Elements

1. **Continuous Distance Pull Peaking at 3.5m ($R_{\text{dist}} = 8.0 \cdot \exp(-0.5 ((d - 3.5)/1.0)^2)$):** Drives chaser toward the 3.5m bucket center.
2. **Active Sprint Closing Potential ($w_{\text{sprint}} = 3.5$):** Prevents the $1/d^2$ pinhole vanishing gradient trap at long distances (> 4.5m).
3. **0.5s Dwell-Gated Fast-Arrival Jackpot ($R_0 = 140.0, \tau = 3.5\text{s}$):** Awarded once after holding continuous lock for 0.5s, eliminating the bullet fly-by exploit.
4. **Dense Bucket Bombardment Scaled to Center ($B = 6.0 \cdot \exp(-0.5 ((d - 3.5)/0.6)^2) + 2.0 \cdot t_{\text{dwell}}$):** Floods the agent with dense rewards inside the bucket without blowing up PPO advantage variance.
5. **0.5m Collision Clearance Radius ($P_{\text{collision}} = -20.0$):** Expands the braking cushion inside the bucket from 1.3m to 2.0m.
6. **Sightline FOV Loss Step Penalty ($P_{\text{lost}} = -1.5$):** Explicit negative reward while blind, training aggressive re-acquisition.
7. **Proportional Navigation Lead Guidance ($w_{\text{pn}} = 2.5$):** Aligns velocity vectors ($\vec{v}_c \cdot \vec{v}_t$) for corner apex anticipation.