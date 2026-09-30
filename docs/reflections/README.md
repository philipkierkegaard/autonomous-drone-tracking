# Engineering & Research Reflections

This directory serves as the persistent log for design decisions, theoretical investigations, and methodological frameworks throughout the DTU Bachelor Thesis (**Autonomous Drone Pursuit & Interception System**).

## Why Keep Structured Reflections?
In academic engineering projects, writing down the rationale behind design choices prevents redundant work, documents failures and breakthroughs, and forms the direct foundation for the **Methodology**, **Architecture**, and **Discussion** chapters of the thesis report.

---

## Index of Reflections

| ID | Date | Title | Key Components / Systems | Status |
| :---: | :---: | :--- | :--- | :---: |
| **01** | `2026-09-13` | [Kalman Filter Evaluation, Covariance Tuning & Foveal Zoom Ablation](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/docs/reflections/01_kalman_tuning_and_foveal_zoom_ablation.md) | `track_pipeline.py`, `KalmanBoxTracker`, Anti-UAV Dataset | **Adopted** |
| **02** | `2026-09-28` | [Guidance Formulation, Recurrent Policy Design (GRU vs. MLP/PID), and Lead-Pursuit Reward Shaping](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/docs/reflections/02_reinforcement_learning_guidance_and_recurrent_policy_design.md) | `control/`, `predictive_controller.py`, `FastPixhawkQuadSim`, RL (PPO) | **Adopted** |
| **03** | `2026-09-30` | [Iterative Reward Shaping, Failure Mode Diagnosis, and Continuous Potential Guidance](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/docs/reflections/03_reward_shaping_and_policy_evolution.md) | `control/environment.py`, `control/train_rl.py`, Recurrent PPO, Visual Servoing | **Active / In Progress** |

---

## Suggested Reflection Document Template

When adding a new reflection, use the following structure:
1. **Context & Motivation**: What triggered the reflection or problem?
2. **Core Research / Design Questions**: Explicit questions being evaluated.
3. **Theoretical & Architectural Analysis**: The mathematical, physical, or algorithmic rationale.
4. **Engineering Trade-offs & Risks**: Real-world constraints (e.g., compute on Jetson, sensor noise, wind disturbances).
5. **Experimental Protocol / Verification Plan**: How this decision will be validated with data.
6. **Action Items & Thesis Placement**: Concrete next steps and where this belongs in the final manuscript.
