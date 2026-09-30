# Research Reflection 02: Guidance Formulation, Recurrent Policy Design (GRU vs. MLP/PID), and Lead-Pursuit Reward Shaping

* **Date:** September 28, 2026  
* **Project:** DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System  
* **Subsystem:** Advanced Guidance & Control (`control/`, `predictive_controller.py`, `dataset_generator.py`)  
* **Milestone:** Track A, Week 3–4 Transition (Simulated Closed-Loop Intercept & Learned Guidance Policy)

---

## 1. Context & Motivation

Having established a functional baseline controller—an Image-Based Visual Servoing (IBVS) PID pipeline stabilized with an 8D Kalman filter—the project schedule calls for an **Advanced Guidance Policy** in simulation. 

Classical visual servoing operates reactively on pixel errors. While functional in simple tail-chase scenarios, it suffers substantial performance degradation against evasive targets, high-speed crossing paths, and frame dropouts. 

To train an autonomous neural policy using **Reinforcement Learning (RL)**, several foundational architectural, algorithmic, and guidance questions must be resolved:
1. *What is the appropriate neural network architecture to handle partial observability without exploding computational budgets on the Jetson Orin Nano?*
2. *Should the policy reward dead-center camera tracking ($e_x = 0, e_y = 0$), or does strict centering actively impair interception efficiency?*
3. *How should the reward function balance range closure, engagement basket retention, velocity matching, control smoothness, and flight safety?*
4. *How should encounter datasets and simulation episodes be structured (cohesive long flights vs. edge-case slices, stopping criteria)?*

---

## 2. Guidance Theory: Pure Pursuit vs. Lead Pursuit

A central breakthrough in this reflection is recognizing the distinction between **Pure Pursuit** (classical PID) and **Lead Pursuit / Proportional Navigation** (optimal interceptor guidance).

### 2.1 The Classic Guidance Trap (Pure Pursuit)
In classical visual PID, the control law treats the camera center as a target setpoint, enforcing:
$$e_x \rightarrow 0, \quad e_y \rightarrow 0$$

```
Target Flight Path ───────────────► (Crossing Perpendicular at 10 m/s)
                  \              /
   Pure Pursuit    \            /   Lead Pursuit (RL Optimal)
  (Forced Center)   \          /    (Target allowed off-center in FOV)
   High-curvature    \        /     Straight, energy-optimal line
   "dog-leg" curve    \      /
                       \    /
                     Chaser Drone
```

* **The Problem on Crossing Targets:** When an evasive target crosses perpendicularly at high speed, steering the chaser directly at the target's *current location* forces the drone onto a curved "dog-leg" trajectory. The chaser is perpetually chasing where the target *was*, demanding massive lateral accelerations and leading to overshoot.
* **The Interceptor Solution (Lead Pursuit):** An optimal interceptor steers toward where the target *will be* (the collision/interception point). In lead pursuit, **the target intentionally sits off-center in the camera frame** while the chaser flies an energy-optimal, straight path. 
* **Key Takeaway:** As long as the target remains within the camera's Field-of-View (FOV), the detector continues to generate valid bounding boxes. **Forcing dead-center tracking is unnecessary and mathematically sub-optimal.**

---

### 2.2 The Quadrotor Pitch-Tilt Paradox

Multirotor dynamics create a direct mechanical conflict between forward acceleration and camera orientation:

1. **Underactuated Propulsion:** A quadrotor must tilt its nose down ($\theta < 0$, typically $20^\circ \text{--} 30^\circ$ at high acceleration) to generate forward thrust ($F_x = T \sin\theta$).
2. **Rigid Camera Coupling:** Although the camera possesses a fixed $+15^\circ$ mechanical up-tilt, pitching down by $25^\circ$ tilts the optical axis $10^\circ$ below horizontal.
3. **The Penalty Conflict:** If the RL reward penalizes vertical pixel error ($e_y \neq 0$), the agent is actively punished whenever it tilts forward to accelerate!
4. **Resolution:** Decoupling the reward from dead-center tracking enables the policy to tilt aggressively, sprint toward the target while allowing the target to rise toward the top of the FOV, and level out upon reaching the engagement basket.

---

## 3. Frustum Slack Cone & Reward Function Formulation

Instead of a point-attractor reward pulling the target to $(0, 0)$, we treat the camera FOV as a **Safe Slack Basket**:
* **Inside the Inner Cone ($|e| \le 0.70$):** Zero penalty. The agent has complete freedom to exploit lead angles and aggressive pitch angles.
* **Near Frustum Boundaries ($|e| > 0.70$):** A soft quadratic barrier repels the state away from the edge to prevent target loss.

```
┌───────────────────────────────────────────────┐
│ Camera Sensor (640x480)                       │
│   ┌───────────────────────────────────────┐   │
│   │ Soft Boundary Repeller (|e| > 0.70)   │   │
│   │   ┌───────────────────────────────┐   │   │
│   │   │ Free Zone (Zero Penalty)      │   │   │
│   │   │ Optimal lead-angles permitted │   │   │
│   │   │                               │   │   │
│   │   └───────────────────────────────┘   │   │
│   └───────────────────────────────────────┘   │
└───────────────────────────────────────────────┘
```

### 3.1 Total Step Reward Function ($R_t$)

At each 50 Hz control step, the scalar reward is:

$$R_t = R_{\text{closure}} + R_{\text{basket}} + R_{\text{velocity\_matching}} - P_{\text{boundary}} - P_{\text{jerk}}$$

#### 1. Range Closure ($R_{\text{closure}}$)
A bounded hyperbolic tangent potential pushing the drone toward the standoff distance $d^* = 6.0\text{ m}$:
$$R_{\text{closure}} = - w_{\text{close}} \cdot \tanh\left( \frac{|d_t - d^*|}{d_{\text{scale}}} \right)$$
* *Parameters:* $w_{\text{close}} = 0.5$, $d^* = 6.0\text{ m}$, $d_{\text{scale}} = 4.0\text{ m}$.
* Yields smooth, non-saturating negative gradients during the long-range approach phase.

#### 2. Firing Basket Reward ($R_{\text{basket}}$)
The primary mission objective—maximizing time spent in a verified firing solution:
$$R_{\text{basket}} = \begin{cases} 
+1.0 + 0.5 \cdot \max\left(0, 1 - \frac{e_x^2 + e_y^2}{0.5^2}\right) & \text{if } 5.0\text{m} \le d_t \le 7.0\text{m} \text{ and Target in FOV} \\
0.0 & \text{otherwise}
\end{cases}$$
* Yields $+1.0$ per step ($+50\text{ pts/second}$ at 50 Hz), incentivizing rapid acquisition and continuous standoff lock.

#### 3. Relative Velocity Matching ($R_{\text{velocity\_matching}}$)
Penalizes relative velocity to eliminate overshoot, activated only near the engagement basket:
$$R_{\text{velocity\_matching}} = - w_{\text{vel}} \cdot \|\mathbf{v}_{\text{chaser}} - \mathbf{v}_{\text{target}}\|^2 \cdot \exp\left( - \frac{(d_t - d^*)^2}{2 \sigma_{\text{vel}}^2} \right)$$
* *Parameters:* $w_{\text{vel}} = 0.05$, $\sigma_{\text{vel}} = 3.0\text{ m}$.
* At long range ($d_t > 15\text{m}$), the Gaussian is near zero, allowing full-speed pursuit. Near $6.0\text{m}$, it forces velocity synchronization.

#### 4. FOV Soft Boundary Repeller ($P_{\text{boundary}}$)
$$P_{\text{boundary}} = w_{\text{bound}} \cdot \left[ \max(0, |e_x| - 0.70)^2 + \max(0, |e_y| - 0.70)^2 \right] + P_{\text{out\_of\_fov}}$$
$$P_{\text{out\_of\_fov}} = \begin{cases} 1.0 & \text{if Target outside FOV} \\ 0.0 & \text{if Target inside FOV} \end{cases}$$
* *Parameters:* $w_{\text{bound}} = 2.0$.

#### 5. Anti-Jerk Smoothness ($P_{\text{jerk}}$)
$$P_{\text{jerk}} = w_{\text{jerk}} \cdot \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2$$
* *Parameters:* $w_{\text{jerk}} = 0.05$, with action setpoints $\mathbf{a}_t \in [-1, 1]^4$.

---

### 3.2 Terminal Costs & Episode Stopping Criteria

| Condition | Flag | Penalty | Rationale |
| :--- | :---: | :---: | :--- |
| **Collision Breach** ($d_t < 2.5\text{m}$) | `terminated` | **$-30.0$** | Prevents suicidal kamikaze behaviors. |
| **Lost Target Timeout** ($T_{\text{lost}} > 2.0\text{s}$) | `terminated` | **$-20.0$** | Punishes losing track beyond recovery horizon. |
| **Ground Strike** ($z_{\text{chaser}} < 1.0\text{m}$) | `terminated` | **$-50.0$** | Floor envelope safety. |
| **Mission Horizon Reached** ($t = 25.0\text{s}$) | `truncated` | **$0.0$** | Successful full engagement run. |

* **Coasting Grace Period:** Terminating instantly when the target clips the FOV edge prevents the policy from learning recovery maneuvers. A $2.0\text{s}$ grace period (100 frames at 50 Hz) gives the recurrent hidden state the opportunity to coast and steer back onto the target track.

---

## 4. Policy Architecture: Why GRU Over Vanilla RNN & LSTM

### 4.1 The Partially Observable Problem (POMDP)
Visual servoing from bounding boxes is fundamentally partially observable: a single bounding box $[x_c, y_c, w, h]$ contains position information, but zero relative velocity. Temporal state memory is required to infer velocity, filter jitter, and bridge occlusions.

### 4.2 Model Evaluation & Comparison

| Architecture | Strengths in Control RL | Shortcomings & Failure Modes | Recommendation |
| :--- | :--- | :--- | :---: |
| **Vanilla RNN (Elman)** | Simple formulation. | Severe vanishing/exploding gradients; truncated BPTT in RL is notoriously unstable. | ❌ Reject |
| **LSTM** | Excellent long-horizon memory. | Two separate hidden states ($h_t, c_t$), higher parameter count, slower inference on embedded hardware. | 🟡 Acceptable |
| **GRU (Gated Recurrent Unit)** | Single hidden state $h_t$; resets/updates gated dynamically; 25–30% faster than LSTM. | Requires recurrent rollout buffers (chunked BPTT). | 🟢 **Selected** |
| **Frame-Stacked MLP ($K=5$)** | Extremely simple feedforward training; no recurrent state management. | Hard fixed-horizon cutoff; cannot retain memory during prolonged frame drops. | 🟢 **Benchmark Baseline** |

### 4.3 Embedded Jetson Orin Nano Feasibility
* A 1-layer GRU with hidden dimension $H = 64$ fed with a 9D observation vector requires only $\sim 14,000$ parameters ($< 60\text{ KB}$ footprint).
* Forward inference latency is **$< 0.2\text{ ms}$** in PyTorch CPU/GPU, consuming less than $1\%$ of the 50 Hz ($20\text{ ms}$) loop budget on the companion computer.

---

## 5. The Thesis 3-Way Comparative Matrix

This reflection solidifies the primary comparative experimental campaign for the thesis:

```
┌────────────────────────────────────────────────────────┐
│                   Classical Baseline                   │
│          Kalman Filter + Visual PID Controller         │
│          (Explicit Analytical State Estimation)        │
└───────────────────────────┬────────────────────────────┘
                            │
            Benchmark Evaluated Against Same Test Suite
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│     Fixed-Memory Policy       │   │    Adaptive Recurrent Policy  │
│   Stacked MLP Policy (K=5)    │   │      Recurrent PPO (GRU)      │
│  (Sliding Window Observation) │   │     (Latent Internal State)   │
└───────────────────────────────┘   └───────────────────────────────┘
```

### Core Research Questions Addressed:
1. *Does an end-to-end recurrent policy eliminate the need for hand-tuned Kalman filters in visual drone interception?*
2. *Does lead pursuit discovered by RL provide quantifiable energy savings and faster time-to-standoff compared to classical pure pursuit PID?*
3. *How robust is the GRU hidden state during severe perception dropouts (0%, 5%, 10%, 20% dropped frames)?*

---

## 6. Implementation Action Plan

1. **Gymnasium Environment (`control/environment.py`):**
   * Wrap `FastPixhawkQuadSim` and `PursuitDatasetGenerator` into `DronePursuitEnv`.
   * Implement the 9D observation space, 4D continuous action space, and the slack-cone reward function.
2. **RL Training Pipeline (`control/train_rl.py`):**
   * Configure `RecurrentPPO` using `sb3-contrib` with vectorized parallel environments.
   * Log reward components, firing basket retention %, and encounter success rates in TensorBoard.
3. **Comparative Evaluation (`benchmarks/control/benchmark_controller.py`):**
   * Extend the benchmarking suite to load and evaluate all three controllers (PID, Stacked MLP, GRU) across the exact same standardized encounter scenarios.
