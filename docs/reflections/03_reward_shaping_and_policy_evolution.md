# Research Reflection 03: Iterative Reward Shaping, Failure Mode Diagnosis, and Continuous Potential Guidance

* **Date:** September 30, 2026  
* **Project:** DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System  
* **Subsystem:** Reinforcement Learning Guidance & Reward Engineering (`control/environment.py`, `control/train_rl.py`)  
* **Milestone:** Track A, Advanced Guidance & Control (Closing the Gap to Classical Visual Servoing)

---

## 1. Executive Summary & Academic Framing

In complex robotics reinforcement learning, the scalar reward function is the sole communication channel between the designer's intent and the optimization algorithm. Naive reward formulations frequently lead to **reward hacking** or **mode collapse**, where the agent maximizes cumulative returns through bizarre, physically undesirable, or destructive behaviors.

This reflection documents the iterative engineering progression of the Recurrent PPO guidance policy across **six distinct design iterations**:
1. **Generation 1 (Isotropic Standoff Shell)**: Discovered the orbital / circling exploit.
2. **Generation 2 (Over-Regularized Scratch)**: Diagnosed exploration entropy collapse induced by action rate penalties.
3. **Generation 3 (Geometric Rear Trail Anchor)**: Formulated the wake anchor $\mathbf{p}_{\text{trail}}$, eliminating circling and achieving yaw smoothness parity with classical PID.
4. **Generation 4 (Continuous Multi-Scale Gaussian Potential Field)**: Resolving the cruise plateau to achieve sub-decimeter standoff precision.
5. **Generation 5A (Transition Bounties & Isotropic Regression)**: Diagnosed the "Lighthouse Radar" spinning exploit where raw transition rewards (+25) induced periodic 120°/s cycling.
6. **Generation 5B (Filtered Optical Rates & Calibrated Potential Field)**: Re-centering on the rear wake potential with 13D causal optical rates and warm-start initialization.

---

## 2. Chronological Policy Evolution & Diagnostics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Generation 1: Isotropic 5-7m Standoff Shell (recurrent_ppo_29_09)           │
│ • Formula: r_basket = +8.0 if 5.0m <= ||p_chaser - p_target|| <= 7.0m       │
│ • Artifact: Orbiting / Circling around the target at 6m.                    │
│ • Root Cause: Spherical symmetry with zero aspect-angle preference, coupled │
│   with a 7.5 m/s forward stick bias against 3.5 m/s targets.                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Generation 2: Heavy Jerk Regularization from Scratch (4M Scratch Run)       │
│ • Formula: p_jerk = 0.20 * ||Δa||^2 applied directly to random actions      │
│ • Artifact: Action standard deviation collapsed (σ -> 0.05), policy frozen. │
│ • Root Cause: High penalty applied to initial exploration noise penalized    │
│   any change in action, strangling the policy gradient before discovery.    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Generation 3: Geometric Rear Trail Anchor (tail_chase_finetune)             │
│ • Formula: p_trail = p_target - 6*h_target; basket if d_trail <= 1.8m       │
│ • Result: Circling 100% eliminated; yaw jerk matched PID (23.4 deg/s²).     │
│ • Artifact: Standoff hung back at ~9.7m on cruising flights (RMSE 8.52m).   │
│ • Root Cause: Zero distance gradient outside the 1.8m basket once speed is  │
│   matched (plateau flaw). Collecting +2.5/step at 9m is risk-free.          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Generation 4: Continuous Multi-Scale Gaussian Potential Field               │
│ • Formula: r_trail = 3.0*exp(-d_trail^2/(2*3.5^2)) + 6.0*exp(-d_trail^2/    │
│   (2*1.2^2)); optical scale w* calibrated to 0.0505 (exact 6.0m).           │
│ • Result: Plateau solved; surpassed classical PID (48.3% vs 46.7% lock).    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Generation 5A: Transition Bounties & Isotropic Regression (recurrent_ppo_g5)│
│ • Formula: r_search = 0.5*yaw/120, r_reacquire = +25.0, r_dist = f(|d-6|).  │
│ • Artifact: "Lighthouse Radar" spinning exploit (120 deg/s), 24 crashes.    │
│ • Root Cause: Raw transition bounties violate policy invariance (Ng 1999);  │
│   agent maximized return (1,830) by intentionally cycling out and in.       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Generation 5B: Filtered Optical Rates + Pure Continuous Rear Potential      │
│ • Formula: 13D obs [d_ex, d_ey, d_w], restored d_trail Gaussian potential,  │
│   zeroed transition bounties, restored 3s terminal lock dwell milestone.    │
│ • Goal: Break through the 55% evasive ceiling without spinning artifacts.    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Deep Dive into Each Generation

### Generation 1: The Isotropic Standoff Shell (The Circling Exploit)

#### Mathematical Formulation
$$R_t = R_{\text{vis}} + R_{\text{basket}} - P_{\text{prox}} - P_{\text{alt}} - P_{\text{jerk}}$$
Where:
$$R_{\text{basket}} = \begin{cases} +8.0 + 2.0 \cdot \max(0, 1 - \frac{e_x^2 + e_y^2}{0.5^2}) & \text{if } 5.0\text{m} \le ||\mathbf{p}_{\text{chaser}} - \mathbf{p}_{\text{target}}|| \le 7.0\text{m} \text{ and in FOV} \\ 0.0 & \text{otherwise} \end{cases}$$

#### Observed Pathology
The trained policy successfully acquired the target, but upon reaching $6\text{m}$, it began **flying continuous high-speed circles around the target drone**.

#### Algorithmic & Physical Root Cause
1. **Spherical Isotropic Symmetry**: The reward function measured Euclidean distance to the target center. Being directly behind the target at $6\text{m}$ yielded $+10.0$; flying perpendicular to the target at $6\text{m}$ yielded the exact same $+10.0$.
2. **Speed Asymmetry**: The forward stick mapping was $v_x = 7.5 + 10.5 \cdot a_0\text{ m/s}$. Neutral stick ($a_0 = 0$) commanded $7.5\text{ m/s}$ ($27\text{ km/h}$). Target cruising speed was only $\sim 3.5\text{ m/s}$. The chaser could not hover at 6m behind without holding heavy reverse stick; orbiting allowed it to maintain its forward aerodynamic speed while keeping the target at 6m.

---

### Generation 2: The Over-Regularization Trap (Entropy Collapse)

#### Mathematical Formulation
To eliminate control chatter, an aggressive action rate (jerk) penalty was introduced:
$$P_{\text{jerk}} = 0.20 \cdot \sum_{i=0}^3 (a_{t, i} - a_{t-1, i})^2$$

#### Observed Pathology
When trained from scratch, the policy completely froze into a fixed forward stick ($a_0 = +0.22, a_1 = 0.0, a_2 = 0.0, a_3 = 0.0$), showing zero learning over 4,000,000 timesteps.

#### Algorithmic Root Cause
At initialization, PPO explores using a Gaussian action distribution with variance $\sigma \approx 0.60$. With random Gaussian noise, $E[(a_t - a_{t-1})^2] = 2\sigma^2 \approx 0.72$. The agent incurred an automatic **$-0.60$ penalty per step purely from exploring**. PPO minimized this penalty by collapsing the exploration variance $\sigma \to 0.05$, freezing the actor before it could discover the target basket.

**Takeaway:** Jerk penalties must be calibrated to $p_{\text{jerk}} \le 0.02$, or introduced via fine-tuning/curriculum after initial policy discovery.

---

### Generation 3: The Geometric Rear Trail Anchor (Solved Circling)

#### Mathematical Formulation
1. **Target Heading Vector**:
   $$\hat{\mathbf{h}}_{\text{target}} = \frac{\mathbf{v}_{\text{target}, xy}}{||\mathbf{v}_{\text{target}, xy}||}$$
2. **Rear Trail Anchor**:
   $$\mathbf{p}_{\text{trail}} = \mathbf{p}_{\text{target}} - d^* \cdot \hat{\mathbf{h}}_{\text{target}} \quad (d^* = 6.0\text{ m})$$
3. **Trail Basket Error**:
   $$d_{\text{trail}} = ||\mathbf{p}_{\text{chaser}} - \mathbf{p}_{\text{trail}}||$$
   $$\text{in\_basket} = (d_{\text{trail}} \le 1.8\text{ m}) \land \text{raw\_in\_view}$$
4. **Action Mapping Re-centering**:
   $$v_x = 5.0 + 10.0 \cdot a_0 \quad ([-5.0\text{ m/s}, +15.0\text{ m/s}])$$

#### Results & Quantitative Verification
- **Circling was 100% eliminated**: The drone settled directly into the wake of the target.
- **Flight Smoothness Parity**: Yaw jerk dropped from $108.5^\circ/\text{s}^2$ to **$23.4^\circ/\text{s}^2$**, matching classical PID ($23.9^\circ/\text{s}^2$).
- **Head-to-Head Benchmark on 120 Scenarios**:
  - Success Rate: PID **$45.8\%$** vs RL **$40.8\%$**
  - Evasive Profile Success: **$42.5\%$ vs $42.5\%$ (Tie)**
  - Hyper-Evasive In-FOV: RL **$84.3\%$** vs PID **$63.4\%$** (+20.9% visual retention advantage).

#### Remaining Limitation: The Cruise Plateau
On cruising flights, the agent averaged $9.70\text{m}$ standoff instead of $6.0\text{m}$ (RMSE $5.80\text{m}$). Why? Outside the $1.8\text{m}$ basket, the distance reward was $0.0$. Once the chaser matched speed at $9\text{m}$, closing rate was $0.0$, so it collected $+2.5/\text{step}$ visual reward with zero gradient pushing it forward.

---

### Generation 4: Continuous Multi-Scale Gaussian Potential Field (The Precision Fix)

#### 1. Multi-Scale Continuous Potential Function
Instead of zero reward outside a discrete threshold, the agent experiences a continuous, everywhere-differentiable potential slope:

$$R_{\text{trail}}(d_{\text{trail}}) = R_{\text{wide}} \cdot \exp\left( -\frac{d_{\text{trail}}^2}{2 \sigma_{\text{wide}}^2} \right) + R_{\text{tight}} \cdot \exp\left( -\frac{d_{\text{trail}}^2}{2 \sigma_{\text{tight}}^2} \right)$$

* **Wide Guiding Slope**: $R_{\text{wide}} = 3.0$, $\sigma_{\text{wide}} = 3.5\text{ m}$. Pulls the drone continuously from $15\text{m} \to 9\text{m} \to 6\text{m}$.
* **Tight Precision Well**: $R_{\text{tight}} = 6.0$, $\sigma_{\text{tight}} = 1.2\text{m}$. Rewards precise sub-meter trail basket retention.
* **Continuous Reward Values**:
  - At $d_{\text{trail}} = 0.0\text{m}$ ($6.0\text{m}$ exact lock): $+9.00/\text{step}$
  - At $d_{\text{trail}} = 1.5\text{m}$ ($7.5\text{m}$ standoff): $+5.74/\text{step}$
  - At $d_{\text{trail}} = 3.0\text{m}$ ($9.0\text{m}$ standoff): $+2.08/\text{step}$
  - At $d_{\text{trail}} = 6.0\text{m}$ ($12.0\text{m}$ standoff): $+0.70/\text{step}$

Even at constant matched speed, flying at $6.0\text{m}$ yields **$+9.0/\text{step}$** vs **$+2.08/\text{step}$** at $9.0\text{m}$. The policy has a relentless mathematical incentive to close the final 3 meters.

#### 2. Exact Optical Standoff Alignment
For a $0.35\text{m}$ target drone on a $640\times 480$ sensor with $60^\circ$ HFOV ($f_x = 554.256\text{ px}$):
$$\text{Pixel Width at 6.0m} = \frac{0.35 \times 554.256}{6.0} = 32.3316\text{ px}$$
$$\mathbf{w}^* = \frac{32.3316}{640} = \mathbf{0.0505}$$

Calibrating $w_{\text{nominal}} = 0.0505$ ensures that visual scale error $\text{scale\_err} = 0.0$ corresponds precisely to $6.0\text{m}$.

---

### Generation 5A: The "Lighthouse Radar" Exploit & Isotropic Regression (recurrent_ppo_gen5)

#### Mathematical Formulation
To address head-on turn freezes and visual target loss during abrupt turns, three modifications were introduced:
1. **Blind Search Yaw Incentive**:
   $$R_{\text{search}} = 0.5 \cdot \text{clip}\left( \frac{\text{sgn}(e_{x, \text{last}}) \cdot \dot{\psi}}{120^\circ/\text{s}}, 0, 1 \right) \quad \text{when } \text{in\_view} = \text{False}$$
2. **Lump-Sum Re-acquisition Bounty**:
   $$R_{\text{reacquire}} = +25.0 \quad \text{when transitioning from out-of-FOV to in-FOV}$$
3. **Isotropic Scalar Distance Potential**:
   $$R_{\text{dist}} = 3.0 \cdot \exp\left( -\frac{(d - 6.0)^2}{2 \times 3.5^2} \right) + 6.0 \cdot \exp\left( -\frac{(d - 6.0)^2}{2 \times 1.2^2} \right)$$
   $$R_{\text{orient}} = 1.5 \cdot \max(0, \cos \theta_{\text{cone}})$$

#### Observed Pathology
* **Training Illusion**: Tensorboard reported an all-time record episode return of **$\bar{R} = 1,830$** (up from 500 in Gen 4) with explained variance $R^2 = 0.95$.
* **Benchmark Collapse**: On the standardized 120-scenario suite, 3.0-second continuous lock success collapsed to **$0.8\%$** (down from $48.3\%$), while yaw jerk spiked to **$71.01^\circ/\text{s}^2$** and physical collisions rose to **24**.

#### Algorithmic & Theoretical Root Cause
1. **The Transition Bounty Trap (Ng et al., 1999)**:
   In potential-based reward shaping, only potential differences $F(s, s') = \gamma \Phi(s') - \Phi(s)$ guarantee policy invariance. Awarding a raw lump-sum $+25.0$ bounty for transitioning from $\text{lost} \to \text{found}$ created a mathematically dominant cyclical exploit:
   - Quietly tracking in the basket at 6m yielded $+9.0/\text{step}$.
   - Yawing at maximum angular velocity ($120^\circ/\text{s}$) swept $360^\circ$ every 3 seconds. Each time the camera flashed past the target, the agent collected $+25.0$.
   - The optimal policy became a **continuous spinning radar scanner** (mean yaw rate: $49.5^\circ/\text{s}$). The agent intentionally lost sight of the target to harvest the re-acquisition jackpot.
2. **Resurrection of the Gen 1 Isotropic Shell**:
   Replacing vector wake error $d_{\text{trail}}$ with scalar error $|d - 6.0|$ destroyed the geometric tail-cone constraint. Slicing directly across the target at 6m yielded identical reward to tail pursuit, causing 24 collision breaches ($d < 1.2\text{m}$).

---

### Generation 5B: Filtered Optical Rates & Calibrated Continuous Potential (recurrent_ppo_gen5_calibrated)

#### Mathematical Formulation
1. **13D Observation Space (Filtered Optical Rates)**:
   Retains causal exponential smoothing ($\alpha = 0.4$) on optical velocities $[\dot{e}_x, \dot{e}_y, \dot{w}_{\text{norm}}]$, eliminating numeric noise amplification while feeding explicit target turn-rate and range-rate trends directly to the recurrent policy.
2. **Restoration of Vector Trail Potential**:
   $$R_{\text{trail}}(d_{\text{trail}}) = 3.0 \cdot \exp\left( -\frac{d_{\text{trail}}^2}{2 \times 3.5^2} \right) + 6.0 \cdot \exp\left( -\frac{d_{\text{trail}}^2}{2 \times 1.2^2} \right)$$
   Centering the potential strictly on $\mathbf{p}_{\text{trail}} = \mathbf{p}_{\text{target}} - 6.0 \cdot \hat{\mathbf{h}}_{\text{target}}$ eliminates isotropic ambiguity and enforces rear wake pursuit.
3. **Elimination of Transition Bounties**:
   $R_{\text{search}} \equiv 0$ and $R_{\text{reacquire}} \equiv 0$. Sightline loss incurs a steady-state penalty ($-0.10/\text{step}$).
4. **Re-enabled 3.0s Continuous Lock Milestone Termination**:
   Achieving 3.0s continuous dwell immediately terminates the episode with $+150.0$, strictly aligning policy optimization with the thesis benchmark metric.
5. **Asymmetric Control Effort Regularization**:
   Zero penalty for positive forward acceleration ($a_0 > 0$), allowing uninhibited sprint closure up to $15.0\text{ m/s}$.

---

## 4. Summary Benchmark & Ablation Matrix

| Generation | Model Checkpoint | Core Reward Formulation | Observed Flight Artifact | 120-Run Success% | Standoff RMSE | In-FOV% | Yaw Jerk |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **G1** | `recurrent_ppo_29_09` | Isotropic 5-7m Shell, $p_{\text{jerk}} = 0.01$ | **Circling / Orbiting** around target | $38.3\%$ | $7.66\text{m}$ | $75.9\%$ | $108.5^\circ/\text{s}^2$ |
| **G2** | `scratch_4m` | Heavy Jerk Penalty ($0.20$), Slew-Rate | **Exploration collapse** ($\sigma \to 0.05$) | $0.0\%$ | N/A | $31.2\%$ | $0.8^\circ/\text{s}^2$ |
| **G3** | `tail_chase_finetune` | Rear Trail Anchor $\mathbf{p}_{\text{trail}}$, $p_{\text{jerk}} = 0.02$ | Circling solved; plateau at $9.7\text{m}$ | **$40.8\%$** | $8.52\text{m}$ | **$79.3\%$** | **$23.4^\circ/\text{s}^2$** |
| **G4** | `continuous_potential` | Multi-Scale Gaussian Potential Field, $w^* = 0.0505$ | Plateau solved; **beats PID overall & on evasive** | **$48.3\%$** | $8.72\text{m}$ | **$82.5\%$** | **$26.6^\circ/\text{s}^2$** |
| **G5A** | `recurrent_ppo_gen5` | Transition bounties (+25), scalar distance shell | **Lighthouse radar spinning exploit** ($120^\circ/\text{s}$) | $0.8\%$ | $18.30\text{m}$ | $42.0\%$ | $71.0^\circ/\text{s}^2$ |
| **G5B** | `recurrent_ppo_gen5_calibrated`| 13D obs (filtered rates), restored $d_{\text{trail}}$ potential | *In training / Calibrated formulation* | *Pending* | *Pending* | *Pending* | *Pending* |
| **Ref** | **Classical PID** | Kinematic Deceleration Curve ($6.0\text{m}$) | Reactive, tight standoff | **$45.8\%$** | **$3.75\text{m}$** | $75.9\%$ | **$23.9^\circ/\text{s}^2$** |
