# Section 3: Tiny-Target Drone Detection: Baseline Formulation, Failure Analysis, and Clutter-Enriched Optimization

## 3.1 Introduction and Perception Requirements
In an autonomous aerial pursuit and interception system, the visual perception pipeline serves as the primary observation model for real-time guidance. To enable effective Image-Based Visual Servoing (IBVS) and line-of-sight regulation, the detector must reliably localize a non-cooperative target drone within a 2D camera stream at high operational frequencies ($\ge 30\text{ Hz}$). 

Target drone detection introduces three severe computer vision challenges:
1. **Extreme Scale Variance & Tiny Target Geometry:** At operational interception ranges ($15\text{ m}$ to $50\text{ m}$), a standard micro-UAV subtends fewer than $15 \times 15\text{ pixels}$ ($<0.02\%$ of a $1080\text{p}$ image), degrading distinctive high-frequency texture and propeller geometry.
2. **Dynamic Background Entropy:** Unlike ground-based tracking where the horizon is relatively stable, an airborne pursuit vehicle experiences continuous camera pitch, roll, and elevation changes. The target must be differentiated against high-frequency urban facades, tree canopies, horizon transitions, and untextured skies.
3. **Strict Embedded Latency Budget:** The detector must execute on an embedded companion computer (e.g., NVIDIA Jetson Orin Nano) with an inference budget under $20\text{ ms}$, restricting model selection to lightweight compact architectures such as YOLOv8n (8.1 GFLOPs, 3.0M parameters).

---

## 3.2 Baseline Model Formulation and Empirical Failure Analysis

### 3.2.1 Initial Training Distribution (Model 1)
The initial baseline model (**Model 1**) was trained on an aggregated sequence dataset combining the **Anti-UAV** multi-video benchmark and sequence `010` of the **Det-Fly** dataset, totaling approximately $18,000\text{ frames}$. Training was executed using YOLOv8n pretrained on COCO, optimized using Stochastic Gradient Descent (SGD) with an initial learning rate $\eta_0 = 0.01$ and input resolution of $640 \times 640\text{ pixels}$.

### 3.2.2 Empirical Failure Diagnosis
Evaluation of Model 1 across held-out flight sequences revealed two catastrophic failure modes that directly compromised downstream visual tracking:

```
                  ┌────────────────────────────────────────────────────────┐
                  │          Model 1 (Baseline) Failure Modes              │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
    ┌───────────────────────────────────┐             ┌───────────────────────────────────┐
    │     1. Clear-Sky Bias             │             │   2. Structural False Positives   │
    │  • 90%+ training frames in sky    │             │  • Zero hard negative structures  │
    │  • Severe recall ceiling (63%)    │             │  • Spurious locks on roof corners │
    │  • 37%+ distant targets missed    │             │  • Tracker trapped in video15     │
    └───────────────────────────────────┘             └───────────────────────────────────┘
```

1. **Clear-Sky Representation Bias & Recall Plateau:**
   Over $90\%$ of the initial training frames depicted targets against clear, uniform skies. Consequently, the convolutional feature extractor learned to associate the presence of a drone with high local contrast against a featureless background rather than learning the intrinsic topological geometry of the airframe. When evaluated against complex backgrounds, recall degraded sharply, reaching a hard ceiling of $63\%$ even at near-zero confidence thresholds ($\text{conf} \to 0.00$). Over $37\%$ of true drone instances were entirely invisible to the detector.
2. **Elevated False Alarms on Structured Geometry:**
   Because the training set lacked explicit negative examples of complex non-drone structures, the network produced spurious high-confidence activations on sharp architectural edges (e.g., roof corners, window mullions, HVAC machinery, antennas). In closed-loop tracking simulations on `video15`, when the target drone temporarily exited the field of view, the detector triggered a false positive on a static roof corner ($\text{conf} \approx 0.28$). The visual servoing controller immediately latched onto the building corner, permanently losing the drone even after it re-entered the scene.

---

## 3.3 Data-Centric Model Optimization (Model 2)

### 3.3.1 Clutter Enrichment via DUT-Anti-UAV
To resolve these structural failure modes without introducing heavier, latency-prohibitive backbones, a **data-centric optimization** strategy was deployed. The training corpus was augmented with the **DUT-Anti-UAV** dataset, expanding the active training set from $18,000$ to **$31,233\text{ images}$**:
* **Visual Entropy & Non-Sequential Diversity:** In contrast to sequential video datasets where adjacent frames are near-identical, DUT-Anti-UAV provides $9,748$ independent, diverse captures featuring drones operating in complex real-world clutter: industrial complexes, dense tree canopies, maritime backgrounds, and varied atmospheric lighting.
* **Scale-Selective Pruning:** An automated morphological filter was applied to prune $252$ extreme close-up captures where the drone bounding box exceeded $15\%$ of the total frame area. Removing these non-representative close-ups forced the multi-scale feature pyramid (P3, P4, P5) to specialize in small, distant drone representations ($<3\%$ frame area).

---

## 3.4 Evaluation Methodology: Sequence-Stratified Splitting for Temporal State Estimation

A critical methodological consideration was the design of the dataset splits. Standard random frame-level shuffling is fundamentally invalid for aerial video evaluation due to two core factors:

### 3.4.1 Prevention of Spatio-Temporal Data Leakage
In video-based tracking datasets, consecutive frames sampled at $30\text{ Hz}$ exhibit near-unity mutual information. Shuffling frames randomly between train and test distributions leaks background textures, illumination profiles, and target trajectories into the training set. This creates artificial overfitting where the model appears to perform exceptionally well on test frames simply by memorizing background features rather than generalizing.

### 3.4.2 Preservation of Continuous Trajectories for Kalman State Estimation
In our visual tracking architecture, detections $z_k = [x_c, y_c, w, h]^T$ are not consumed in isolation; they are fed into an **8-State Recursive Kalman Filter** (`KalmanBoxTracker`) modeling position and velocity:

$$\mathbf{x}_k = [x_c, y_c, w, h, \dot{x}_c, \dot{y}_c, \dot{w}, \dot{h}]^T$$

To validate the joint perception-tracking system, **complete, unbroken flight trajectories were held out** rather than isolated frames:
* **Validation Set (6 Continuous Videos, 6,821 Frames):** Comprising `video01`, `video04`, `video13`, `video15`, `video16`, and `video17`. Serves as the tuning ground for calibrating measurement noise covariance ($R$) and process noise covariance ($Q$).
* **Test Set (3 Continuous Videos + Det-Fly 020, 8,052 Frames):** Comprising `video05`, `video12`, `video20`, Det-Fly sequence `020`, and $98$ hard-negative background distractor frames. Held strictly isolated as an unbiased benchmark.

Holding out contiguous sequences allows rigorous evaluation of real-world temporal behaviors:
1. **Target Coasting:** Maintaining state prediction through the Kalman velocity vector during brief target occlusions or missed detections.
2. **Re-Acquisition:** Re-locking onto the target without track ID fragmentation once the drone re-enters the field of view.
3. **Scale Regulation:** Smoothly commanding digital PTZ (foveal zoom) across dynamic range transitions.

---

## 3.5 Experimental Results and Comparative Analysis

Both the baseline (**Model 1**) and clutter-enriched (**Model 2**) checkpoints were benchmarked on the identical held-out test split of **$8,052\text{ images}$** ($7,807$ ground-truth drone instances and $245$ pure background distractor frames). Testing was conducted on Apple Silicon MPS hardware with batch size $16$ and input resolution $640 \times 640\text{ pixels}$.

### 3.5.1 Quantitative Performance Metrics

| Evaluation Metric | Baseline Model (`v1`) | Clutter-Enriched Model (`v2`) | Absolute Gain | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Precision ($P$)** | $82.29\%$ | **$88.29\%$** | $+6.00\%$ | $+7.3\%$ |
| **Recall ($R$)** | $53.13\%$ | **$66.91\%$** | $+13.78\%$ | **$+26.0\%$** |
| **$\text{mAP}@0.5$** | $56.21\%$ | **$71.40\%$** | $+15.19\%$ | **$+27.0\%$** |
| **$\text{mAP}@0.5:0.95$** | $23.06\%$ | **$30.45\%$** | $+7.39\%$ | **$+32.0\%$** |
| **True Positive Detections** | $4,532$ | **$5,693$** | $+1,161\text{ frames}$ | $+25.6\%$ |
| **False Negative Misses** | $3,275$ | **$2,114$** | $-1,161\text{ frames}$ | **$-35.5\%$** |
| **Optimal F1 Peak** | $0.65\text{ (at conf } 0.33\text{)}$ | **$0.76\text{ (at conf } 0.39\text{)}$** | $+0.11$ | $+16.9\%$ |
| **Operating F1 Plateau** | Narrow ($[0.28, 0.40]$) | **Broad ($[0.25, 0.55]$)** | $+150\%$ bandwidth | Robust tuning |
| **Inference Latency** | $0.32\text{ ms}$ | **$0.28\text{ ms}$** | — | $>230\text{ FPS}$ |
| **Post-Processing Latency** | $3.88\text{ ms}$ | **$3.87\text{ ms}$** | — | Fixed NMS budget |

```
    mAP@0.5 Comparison:
    Model 1 (v1): [██████████████░░░░░░] 56.21%
    Model 2 (v2): [██████████████████░░] 71.40%  (+15.19% Absolute / +27.0% Relative)

    Recall Comparison:
    Model 1 (v1): [█████████████░░░░░░░] 53.13%
    Model 2 (v2): [█████████████████░░░] 66.91%  (+13.78% Absolute / +26.0% Relative)

    Missed Target Frames (Lower is Better):
    Model 1 (v1): [████████████████] 3,275 frames
    Model 2 (v2): [██████████░░░░░░] 2,114 frames  (-35.5% Miss Reduction)
```

### 3.5.2 Precision-Recall Dynamics and Operating Threshold Selection
In standard detector optimization, an increase in recall typically comes at the expense of precision, as the classifier admits more false positive proposals. However, as demonstrated in the benchmark:
* **Simultaneous Improvement:** Precision increased by $+6.00\%$ alongside a $+13.78\%$ surge in recall. Injecting diverse background clutter resolved feature ambiguity, enabling the network to eliminate background false positives while simultaneously lowering the activation threshold for genuine distant drones.
* **Shelf-Shaped PR Curve:** As illustrated in the Precision-Recall curve, Model 2 maintains precision above $95\%$ up to $50\%$ recall, and stays above $90\%$ precision out to $67\%$ recall before rolling off gracefully. In contrast, Model 1 experienced an abrupt drop-off past $55\%$ recall.
* **Operating Confidence Discovery:** Analysis of the F1-confidence curve reveals a wide, flat plateau between $\text{conf} = 0.25$ and $\text{conf} = 0.55$, peaking at $F_1 = 0.76$ at $\text{conf} = 0.387$. Operating the pursuit pipeline at $\text{conf} = 0.35$ places the system directly in this optimal zone, ensuring high tracking sensitivity while guaranteeing $>90\%$ detection precision.

---

## 3.6 Closed-Loop Tracking Impact and System Integration

The practical utility of the optimized detector was validated by deploying Model 2 into the integrated visual tracking pipeline (`track_pipeline.py`) across three challenging flight regimes:

```text
┌───────────────────────────────┐     2D Detections      ┌───────────────────────────────┐
│     YOLOv8n Detector (v2)     │ ─────────────────────► │   Kalman State Estimator      │
│  • 71.4% mAP50 / 88.3% Prec   │   zk = [xc, yc, w, h]  │   8D State [x, v_x, w, ...]   │
└───────────────────────────────┘                        └───────────────┬───────────────┘
                                                                         │
                                       ┌─────────────────────────────────┴────────────────┐
                                       ▼                                                  ▼
                        ┌───────────────────────────────┐                  ┌───────────────────────────────┐
                        │ Dynamic Digital Foveal Zoom   │                  │ Kinematic IBVS Controller     │
                        │ • Normalizes target to 140px  │                  │ • Range: v = sqrt(2 * a * d)  │
                        │ • Mitigates small-target loss │                  │ • Pitch de-rotation feedfwd   │
                        └───────────────────────────────┘                  └───────────────────────────────┘
```

1. **Elimination of Structural Artifact Traps (`video15`):**
   In the complex rooftop sequence where Model 1 catastrophically failed, Model 2 exhibited zero false alarms on the building facade. When the target performed an out-of-frame egress, the Kalman filter smoothly coasted along the target's estimated velocity vector in `COASTING` state, and re-acquired the drone within a single frame ($33\text{ ms}$) upon re-entry.
2. **High-Dynamic Angular Acceleration (`video17`):**
   During rapid break turns exceeding line-of-sight rates of $45^\circ/\text{s}$, Model 2 maintained a tracking lock duty cycle exceeding $98\%$, providing continuous, low-variance innovation terms to the Kalman measurement update.
3. **Multi-Scale Invariance (`video01`):**
   Model 2 maintained unbroken tracking across an $11.3\times$ target scale variation (from a distant $18\text{-pixel}$ target at $45\text{ m}$ to a $205\text{-pixel}$ close-up), validating the effectiveness of the scale-pruned training corpus.

---

## 3.8 Stochastic Target Trajectory Simulation and Monte Carlo Pursuit Validation

To fulfill the Week 2 milestone (*"Research and implement drone trajectory simulations"*), the pursuit pipeline was evaluated beyond deterministic, hand-crafted trajectories (such as figure-8 or slalom maneuvers). Evaluating state estimation and Image-Based Visual Servoing (IBVS) against realistic evasive targets requires an uncooperative flight trajectory generator that satisfies three criteria:
1. **Stochasticity:** Flight paths must be drawn from parameterized random distributions rather than fixed equations, eliminating pursuit path memorization.
2. **Physical Dynamic Feasibility:** The target trajectory must adhere to multirotor flight dynamics (differential flatness), ensuring continuity in position $\mathbf{p}(t)$, velocity $\mathbf{v}(t)$, and acceleration $\mathbf{a}(t)$ with strict motor thrust ceilings ($a(t) \le a_{\max}$).
3. **Black-Box Oracle Architecture:** The generator must function as an external oracle $\mathcal{T}(t) \to [\mathbf{p}(t), \mathbf{v}(t), \mathbf{a}(t)]$ providing exact closed-form derivatives for unbiased Kalman ground-truth benchmarking without numerical differentiation noise.

### 3.8.1 Mathematical Formulation: Markov Random Walk with $\mathcal{C}^2$ B-Spline Smoothing
The trajectory generator (`StochasticTargetTrajectory`) synthesizes flight paths in a two-stage process:
1. **Markov Anchor Sampling:** Temporal waypoints $\mathbf{w}_k$ spaced at $\Delta t_{\text{wp}}$ are generated by drawing forward cruise speed $v_k \sim \mathcal{N}(\mu_v, \sigma_v^2)$, horizontal yaw turning rate $\dot{\psi}_k \sim \mathcal{N}(0, \sigma_\psi^2)$, and vertical climb rate $\dot{z}_k \sim \mathcal{N}(0, \sigma_z^2)$, coupled with soft-boundary potential repulsion near altitude boundaries $[z_{\min}, z_{\max}]$.
2. **$\mathcal{C}^2$ Cubic B-Spline Fitting:** A piecewise cubic polynomial spline is fitted through the waypoints:
   $$\mathbf{p}(t) = \sum_{i=0}^{n-1} \mathbf{c}_i B_{i, 3}(t)$$
   Because cubic basis functions $B_{i,3}(t)$ possess continuous second derivatives, analytical velocity $\mathbf{v}(t) = \mathbf{p}'(t)$ and acceleration $\mathbf{a}(t) = \mathbf{p}''(t)$ are evaluated directly in closed form.
3. **Thrust Ceiling Compliance via Time-Warp Scaling:** To prevent unphysical accelerations during sharp transitions, the peak acceleration $a_{\text{peak}} = \max_t \|\mathbf{a}(t)\|$ is checked against the physical motor thrust limit $a_{\max}$. If $a_{\text{peak}} > a_{\max}$, a temporal scaling factor $\kappa = \sqrt{a_{\max} / a_{\text{peak}}}$ is applied, strictly guaranteeing that $\forall t, \|\mathbf{a}(t)\| \le a_{\max}$.

```text
┌───────────────────────────────┐      Anchor Waypoints       ┌───────────────────────────────┐
│      Markov Random Walk       │ ──────────────────────────► │     Cubic B-Spline Solver     │
│   v ~ N(μ, σ²), ψ_dot ~ N(0)  │        w_k in R³            │   C² Continuous Curves p(t)   │
└───────────────────────────────┘                             └───────────────┬───────────────┘
                                                                              │
                                      ┌───────────────────────────────────────┴────────────────┐
                                      ▼                                                        ▼
                       ┌───────────────────────────────┐                        ┌───────────────────────────────┐
                       │  Thrust Ceiling Verification  │                        │ Analytical Oracle Ground-Truth│
                       │  Time-warp if ||a|| > a_max   │                        │ Exact p(t), v(t), a(t) for KF │
                       └───────────────────────────────┘                        └───────────────────────────────┘
```

Three standardized flight profiles were established:
* **Cruising:** $\mu_v = 3.0\text{ m/s}, a_{\max} = 2.2\text{ m/s}^2$ (standard non-evasive transit).
* **Evasive:** $\mu_v = 4.5\text{ m/s}, a_{\max} = 3.5\text{ m/s}^2$ (dynamic zigzagging and sudden altitude shifts).
* **Aerobatic:** $\mu_v = 6.0\text{ m/s}, a_{\max} = 5.0\text{ m/s}^2$ (extreme aggressive maneuvers near multirotor control limits).

### 3.8.2 Monte Carlo Empirical Evaluation
A 75-trial Monte Carlo benchmark ($25\text{ independent runs}$ per flight profile, $20.0\text{ s}$ per flight at $50\text{ Hz}$) was executed using the full closed-loop 6-DOF pursuer simulator (`FastPixhawkQuadSim`) and visual servoing controller (`DroneVisualPIDController`):

| Flight Profile | Cruise Speed ($v$) | Thrust Limit ($a_{\max}$) | Trials ($N$) | Optical Lock Retention | Post-Convergence Lock | Mean Standoff Distance | Radial Centering Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cruising** | $3.0\text{ m/s}$ | $2.2\text{ m/s}^2$ | $25$ | **$99.3\% \pm 1.4\%$** | $99.2\% \pm 1.7\%$ | $11.40 \pm 0.76\text{ m}$ | $0.697 \pm 0.029$ |
| **Evasive** | $4.5\text{ m/s}$ | $3.5\text{ m/s}^2$ | $25$ | **$99.3\% \pm 1.5\%$** | $99.2\% \pm 1.8\%$ | $12.94 \pm 2.33\text{ m}$ | $0.692 \pm 0.049$ |
| **Aerobatic** | $6.0\text{ m/s}$ | $5.0\text{ m/s}^2$ | $25$ | **$96.0\% \pm 9.7\%$** | $95.2\% \pm 11.4\%$ | $12.06 \pm 2.72\text{ m}$ | $0.706 \pm 0.063$ |

**Key Findings:**
1. **Unbroken Lock Retention in High-Clutter Regimes:** In both Cruising and Evasive profiles, the pursuer maintained target visibility for $>99.3\%$ of flight time, with zero catastrophic loss-of-lock events.
2. **Stress-Testing Aerobatic Limits:** In the Aerobatic profile, the target's rapid accelerations ($5.0\text{ m/s}^2$) pushed the pursuer's kinematic visual servoing limits, identifying brief out-of-frame excursions during abrupt direction reversals. This establishes a concrete baseline for integrating feed-forward target acceleration estimation in subsequent weeks.

---

## 3.9 Section Summary
Through targeted empirical diagnosis, the fundamental bottleneck of the baseline drone detector was traced to clear-sky training bias and a lack of negative architectural clutter. By executing a data-centric augmentation with curated non-sequential imagery from DUT-Anti-UAV, test mAP@0.5 improved from $56.21\%$ to $71.40\%$ and recall surged from $53.13\%$ to $66.91\%$ with zero increase in computational latency. Furthermore, adopting a sequence-stratified evaluation split preserved temporal continuity for Kalman filter covariance calibration. Finally, a physically feasible stochastic trajectory simulation generator was developed and validated through a 75-trial Monte Carlo benchmark, confirming that the visual pursuit system maintains robust $>96-99\%$ optical lock retention even under highly unpredictable evasive maneuvers.
