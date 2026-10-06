# State Estimation & The Kalman Filter in Autonomous Drone Tracking
## Theoretical Foundations, Mathematical Derivations & Pipeline Architecture

**DTU Bachelor Project — Autonomous Drone Pursuit & Vision-Based Interception**  
**Author:** Philip Kierkegaard  
**Subsystems:** Perception, State Estimation & Visual Servoing (`perception/pipeline.py`, `control/spatial.py`, `control/pid_controller.py`)

---

## Table of Contents
1. [Executive Summary & Motivation](#1-executive-summary--motivation)
2. [The Fundamental Problem of Airborne Visual Tracking](#2-the-fundamental-problem-of-airborne-visual-tracking)
3. [Optimal Recursive Bayesian Estimation Theory](#3-optimal-recursive-bayesian-estimation-theory)
   - [3.1 State-Space Formulation](#31-state-space-formulation)
   - [3.2 The Recursive Bayesian Framework](#32-the-recursive-bayesian-framework)
   - [3.3 The Gaussian Hypothesis & The Kalman Derivation](#33-the-gaussian-hypothesis--the-kalman-derivation)
4. [The 8D Constant-Velocity Kalman Filter (`KalmanBoxTracker`)](#4-the-8d-constant-velocity-kalman-filter-kalmanboxtracker)
   - [4.1 State & Measurement Representation](#41-state--measurement-representation)
   - [4.2 Kinematic Transition Model ($F$)](#42-kinematic-transition-model-f)
   - [4.3 Observation Model ($H$)](#43-observation-model-h)
   - [4.4 Process ($Q$) and Measurement ($R$) Covariance Tuning](#44-process-q-and-measurement-r-covariance-tuning)
   - [4.5 The Step-by-Step Filter Equations](#45-the-step-by-step-filter-equations)
5. [Ego-Motion Compensation: Full 3D SO(3) Attitude Decoupling](#5-ego-motion-compensation-full-3d-so3-attitude-decoupling)
   - [5.1 The Moving Platform Dilemma](#51-the-moving-platform-dilemma)
   - [5.2 3D Ray Back-Projection & Mount Up-Tilt Compensation](#52-3d-ray-back-projection--mount-up-tilt-compensation)
   - [5.3 Deterministic Control Injection into the Predict Step](#53-deterministic-control-injection-into-the-predict-step)
6. [Data Association & Multi-Track Management (`DroneTracker`)](#6-data-association--multi-track-management-dronetracker)
   - [6.1 Hungarian Matching on IoU Cost Matrix](#61-hungarian-matching-on-iou-cost-matrix)
   - [6.2 Euclidean Center Distance Gating Fallback](#62-euclidean-center-distance-gating-fallback)
   - [6.3 Track Lifecycle: Confirmation, Coasting & Pruning](#63-track-lifecycle-confirmation-coasting--pruning)
7. [Dynamic Foveal Zoom (PTZ) Integration](#7-dynamic-foveal-zoom-ptz-integration)
   - [7.1 Resolving the Latency Catch-22 with 1-Step Prior State](#71-resolving-the-latency-catch-22-with-1-step-prior-state)
   - [7.2 Sensor-Frame Coordinate Invariance](#72-sensor-frame-coordinate-invariance)
8. [Downstream Flight Control Interface](#8-downstream-flight-control-interface)
   - [8.1 Feedforward & Derivative Damping for IBVS](#81-feedforward--derivative-damping-for-ibvs)
   - [8.2 Recurrent Policy Observation Vector](#82-recurrent-policy-observation-vector)
9. [Report Writing Guide & Benchmark Evaluation](#9-report-writing-guide--benchmark-evaluation)
   - [9.1 Evaluation Metrics (CLE, Jitter Index, Coasting Recall, AUC)](#91-evaluation-metrics-cle-jitter-index-coasting-recall-auc)
   - [9.2 Key Arguments & Takeaways for the Thesis Committee](#92-key-arguments--takeaways-for-the-thesis-committee)

---

## 1. Executive Summary & Motivation

In an airborne drone pursuit system, a chase quadrotor must track and intercept an agile target drone using visual feedback from an onboard camera. The perception system employs a deep neural network (YOLOv8) to detect the target bounding box on each video frame.

However, **a single-frame object detector is inherently memoryless, noisy, and incomplete:**
* It produces bounding box corner jitter due to pixel discretization and feature ambiguity.
* It occasionally drops detections (false negatives) during rapid maneuvers, sun glare, or motion blur.
* It measures only 2D position and size—it has **no concept of velocity, momentum, or acceleration**.

Feeding raw YOLO detections directly into the quadrotor flight controller results in violent derivative kicks ($\frac{de}{dt}$), actuator saturation, high-frequency motor heating, and immediate tracking failure whenever the detector misses a single frame.

To solve this, the pipeline integrates a **discrete-time 8D Linear Kalman Filter** coupled with **3D ego-motion compensation** and **Hungarian data association**. This subsystem performs three vital tasks:
1. **Denoising & Smoothing:** Extracts an optimal estimate of target centroid and dimensions by filtering out high-frequency detection jitter.
2. **Velocity Estimation (Hidden State Observer):** Reconstructs target 2D image velocities $(\dot{c}_x, \dot{c}_y)$ and scale expansion rates $(\dot{w}, \dot{h})$ without dirty numerical differentiation.
3. **Dead Reckoning (Coasting):** Propagates target position forward across occlusions and detector dropouts, allowing the drone to maintain pursuit when the vision model temporarily fails.

```
┌────────────────────────┐         Noisy Bounding Boxes          ┌───────────────────────────────────┐
│     YOLOv8 Detector    │ ────────────────────────────────────► │        Hungarian Associator       │
│  (Spatial CNN, 30 FPS) │           z_k = [cx, cy, w, h]^T      │      (IoU + Distance Gating)      │
└────────────────────────┘                                       └─────────────────┬─────────────────┘
                                                                                   │ Associated Detections
                                                                                   ▼
┌────────────────────────┐         Ego-Motion Shift (dx, dy)     ┌───────────────────────────────────┐
│  Pixhawk Telemetry IMU │ ────────────────────────────────────► │      8D Kalman State Estimator    │
│  (Roll, Pitch, Yaw, v) │   (SO(3) Mount Tilt & Attitude)       │   x_k = [cx, cy, w, h, vx..vh]^T  │
└────────────────────────┘                                       └─────────────────┬─────────────────┘
                                                                                   │
                                                     ┌─────────────────────────────┴─────────────────────────────┐
                                                     ▼                                                           ▼
                                      ┌─────────────────────────────┐                             ┌─────────────────────────────┐
                                      │ Dynamic Foveal Zoom (PTZ)   │                             │   Flight Controller (IBVS)  │
                                      │ Crops high-res sensor patch │                             │  Smooth setpoint & D-term   │
                                      └─────────────────────────────┘                             └─────────────────────────────┘
```

---

## 2. The Fundamental Problem of Airborne Visual Tracking

### 2.1 Why Raw Detections Fail Control Loops
Let the detected bounding box center at discrete time $k$ be $\mathbf{z}_k = [c_{x,k}, c_{y,k}]^T$. In classical Image-Based Visual Servoing (IBVS), the controller regulates error $\mathbf{e}_k = \mathbf{z}_k - \mathbf{c}_{optical}$ to zero. 

A standard PID controller requires the error derivative $\mathbf{\dot{e}}_k \approx \frac{\mathbf{e}_k - \mathbf{e}_{k-1}}{\Delta t}$. 

If the detector output fluctuates by just $\pm 3\text{ pixels}$ between successive 30 FPS frames ($\Delta t = 0.033\text{ s}$), the apparent velocity fluctuates by:
$$\Delta v = \frac{\pm 6\text{ px}}{0.033\text{ s}} \approx \pm 180\text{ px/s}$$

When amplified by the controller's derivative gain $K_d$, this creates massive voltage/torque spikes in the motor electronic speed controllers (ESCs), causing the airframe to shudder and destabilize.

### 2.2 Measurement Gaps and Target Dropout
Target drones execute rapid evasive maneuvers (accelerations exceeding $2g$). During bank turns, the target's cross-sectional area changes rapidly, causing YOLO's confidence to dip below the detection threshold $\tau_{conf}$. Without a state estimator, the tracking error abruptly drops to zero or becomes undefined, resetting controller integral terms and causing control divergence.

---

## 3. Optimal Recursive Bayesian Estimation Theory

### 3.1 State-Space Formulation
We model the physical reality of the target on the camera image plane as a linear dynamic system governed by state vector $\mathbf{x}_k \in \mathbb{R}^n$ and discrete observations $\mathbf{z}_k \in \mathbb{R}^m$:

$$\mathbf{x}_k = F_k \mathbf{x}_{k-1} + B_k \mathbf{u}_k + \mathbf{w}_k$$
$$\mathbf{z}_k = H_k \mathbf{x}_k + \mathbf{v}_k$$

Where:
* $F_k \in \mathbb{R}^{n \times n}$ is the **State Transition Matrix** encoding the physical kinematic laws.
* $B_k \in \mathbb{R}^{n \times l}$ is the **Control Input Matrix**, and $\mathbf{u}_k$ is a known control or platform displacement (ego-motion).
* $\mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, Q_k)$ is the **Process Noise**, representing unmodeled accelerations, aerodynamic buffeting, and maneuvers.
* $H_k \in \mathbb{R}^{m \times n}$ is the **Measurement Matrix**, mapping the hidden state space into the observable sensor space.
* $\mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, R_k)$ is the **Measurement Noise**, representing sensor pixel noise and YOLO bounding-box estimation error.

Both noise sources are assumed to be zero-mean, white, Gaussian, and mutually uncorrelated:
$$\mathbb{E}[\mathbf{w}_k \mathbf{w}_j^T] = Q_k \delta_{kj}, \quad \mathbb{E}[\mathbf{v}_k \mathbf{v}_j^T] = R_k \delta_{kj}, \quad \mathbb{E}[\mathbf{w}_k \mathbf{v}_j^T] = 0$$

---

### 3.2 The Recursive Bayesian Framework
State estimation is fundamentally about tracking the **Posterior Probability Density Function (PDF)** of the state given all measurements accumulated up to time $k$, denoted $p(\mathbf{x}_k \mid \mathbf{z}_{1:k})$.

The estimation proceeds in two recursive phases:

```
                          ┌─────────────────────────────────────┐
                          │         Prior Distribution          │
                          │      p(x_{k-1} | z_{1:k-1})         │
                          └──────────────────┬──────────────────┘
                                             │
                                             ▼ Time Propagation (Chapman-Kolmogorov)
                          ┌─────────────────────────────────────┐
                          │        Predicted Prior PDF          │
                          │        p(x_k | z_{1:k-1})           │
                          └──────────────────┬──────────────────┘
                                             │
                                             ▼ Measurement Arrival (Bayes' Rule)
  ┌───────────────────────┐                  │
  │ Observation Likelihood│ ─────────► ( x ) │
  │      p(z_k | x_k)     │                  │
  └───────────────────────┘                  ▼
                          ┌─────────────────────────────────────┐
                          │        Updated Posterior PDF        │
                          │          p(x_k | z_{1:k})           │
                          └─────────────────────────────────────┘
```

#### Step 1: Time Update (Prediction / Chapman-Kolmogorov Equation)
Before receiving the measurement $\mathbf{z}_k$, we project the probability density forward in time using our kinematic model:
$$p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}) = \int_{\mathbb{R}^n} \underbrace{p(\mathbf{x}_k \mid \mathbf{x}_{k-1})}_{\text{Transition Model}} \cdot \underbrace{p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1})}_{\text{Previous Posterior}} \, d\mathbf{x}_{k-1}$$

#### Step 2: Measurement Update (Correction / Bayes' Rule)
When the new measurement $\mathbf{z}_k$ arrives from YOLO, we condition on it via Bayes' rule:
$$p(\mathbf{x}_k \mid \mathbf{z}_{1:k}) = \frac{p(\mathbf{z}_k \mid \mathbf{x}_k) \cdot p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1})}{p(\mathbf{z}_k \mid \mathbf{z}_{1:k-1})}$$
where the evidence denominator is:
$$p(\mathbf{z}_k \mid \mathbf{z}_{1:k-1}) = \int_{\mathbb{R}^n} p(\mathbf{z}_k \mid \mathbf{x}_k) \cdot p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}) \, d\mathbf{x}_k$$

---

### 3.3 The Gaussian Hypothesis & The Kalman Derivation
In general, computing these continuous integrals for arbitrary non-linear PDFs is analytically intractable (requiring particle filters or grid approximations).

However, **if both the transition model and measurement model are linear, and all noise distributions are Gaussian**, the family of Gaussian distributions is closed under linear combinations and conditioning. 

A Gaussian distribution is completely characterized by its **Mean Vector** $\mathbf{\hat{x}}$ and **Covariance Matrix** $P$:
$$p(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^n |P|}} \exp\left( -\frac{1}{2} (\mathbf{x} - \mathbf{\hat{x}})^T P^{-1} (\mathbf{x} - \mathbf{\hat{x}}) \right)$$

The **Kalman Filter** is the exact, closed-form analytic solution to the continuous recursive Bayesian filter for linear Gaussian systems. Furthermore, by the **Gauss-Markov Theorem**, even if the noise is not Gaussian, the Kalman filter is the **Best Linear Unbiased Estimator (BLUE)**—no linear estimator can achieve smaller error variance.

---

## 4. The 8D Constant-Velocity Kalman Filter (`KalmanBoxTracker`)

In [`perception/pipeline.py`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/perception/pipeline.py#L34-L169), the state estimator is implemented in the `KalmanBoxTracker` class.

### 4.1 State & Measurement Representation
To decouple target position and physical bounding box scaling while tracking momentum, we formulate an **8-dimensional state vector**:

$$\mathbf{x} = \begin{bmatrix} c_x \\ c_y \\ w \\ h \\ v_x \\ v_y \\ v_w \\ v_h \end{bmatrix} \in \mathbb{R}^8$$

Where:
* $c_x, c_y$: Bounding box geometric center on the camera image plane (in pixels).
* $w, h$: Bounding box width and height (in pixels).
* $v_x, v_y$: Linear velocity of the centroid on the image plane (in pixels/frame or pixels/second).
* $v_w, v_h$: Rate of change of box dimensions (scale expansion/contraction rate, indicating approaching or retreating depth motion).

The sensor measurement vector provided by YOLOv8 consists only of the observable bounding box coordinates:
$$\mathbf{z} = \begin{bmatrix} c_x \\ c_y \\ w \\ h \end{bmatrix} \in \mathbb{R}^4$$

---

### 4.2 Kinematic Transition Model ($F$)
We adopt a **Discrete-Time Constant Velocity (CV) Kinematic Model**. Over an update interval $\Delta t$ ($1/30\text{ s} \approx 0.0333\text{ s}$), we assume:
$$c_x(k) = c_x(k-1) + v_x(k-1) \Delta t$$
$$c_y(k) = c_y(k-1) + v_y(k-1) \Delta t$$
$$w(k) = w(k-1) + v_w(k-1) \Delta t$$
$$h(k) = h(k-1) + v_h(k-1) \Delta t$$
$$\mathbf{v}(k) = \mathbf{v}(k-1) + \mathbf{w}_v(k)$$

In matrix block form, $F \in \mathbb{R}^{8 \times 8}$ is:
$$F = \begin{bmatrix} I_{4 \times 4} & \Delta t \cdot I_{4 \times 4} \\ 0_{4 \times 4} & I_{4 \times 4} \end{bmatrix}$$

Explicitly:
$$F = \begin{bmatrix}
1 & 0 & 0 & 0 & \Delta t & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & \Delta t & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & \Delta t & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & \Delta t \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}$$

---

### 4.3 Observation Model ($H$)
The measurement matrix $H \in \mathbb{R}^{4 \times 8}$ projects the 8D state into the 4D measurement subspace:
$$H = \begin{bmatrix} I_{4 \times 4} & 0_{4 \times 4} \end{bmatrix} = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0
\end{bmatrix}$$

---

### 4.4 Process ($Q$) and Measurement ($R$) Covariance Tuning

The performance of any Kalman filter hinges entirely on the ratio between **Process Noise Covariance $Q$** and **Measurement Noise Covariance $R$**.

```
                Trust the Physics Model                   Trust the YOLO Detector
             ◄─────────────────────────────             ─────────────────────────────►
                   (Large R, Small Q)                         (Small R, Large Q)
              Heavy smoothing, high lag                  Follows every jitter, fast response
```

#### Physical Meaning of Process Noise $Q$
Process noise represents deviations from the constant velocity assumption—namely, target maneuvers, wind gusts, and pilot accelerations. In [`perception/pipeline.py`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/perception/pipeline.py#L69-L74):
$$Q = \text{diag}\left( q_{pos}, \, q_{pos}, \, 2.0, \, 2.0, \, q_{vel}, \, q_{vel}, \, 10.0, \, 10.0 \right)$$
* **Position terms ($q_{pos} = 1.0$):** Kinematic position integration uncertainty over $\Delta t$.
* **Dimension terms ($2.0$):** Physical drone geometry does not morph instantaneously; dimension changes occur gradually as relative distance and aspect angles shift.
* **Velocity terms ($q_{vel} = 18.0 \text{ to } 30.0$):** High variance assigned to velocity reflects the high agility of quadrotors capable of rapid direction reversals.
* **Scale rate terms ($10.0$):** Allows the filter to adapt when the target aggressively accelerates toward or away from the camera.

#### Physical Meaning of Measurement Noise $R$
Measurement noise represents the uncertainty and jitter of YOLOv8 bounding box predictions. In [`perception/pipeline.py`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/perception/pipeline.py#L76-L79):
$$R = \text{diag}\left( r_{pos}, \, r_{pos}, \, 4.0, \, 4.0 \right)$$
* **Center terms ($r_{pos} = 2.0$):** YOLO estimates object centroids with relatively high precision ($\pm 1.4\text{ px}$).
* **Dimension terms ($4.0$):** Bounding box edges are noisier than centroids due to varying rotor angles, motion blur at extremities, and aspect-ratio discretization in anchor-free detection heads.

---

### 4.5 The Step-by-Step Filter Equations

#### Phase 1: Time Update (A Priori Prediction)
At frame $k$, the state and its error covariance are projected forward from step $k-1$:

$$\mathbf{\hat{x}}_{k \mid k-1} = F \mathbf{\hat{x}}_{k-1 \mid k-1} + \mathbf{u}_{\text{ego}}$$
$$P_{k \mid k-1} = F P_{k-1 \mid k-1} F^T + Q$$

* $\mathbf{\hat{x}}_{k \mid k-1}$ is the **a priori state estimate**.
* $P_{k \mid k-1}$ is the **a priori error covariance matrix**. The term $F P F^T$ propagates existing uncertainty through the physics model, while $+ Q$ adds new uncertainty due to unmodeled target acceleration.

#### Phase 2: Measurement Update (A Posteriori Correction)
When YOLO provides a detection $\mathbf{z}_k = [cx_{det}, cy_{det}, w_{det}, h_{det}]^T$:

1. **Measurement Innovation (Residual):**
   $$\mathbf{y}_k = \mathbf{z}_k - H \mathbf{\hat{x}}_{k \mid k-1}$$
   * $\mathbf{y}_k \in \mathbb{R}^4$ measures the difference between actual observation and predicted observation. It is the "surprise" factor.

2. **Innovation Covariance:**
   $$S_k = H P_{k \mid k-1} H^T + R$$
   * $S_k \in \mathbb{R}^{4 \times 4}$ represents the total uncertainty of the innovation, summing the projected state uncertainty $H P H^T$ and detector noise $R$.

3. **Optimal Kalman Gain:**
   $$K_k = P_{k \mid k-1} H^T S_k^{-1} = P_{k \mid k-1} H^T \left( H P_{k \mid k-1} H^T + R \right)^{-1}$$
   * $K_k \in \mathbb{R}^{8 \times 4}$ is the optimal weighting factor that minimizes the trace of the a posteriori error covariance $\text{Tr}(P_{k \mid k})$ (Mean Squared Error).
   * **Limiting Behavior:**
     * If detector precision is perfect ($R \to 0$): $K_k \to H^{-1}$, meaning $\mathbf{\hat{x}}_k$ is driven entirely by the sensor.
     * If detector is completely unreliable ($R \to \infty$): $K_k \to 0$, meaning the filter ignores the observation and trusts pure physics prediction.

4. **A Posteriori State Update:**
   $$\mathbf{\hat{x}}_{k \mid k} = \mathbf{\hat{x}}_{k \mid k-1} + K_k \mathbf{y}_k$$
   * Blends the prior prediction with the measurement residual scaled by the Kalman gain.

5. **A Posteriori Covariance Update:**
   $$P_{k \mid k} = (I - K_k H) P_{k \mid k-1}$$
   * Incorporating new information always reduces uncertainty: $P_{k \mid k} \le P_{k \mid k-1}$.

---

## 5. Ego-Motion Compensation: Full 3D SO(3) Attitude Decoupling

### 5.1 The Moving Platform Dilemma
A standard Kalman tracker assumes that any displacement of the target on the image plane is caused by target motion. 

**This assumption catastrophically fails on an airborne chase drone:**
* When the chase drone pitches forward to accelerate ($\theta < 0$), the camera tilts down, causing the target on screen to jump upward rapidly.
* When the chase drone rolls right to bank ($\phi > 0$), the image plane rotates, displacing the target laterally.
* When the chase drone yaws to track ($\dot{\psi}$), the target sweeps across the horizontal field of view.

If the Kalman filter does not know about the chase drone's own movement, it interprets these camera rotations as **massive, violent target accelerations**, corrupting the velocity state $[v_x, v_y]$ and destabilizing the tracking loop.

```
Camera Sensor (t=0)                          Camera Sensor (t=1, after drone pitch)
┌─────────────────────────────────┐          ┌─────────────────────────────────┐
│                                 │          │             [TARGET]            │ ◄── Target jumped upward
│            [TARGET]             │          │                                 │     due to CHASER pitch,
│                                 │  ─────►  │                                 │     NOT target motion!
│                                 │          │                                 │
└─────────────────────────────────┘          └─────────────────────────────────┘
```

---

### 5.2 3D Ray Back-Projection & Mount Up-Tilt Compensation
To solve this, our system implements exact **3D Ray Back-Projection and SO(3) Attitude Decoupling** in [`control/spatial.py`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/spatial.py#L87-L182) via `compute_ego_motion_compensation`:

```
2D Image Point [cx, cy]
       │
       ▼ (1. Pinhole Back-Projection)
3D Unit Ray in Camera Optical Frame [X_cam, Y_cam, Z_cam]
       │
       ▼ (2. Invert Mechanical Mount Up-Tilt: +15 deg)
3D Vector in Drone Body Frame [X_body, Y_body, Z_body]
       │
       ├─► (3. Subtract Body Translation: v_body * dt)
       ├─► (4. Invert Attitude Increments: d_roll, d_pitch, d_yaw)
       │
       ▼ (5. Re-project through Mount Up-Tilt)
Updated 3D Position in New Camera Frame
       │
       ▼ (6. Perspective Re-Projection to Sensor)
Compensated Pixel Coordinates [cx', cy', w', h']
```

#### Step 1: Camera Ray Reconstruction
From normalized screen coordinates $e_x = \frac{cx - W/2}{W/2} \in [-1, 1]$ and $e_y = \frac{cy - H/2}{H/2} \in [-1, 1]$:
$$r_{x,\text{cam}} = e_x \cdot \tan\left(\frac{\text{HFOV}}{2}\right), \quad r_{y,\text{cam}} = e_y \cdot \tan\left(\frac{\text{VFOV}}{2}\right), \quad r_{z,\text{cam}} = 1.0$$
Scaled by estimated target distance $d$: $\mathbf{P}_{\text{cam}} = [r_{x,\text{cam}} \cdot d, \; r_{y,\text{cam}} \cdot d, \; d]^T$.

#### Step 2: Mechanical Mount Transformation
The camera is physically angled upward by $\alpha_{\text{uptilt}} = +15^\circ$ ($0.2618\text{ rad}$) to keep the target centered while the chaser pitches forward in high-speed pursuit. The rotation matrix from Camera Frame to Body Frame is:

$$R_{\text{body}}^{\text{cam}} = \begin{bmatrix}
\cos\alpha & 0 & \sin\alpha \\
0 & 1 & 0 \\
-\sin\alpha & 0 & \cos\alpha
\end{bmatrix}$$

$$\mathbf{P}_{\text{body}} = R_{\text{body}}^{\text{cam}} \mathbf{P}_{\text{cam}}$$

#### Step 3 & 4: Applying Chaser Ego-Motion
Over time step $\Delta t$, the chaser translates by $\Delta \mathbf{p}_{\text{body}} = \mathbf{v}_{\text{body}} \Delta t$ and rotates by incremental Euler angles $(\Delta \phi, \Delta \theta, \Delta \psi)$:
$$\mathbf{P}'_{\text{body}} = R_{\text{incremental}}^T \left( \mathbf{P}_{\text{body}} - \mathbf{v}_{\text{body}} \Delta t \right)$$
where $R_{\text{incremental}} = R_z(\Delta \psi) R_y(\Delta \theta) R_x(\Delta \phi)$.

#### Step 5 & 6: Re-projection to Image Plane
The rotated 3D point is transformed back to the camera frame $\mathbf{P}'_{\text{cam}} = (R_{\text{body}}^{\text{cam}})^T \mathbf{P}'_{\text{body}}$ and projected back onto the image sensor:
$$cx_{\text{ego}} = \frac{W}{2} + \frac{P'_{x,\text{cam}} / P'_{z,\text{cam}}}{\tan(\text{HFOV}/2)} \cdot \frac{W}{2}, \quad cy_{\text{ego}} = \frac{H}{2} + \frac{P'_{y,\text{cam}} / P'_{z,\text{cam}}}{\tan(\text{VFOV}/2)} \cdot \frac{H}{2}$$

The apparent shifts are:
$$\Delta c_x = cx_{\text{ego}} - cx, \quad \Delta c_y = cy_{\text{ego}} - cy$$
$$\Delta w = w \cdot \left(\frac{d}{P'_{z,\text{cam}}} - 1\right), \quad \Delta h = h \cdot \left(\frac{d}{P'_{z,\text{cam}}} - 1\right)$$

---

### 5.3 Deterministic Control Injection into the Predict Step
These calculated pixel shifts $(\Delta c_x, \Delta c_y, \Delta w, \Delta h)$ represent the exact deterministic transformation induced by the observer's own motion.

In [`perception/pipeline.py`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/perception/pipeline.py#L108-L115), this shift is directly injected as a control vector $\mathbf{u}_{\text{ego}}$ during the filter's **prediction step**:

$$\mathbf{\hat{x}}_{k \mid k-1} = F \mathbf{\hat{x}}_{k-1 \mid k-1} + \begin{bmatrix} \Delta c_x \\ \Delta c_y \\ \Delta w \\ \Delta h \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}$$

**Theoretical Consequence:**
Because the ego-motion displacement is accounted for *prior* to computing the measurement innovation $\mathbf{y}_k = \mathbf{z}_k - H \mathbf{\hat{x}}_{k \mid k-1}$, the innovation reflects **only the true independent maneuver of the target drone**, completely decoupling chaser airframe vibrations and control actions from target state estimation.

---

## 6. Data Association & Multi-Track Management (`DroneTracker`)

When multiple objects or false detections appear in the scene, the pipeline must decide which detection belongs to which existing Kalman filter track.

### 6.1 Hungarian Matching on IoU Cost Matrix
Data association is formulated as a **Bipartite Minimum Weight Matching Problem**. 

For $M$ active tracks and $N$ incoming YOLO detections, we construct a cost matrix $C \in \mathbb{R}^{M \times N}$ based on **Intersection over Union (IoU)**:
$$C_{i,j} = 1.0 - \text{IoU}(\mathbf{B}_{i}^{\text{pred}}, \mathbf{B}_{j}^{\text{det}})$$

Where:
$$\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}$$

The optimal global assignment $(i^*, j^*)$ is solved in polynomial time $\mathcal{O}(n^3)$ via the **Hungarian (Munkres) Algorithm** (`scipy.optimize.linear_sum_assignment`):
$$\min_{\pi} \sum_{i} C_{i, \pi(i)}$$

An assignment is accepted only if the match cost satisfies the gating threshold:
$$C_{i,j} \le \tau_{\text{IoU\_match}} \quad (\text{i.e. } \text{IoU} \ge 0.25)$$

---

### 6.2 Euclidean Center Distance Gating Fallback
Under aggressive target evasive maneuvers or dropped frames, the 1-step prediction $\mathbf{B}_{i}^{\text{pred}}$ and the detection $\mathbf{B}_{j}^{\text{det}}$ may experience large displacement such that their bounding boxes no longer overlap ($\text{IoU} = 0.0, C_{i,j} = 1.0$).

To prevent the tracker from discarding the target during extreme maneuvers, [`perception/pipeline.py`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/perception/pipeline.py#L261-L273) implements a **Dynamic Spatial Gating Fallback**:

```
If Hungarian IoU match fails:
  Compute centroid Euclidean distance:  d_center = || p_pred - p_det ||_2
  Compute target bounding box diagonal: diag = || [w_pred, h_pred] ||_2
  Accept match if:                      d_center < max(80.0 px, 2.5 * diag)
```

This gating region scales dynamically with target size, ensuring robust association when the target is close while remaining selective at long range.

---

### 6.3 Track Lifecycle: Confirmation, Coasting & Pruning

To handle spurious false alarms and temporary visual dropouts, tracks evolve through a discrete state machine:

```
                  detection
       ┌───────────────────────────────┐
       │                               │
       ▼                               │
┌──────────────┐   hits >= 2    ┌──────────────┐  missed frame   ┌──────────────┐
│  Tentative   │ ─────────────► │    Locked    │ ──────────────► │   Coasting   │
│ (Unconfirmed)│                │  (Confirmed) │ ◄────────────── │ (Dead Reck.) │
└──────┬───────┘                └──────────────┘    detection    └──────┬───────┘
       │                                                                │
       │ misses >= 1                                                    │ misses > 15
       ▼                                                                ▼
┌──────────────┐                                                 ┌──────────────┐
│    Pruned    │ ◄───────────────────────────────────────────────│    Pruned    │
│  (Destroyed) │                                                 │  (Destroyed) │
└──────────────┘                                                 └──────────────┘
```

1. **Tentative State:** When an unmatched detection appears, a new `KalmanBoxTracker` is instantiated with high initial velocity uncertainty ($P_{4:8, 4:8} = 100 \cdot I$). It is not reported to the flight controller until confirmed.
2. **Confirmed State:** Once detected across $\ge 2$ consecutive frames (`min_hits_to_confirm = 2`), the track is promoted to confirmed status.
3. **Coasting State (Dead Reckoning):** If YOLO fails to detect the target on frame $k$:
   * No measurement update is performed ($\mathbf{y}_k$ is null).
   * The filter executes **only the prediction step**:
     $$\mathbf{\hat{x}}_{k \mid k} = \mathbf{\hat{x}}_{k \mid k-1} = F \mathbf{\hat{x}}_{k-1 \mid k-1} + \mathbf{u}_{\text{ego}}$$
     $$P_{k \mid k} = P_{k \mid k-1} = F P_{k-1 \mid k-1} F^T + Q$$
   * The state covariance $P$ naturally expands, mathematically reflecting increasing uncertainty during occlusion.
4. **Pruning (Termination):** If a track receives no measurement updates for $\ge 15\text{ frames}$ (`max_lost_frames = 15`, equivalent to $0.5\text{ s}$ of blind coasting), it is permanently removed from memory.

---

## 7. Dynamic Foveal Zoom (PTZ) Integration

### 7.1 Resolving the Latency Catch-22 with 1-Step Prior State
At long engagement ranges ($40\text{--}60\text{ m}$), a target drone occupies only $12 \times 12\text{ pixels}$ on a 1080p sensor. Downscaling the full frame to $640 \times 640$ for YOLO obliterates the target into an unidentifiable blur.

The pipeline incorporates a **Dynamic Digital PTZ (Foveal Zoom)** that crops native sensor pixels around the target before inference.

**The Architectural Dilemma:**
To run YOLO on a cropped region of frame $k$, the system must know where the target is *before running YOLO on frame $k$*.

**The Kalman Solution:**
The pipeline queries the Kalman filter's **a priori state prediction** $\mathbf{\hat{x}}_{k \mid k-1}$:
$$\mathbf{p}_{\text{crop\_center}} = \left[ \hat{c}_{x, k \mid k-1}, \, \hat{c}_{y, k \mid k-1} \right]^T$$

Because the Kalman filter incorporates velocity and chaser ego-motion compensation, it accurately anticipates where the target will appear in frame $k$, placing the $640 \times 640$ high-acuity foveal crop precisely over the target without latency lag.

---

### 7.2 Sensor-Frame Coordinate Invariance
When YOLO detects a bounding box $\mathbf{B}_{\text{crop}} = [x_{1}, y_{1}, x_{2}, y_{2}]$ inside a cropped window offset by $(X_{\text{crop}}, Y_{\text{crop}})$, the coordinates are immediately transformed back into global sensor coordinates:

$$\mathbf{B}_{\text{global}} = \begin{bmatrix} X_{\text{crop}} + x_{1} \\ Y_{\text{crop}} + y_{1} \\ X_{\text{crop}} + x_{2} \\ Y_{\text{crop}} + y_{2} \end{bmatrix}$$

**Why this is critical:**
The Kalman filter maintains its state vector $\mathbf{x}$ strictly in continuous, un-cropped full sensor space $[0, W_{\text{sensor}}] \times [0, H_{\text{sensor}}]$. Consequently, zoom magnification changes, digital crops, and aspect adjustments **never induce coordinate jumps or discontinuities in the state estimator**.

---

## 8. Downstream Flight Control Interface

The state estimator outputs a structured telemetry packet consumed directly by the pursuit guidance controllers:

```python
telemetry = {
    "status": "LOCKED",        # "LOCKED", "COASTING", "SEARCHING"
    "error_x": float,          # Normalized azimuth error [-1.0, 1.0]
    "error_y": float,          # Normalized elevation error [-1.0, 1.0]
    "error_range": float,      # Normalized standoff distance error [-1.0, 1.0]
    "velocity_x": float,       # Filtered image velocity vx (px/s)
    "velocity_y": float,       # Filtered image velocity vy (px/s)
    "is_coasting": bool        # True if operating on dead reckoning
}
```

### 8.1 Feedforward & Derivative Damping for IBVS
In the classical controller ([`control/pid_controller.py`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/control/pid_controller.py)), the yaw and pitch loops utilize the Kalman state directly:

$$\dot{\psi}_{\text{cmd}} = -K_p \cdot e_x - K_d \cdot \left( \frac{\hat{v}_x}{W/2} \right)$$
$$v_{z,\text{cmd}} = K_{p,z} \cdot e_y + v_{z,\text{attitude\_feedforward}} + K_{d,z} \cdot \left( \frac{\hat{v}_y}{H/2} \right)$$

* **Elimination of Noise:** Replacing finite-difference derivative $\frac{\Delta e}{\Delta t}$ with the Kalman estimated velocity $\hat{v}_x, \hat{v}_y$ completely eliminates derivative jitter, allowing higher proportional gains without actuator buzz.
* **Lead Pursuit:** The velocity term provides natural lead-angle pursuit, commanding the chaser to turn ahead of an accelerating target.

### 8.2 Recurrent Policy Observation Vector
In the deep reinforcement learning controller (Generation 5C, `sb3_contrib.RecurrentPPO`), the state estimation telemetry forms the first 5 elements of the 13-dimensional observation vector:
$$\mathbf{o}_t = \begin{bmatrix} e_x & e_y & e_{\text{range}} & \frac{\hat{v}_x}{300} & \frac{\hat{v}_y}{300} & \dots \end{bmatrix}^T$$

Providing clean, filtered visual velocities enables the LSTM recurrent network to internalize target trajectory history and execute coordinated banking maneuvers during sharp evasions.

---

## 9. Report Writing Guide & Benchmark Evaluation

When writing the Perception and State Estimation chapters of your thesis report, structure your evaluation around the following standardized metrics:

### 9.1 Evaluation Metrics

#### 1. Center Location Error (CLE)
Measures tracking precision against ground-truth centroid $(cx^*_t, cy^*_t)$:
$$\text{CLE}_t = \sqrt{ (cx_t - cx^*_t)^2 + (cy_t - cy^*_t)^2 } \quad [\text{pixels}]$$
* *Report:* Mean CLE and Precision @ 20px threshold (percentage of frames with $\text{CLE} \le 20\text{ px}$).

#### 2. Spatial Jitter Index (Control Smoothness)
Measures high-frequency setpoint oscillation by computing the second-order temporal difference (discrete jerk) of the estimated target position:
$$\text{Jitter} = \frac{1}{T - 2} \sum_{t=2}^{T-1} \left\| \mathbf{p}_{t+1} - 2\mathbf{p}_t + \mathbf{p}_{t-1} \right\|_2 \quad \text{where } \mathbf{p}_t = [cx_t, cy_t]^T$$
* *Significance:* A lower Jitter Index correlates directly with lower ESC current draw, lower motor temperatures, and smoother airframe dynamics. **The Kalman filter should demonstrate an 80%+ reduction in Jitter compared to raw YOLO.**

#### 3. Coasting Recall Under Dropout
$$\text{Recall}_{\text{coast}} = \frac{N_{\text{maintained}}(\text{IoU} \ge 0.3 \mid \text{YOLO missed})}{N_{\text{total}}(\text{YOLO missed})}$$
* *Significance:* Quantifies the filter's dead-reckoning ability to bridge visual dropouts caused by sun glare or motion blur.

#### 4. Bounding Box Overlap Success (AUC)
The Area Under the Curve of the IoU threshold curve across $\tau \in [0.0, 1.0]$. Demonstrates that bounding box scale rates $(\dot{w}, \dot{h})$ accurately track target distance changes.

---

### 9.2 Key Arguments & Takeaways for the Thesis Committee

1. **Separation of Concerns (Spatial vs. Temporal):**
   * *Argument:* Deep learning detectors (YOLO) excel at spatial feature extraction (mapping pixels to bounding boxes), but are fundamentally memoryless. Combining a lightweight neural detector with a physically grounded Bayesian state estimator is computationally superior to end-to-end recurrent vision networks on embedded edge hardware (Jetson Orin Nano).
2. **Ego-Motion Decoupling as an Inertial Observer:**
   * *Argument:* Vision-based tracking on an agile quadrotor cannot treat the camera as a stationary observer. Decoupling chaser roll, pitch, and mount uptilt via 3D ray backprojection transforms the 2D image tracker into an inertial-stabilized visual observer.
3. **Enabler of Dynamic Foveal Zoom:**
   * *Argument:* The Kalman filter does not merely filter past observations; its forward prediction $\mathbf{\hat{x}}_{k \mid k-1}$ is the architectural enabler of active foveal zoom, breaking the traditional trade-off between wide-field search and high-resolution distant target tracking.
