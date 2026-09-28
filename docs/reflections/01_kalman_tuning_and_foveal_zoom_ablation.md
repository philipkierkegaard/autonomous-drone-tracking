# Research Reflection 01: Kalman Filter Evaluation, Covariance Tuning & Foveal Zoom Ablation

* **Date:** September 13, 2026  
* **Project:** DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System  
* **Subsystem:** Perception & State Estimation (`object-detection/`, `track_pipeline.py`, `run_tracker.py`)  
* **Datasets Involved:** Anti-UAV 3-Way Sequence-Stratified Split (`train`, `val`, `test`)

---

## 1. Context & Motivation

A custom YOLOv8n object detection model was trained on the sequence-stratified Anti-UAV dataset to detect target drones in real time. To interface with the kinematic Image-Based Visual Servoing (IBVS) controller on the airborne Jetson Orin Nano, the detection pipeline incorporates:
1. An **8D State Kalman Filter** (`KalmanBoxTracker`) to estimate position and velocity while smoothing bounding-box jitter.
2. A **Dynamic Digital PTZ / Foveal Zoom** mechanism that crops native sensor pixels around the Kalman-predicted target location to boost visual acuity on small, distant targets.

### The Central Dilemma
Because the YOLO model was trained as a standard single-frame detector—and was **not** specifically optimized for or co-trained with a Kalman filter—several foundational questions emerge:
1. *What is the mathematically and scientifically correct way to evaluate the accuracy of the combined YOLO + Kalman filter pipeline?*
2. *Can Kalman filter parameters ($R, Q$) be tuned without corrupting the test dataset or requiring external video collection?*
3. *Does the Foveal Zoom mechanism introduce evaluation bias favoring the Kalman filter, and how should it be rigorously evaluated in the thesis?*

---

## 2. Theoretical Clarification: The Tracking-by-Detection Paradigm

In visual tracking and robotics, **object detectors are almost never trained or optimized for a Kalman filter**. 

```
┌───────────────────────────┐         Raw Observations (Noisy)         ┌───────────────────────────┐
│       YOLO Detector       │ ───────────────────────────────────────► │   Kalman State Estimator  │
│ (Single-Frame Spatial CNN)│         z_k = [cx, cy, w, h]^T           │ (Temporal Bayesian Filter)│
└───────────────────────────┘                                          └─────────────┬─────────────┘
                                                                                     │ Filtered State
                                                                                     ▼ x_k = [cx, cy, w, h, vx, vy, vw, vh]^T
                                                                       ┌───────────────────────────┐
                                                                       │ Visual Servoing (IBVS)    │
                                                                       │ (Controller Setpoints)    │
                                                                       └───────────────────────────┘
```

* **YOLO's Role:** Learns non-linear spatial mappings from raw image pixels to bounding box coordinates $P(B \mid I_k)$. It is memoryless and has no temporal state.
* **Kalman Filter's Role:** Acts as an optimal recursive Bayesian estimator. It assumes observations $z_k$ contain Gaussian measurement noise with covariance $R$, and fuses them with a physical kinematic transition model (constant velocity) subjected to process noise $Q$.
* **Key Takeaway:** You do not train the neural network to fit the filter. Instead, **you calibrate the filter's noise covariances ($R$ and $Q$) to reflect the empirical precision of the detector and the physical maneuverability of the target drone.**

---

## 3. Dataset Discipline: Tuning Without Test Leakage

A common concern is that tuning the Kalman filter parameters requires finding additional real-world test videos. **This is not necessary.** The project's existing 3-way sequence-stratified split completely addresses this:

| Split | Sequences | Frames | Role in Experimental Pipeline |
| :--- | :--- | :--- | :--- |
| **Train** | 14 videos | 17,628 | Trains the YOLO neural network weights ($W, b$). |
| **Val** | 3 videos (`video04`, `video15`, `video13`) | 3,606 | **Tuning Ground:** Calibrate $R$, sweep $Q$, and optimize track thresholds. |
| **Test** | 3 videos (`video05`, `video12`, `video20`) | 3,570 | **Frozen Vault:** Run once for final, unbiased benchmark metrics in the thesis. |

### Procedure for Tuning on the Validation Set:
1. **Calibrate Measurement Noise ($R$):**
   * Run the frozen YOLO model across the 3 validation videos.
   * Match detections with ground truth using Hungarian/IoU assignment.
   * Compute the residual error variances:
     $$\sigma_{cx}^2 = \text{Var}(cx_{\text{det}} - cx_{\text{gt}}), \quad \sigma_{w}^2 = \text{Var}(w_{\text{det}} - w_{\text{gt}})$$
   * Directly assign $R = \text{diag}(\sigma_{cx}^2, \sigma_{cy}^2, \sigma_w^2, \sigma_h^2)$. This eliminates arbitrary guesswork.
2. **Sweep Process Noise ($Q$) & Life-Cycle Thresholds:**
   * Process noise represents target agility. On the validation set, sweep the velocity noise scale $\alpha$ ($Q_{4:6, 4:6} = \alpha \cdot I$) alongside track management thresholds (`max_lost_frames`, association IoU threshold).
   * Select the parameters that minimize Center Location Error (CLE) and minimize Jitter Index without causing track divergence during sharp turns.

---

## 4. Analysis of the Foveal Zoom Mechanism

### Why It Is Architecturally Valuable
1. **Mitigates Tiny-Target Pixel Decimation:** A drone at a distance of 40–60 meters may only occupy $12 \times 12$ pixels on a 1080p sensor. Resizing a full 1080p image down to $640 \times 640$ for YOLO destroys high-frequency spatial features (rotors, struts). Cropping native sensor pixels preserves full optical resolution.
2. **Compute Efficiency on Edge Hardware:** Running inference on full 4K or 1080p frames is too slow for the Jetson Orin Nano. Running inference on a dynamic $640 \times 640$ crop achieves high-acuity perception at **30+ FPS**.

### The Evaluation Bias Concern
The user's intuition is correct: **testing "YOLO + Kalman + Zoom" directly against "Raw YOLO" introduces a confounding variable.**
* The zoom mechanism **requires** the Kalman filter: on frame $t$, the system cannot know where to crop *before* running YOLO unless it uses the Kalman filter's forward prediction $\hat{x}_{t \mid t-1}$.
* If performance improves, one cannot discern whether the gain was driven by **kinematic state filtering** or by **increased pixel resolution** of the target.

### Real-World Operational Risks
1. **The "Tunnel Vision" / Cascading Failure Loop:** If the target drone executes an aggressive maneuver exceeding the crop margins, it exits the FOV in 1–2 frames. Because YOLO sees empty sky, the Kalman filter gets no updates and drifts until the snap-back failsafe resets the zoom.
2. **Chase Drone Ego-Motion:** In flight, angular rates (pitch/roll gusts) cause large pixel displacements. A $2.5\times$ zoom narrows the FOV from $\approx 80^\circ$ to $\approx 32^\circ$, dramatically increasing sensitivity to camera vibrations.

---

## 5. Methodological Solution: The 3-Stage Ablation Study

To eliminate confounding bias and turn this into a core academic contribution, the experimental benchmark will be structured as a **3-stage ablation study**:

```
[Config A: Raw YOLO]            Full FOV Frame ──► YOLO ──────────────────────────────► Raw Detections
                                                                                              │
[Config B: Filtered YOLO]         Full FOV Frame ──► YOLO ──► Kalman Filter ────────────► Filtered State
                                                                    ▲                         │
                                                                    │ (Prior State)           ▼
[Config C: Integrated Pipeline]   Foveal Crop   ──► YOLO ──► Kalman Filter ────────────► Zoomed State
                                 (Guided by KF)
```

| Configuration | Sensor Input | Temporal Filter | Scientific Question Answered |
| :--- | :--- | :--- | :--- |
| **Config A: Raw Baseline** | Full Frame (Wide FOV) | None (Raw YOLO) | How well does the spatial detector perform on its own? |
| **Config B: Isolated Filter** | Full Frame (Wide FOV) | 8D Kalman Filter | **What is the exact gain provided by temporal state filtering?** |
| **Config C: Integrated System** | Dynamic Foveal Crop | 8D Kalman Filter | What is the total gain when combining kinematic estimation with optical foveation? |

---

## 6. Evaluation Metrics Taxonomy

The test sequences (`video05`, `video12`, `video20`) will be evaluated across four distinct quantitative dimensions:

### 1. Accuracy: Center Location Error (CLE)
$$\text{CLE}_t = \sqrt{(cx_t - cx^*_t)^2 + (cy_t - cy^*_t)^2} \quad [\text{pixels}]$$
* Measures absolute tracking accuracy relative to ground truth.

### 2. Smoothness: Spatial Jitter Index (Second-Order Difference / Jerk)
$$\text{Jitter} = \frac{1}{T-2} \sum_{t=2}^{T-1} \| \mathbf{p}_{t+1} - 2\mathbf{p}_t + \mathbf{p}_{t-1} \| \quad \text{where } \mathbf{p}_t = [cx_t, cy_t]^T$$
* **Relevance for Visual Servoing:** High-frequency jitter creates massive derivative spikes ($\frac{de}{dt}$) in IBVS PID controllers, leading to motor saturation and oscillation. The Kalman filter must show a significant reduction in this metric.

### 3. Robustness: Coasting Recall Under Dropout
* Percentage of frames where ground truth exists but YOLO fails to detect the target ($\text{conf} < \tau_{\text{det}}$), yet the Kalman filter's dead-reckoning state maintains $\text{IoU} \ge 0.3$.

### 4. Overlap: Success Plot (AUC)
* Area Under Curve of the bounding box IoU curve across all thresholds $\tau \in [0.0, 1.0]$.

---

## 7. Next Actions for Implementation

1. [ ] **Build Validation Calibrator:** Create a script (`object-detection/calibrate_kalman.py`) that executes YOLO over `val` sequences and auto-computes the empirical covariance matrix $R$.
2. [ ] **Implement Ablation Harness:** Extend `track_pipeline.py` or create a benchmark runner (`benchmark_perception.py`) supporting an `--ablation` argument (`raw`, `kf`, `kf_zoom`).
3. [ ] **Generate Comparative Plots:** Produce side-by-side trajectory curves, jitter bar charts, and IoU success plots across the 3 test sequences for inclusion in Chapter 4 (Results) of the thesis.
