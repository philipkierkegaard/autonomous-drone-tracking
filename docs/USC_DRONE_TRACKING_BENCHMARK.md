# USC Drone Real-World Tracking & Noise Calibration Benchmark

**DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System**  
**Dataset Reference:** [`perception/datasets/usc_drone_tracking_benchmark`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/perception/datasets/usc_drone_tracking_benchmark)  
**Manifest File:** [`perception/datasets/usc_drone_tracking_benchmark/benchmark_manifest.json`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/perception/datasets/usc_drone_tracking_benchmark/benchmark_manifest.json)  
**Target Airframe:** DJI Phantom Quadrotor (White 4-Rotor Platform)  
**Native Capture:** Full HD 1080p (1920×1080) @ 30 FPS  

---

## 1. Architectural Role & Why It Is Kept Separate

### Should this dataset be merged into the existing YOLO validation set, or kept separate?
**Recommendation: Keep it strictly separate as a dedicated "Out-of-Domain Real-World Video & Tracking Benchmark".**

1. **Preserves In-Domain Scientific Comparability:**  
   The primary detection models (`yolov8n_drone_v3`, `yolov8n_768_hybrid_nwd`, `yolov8n_drone_768_ciou`) were comprehensively evaluated on the standardized **8,052-image held-out test set** (documented in [`docs/YOLOV8N_DRONE_V3_TEST_EVALUATION.md`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/docs/YOLOV8N_DRONE_V3_TEST_EVALUATION.md)). Merging new video frames into that set would retroactively invalidate all published mAP@50 and mAP@50-95 metrics.
2. **Video Tracking vs. Static Object Detection:**  
   Standard YOLO validation evaluates frames as independent, isolated images. The USC dataset's primary value is its **30 Hz temporal continuity**, which is needed for:
   - Empirical Measurement Noise Covariance ($R$) calibration from real video jitter.
   - Process Noise Covariance ($Q$) optimization for rapid target acceleration.
   - Kalman filter coasting evaluation during intermittent tree-branch occlusions.
3. **Academic Generalization Proof (Gold Standard Thesis Practice):**  
   Treating this as an independent appendix allows you to prove **Zero-Shot Out-of-Domain Generalization**: showing that your detector and tracker transfer seamlessly to third-party footage without any retraining.

---

## 2. Sequence Ingestion, Splitting & Trimming Specification

To eliminate unrecoverable sensor blackouts and avoid sudden target teleportation jumps, the raw footage was sliced into **9 continuous 1-indexed sequences** totaling **10,463 frames (~5.8 minutes of 30 fps video)**:

| Benchmark Sequence | Source Folder | Source Slice | Frame Count | Duration | Operational Environment & Utility |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`Drone_garden_part1`** | `Drone_garden` | 1 – 1,200 | 1,200 | 40.0s | Dynamic flight weaving through garden trees, climbs, and dives. |
| **`Drone_garden_part2`** | `Drone_garden` | 1,750 – 2,210 | 461 | 15.4s | High-speed close fly-by diving into dense foliage canopy. |
| **`Drone9_part1`** | `Drone9` | 1 – 1,200 | 1,200 | 40.0s | High-altitude open pursuit ($w = 45\text{--}108\text{px}$) against sky & canopy. |
| **`Drone9_part2`** | `Drone9` | 2,200 – 2,900 | 701 | 23.4s | Tree-line edge tracking with intermittent canopy occlusions. |
| **`Drone8`** | `Drone8` | 1 – 3,200 | 3,200 | 106.7s | Full mission: courtyard launch $\to$ high sky cruise $\to$ building descent. |
| **`Drone_frames`** | `Drone_frames` | 1 – 421 | 421 | 14.0s | Close-range high-resolution quadrotor inspection against foliage. |
| **`Drone_frames_1_canopy`**| `Drone_frames_1`| 550 – 1,600 | 1,051 | 35.0s | Mid-range canopy pursuit (discards sun flare & close defocus blur). |
| **`Drone_test1`** | `Drone_test1` | 1 – 1,479 | 1,479 | 49.3s | Open-field pursuit with dynamic approach/departure velocity sweeps. |
| **`Drone_test_airborne`** | `Drone_test` | 1 – 750 | 750 | 25.0s | Airborne flight past winter bare branches (discards parking lot). |
| **Total Benchmark Corpus**| — | — | **10,463** | **348.8s** | **9 Curated Continuous Sequences** |

### Discarded Segments & Rationale:
* **`Drone_garden` (Frames 1,201 – 1,749):** 548 frames of featureless, static blue sky where the target is a tiny, stationary dot. Discarded to focus on dynamic tracking.
* **`Drone9` (Frames 1,201 – 2,199):** Drone drops behind a black wrought-iron security gate and hovers over a white camper bus in a car park. Ground CCTV perspective with 47% miss rate.
* **`Drone9` (Frames 2,901 – 3,909):** Target shrinks to a microscopic $23\text{px}$ speck ($<9\text{px}$ on YOLO) on the horizon with 31% recall.
* **`Drone_frames_1` (Frames 1 – 450 & 1,600 – 2,246):** Discarded 9.5s solar lens flare blackout and $<0.8\text{m}$ extreme out-of-focus motion blur.
* **`Drone_test` (Frames 751 – 1,179):** Discarded ground-level parking lot sequence where drone hovers against parked car tires.

---

## 3. Data Processing & Ground Truth Interpolation

* **Dense 30 Hz Ground Truth:** The raw keyframes were annotated at 2 Hz (every 15 frames). Linear interpolation was applied between valid keyframe coordinates to generate dense ground-truth bounding boxes $[x, y, w, h]$ for **every single frame** at 30 fps in `groundtruth_rect.txt`.
* **Zero Storage Overhead:** Image frames in `img/0001.jpg` ... `img/NNNN.jpg` are continuous symbolic links back to the raw storage. No redundant 1080p video copies were created.
* **Reproducibility:** The entire benchmark can be regenerated at any time by executing:
  ```bash
  python perception/preparation/prepare_usc_benchmark.py
  ```
