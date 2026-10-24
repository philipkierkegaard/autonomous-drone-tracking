# Perception Evaluation: Zero-Leakage Video Split Benchmark
## YOLOv8n Hybrid NWD Base (Zero-Shot) vs. YOLOv8n Video-Split Fine-Tuned Specialist

**DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System**  
**Date:** October 2026  
**Author:** Philip Kierkegaard  
**Scope:** Perception Subsystem (`perception/datasets/nougat_pede_video_split/`, `benchmarks/perception/`)  
**Target Hardware:** NVIDIA Jetson Orin Nano (MAXN Power Mode, TensorRT 10.3 FP16)  

---

## 1. Executive Summary & Scientific Motivation

### 1.1 The Video Data Leakage Problem
In aerial drone detection datasets extracted from continuous video streams, default random frame-level splitting (e.g. standard Roboflow 70/20/10 random partitioning) creates **severe temporal and spatial data leakage**. 

When consecutive frames from the same 30/60 FPS flight session are distributed across both `train` and `test` splits, the model is tested on near-identical backgrounds, lighting, airframe poses, and camera angles that it saw milliseconds earlier during training. As proven in our preliminary investigation, this caused an inflated score of **$96.7\%$ mAP@50**, measuring sequence memorization rather than true generalizability.

### 1.2 Zero-Leakage Video Partitioning
To produce a methodologically rigorous, scientifically defensible evaluation for the bachelor thesis, the dataset was completely repartitioned at the **video level** (Group/Sequence Split) in [`perception/datasets/nougat_pede_video_split/`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/perception/datasets/nougat_pede_video_split/):

- **Train Split (313 frames):** `nougat-flyver-drone-kun-drone-`, `drone_vs_trees_1`, `flight_raw_20261002`, `raw_flight_in_front_of_house`, and 40 background negative frames.
- **Validation Split (43 frames):** `drone_vs_trees_2` (distant tree targets, mean $45\text{ px}$) and 14 background negative frames.
- **Held-Out Test Split (161 frames):** **`pede-flyver-drone`** (122 target frames, 25 background frames) and 14 background negative frames (39 total negatives).

The test sequence represents a completely unseen pilot, trajectory, and session. A new model was fine-tuned directly from the **`yolov8n_drone_768_hybrid_nwd_c51_best.pt`** base checkpoint on the clean training set using Hybrid NWD loss ($\alpha=0.5, C=51.3$) and evaluated side-by-side with the zero-shot base model.

---

## 2. Master Comparative Benchmark (Zero-Leakage Test Split)

| Metric / Evaluation Domain | Base YOLOv8n Hybrid NWD (Zero-Shot) | Fine-Tuned YOLOv8n (Video-Split Specialist) | Delta / Fine-Tuning Impact |
| :--- | :---: | :---: | :--- |
| **Model Weights File** | [`yolov8n_drone_768_hybrid_nwd_c51_best.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/perception/weights/yolov8n_drone_768_hybrid_nwd_c51_best.pt) | [`yolov8n_nougat_pede_768_video_split_best.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/perception/weights/yolov8n_nougat_pede_768_video_split_best.pt) | Published weights |
| **Pretrained Initialization** | Anti-UAV + Det-Fly (COCO/Public) | Hybrid NWD 768 Base Checkpoint | Initialized from generalist |
| **Evaluation Sequence** | `pede-flyver-drone` (Held-Out) | `pede-flyver-drone` (Held-Out) | Strictly unseen flight video |
| **Held-Out Test mAP@50 (161 images)** | $47.27\%$ | **$85.33\%$** | **$+38.06\%$ absolute gain on unseen flight** |
| **Held-Out Test mAP@50-95** | $29.74\%$ | **$53.05\%$** | **$+23.31\%$ tighter localization** |
| **Operational Precision (@ conf 0.25)** | $62.34\%$ | **$93.48\%$** | **$+31.14\%$ higher precision** |
| **Operational Recall (@ conf 0.25)** | $39.34\%$ ($48 / 122$) | **$70.49\%$ ($86 / 122$)** | **$+31.15\%$ higher catch rate** |
| **Operational $F_1$ Score (@ conf 0.25)** | $0.482$ | **$0.804$** | **$+0.322$ balanced accuracy** |
| **Small Target Recall ($35\text{--}60\text{ px}$, @ conf 0.25)** | $20.00\%$ ($7 / 35$) | **$60.00\%$ ($21 / 35$)** | **$3\times$ catch rate on small/distant drone** |
| **Medium Target Recall ($> 60\text{ px}$, @ conf 0.25)** | $47.13\%$ ($41 / 87$) | **$74.71\%$ ($65 / 87$)** | **$+27.58\%$ catch rate on close drone** |
| **Matched Center Location Error (CLE @ conf 0.25)** | $2.86\text{ px}$ (Median $2.41\text{ px}$) | **$2.47\text{ px}$ (Median $1.83\text{ px}$)** | **$0.39\text{ px}$ closer centroid alignment** |
| **Mean Bounding-Box IoU (@ conf 0.25)** | $0.815$ | **$0.813$** | High aspect-ratio preservation |
| **False Alarm Rate on Background (39 frames, @ conf 0.25)** | **$5.13\%$** ($2 / 39$) | $7.69\%$ ($3 / 39$) | Both maintain $<8\%$ false alarms |
| **False Alarm Rate on Background (39 frames, @ conf 0.50)** | **$2.56\%$** ($1 / 39$) | **$2.56\%$** ($1 / 39$) | Near-zero false alarms at high conf |
| **Validation Split mAP@50 (`drone_vs_trees_2`)** | $81.81\%$ | **$84.78\%$** | $+2.97\%$ validation mAP |
| **Validation Split mAP@50-95** | $49.96\%$ | **$55.53\%$** | $+5.57\%$ validation mAP@50-95 |
| **Inference Latency (MPS, Batch=1)** | $9.17\text{ ms}$ ($109.1\text{ FPS}$) | $9.75\text{ ms}$ ($102.5\text{ FPS}$) | Real-time edge throughput |

---

## 3. Benchmark Visualizations

### 3.1 Empirical Performance Metrics Comparison
![Empirical Performance Metrics Comparison](/Users/philipkierkegaard/.gemini/antigravity-ide/brain/5967d853-7a7e-4cdd-8792-4a77e6d704a2/video_split_model_comparison.png)

### 3.2 Visual Detection Qualitative Comparison
![Visual Qualitative Comparison](/Users/philipkierkegaard/.gemini/antigravity-ide/brain/5967d853-7a7e-4cdd-8792-4a77e6d704a2/video_split_detection_examples.png)

---

## 4. Key Engineering Insights

### 4.1 True Domain Generalization vs. Memorization
- Under the original corrupted random split, the fine-tuned model scored $96.7\%$ mAP@50 because the test set was populated by adjacent video frames from the training set.
- Under the strict zero-leakage video split, the base model achieved **$47.3\%$ mAP@50** in true zero-shot transfer, while the video-split fine-tuned specialist achieved **$85.3\%$ mAP@50**.
- This proves that the fine-tuning process genuinely adapted the detector to the target environment: **$+38.1\%$ mAP@50**, **$+31.1\%$ precision**, and **$+31.2\%$ recall** across an entirely unseen flight path and pilot.

### 4.2 Distant & Small Target Advantage
On small targets ($35\text{--}60\text{ px}$ max dimension):
- The zero-shot base model caught only **$20.0\%$** ($7 / 35$) of targets.
- The fine-tuned specialist caught **$60.0\%$** ($21 / 35$), demonstrating a **$3\times$ improvement in small-target detection range**.

### 4.3 Qualitative Distractor Discrimination (Pilot vs. Drone)
As demonstrated in the qualitative comparison grid (middle row of [`video_split_detection_examples.png`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/outputs/benchmarks/perception/video_split_detection_examples.png)):
- When the pilot is standing in the garden holding the transmitter, the zero-shot base model produced a false positive detection on the human pilot's torso (`Base: 0.29`) and missed the low-hovering drone.
- The fine-tuned specialist completely suppressed the human distractor and accurately locked onto the hovering drone (`Fine-Tuned: 0.59`).

---

## 5. Deployment Recommendation

Deploy **[`yolov8n_nougat_pede_768_video_split_best.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/perception/weights/yolov8n_nougat_pede_768_video_split_best.pt)** for all live tracking and flight interception trials.  
- **Operational Confidence Threshold:** Set $\text{conf} = 0.25\text{--}0.35$ for tracking acquisition.
- **Reproducibility:** Re-run the zero-leakage benchmark at any time via:
  ```bash
  ./.venv/bin/python benchmarks/perception/benchmark_video_split_model_comparison.py
  ```
