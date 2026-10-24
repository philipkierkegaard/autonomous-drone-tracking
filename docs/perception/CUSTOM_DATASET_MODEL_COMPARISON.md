# Perception Evaluation: YOLOv8n Hybrid NWD (Generalist) vs. YOLOv8n Nougat Pede (Custom Specialist)

**DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System**  
**Date:** October 2026  
**Author:** Philip Kierkegaard  
**Scope:** Perception Subsystem (`perception/raw_data/nougat-og-pede-flyver-drone/`, `benchmarks/perception/`)  
**Target Hardware:** NVIDIA Jetson Orin Nano (MAXN Power Mode, TensorRT 10.3 FP16)  

---

## 1. Executive Summary & Master Comparative Benchmark

This benchmark evaluates the domain adaptation performance between two key detector models on the user's custom-captured flight dataset:

1. **YOLOv8n Hybrid NWD (`yolov8n_drone_768_hybrid_nwd_c51_best.pt`):** The generalist detector trained on extensive public aerial benchmarks (Anti-UAV, Det-Fly 010/020) with $768\times 768$ input and Normalized Wasserstein Distance (NWD) loss ($\alpha=0.5, C=51.3\text{ px}$).
2. **YOLOv8n Nougat Pede (`yolov8n_nougat_pede_768_best.pt`):** The fine-tuned specialist detector trained directly on the user's custom-captured drone imagery (`nougat-og-pede-flyver-drone`).

The evaluation was executed on Apple Silicon MPS using the custom benchmark harness in [`benchmarks/perception/benchmark_custom_dataset_model_comparison.py`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/benchmarks/perception/benchmark_custom_dataset_model_comparison.py).

### Master Comparative Results

| Metric / Evaluation Domain | YOLOv8n Hybrid NWD (Generalist) | YOLOv8n Nougat Pede (Specialist) | Delta / Specialist Advantage |
| :--- | :---: | :---: | :--- |
| **Model Weights File** | [`yolov8n_drone_768_hybrid_nwd_c51_best.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/perception/weights/yolov8n_drone_768_hybrid_nwd_c51_best.pt) | [`yolov8n_nougat_pede_768_best.pt`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/perception/weights/yolov8n_nougat_pede_768_best.pt) | — |
| **Training Pretext / Regime** | Anti-UAV + Det-Fly (Generalist) | Custom Drone Capture (Fine-Tuned) | Domain-adapted on target environment |
| **Inference Resolution** | $768 \times 768$ | $768 \times 768$ | Identical input dimensions |
| **Held-Out Test mAP@50 (51 images)** | $56.16\%$ | **$96.74\%$** | **$+40.58\%$ absolute increase** |
| **Held-Out Test mAP@50-95** | $37.23\%$ | **$63.20\%$** | **$+25.97\%$ tighter bounding box fit** |
| **Held-Out Test Recall (Catch Rate)** | $56.98\%$ | **$94.87\%$** | **$+37.89\%$ higher target recall** |
| **Held-Out Test Precision** | $58.13\%$ | **$99.66\%$** | **$+41.53\%$ precision on valid targets** |
| **Validation Split mAP@50 (105 images)** | $40.63\%$ | **$94.63\%$** | **$+54.00\%$ validation mAP@50** |
| **Validation Split mAP@50-95** | $24.65\%$ | **$59.48\%$** | **$+34.83\%$ validation mAP@50-95** |
| **Validation Split Recall** | $37.66\%$ | **$88.31\%$** | **$+50.65\%$ validation recall** |
| **Validation Split Precision** | $59.96\%$ | **$98.43\%$** | **$+38.47\%$ validation precision** |
| **Matched Center Location Error (CLE @ conf 0.25)** | $3.39\text{ px}$ (Median $1.69\text{ px}$) | **$2.11\text{ px}$ (Median $1.85\text{ px}$)** | **$1.28\text{ px}$ closer centroid alignment** |
| **Mean Bounding-Box IoU (@ conf 0.25)** | $0.805$ | **$0.823$** | Tighter aspect-ratio preservation |
| **Small Targets Recall ($35\text{--}60\text{ px}$, @ conf 0.25)** | $53.85\%$ ($7 / 13$) | **$100.00\%$ ($13 / 13$)** | **$+46.15\%$ catch rate on distant drone** |
| **Medium Targets Recall ($> 60\text{ px}$, @ conf 0.25)** | $61.54\%$ ($16 / 26$) | **$92.31\%$ ($24 / 26$)** | **$+30.77\%$ catch rate on close drone** |
| **Background False Alarm Rate (@ conf 0.25)** | **$8.33\%$** ($1 / 12$ frames) | $16.67\%$ ($2 / 12$ frames) | Generalist slightly more conservative |
| **Background False Alarm Rate (@ conf 0.50)** | **$0.00\%$** ($0 / 12$ frames) | **$0.00\%$** ($0 / 12$ frames) | **Zero hallucinations at conf $\ge 0.50$** |
| **Recall @ conf 0.50** | $38.46\%$ ($15 / 39$) | **$94.87\%$ ($37 / 39$)** | Nougat retains full recall at high conf |
| **Inference Latency (MPS, Batch=1)** | $9.26\text{ ms}$ ($108\text{ FPS}$) | $9.40\text{ ms}$ ($106\text{ FPS}$) | Identical backbone architecture |

---

## 2. Benchmark Visualizations

### 2.1 Empirical Performance Metrics Comparison
![Empirical Performance Metrics Comparison](/Users/philipkierkegaard/.gemini/antigravity-ide/brain/5967d853-7a7e-4cdd-8792-4a77e6d704a2/nougat_pede_model_comparison.png)

### 2.2 Visual Detection Qualitative Comparison
![Visual Qualitative Comparison](/Users/philipkierkegaard/.gemini/antigravity-ide/brain/5967d853-7a7e-4cdd-8792-4a77e6d704a2/nougat_pede_detection_examples.png)

---

## 3. Dataset Integrity & Characterization

The user-captured dataset is located in [`perception/raw_data/nougat-og-pede-flyver-drone/`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/perception/raw_data/nougat-og-pede-flyver-drone/) and configured by [`data.yaml`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/perception/raw_data/nougat-og-pede-flyver-drone/data.yaml).

### 3.1 Dataset Verification Findings
A thorough verification check of the dataset revealed:
1. **Directory Structure:** Standard YOLO format with `train/images`, `train/labels`, `valid/images`, `valid/labels`, `test/images`, and `test/labels`.
2. **File Alignment:** Every label file has a matching image file.
3. **Bounding Box Integrity:** All bounding box coordinates $(c_x, c_y, w, h)$ are strictly normalized within $[0.0, 1.0]$.
4. **Class Consistency:** Single class (`0: drone`) throughout all splits (`nc: 1`).
5. **Frame Resolution:** All frames are uniform $1280 \times 720$ (native camera capture resolution).
6. **Negative Background Inclusion:** Crucially, the dataset contains authentic background negative images (frames with zero drone instances, empty `.txt` files), allowing realistic measurement of edge false alarms.

### 3.2 Dataset Split Breakdown

| Split | Images | Drone Annotations | Pure Negative Frames | Negative Ratio | Native Resolution |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 361 | 266 | 95 | $26.3\%$ | $1280 \times 720$ |
| **Valid** | 105 | 77 | 28 | $26.7\%$ | $1280 \times 720$ |
| **Test** | 51 | 39 | 12 | $23.5\%$ | $1280 \times 720$ |
| **Total** | **517** | **382** | **135** | **$26.1\%$** | **$1280 \times 720$** |

---

## 4. Key Engineering Insights

### 4.1 Significant Domain Shift from Anti-UAV
The generalist model (`yolov8n_drone_768_hybrid_nwd_c51_best.pt`) achieves $76.51\%$ mAP@50 on the public Anti-UAV benchmark, but drops to **$56.16\%$** on the test split and **$40.63\%$** on the validation split of the custom dataset. 

This drop is attributable to three key environmental factors:
- **Background Clutter:** The custom imagery features complex Danish residential and agricultural scenery (dense green tree canopies, wooden fencing, garden sheds, foliage edges) rather than high-contrast open sky or thermal IR imagery.
- **Drone Form Factor & Scale:** The user's quadcopter airframe and propeller geometry differ from the military and industrial UAVs predominant in public benchmarks.
- **Camera Optics & Color Grading:** The natural lens distortion, white balance, and contrast curve of the test drone camera produce distinctive feature activations.

### 4.2 Fine-Tuned Specialist Dominance
The fine-tuned model (`yolov8n_nougat_pede_768_best.pt`) demonstrates near-perfect domain adaptation:
- **mAP@50 increases from $56.16\%$ to $96.74\%$** on test (+40.58%), and from $40.63\%$ to $94.63\%$ on valid (+54.00%).
- **Small Target Robustness:** On small distant targets ($35\text{--}60\text{ px}$), the specialist achieves **$100\%$ recall** ($13/13$), compared to only $53.85\%$ ($7/13$) for the generalist.
- **High Confidence Persistence:** At $\text{conf}=0.50$, the generalist drops to $38.46\%$ recall, while Nougat Pede maintains a flawless **$94.87\%$ recall** with **$97.37\%$ precision** and **$0\%$ background false alarms**.

### 4.3 Center Location Error (CLE) & Tracking Gating
Both models exhibit sub-pixel to few-pixel localization accuracy when a detection is matched:
- **Nougat Pede Mean CLE:** $2.11\text{ px}$ (Median $1.85\text{ px}$)
- **Hybrid NWD Mean CLE:** $3.39\text{ px}$ (Median $1.69\text{ px}$)
- **Implication for Kalman Filtering:** Both models provide centroid measurements with error well within the standard Kalman innovation gating radius ($R \approx 15\text{--}30\text{ px}$). However, because Nougat Pede provides far more continuous detections ($94.9\%$ vs $57.0\%$ recall), the Kalman filter is drastically less prone to coasting on pure motion prediction or suffering track dropouts.

---

## 5. Deployment Recommendations for Autonomous Pursuit

1. **Primary Interception Detector:**  
   Deploy **`yolov8n_nougat_pede_768_best.pt`** as the primary detection engine when operating in the target flight test environment.
2. **Operational Confidence Threshold:**  
   Set `conf_threshold = 0.35` (or up to `0.50`). At `0.50`, Nougat Pede completely suppresses background false alarms ($0\%$ false alarm rate on trees/sheds) while preserving $94.87\%$ target recall.
3. **Benchmarking Script:**  
   The evaluation can be re-run at any time using:
   ```bash
   ./.venv/bin/python benchmarks/perception/benchmark_custom_dataset_model_comparison.py
   ```
4. **Artifact Locations:**  
   - Benchmark script: [`benchmarks/perception/benchmark_custom_dataset_model_comparison.py`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/benchmarks/perception/benchmark_custom_dataset_model_comparison.py)
   - JSON Metrics: [`outputs/benchmarks/perception/nougat_pede_model_comparison_benchmark.json`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/outputs/benchmarks/perception/nougat_pede_model_comparison_benchmark.json)
   - Metrics Chart: [`outputs/benchmarks/perception/nougat_pede_model_comparison.png`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/outputs/benchmarks/perception/nougat_pede_model_comparison.png)
   - Visual Detection Chart: [`outputs/benchmarks/perception/nougat_pede_detection_examples.png`](file:///Users/philipkierkegaard/Development/autonomous-drone-tracking/outputs/benchmarks/perception/nougat_pede_detection_examples.png)
