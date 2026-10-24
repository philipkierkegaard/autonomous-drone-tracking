#!/usr/bin/env python3
"""
Benchmark: YOLOv8n Hybrid NWD (Anti-UAV/Det-Fly Generalist) vs. YOLOv8n Nougat Pede (Custom Fine-Tuned)
Evaluates on the user's custom-captured dataset: perception/raw_data/nougat-og-pede-flyver-drone/

DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception.
"""

import sys
import time
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO

DATA_DIR = REPO_ROOT / "perception" / "raw_data" / "nougat-og-pede-flyver-drone"
DATA_YAML = DATA_DIR / "data.yaml"

WEIGHT_HYBRID = REPO_ROOT / "perception" / "weights" / "yolov8n_drone_768_hybrid_nwd_c51_best.pt"
WEIGHT_NOUGAT = REPO_ROOT / "perception" / "weights" / "yolov8n_nougat_pede_768_best.pt"

OUT_DIR = REPO_ROOT / "outputs" / "benchmarks" / "perception"
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_OUT = OUT_DIR / "nougat_pede_model_comparison_benchmark.json"
FIG_METRICS = OUT_DIR / "nougat_pede_model_comparison.png"
FIG_VISUAL = OUT_DIR / "nougat_pede_detection_examples.png"
ARTIFACT_DIR = Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/5967d853-7a7e-4cdd-8792-4a77e6d704a2")


def compute_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    boxAArea = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return float(interArea / (boxAArea + boxBArea - interArea))


def load_dataset_gt(split: str = "test") -> List[Dict[str, Any]]:
    """Loads all image paths, image sizes, and bounding box ground truths."""
    img_dir = DATA_DIR / split / "images"
    lbl_dir = DATA_DIR / split / "labels"

    records = []
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))

    for img_p in img_files:
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        gt_boxes = []

        im = cv2.imread(str(img_p))
        if im is None:
            continue
        h, w = im.shape[:2]

        if lbl_p.exists():
            with open(lbl_p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = [float(v) for v in parts[1:5]]
                    # Convert normalized xywh to absolute pixel xyxy
                    x1 = (cx - bw / 2.0) * w
                    y1 = (cy - bh / 2.0) * h
                    x2 = (cx + bw / 2.0) * w
                    y2 = (cy + bh / 2.0) * h
                    gt_boxes.append({
                        "cls": cls_id,
                        "box": np.array([x1, y1, x2, y2], dtype=np.float32),
                        "norm_box": [cx, cy, bw, bh],
                        "max_dim_px": max(bw * w, bh * h),
                        "area_px": (bw * w) * (bh * h)
                    })

        records.append({
            "stem": img_p.stem,
            "img_path": img_p,
            "width": w,
            "height": h,
            "gt_boxes": gt_boxes,
            "is_background": len(gt_boxes) == 0
        })

    return records


def evaluate_model_granular(
    model: YOLO,
    records: List[Dict[str, Any]],
    conf_thresholds: List[float] = [0.15, 0.25, 0.50],
    match_iou_thresh: float = 0.30,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
) -> Dict[str, Any]:
    """Runs granular target-level evaluation, scale-binning, and FP counting."""
    results_by_conf = {}

    for conf in conf_thresholds:
        total_gt = 0
        total_matched = 0
        total_fp = 0
        bg_images_with_fp = 0
        total_bg_images = 0

        cle_list = []
        iou_list = []

        # Scale bins: tiny (<= 35px), small (35-60px), medium (> 60px)
        scale_stats = {
            "tiny": {"gt": 0, "matched": 0},
            "small": {"gt": 0, "matched": 0},
            "medium": {"gt": 0, "matched": 0}
        }

        infer_times = []

        for rec in records:
            img = cv2.imread(str(rec["img_path"]))
            t0 = time.perf_counter()
            preds = model.predict(img, imgsz=768, conf=conf, iou=0.45, device=device, verbose=False)
            t_infer = (time.perf_counter() - t0) * 1000.0
            infer_times.append(t_infer)

            pred_boxes = []
            if len(preds) > 0 and len(preds[0].boxes) > 0:
                p_xyxy = preds[0].boxes.xyxy.cpu().numpy()
                p_conf = preds[0].boxes.conf.cpu().numpy()
                for b, c in zip(p_xyxy, p_conf):
                    pred_boxes.append({"box": b, "conf": float(c), "matched": False})

            gts = rec["gt_boxes"]
            if rec["is_background"]:
                total_bg_images += 1
                if len(pred_boxes) > 0:
                    bg_images_with_fp += 1
                    total_fp += len(pred_boxes)
            else:
                for gt in gts:
                    total_gt += 1
                    dim = gt["max_dim_px"]
                    if dim <= 35.0:
                        bin_name = "tiny"
                    elif dim <= 60.0:
                        bin_name = "small"
                    else:
                        bin_name = "medium"
                    scale_stats[bin_name]["gt"] += 1

                    best_iou = 0.0
                    best_idx = -1
                    for i, pb in enumerate(pred_boxes):
                        if pb["matched"]:
                            continue
                        iou = compute_iou(pb["box"], gt["box"])
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = i
                    
                    if best_iou >= match_iou_thresh and best_idx >= 0:
                        pred_boxes[best_idx]["matched"] = True
                        total_matched += 1
                        scale_stats[bin_name]["matched"] += 1

                        p_box = pred_boxes[best_idx]["box"]
                        g_box = gt["box"]
                        pcx = (p_box[0] + p_box[2]) / 2.0
                        pcy = (p_box[1] + p_box[3]) / 2.0
                        gcx = (g_box[0] + g_box[2]) / 2.0
                        gcy = (g_box[1] + g_box[3]) / 2.0
                        cle = np.sqrt((pcx - gcx) ** 2 + (pcy - gcy) ** 2)

                        cle_list.append(float(cle))
                        iou_list.append(float(best_iou))

                # Any unmatched prediction is a false positive
                for pb in pred_boxes:
                    if not pb["matched"]:
                        total_fp += 1

        recall = (total_matched / max(1, total_gt)) * 100.0
        precision = (total_matched / max(1, total_matched + total_fp)) * 100.0
        f1 = (2 * precision * recall) / max(1e-4, precision + recall)

        results_by_conf[f"conf_{conf:.2f}"] = {
            "conf_threshold": conf,
            "total_gt": total_gt,
            "total_matched": total_matched,
            "total_fp": total_fp,
            "recall_pct": round(recall, 2),
            "precision_pct": round(precision, 2),
            "f1_score": round(f1 / 100.0, 3),
            "bg_images": total_bg_images,
            "bg_images_with_fp": bg_images_with_fp,
            "bg_fp_rate_pct": round((bg_images_with_fp / max(1, total_bg_images)) * 100.0, 2),
            "mean_cle_px": round(float(np.mean(cle_list)), 2) if cle_list else float("nan"),
            "median_cle_px": round(float(np.median(cle_list)), 2) if cle_list else float("nan"),
            "mean_iou": round(float(np.mean(iou_list)), 3) if iou_list else float("nan"),
            "scale_breakdown": {
                k: {
                    "gt": v["gt"],
                    "matched": v["matched"],
                    "recall_pct": round((v["matched"] / max(1, v["gt"])) * 100.0, 2)
                } for k, v in scale_stats.items()
            },
            "mean_inference_ms": round(float(np.mean(infer_times)), 2),
            "fps": round(1000.0 / max(0.01, float(np.mean(infer_times))), 1)
        }

    return results_by_conf


def run_benchmark():
    print("=" * 85)
    print("BENCHMARK: YOLOv8n Hybrid NWD vs. YOLOv8n Nougat Pede (Custom Dataset)")
    print(f"Dataset Location: {DATA_DIR}")
    print(f"Data YAML:        {DATA_YAML}")
    print(f"Model A (Hybrid): {WEIGHT_HYBRID.name}")
    print(f"Model B (Nougat): {WEIGHT_NOUGAT.name}")
    print("=" * 85)

    assert DATA_YAML.exists(), f"Missing dataset yaml: {DATA_YAML}"
    assert WEIGHT_HYBRID.exists(), f"Missing weight: {WEIGHT_HYBRID}"
    assert WEIGHT_NOUGAT.exists(), f"Missing weight: {WEIGHT_NOUGAT}"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"---> Evaluation Device: {device.upper()}")

    # 1. Dataset Verification
    records_test = load_dataset_gt("test")
    records_valid = load_dataset_gt("valid")
    records_train = load_dataset_gt("train")

    dataset_summary = {
        "test": {
            "images": len(records_test),
            "drones": sum(len(r["gt_boxes"]) for r in records_test),
            "backgrounds": sum(1 for r in records_test if r["is_background"]),
            "resolution": "1280x720"
        },
        "valid": {
            "images": len(records_valid),
            "drones": sum(len(r["gt_boxes"]) for r in records_valid),
            "backgrounds": sum(1 for r in records_valid if r["is_background"]),
            "resolution": "1280x720"
        },
        "train": {
            "images": len(records_train),
            "drones": sum(len(r["gt_boxes"]) for r in records_train),
            "backgrounds": sum(1 for r in records_train if r["is_background"]),
            "resolution": "1280x720"
        }
    }
    print("\n---> Dataset Summary:")
    for k, v in dataset_summary.items():
        print(f"     [{k.upper():<5}]: {v['images']:3d} images ({v['resolution']}), {v['drones']:3d} drone instances, {v['backgrounds']:2d} background negatives")

    # 2. Load Models
    print("\n---> Loading Models...")
    model_hybrid = YOLO(str(WEIGHT_HYBRID))
    model_nougat = YOLO(str(WEIGHT_NOUGAT))

    # 3. Standard COCO Evaluation via Ultralytics val() on TEST split
    print("\n---> Running standard Ultralytics val() on [TEST] split...")
    val_hybrid_test = model_hybrid.val(data=str(DATA_YAML), split="test", imgsz=768, device=device, verbose=False)
    val_nougat_test = model_nougat.val(data=str(DATA_YAML), split="test", imgsz=768, device=device, verbose=False)

    print("\n---> Running standard Ultralytics val() on [VALID] split...")
    val_hybrid_val = model_hybrid.val(data=str(DATA_YAML), split="val", imgsz=768, device=device, verbose=False)
    val_nougat_val = model_nougat.val(data=str(DATA_YAML), split="val", imgsz=768, device=device, verbose=False)

    standard_eval = {
        "test_split": {
            "hybrid_nwd": {
                "map50": round(float(val_hybrid_test.box.map50) * 100.0, 2),
                "map50_95": round(float(val_hybrid_test.box.map) * 100.0, 2),
                "precision": round(float(val_hybrid_test.box.mp) * 100.0, 2),
                "recall": round(float(val_hybrid_test.box.mr) * 100.0, 2)
            },
            "nougat_pede": {
                "map50": round(float(val_nougat_test.box.map50) * 100.0, 2),
                "map50_95": round(float(val_nougat_test.box.map) * 100.0, 2),
                "precision": round(float(val_nougat_test.box.mp) * 100.0, 2),
                "recall": round(float(val_nougat_test.box.mr) * 100.0, 2)
            }
        },
        "valid_split": {
            "hybrid_nwd": {
                "map50": round(float(val_hybrid_val.box.map50) * 100.0, 2),
                "map50_95": round(float(val_hybrid_val.box.map) * 100.0, 2),
                "precision": round(float(val_hybrid_val.box.mp) * 100.0, 2),
                "recall": round(float(val_hybrid_val.box.mr) * 100.0, 2)
            },
            "nougat_pede": {
                "map50": round(float(val_nougat_val.box.map50) * 100.0, 2),
                "map50_95": round(float(val_nougat_val.box.map) * 100.0, 2),
                "precision": round(float(val_nougat_val.box.mp) * 100.0, 2),
                "recall": round(float(val_nougat_val.box.mr) * 100.0, 2)
            }
        }
    }

    # 4. Granular Analysis on Test Set (51 images)
    print("\n---> Running Granular Evaluation on [TEST] Split...")
    granular_hybrid = evaluate_model_granular(model_hybrid, records_test, conf_thresholds=[0.15, 0.25, 0.50], device=device)
    granular_nougat = evaluate_model_granular(model_nougat, records_test, conf_thresholds=[0.15, 0.25, 0.50], device=device)

    # 5. Compile Master Benchmark Results
    master_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": "nougat-og-pede-flyver-drone",
        "dataset_summary": dataset_summary,
        "models": {
            "hybrid_nwd": {
                "weights_file": WEIGHT_HYBRID.name,
                "description": "YOLOv8n trained on Anti-UAV & Det-Fly with 768px Hybrid NWD-CIoU loss"
            },
            "nougat_pede": {
                "weights_file": WEIGHT_NOUGAT.name,
                "description": "YOLOv8n fine-tuned directly on user custom drone data (768px)"
            }
        },
        "standard_validation_metrics": standard_eval,
        "granular_test_evaluation": {
            "hybrid_nwd": granular_hybrid,
            "nougat_pede": granular_nougat
        }
    }

    with open(JSON_OUT, "w") as f:
        json.dump(master_payload, f, indent=2)
    print(f"\n[SAVED] Benchmark results saved to: {JSON_OUT}")

    if ARTIFACT_DIR.exists():
        shutil.copy2(JSON_OUT, ARTIFACT_DIR / JSON_OUT.name)

    # 6. Generate Publication Comparison Chart
    print("\n---> Generating Publication Comparison Figures...")
    generate_comparison_chart(standard_eval, granular_hybrid, granular_nougat)
    generate_qualitative_visuals(model_hybrid, model_nougat, records_test)

    return master_payload


def generate_comparison_chart(standard_eval: Dict[str, Any], granular_hybrid: Dict[str, Any], granular_nougat: Dict[str, Any]):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), dpi=220, facecolor="#0B0F19")
    fig.suptitle(
        "Empirical Model Evaluation: Hybrid NWD Generalist vs. Nougat-Pede Fine-Tuned Specialist\n"
        "Held-Out Test Set (51 Images, 1280×720 Native Resolution — Custom Captured Drone Data)",
        fontsize=13.5, weight="bold", color="#F8FAFC", y=1.02
    )

    models = ["Hybrid NWD (Generalist)", "Nougat-Pede (Specialist)"]
    x = np.arange(len(models))
    colors = ["#EF4444", "#8B5CF6"]

    # Panel 1: Detection mAP50 and mAP50-95
    ax1 = axes[0]
    ax1.set_facecolor("#111827")
    m50 = [standard_eval["test_split"]["hybrid_nwd"]["map50"], standard_eval["test_split"]["nougat_pede"]["map50"]]
    m50_95 = [standard_eval["test_split"]["hybrid_nwd"]["map50_95"], standard_eval["test_split"]["nougat_pede"]["map50_95"]]
    w = 0.35
    b1 = ax1.bar(x - w/2, m50, w, label="mAP@50 (%)", color="#38BDF8", edgecolor="#1E293B", linewidth=1.2)
    b2 = ax1.bar(x + w/2, m50_95, w, label="mAP@50-95 (%)", color="#818CF8", edgecolor="#1E293B", linewidth=1.2)
    ax1.set_title("Detection Accuracy (COCO Benchmark)", fontsize=11, weight="bold", color="#E2E8F0", pad=10)
    ax1.set_ylabel("Metric (%)", fontsize=10, color="#94A3B8")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9.5, weight="bold", color="#CBD5E1")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", linestyle="--", alpha=0.25, color="#475569")
    ax1.legend(loc="upper left", facecolor="#1F2937", edgecolor="#374151", fontsize=8.5, labelcolor="#F1F5F9")
    for bar in b1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8.5, weight="bold", color="#38BDF8")
    for bar in b2:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8.5, weight="bold", color="#818CF8")

    # Panel 2: Precision & Recall at Operational Operating Point (conf=0.25)
    ax2 = axes[1]
    ax2.set_facecolor("#111827")
    h25 = granular_hybrid["conf_0.25"]
    n25 = granular_nougat["conf_0.25"]
    prec = [h25["precision_pct"], n25["precision_pct"]]
    rec = [h25["recall_pct"], n25["recall_pct"]]
    b3 = ax2.bar(x - w/2, prec, w, label="Precision (%)", color="#34D399", edgecolor="#1E293B", linewidth=1.2)
    b4 = ax2.bar(x + w/2, rec, w, label="Recall (%)", color="#F59E0B", edgecolor="#1E293B", linewidth=1.2)
    ax2.set_title("Operational Precision vs. Recall\n(@ Confidence = 0.25)", fontsize=11, weight="bold", color="#E2E8F0", pad=10)
    ax2.set_ylabel("Rate (%)", fontsize=10, color="#94A3B8")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9.5, weight="bold", color="#CBD5E1")
    ax2.set_ylim(0, 110)
    ax2.grid(axis="y", linestyle="--", alpha=0.25, color="#475569")
    ax2.legend(loc="upper left", facecolor="#1F2937", edgecolor="#374151", fontsize=8.5, labelcolor="#F1F5F9")
    for bar in b3:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8.5, weight="bold", color="#34D399")
    for bar in b4:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8.5, weight="bold", color="#F59E0B")

    # Panel 3: False Positive Rate on Pure Background Negatives
    ax3 = axes[2]
    ax3.set_facecolor("#111827")
    bg_fps = [h25["bg_fp_rate_pct"], n25["bg_fp_rate_pct"]]
    b5 = ax3.bar(x, bg_fps, width=0.55, color=["#EF4444", "#10B981"], edgecolor="#1E293B", linewidth=1.2)
    ax3.set_title("False Alarm Rate on Background Frames\n(Lower is Better)", fontsize=11, weight="bold", color="#E2E8F0", pad=10)
    ax3.set_ylabel("Background Frames with FP (%)", fontsize=10, color="#94A3B8")
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontsize=9.5, weight="bold", color="#CBD5E1")
    ax3.set_ylim(0, max(bg_fps) * 1.35 if max(bg_fps) > 0 else 50)
    ax3.grid(axis="y", linestyle="--", alpha=0.25, color="#475569")
    for bar in b5:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha="center", va="bottom", fontsize=10, weight="bold", color="#F8FAFC")

    # Panel 4: Spatial Error (CLE in pixels) & IoU
    ax4 = axes[3]
    ax4.set_facecolor("#111827")
    cle = [h25["mean_cle_px"], n25["mean_cle_px"]]
    b6 = ax4.bar(x, cle, width=0.55, color=["#F97316", "#06B6D4"], edgecolor="#1E293B", linewidth=1.2)
    ax4.set_title("Matched Center Location Error (CLE)\n(Lower is Better)", fontsize=11, weight="bold", color="#E2E8F0", pad=10)
    ax4.set_ylabel("Mean CLE (pixels)", fontsize=10, color="#94A3B8")
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, fontsize=9.5, weight="bold", color="#CBD5E1")
    ax4.set_ylim(0, max(cle) * 1.35)
    ax4.grid(axis="y", linestyle="--", alpha=0.25, color="#475569")
    for bar in b6:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.1f}px", ha="center", va="bottom", fontsize=10, weight="bold", color="#F8FAFC")

    for ax in axes:
        ax.tick_params(colors="#94A3B8", labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#374151")

    plt.tight_layout()
    plt.savefig(FIG_METRICS, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Metrics comparison figure saved to: {FIG_METRICS}")

    if ARTIFACT_DIR.exists():
        shutil.copy2(FIG_METRICS, ARTIFACT_DIR / FIG_METRICS.name)


def generate_qualitative_visuals(model_hybrid: YOLO, model_nougat: YOLO, records: List[Dict[str, Any]]):
    """Generates a 3-image visual comparison grid showing Ground Truth vs Hybrid NWD vs Nougat Pede."""
    # Pick 3 interesting examples: 2 with drone, 1 background
    drone_recs = [r for r in records if not r["is_background"]]
    bg_recs = [r for r in records if r["is_background"]]

    selected = []
    if len(drone_recs) >= 2:
        selected.append(drone_recs[0])
        selected.append(drone_recs[len(drone_recs) // 2])
    if len(bg_recs) >= 1:
        selected.append(bg_recs[0])

    fig, axes = plt.subplots(len(selected), 3, figsize=(18, 4.5 * len(selected)), dpi=180, facecolor="#0B0F19")
    fig.suptitle(
        "Visual Qualitative Comparison: Ground Truth vs. Hybrid NWD vs. Nougat-Pede Fine-Tuned",
        fontsize=14, weight="bold", color="#F8FAFC", y=0.99
    )

    col_titles = ["Ground Truth Annotation", "YOLOv8n Hybrid NWD (Generalist)", "YOLOv8n Nougat Pede (Specialist)"]

    for row_idx, rec in enumerate(selected):
        im_bgr = cv2.imread(str(rec["img_path"]))
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

        # Predict with both models
        res_h = model_hybrid.predict(im_bgr, imgsz=768, conf=0.25, verbose=False)
        res_n = model_nougat.predict(im_bgr, imgsz=768, conf=0.25, verbose=False)

        # Col 0: Ground Truth
        ax0 = axes[row_idx, 0] if len(selected) > 1 else axes[0]
        ax0.imshow(im_rgb)
        ax0.set_facecolor("#111827")
        if row_idx == 0:
            ax0.set_title(col_titles[0], fontsize=11, weight="bold", color="#10B981", pad=8)
        for gt in rec["gt_boxes"]:
            b = gt["box"]
            rect = patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], linewidth=2.5, edgecolor="#10B981", facecolor="none")
            ax0.add_patch(rect)
            ax0.text(b[0], max(0, b[1]-5), f"GT Drone", color="#10B981", fontsize=9, weight="bold", bbox=dict(facecolor="#111827", alpha=0.7, edgecolor="none", pad=1))
        if rec["is_background"]:
            ax0.text(50, 80, "[Pure Negative Background]", color="#FCD34D", fontsize=11, weight="bold", bbox=dict(facecolor="#1F2937", alpha=0.8, pad=3))

        # Col 1: Hybrid NWD
        ax1 = axes[row_idx, 1] if len(selected) > 1 else axes[1]
        ax1.imshow(im_rgb)
        ax1.set_facecolor("#111827")
        if row_idx == 0:
            ax1.set_title(col_titles[1], fontsize=11, weight="bold", color="#EF4444", pad=8)
        h_boxes = res_h[0].boxes.xyxy.cpu().numpy() if len(res_h) > 0 and len(res_h[0].boxes) > 0 else []
        h_confs = res_h[0].boxes.conf.cpu().numpy() if len(res_h) > 0 and len(res_h[0].boxes) > 0 else []
        for b, c in zip(h_boxes, h_confs):
            rect = patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], linewidth=2.5, edgecolor="#EF4444", facecolor="none")
            ax1.add_patch(rect)
            ax1.text(b[0], max(0, b[1]-5), f"Hybrid: {c:.2f}", color="#EF4444", fontsize=9, weight="bold", bbox=dict(facecolor="#111827", alpha=0.7, edgecolor="none", pad=1))
        if len(h_boxes) == 0:
            ax1.text(50, 80, "No Detection", color="#94A3B8", fontsize=11, weight="bold", bbox=dict(facecolor="#1F2937", alpha=0.8, pad=3))

        # Col 2: Nougat Pede
        ax2 = axes[row_idx, 2] if len(selected) > 1 else axes[2]
        ax2.imshow(im_rgb)
        ax2.set_facecolor("#111827")
        if row_idx == 0:
            ax2.set_title(col_titles[2], fontsize=11, weight="bold", color="#8B5CF6", pad=8)
        n_boxes = res_n[0].boxes.xyxy.cpu().numpy() if len(res_n) > 0 and len(res_n[0].boxes) > 0 else []
        n_confs = res_n[0].boxes.conf.cpu().numpy() if len(res_n) > 0 and len(res_n[0].boxes) > 0 else []
        for b, c in zip(n_boxes, n_confs):
            rect = patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], linewidth=2.5, edgecolor="#8B5CF6", facecolor="none")
            ax2.add_patch(rect)
            ax2.text(b[0], max(0, b[1]-5), f"Nougat: {c:.2f}", color="#8B5CF6", fontsize=9, weight="bold", bbox=dict(facecolor="#111827", alpha=0.7, edgecolor="none", pad=1))
        if len(n_boxes) == 0:
            ax2.text(50, 80, "No Detection", color="#94A3B8", fontsize=11, weight="bold", bbox=dict(facecolor="#1F2937", alpha=0.8, pad=3))

        for ax in [ax0, ax1, ax2]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(FIG_VISUAL, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Qualitative visual comparison figure saved to: {FIG_VISUAL}")

    if ARTIFACT_DIR.exists():
        shutil.copy2(FIG_VISUAL, ARTIFACT_DIR / FIG_VISUAL.name)


if __name__ == "__main__":
    run_benchmark()
