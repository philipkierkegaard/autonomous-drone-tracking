#!/usr/bin/env python3
"""
Evaluates YOLOv8n v3 Baseline (640x640), 768 CIoU, and 768 Hybrid NWD-CIoU on the held-out test set (8,052 frames).
Computes exact empirical detection catch rates (recall) across standardized scale categories:
1. Canonical 768 Reference Grid:
   - COCO Small Area: area <= 1024 px^2 (32^2) on 768 grid
   - Small Target Dimension: max(w, h) <= 32 px on 768 grid
   - Historical Scale Bin: scale < 0.035 (< 26.88 px on 768)
2. Native Input Grid:
   - Small Target Dimension: max(w, h) <= 32 px on native model input
   - COCO Small Area: area <= 1024 px^2 on native model input
3. All Targets:
   - Full held-out test set recall (7,807 ground-truth instances)
   - Evaluated at both IoU >= 0.25 (tracker acquisition gate) and IoU >= 0.50 (strict mAP50 gate)
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils.nms import non_max_suppression


def compute_iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """Computes pairwise IoU between box1 (N, 4) and box2 (M, 4) in xyxy format."""
    if len(box1) == 0 or len(box2) == 0:
        return np.zeros((len(box1), len(box2)), dtype=np.float32)

    x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = area1[:, None] + area2[None, :] - inter_area

    return inter_area / np.maximum(1e-6, union_area)


def evaluate_model_on_test(
    weights_path: str,
    imgsz: int = 768,
    data_yaml: str = "perception/datasets/anti_uav_yolo/data.yaml",
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    match_iou_thresh: float = 0.25,
    batch_size: int = 64,
    device: str = "mps"
):
    print("=" * 80)
    print(f"EVALUATING: {Path(weights_path).name} (imgsz={imgsz})")
    print("=" * 80)

    data = check_det_dataset(data_yaml)
    cfg = get_cfg()
    cfg.data = data_yaml
    cfg.imgsz = imgsz
    cfg.batch = batch_size

    # Build evaluation dataset & dataloader (rect=False ensures square letterbox of imgsz x imgsz)
    dataset = build_yolo_dataset(cfg, data["test"], batch_size, data, mode="val", rect=False)
    loader = build_dataloader(dataset, batch=batch_size, workers=8, shuffle=False)

    yolo = YOLO(weights_path)
    model = yolo.model.to(device)
    model.eval()

    categories = [
        "coco_small_768_area",     # area <= 1024 px^2 on 768 reference
        "small_768_dim",           # max(w, h) <= 32 px on 768 reference
        "tiny_hist_sub27",         # normalized scale < 0.035 (< 26.88 px on 768)
        "native_small_dim",        # max(w, h) <= 32 px on native model grid
        "native_small_area",       # area <= 1024 px^2 on native model grid
        "all"
    ]

    stats = {
        cat: {
            "total_gt": 0,
            "detected_iou25": 0,
            "detected_iou50": 0
        }
        for cat in categories
    }

    total_predictions = 0
    total_true_positives_iou25 = 0
    total_true_positives_iou50 = 0
    total_frames = 0
    empty_frames = 0

    t0 = time.time()
    for batch_i, batch in enumerate(loader):
        imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
        B = imgs.shape[0]
        total_frames += B

        with torch.no_grad():
            preds = model(imgs)

        # Apply NMS: returns list of (N, 6) [x1, y1, x2, y2, conf, cls] in [0, imgsz] pixel space
        nms_preds = non_max_suppression(preds, conf_thres=conf_thresh, iou_thres=iou_thresh)

        # Batch ground truth: bboxes are normalized [cx, cy, w, h] in [0, 1] of the letterboxed (imgsz, imgsz) image!
        batch_idx = batch["batch_idx"].cpu().numpy()
        batch_bboxes = batch["bboxes"].cpu().numpy()  # in letterbox normalized coords
        ori_shapes = batch["ori_shape"]

        for i in range(B):
            ori_h, ori_w = ori_shapes[i]
            if isinstance(ori_h, torch.Tensor):
                ori_h, ori_w = int(ori_h.item()), int(ori_w.item())

            mask = (batch_idx == i)
            gt_boxes_norm = batch_bboxes[mask]  # [cx, cy, w, h] normalized in letterboxed imgsz

            pred_tensor = nms_preds[i]
            if len(pred_tensor) > 0:
                pred_xyxy = pred_tensor[:, :4].cpu().numpy()
                pred_confs = pred_tensor[:, 4].cpu().numpy()
            else:
                pred_xyxy = np.empty((0, 4), dtype=np.float32)
                pred_confs = np.empty((0,), dtype=np.float32)

            total_predictions += len(pred_xyxy)

            if len(gt_boxes_norm) == 0:
                empty_frames += 1
                continue

            # Convert GT to letterbox pixel xyxy in [0, imgsz]
            gt_cx_px = gt_boxes_norm[:, 0] * imgsz
            gt_cy_px = gt_boxes_norm[:, 1] * imgsz
            gt_w_px = gt_boxes_norm[:, 2] * imgsz
            gt_h_px = gt_boxes_norm[:, 3] * imgsz

            gt_xyxy = np.column_stack([
                gt_cx_px - gt_w_px / 2.0,
                gt_cy_px - gt_h_px / 2.0,
                gt_cx_px + gt_w_px / 2.0,
                gt_cy_px + gt_h_px / 2.0
            ])

            # Dimension on native model input
            dim_native = np.maximum(gt_w_px, gt_h_px)
            area_native = gt_w_px * gt_h_px

            # Projected dimension on canonical 768x768 reference
            scale_to_768 = 768.0 / float(imgsz)
            dim_768 = dim_native * scale_to_768
            area_768 = area_native * (scale_to_768 ** 2)

            # Historical normalized scale (relative to 768: scale < 0.035, i.e. < 26.88 px)
            norm_scale_768 = dim_768 / 768.0

            # Match GT with predictions using pairwise IoU in letterbox pixel coords
            iou_mat = compute_iou_xyxy(gt_xyxy, pred_xyxy)

            for k in range(len(gt_boxes_norm)):
                is_detected_iou25 = False
                is_detected_iou50 = False
                if len(pred_xyxy) > 0:
                    max_iou = np.max(iou_mat[k])
                    if max_iou >= match_iou_thresh:
                        is_detected_iou25 = True
                    if max_iou >= 0.50:
                        is_detected_iou50 = True

                # Category membership
                flags = {
                    "all": True,
                    "coco_small_768_area": (area_768[k] <= 1024.0),
                    "small_768_dim": (dim_768[k] <= 32.0),
                    "tiny_hist_sub27": (norm_scale_768[k] < 0.035),
                    "native_small_dim": (dim_native[k] <= 32.0),
                    "native_small_area": (area_native[k] <= 1024.0)
                }

                for cat, applies in flags.items():
                    if applies:
                        stats[cat]["total_gt"] += 1
                        if is_detected_iou25:
                            stats[cat]["detected_iou25"] += 1
                        if is_detected_iou50:
                            stats[cat]["detected_iou50"] += 1

            # Precision tracking
            if len(pred_xyxy) > 0:
                for p_idx in range(len(pred_xyxy)):
                    p_max_iou = np.max(iou_mat[:, p_idx])
                    if p_max_iou >= match_iou_thresh:
                        total_true_positives_iou25 += 1
                    if p_max_iou >= 0.50:
                        total_true_positives_iou50 += 1

        if (batch_i + 1) % 25 == 0 or (batch_i + 1) == len(loader):
            print(f"  Processed [{batch_i+1}/{len(loader)}] batches ({total_frames} frames)...")

    elapsed = time.time() - t0
    fps = total_frames / elapsed

    print(f"\nEvaluation finished in {elapsed:.2f}s ({fps:.1f} FPS on {device}).")
    print(f"Total Frames: {total_frames} | Empty Frames: {empty_frames}")

    results = {
        "model": Path(weights_path).name,
        "imgsz": imgsz,
        "elapsed_s": round(elapsed, 2),
        "fps": round(fps, 1),
        "total_frames": total_frames,
        "empty_frames": empty_frames,
        "categories": {}
    }

    print("\n--- RESULTS BY TARGET SCALE CATEGORY ---")
    for cat, data_cat in stats.items():
        tot = data_cat["total_gt"]
        det25 = data_cat["detected_iou25"]
        det50 = data_cat["detected_iou50"]
        rec25 = (det25 / tot * 100.0) if tot > 0 else 0.0
        rec50 = (det50 / tot * 100.0) if tot > 0 else 0.0
        results["categories"][cat] = {
            "total_gt": tot,
            "detected_iou25": det25,
            "recall_iou25": round(rec25, 2),
            "detected_iou50": det50,
            "recall_iou50": round(rec50, 2)
        }
        print(f"• {cat:<20}: GT={tot:5d} | Det(IoU>=0.25)={det25:5d} ({rec25:6.2f}%) | Det(IoU>=0.50)={det50:5d} ({rec50:6.2f}%)")

    prec25 = (total_true_positives_iou25 / total_predictions * 100.0) if total_predictions > 0 else 0.0
    prec50 = (total_true_positives_iou50 / total_predictions * 100.0) if total_predictions > 0 else 0.0
    fp_per_frame = (total_predictions - total_true_positives_iou25) / total_frames if total_frames > 0 else 0.0

    results["precision_iou25"] = round(prec25, 2)
    results["precision_iou50"] = round(prec50, 2)
    results["total_predictions"] = total_predictions
    results["fp_per_frame"] = round(fp_per_frame, 4)

    print(f"\nOverall Precision (IoU>=0.25): {prec25:.2f}% | Precision (IoU>=0.50): {prec50:.2f}%")
    print(f"False Positives per frame: {fp_per_frame:.4f}")
    print("=" * 80 + "\n")

    return results


def main():
    models = [
        {
            "name": "YOLOv8n v3 Baseline",
            "weights": "perception/weights/yolov8n_v3_best.pt",
            "imgsz": 640
        },
        {
            "name": "YOLOv8n 768 CIoU",
            "weights": "perception/weights/yolov8n_drone_768_ciou_best.pt",
            "imgsz": 768
        },
        {
            "name": "YOLOv8n 768 Hybrid NWD-CIoU",
            "weights": "perception/weights/yolov8n_drone_768_hybrid_nwd_c51_best.pt",
            "imgsz": 768
        }
    ]

    all_results = {}
    for m in models:
        res = evaluate_model_on_test(
            weights_path=m["weights"],
            imgsz=m["imgsz"],
            batch_size=64,
            device="mps"
        )
        all_results[m["name"]] = res

    out_file = Path("perception/held_out_test_scale_ablation.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"All evaluation results saved to: {out_file}")


if __name__ == "__main__":
    main()
