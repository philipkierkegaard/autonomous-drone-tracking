#!/usr/bin/env python3
"""
Find and categorize 25 distinct failure cases of YOLOv8n-768-Hybrid on the validation set.
Categories (5x5 grid):
Row 1: Tree Canopy & Foliage Camouflage (FN)
Row 2: Extreme Distance & Sub-25px Diminutive Target (FN)
Row 3: Deep Shadow, Dark Occlusion & Backlit Contrast (FN)
Row 4: Urban & Structural Background Clutter (FN)
Row 5: False Positives & Lookalikes on Hard Negatives (FP)
"""

import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

val_img_dir = PROJECT_ROOT / "perception" / "datasets" / "anti_uav_yolo" / "images" / "val"
val_lbl_dir = PROJECT_ROOT / "perception" / "datasets" / "anti_uav_yolo" / "labels" / "val"
weights_path = PROJECT_ROOT / "perception" / "weights" / "yolov8n_drone_768_hybrid_nwd_c51_best.pt"

model = YOLO(str(weights_path))

# Let's search across validation videos and hard negatives
vids = ['video01', 'video04', 'video13', 'video15', 'video16', 'video17']
candidates = {
    'foliage_fn': [],
    'tiny_fn': [],
    'shadow_lighting_fn': [],
    'urban_clutter_fn': [],
    'hardneg_fp': []
}

# 1. Scan hard negatives for FP
hard_negs = sorted(list(val_img_dir.glob('hardneg_*.jpg')))
print(f"Scanning {len(hard_negs)} hard negatives...")
for p in hard_negs:
    res = model.predict(str(p), imgsz=768, conf=0.15, verbose=False)[0]
    for b in res.boxes:
        conf = float(b.conf[0])
        box = b.xyxy[0].cpu().numpy().astype(int)
        candidates['hardneg_fp'].append({
            'file': p.name,
            'conf': conf,
            'pred_box': box.tolist(),
            'gt_box': None,
            'w': box[2] - box[0],
            'h': box[3] - box[1]
        })

print(f"Found {len(candidates['hardneg_fp'])} FPs in hard negatives")

# 2. Scan video frames for FNs
for vid in vids:
    imgs = sorted(list(val_img_dir.glob(f"{vid}_*.jpg")))
    step = max(1, len(imgs) // 60) # sample 60 frames per video
    sampled = imgs[::step]
    print(f"Scanning {vid} ({len(sampled)} frames)...")
    
    for p in sampled:
        lbl_file = val_lbl_dir / (p.stem + ".txt")
        lines = [l.strip().split() for l in lbl_file.read_text().splitlines() if l.strip()]
        if not lines:
            continue
        
        im = cv2.imread(str(p))
        if im is None:
            continue
        ih, iw = im.shape[:2]
        
        gt_boxes = []
        for l in lines:
            cls_id, cx, cy, bw, bh = map(float, l)
            gx1, gy1 = int((cx - bw/2) * iw), int((cy - bh/2) * ih)
            gx2, gy2 = int((cx + bw/2) * iw), int((cy + bh/2) * ih)
            gt_boxes.append((gx1, gy1, gx2, gy2, int(bw * iw), int(bh * ih)))
            
        res = model.predict(str(p), imgsz=768, conf=0.20, verbose=False)[0]
        preds = []
        for b in res.boxes:
            box = b.xyxy[0].cpu().numpy().astype(int)
            conf = float(b.conf[0])
            preds.append((box[0], box[1], box[2], box[3], conf))
            
        for gx1, gy1, gx2, gy2, gw, gh in gt_boxes:
            matched = False
            best_iou = 0.0
            best_conf = 0.0
            for px1, py1, px2, py2, pconf in preds:
                ix1, iy1 = max(gx1, px1), max(gy1, py1)
                ix2, iy2 = min(gx2, px2), min(gy2, py2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (gx2 - gx1)*(gy2 - gy1) + (px2 - px1)*(py2 - py1) - inter
                iou = inter / max(1e-6, union)
                if iou > best_iou:
                    best_iou = iou
                    best_conf = pconf
                if iou >= 0.20:
                    matched = True
            
            if not matched:
                # Analyze patch properties
                pad = 40
                y_min, y_max = max(0, gy1 - pad), min(ih, gy2 + pad)
                x_min, x_max = max(0, gx1 - pad), min(iw, gx2 + pad)
                patch = im[y_min:y_max, x_min:x_max]
                mean_bgr = patch.mean(axis=(0,1))
                std_bgr = patch.std(axis=(0,1))
                brightness = mean_bgr.mean()
                
                # Check greenness (foliage)
                # G > R and G > B by a margin
                is_foliage = (mean_bgr[1] > mean_bgr[2] * 1.05 and mean_bgr[1] > mean_bgr[0] * 1.05) or (vid in ['video16', 'video12'])
                is_tiny = max(gw, gh) <= 35
                is_dark = brightness < 65
                is_overcast = brightness > 180 and std_bgr.mean() < 30
                
                item = {
                    'file': p.name,
                    'vid': vid,
                    'gt_box': [gx1, gy1, gx2, gy2],
                    'w': gw,
                    'h': gh,
                    'best_iou': best_iou,
                    'best_conf': best_conf,
                    'brightness': float(brightness),
                    'mean_bgr': mean_bgr.tolist(),
                    'std_bgr': std_bgr.tolist(),
                    'preds': preds
                }
                
                if vid == 'video16':
                    candidates['foliage_fn'].append(item)
                elif is_tiny:
                    candidates['tiny_fn'].append(item)
                elif is_dark or is_overcast or vid == 'video01' and brightness < 80:
                    candidates['shadow_lighting_fn'].append(item)
                else:
                    candidates['urban_clutter_fn'].append(item)

print("\n--- Candidate Counts ---")
for k, v in candidates.items():
    print(f"{k}: {len(v)}")

import json
with open(PROJECT_ROOT / "scripts" / "val_failure_candidates.json", "w") as f:
    json.dump(candidates, f, indent=2)
print("Saved val_failure_candidates.json")
