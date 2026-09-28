"""
Visual Verification for Prepared YOLO Dataset
Draws bounding boxes from labels onto images to verify alignment and aspect ratio.
"""

import os
import glob
import random
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List


def draw_yolo_box(img, label_line, color=(0, 255, 0), thickness=3):
    parts = label_line.strip().split()
    if len(parts) != 5:
        return img
    
    cls_id, xc, yc, nw, nh = map(float, parts)
    h, w = img.shape[:2]
    
    xmin = int((xc - nw / 2) * w)
    ymin = int((yc - nh / 2) * h)
    xmax = int((xc + nw / 2) * w)
    ymax = int((yc + nh / 2) * h)
    
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
    label_text = f"Drone ({int(nw*w)}x{int(nh*h)})"
    cv2.putText(img, label_text, (xmin, max(20, ymin - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img


BASE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BASE_DIR.parent


def verify_dataset(
    dataset_dir: str = None,
    num_samples: int = 6,
    output_path: str = None
):
    if dataset_dir is None:
        dataset_dir = str(BASE_DIR / "datasets/anti_uav_yolo")
    if output_path is None:
        output_path = str(ROOT_DIR / "outputs/yolo_label_verification.png")

    img_dir = Path(dataset_dir) / "images" / "train"
    lbl_dir = Path(dataset_dir) / "labels" / "train"
    
    label_files = [f for f in glob.glob(f"{lbl_dir}/*.txt") if os.path.getsize(f) > 0]
    
    if not label_files:
        print(f"No non-empty label files found in {lbl_dir}.")
        return

    sampled_labels = random.sample(label_files, min(num_samples, len(label_files)))
    
    rows = (len(sampled_labels) + 2) // 3
    fig, axs = plt.subplots(rows, 3, figsize=(18, 5 * rows))
    axs = axs.flatten() if isinstance(axs, (list, tuple, np.ndarray)) else [axs]
    
    for i, lbl_path in enumerate(sampled_labels):
        stem = Path(lbl_path).stem
        img_path = img_dir / f"{stem}.jpg"
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        with open(lbl_path, "r") as f:
            for line in f:
                img_rgb = draw_yolo_box(img_rgb, line)
                
        axs[i].imshow(img_rgb)
        axs[i].set_title(stem, fontsize=10)
        axs[i].axis("off")
        
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved visual verification plot to: {output_path}")
    plt.close()


if __name__ == "__main__":
    import numpy as np
    verify_dataset()
