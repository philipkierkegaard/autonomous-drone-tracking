#!/usr/bin/env python3
"""
Generate a 2x3 Publication-Quality Grid of Validation Failure Images for Academic Thesis.
Features:
- White background (paper / print ready)
- No large title
- Subfigures labeled (a) through (f)
- Exact same 320x320 zoom & centering as the 5x5 grid
- Very subtle bounding box around the target in an elegant, understated color

Images:
  Row 1:
    (a) video15_00861
    (b) video16_00117
    (c) video01_00211
  Row 2:
    (d) video13_00757
    (e) video17_00105
    (f) video17_00345
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

val_img_dir = PROJECT_ROOT / "perception" / "datasets" / "anti_uav_yolo" / "images" / "val"
val_lbl_dir = PROJECT_ROOT / "perception" / "datasets" / "anti_uav_yolo" / "labels" / "val"

OUT_DIR = PROJECT_ROOT / "outputs" / "videos" / "perception"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "yolov8_val_failures_subtle_2x3_grid.png"
ARTIFACT_DIR = Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/5967d853-7a7e-4cdd-8792-4a77e6d704a2")

# 2 rows x 3 columns mapping
image_grid = [
    [
        {"name": "video15_00861", "tag": "(a)"},
        {"name": "video16_00117", "tag": "(b)"},
        {"name": "video01_00211", "tag": "(c)"}
    ],
    [
        {"name": "video13_00757", "tag": "(d)"},
        {"name": "video17_00105", "tag": "(e)"},
        {"name": "video17_00345", "tag": "(f)"}
    ]
]

crop_window_size = 320  # Identical 320x320 crop window


def main():
    print("---> Generating 2x3 white grid with subtle bounding boxes...")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    # White canvas
    fig = plt.figure(figsize=(15, 10.2), dpi=300, facecolor="#FFFFFF")
    gs = gridspec.GridSpec(2, 3, wspace=0.10, hspace=0.20, top=0.96, bottom=0.06, left=0.04, right=0.96)

    for r_idx, row in enumerate(image_grid):
        for c_idx, item in enumerate(row):
            ax = fig.add_subplot(gs[r_idx, c_idx])
            img_name = item["name"]
            tag = item["tag"]
            img_file = f"{img_name}.jpg"
            img_path = val_img_dir / img_file
            lbl_path = val_lbl_dir / f"{img_name}.txt"

            img_bgr = cv2.imread(str(img_path))
            assert img_bgr is not None, f"Could not read {img_path}"
            ih, iw = img_bgr.shape[:2]

            # Read Ground Truth box
            gt_boxes = []
            if lbl_path.exists():
                lines = [l.strip().split() for l in lbl_path.read_text().splitlines() if l.strip()]
                for l in lines:
                    cls_id, cx, cy, bw, bh = map(float, l)
                    gx1 = int((cx - bw / 2) * iw)
                    gy1 = int((cy - bh / 2) * ih)
                    gx2 = int((cx + bw / 2) * iw)
                    gy2 = int((cy + bh / 2) * ih)
                    gt_boxes.append((gx1, gy1, gx2, gy2))

            if len(gt_boxes) > 0:
                gx1, gy1, gx2, gy2 = gt_boxes[0]
                cx = (gx1 + gx2) // 2
                cy = (gy1 + gy2) // 2
            else:
                cx, cy = iw // 2, ih // 2

            # Same clamping as 5x5 grid
            half_win = crop_window_size // 2
            x1 = max(0, min(iw - crop_window_size, cx - half_win))
            y1 = max(0, min(ih - crop_window_size, cy - half_win))
            x2 = x1 + crop_window_size
            y2 = y1 + crop_window_size

            crop_bgr = img_bgr[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

            ax.imshow(crop_rgb)
            ax.set_xlim(0, crop_window_size)
            ax.set_ylim(crop_window_size, 0)

            # Draw very subtle bounding box around target
            if len(gt_boxes) > 0:
                cgx1 = gx1 - x1
                cgy1 = gy1 - y1
                cgx2 = gx2 - x1
                cgy2 = gy2 - y1

                # Subtle, elegant bounding box (subtle azure cyan #0284C7, 1.4px line width)
                rect = Rectangle(
                    (cgx1, cgy1), cgx2 - cgx1, cgy2 - cgy1,
                    edgecolor="#0284C7", facecolor="none",
                    linewidth=1.4, linestyle="-", alpha=0.9
                )
                ax.add_patch(rect)

            # Subfigure label (a)-(f) below each image
            ax.set_xlabel(tag, fontsize=13, weight="bold", color="#1E293B", labelpad=8)

            # Remove ticks but keep subtle 1px border so sky doesn't bleed into white page
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#94A3B8")
                spine.set_linewidth(1.0)

    print(f"---> Saving figure to {OUT_FILE}...")
    plt.savefig(OUT_FILE, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved subtle 2x3 grid to: {OUT_FILE}")

    # Copy to artifact directory
    if ARTIFACT_DIR.exists():
        dest = ARTIFACT_DIR / OUT_FILE.name
        shutil.copy2(OUT_FILE, dest)
        # Also copy over the plain grid name if requested
        shutil.copy2(OUT_FILE, ARTIFACT_DIR / "yolov8_val_failures_plain_2x3_grid.png")
        print(f"[COPIED] Copied to artifact dir: {dest}")


if __name__ == "__main__":
    main()
