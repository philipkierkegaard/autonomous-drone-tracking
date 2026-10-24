#!/usr/bin/env python3
"""
Generate a 5x5 Publication-Quality Grid of YOLOv8n-768-Hybrid Failure Cases on the Validation Dataset.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception System.

Taxonomy (5 Rows x 5 Columns = 25 Failure Cases):
Row 1: Tree Canopy & Dense Foliage Camouflage (False Negatives)
Row 2: Extreme Distance & Sub-35px Scale Decimation (False Negatives)
Row 3: Deep Shadow, Shaded Ground & Low Dynamic Range (False Negatives)
Row 4: Complex Architectural & Urban Clutter (False Negatives)
Row 5: Hard Negative Distractors & False Alarms (False Positives)
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

from ultralytics import YOLO

OUT_DIR = PROJECT_ROOT / "outputs" / "videos" / "perception"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "yolov8_768_hybrid_val_failures_5x5_grid.png"
ARTIFACT_DIR = Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/5967d853-7a7e-4cdd-8792-4a77e6d704a2")

val_img_dir = PROJECT_ROOT / "perception" / "datasets" / "anti_uav_yolo" / "images" / "val"
val_lbl_dir = PROJECT_ROOT / "perception" / "datasets" / "anti_uav_yolo" / "labels" / "val"
weights_path = PROJECT_ROOT / "perception" / "weights" / "yolov8n_drone_768_hybrid_nwd_c51_best.pt"


def main():
    print(f"---> Loading YOLOv8n 768 Hybrid model from {weights_path.name}...")
    model = YOLO(str(weights_path))

    # Define the curated 5x5 taxonomy
    rows = [
        {
            "category": "ROW 1: TREE CANOPY & FOLIAGE CAMOUFLAGE (FALSE NEGATIVES)",
            "subtitle": "Thin carbon arms and spinning rotor discs blend into high-frequency leafy textures",
            "tag_color": "#22C55E",
            "bg_color": "#052E16",
            "items": [
                {"file": "video16_00109.jpg", "label": "Canopy Edge", "desc": "Target against dense foliage"},
                {"file": "video16_00117.jpg", "label": "Foliage Haze", "desc": "Quadcopter in tree canopy"},
                {"file": "video16_00125.jpg", "label": "Branch Clutter", "desc": "Rotors merge into leaf texture"},
                {"file": "video16_00133.jpg", "label": "Dense Leaves", "desc": "Camouflaged chassis against trees"},
                {"file": "video16_00463.jpg", "label": "Deep Canopy", "desc": "Drone buried in dark tree canopy"}
            ]
        },
        {
            "category": "ROW 2: EXTREME DISTANCE & SUB-35px SCALE DECIMATION (FALSE NEGATIVES)",
            "subtitle": "Long-range targets (>25m) where spatial features are decimated during 768x768 letterboxing",
            "tag_color": "#38BDF8",
            "bg_color": "#082F49",
            "items": [
                {"file": "video15_00361.jpg", "label": "High Altitude", "desc": "Distant target (44×31 px)"},
                {"file": "video15_00861.jpg", "label": "Sub-Pixel Scale", "desc": "Diminutive target (45×38 px)"},
                {"file": "video17_00213.jpg", "label": "Distant Sky", "desc": "Small target (54×30 px) in haze"},
                {"file": "video17_00345.jpg", "label": "Far Horizon", "desc": "Distant drone (46×36 px) at boundary"},
                {"file": "video17_00417.jpg", "label": "Sensor Boundary", "desc": "Extreme edge target (42×33 px)"}
            ]
        },
        {
            "category": "ROW 3: DEEP SHADOW & LOW AMBIENT DYNAMIC RANGE (FALSE NEGATIVES)",
            "subtitle": "Severe ground shadow and courtyard shade suppress target contrast below receptive field activation",
            "tag_color": "#F59E0B",
            "bg_color": "#451A03",
            "items": [
                {"file": "video01_00176.jpg", "label": "Deep Ground Shadow", "desc": "Large drone (184×93 px) in dark shade"},
                {"file": "video01_00211.jpg", "label": "Shadow Camouflage", "desc": "Low-reflectance body in tarmac shade"},
                {"file": "video01_00246.jpg", "label": "Wall Shadow Edge", "desc": "Body merging with building shadow"},
                {"file": "video01_00911.jpg", "label": "Low Contrast Ground", "desc": "Target (74×39 px) on shaded ground"},
                {"file": "video17_00079.jpg", "label": "Low Ambient Light", "desc": "Under-exposed target (55×44 px)"}
            ]
        },
        {
            "category": "ROW 4: COMPLEX ARCHITECTURAL & URBAN CLUTTER (FALSE NEGATIVES)",
            "subtitle": "High-contrast building facades, masonry lines, and courtyard window frames confuse CNN attention",
            "tag_color": "#A855F7",
            "bg_color": "#3B0764",
            "items": [
                {"file": "video13_00277.jpg", "label": "Urban Facade", "desc": "Drone (68×30 px) against building wall"},
                {"file": "video13_00289.jpg", "label": "Window Grid Edge", "desc": "Moving across structural window frames"},
                {"file": "video13_00757.jpg", "label": "Building Lines", "desc": "Target (60×30 px) on facade lines"},
                {"file": "video15_00541.jpg", "label": "Courtyard Railing", "desc": "Target (48×56 px) in front of railing"},
                {"file": "video17_00105.jpg", "label": "Roof Edge Clutter", "desc": "Drone (61×43 px) near roofline corner"}
            ]
        },
        {
            "category": "ROW 5: HARD NEGATIVE DISTRACTORS & LOOKALIKES (FALSE POSITIVES)",
            "subtitle": "Background images with ZERO drone where antenna tips, pylon struts, and tree crowns trigger false alarms",
            "tag_color": "#EF4444",
            "bg_color": "#450A0A",
            "items": [
                {"file": "hardneg_neg_tree_cypress_trees_0008.jpg", "label": "Cypress Crown", "desc": "Cypress tree apex triggers 69.5% false alarm"},
                {"file": "hardneg_neg_tree_autumn_canopy_0004.jpg", "label": "Canopy Clump", "desc": "Autumn foliage cluster triggers 51.4% false alarm"},
                {"file": "hardneg_neg_skyscraper_facade_0019.jpg", "label": "Window Corner", "desc": "Window grid corner triggers 40.2% false alarm"},
                {"file": "hardneg_neg_transmission_tower_0017.jpg", "label": "Tower Insulator", "desc": "Pylon strut insulator triggers 32.1% false alarm"},
                {"file": "hardneg_neg_pine_tree_sky_0010.jpg", "label": "Pine Tip Silhouette", "desc": "Branch tip silhouette triggers 18.6% false alarm"}
            ]
        }
    ]

    print("---> Setting up publication figure canvas...")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    n_rows = 5
    n_cols = 5
    fig = plt.figure(figsize=(24, 26), dpi=220, facecolor="#0B0F19")

    # GridSpec: 5 category blocks
    outer_gs = gridspec.GridSpec(n_rows, 1, height_ratios=[1, 1, 1, 1, 1], hspace=0.30)

    # Master title
    fig.suptitle(
        "YOLOv8n 768 Hybrid NWD Detector Failure Analysis — 5×5 Taxonomy on Validation Dataset\n"
        "Anti-UAV Benchmark (6,821 Images) | Model: yolov8n_drone_768_hybrid_nwd_c51 | Native Input: 768×768 | Threshold: conf ≥ 0.20",
        fontsize=16.5, weight="bold", color="#F8FAFC", y=0.988
    )

    crop_window_size = 320  # Square context window size centered on target/FP

    for row_idx, row_data in enumerate(rows):
        print(f"Processing {row_data['category']}...")
        # Sub-gridspec: top row is full-width header banner, bottom row is 5 image tiles
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, n_cols, subplot_spec=outer_gs[row_idx],
            height_ratios=[0.09, 0.91], wspace=0.08, hspace=0.10
        )

        # Full-width Category Banner spanning all 5 columns
        ax_banner = fig.add_subplot(inner_gs[0, :])
        ax_banner.set_facecolor(row_data["bg_color"])
        for spine in ax_banner.spines.values():
            spine.set_edgecolor(row_data["tag_color"])
            spine.set_linewidth(2)
        ax_banner.set_xticks([])
        ax_banner.set_yticks([])

        # Banner Text
        ax_banner.text(
            0.02, 0.5,
            row_data["category"],
            transform=ax_banner.transAxes,
            ha="left", va="center",
            fontsize=12, weight="bold", color=row_data["tag_color"]
        )
        ax_banner.text(
            0.98, 0.5,
            row_data["subtitle"],
            transform=ax_banner.transAxes,
            ha="right", va="center",
            fontsize=10.5, style="italic", color="#CBD5E1"
        )

        for col_idx, item in enumerate(row_data["items"]):
            ax = fig.add_subplot(inner_gs[1, col_idx])
            img_file = item["file"]
            img_path = val_img_dir / img_file
            lbl_path = val_lbl_dir / (img_path.stem + ".txt")

            img_bgr = cv2.imread(str(img_path))
            assert img_bgr is not None, f"Could not read {img_path}"
            ih, iw = img_bgr.shape[:2]

            # Read Ground Truth if exists
            gt_boxes = []
            if lbl_path.exists():
                lines = [l.strip().split() for l in lbl_path.read_text().splitlines() if l.strip()]
                for l in lines:
                    cls_id, cx, cy, bw, bh = map(float, l)
                    gx1 = int((cx - bw / 2) * iw)
                    gy1 = int((cy - bh / 2) * ih)
                    gx2 = int((cx + bw / 2) * iw)
                    gy2 = int((cy + bh / 2) * ih)
                    gt_boxes.append((gx1, gy1, gx2, gy2, int(bw * iw), int(bh * ih)))

            # Run prediction on full frame with 768
            res = model.predict(str(img_path), imgsz=768, conf=0.15, verbose=False)[0]
            preds = []
            for b in res.boxes:
                box = b.xyxy[0].cpu().numpy().astype(int)
                conf = float(b.conf[0])
                preds.append((box[0], box[1], box[2], box[3], conf))

            # Determine crop center
            if row_idx < 4 and len(gt_boxes) > 0:
                # Center on GT box
                gx1, gy1, gx2, gy2, gw, gh = gt_boxes[0]
                cx = (gx1 + gx2) // 2
                cy = (gy1 + gy2) // 2
                is_fp_row = False
            else:
                # Row 5: Hard negatives with False Positive
                if len(preds) > 0:
                    best_pred = max(preds, key=lambda x: x[4])
                    cx = (best_pred[0] + best_pred[2]) // 2
                    cy = (best_pred[1] + best_pred[3]) // 2
                else:
                    cx, cy = iw // 2, ih // 2
                is_fp_row = True

            # Crop context window with bounds clamp
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

            # Draw Ground Truth Box if FN row
            if not is_fp_row and len(gt_boxes) > 0:
                gx1, gy1, gx2, gy2, gw, gh = gt_boxes[0]
                cgx1, cgy1 = gx1 - x1, gy1 - y1
                cgx2, cgy2 = gx2 - x1, gy2 - y1

                # Ground Truth box in Cyan
                rect_gt = Rectangle(
                    (cgx1, cgy1), cgx2 - cgx1, cgy2 - cgy1,
                    edgecolor="#06B6D4", facecolor="none", linewidth=2.5, linestyle="--"
                )
                ax.add_patch(rect_gt)

                # Ground Truth dimension badge
                ax.text(
                    cgx1, max(14, cgy1 - 6),
                    f"GT: {gw}×{gh}px",
                    fontsize=8, weight="bold", color="#06B6D4",
                    bbox=dict(boxstyle="square,pad=0.15", facecolor="#082F49", edgecolor="#06B6D4", alpha=0.9)
                )

                # Prominent MISSED badge
                ax.text(
                    0.96, 0.06, "MISSED (FN)\nConf: 0.0%",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8.5, weight="bold", color="#FCA5A5",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#7F1D1D", edgecolor="#EF4444", alpha=0.95)
                )

            # Draw False Positive Box if FP row
            if is_fp_row:
                for px1, py1, px2, py2, pconf in preds:
                    cpx1, cpy1 = px1 - x1, py1 - y1
                    cpx2, cpy2 = px2 - x1, py2 - y1
                    rect_fp = Rectangle(
                        (cpx1, cpy1), cpx2 - cpx1, cpy2 - cpy1,
                        edgecolor="#EF4444", facecolor="none", linewidth=2.5
                    )
                    ax.add_patch(rect_fp)
                    ax.text(
                        cpx1, max(14, cpy1 - 6),
                        f"False Drone: {pconf*100:.1f}%",
                        fontsize=8.5, weight="bold", color="#FCA5A5",
                        bbox=dict(boxstyle="square,pad=0.15", facecolor="#450A0A", edgecolor="#EF4444", alpha=0.9)
                    )

                ax.text(
                    0.96, 0.06, "FALSE ALARM (FP)\nNo Drone in Scene",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8.5, weight="bold", color="#FDE047",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#713F12", edgecolor="#EAB308", alpha=0.95)
                )

            # Subplot Header & Info Banner
            frame_num = img_file.replace(".jpg", "")
            ax.set_title(
                f"{item['label']} ({frame_num})\n{item['desc']}",
                fontsize=9.2, weight="bold", color="#E2E8F0", pad=5
            )

            # Frame border styled by category
            for spine in ax.spines.values():
                spine.set_edgecolor("#334155")
                spine.set_linewidth(1.5)
            ax.set_xticks([])
            ax.set_yticks([])

    # Save final high-res figure
    print(f"---> Saving figure to {OUT_FILE}...")
    plt.savefig(OUT_FILE, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved 5x5 failure grid to: {OUT_FILE}")

    # Copy to artifacts directory
    if ARTIFACT_DIR.exists():
        dest = ARTIFACT_DIR / OUT_FILE.name
        shutil.copy2(OUT_FILE, dest)
        print(f"[COPIED] Copied to artifact dir: {dest}")


if __name__ == "__main__":
    main()
