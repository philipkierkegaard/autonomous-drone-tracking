import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

OUT_DIR = Path("docs/reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

CROP_SIZE = 420  # Consistent 1:1 square crop size

# =============================================================================
# 1. Clean Side-by-Side: Video 20 Facade Miss vs Building Window Facade False Alarm
# =============================================================================
def generate_clean_side_by_side_building():
    # Panel (a): Anti-UAV video20 frame 00780
    img20_p = "perception/raw_data/Anti-UAV/Anti-UAV-Tracking-V0/video20/00780.jpg"
    img20 = cv2.cvtColor(cv2.imread(img20_p), cv2.COLOR_BGR2RGB)
    gx, gy, gw, gh = 736.0, 488.0, 55.0, 38.0
    cx20, cy20 = gx + gw / 2, gy + gh / 2
    x1_20 = int(cx20 - CROP_SIZE / 2)
    y1_20 = int(cy20 - CROP_SIZE / 2) + 20
    crop20 = img20[y1_20:y1_20 + CROP_SIZE, x1_20:x1_20 + CROP_SIZE]
    lgx = gx - x1_20
    lgy = gy - y1_20

    # Panel (b): Building Window Grid (neg_window_grid_0025.jpg)
    img_wg_p = "perception/raw_data/hard_negatives/images/neg_window_grid_0025.jpg"
    img_wg = cv2.cvtColor(cv2.imread(img_wg_p), cv2.COLOR_BGR2RGB)
    wb = [228, 1148, 297, 1198]  # YOLOv8n v1 Conf = 0.652
    wcx, wcy = (wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2
    wx1 = max(0, int(wcx - CROP_SIZE / 2))
    wy1 = max(0, int(wcy - CROP_SIZE / 2))
    crop_wg = img_wg[wy1:wy1 + CROP_SIZE, wx1:wx1 + CROP_SIZE]
    lwx = wb[0] - wx1
    lwy = wb[1] - wy1
    lww = wb[2] - wb[0]
    lwh = wb[3] - wb[1]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.5), dpi=300, facecolor='#FFFFFF')
    plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.03, wspace=0.06)

    # (a) False Negative
    ax_a = axes[0]
    ax_a.imshow(crop20)
    ax_a.axis('off')
    rect_gt = patches.Rectangle((lgx, lgy), gw, gh, linewidth=2.5, edgecolor='#00C853', facecolor='none')
    ax_a.add_patch(rect_gt)
    ax_a.text(lgx, lgy - 7, 'Ground Truth: Drone (Missed)', color='#FFFFFF', fontsize=9.0, fontweight='bold',
              bbox=dict(facecolor='#00C853', edgecolor='none', boxstyle='square,pad=0.22'))
    ax_a.set_title('(a) False Negative (Missed Detection)', fontsize=10.5, fontweight='bold', color='#1E293B', pad=8)

    # (b) False Positive
    ax_b = axes[1]
    ax_b.imshow(crop_wg)
    ax_b.axis('off')
    rect_fp = patches.Rectangle((lwx, lwy), lww, lwh, linewidth=2.5, edgecolor='#E53935', facecolor='none')
    ax_b.add_patch(rect_fp)
    ax_b.text(lwx, lwy - 7, 'False Positive (conf = 0.65)', color='#FFFFFF', fontsize=9.0, fontweight='bold',
              bbox=dict(facecolor='#E53935', edgecolor='none', boxstyle='square,pad=0.22'))
    ax_b.set_title('(b) False Positive (Spurious Detection)', fontsize=10.5, fontweight='bold', color='#1E293B', pad=8)

    out_p = OUT_DIR / 'detector_v1_failures_clean_side_by_side.png'
    plt.savefig(out_p, dpi=300, facecolor='#FFFFFF', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_p}")


# =============================================================================
# 2. Standalone Subfigure Exports (for LaTeX \\begin{subfigure})
# =============================================================================
def generate_clean_standalone_crops():
    # 1. False Negative Crop (video20 frame 00780)
    img20 = cv2.cvtColor(cv2.imread('perception/raw_data/Anti-UAV/Anti-UAV-Tracking-V0/video20/00780.jpg'), cv2.COLOR_BGR2RGB)
    gx, gy, gw, gh = 736.0, 488.0, 55.0, 38.0
    cx20, cy20 = gx + gw / 2, gy + gh / 2
    x1_20 = int(cx20 - CROP_SIZE / 2)
    y1_20 = int(cy20 - CROP_SIZE / 2) + 20
    crop20 = img20[y1_20:y1_20 + CROP_SIZE, x1_20:x1_20 + CROP_SIZE]
    lgx = gx - x1_20
    lgy = gy - y1_20

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(crop20)
    ax.axis('off')
    rect = patches.Rectangle((lgx, lgy), gw, gh, linewidth=2.5, edgecolor='#00C853', facecolor='none')
    ax.add_patch(rect)
    ax.text(lgx, lgy - 8, 'Ground Truth: Drone (Missed)', color='#FFFFFF', fontsize=9.5, fontweight='bold',
            bbox=dict(facecolor='#00C853', edgecolor='none', boxstyle='square,pad=0.25'))
    out_fn = OUT_DIR / 'v1_clean_false_negative.png'
    plt.savefig(out_fn, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 2. False Positive Crop (Window Grid)
    img_wg = cv2.cvtColor(cv2.imread('perception/raw_data/hard_negatives/images/neg_window_grid_0025.jpg'), cv2.COLOR_BGR2RGB)
    wb = [228, 1148, 297, 1198]
    wcx, wcy = (wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2
    wx1 = max(0, int(wcx - CROP_SIZE / 2))
    wy1 = max(0, int(wcy - CROP_SIZE / 2))
    crop_wg = img_wg[wy1:wy1 + CROP_SIZE, wx1:wx1 + CROP_SIZE]
    lwx = wb[0] - wx1
    lwy = wb[1] - wy1
    lww = wb[2] - wb[0]
    lwh = wb[3] - wb[1]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(crop_wg)
    ax.axis('off')
    rect = patches.Rectangle((lwx, lwy), lww, lwh, linewidth=2.5, edgecolor='#E53935', facecolor='none')
    ax.add_patch(rect)
    ax.text(lwx, lwy - 8, 'False Positive (conf = 0.65)', color='#FFFFFF', fontsize=9.5, fontweight='bold',
            bbox=dict(facecolor='#E53935', edgecolor='none', boxstyle='square,pad=0.25'))
    out_fp = OUT_DIR / 'v1_clean_false_positive_window_grid.png'
    plt.savefig(out_fp, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Saved standalone crops: {out_fn} and {out_fp}")


if __name__ == "__main__":
    generate_clean_side_by_side_building()
    generate_clean_standalone_crops()

