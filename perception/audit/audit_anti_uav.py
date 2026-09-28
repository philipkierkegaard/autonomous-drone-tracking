"""
Comprehensive Dataset Audit for Anti-UAV Benchmark
Analyzes:
1. Scale bias (COCO metric: small, medium, large).
2. Aspect ratio distribution (width / height).
3. Spatial centroid heatmap (PTZ operator center bias).
4. Kinematic displacement (inter-frame velocity in pixels).
5. Label integrity (out-of-bounds, background proportion).
Saves a 4-panel academic audit figure to outputs/anti_uav_dataset_audit.png.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BASE_DIR.parent


def run_audit():
    gt_dir = BASE_DIR / "data/Anti-UAV/Anti-UAV-Tracking-V0GT"
    video_dir = BASE_DIR / "data/Anti-UAV/Anti-UAV-Tracking-V0"

    gt_files = sorted(glob.glob(f"{gt_dir}/*.txt"))
    
    total_frames = 0
    total_positive = 0
    total_negative = 0
    out_of_bounds = 0

    widths = []
    heights = []
    areas = []
    aspect_ratios = []
    centroids_x = []
    centroids_y = []
    displacements = []

    small_count = 0
    medium_count = 0
    large_count = 0

    for gf in gt_files:
        vname = Path(gf).stem.replace("_gt", "")
        v_path = Path(video_dir) / vname
        img_files = sorted(glob.glob(f"{v_path}/*.jpg"))
        if not img_files:
            continue
        
        with Image.open(img_files[0]) as im:
            iw, ih = im.size

        with open(gf, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        prev_center = None
        for l in lines:
            total_frames += 1
            parts = l.replace(",", " ").split()
            has_box = False
            if len(parts) >= 4:
                try:
                    x, y, w, h = map(float, parts[:4])
                    if w > 0 and h > 0:
                        total_positive += 1
                        has_box = True
                        
                        # Boundary check
                        if x < 0 or y < 0 or (x + w) > iw or (y + h) > ih:
                            out_of_bounds += 1

                        area = w * h
                        areas.append(area)
                        widths.append(w)
                        heights.append(h)
                        aspect_ratios.append(w / h)

                        # Normalized centroid
                        cx = (x + w / 2.0) / iw
                        cy = (y + h / 2.0) / ih
                        centroids_x.append(cx)
                        centroids_y.append(cy)

                        # Scale binning (COCO standard)
                        if area < 32 * 32:
                            small_count += 1
                        elif area <= 96 * 96:
                            medium_count += 1
                        else:
                            large_count += 1

                        # Inter-frame displacement
                        curr_center = (x + w / 2.0, y + h / 2.0)
                        if prev_center is not None:
                            d = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
                            displacements.append(d)
                        prev_center = curr_center
                except ValueError:
                    pass

            if not has_box:
                total_negative += 1
                prev_center = None

    # Compute summaries
    print("=" * 65)
    print("ANTI-UAV DATASET AUDIT SUMMARY")
    print("=" * 65)
    print(f"Total Video Sequences:     {len(gt_files)}")
    print(f"Total Evaluated Frames:    {total_frames}")
    print(f"Frames with Target Drone:  {total_positive} ({total_positive / total_frames * 100:.1f}%)")
    print(f"Background/Occluded Frames:{total_negative} ({total_negative / total_frames * 100:.1f}%)")
    print(f"Out-of-Bounds Label Boxes: {out_of_bounds} ({out_of_bounds / total_positive * 100:.2f}%)")
    print("-" * 65)
    print("SCALE DISTRIBUTION (COCO Metric):")
    print(f"  Small  (< 32x32 px):     {small_count} ({small_count / total_positive * 100:.1f}%)")
    print(f"  Medium (32x32 - 96x96):  {medium_count} ({medium_count / total_positive * 100:.1f}%)")
    print(f"  Large  (> 96x96 px):     {large_count} ({large_count / total_positive * 100:.1f}%)")
    print("-" * 65)
    print(f"ASPECT RATIO (w / h):      Median = {np.median(aspect_ratios):.2f}, Mean = {np.mean(aspect_ratios):.2f}")
    print(f"CENTROID SPATIAL BIAS:     X_mean = {np.mean(centroids_x):.3f}, Y_mean = {np.mean(centroids_y):.3f}")
    print(f"FRAME DISPLACEMENT (px):   Median = {np.median(displacements):.2f} px, 95th-pct = {np.percentile(displacements, 95):.2f} px")
    print("=" * 65)

    # -----------------------------------------------------------------
    # Generate 4-Panel Academic Audit Plot
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=200)
    fig.patch.set_facecolor('#0F1117')

    # Color palette
    bar_color = '#00FF88'
    accent_color = '#00C8FF'

    # Panel 1: Target Scale Distribution
    ax1 = axes[0, 0]
    ax1.set_facecolor('#181B24')
    scale_labels = ['Small\n(<32×32px)', 'Medium\n(32–96px)', 'Large\n(>96×96px)']
    scale_counts = [small_count, medium_count, large_count]
    bars = ax1.bar(scale_labels, scale_counts, color=['#FFA726', '#42A5F5', '#66BB6A'], width=0.55, edgecolor='#FFFFFF', linewidth=1)
    for bar, count in zip(bars, scale_counts):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 350, f"{count:,}\n({count/total_positive*100:.1f}%)", 
                 ha='center', va='bottom', color='#FFFFFF', fontsize=11, fontweight='bold')
    ax1.set_title("1. Target Scale Breakdown (COCO Metric)", fontsize=13, fontweight='bold', color='#FFFFFF', pad=12)
    ax1.set_ylabel("Bounding Box Count", color='#E0E6ED', fontsize=11)
    ax1.tick_params(colors='#E0E6ED', labelsize=10)
    ax1.grid(axis='y', color='#2C303E', linestyle='--', alpha=0.7)
    ax1.set_ylim(0, max(scale_counts) * 1.2)

    # Panel 2: Aspect Ratio Histogram
    ax2 = axes[0, 1]
    ax2.set_facecolor('#181B24')
    ax2.hist(aspect_ratios, bins=50, range=(0.5, 4.0), color=accent_color, edgecolor='#0F1117', alpha=0.85)
    ax2.axvline(np.median(aspect_ratios), color='#FF5252', linestyle='--', linewidth=2.5, 
                label=f"Median: {np.median(aspect_ratios):.2f} (Horizontal Arm Bias)")
    ax2.set_title("2. Aspect Ratio Distribution (Width / Height)", fontsize=13, fontweight='bold', color='#FFFFFF', pad=12)
    ax2.set_xlabel("Aspect Ratio (w / h)", color='#E0E6ED', fontsize=11)
    ax2.set_ylabel("Frequency", color='#E0E6ED', fontsize=11)
    ax2.tick_params(colors='#E0E6ED', labelsize=10)
    ax2.grid(color='#2C303E', linestyle='--', alpha=0.7)
    ax2.legend(facecolor='#0F1117', edgecolor='#42A5F5', labelcolor='#FFFFFF', fontsize=10)

    # Panel 3: 2D Spatial Centroid Heatmap
    ax3 = axes[1, 0]
    ax3.set_facecolor('#181B24')
    h2d = ax3.hist2d(centroids_x, centroids_y, bins=45, range=[[0, 1], [0, 1]], cmap='inferno')
    ax3.invert_yaxis()  # Image coordinate system (0,0 at top-left)
    ax3.axhline(0.5, color='#FFFFFF', linestyle=':', alpha=0.4)
    ax3.axvline(0.5, color='#FFFFFF', linestyle=':', alpha=0.4)
    ax3.plot(np.mean(centroids_x), np.mean(centroids_y), 'cx', markersize=14, markeredgewidth=3, 
             label=f"Center of Mass: ({np.mean(centroids_x):.2f}, {np.mean(centroids_y):.2f})")
    ax3.set_title("3. Spatial Centroid Heatmap (Operator Framing Bias)", fontsize=13, fontweight='bold', color='#FFFFFF', pad=12)
    ax3.set_xlabel("Normalized X Coordinate (0 = Left, 1 = Right)", color='#E0E6ED', fontsize=11)
    ax3.set_ylabel("Normalized Y Coordinate (0 = Top, 1 = Bottom)", color='#E0E6ED', fontsize=11)
    ax3.tick_params(colors='#E0E6ED', labelsize=10)
    ax3.legend(facecolor='#0F1117', edgecolor='#42A5F5', labelcolor='#FFFFFF', fontsize=10, loc='lower left')

    # Panel 4: Inter-Frame Motion Displacement
    ax4 = axes[1, 1]
    ax4.set_facecolor('#181B24')
    ax4.hist(displacements, bins=50, range=(0, 35), color=bar_color, edgecolor='#0F1117', alpha=0.85)
    ax4.axvline(np.median(displacements), color='#FF5252', linestyle='--', linewidth=2.5, 
                label=f"Median: {np.median(displacements):.1f} px/frame")
    ax4.axvline(np.percentile(displacements, 95), color='#FFA726', linestyle=':', linewidth=2, 
                label=f"95th Percentile: {np.percentile(displacements, 95):.1f} px/frame")
    ax4.set_title("4. Inter-Frame Kinematic Velocity Distribution", fontsize=13, fontweight='bold', color='#FFFFFF', pad=12)
    ax4.set_xlabel("Inter-Frame Centroid Displacement (Pixels at 30 FPS)", color='#E0E6ED', fontsize=11)
    ax4.set_ylabel("Frequency", color='#E0E6ED', fontsize=11)
    ax4.tick_params(colors='#E0E6ED', labelsize=10)
    ax4.grid(color='#2C303E', linestyle='--', alpha=0.7)
    ax4.legend(facecolor='#0F1117', edgecolor=bar_color, labelcolor='#FFFFFF', fontsize=10)

    fig.suptitle(
        "Anti-UAV Benchmark Dataset Audit: Scale, Morphology, Spatial Bias & Motion Dynamics",
        fontsize=16,
        fontweight='bold',
        color='#FFFFFF',
        y=0.99
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    out_dir = ROOT_DIR / "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_plot = out_dir / "anti_uav_dataset_audit.png"
    plt.savefig(out_plot, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"\nSaved 4-panel visual audit graphic to: {out_plot}")


if __name__ == "__main__":
    run_audit()
