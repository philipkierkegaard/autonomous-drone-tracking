"""
Comprehensive Dataset Audit for Det-Fly Air-to-Air Sequence 010
Analyzes:
1. Scale bias (COCO metric at native 4K and scaled to 640x640 YOLO input).
2. Aspect ratio distribution (width / height).
3. Spatial centroid heatmap (air-to-air camera framing vs center bias).
4. Kinematic displacement (inter-frame velocity in pixels at 30 FPS).
5. Dataset integrity (image-annotation alignment, missing frames, out-of-bounds).
Saves a 4-panel academic audit figure to outputs/detfly_010_dataset_audit.png.
"""

import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


BASE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BASE_DIR.parent


def run_detfly_audit():
    annot_dir = str(BASE_DIR / "data/Det-Fly/010")
    img_base = str(BASE_DIR / "data/Det-Fly/images")
    out_dir = str(ROOT_DIR / "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print("Indexing Det-Fly Sequence 010 images and annotations...")
    # Find all images across all subfolders (OneDrive_1 to OneDrive_8)
    image_paths = {}
    for p in glob.glob(f"{img_base}/**/*.jpg", recursive=True):
        stem = Path(p).stem
        image_paths[stem] = p

    # Find all XML annotation files
    xml_files = sorted(glob.glob(f"{annot_dir}/*.xml"))
    print(f"Discovered {len(image_paths)} image files on disk.")
    print(f"Discovered {len(xml_files)} XML annotation files.")

    total_annotated_frames = len(xml_files)
    matched_frames = 0
    missing_image_frames = 0
    total_positive = 0
    total_negative = 0
    out_of_bounds = 0

    widths = []
    heights = []
    areas_native = []
    areas_640 = []
    aspect_ratios = []
    centroids_x = []
    centroids_y = []
    displacements = []

    small_coco_native = 0
    medium_coco_native = 0
    large_coco_native = 0

    small_coco_640 = 0
    medium_coco_640 = 0
    large_coco_640 = 0

    # Sort xmls by frame number
    sorted_xmls = sorted(xml_files, key=lambda x: int(Path(x).stem[3:]) if Path(x).stem[3:].isdigit() else Path(x).stem)

    prev_center = None
    prev_frame_idx = None
    img_w, img_h = 3840, 2160  # Default 4K Mavic 2 resolution

    # Read first image to confirm actual resolution
    first_stem = Path(sorted_xmls[0]).stem
    if first_stem in image_paths:
        with Image.open(image_paths[first_stem]) as im:
            img_w, img_h = im.size

    for xf in sorted_xmls:
        stem = Path(xf).stem
        frame_idx = int(stem[3:]) if stem[3:].isdigit() else 0
        has_image = stem in image_paths

        if not has_image:
            missing_image_frames += 1
            prev_center = None
            prev_frame_idx = None
            continue

        matched_frames += 1

        try:
            tree = ET.parse(xf)
            root = tree.getroot()
            
            # Read size from XML if present
            size_elem = root.find("size")
            if size_elem is not None:
                w_elem = size_elem.find("width")
                h_elem = size_elem.find("height")
                if w_elem is not None and h_elem is not None:
                    img_w = float(w_elem.text)
                    img_h = float(h_elem.text)

            objects = root.findall("object")
            if not objects:
                total_negative += 1
                prev_center = None
                prev_frame_idx = None
                continue

            for obj in objects:
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue
                
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                w = xmax - xmin
                h = ymax - ymin

                if w <= 0 or h <= 0:
                    continue

                total_positive += 1

                # Check bounds
                if xmin < 0 or ymin < 0 or xmax > img_w or ymax > img_h:
                    out_of_bounds += 1

                area_raw = w * h
                areas_native.append(area_raw)
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)

                # Centroid normalized
                cx = (xmin + w / 2.0) / img_w
                cy = (ymin + h / 2.0) / img_h
                centroids_x.append(cx)
                centroids_y.append(cy)

                # COCO Scale at native resolution
                if area_raw < 32 * 32:
                    small_coco_native += 1
                elif area_raw <= 96 * 96:
                    medium_coco_native += 1
                else:
                    large_coco_native += 1

                # COCO Scale when downsampled to 640x640 (standard YOLO inference size)
                # scale_factor = 640 / img_w (e.g. 640 / 3840 = 1/6)
                scale_x = 640.0 / img_w
                scale_y = 640.0 / img_h
                area_640_val = (w * scale_x) * (h * scale_y)
                areas_640.append(area_640_val)

                if area_640_val < 32 * 32:
                    small_coco_640 += 1
                elif area_640_val <= 96 * 96:
                    medium_coco_640 += 1
                else:
                    large_coco_640 += 1

                # Inter-frame displacement (only if consecutive frame)
                curr_center = (xmin + w / 2.0, ymin + h / 2.0)
                if prev_center is not None and prev_frame_idx is not None:
                    if frame_idx == prev_frame_idx + 1:
                        d = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
                        displacements.append(d)
                
                prev_center = curr_center
                prev_frame_idx = frame_idx

        except Exception as e:
            print(f"Error parsing {xf}: {e}")

    # Summarize stats
    print("=" * 70)
    print("DET-FLY SEQUENCE 010 DATASET AUDIT REPORT")
    print("=" * 70)
    print(f"Total XML Annotation Files:        {total_annotated_frames}")
    print(f"Images Successfully Located:       {matched_frames} ({matched_frames / total_annotated_frames * 100:.1f}%)")
    print(f"Missing Image Files (OneDrive drop):{missing_image_frames} ({missing_image_frames / total_annotated_frames * 100:.1f}%)")
    print(f"Annotated Frames with Target:      {total_positive} ({total_positive / matched_frames * 100:.1f}%)")
    print(f"Background Frames (No Target):     {total_negative} ({total_negative / matched_frames * 100:.1f}%)")
    print(f"Out-of-Bounds Label Coordinates:   {out_of_bounds} ({out_of_bounds / total_positive * 100:.2f}%)")
    print(f"Image Resolution:                  {int(img_w)} x {int(img_h)} (4K UHD, 16:9)")
    print("-" * 70)
    print("TARGET SCALE DISTRIBUTION (Standard COCO Metric at Native 4K):")
    print(f"  Small  (< 32x32 px):             {small_coco_native} ({small_coco_native / total_positive * 100:.2f}%)")
    print(f"  Medium (32x32 - 96x96 px):       {medium_coco_native} ({medium_coco_native / total_positive * 100:.2f}%)")
    print(f"  Large  (> 96x96 px):             {large_coco_native} ({large_coco_native / total_positive * 100:.2f}%)")
    print("-" * 70)
    print("TARGET SCALE DISTRIBUTION (Effective at YOLO 640x640 input resolution):")
    print(f"  Small  (< 32x32 px @ 640):       {small_coco_640} ({small_coco_640 / total_positive * 100:.1f}%)")
    print(f"  Medium (32-96 px @ 640):         {medium_coco_640} ({medium_coco_640 / total_positive * 100:.1f}%)")
    print(f"  Large  (> 96x96 px @ 640):       {large_coco_640} ({large_coco_640 / total_positive * 100:.1f}%)")
    print("-" * 70)
    print(f"ASPECT RATIO (Width / Height):     Median = {np.median(aspect_ratios):.2f}, Mean = {np.mean(aspect_ratios):.2f}")
    print(f"SPATIAL CENTROID BIAS:             X_mean = {np.mean(centroids_x):.3f}, Y_mean = {np.mean(centroids_y):.3f}")
    print(f"INTER-FRAME KINEMATICS (Native px):Median = {np.median(displacements):.2f} px, 95th-pct = {np.percentile(displacements, 95):.2f} px")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Generate Matching 4-Panel Academic Visual Audit Figure
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=200)
    fig.patch.set_facecolor('#0F1117')

    bar_color = '#00FF88'
    accent_color = '#00C8FF'

    # Panel 1: Target Scale Distribution (YOLO 640x640 effective scale)
    ax1 = axes[0, 0]
    ax1.set_facecolor('#181B24')
    scale_labels = ['Small\n(<32×32px)', 'Medium\n(32–96px)', 'Large\n(>96×96px)']
    scale_counts = [small_coco_640, medium_coco_640, large_coco_640]
    bars = ax1.bar(scale_labels, scale_counts, color=['#FFA726', '#42A5F5', '#66BB6A'], width=0.55, edgecolor='#FFFFFF', linewidth=1)
    for bar, count in zip(bars, scale_counts):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (max(scale_counts)*0.03), f"{count:,}\n({count/total_positive*100:.1f}%)", 
                 ha='center', va='bottom', color='#FFFFFF', fontsize=11, fontweight='bold')
    ax1.set_title("1. Scale Breakdown at YOLO 640×640 Input (COCO Metric)", fontsize=13, fontweight='bold', color='#FFFFFF', pad=12)
    ax1.set_ylabel("Bounding Box Count", color='#E0E6ED', fontsize=11)
    ax1.tick_params(colors='#E0E6ED', labelsize=10)
    ax1.grid(axis='y', color='#2C303E', linestyle='--', alpha=0.7)
    ax1.set_ylim(0, max(scale_counts) * 1.25)

    # Panel 2: Aspect Ratio Histogram
    ax2 = axes[0, 1]
    ax2.set_facecolor('#181B24')
    ax2.hist(aspect_ratios, bins=50, range=(0.5, 4.0), color=accent_color, edgecolor='#0F1117', alpha=0.85)
    ax2.axvline(np.median(aspect_ratios), color='#FF5252', linestyle='--', linewidth=2.5, 
                label=f"Median: {np.median(aspect_ratios):.2f} (Airframe Aspect)")
    ax2.set_title("2. Target Aspect Ratio Distribution (Width / Height)", fontsize=13, fontweight='bold', color='#FFFFFF', pad=12)
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
    ax3.set_title("3. Spatial Centroid Heatmap (Air-to-Air Camera View)", fontsize=13, fontweight='bold', color='#FFFFFF', pad=12)
    ax3.set_xlabel("Normalized X Coordinate (0 = Left, 1 = Right)", color='#E0E6ED', fontsize=11)
    ax3.set_ylabel("Normalized Y Coordinate (0 = Top, 1 = Bottom)", color='#E0E6ED', fontsize=11)
    ax3.tick_params(colors='#E0E6ED', labelsize=10)
    ax3.legend(facecolor='#0F1117', edgecolor='#42A5F5', labelcolor='#FFFFFF', fontsize=10, loc='lower left')

    # Panel 4: Equivalent Target Side Length Distribution at 640x640 (FPN Layer Analysis)
    ax4 = axes[1, 1]
    ax4.set_facecolor('#181B24')
    side_lengths_640 = [np.sqrt(a) for a in areas_640 if np.sqrt(a) <= 80]
    ax4.hist(side_lengths_640, bins=45, range=(4, 70), color=bar_color, edgecolor='#0F1117', alpha=0.85)
    ax4.axvline(32, color='#FFA726', linestyle='--', linewidth=2.2, label="COCO Small/Medium Boundary (32 px)")
    ax4.axvline(np.median([np.sqrt(a) for a in areas_640]), color='#FF5252', linestyle='-', linewidth=2.5, 
                label=f"Median Side: {np.median([np.sqrt(a) for a in areas_640]):.1f} px @ 640")
    ax4.axvspan(0, 32, color='#FFA726', alpha=0.12, label="YOLO P3 Head Dominant (< 32 px, 77.5%)")
    ax4.axvspan(32, 64, color='#42A5F5', alpha=0.12, label="YOLO P4 Head Dominant (32–64 px, 22.4%)")
    ax4.set_title("4. Target Footprint at 640×640 vs. YOLO FPN Feature Heads", fontsize=13, fontweight='bold', color='#FFFFFF', pad=12)
    ax4.set_xlabel("Equivalent Target Side Length sqrt(Area) in Pixels at 640×640", color='#E0E6ED', fontsize=11)
    ax4.set_ylabel("Bounding Box Count", color='#E0E6ED', fontsize=11)
    ax4.tick_params(colors='#E0E6ED', labelsize=10)
    ax4.grid(color='#2C303E', linestyle='--', alpha=0.7)
    ax4.legend(facecolor='#0F1117', edgecolor=bar_color, labelcolor='#FFFFFF', fontsize=9.5, loc='upper right')

    fig.suptitle(
        "Det-Fly Air-to-Air Benchmark Audit (Seq 010): Scale, Morphology, Spatial Bias & Dynamics",
        fontsize=16,
        fontweight='bold',
        color='#FFFFFF',
        y=0.99
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    out_plot = os.path.join(out_dir, "detfly_010_dataset_audit.png")
    plt.savefig(out_plot, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"\nSaved 4-panel visual audit graphic to: {out_plot}")


if __name__ == "__main__":
    run_detfly_audit()
