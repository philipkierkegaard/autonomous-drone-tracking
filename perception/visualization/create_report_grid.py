"""
Generate Publication-Quality 2x3 Labeled Image Grid from Anti-UAV
Creates a 2-row by 3-column figure of full-frame images with clean bounding boxes
and labels, without zoom insets.
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def draw_clean_bbox(
    img_bgr: np.ndarray,
    label_line: str,
    color=(0, 255, 64)
) -> np.ndarray:
    """
    Renders high-contrast bounding box and clean label badge on image.
    """
    img = img_bgr.copy()
    h, w = img.shape[:2]

    parts = label_line.strip().split()
    if len(parts) < 5:
        return img

    cls_id, xc, yc, nw, nh = map(float, parts[:5])
    
    # Pixel coordinates
    box_w = int(nw * w)
    box_h = int(nh * h)
    xmin = int((xc - nw / 2.0) * w)
    ymin = int((yc - nh / 2.0) * h)
    xmax = xmin + box_w
    ymax = ymin + box_h

    # Clamping
    xmin_c = max(0, min(w - 1, xmin))
    ymin_c = max(0, min(h - 1, ymin))
    xmax_c = max(0, min(w - 1, xmax))
    ymax_c = max(0, min(h - 1, ymax))

    # 1. Dual Bounding Box (Dark outer border + Bright Neon Green)
    cv2.rectangle(img, (xmin_c - 2, ymin_c - 2), (xmax_c + 2, ymax_c + 2), (0, 0, 0), 4)
    cv2.rectangle(img, (xmin_c, ymin_c), (xmax_c, ymax_c), color, 2)

    # 2. Text Label Badge
    label_str = f"Target Drone [{box_w}x{box_h}px]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thick = 2
    (text_w, text_h), baseline = cv2.getTextSize(label_str, font, font_scale, font_thick)

    badge_ymin = max(5, ymin_c - text_h - 14)
    badge_ymax = badge_ymin + text_h + 10
    badge_xmin = max(5, xmin_c)
    badge_xmax = min(w - 5, badge_xmin + text_w + 14)

    # Dark badge background with high opacity
    badge_overlay = img.copy()
    cv2.rectangle(badge_overlay, (badge_xmin, badge_ymin), (badge_xmax, badge_ymax), (15, 15, 15), -1)
    cv2.addWeighted(badge_overlay, 0.85, img, 0.15, 0, img)
    cv2.rectangle(img, (badge_xmin, badge_ymin), (badge_xmax, badge_ymax), color, 1)

    # Text inside badge
    cv2.putText(
        img,
        label_str,
        (badge_xmin + 7, badge_ymin + text_h + 4),
        font,
        font_scale,
        (255, 255, 255),
        font_thick,
        cv2.LINE_AA
    )

    return img


BASE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BASE_DIR.parent


def generate_2x3_grid():
    target_sequences = [
        "video01", "video04", "video06",
        "video07", "video08", "video10"
    ]

    all_labels = glob.glob(str(BASE_DIR / "datasets/anti_uav_yolo/labels/*/*.txt"))
    
    samples = []
    for seq in target_sequences:
        seq_files = [f for f in all_labels if Path(f).name.startswith(f"{seq}_") and os.path.getsize(f) > 0]
        if not seq_files:
            continue
        seq_files.sort()
        # Choose a stable middle frame where target is tracked
        mid_file = seq_files[len(seq_files) // 2]
        stem = Path(mid_file).stem
        
        # Locate corresponding image
        split = Path(mid_file).parent.name
        img_path = BASE_DIR / f"datasets/anti_uav_yolo/images/{split}/{stem}.jpg"
        
        with open(mid_file, "r") as fp:
            lbl_line = fp.readline().strip()
            
        samples.append({
            "seq": seq,
            "stem": stem,
            "img_path": img_path,
            "lbl_line": lbl_line
        })

    print(f"Collected {len(samples)} representative samples for 2x3 grid.")

    # Process and annotate images
    rendered_images = []
    for s in samples:
        img_bgr = cv2.imread(str(s["img_path"]))
        if img_bgr is None:
            print(f"Warning: Could not read {s['img_path']}")
            continue
        
        annotated = draw_clean_bbox(img_bgr, s["lbl_line"])
        img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        rendered_images.append((s, img_rgb))

    os.makedirs("outputs", exist_ok=True)

    # -------------------------------------------------------------
    # Generate 2 x 3 Grid (2 Rows, 3 Columns)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), dpi=200)
    fig.patch.set_facecolor('#0F1117')
    axes_flat = axes.flatten()

    for i, (meta, img_rgb) in enumerate(rendered_images[:6]):
        ax = axes_flat[i]
        ax.imshow(img_rgb)
        frame_num = meta["stem"].split("_")[-1]
        ax.set_title(
            f"Sequence: {meta['seq']} | Frame #{frame_num}",
            fontsize=13,
            fontweight='bold',
            color='#E0E6ED',
            pad=8
        )
        ax.axis("off")

    fig.suptitle(
        "Anti-UAV Dataset: Labeled Target Drone Detection Samples (2 x 3 Grid)",
        fontsize=17,
        fontweight='bold',
        color='#FFFFFF',
        y=0.98
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    out_dir = ROOT_DIR / "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "anti_uav_2x3_grid.png"
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Successfully saved 2x3 grid to: {out_path}")


if __name__ == "__main__":
    generate_2x3_grid()
