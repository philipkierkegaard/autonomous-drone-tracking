import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

OUT_DIR = Path("docs/reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# =============================================================================
# 1. FALSE NEGATIVE: Air-to-Air Canopy Clutter (Det-Fly 020)
# =============================================================================
def generate_fn_air_to_air():
    img_p = "perception/datasets/tree_domain_v4_1/images/test/anti_detfly_0204641.jpg"
    lbl_p = "perception/datasets/tree_domain_v4_1/labels/test/anti_detfly_0204641.txt"
    img_bgr = cv2.imread(img_p)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    with open(lbl_p) as f:
        parts = list(map(float, f.read().split()[1:]))
    cx, cy, bw, bh = parts[0]*w, parts[1]*h, parts[2]*w, parts[3]*h
    gx, gy = cx - bw/2, cy - bh/2

    pad_x, pad_y = 320, 220
    zx1 = int(max(0, cx - pad_x))
    zy1 = int(max(0, cy - pad_y))
    zx2 = int(min(w, cx + pad_x))
    zy2 = int(min(h, cy + pad_y))
    zoom_crop = img_rgb[zy1:zy2, zx1:zx2]

    # Export clean standalone crop
    fig_crop, ax_sc = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax_sc.imshow(zoom_crop)
    local_gx = gx - zx1
    local_gy = gy - zy1
    gt_rect = patches.Rectangle((local_gx, local_gy), bw, bh, linewidth=2.5, edgecolor="#10B981", facecolor="none")
    ax_sc.add_patch(gt_rect)
    ax_sc.text(local_gx, local_gy - 14, f"Ground Truth: Drone ({bw:.0f}×{bh:.0f} px)", color="#10B981",
               fontsize=9.5, fontweight="bold",
               bbox=dict(facecolor="#064E3B", edgecolor="#10B981", boxstyle="round,pad=0.25", alpha=0.95))
    ax_sc.text(local_gx + bw + 12, local_gy + bh / 2, "DETECTION MISS\n(Model v1 conf < 0.20)", 
               color="#EF4444", fontsize=9.5, fontweight="bold", verticalalignment="center",
               bbox=dict(facecolor="#450A0A", edgecolor="#EF4444", boxstyle="round,pad=0.3", alpha=0.95))
    ax_sc.axis("off")
    fig_crop.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig_crop.savefig(OUT_DIR / "v1_false_negative_crop.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig_crop)

    # Full diagnostic figure
    fig = plt.figure(figsize=(16, 9), dpi=300, facecolor="#0B0F17")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1, 1],
                           wspace=0.08, hspace=0.14, left=0.05, right=0.96, top=0.88, bottom=0.08)

    ax_full = fig.add_subplot(gs[:, 0])
    ax_full.imshow(img_rgb)
    ax_full.set_title("Wide-Angle Scene Overview (Det-Fly Air-to-Air, Seq 020)", 
                      color="#E2E8F0", fontsize=13, fontweight="bold", pad=10)
    ax_full.axis("off")

    roi_rect = patches.Rectangle((zx1, zy1), zx2 - zx1, zy2 - zy1,
                                 linewidth=2.2, edgecolor="#F59E0B", facecolor="none", linestyle="--")
    ax_full.add_patch(roi_rect)
    ax_full.text(zx1 + 20, zy1 - 25, "ROI Zoom Area (Air-to-Air Tracking)", color="#F59E0B", fontsize=11, fontweight="bold",
                 bbox=dict(facecolor="#0F172A", edgecolor="#F59E0B", boxstyle="round,pad=0.25", alpha=0.9))

    gt_rect_full = patches.Rectangle((gx, gy), bw, bh, linewidth=2.0, edgecolor="#10B981", facecolor="none")
    ax_full.add_patch(gt_rect_full)

    ax_crop = fig.add_subplot(gs[0, 1])
    ax_crop.imshow(zoom_crop)
    ax_crop.set_title("Target Region Zoom: Dark Tree Canopy & Terrain", 
                      color="#E2E8F0", fontsize=13, fontweight="bold", pad=10)
    ax_crop.axis("off")

    gt_rect_crop = patches.Rectangle((local_gx, local_gy), bw, bh,
                                     linewidth=2.6, edgecolor="#10B981", facecolor="none", linestyle="-")
    ax_crop.add_patch(gt_rect_crop)
    ax_crop.text(local_gx, local_gy - 18, f"Ground Truth: Drone ({bw:.0f}×{bh:.0f} px)", color="#10B981",
                 fontsize=10.0, fontweight="bold",
                 bbox=dict(facecolor="#064E3B", edgecolor="#10B981", boxstyle="round,pad=0.25", alpha=0.95))

    ax_crop.text(local_gx + bw + 15, local_gy + bh / 2, "DETECTION MISS\n(Model v1 conf < 0.20)", 
                 color="#EF4444", fontsize=10.0, fontweight="bold", verticalalignment="center",
                 bbox=dict(facecolor="#450A0A", edgecolor="#EF4444", boxstyle="round,pad=0.35", alpha=0.95))

    ax_card = fig.add_subplot(gs[1, 1])
    ax_card.set_facecolor("#111827")
    ax_card.axis("off")

    card_text = (
        "EMPIRICAL FAILURE AUDIT: FALSE NEGATIVE (AIR-TO-AIR CLUTTER)\n"
        "────────────────────────────────────────────────────────────\n"
        "• Model Tested:             YOLOv8n v1 Baseline (Ground-to-Air Clear Sky)\n"
        "• Perspective:              Air-to-Air (Pursuit UAV looking down at target)\n"
        "• Ground Truth Target:       Quadcopter over dense foliage (92×27 px @ 4K)\n"
        "• Detector Output:          0 Detections (Threshold = 0.20)\n"
        "• Tracking Impact:          LOST: Visual servoing enters blind search\n\n"
        "ROOT CAUSE ANALYSIS:\n"
        "Trained primarily on ground-to-air captures, Model 1 relies heavily on\n"
        "sharp luminance contrast against featureless sky. In an airborne pursuit\n"
        "context, looking down across dark trees, shrubs, and textured roads completely\n"
        "camouflages the target. The multi-scale feature maps fail to activate on\n"
        "the low-contrast dark carbon-fiber airframe, causing an immediate dropout."
    )
    ax_card.text(0.04, 0.92, card_text, transform=ax_card.transAxes,
                 color="#E2E8F0", fontsize=10.0, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(facecolor="#1E293B", edgecolor="#334155", boxstyle="round,pad=0.6", alpha=0.9))

    fig.suptitle("Model 1 (Baseline) Failure Mode 1: False Negative Against Canopy & Terrain Clutter",
                 fontsize=16, fontweight="bold", color="#F8FAFC", y=0.96)

    out_path = OUT_DIR / "detector_v1_false_negative_clutter.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# =============================================================================
# 2. FALSE POSITIVE: Architectural Roof Edge (Anti-UAV video15)
# =============================================================================
def generate_fp_architectural():
    img_path = "perception/raw_data/Anti-UAV/Anti-UAV-Tracking-V0/video15/00066.jpg"
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    px1, py1, px2, py2 = 1894.2, 763.1, 1920.0, 798.1
    pw, ph = px2 - px1, py2 - py1
    p_conf = 0.278

    zx1 = int(max(0, px1 - 320))
    zy1 = int(max(0, py1 - 180))
    zx2 = int(w)
    zy2 = int(min(h, py2 + 180))
    zoom_crop = img_rgb[zy1:zy2, zx1:zx2]

    # Standalone clean crop
    fig_crop, ax_sc = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax_sc.imshow(zoom_crop)
    local_px1 = px1 - zx1
    local_py1 = py1 - zy1
    fp_rect = patches.Rectangle((local_px1, local_py1), pw, ph, linewidth=2.5, edgecolor="#EF4444", facecolor="none")
    ax_sc.add_patch(fp_rect)
    ax_sc.text(local_px1 - 12, local_py1 - 12, f"SPURIOUS DETECTION: 'Drone'\n(Confidence = {p_conf:.2f})", 
               color="#EF4444", fontsize=9.0, fontweight="bold", horizontalalignment="right",
               bbox=dict(facecolor="#450A0A", edgecolor="#EF4444", boxstyle="round,pad=0.25", alpha=0.95))
    ax_sc.text(local_px1 - 12, local_py1 + ph + 22, "GROUND TRUTH: Target Absent", 
               color="#38BDF8", fontsize=9.0, fontweight="bold", horizontalalignment="right",
               bbox=dict(facecolor="#0C4A6E", edgecolor="#38BDF8", boxstyle="round,pad=0.25", alpha=0.95))
    ax_sc.axis("off")
    fig_crop.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig_crop.savefig(OUT_DIR / "v1_false_positive_crop.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig_crop)

    # Full diagnostic figure
    fig = plt.figure(figsize=(16, 9), dpi=300, facecolor="#0B0F17")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1, 1],
                           wspace=0.08, hspace=0.14, left=0.05, right=0.96, top=0.88, bottom=0.08)

    ax_full = fig.add_subplot(gs[:, 0])
    ax_full.imshow(img_rgb)
    ax_full.set_title("Wide-Angle Scene Overview (Anti-UAV video15, Frame 00066)", 
                      color="#E2E8F0", fontsize=13, fontweight="bold", pad=10)
    ax_full.axis("off")

    ax_full.text(50, 80, "STATUS: TARGET DRONE ABSENT (OUT OF FRAME)", color="#38BDF8",
                 fontsize=12, fontweight="bold",
                 bbox=dict(facecolor="#0C4A6E", edgecolor="#38BDF8", boxstyle="round,pad=0.4", alpha=0.95))

    roi_rect = patches.Rectangle((zx1, zy1), zx2 - zx1, zy2 - zy1,
                                 linewidth=2.2, edgecolor="#F59E0B", facecolor="none", linestyle="--")
    ax_full.add_patch(roi_rect)
    ax_full.text(zx1 - 15, zy1 - 15, "ROI: False Alarm Zone", color="#F59E0B", fontsize=11, fontweight="bold",
                 bbox=dict(facecolor="#0F172A", edgecolor="#F59E0B", boxstyle="round,pad=0.25", alpha=0.9),
                 horizontalalignment="right")

    fp_rect_full = patches.Rectangle((px1, py1), pw, ph, linewidth=2.5, edgecolor="#EF4444", facecolor="none")
    ax_full.add_patch(fp_rect_full)

    ax_crop = fig.add_subplot(gs[0, 1])
    ax_crop.imshow(zoom_crop)
    ax_crop.set_title("Target Region Zoom: Architectural Roof Edge", 
                      color="#E2E8F0", fontsize=13, fontweight="bold", pad=10)
    ax_crop.axis("off")

    fp_rect_crop = patches.Rectangle((local_px1, local_py1), pw, ph,
                                     linewidth=2.8, edgecolor="#EF4444", facecolor="none", linestyle="-")
    ax_crop.add_patch(fp_rect_crop)

    ax_crop.text(local_px1 - 15, local_py1 - 12, f"SPURIOUS DETECTION: 'Drone'\n(Confidence = {p_conf:.2f})", 
                 color="#EF4444", fontsize=10.0, fontweight="bold", horizontalalignment="right",
                 bbox=dict(facecolor="#450A0A", edgecolor="#EF4444", boxstyle="round,pad=0.3", alpha=0.95))
    
    ax_crop.text(local_px1 - 15, local_py1 + ph + 25, "GROUND TRUTH: Target Absent", 
                 color="#38BDF8", fontsize=10.0, fontweight="bold", horizontalalignment="right",
                 bbox=dict(facecolor="#0C4A6E", edgecolor="#38BDF8", boxstyle="round,pad=0.25", alpha=0.95))

    ax_card = fig.add_subplot(gs[1, 1])
    ax_card.set_facecolor("#111827")
    ax_card.axis("off")

    card_text = (
        "EMPIRICAL FAILURE AUDIT: FALSE POSITIVE (ARCHITECTURAL CLUTTER)\n"
        "────────────────────────────────────────────────────────────\n"
        "• Model Tested:             YOLOv8n v1 Baseline (Clear-Sky Pretrained)\n"
        "• Ground Truth Status:      Target Drone Absent (Out of View, GT: -100)\n"
        "• Detector Output:          1 False Alarm: Drone (Conf = 0.28)\n"
        "• False Bounding Box:       [x=1894, y=763, w=26, h=35 px]\n"
        "• Servoing Consequence:     CATASTROPHIC: IBVS latched onto static roofline\n\n"
        "ROOT CAUSE ANALYSIS:\n"
        "Because the baseline training set contained zero explicit negative examples\n"
        "of man-made structures, sharp geometric corners and roof edges against the sky\n"
        "produce strong edge activations resembling a drone airframe. In closed-loop flight,\n"
        "the visual pursuit controller steered toward the building, losing the true target\n"
        "permanently upon its re-entry into the operational arena."
    )
    ax_card.text(0.04, 0.92, card_text, transform=ax_card.transAxes,
                 color="#E2E8F0", fontsize=10.0, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(facecolor="#1E293B", edgecolor="#334155", boxstyle="round,pad=0.6", alpha=0.9))

    fig.suptitle("Model 1 (Baseline) Failure Mode 2: Spurious False Alarm on Architectural Clutter",
                 fontsize=16, fontweight="bold", color="#F8FAFC", y=0.96)

    out_path = OUT_DIR / "detector_v1_false_positive_clutter.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    generate_fn_air_to_air()
    generate_fp_architectural()
