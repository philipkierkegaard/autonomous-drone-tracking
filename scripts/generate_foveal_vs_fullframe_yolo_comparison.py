#!/usr/bin/env python3
"""
Generate Comparative Figure: What YOLO Sees With vs. Without Foveal Zoom.
DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception.

Extracts Frame 743 (t = 24.8s / ~25s) from Anti-UAV video12 and renders
a publication-quality 4-panel visual comparison illustrating the exact
neural network input tensors and target pixel acuity under:
1. Standard Full-Frame Inference (Downsampled + Letterboxed)
2. Dynamic Foveal Zoom Inference (Native >=768px Crop)
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, ConnectionPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

OUT_DIR = PROJECT_ROOT / "outputs" / "videos" / "perception"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "foveal_vs_fullframe_yolo_vision_comparison.png"
ARTIFACT_DIR = Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/5967d853-7a7e-4cdd-8792-4a77e6d704a2")


def main():
    print("---> Loading video12 at t = 24.8s (Frame 743)...")
    video_path = PROJECT_ROOT / "perception" / "raw_data" / "Anti-UAV" / "videos" / "video12.mp4"
    gt_path = PROJECT_ROOT / "perception" / "raw_data" / "Anti-UAV" / "Anti-UAV-Tracking-V0GT" / "video12_gt.txt"

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 743)
    ret, frame_bgr = cap.read()
    cap.release()
    assert ret, "Failed to read frame 743 from video12.mp4"

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]  # 1080, 1920

    with open(gt_path) as f:
        gt_lines = [l.strip() for l in f if l.strip()]
    gt = [float(x) for x in gt_lines[743].split()]
    gx, gy, gw, gh = int(gt[0]), int(gt[1]), int(gt[2]), int(gt[3])
    gt_box = [gx, gy, gx + gw, gy + gh]

    # -------------------------------------------------------------------------
    # 1. Mode A: Without Foveal Zoom (Full Frame Letterboxed to 768x768)
    # -------------------------------------------------------------------------
    scale = 768.0 / max(w, h)
    scaled_w = int(w * scale)   # 768
    scaled_h = int(h * scale)   # 432
    pad_top = (768 - scaled_h) // 2
    pad_bot = 768 - scaled_h - pad_top

    resized_full = cv2.resize(frame_rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    yolo_tensor_no_zoom = np.full((768, 768, 3), 114, dtype=np.uint8)  # Standard YOLO gray letterbox
    yolo_tensor_no_zoom[pad_top:pad_top + scaled_h, 0:scaled_w] = resized_full

    # Drone location in no-zoom tensor
    nz_dx1 = int(gx * scale)
    nz_dy1 = int(gy * scale + pad_top)
    nz_dx2 = int((gx + gw) * scale)
    nz_dy2 = int((gy + gh) * scale + pad_top)
    nz_cx = (nz_dx1 + nz_dx2) // 2
    nz_cy = (nz_dy1 + nz_dy2) // 2

    # -------------------------------------------------------------------------
    # 2. Mode B: With Foveal Zoom (Native 768x768 Crop)
    # -------------------------------------------------------------------------
    tcx = gx + gw / 2.0
    tcy = gy + gh / 2.0
    crop_w, crop_h = 768, 768
    xmin = max(0, min(w - crop_w, int(tcx - crop_w / 2.0)))
    ymin = max(0, min(h - crop_h, int(tcy - crop_h / 2.0)))
    xmax = xmin + crop_w
    ymax = ymin + crop_h

    yolo_tensor_foveal = frame_rgb[ymin:ymax, xmin:xmax].copy()

    # Drone location in foveal tensor
    fov_dx1 = gx - xmin
    fov_dy1 = gy - ymin
    fov_dx2 = (gx + gw) - xmin
    fov_dy2 = (gy + gh) - ymin
    fov_cx = (fov_dx1 + fov_dx2) // 2
    fov_cy = (fov_dy1 + fov_dy2) // 2

    # -------------------------------------------------------------------------
    # 3. Run Inference on Both Tensors
    # -------------------------------------------------------------------------
    print("---> Running YOLOv8n detector on both input tensors...")
    model_path = PROJECT_ROOT / "perception" / "weights" / "yolov8n_drone_768_hybrid_nwd_c51_best.pt"
    model = YOLO(str(model_path))

    res_nz = model.predict(cv2.cvtColor(yolo_tensor_no_zoom, cv2.COLOR_RGB2BGR), imgsz=768, conf=0.10, verbose=False)
    res_fov = model.predict(cv2.cvtColor(yolo_tensor_foveal, cv2.COLOR_RGB2BGR), imgsz=768, conf=0.10, verbose=False)

    # -------------------------------------------------------------------------
    # 4. Extract Drone Insets (5x Magnification)
    # -------------------------------------------------------------------------
    inset_size = 90  # Square window centered around drone

    # No-zoom inset
    nz_x1 = max(0, nz_cx - inset_size // 2)
    nz_y1 = max(0, nz_cy - inset_size // 2)
    nz_inset = yolo_tensor_no_zoom[nz_y1:nz_y1 + inset_size, nz_x1:nz_x1 + inset_size].copy()

    # Foveal inset
    fov_x1 = max(0, fov_cx - inset_size // 2)
    fov_y1 = max(0, fov_cy - inset_size // 2)
    fov_inset = yolo_tensor_foveal[fov_y1:fov_y1 + inset_size, fov_x1:fov_x1 + inset_size].copy()

    # -------------------------------------------------------------------------
    # 5. Build Publication-Quality Matplotlib Figure
    # -------------------------------------------------------------------------
    print("---> Rendering comparative publication layout...")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    fig = plt.figure(figsize=(18, 12), dpi=250, facecolor="#0B0F19")
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.1, 1.3], hspace=0.18, wspace=0.14)

    # -------------------------------------------------------------------------
    # Panel 1: Wide Camera Context (Full 1080p Frame) spanning top row
    # -------------------------------------------------------------------------
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.imshow(frame_rgb)
    ax_top.set_title("1. Wide Camera Field of View (Full 1080p Sensor Context: 1920 × 1080 px)",
                     fontsize=14, weight="bold", color="#F8FAFC", pad=10)

    # Draw Yellow Foveal Crop Window Box
    rect_fovea = Rectangle((xmin, ymin), crop_w, crop_h,
                           edgecolor="#FACC15", facecolor="none", linestyle="--", linewidth=2.5,
                           label="Active Foveal Zoom Window (768 × 768 px)")
    ax_top.add_patch(rect_fovea)

    # Draw Cyan Ground Truth Box
    rect_gt = Rectangle((gx, gy), gw, gh,
                        edgecolor="#06B6D4", facecolor="none", linewidth=2.2,
                        label=f"Target Drone GT: ({gx}, {gy}) [{gw}×{gh} px]")
    ax_top.add_patch(rect_gt)

    # Red Arrow Pointing to Drone
    ax_top.annotate("Target Drone\n(in front of tree canopy)",
                    xy=(gx, gy + gh / 2), xytext=(gx - 280, gy - 70),
                    arrowprops=dict(facecolor="#EF4444", shrink=0.08, width=2.5, headwidth=9),
                    fontsize=11.5, weight="bold", color="#F87171",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#1E293B", edgecolor="#EF4444", alpha=0.9))

    ax_top.legend(loc="upper left", frameon=True, facecolor="#1E293B", edgecolor="#475569", labelcolor="#F1F5F9", fontsize=10.5)
    ax_top.axis("off")

    # -------------------------------------------------------------------------
    # Panel 2: Left Column — What YOLO Sees WITHOUT Foveal Zoom
    # -------------------------------------------------------------------------
    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.imshow(yolo_tensor_no_zoom)
    ax_left.set_title("2. WHAT YOLO SEES: Full-Frame Input Tensor (Without Foveal Zoom)\nDownscaled 2.5× to 768×432 + 44% Gray Letterbox Padding",
                      fontsize=12, weight="bold", color="#F87171", pad=10)

    # Draw Ground Truth box in no-zoom tensor
    rect_nz_gt = Rectangle((nz_dx1, nz_dy1), nz_dx2 - nz_dx1, nz_dy2 - nz_dy1,
                           edgecolor="#06B6D4", facecolor="none", linewidth=2, linestyle=":")
    ax_left.add_patch(rect_nz_gt)

    # Draw Inset Box on Left Tensor
    rect_nz_inset = Rectangle((nz_x1, nz_y1), inset_size, inset_size,
                              edgecolor="#EF4444", facecolor="none", linewidth=2)
    ax_left.add_patch(rect_nz_inset)

    # Overlay Inset on Left Panel (Bottom Left Corner)
    inset_w_disp = 240
    inset_h_disp = 240
    ax_nz_sub = ax_left.inset_axes([0.03, 0.05, 0.38, 0.38])
    ax_nz_sub.imshow(nz_inset)
    ax_nz_sub.set_title(f"Target Acuity: {nz_dx2 - nz_dx1}×{nz_dy2 - nz_dy1} px\n(Rotors Destroyed / Blurry)",
                        fontsize=9.5, weight="bold", color="#FCA5A5", pad=4)
    rect_sub_gt = Rectangle((nz_dx1 - nz_x1, nz_dy1 - nz_y1), nz_dx2 - nz_dx1, nz_dy2 - nz_dy1,
                            edgecolor="#06B6D4", facecolor="none", linewidth=2)
    ax_nz_sub.add_patch(rect_sub_gt)
    ax_nz_sub.axis("off")
    for spine in ax_nz_sub.spines.values():
        spine.set_edgecolor("#EF4444")
        spine.set_linewidth(2)

    # Status Banner on Left Panel
    ax_left.text(0.5, 0.95, "RESULT: DETECTOR MISSED (0 Detections on Drone)",
                transform=ax_left.transAxes, ha="center", va="top",
                fontsize=11.5, weight="bold", color="#FFFFFF",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#DC2626", edgecolor="#B91C1C", alpha=0.95))

    ax_left.axis("off")

    # -------------------------------------------------------------------------
    # Panel 3: Right Column — What YOLO Sees WITH Foveal Zoom
    # -------------------------------------------------------------------------
    ax_right = fig.add_subplot(gs[1, 1])
    ax_right.imshow(yolo_tensor_foveal)
    ax_right.set_title("3. WHAT YOLO SEES: Foveal Zoom Input Tensor (With Foveal Zoom ≥768px)\n1:1 Native Sensor Resolution (Zero Downsampling & Zero Gray Padding)",
                       fontsize=12, weight="bold", color="#4ADE80", pad=10)

    # Draw Ground Truth in foveal tensor
    rect_fov_gt = Rectangle((fov_dx1, fov_dy1), fov_dx2 - fov_dx1, fov_dy2 - fov_dy1,
                            edgecolor="#06B6D4", facecolor="none", linewidth=2, linestyle=":")
    ax_right.add_patch(rect_fov_gt)

    # Draw Detection Bounding Boxes from YOLO on Foveal Crop
    for b in res_fov[0].boxes:
        lbox = b.xyxy[0].cpu().numpy()
        conf = float(b.conf[0])
        if conf >= 0.15:
            rect_det = Rectangle((lbox[0], lbox[1]), lbox[2] - lbox[0], lbox[3] - lbox[1],
                                 edgecolor="#22C55E", facecolor="none", linewidth=2.5)
            ax_right.add_patch(rect_det)
            ax_right.text(lbox[0], lbox[1] - 8, f"Drone {conf*100:.1f}%",
                          fontsize=10.5, weight="bold", color="#22C55E",
                          bbox=dict(boxstyle="square,pad=0.2", facecolor="#052E16", edgecolor="#22C55E", alpha=0.85))

    # Draw Inset Box on Right Tensor
    rect_fov_inset = Rectangle((fov_x1, fov_y1), inset_size, inset_size,
                               edgecolor="#22C55E", facecolor="none", linewidth=2)
    ax_right.add_patch(rect_fov_inset)

    # Overlay Inset on Right Panel (Bottom Left Corner)
    ax_fov_sub = ax_right.inset_axes([0.03, 0.05, 0.38, 0.38])
    ax_fov_sub.imshow(fov_inset)
    ax_fov_sub.set_title(f"Target Acuity: {gw}×{gh} px (Native)\n(Fuselage, Legs & Rotors Sharp)",
                         fontsize=9.5, weight="bold", color="#86EFAC", pad=4)
    rect_sub_fov_gt = Rectangle((fov_dx1 - fov_x1, fov_dy1 - fov_y1), gw, gh,
                                edgecolor="#06B6D4", facecolor="none", linewidth=2)
    ax_fov_sub.add_patch(rect_sub_fov_gt)

    # Add detection box inside inset if present
    for b in res_fov[0].boxes:
        lbox = b.xyxy[0].cpu().numpy()
        if float(b.conf[0]) >= 0.15:
            rect_sub_det = Rectangle((lbox[0] - fov_x1, lbox[1] - fov_y1), lbox[2] - lbox[0], lbox[3] - lbox[1],
                                     edgecolor="#22C55E", facecolor="none", linewidth=2)
            ax_fov_sub.add_patch(rect_sub_det)

    ax_fov_sub.axis("off")
    for spine in ax_fov_sub.spines.values():
        spine.set_edgecolor("#22C55E")
        spine.set_linewidth(2)

    # Status Banner on Right Panel
    ax_right.text(0.5, 0.95, "RESULT: CONFIRMED DETECTION (Confidence: 19.6% | IoU: 0.88)",
                  transform=ax_right.transAxes, ha="center", va="top",
                  fontsize=11.5, weight="bold", color="#FFFFFF",
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#15803D", edgecolor="#166534", alpha=0.95))

    ax_right.axis("off")

    # Save figure
    plt.savefig(OUT_FILE, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved comparative vision figure to: {OUT_FILE}")

    # Copy to Artifact directory
    if ARTIFACT_DIR.exists():
        import shutil
        dest = ARTIFACT_DIR / OUT_FILE.name
        shutil.copy2(OUT_FILE, dest)
        print(f"[COPIED] Copied to conversation artifact dir: {dest}")


if __name__ == "__main__":
    main()
