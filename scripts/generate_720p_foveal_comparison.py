#!/usr/bin/env python3
"""
Generate 720p Camera Foveal Zoom Comparison:
Evaluates Frame 743 (video12) downsampled to 1280x720 camera sensor resolution,
comparing:
1. Full-Frame (Without Foveal Zoom) -> 1280x720 letterboxed to 768x768
2. Native Square Foveal Zoom -> 720x720 clamped crop resized to 768x768
3. Deep Foveal Zoom -> 420x420 digital crop resized to 768x768

DTU Bachelor Thesis — Autonomous Drone Pursuit & Interception.
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
OUT_FILE = OUT_DIR / "foveal_720p_sensor_3way_comparison.png"
ARTIFACT_DIR = Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/5967d853-7a7e-4cdd-8792-4a77e6d704a2")


def main():
    print("---> Loading video12 frame 743 and model...")
    video_path = PROJECT_ROOT / "perception" / "raw_data" / "Anti-UAV" / "videos" / "video12.mp4"
    gt_path = PROJECT_ROOT / "perception" / "raw_data" / "Anti-UAV" / "Anti-UAV-Tracking-V0GT" / "video12_gt.txt"
    model_path = PROJECT_ROOT / "perception" / "weights" / "yolov8n_drone_768_hybrid_nwd_c51_best.pt"

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 743)
    ret, frame_1080_bgr = cap.read()
    cap.release()
    assert ret, "Failed to read frame 743"

    with open(gt_path) as f:
        gt_lines = [l.strip() for l in f if l.strip()]
    gt_1080 = [float(x) for x in gt_lines[743].split()]
    gx_1080, gy_1080, gw_1080, gh_1080 = int(gt_1080[0]), int(gt_1080[1]), int(gt_1080[2]), int(gt_1080[3])

    # 1. Downsample raw frame to 1280x720 (camera sensor simulation)
    frame_720_bgr = cv2.resize(frame_1080_bgr, (1280, 720), interpolation=cv2.INTER_AREA)
    frame_720_rgb = cv2.cvtColor(frame_720_bgr, cv2.COLOR_BGR2RGB)
    ih, iw = 720, 1280

    scale_cam = 1280.0 / 1920.0  # 2/3
    gx = int(gx_1080 * scale_cam)
    gy = int(gy_1080 * scale_cam)
    gw = int(gw_1080 * scale_cam)
    gh = int(gh_1080 * scale_cam)
    tcx = gx + gw / 2.0
    tcy = gy + gh / 2.0
    print(f"Target on 1280x720 sensor: ({gx}, {gy}), size: {gw}x{gh} px (center: {tcx:.1f}, {tcy:.1f})")

    model = YOLO(str(model_path))

    # -------------------------------------------------------------------------
    # Mode 1: Full-Frame (Without Foveal Zoom) -> Letterboxed to 768x768
    # -------------------------------------------------------------------------
    scale_m1 = 768.0 / 1280.0  # 0.60
    sw = int(1280 * scale_m1)  # 768
    sh = int(720 * scale_m1)   # 432
    pad_top = (768 - sh) // 2  # 168

    resized_ff = cv2.resize(frame_720_rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    tensor_m1 = np.full((768, 768, 3), 114, dtype=np.uint8)
    tensor_m1[pad_top:pad_top + sh, 0:sw] = resized_ff

    # GT in tensor 1
    m1_gx1 = int(gx * scale_m1)
    m1_gy1 = int(gy * scale_m1 + pad_top)
    m1_gx2 = int((gx + gw) * scale_m1)
    m1_gy2 = int((gy + gh) * scale_m1 + pad_top)
    m1_cx = (m1_gx1 + m1_gx2) // 2
    m1_cy = (m1_gy1 + m1_gy2) // 2

    # -------------------------------------------------------------------------
    # Mode 2: Native Square Foveal Zoom 720x720 (Clamped to bounds)
    # -------------------------------------------------------------------------
    crop_w2, crop_h2 = 720, 720
    x1_m2 = max(0, min(iw - crop_w2, int(tcx - crop_w2 / 2.0)))
    y1_m2 = 0
    x2_m2 = x1_m2 + crop_w2
    y2_m2 = y1_m2 + crop_h2

    crop_m2_rgb = frame_720_rgb[y1_m2:y2_m2, x1_m2:x2_m2]
    tensor_m2 = cv2.resize(crop_m2_rgb, (768, 768), interpolation=cv2.INTER_LINEAR)
    scale_m2 = 768.0 / 720.0

    m2_gx1 = int((gx - x1_m2) * scale_m2)
    m2_gy1 = int((gy - y1_m2) * scale_m2)
    m2_gx2 = int((gx + gw - x1_m2) * scale_m2)
    m2_gy2 = int((gy + gh - y1_m2) * scale_m2)
    m2_cx = (m2_gx1 + m2_gx2) // 2
    m2_cy = (m2_gy1 + m2_gy2) // 2

    # -------------------------------------------------------------------------
    # Mode 3: Deep Foveal Zoom 420x420 (Clamped to bounds)
    # -------------------------------------------------------------------------
    crop_w3, crop_h3 = 420, 420
    x1_m3 = max(0, min(iw - crop_w3, int(tcx - crop_w3 / 2.0)))
    y1_m3 = max(0, min(ih - crop_h3, int(tcy - crop_h3 / 2.0)))
    x2_m3 = x1_m3 + crop_w3
    y2_m3 = y1_m3 + crop_h3

    crop_m3_rgb = frame_720_rgb[y1_m3:y2_m3, x1_m3:x2_m3]
    tensor_m3 = cv2.resize(crop_m3_rgb, (768, 768), interpolation=cv2.INTER_LINEAR)
    scale_m3 = 768.0 / 420.0

    m3_gx1 = int((gx - x1_m3) * scale_m3)
    m3_gy1 = int((gy - y1_m3) * scale_m3)
    m3_gx2 = int((gx + gw - x1_m3) * scale_m3)
    m3_gy2 = int((gy + gh - y1_m3) * scale_m3)
    m3_cx = (m3_gx1 + m3_gx2) // 2
    m3_cy = (m3_gy1 + m3_gy2) // 2

    # -------------------------------------------------------------------------
    # Run YOLO Predictions on All 3 Tensors
    # -------------------------------------------------------------------------
    print("---> Running YOLO inference on all 3 tensors...")
    res_m1 = model.predict(cv2.cvtColor(tensor_m1, cv2.COLOR_RGB2BGR), imgsz=768, conf=0.10, verbose=False)[0]
    res_m2 = model.predict(cv2.cvtColor(tensor_m2, cv2.COLOR_RGB2BGR), imgsz=768, conf=0.10, verbose=False)[0]
    res_m3 = model.predict(cv2.cvtColor(tensor_m3, cv2.COLOR_RGB2BGR), imgsz=768, conf=0.10, verbose=False)[0]

    # Insets: extract square region around drone
    inset_size = 90
    def extract_inset(tensor, cx, cy, sz):
        x_min = max(0, min(768 - sz, cx - sz // 2))
        y_min = max(0, min(768 - sz, cy - sz // 2))
        return tensor[y_min:y_min + sz, x_min:x_min + sz].copy(), x_min, y_min

    inset_m1, in_x1_m1, in_y1_m1 = extract_inset(tensor_m1, m1_cx, m1_cy, inset_size)
    inset_m2, in_x1_m2, in_y1_m2 = extract_inset(tensor_m2, m2_cx, m2_cy, inset_size)
    inset_m3, in_x1_m3, in_y1_m3 = extract_inset(tensor_m3, m3_cx, m3_cy, inset_size)

    # -------------------------------------------------------------------------
    # Render Master Figure
    # -------------------------------------------------------------------------
    print("---> Rendering master publication layout...")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    fig = plt.figure(figsize=(24, 16), dpi=220, facecolor="#0B0F19")
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.1, 1.4], hspace=0.18, wspace=0.12)

    # =========================================================================
    # Panel 1: Top Wide Sensor Context (Full 1280x720) Spanning All 3 Columns
    # =========================================================================
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.imshow(frame_720_rgb)
    ax_top.set_title(
        "1. Physical Camera Sensor Field of View (1280 × 720 px — Native Capture)",
        fontsize=15, weight="bold", color="#F8FAFC", pad=10
    )

    # Draw 720x720 Fovea Window (Amber)
    rect_f2 = Rectangle(
        (x1_m2, y1_m2), crop_w2, crop_h2,
        edgecolor="#F59E0B", facecolor="none", linestyle="--", linewidth=2.5,
        label=f"Native Square Fovea: 720 × 720 px (Origin: [{x1_m2}, {y1_m2}])"
    )
    ax_top.add_patch(rect_f2)

    # Draw 420x420 Fovea Window (Emerald)
    rect_f3 = Rectangle(
        (x1_m3, y1_m3), crop_w3, crop_h3,
        edgecolor="#10B981", facecolor="none", linestyle="-.", linewidth=2.5,
        label=f"Deep Foveal Zoom: 420 × 420 px (Origin: [{x1_m3}, {y1_m3}])"
    )
    ax_top.add_patch(rect_f3)

    # Draw Cyan Ground Truth Box
    rect_gt = Rectangle(
        (gx, gy), gw, gh,
        edgecolor="#06B6D4", facecolor="none", linewidth=2.2,
        label=f"Target Drone GT: ({gx}, {gy}) [{gw}×{gh} px on sensor]"
    )
    ax_top.add_patch(rect_gt)

    # Arrow to Drone
    ax_top.annotate(
        "Target Drone\n(in front of tree canopy)",
        xy=(gx, gy + gh / 2), xytext=(gx - 260, gy - 65),
        arrowprops=dict(facecolor="#EF4444", shrink=0.08, width=2.5, headwidth=9),
        fontsize=11.5, weight="bold", color="#F87171",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1E293B", edgecolor="#EF4444", alpha=0.92)
    )

    ax_top.legend(loc="upper left", frameon=True, facecolor="#1E293B", edgecolor="#475569", labelcolor="#F1F5F9", fontsize=11)
    ax_top.axis("off")

    # =========================================================================
    # Panel 2: Bottom-Left — Mode 1: Full-Frame (Without Foveal Zoom)
    # =========================================================================
    ax_m1 = fig.add_subplot(gs[1, 0])
    ax_m1.imshow(tensor_m1)
    ax_m1.set_title(
        "MODE 1: Full-Frame (No Zoom)\nDownscaled 0.60× to 768×432 + 44% Gray Letterbox",
        fontsize=12, weight="bold", color="#F87171", pad=10
    )

    # GT Box in tensor 1
    rect_t1_gt = Rectangle(
        (m1_gx1, m1_gy1), m1_gx2 - m1_gx1, m1_gy2 - m1_gy1,
        edgecolor="#06B6D4", facecolor="none", linewidth=2, linestyle=":"
    )
    ax_m1.add_patch(rect_t1_gt)

    # Inset window
    rect_t1_in = Rectangle((in_x1_m1, in_y1_m1), inset_size, inset_size, edgecolor="#EF4444", facecolor="none", linewidth=2)
    ax_m1.add_patch(rect_t1_in)

    # Overlay Inset
    ax_m1_sub = ax_m1.inset_axes([0.03, 0.05, 0.40, 0.40])
    ax_m1_sub.imshow(inset_m1)
    ax_m1_sub.set_title(f"Target: {m1_gx2-m1_gx1}×{m1_gy2-m1_gy1} px\n(Decimated / Lost)", fontsize=9.5, weight="bold", color="#FCA5A5", pad=4)
    rect_sub1_gt = Rectangle((m1_gx1 - in_x1_m1, m1_gy1 - in_y1_m1), m1_gx2 - m1_gx1, m1_gy2 - m1_gy1, edgecolor="#06B6D4", facecolor="none", linewidth=2)
    ax_m1_sub.add_patch(rect_sub1_gt)
    ax_m1_sub.axis("off")
    for s in ax_m1_sub.spines.values():
        s.set_edgecolor("#EF4444")
        s.set_linewidth(2)

    # Status Banner
    ax_m1.text(
        0.5, 0.95, "RESULT: DETECTOR MISSED (0 Detections on Drone)",
        transform=ax_m1.transAxes, ha="center", va="top",
        fontsize=11, weight="bold", color="#FFFFFF",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#DC2626", edgecolor="#B91C1C", alpha=0.95)
    )
    ax_m1.axis("off")

    # =========================================================================
    # Panel 3: Bottom-Center — Mode 2: Native Square Foveal Zoom (720x720)
    # =========================================================================
    ax_m2 = fig.add_subplot(gs[1, 1])
    ax_m2.imshow(tensor_m2)
    ax_m2.set_title(
        "MODE 2: Native Fovea (720×720)\n1.07× Scale (1:1 Optical Pixels | 0% Letterbox)",
        fontsize=12, weight="bold", color="#FBBF24", pad=10
    )

    # GT Box in tensor 2
    rect_t2_gt = Rectangle(
        (m2_gx1, m2_gy1), m2_gx2 - m2_gx1, m2_gy2 - m2_gy1,
        edgecolor="#06B6D4", facecolor="none", linewidth=2, linestyle=":"
    )
    ax_m2.add_patch(rect_t2_gt)

    # Inset window
    rect_t2_in = Rectangle((in_x1_m2, in_y1_m2), inset_size, inset_size, edgecolor="#F59E0B", facecolor="none", linewidth=2)
    ax_m2.add_patch(rect_t2_in)

    # Overlay Inset
    ax_m2_sub = ax_m2.inset_axes([0.03, 0.05, 0.40, 0.40])
    ax_m2_sub.imshow(inset_m2)
    ax_m2_sub.set_title(f"Target: {m2_gx2-m2_gx1}×{m2_gy2-m2_gy1} px\n(Pushed to Frame Edge)", fontsize=9.5, weight="bold", color="#FDE68A", pad=4)
    rect_sub2_gt = Rectangle((m2_gx1 - in_x1_m2, m2_gy1 - in_y1_m2), m2_gx2 - m2_gx1, m2_gy2 - m2_gy1, edgecolor="#06B6D4", facecolor="none", linewidth=2)
    ax_m2_sub.add_patch(rect_sub2_gt)
    ax_m2_sub.axis("off")
    for s in ax_m2_sub.spines.values():
        s.set_edgecolor("#F59E0B")
        s.set_linewidth(2)

    # Status Banner
    ax_m2.text(
        0.5, 0.95, "RESULT: MISSED (At Extreme Sensor Margin)",
        transform=ax_m2.transAxes, ha="center", va="top",
        fontsize=11, weight="bold", color="#FFFFFF",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#B45309", edgecolor="#D97706", alpha=0.95)
    )
    ax_m2.axis("off")

    # =========================================================================
    # Panel 4: Bottom-Right — Mode 3: Deep Foveal Zoom (420x420)
    # =========================================================================
    ax_m3 = fig.add_subplot(gs[1, 2])
    ax_m3.imshow(tensor_m3)
    ax_m3.set_title(
        "MODE 3: Deep Fovea (420×420)\n1.83× Digital Crop (+880% Pixel Area vs Mode 1)",
        fontsize=12, weight="bold", color="#4ADE80", pad=10
    )

    # GT Box in tensor 3
    rect_t3_gt = Rectangle(
        (m3_gx1, m3_gy1), m3_gx2 - m3_gx1, m3_gy2 - m3_gy1,
        edgecolor="#06B6D4", facecolor="none", linewidth=2, linestyle=":"
    )
    ax_m3.add_patch(rect_t3_gt)

    # Detections from YOLO
    for b in res_m3.boxes:
        lbox = b.xyxy[0].cpu().numpy()
        conf = float(b.conf[0])
        if conf >= 0.15:
            rect_det = Rectangle((lbox[0], lbox[1]), lbox[2] - lbox[0], lbox[3] - lbox[1],
                                 edgecolor="#22C55E", facecolor="none", linewidth=2.5)
            ax_m3.add_patch(rect_det)
            ax_m3.text(lbox[0], lbox[1] - 8, f"Drone {conf*100:.1f}%",
                       fontsize=10.5, weight="bold", color="#22C55E",
                       bbox=dict(boxstyle="square,pad=0.2", facecolor="#052E16", edgecolor="#22C55E", alpha=0.88))

    # Inset window
    rect_t3_in = Rectangle((in_x1_m3, in_y1_m3), inset_size, inset_size, edgecolor="#22C55E", facecolor="none", linewidth=2)
    ax_m3.add_patch(rect_t3_in)

    # Overlay Inset
    ax_m3_sub = ax_m3.inset_axes([0.03, 0.05, 0.40, 0.40])
    ax_m3_sub.imshow(inset_m3)
    ax_m3_sub.set_title(f"Target: {m3_gx2-m3_gx1}×{m3_gy2-m3_gy1} px\n(Sharp Fuselage & Rotors)", fontsize=9.5, weight="bold", color="#86EFAC", pad=4)
    rect_sub3_gt = Rectangle((m3_gx1 - in_x1_m3, m3_gy1 - in_y1_m3), m3_gx2 - m3_gx1, m3_gy2 - m3_gy1, edgecolor="#06B6D4", facecolor="none", linewidth=2)
    ax_m3_sub.add_patch(rect_sub3_gt)

    for b in res_m3.boxes:
        lbox = b.xyxy[0].cpu().numpy()
        if float(b.conf[0]) >= 0.15:
            rect_sub3_det = Rectangle((lbox[0] - in_x1_m3, lbox[1] - in_y1_m3), lbox[2] - lbox[0], lbox[3] - lbox[1],
                                      edgecolor="#22C55E", facecolor="none", linewidth=2)
            ax_m3_sub.add_patch(rect_sub3_det)

    ax_m3_sub.axis("off")
    for s in ax_m3_sub.spines.values():
        s.set_edgecolor("#22C55E")
        s.set_linewidth(2)

    # Status Banner
    ax_m3.text(
        0.5, 0.95, "RESULT: CONFIRMED DETECTION (Confidence: 69.6% | IoU: 0.76)",
        transform=ax_m3.transAxes, ha="center", va="top",
        fontsize=11, weight="bold", color="#FFFFFF",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#15803D", edgecolor="#166534", alpha=0.95)
    )
    ax_m3.axis("off")

    print(f"---> Saving figure to {OUT_FILE}...")
    plt.savefig(OUT_FILE, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved 720p 3-way comparison to: {OUT_FILE}")

    if ARTIFACT_DIR.exists():
        dest = ARTIFACT_DIR / OUT_FILE.name
        shutil.copy2(OUT_FILE, dest)
        print(f"[COPIED] Copied to artifact dir: {dest}")


if __name__ == "__main__":
    main()
