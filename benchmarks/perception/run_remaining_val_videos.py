#!/usr/bin/env python3
"""
Evaluate the remaining 3 validation videos (video01, video16, video17: 3,115 frames)
and merge with existing video04, video13, video15 results to produce the complete
6-video validation benchmark (6,721 frames) on 720p camera capture.
"""

import sys
import time
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.perception.benchmark_foveal_zoom_ablation import evaluate_video, load_gt_boxes_720p, MODEL_WEIGHTS

OUT_DIR = REPO_ROOT / "outputs" / "benchmarks" / "perception"
EXISTING_JSON = OUT_DIR / "foveal_zoom_1x_2x_3x_val_benchmark.json"
FULL_JSON = OUT_DIR / "foveal_zoom_1x_2x_3x_val_6videos_benchmark.json"
FULL_FIG = OUT_DIR / "foveal_zoom_1x_2x_3x_val_6videos_comparison.png"
ARTIFACT_DIR = Path("/Users/philipkierkegaard/.gemini/antigravity-ide/brain/5967d853-7a7e-4cdd-8792-4a77e6d704a2")

REMAINING_VIDEOS = ["video01", "video16", "video17"]

configs = {
    "No Foveal Zoom (1x)": {
        "weights": str(MODEL_WEIGHTS),
        "conf_threshold": 0.25,
        "enable_kalman": True,
        "enable_dynamic_zoom": False,
        "imgsz": 768
    },
    "Foveal Zoom (2x Max)": {
        "weights": str(MODEL_WEIGHTS),
        "conf_threshold": 0.25,
        "enable_kalman": True,
        "enable_dynamic_zoom": True,
        "max_zoom": 2.0,
        "min_crop_size": None,
        "desired_target_size": 110.0,
        "zoom_lost_threshold": 3,
        "imgsz": 768
    },
    "Foveal Zoom (3x Max)": {
        "weights": str(MODEL_WEIGHTS),
        "conf_threshold": 0.25,
        "enable_kalman": True,
        "enable_dynamic_zoom": True,
        "max_zoom": 3.0,
        "min_crop_size": None,
        "desired_target_size": 110.0,
        "zoom_lost_threshold": 3,
        "imgsz": 768
    }
}


def main():
    print("=" * 80)
    print("EVALUATING REMAINING VALIDATION VIDEOS (video01, video16, video17)")
    print("=" * 80)

    # Load existing 3-video results
    with open(EXISTING_JSON, "r") as f:
        existing_data = json.load(f)

    per_video_results = list(existing_data["per_video_results"])

    for vid in REMAINING_VIDEOS:
        for cfg_name, kwargs in configs.items():
            print(f"\n---> Evaluating {vid} under [{cfg_name}]...")
            res = evaluate_video(vid, cfg_name, kwargs)
            print(f"     Locked: {res['lock_rate_pct']}% | CLE: {res['mean_cle_px']}px | P@20px: {res['precision_20px_pct']}% | Catch (IoU>0.3): {res['success_iou30_pct']}% | FPS: {res['fps']}")
            per_video_results.append(res)

    all_vids = ["video01", "video04", "video13", "video15", "video16", "video17"]
    summary_by_config = {cfg: [r for r in per_video_results if r["config"] == cfg] for cfg in configs}

    aggregate = {}
    for cfg_name, res_list in summary_by_config.items():
        tot_vis = sum(r["visible_frames"] for r in res_list)
        tot_frames = sum(r["total_frames"] for r in res_list)
        w_p20 = sum(r["precision_20px_pct"] * r["visible_frames"] for r in res_list) / max(1, tot_vis)
        w_succ30 = sum(r["success_iou30_pct"] * r["visible_frames"] for r in res_list) / max(1, tot_vis)
        w_succ50 = sum(r["success_iou50_pct"] * r["visible_frames"] for r in res_list) / max(1, tot_vis)
        w_cle = sum(r["mean_cle_px"] * r["visible_frames"] for r in res_list) / max(1, tot_vis)
        w_iou = sum(r["mean_iou"] * r["visible_frames"] for r in res_list) / max(1, tot_vis)
        w_lock = sum(r["lock_rate_pct"] * r["total_frames"] for r in res_list) / max(1, tot_frames)
        mean_fps = np.mean([r["fps"] for r in res_list])
        mean_zoom = np.mean([r["mean_zoom"] for r in res_list])

        aggregate[cfg_name] = {
            "total_frames": tot_frames,
            "visible_frames": tot_vis,
            "mean_cle_px": round(w_cle, 2),
            "mean_iou": round(w_iou, 3),
            "precision_20px_pct": round(w_p20, 2),
            "success_iou30_pct": round(w_succ30, 2),
            "success_iou50_pct": round(w_succ50, 2),
            "lock_rate_pct": round(w_lock, 2),
            "mean_zoom_level": round(mean_zoom, 2),
            "fps": round(float(mean_fps), 1)
        }

    output_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sensor_resolution": "1280x720",
        "validation_videos": all_vids,
        "aggregate": aggregate,
        "per_video_results": per_video_results
    }

    with open(FULL_JSON, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[SAVED] Full 6-video benchmark saved to: {FULL_JSON}")

    # Generate Chart
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), dpi=220, facecolor="#0B0F19")
    fig.suptitle(
        "Foveal Zoom Ablation on 720p Camera Sensor (1280×720 Capture)\n"
        "Full Anti-UAV Validation Suite (All 6 Videos: video01, 04, 13, 15, 16, 17 — 6,721 Frames)",
        fontsize=14, weight="bold", color="#F8FAFC", y=1.03
    )

    cfg_labels = ["No Zoom\n(1x Full-Frame)", "Foveal Zoom\n(2x Max)", "Foveal Zoom\n(3x Max)"]
    colors = ["#EF4444", "#38BDF8", "#10B981"]

    p20_vals = [aggregate[c]["precision_20px_pct"] for c in configs]
    succ30_vals = [aggregate[c]["success_iou30_pct"] for c in configs]
    lock_vals = [aggregate[c]["lock_rate_pct"] for c in configs]
    cle_vals = [aggregate[c]["mean_cle_px"] for c in configs]

    # Plot 1: P@20px
    ax1 = axes[0]
    ax1.set_facecolor("#111827")
    b1 = ax1.bar(cfg_labels, p20_vals, color=colors, width=0.55, edgecolor="#374151", linewidth=1.5)
    ax1.set_title("Tracking Precision @ 20px\n(CLE ≤ 20 px)", fontsize=11.5, weight="bold", color="#E2E8F0", pad=10)
    ax1.set_ylabel("Precision (%)", fontsize=10.5, color="#94A3B8")
    ax1.set_ylim(80, 102)
    ax1.grid(axis="y", linestyle="--", alpha=0.3, color="#475569")
    for b in b1:
        y = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2.0, y + 0.6, f"{y:.1f}%", ha="center", va="bottom", fontsize=11, weight="bold", color="#F8FAFC")

    # Plot 2: Catch Rate
    ax2 = axes[1]
    ax2.set_facecolor("#111827")
    b2 = ax2.bar(cfg_labels, succ30_vals, color=colors, width=0.55, edgecolor="#374151", linewidth=1.5)
    ax2.set_title("Tracking Acquisition Catch Rate\n(IoU ≥ 0.30)", fontsize=11.5, weight="bold", color="#E2E8F0", pad=10)
    ax2.set_ylabel("Catch Rate (%)", fontsize=10.5, color="#94A3B8")
    ax2.set_ylim(80, 102)
    ax2.grid(axis="y", linestyle="--", alpha=0.3, color="#475569")
    for b in b2:
        y = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2.0, y + 0.6, f"{y:.1f}%", ha="center", va="bottom", fontsize=11, weight="bold", color="#F8FAFC")

    # Plot 3: Lock Rate
    ax3 = axes[2]
    ax3.set_facecolor("#111827")
    b3 = ax3.bar(cfg_labels, lock_vals, color=colors, width=0.55, edgecolor="#374151", linewidth=1.5)
    ax3.set_title("Active Track Lock Rate\n(Continuous Locked State)", fontsize=11.5, weight="bold", color="#E2E8F0", pad=10)
    ax3.set_ylabel("Lock Rate (%)", fontsize=10.5, color="#94A3B8")
    ax3.set_ylim(70, 100)
    ax3.grid(axis="y", linestyle="--", alpha=0.3, color="#475569")
    for b in b3:
        y = b.get_height()
        ax3.text(b.get_x() + b.get_width() / 2.0, y + 0.8, f"{y:.1f}%", ha="center", va="bottom", fontsize=11, weight="bold", color="#F8FAFC")

    # Plot 4: CLE
    ax4 = axes[3]
    ax4.set_facecolor("#111827")
    b4 = ax4.bar(cfg_labels, cle_vals, color=colors, width=0.55, edgecolor="#374151", linewidth=1.5)
    ax4.set_title("Mean Center Location Error\n(Lower is Better)", fontsize=11.5, weight="bold", color="#E2E8F0", pad=10)
    ax4.set_ylabel("Mean Error (px)", fontsize=10.5, color="#94A3B8")
    ax4.set_ylim(0, max(cle_vals) * 1.3)
    ax4.grid(axis="y", linestyle="--", alpha=0.3, color="#475569")
    for b in b4:
        y = b.get_height()
        ax4.text(b.get_x() + b.get_width() / 2.0, y + 0.15, f"{y:.2f} px", ha="center", va="bottom", fontsize=11, weight="bold", color="#F8FAFC")

    for ax in axes:
        ax.tick_params(colors="#94A3B8", labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor("#374151")

    plt.tight_layout()
    plt.savefig(FULL_FIG, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Comparison plot saved to: {FULL_FIG}")

    if ARTIFACT_DIR.exists():
        shutil.copy2(FULL_JSON, ARTIFACT_DIR / FULL_JSON.name)
        shutil.copy2(FULL_FIG, ARTIFACT_DIR / FULL_FIG.name)


if __name__ == "__main__":
    main()
