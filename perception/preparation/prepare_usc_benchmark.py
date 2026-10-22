#!/usr/bin/env python3
"""
USC Drone Tracking Benchmark Preparation & Ingestion.
Creates a curated, continuous 30 Hz visual tracking validation suite from raw USC footage.
Splits environment-shifting sequences, trims impossible blackouts, re-indexes frames smoothly
via symlinks (zero disk waste), and interpolates ground-truth bounding boxes to dense 30 Hz.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_USC_DIR = PROJECT_ROOT / "perception" / "raw_data" / "usc%20drone-001"
OUTPUT_DIR = PROJECT_ROOT / "perception" / "datasets" / "usc_drone_tracking_benchmark"


def interpolate_boxes(gt_boxes: list, total_frames: int, start_frame: int, end_frame: int) -> list:
    """
    Interpolates 15-frame keyframes to 30 Hz dense ground-truth bounding boxes
    for the sub-slice [start_frame, end_frame].
    """
    dense_boxes = {}
    
    # 1. Map existing keyframes (1, 16, 31, ...)
    kf_map = {}
    for i, box in enumerate(gt_boxes):
        f_idx = 1 + i * 15
        kf_map[f_idx] = np.array(box, dtype=np.float64)
        
    kf_indices = sorted(kf_map.keys())
    
    # 2. Dense interpolation for all frames from 1 to total_frames
    for i in range(len(kf_indices) - 1):
        f1 = kf_indices[i]
        f2 = kf_indices[i + 1]
        b1 = kf_map[f1]
        b2 = kf_map[f2]
        
        is_valid_1 = (b1[2] > 0 and b1[3] > 0)
        is_valid_2 = (b2[2] > 0 and b2[3] > 0)
        
        for f in range(f1, f2):
            if is_valid_1 and is_valid_2:
                alpha = (f - f1) / float(f2 - f1)
                dense_boxes[f] = b1 + alpha * (b2 - b1)
            elif is_valid_1 and not is_valid_2:
                dense_boxes[f] = b1 if (f - f1) < (f2 - f1) / 2 else np.zeros(4)
            elif not is_valid_1 and is_valid_2:
                dense_boxes[f] = b2 if (f - f1) >= (f2 - f1) / 2 else np.zeros(4)
            else:
                dense_boxes[f] = np.zeros(4)
                
    # End frame boundary
    last_kf = kf_indices[-1]
    last_box = kf_map[last_kf]
    for f in range(last_kf, total_frames + 1):
        dense_boxes[f] = last_box
        
    # 3. Slice [start_frame, end_frame]
    sliced_boxes = []
    for f in range(start_frame, end_frame + 1):
        box = dense_boxes.get(f, np.zeros(4))
        sliced_boxes.append(box)
        
    return sliced_boxes


def main():
    print("=" * 80)
    print("  PREPARING USC DRONE TRACKING & NOISE BENCHMARK (30 Hz)")
    print("=" * 80)
    
    if not RAW_USC_DIR.exists():
        print(f"[ERROR] Raw USC directory not found at: {RAW_USC_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sequence slice definitions
    # (name, raw_folder, start_frame, end_frame, notes)
    sequences = [
        (
            "Drone_garden_part1",
            "Drone_garden",
            1, 1200,
            "Low-to-mid altitude flight in garden/park weaving past trees and stadium structures."
        ),
        (
            "Drone_garden_part2",
            "Drone_garden",
            1750, 2210,
            "High-speed close-in fly-by diving sharply into dense tree canopy foliage."
        ),
        (
            "Drone9_part1",
            "Drone9",
            1, 1200,
            "High-altitude open pursuit against sky and tree crowns (target scale: 45-108 px)."
        ),
        (
            "Drone9_part2",
            "Drone9",
            2200, 2900,
            "Tree-line boundary tracking with intermittent canopy edge occlusions (scale: 35-55 px)."
        ),
        (
            "Drone8",
            "Drone8",
            1, 3200,
            "Full 3,200-frame mission: courtyard launch -> high sky cruising -> building facade descent."
        ),
        (
            "Drone_frames",
            "Drone_frames",
            1, 421,
            "Close-range high-resolution quadrotor inspection against dark tree leaves."
        ),
        (
            "Drone_frames_1_canopy",
            "Drone_frames_1",
            550, 1600,
            "Mid-range tree canopy pursuit (discards frames 1-450 sun flare & 1600+ defocus blur)."
        ),
        (
            "Drone_test1",
            "Drone_test1",
            1, 1479,
            "Open test field with dynamic approach and departure velocity sweeps."
        ),
        (
            "Drone_test_airborne",
            "Drone_test",
            1, 750,
            "Airborne tree canopy and bare branch crossings (discards ground parking lot segment)."
        ),
    ]

    manifest = {"sequences": {}, "metadata": {}}
    total_benchmark_frames = 0

    for out_name, raw_folder, start_f, end_f, notes in sequences:
        raw_seq_dir = RAW_USC_DIR / raw_folder / "img"
        gt_file = raw_seq_dir / "groundtruth_rect.txt"
        
        with open(gt_file) as f:
            raw_gt_boxes = [list(map(float, line.strip().split(","))) for line in f if line.strip()]
            
        all_raw_imgs = sorted([f for f in os.listdir(raw_seq_dir) if f.lower().endswith((".jpg", ".png"))])
        n_raw = len(all_raw_imgs)
        
        end_f = min(end_f, n_raw)
        n_slice = end_f - start_f + 1
        
        print(f"\nProcessing [{out_name}]:")
        print(f"  Source:       {raw_folder} (Frames {start_f} to {end_f})")
        print(f"  Frame Count:  {n_slice} frames (~{n_slice/30:.1f}s at 30 fps)")
        print(f"  Context:      {notes}")
        
        out_seq_dir = OUTPUT_DIR / out_name
        out_img_dir = out_seq_dir / "img"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        
        # Dense 30 Hz ground-truth interpolation
        dense_gt = interpolate_boxes(raw_gt_boxes, n_raw, start_f, end_f)
        
        # Write dense 30 Hz groundtruth_rect.txt
        gt_out_path = out_seq_dir / "groundtruth_rect.txt"
        with open(gt_out_path, "w", encoding="utf-8") as f:
            for b in dense_gt:
                f.write(f"{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f}\n")
                
        # Create continuous 1-indexed symlinks (0001.jpg, 0002.jpg, ...)
        for new_idx, raw_f_idx in enumerate(range(start_f, end_f + 1), 1):
            src_file = raw_seq_dir / f"{raw_f_idx:04d}.jpg"
            dst_file = out_img_dir / f"{new_idx:04d}.jpg"
            if dst_file.exists() or dst_file.is_symlink():
                dst_file.unlink()
            # Relative symlink for portability
            rel_src = os.path.relpath(src_file, out_img_dir)
            dst_file.symlink_to(rel_src)
            
        manifest["sequences"][out_name] = {
            "source_sequence": raw_folder,
            "source_start_frame": start_f,
            "source_end_frame": end_f,
            "frame_count": n_slice,
            "duration_s": round(n_slice / 30.0, 2),
            "fps": 30.0,
            "ground_truth_file": str(gt_out_path.relative_to(PROJECT_ROOT)),
            "image_dir": str(out_img_dir.relative_to(PROJECT_ROOT)),
            "description": notes,
        }
        total_benchmark_frames += n_slice

    manifest["metadata"] = {
        "dataset_name": "USC Drone Tracking & Noise Calibration Benchmark",
        "total_sequences": len(sequences),
        "total_frames": total_benchmark_frames,
        "total_duration_minutes": round(total_benchmark_frames / (30.0 * 60.0), 2),
        "target_type": "DJI Phantom Quadrotor",
        "intended_use": "Kalman Filter Tuning & Empirical YOLO Noise Modeling (Validation / Test Split)",
    }

    manifest_path = OUTPUT_DIR / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[DONE] Successfully prepared {len(sequences)} benchmark sequences ({total_benchmark_frames:,} total frames)!")
    print(f"[SAVED] Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
