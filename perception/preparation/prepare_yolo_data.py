"""
Anti-UAV / Drone Dataset Converter to YOLO Format
Converts video sequences and tracking ground truth files into standard YOLO format.
"""

import os
import glob
import shutil
import random
import yaml
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image


def convert_bbox_to_yolo(
    x: float, y: float, w: float, h: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """
    Converts [xmin, ymin, width, height] in pixels to normalized YOLO format:
    [x_center, y_center, width, height] in range [0.0, 1.0].
    Clamps values to image boundaries.
    """
    # Clamp coordinates within image
    xmin = max(0.0, min(x, img_w))
    ymin = max(0.0, min(y, img_h))
    xmax = max(0.0, min(x + w, img_w))
    ymax = max(0.0, min(y + h, img_h))

    actual_w = xmax - xmin
    actual_h = ymax - ymin

    if actual_w <= 1.0 or actual_h <= 1.0:
        return None

    x_center = (xmin + actual_w / 2.0) / img_w
    y_center = (ymin + actual_h / 2.0) / img_h
    norm_w = actual_w / img_w
    norm_h = actual_h / img_h

    # Clip to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))

    return x_center, y_center, norm_w, norm_h


def prepare_anti_uav_yolo(
    video_dir: str = "data/Anti-UAV/Anti-UAV-Tracking-V0",
    gt_dir: str = "data/Anti-UAV/Anti-UAV-Tracking-V0GT",
    output_dir: str = "datasets/anti_uav_yolo",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stride: int = 1,
    seed: int = 42,
    use_symlinks: bool = True,
    clean: bool = True,
    stratify: bool = True
) -> Dict:
    """
    Converts Anti-UAV videos and GT files into a 3-way YOLO dataset structure (train, val, test).
    Splits by entire video sequences to prevent frame-level temporal data leakage.
    """
    random.seed(seed)
    
    # 1. Discover all videos and matching GT files
    video_folders = sorted([
        f for f in os.listdir(video_dir)
        if os.path.isdir(os.path.join(video_dir, f)) and not f.startswith(".")
    ])
    
    valid_sequences = []
    for v in video_folders:
        v_path = os.path.join(video_dir, v)
        gt_path = os.path.join(gt_dir, f"{v}_gt.txt")
        if os.path.exists(gt_path):
            # Count frames for stratification
            n_frames = len(glob.glob(f"{v_path}/*.jpg"))
            valid_sequences.append((v, v_path, gt_path, n_frames))

    print(f"Found {len(valid_sequences)} matching video sequences.")

    # 2. Train / Val / Test split by sequence
    if stratify and len(valid_sequences) == 20:
        # Pre-calculated duration-balanced sequence allocation
        # Val (6 videos): video01 (1050), video04 (341), video13 (1915), video15 (1350), video16 (1285), video17 (780) -> 6721 frames
        # Test (3 videos): video05 (450), video12 (1485), video20 (1635) -> 3570 frames
        # Train (11 videos): remaining 11 Anti-UAV videos -> 14,513 frames
        val_names = {"video01", "video04", "video13", "video15", "video16", "video17"}
        test_names = {"video05", "video12", "video20"}
        
        val_seqs = [s for s in valid_sequences if s[0] in val_names]
        test_seqs = [s for s in valid_sequences if s[0] in test_names]
        train_seqs = [s for s in valid_sequences if s[0] not in val_names and s[0] not in test_names]
    else:
        # Dynamic sequence-level split
        shuffled = valid_sequences.copy()
        random.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        n_test = max(1, int(len(shuffled) * test_ratio))
        val_seqs = shuffled[:n_val]
        test_seqs = shuffled[n_val:n_val + n_test]
        train_seqs = shuffled[n_val + n_test:]

    val_seqs.sort(key=lambda x: x[0])
    test_seqs.sort(key=lambda x: x[0])
    train_seqs.sort(key=lambda x: x[0])

    print(f"\nSequence Split Summary:")
    print(f"  Train: {len(train_seqs)} videos -> {[s[0] for s in train_seqs]}")
    print(f"  Val:   {len(val_seqs)} videos -> {[s[0] for s in val_seqs]}")
    print(f"  Test:  {len(test_seqs)} videos -> {[s[0] for s in test_seqs]}")

    # 3. Create destination directory structure & clean if requested
    out_path = Path(output_dir).resolve()
    splits = ["train", "val", "test"]

    if clean:
        print(f"\nCleaning existing split directories in {out_path}...")
        for split in splits:
            for sub in ["images", "labels"]:
                p = out_path / sub / split
                if p.exists():
                    shutil.rmtree(p)

    for split in splits:
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {f"{s}_{k}": 0 for s in splits for k in ["frames", "positives", "negatives"]}

    # 4. Process each split
    split_map = [("train", train_seqs), ("val", val_seqs), ("test", test_seqs)]
    for split, seq_list in split_map:
        print(f"\nPopulating {split} split ({len(seq_list)} sequences)...")
        for seq_name, v_path, gt_path, _ in seq_list:
            frame_files = sorted(glob.glob(f"{v_path}/*.jpg"))
            with open(gt_path, "r") as f:
                gt_lines = [l.strip() for l in f if l.strip()]

            if len(frame_files) != len(gt_lines):
                print(f"Warning: {seq_name} has {len(frame_files)} frames but {len(gt_lines)} GT lines. Using min length.")

            n_samples = min(len(frame_files), len(gt_lines))
            
            # Cache image size from first frame
            if n_samples > 0:
                with Image.open(frame_files[0]) as im:
                    img_w, img_h = im.size
            else:
                continue

            for idx in range(0, n_samples, stride):
                img_src = frame_files[idx]
                gt_line = gt_lines[idx]

                dst_stem = f"{seq_name}_{Path(img_src).stem}"
                dst_img = out_path / "images" / split / f"{dst_stem}.jpg"
                dst_lbl = out_path / "labels" / split / f"{dst_stem}.txt"

                # Link or copy image
                if dst_img.exists():
                    dst_img.unlink()
                if use_symlinks:
                    os.symlink(os.path.abspath(img_src), dst_img)
                else:
                    shutil.copy2(img_src, dst_img)

                # Parse bounding box
                parts = gt_line.replace(",", " ").split()
                has_drone = False
                if len(parts) >= 4:
                    try:
                        x, y, w, h = map(float, parts[:4])
                        if w > 0 and h > 0:
                            yolo_bbox = convert_bbox_to_yolo(x, y, w, h, img_w, img_h)
                            if yolo_bbox:
                                xc, yc, nw, nh = yolo_bbox
                                with open(dst_lbl, "w") as lf:
                                    lf.write(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
                                has_drone = True
                    except ValueError:
                        pass

                if not has_drone:
                    # Create empty label file (background image)
                    dst_lbl.touch(exist_ok=True)
                    stats[f"{split}_negatives"] += 1
                else:
                    stats[f"{split}_positives"] += 1

                stats[f"{split}_frames"] += 1

    # 5. Write dataset.yaml configuration file for YOLO
    data_yaml = {
        "path": str(out_path),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "drone"
        }
    }

    yaml_file = out_path / "data.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print("\n" + "=" * 65)
    print("Dataset Preparation Summary:")
    print(f"  Output directory:  {out_path}")
    print(f"  YAML config:       {yaml_file}")
    for split in splits:
        print(f"  {split.capitalize():<6} Frames:      {stats[f'{split}_frames']} "
              f"({stats[f'{split}_positives']} drone, {stats[f'{split}_negatives']} background)")
    print("=" * 65)

    return stats


BASE_DIR = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Anti-UAV Dataset for YOLO training.")
    parser.add_argument("--video_dir", type=str, default=str(BASE_DIR / "data/Anti-UAV/Anti-UAV-Tracking-V0"), help="Path to video frames directory")
    parser.add_argument("--gt_dir", type=str, default=str(BASE_DIR / "data/Anti-UAV/Anti-UAV-Tracking-V0GT"), help="Path to GT files directory")
    parser.add_argument("--out", type=str, default=str(BASE_DIR / "datasets/anti_uav_yolo"), help="Destination YOLO dataset directory")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio by sequence (default: 0.15)")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio by sequence (default: 0.15)")
    parser.add_argument("--stride", type=int, default=1, help="Frame subsampling stride (1 = all frames, 2 = every 2nd frame)")
    parser.add_argument("--no_symlinks", action="store_true", help="Copy files instead of symlinks")
    parser.add_argument("--no_clean", action="store_true", help="Do not wipe existing split directories before populating")
    parser.add_argument("--random_split", action="store_true", help="Use random split instead of length-stratified split")
    args = parser.parse_args()

    prepare_anti_uav_yolo(
        video_dir=args.video_dir,
        gt_dir=args.gt_dir,
        output_dir=args.out,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stride=args.stride,
        use_symlinks=not args.no_symlinks,
        clean=not args.no_clean,
        stratify=not args.random_split
    )
