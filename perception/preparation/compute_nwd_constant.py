"""
Compute Normalized Wasserstein Distance (NWD) Scale Constant C for YOLO Drone Detection.

Formula:
    C = sqrt( (1 / N) * \sum_{i=1}^N (w_i * h_i) )

Where:
    - N is the total number of ground-truth bounding box instances in the dataset split.
    - w_i, h_i are the bounding box width and height in pixel coordinates after resizing
      to the training input resolution (e.g., 768 x 768).
"""

import argparse
from pathlib import Path
import numpy as np
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent


def compute_nwd_c(data_yaml_path: str, split: str = "train", imgsz: int = 768):
    data_yaml = Path(data_yaml_path).resolve()
    with open(data_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)

    root_dir = Path(data_cfg.get("path", data_yaml.parent))
    split_img_rel = data_cfg.get(split, f"images/{split}")
    img_dir = (root_dir / split_img_rel).resolve()
    label_dir = (root_dir / "labels" / split).resolve()

    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found at: {label_dir}")

    cache_file = label_dir.with_suffix(".cache")
    if not cache_file.exists():
        # Check inside label_dir or root
        alt_cache = label_dir.parent / f"{split}.cache"
        if alt_cache.exists():
            cache_file = alt_cache

    w_norms = []
    h_norms = []

    if cache_file.exists():
        print(f"[INFO] Loading cached dataset annotations from: {cache_file}")
        cache_data = np.load(cache_file, allow_pickle=True).item()
        labels = cache_data.get("labels", [])
        for item in labels:
            bboxes = item.get("bboxes")
            if bboxes is not None and len(bboxes) > 0:
                for b in bboxes:
                    w_norms.append(float(b[2]))
                    h_norms.append(float(b[3]))
    else:
        print(f"[INFO] Scanning raw text label files from: {label_dir}...")
        for lf in label_dir.glob("*.txt"):
            with open(lf, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        w_norms.append(float(parts[3]))
                        h_norms.append(float(parts[4]))

    if not w_norms:
        raise ValueError(f"No bounding box labels found in {label_dir}")

    w_norms = np.array(w_norms)
    h_norms = np.array(h_norms)
    N = len(w_norms)

    # 1. Resized to input resolution (imgsz x imgsz)
    w_px = w_norms * imgsz
    h_px = h_norms * imgsz
    areas_px = w_px * h_px

    mean_area = np.mean(areas_px)
    C = np.sqrt(mean_area)

    print("\n" + "=" * 70)
    print(f" NWD Constant C Calibration: {split.upper()} split (imgsz={imgsz}x{imgsz})")
    print("=" * 70)
    print(f"Total Target Instances (N) : {N:,}")
    print(f"Target Width  (mean ± std) : {np.mean(w_px):.2f} ± {np.std(w_px):.2f} px")
    print(f"Target Width  (median)     : {np.median(w_px):.2f} px (IQR: {np.percentile(w_px, 25):.2f} - {np.percentile(w_px, 75):.2f})")
    print(f"Target Height (mean ± std) : {np.mean(h_px):.2f} ± {np.std(h_px):.2f} px")
    print(f"Target Height (median)     : {np.median(h_px):.2f} px (IQR: {np.percentile(h_px, 25):.2f} - {np.percentile(h_px, 75):.2f})")
    print(f"Mean Bbox Area (1/N \u2211 wi*hi) : {mean_area:.2f} px²")
    print(f"RMS Scale Dimension        : {np.sqrt(mean_area):.4f} px")
    print("-" * 70)
    print(f"==> Calibrated NWD Constant C = {C:.4f} px (recommend: {round(C, 1)})")
    print("=" * 70 + "\n")

    return C


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute NWD constant C from YOLO dataset.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(BASE_DIR / "datasets/anti_uav_yolo/data.yaml"),
        help="Path to data.yaml",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split ('train', 'val', 'test')")
    parser.add_argument("--imgsz", type=int, default=768, help="Target input resolution")
    args = parser.parse_args()

    compute_nwd_c(args.data, split=args.split, imgsz=args.imgsz)
