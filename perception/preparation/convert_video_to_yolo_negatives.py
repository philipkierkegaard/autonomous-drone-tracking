#!/usr/bin/env python3
"""
Convert Video to YOLO Negative (Background) Frames

Extracts frames from a video with NO target in frame and generates matching
empty (0-byte) .txt annotation files for YOLO training.

Ultralytics YOLO uses empty label files to represent background images,
which teaches the network not to trigger false positives on trees, roofs, or sky.
Optionally blends a balanced ratio (e.g. 10%) directly into an existing YOLO dataset.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import Optional
import cv2
import numpy as np


def convert_video_to_yolo_negatives(
    video_path: str,
    output_dir: str,
    prefix: str = "ingen_drone",
    stride: int = 1,
    quality: int = 95,
) -> int:
    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Processing: {video_path.name}")
    print(f"       Resolution: {width}x{height} | FPS: {fps:.2f} | Total Video Frames: {total_video_frames}")
    print(f"       Sampling Stride: {stride} (Extracting every {stride} frame(s))")
    print(f"       Destination: {output_dir}")

    extracted_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            frame_name = f"{prefix}_frame_{extracted_count:04d}"
            img_path = images_dir / f"{frame_name}.jpg"
            lbl_path = labels_dir / f"{frame_name}.txt"

            # Write high-quality JPEG image
            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

            # Write empty label file (0 bytes for YOLO negative background)
            lbl_path.touch()

            extracted_count += 1

        frame_idx += 1

    cap.release()

    print(f"[SUCCESS] Extracted {extracted_count} frames and generated {extracted_count} empty label files.")
    print(f"          Images: {images_dir}")
    print(f"          Labels: {labels_dir}")
    return extracted_count


def merge_negatives_into_dataset(
    negatives_dir: str,
    dataset_dir: str,
    ratio: float = 0.10,
    symlink: bool = False,
):
    """
    Uniformly samples negatives from negatives_dir and copies/symlinks them into
    the train, valid, and test splits of dataset_dir matching the target ratio.
    """
    neg_images_dir = Path(negatives_dir) / "images"
    neg_labels_dir = Path(negatives_dir) / "labels"
    target_root = Path(dataset_dir)

    all_neg_images = sorted(list(neg_images_dir.glob("*.jpg")))
    if not all_neg_images:
        print(f"[ERROR] No negative images found in {neg_images_dir}")
        return

    print(f"\n[INFO] Merging background negatives into YOLO dataset: {target_root.name}")
    print(f"       Available negative frames: {len(all_neg_images)}")
    print(f"       Target background ratio: {ratio * 100:.1f}%")

    splits = ["train", "valid", "test"]
    offset = 0

    for split in splits:
        split_img_dir = target_root / split / "images"
        split_lbl_dir = target_root / split / "labels"

        if not split_img_dir.exists():
            continue

        existing_positives = [
            f for f in split_img_dir.glob("*.jpg")
            if not f.name.startswith("ingen_drone") and not f.name.startswith("neg_")
        ]
        n_pos = len(existing_positives)
        n_neg_needed = int(round(n_pos * ratio))

        if n_neg_needed <= 0:
            continue

        # Uniformly sample indices from remaining negatives
        available_pool = all_neg_images[offset:]
        if len(available_pool) < n_neg_needed:
            sample_negatives = available_pool
        else:
            step = len(available_pool) / n_neg_needed
            sample_negatives = [available_pool[int(i * step)] for i in range(n_neg_needed)]

        offset += len(sample_negatives)

        # Copy or symlink into split
        added = 0
        for src_img in sample_negatives:
            dst_img = split_img_dir / src_img.name
            src_lbl = neg_labels_dir / (src_img.stem + ".txt")
            dst_lbl = split_lbl_dir / (src_img.stem + ".txt")

            if dst_img.exists():
                continue

            if symlink:
                if not dst_img.exists():
                    os.symlink(src_img.resolve(), dst_img)
                if not dst_lbl.exists():
                    os.symlink(src_lbl.resolve(), dst_lbl)
            else:
                shutil.copy2(src_img, dst_img)
                dst_lbl.touch()

            added += 1

        print(f"       • {split:>5}: Added {added} negatives ({n_pos} positive + {added} negative = {n_pos + added} total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from video and generate empty YOLO label files for background training."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="perception/raw_data/own-collected-data/ingen-drone.mp4",
        help="Path to input background video.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="perception/raw_data/own-collected-data/ingen-drone-negatives",
        help="Path to output directory (will create images/ and labels/).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="ingen_drone",
        help="Filename prefix for extracted frames.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Extract every Nth frame (default: 1 = extract all frames).",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality 1-100 (default: 95).",
    )
    parser.add_argument(
        "--merge-into",
        type=str,
        default=None,
        help="Optional path to existing YOLO dataset to automatically blend negatives into train/valid/test splits.",
    )
    parser.add_argument(
        "--merge-ratio",
        type=float,
        default=0.10,
        help="Ratio of background images to add (default: 0.10 = 10%% background).",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks instead of copying files when merging.",
    )

    args = parser.parse_args()
    convert_video_to_yolo_negatives(
        video_path=args.video,
        output_dir=args.output_dir,
        prefix=args.prefix,
        stride=args.stride,
        quality=args.quality,
    )

    if args.merge_into:
        merge_negatives_into_dataset(
            negatives_dir=args.output_dir,
            dataset_dir=args.merge_into,
            ratio=args.merge_ratio,
            symlink=args.symlink,
        )
