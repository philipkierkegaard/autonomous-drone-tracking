#!/usr/bin/env python3
"""
DUT-Anti-UAV Ingestion & YOLO Formatter
Ingests curated DUT-Anti-UAV train images and Pascal VOC XML annotations into the YOLO training split.
Also ensures negative bird samples are linked into the training split.
"""

import os
import glob
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image


def parse_voc_xml(xml_path: str) -> Tuple[List[Tuple[float, float, float, float]], Optional[int], Optional[int]]:
    """Parses Pascal VOC XML annotation and returns (list of [xmin, ymin, xmax, ymax], width, height)."""
    boxes = []
    img_w, img_h = None, None
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        sz = root.find("size")
        if sz is not None:
            try:
                img_w = int(float(sz.find("width").text))
                img_h = int(float(sz.find("height").text))
            except (ValueError, AttributeError):
                pass
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            if bndbox is not None:
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                boxes.append((xmin, ymin, xmax, ymax))
    except Exception as e:
        print(f"Warning: Failed to parse XML {xml_path}: {e}")
    return boxes, img_w, img_h


def voc_to_yolo(
    xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int
) -> Optional[Tuple[float, float, float, float]]:
    """Converts pixel [xmin, ymin, xmax, ymax] to normalized YOLO [x_center, y_center, w, h]."""
    xmin = max(0.0, min(xmin, float(img_w)))
    ymin = max(0.0, min(ymin, float(img_h)))
    xmax = max(0.0, min(xmax, float(img_w)))
    ymax = max(0.0, min(ymax, float(img_h)))

    w = xmax - xmin
    h = ymax - ymin

    if w <= 1.0 or h <= 1.0:
        return None

    x_center = (xmin + w / 2.0) / img_w
    y_center = (ymin + h / 2.0) / img_h
    norm_w = w / img_w
    norm_h = h / img_h

    # Clip to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))

    return x_center, y_center, norm_w, norm_h


def ingest_dut_anti_uav(
    dut_img_dir: str = "data/DUT-Anti-UAV/train/img",
    dut_xml_dir: str = "data/DUT-Anti-UAV/train/xml",
    yolo_dataset_dir: str = "datasets/anti_uav_yolo",
    use_symlinks: bool = True
) -> Dict[str, int]:
    """Ingests DUT-Anti-UAV training images and annotations into YOLO train split."""
    yolo_base = Path(yolo_dataset_dir).resolve()
    dest_img_dir = yolo_base / "images" / "train"
    dest_lbl_dir = yolo_base / "labels" / "train"

    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_dir_path = Path(dut_img_dir).resolve()
    xml_dir_path = Path(dut_xml_dir).resolve()

    img_files = sorted([f for f in img_dir_path.glob("*.jpg") if not f.name.startswith(".")])
    print(f"\nDiscovered {len(img_files)} active images in {img_dir_path}...")

    stats = {"processed": 0, "boxes_written": 0, "negatives": 0}

    for idx, img_p in enumerate(img_files, 1):
        if idx % 2000 == 0 or idx == len(img_files):
            print(f"  [{idx}/{len(img_files)}] Ingesting {img_p.name}...")

        stem = img_p.stem
        xml_p = xml_dir_path / f"{stem}.xml"

        yolo_lines = []
        if xml_p.exists():
            raw_boxes, xml_w, xml_h = parse_voc_xml(str(xml_p))
            if not xml_w or not xml_h:
                try:
                    with Image.open(img_p) as im:
                        xml_w, xml_h = im.size
                except Exception:
                    xml_w, xml_h = 1920, 1080

            for box in raw_boxes:
                yolo_box = voc_to_yolo(box[0], box[1], box[2], box[3], xml_w, xml_h)
                if yolo_box is not None:
                    xc, yc, nw, nh = yolo_box
                    yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                    stats["boxes_written"] += 1

        dest_stem = f"dut_{stem}"
        dest_img_path = dest_img_dir / f"{dest_stem}.jpg"
        dest_lbl_path = dest_lbl_dir / f"{dest_stem}.txt"

        # Write label file
        with open(dest_lbl_path, "w") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines) + "\n")
            else:
                stats["negatives"] += 1

        # Link or copy image
        if dest_img_path.exists() or dest_img_path.is_symlink():
            dest_img_path.unlink()

        if use_symlinks:
            try:
                os.symlink(str(img_p), str(dest_img_path))
            except OSError:
                import shutil
                shutil.copy2(str(img_p), str(dest_img_path))
        else:
            import shutil
            shutil.copy2(str(img_p), str(dest_img_path))

        stats["processed"] += 1

    print("\n" + "=" * 55)
    print("DUT-ANTI-UAV INGESTION COMPLETED")
    print(f"  Added to YOLO train split: {stats['processed']} images")
    print(f"  Bounding boxes written:    {stats['boxes_written']}")
    print(f"  Background (zero box):     {stats['negatives']}")
    print("=" * 55)
    return stats


def ingest_bird_negatives(
    bird_dir: str = "data/Bird",
    yolo_dataset_dir: str = "datasets/anti_uav_yolo",
    use_symlinks: bool = True
):
    """Links bird images into train as negative background samples."""
    bird_path = Path(bird_dir).resolve()
    if not bird_path.exists():
        return

    yolo_base = Path(yolo_dataset_dir).resolve()
    dest_img_dir = yolo_base / "images" / "train"
    dest_lbl_dir = yolo_base / "labels" / "train"

    bird_imgs = sorted([f for f in bird_path.glob("*.*") if not f.name.startswith(".") and f.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    added = 0
    for bp in bird_imgs:
        dest_stem = f"bird_{bp.stem}"
        dest_img = dest_img_dir / f"{dest_stem}.jpg"
        dest_lbl = dest_lbl_dir / f"{dest_stem}.txt"

        if not dest_lbl.exists():
            dest_lbl.touch()

        if dest_img.exists() or dest_img.is_symlink():
            dest_img.unlink()

        if use_symlinks:
            try:
                os.symlink(str(bp), str(dest_img))
            except OSError:
                import shutil
                shutil.copy2(str(bp), str(dest_img))
        else:
            import shutil
            shutil.copy2(str(bp), str(dest_img))
        added += 1

    print(f"Linked {added} bird negative images into train.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest DUT-Anti-UAV dataset into YOLO train split")
    parser.add_argument("--dut-img-dir", type=str, default="data/DUT-Anti-UAV/train/img")
    parser.add_argument("--dut-xml-dir", type=str, default="data/DUT-Anti-UAV/train/xml")
    parser.add_argument("--yolo-dir", type=str, default="datasets/anti_uav_yolo")
    parser.add_argument("--no-symlink", action="store_true")

    args = parser.parse_args()
    ingest_dut_anti_uav(
        dut_img_dir=args.dut_img_dir,
        dut_xml_dir=args.dut_xml_dir,
        yolo_dataset_dir=args.yolo_dir,
        use_symlinks=not args.no_symlink
    )
    ingest_bird_negatives(
        yolo_dataset_dir=args.yolo_dir,
        use_symlinks=not args.no_symlink
    )
