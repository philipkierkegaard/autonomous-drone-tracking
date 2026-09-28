"""
Det-Fly Air-to-Air Dataset Ingestion & YOLO Formatter

Integrates Det-Fly air-to-air video sequences (e.g., Sequence 010) into the YOLO training split.
Features:
1. Automatic file existence verification: seamlessly ignores any missing frames (e.g., OneDrive dropped files).
2. Temporal subsampling (stride k=5 or 10) to prevent redundant consecutive frames.
3. Multi-format annotation parsing (supports Pascal VOC XML and TXT formats).
4. Symlink generation or file copying into datasets/anti_uav_yolo/ without duplicating gigabytes of storage.
5. Strict Sequence-level allocation: Sequence 010 -> train, Sequence 020 (if present) -> test.
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
                img_w = int(sz.find("width").text)
                img_h = int(sz.find("height").text)
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


def parse_txt_annotation(txt_path: str) -> List[Tuple[float, float, float, float]]:
    """Parses text annotation (xmin, ymin, xmax, ymax or xmin, ymin, w, h)."""
    boxes = []
    try:
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                # If 4 numbers: assume xmin, ymin, xmax, ymax or xmin, ymin, w, h
                coords = [float(p) for p in parts if p.replace(".", "", 1).replace("-", "", 1).isdigit()]
                if len(coords) >= 4:
                    boxes.append((coords[0], coords[1], coords[2], coords[3]))
    except Exception as e:
        print(f"Warning: Failed to parse TXT {txt_path}: {e}")
    return boxes


def voc_to_yolo(
    xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int
) -> Optional[Tuple[float, float, float, float]]:
    """Converts pixel [xmin, ymin, xmax, ymax] to normalized YOLO [x_center, y_center, w, h]."""
    # Clamp coordinates
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


def ingest_detfly(
    detfly_img_dir: str = "data/Det-Fly/images",
    detfly_annot_dir: str = "data/Det-Fly/annotations",
    yolo_dataset_dir: str = "datasets/anti_uav_yolo",
    stride: int = 1,
    use_symlinks: bool = True,
    default_img_size: Tuple[int, int] = (3840, 2160)
) -> Dict[str, int]:
    """
    Ingests Det-Fly images and annotations into the existing YOLO dataset directory.
    Sequence '010' -> train split
    Sequence '020' -> test split (if present)
    """
    yolo_base = Path(yolo_dataset_dir).resolve()
    if not (yolo_base / "images" / "train").exists():
        raise FileNotFoundError(
            f"Base YOLO dataset at {yolo_base} not found. Please run prepare_yolo_data.py first."
        )

    # 1. Index all available Det-Fly JPG images recursively
    print(f"Scanning for Det-Fly images in '{detfly_img_dir}'...")
    all_jpg_paths = glob.glob(f"{detfly_img_dir}/**/*.jpg", recursive=True)
    image_dict = {}
    for p in all_jpg_paths:
        fname = os.path.basename(p)
        stem = Path(fname).stem
        image_dict[stem] = Path(p).resolve()

    print(f"Discovered {len(image_dict)} total image files on disk.")

    # 2. Index all available annotations (XML or TXT)
    print(f"Scanning for Det-Fly annotations in '{detfly_annot_dir}'...")
    all_xml_paths = glob.glob(f"{detfly_annot_dir}/**/*.xml", recursive=True)
    all_txt_paths = glob.glob(f"{detfly_annot_dir}/**/*.txt", recursive=True)

    annot_dict = {}
    for p in all_xml_paths:
        stem = Path(p).stem
        annot_dict[stem] = (Path(p).resolve(), "xml")
    for p in all_txt_paths:
        stem = Path(p).stem
        if stem not in annot_dict and not stem.startswith("_"):
            annot_dict[stem] = (Path(p).resolve(), "txt")

    print(f"Discovered {len(annot_dict)} total annotation files.")

    if not annot_dict:
        print("\n[!] WARNING: No annotation files (.xml or .txt) found in detfly_annot_dir.")
        print(f"    Please place the extracted Det-Fly annotations into: {detfly_annot_dir}")
        return {"matched": 0, "processed": 0}

    # 3. Find matched pairs (automatically skipping any missing frames/dropped downloads)
    matched_stems = sorted(list(set(image_dict.keys()).intersection(set(annot_dict.keys()))))
    skipped_count = len(annot_dict) - len(matched_stems)
    print(f"\nMatched {len(matched_stems)} complete (Image + Annotation) pairs.")
    if skipped_count > 0:
        print(f"Note: Automatically skipped {skipped_count} frames that were missing either the image or label.")

    # 4. Group by sequence
    sequences = {}
    for stem in matched_stems:
        seq_id = stem[:3]  # e.g., '010'
        if seq_id not in sequences:
            sequences[seq_id] = []
        sequences[seq_id].append(stem)

    stats = {"train_added": 0, "test_added": 0, "skipped_stride": 0}

    # Determine image dimensions from first image if available
    sample_img_w, sample_img_h = default_img_size
    if matched_stems:
        try:
            with Image.open(image_dict[matched_stems[0]]) as img:
                sample_img_w, sample_img_h = img.size
                print(f"Detected image resolution: {sample_img_w} x {sample_img_h}")
        except Exception:
            pass

    # 5. Process and insert into splits
    for seq_id, stems in sorted(sequences.items()):
        # Sequence 010 -> train, 020 -> test (unseen air-to-air evaluation), other -> train
        target_split = "test" if seq_id == "020" else "train"
        print(f"\nProcessing Sequence {seq_id} ({len(stems)} frames) -> '{target_split}' split with stride={stride}:")

        dest_img_dir = yolo_base / "images" / target_split
        dest_lbl_dir = yolo_base / "labels" / target_split

        # Subsample with stride
        subsampled_stems = stems[::stride]
        total_sub = len(subsampled_stems)
        print(f"  Selecting {total_sub} frames after temporal striding...")

        for idx, stem in enumerate(subsampled_stems, 1):
            if idx % 1000 == 0 or idx == total_sub:
                print(f"    [{idx}/{total_sub}] Processed {stem}...")

            img_src = image_dict[stem]
            annot_src, annot_type = annot_dict[stem]

            # Parse bounding boxes
            if annot_type == "xml":
                raw_boxes, xml_w, xml_h = parse_voc_xml(str(annot_src))
                cur_w = xml_w or sample_img_w
                cur_h = xml_h or sample_img_h
            else:
                raw_boxes = parse_txt_annotation(str(annot_src))
                cur_w, cur_h = sample_img_w, sample_img_h

            # Convert to YOLO format
            yolo_lines = []
            for box in raw_boxes:
                # If VOC format [xmin, ymin, xmax, ymax]
                if len(box) == 4:
                    b_yolo = voc_to_yolo(box[0], box[1], box[2], box[3], cur_w, cur_h)
                    if b_yolo is not None:
                        yolo_lines.append(f"0 {b_yolo[0]:.6f} {b_yolo[1]:.6f} {b_yolo[2]:.6f} {b_yolo[3]:.6f}")

            # Destination filenames prefixed with detfly_ to avoid collision
            dest_stem = f"detfly_{stem}"
            dest_img_path = dest_img_dir / f"{dest_stem}.jpg"
            dest_lbl_path = dest_lbl_dir / f"{dest_stem}.txt"

            # Write label file
            with open(dest_lbl_path, "w") as f:
                if yolo_lines:
                    f.write("\n".join(yolo_lines) + "\n")

            # Link or copy image
            if dest_img_path.exists() or dest_img_path.is_symlink():
                dest_img_path.unlink()

            if use_symlinks:
                try:
                    os.symlink(img_src, dest_img_path)
                except OSError:
                    import shutil
                    shutil.copy2(img_src, dest_img_path)
            else:
                import shutil
                shutil.copy2(img_src, dest_img_path)

            if target_split == "train":
                stats["train_added"] += 1
            else:
                stats["test_added"] += 1

    print("\n" + "=" * 50)
    print("DET-FLY INGESTION COMPLETED")
    print(f"  Added to Train Split: {stats['train_added']} images")
    print(f"  Added to Test Split:  {stats['test_added']} images")
    print(f"  Destination Dataset:  {yolo_base}")
    print("=" * 50)
    return stats


BASE_DIR = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Det-Fly dataset into YOLO structure")
    parser.add_argument("--img-dir", type=str, default=str(BASE_DIR / "data/Det-Fly/images"), help="Path to Det-Fly images")
    parser.add_argument("--annot-dir", type=str, default=str(BASE_DIR / "data/Det-Fly/annotations"), help="Path to Det-Fly annotations")
    parser.add_argument("--yolo-dir", type=str, default=str(BASE_DIR / "datasets/anti_uav_yolo"), help="YOLO dataset root")
    parser.add_argument("--stride", type=int, default=1, help="Temporal subsampling stride (default: 1 for full dataset)")
    parser.add_argument("--no-symlink", action="store_true", help="Copy files instead of symlinks")

    args = parser.parse_args()
    ingest_detfly(
        detfly_img_dir=args.img_dir,
        detfly_annot_dir=args.annot_dir,
        yolo_dataset_dir=args.yolo_dir,
        stride=args.stride,
        use_symlinks=not args.no_symlink,
    )
