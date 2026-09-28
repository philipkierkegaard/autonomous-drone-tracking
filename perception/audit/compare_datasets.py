"""
Comparative Dataset Audit: Anti-UAV (Ground-to-Air) vs. Det-Fly (Air-to-Air)
Generates a publication-grade 2x2 comparison plot focusing on the two most critical control metrics:
1. Scale Distribution (COCO Bounding Box Scale - Range Proxy)
2. Spatial Centroid Heatmap (Center of Mass & Framing Bias - Visual Servoing Error)
"""

import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BASE_DIR.parent


def load_anti_uav_data():
    gt_dir = str(BASE_DIR / "data/Anti-UAV/Anti-UAV-Tracking-V0GT")
    video_dir = str(BASE_DIR / "data/Anti-UAV/Anti-UAV-Tracking-V0")

    gt_files = sorted(glob.glob(f"{gt_dir}/*.txt"))
    areas = []
    centroids_x = []
    centroids_y = []

    small_c = 0
    medium_c = 0
    large_c = 0

    for gf in gt_files:
        vname = Path(gf).stem.replace("_gt", "")
        v_path = Path(video_dir) / vname
        img_files = sorted(glob.glob(f"{v_path}/*.jpg"))
        if not img_files:
            continue
        with Image.open(img_files[0]) as im:
            iw, ih = im.size

        scale_x = 640.0 / iw
        scale_y = 640.0 / ih

        with open(gf, "r") as f:
            for line in f:
                parts = line.strip().replace(",", " ").split()
                if len(parts) >= 4:
                    try:
                        x, y, w, h = map(float, parts[:4])
                        if w > 0 and h > 0:
                            area_640 = (w * scale_x) * (h * scale_y)
                            areas.append(area_640)
                            centroids_x.append((x + w / 2.0) / iw)
                            centroids_y.append((y + h / 2.0) / ih)

                            if area_640 < 32 * 32:
                                small_c += 1
                            elif area_640 <= 96 * 96:
                                medium_c += 1
                            else:
                                large_c += 1
                    except ValueError:
                        pass

    return {
        "small": small_c,
        "medium": medium_c,
        "large": large_c,
        "total": small_c + medium_c + large_c,
        "cx": centroids_x,
        "cy": centroids_y,
    }


def load_detfly_data():
    # Check possible locations for 010 annotations
    possible_annot_dirs = [
        str(BASE_DIR / "data/Det-Fly/annotations/010"),
        str(BASE_DIR / "data/Det-Fly/010"),
    ]
    annot_dir = None
    for d in possible_annot_dirs:
        if os.path.isdir(d):
            annot_dir = d
            break
    if annot_dir is None:
        # Fallback recursive search
        found = glob.glob(str(BASE_DIR / "data/Det-Fly/**/010"), recursive=True)
        annot_dir = found[0] if found else str(BASE_DIR / "data/Det-Fly/010")

    img_base = str(BASE_DIR / "data/Det-Fly/images")
    image_paths = {Path(p).stem: p for p in glob.glob(f"{img_base}/**/*.jpg", recursive=True)}
    xml_files = sorted(glob.glob(f"{annot_dir}/*.xml"))

    centroids_x = []
    centroids_y = []

    small_c = 0
    medium_c = 0
    large_c = 0

    img_w, img_h = 3840.0, 2160.0
    scale_x = 640.0 / img_w
    scale_y = 640.0 / img_h

    for xf in xml_files:
        stem = Path(xf).stem
        if stem not in image_paths:
            continue

        try:
            tree = ET.parse(xf)
            root = tree.getroot()
            for obj in root.findall("object"):
                bb = obj.find("bndbox")
                if bb is None:
                    continue
                xmin = float(bb.find("xmin").text)
                ymin = float(bb.find("ymin").text)
                xmax = float(bb.find("xmax").text)
                ymax = float(bb.find("ymax").text)

                w = xmax - xmin
                h = ymax - ymin
                if w <= 0 or h <= 0:
                    continue

                # Centroids
                centroids_x.append((xmin + w / 2.0) / img_w)
                centroids_y.append((ymin + h / 2.0) / img_h)

                # Scaled area at 640x640 YOLO input
                area_640 = (w * scale_x) * (h * scale_y)
                if area_640 < 32 * 32:
                    small_c += 1
                elif area_640 <= 96 * 96:
                    medium_c += 1
                else:
                    large_c += 1
        except Exception:
            pass

    return {
        "small": small_c,
        "medium": medium_c,
        "large": large_c,
        "total": small_c + medium_c + large_c,
        "cx": centroids_x,
        "cy": centroids_y,
    }


def generate_comparison_plot():
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import gaussian_filter
    import shutil

    print("Loading Anti-UAV data...")
    anti = load_anti_uav_data()
    print("Loading Det-Fly data...")
    detfly = load_detfly_data()

    print("Generating 2x2 publication comparison figure (Jewel Ocean palette)...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")

    # Academic Jewel Palette: Royal Cobalt (Small), Luminous Teal (Medium), Warm Amber (Large)
    scale_labels = ["Small\n(< 32×32 px)", "Medium\n(32–96 px)", "Large\n(> 96×96 px)"]
    palette = ["#2563EB", "#0D9488", "#F59E0B"]

    # Smooth Oceanic Density Colormap: Pure White -> Sky -> Cerulean -> Royal Cobalt -> Midnight
    cmap_ocean = LinearSegmentedColormap.from_list(
        "ocean_density",
        [
            (0.0, "#FFFFFF"),
            (0.12, "#E0F2FE"),
            (0.30, "#7DD3FC"),
            (0.55, "#0284C7"),
            (0.80, "#1D4ED8"),
            (1.0, "#0F172A"),
        ],
    )

    # =========================================================================
    # Row 1, Col 0: Anti-UAV Scale Breakdown
    # =========================================================================
    ax00 = axes[0, 0]
    ax00.set_facecolor("#FFFFFF")
    anti_counts = [anti["small"], anti["medium"], anti["large"]]
    bars00 = ax00.bar(scale_labels, anti_counts, color=palette, width=0.48, zorder=3)
    for bar, count in zip(bars00, anti_counts):
        yval = bar.get_height()
        pct = count / anti["total"] * 100
        ax00.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + (17000 * 0.02),
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            color="#0F172A",
            fontsize=11,
            fontweight="bold",
        )
    ax00.set_title(
        "A1. Anti-UAV (Ground-to-Air): Target Scale Breakdown (@ 640×640)",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
        pad=12,
    )
    ax00.set_ylabel("Bounding Box Count", color="#475569", fontsize=11)
    ax00.tick_params(colors="#64748B", labelsize=10.5, bottom=False)
    ax00.grid(axis="y", color="#F1F5F9", linestyle="-", linewidth=1.2, zorder=1)
    ax00.spines["top"].set_visible(False)
    ax00.spines["right"].set_visible(False)
    ax00.spines["left"].set_color("#E2E8F0")
    ax00.spines["bottom"].set_color("#CBD5E1")
    ax00.set_ylim(0, 17000)

    # =========================================================================
    # Row 1, Col 1: Det-Fly Scale Breakdown
    # =========================================================================
    ax01 = axes[0, 1]
    ax01.set_facecolor("#FFFFFF")
    det_counts = [detfly["small"], detfly["medium"], detfly["large"]]
    bars01 = ax01.bar(scale_labels, det_counts, color=palette, width=0.48, zorder=3)
    for bar, count in zip(bars01, det_counts):
        yval = bar.get_height()
        pct = count / detfly["total"] * 100
        ax01.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + (6000 * 0.02),
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            color="#0F172A",
            fontsize=11,
            fontweight="bold",
        )
    ax01.set_title(
        "A2. Det-Fly Seq 010 (Air-to-Air): Target Scale Breakdown (@ 640×640)",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
        pad=12,
    )
    ax01.set_ylabel("Bounding Box Count", color="#475569", fontsize=11)
    ax01.tick_params(colors="#64748B", labelsize=10.5, bottom=False)
    ax01.grid(axis="y", color="#F1F5F9", linestyle="-", linewidth=1.2, zorder=1)
    ax01.spines["top"].set_visible(False)
    ax01.spines["right"].set_visible(False)
    ax01.spines["left"].set_color("#E2E8F0")
    ax01.spines["bottom"].set_color("#CBD5E1")
    ax01.set_ylim(0, 6000)

    # =========================================================================
    # Row 2, Col 0: Anti-UAV Spatial Centroid Heatmap
    # =========================================================================
    ax10 = axes[1, 0]
    ax10.set_facecolor("#FFFFFF")
    H0, _, _ = np.histogram2d(anti["cx"], anti["cy"], bins=80, range=[[0, 1], [0, 1]])
    H0_smooth = gaussian_filter(H0.T, sigma=1.4)
    H0_masked = np.ma.masked_where(H0_smooth < 0.08, H0_smooth)

    ax10.imshow(
        H0_masked,
        extent=[0, 1, 1, 0],
        origin="upper",
        cmap=cmap_ocean,
        interpolation="bicubic",
        aspect="auto",
        zorder=2,
    )
    ax10.axhline(0.5, color="#94A3B8", linestyle="--", alpha=0.5, linewidth=1.1, zorder=4)
    ax10.axvline(0.5, color="#94A3B8", linestyle="--", alpha=0.5, linewidth=1.1, zorder=4)

    mean_anti_x = np.mean(anti["cx"])
    mean_anti_y = np.mean(anti["cy"])
    ax10.plot(
        mean_anti_x,
        mean_anti_y,
        marker="P",
        color="#E11D48",
        markersize=13,
        markeredgewidth=2.2,
        markeredgecolor="#FFFFFF",
        linestyle="None",
        zorder=6,
        label=f"Center of Mass: ({mean_anti_x:.2f}, {mean_anti_y:.2f})",
    )
    ax10.set_title(
        "B1. Anti-UAV (Ground-to-Air): Centroid Distribution Heatmap",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
        pad=12,
    )
    ax10.set_xlabel("Normalized X Coordinate (0 = Left, 1 = Right)", color="#475569", fontsize=11)
    ax10.set_ylabel("Normalized Y Coordinate (0 = Top, 1 = Bottom)", color="#475569", fontsize=11)
    ax10.tick_params(colors="#64748B", labelsize=10.5)
    for spine in ax10.spines.values():
        spine.set_color("#CBD5E1")
    ax10.legend(
        facecolor="#FFFFFF",
        edgecolor="#E2E8F0",
        labelcolor="#0F172A",
        fontsize=10.5,
        loc="lower left",
        framealpha=0.95,
    )

    # =========================================================================
    # Row 2, Col 1: Det-Fly Spatial Centroid Heatmap
    # =========================================================================
    ax11 = axes[1, 1]
    ax11.set_facecolor("#FFFFFF")
    H1, _, _ = np.histogram2d(detfly["cx"], detfly["cy"], bins=80, range=[[0, 1], [0, 1]])
    H1_smooth = gaussian_filter(H1.T, sigma=1.4)
    H1_masked = np.ma.masked_where(H1_smooth < 0.08, H1_smooth)

    ax11.imshow(
        H1_masked,
        extent=[0, 1, 1, 0],
        origin="upper",
        cmap=cmap_ocean,
        interpolation="bicubic",
        aspect="auto",
        zorder=2,
    )
    ax11.axhline(0.5, color="#94A3B8", linestyle="--", alpha=0.5, linewidth=1.1, zorder=4)
    ax11.axvline(0.5, color="#94A3B8", linestyle="--", alpha=0.5, linewidth=1.1, zorder=4)

    mean_det_x = np.mean(detfly["cx"])
    mean_det_y = np.mean(detfly["cy"])
    ax11.plot(
        mean_det_x,
        mean_det_y,
        marker="P",
        color="#E11D48",
        markersize=13,
        markeredgewidth=2.2,
        markeredgecolor="#FFFFFF",
        linestyle="None",
        zorder=6,
        label=f"Center of Mass: ({mean_det_x:.2f}, {mean_det_y:.2f})",
    )
    ax11.set_title(
        "B2. Det-Fly Seq 010 (Air-to-Air): Centroid Distribution Heatmap",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
        pad=12,
    )
    ax11.set_xlabel("Normalized X Coordinate (0 = Left, 1 = Right)", color="#475569", fontsize=11)
    ax11.set_ylabel("Normalized Y Coordinate (0 = Top, 1 = Bottom)", color="#475569", fontsize=11)
    ax11.tick_params(colors="#64748B", labelsize=10.5)
    for spine in ax11.spines.values():
        spine.set_color("#CBD5E1")
    ax11.legend(
        facecolor="#FFFFFF",
        edgecolor="#E2E8F0",
        labelcolor="#0F172A",
        fontsize=10.5,
        loc="lower left",
        framealpha=0.95,
    )

    # Master Figure Title
    fig.suptitle(
        "Ground-to-Air vs. Air-to-Air Benchmark Comparison: Scale & Centroid Distribution",
        fontsize=16,
        fontweight="bold",
        color="#0F172A",
        y=0.99,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    out_dir = ROOT_DIR / "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_file = str(out_dir / "dataset_comparison_2x2.png")
    plt.savefig(out_file, dpi=200, facecolor="#FFFFFF", edgecolor="none")
    plt.close()

    print(f"\n[✓] Successfully saved comparison graphic to: {out_file}")


if __name__ == "__main__":
    generate_comparison_plot()
