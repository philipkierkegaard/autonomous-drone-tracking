"""
Train a Small YOLO Model on Drone Detection Dataset (Anti-UAV + Det-Fly + Bird Negatives)
Supports YOLOv8 / YOLO11 nano/small architectures with Apple Silicon (MPS) & CUDA acceleration.
Includes:
  1. Aerospace/flight-specific geometric & optical data augmentations.
  2. High-robustness low-contrast & atmospheric haze Albumentations (CLAHE, Gamma, Contrast drops, ToGray, MotionBlur).
  3. Hybrid CIoU + NWD (Normalized Wasserstein Distance) loss for non-vanishing tiny-drone regression gradients.
"""

import os
import shutil
from pathlib import Path
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent


# ==============================================================================
# 1. Low-Contrast & Atmospheric Haze Data Augmentation Pipeline
# ==============================================================================

def setup_low_contrast_augmentations(enabled: bool = True):
    """
    Patches Ultralytics Albumentations module with aggressive low-contrast,
    overcast desaturation, atmospheric haze, and motion blur augmentations.
    """
    if not enabled:
        return

    try:
        import albumentations as A
        import ultralytics.data.augment as u_augment
    except ImportError:
        print("[WARNING] Albumentations not installed. Skipping low-contrast augmentation hook.")
        return

    class LowContrastAlbumentations(u_augment.Albumentations):
        def __init__(self, p: float = 1.0, transforms: list | None = None, flip_idx: list[int] | None = None):
            if transforms is None:
                transforms = [
                    # 1. Negative contrast drop (simulates flat overcast gray skies & thick haze)
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.25, 0.20),
                        contrast_limit=(-0.50, 0.15),
                        p=0.60
                    ),
                    # 2. Gamma variations (simulates non-linear optical haze, backlighting, and sensor dynamic range loss)
                    A.RandomGamma(gamma_limit=(65, 145), p=0.40),
                    # 3. Grayscale conversion (forces the network to learn pure geometry without color cues)
                    A.ToGray(p=0.20),
                    # 4. CLAHE (trains on local contrast gradient normalization)
                    A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.30),
                    # 5. Motion blur (simulates camera shutter blur from high-speed drone chase)
                    A.MotionBlur(blur_limit=(3, 7), p=0.15),
                    # 6. Sensor compression noise
                    A.ImageCompression(quality_range=(55, 95), p=0.20),
                ]
            super().__init__(p=p, transforms=transforms, flip_idx=flip_idx)

    u_augment.Albumentations = LowContrastAlbumentations
    print("[INFO] Successfully injected Low-Contrast & Atmospheric Haze Albumentations into DataLoader:")
    print("       • Contrast Drop (-50% to +15%, p=0.60)")
    print("       • Random Gamma / Haze (65–145, p=0.40)")
    print("       • ToGray Desaturation (p=0.20)")
    print("       • CLAHE Adaptive Contrast (p=0.30)")
    print("       • Camera Motion Blur (p=0.15)")


# ==============================================================================
# 2. Hybrid CIoU + NWD (Normalized Wasserstein Distance) Bounding Box Loss
# ==============================================================================

def setup_hybrid_nwd_loss(alpha: float = 0.5, C: float = 12.0):
    """
    Patches Ultralytics BboxLoss with a Hybrid CIoU + NWD loss formulation.
    Protects small/tiny target regression against vanishing gradients when IoU = 0.
    """
    try:
        import ultralytics.utils.loss as u_loss
        from ultralytics.utils.metrics import bbox_iou
        from ultralytics.utils.tal import bbox2dist
    except ImportError:
        print("[WARNING] Could not import Ultralytics loss modules. Running standard loss.")
        return

    class HybridNWD_BboxLoss(u_loss.BboxLoss):
        def __init__(self, reg_max: int = 16):
            super().__init__(reg_max)
            self.alpha = alpha  # Weight for NWD vs CIoU (e.g., 0.5 = 50% CIoU, 50% NWD)
            self.C = C          # NWD scale parameter calibrated for tiny objects

        def forward(
            self,
            pred_dist: torch.Tensor,
            pred_bboxes: torch.Tensor,
            anchor_points: torch.Tensor,
            target_bboxes: torch.Tensor,
            target_scores: torch.Tensor,
            target_scores_sum: torch.Tensor,
            fg_mask: torch.Tensor,
            imgsz: torch.Tensor,
            stride: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            weight = target_scores[fg_mask].sum(-1, keepdim=True)
            pbox = pred_bboxes[fg_mask]
            tbox = target_bboxes[fg_mask]

            # 1. Standard CIoU Loss (Tight rectangular boundary alignment)
            iou = bbox_iou(pbox, tbox, xywh=False, CIoU=True)
            loss_ciou = 1.0 - iou

            # 2. Normalized Wasserstein Distance (Smooth non-zero Gaussian gradient for tiny boxes)
            cx1 = (pbox[..., 0] + pbox[..., 2]) / 2.0
            cy1 = (pbox[..., 1] + pbox[..., 3]) / 2.0
            w1 = (pbox[..., 2] - pbox[..., 0]).clamp(min=1e-6)
            h1 = (pbox[..., 3] - pbox[..., 1]).clamp(min=1e-6)

            cx2 = (tbox[..., 0] + tbox[..., 2]) / 2.0
            cy2 = (tbox[..., 1] + tbox[..., 3]) / 2.0
            w2 = (tbox[..., 2] - tbox[..., 0]).clamp(min=1e-6)
            h2 = (tbox[..., 3] - tbox[..., 1]).clamp(min=1e-6)

            w2_dist = (cx1 - cx2)**2 + (cy1 - cy2)**2 + ((w1 - w2)**2 + (h1 - h2)**2) / 4.0
            nwd = torch.exp(-torch.sqrt(w2_dist.clamp(min=1e-6)) / self.C)
            if nwd.ndim < loss_ciou.ndim:
                nwd = nwd.unsqueeze(-1)
            loss_nwd = 1.0 - nwd

            # 3. Hybrid Box Loss Combination
            loss_box = (1.0 - self.alpha) * loss_ciou + self.alpha * loss_nwd
            loss_iou = (loss_box * weight).sum() / target_scores_sum

            # DFL Loss
            if self.dfl_loss:
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
                loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask])
                loss_dfl = (loss_dfl * weight).sum() / target_scores_sum
            else:
                loss_dfl = torch.tensor(0.0, device=pred_dist.device)

            return loss_iou, loss_dfl

    # Inject into Ultralytics loss module
    u_loss.BboxLoss = HybridNWD_BboxLoss
    print(f"[INFO] Successfully activated Hybrid CIoU + NWD Bbox Loss (alpha={alpha}, C={C})")


# ==============================================================================
# 3. Main Training Function
# ==============================================================================

def train_drone_detector(
    model_name: str = "yolov8n.pt",
    data_yaml: str = None,
    epochs: int = 40,
    imgsz: int = 768,
    batch_size: int = 32,
    device: str = None,
    project: str = None,
    name: str = "yolov8n_drone_v4_nwd_768",
    save_period: int = 10,
    workers: int = 4,
    patience: int = 0,
    close_mosaic: int = 10,
    lr0: float = None,
    use_nwd: bool = True,
    nwd_alpha: float = 0.5,
    nwd_c: float = 12.0,
    low_contrast: bool = True
):
    if data_yaml is None:
        data_yaml = str(BASE_DIR / "datasets/anti_uav_yolo/data.yaml")
    if project is None:
        project = str(BASE_DIR / "runs/detect")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics is not installed. Run: pip install ultralytics")
        return

    # 1. Enable Low-Contrast Augmentations
    if low_contrast:
        setup_low_contrast_augmentations(enabled=True)

    # 2. Enable Hybrid NWD loss if requested
    if use_nwd:
        setup_hybrid_nwd_loss(alpha=nwd_alpha, C=nwd_c)

    # Auto-detect device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "0"
        else:
            device = "cpu"

    print(f"\nLoading pretrained base model: {model_name}")
    model = YOLO(model_name)

    yaml_path = str(Path(data_yaml).resolve())
    project_path = str(Path(project).resolve())
    print(f"Starting training on device='{device}' using dataset: {yaml_path}")
    print(f"Hyperparameters: epochs={epochs}, batch={batch_size}, imgsz={imgsz}, patience={patience}")
    print(f"Output directory: {project_path}/{name}\n")

    train_kwargs = dict(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project_path,
        name=name,
        save_period=save_period,
        workers=workers,
        plots=True,
        deterministic=False,  # Speeds up MPS backprop and silences non-deterministic warning
        # --- Flight-Specific & Optical Augmentations ---
        degrees=12.0,       # Simulates banking/roll during flight maneuvers
        fliplr=0.5,         # Horizontal flight direction symmetry
        flipud=0.0,         # Disabled: preserves ground vs sky horizon prior
        hsv_h=0.015,        # Subtle hue variation
        hsv_s=0.7,          # Saturation variation (overcast vs clear blue sky)
        hsv_v=0.5,          # Expanded value/brightness jitter (sun glare & backlighting)
        scale=0.5,          # Multi-scale distance variations (+/- 50%)
        erasing=0.3,        # Cutout simulating partial occlusions (trees, poles, clouds)
        mosaic=1.0,         # Multi-scale context aggregation
        close_mosaic=close_mosaic,  # Turn off mosaic for final epochs for crisp bounding boxes
        mixup=0.0,          # Disabled: prevents ghosting/fading tiny distant targets
        copy_paste=0.0,     # Disabled: avoids rectangular edge artifacts
        # --- Training Dynamics ---
        patience=patience,  # 0 disables early stopping
        verbose=True
    )
    if lr0 is not None:
        train_kwargs["lr0"] = lr0

    results = model.train(**train_kwargs)

    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"

    print("\n" + "=" * 60)
    print("Training Completed!")
    print(f"Best weights saved to: {best_weights}")
    print("=" * 60)

    # Save dedicated copy in perception/weights/
    target_weights_dir = BASE_DIR / "weights"
    target_weights_dir.mkdir(parents=True, exist_ok=True)
    published_weights = target_weights_dir / f"{name}_best.pt"
    if best_weights.exists():
        shutil.copy2(best_weights, published_weights)
        print(f"Published model weights to: {published_weights}")

    # Automatically run evaluation on the held-out test split (Dual-Domain: Anti-UAV + Det-Fly 020)
    if best_weights.exists():
        print("\nRunning post-training benchmark on Held-Out Test Set (Anti-UAV + Det-Fly 020)...")
        best_model = YOLO(str(best_weights))
        test_results = best_model.val(
            data=yaml_path,
            split="test",
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            plots=True,
            project=project_path,
            name=f"{name}_test_eval"
        )
        print("Test Evaluation Finished!")
        return results, test_results

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLO Drone Detector with Hybrid NWD Loss & Low-Contrast Augmentation.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base pretrained YOLO model or checkpoint path")
    parser.add_argument("--data", type=str, default=str(BASE_DIR / "datasets/anti_uav_yolo/data.yaml"), help="Path to data.yaml")
    parser.add_argument("--project", type=str, default=str(BASE_DIR / "runs/detect"), help="Output project directory")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=768, help="Input image size (default: 768)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--device", type=str, default=None, help="Device ('mps', '0', 'cpu')")
    parser.add_argument("--name", type=str, default="yolov8n_drone_v4_nwd_768", help="Experiment run name")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (0 to disable early stopping)")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Epochs before end to disable mosaic augmentation")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate (optional)")
    parser.add_argument("--no-nwd", action="store_true", help="Disable NWD loss and use standard CIoU")
    parser.add_argument("--nwd-alpha", type=float, default=0.5, help="NWD loss weight alpha (default: 0.5)")
    parser.add_argument("--nwd-c", type=float, default=12.0, help="NWD scale parameter C (default: 12.0)")
    parser.add_argument("--no-low-contrast", action="store_true", help="Disable low-contrast Albumentations")
    args = parser.parse_args()

    train_drone_detector(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        lr0=args.lr0,
        use_nwd=not args.no_nwd,
        nwd_alpha=args.nwd_alpha,
        nwd_c=args.nwd_c,
        low_contrast=not args.no_low_contrast
    )
