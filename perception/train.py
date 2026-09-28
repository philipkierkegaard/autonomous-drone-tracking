"""
Train a Small YOLO Model on Drone Detection Dataset (Anti-UAV + Det-Fly + Bird Negatives)
Supports YOLOv8 / YOLO11 nano/small architectures with Apple Silicon (MPS) & CUDA acceleration.
Includes aerospace/flight-specific data augmentations for autonomous drone visual tracking.
"""

import os
from pathlib import Path
import torch


BASE_DIR = Path(__file__).resolve().parent


def train_drone_detector(
    model_name: str = "yolov8n.pt",
    data_yaml: str = None,
    epochs: int = 60,
    imgsz: int = 640,
    batch_size: int = 32,
    device: str = None,
    project: str = None,
    name: str = "yolov8n_drone",
    save_period: int = 10,
    workers: int = 4
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

    # Auto-detect device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "0"
        else:
            device = "cpu"

    print(f"Loading pretrained base model: {model_name}")
    model = YOLO(model_name)

    yaml_path = str(Path(data_yaml).resolve())
    project_path = str(Path(project).resolve())
    print(f"Starting training on device='{device}' using dataset: {yaml_path}")
    print(f"Hyperparameters: epochs={epochs}, batch={batch_size}, imgsz={imgsz}")
    print(f"Output directory: {project_path}/{name}")

    results = model.train(
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
        hsv_v=0.4,          # Value/brightness jitter (sun glare & backlighting)
        scale=0.5,          # Multi-scale distance variations (+/- 50%)
        erasing=0.3,        # Cutout simulating partial occlusions (trees, poles, clouds)
        mosaic=1.0,         # Multi-scale context aggregation
        close_mosaic=10,    # Turn off mosaic for final 10 epochs for crisp bounding boxes
        mixup=0.0,          # Disabled: prevents ghosting/fading tiny distant targets
        copy_paste=0.0,     # Disabled: avoids rectangular edge artifacts
        # --- Training Dynamics ---
        patience=15,        # Early stopping patience
        verbose=True
    )

    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"

    print("\n" + "=" * 60)
    print("Training Completed!")
    print(f"Best weights saved to: {best_weights}")
    print("=" * 60)

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

    parser = argparse.ArgumentParser(description="Train YOLO Drone Detector.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", choices=["yolov8n.pt", "yolov8s.pt", "yolo11n.pt", "yolo11s.pt"], help="Base pretrained YOLO model")
    parser.add_argument("--data", type=str, default=str(BASE_DIR / "datasets/anti_uav_yolo/data.yaml"), help="Path to data.yaml")
    parser.add_argument("--project", type=str, default=str(BASE_DIR / "runs/detect"), help="Output project directory")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (640 or 960)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device ('mps', '0', 'cpu')")
    parser.add_argument("--name", type=str, default="yolov8n_drone", help="Experiment run name")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
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
        workers=args.workers
    )
