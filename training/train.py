"""Train a YOLOv8 model on the Roboflow football-players-detection dataset.

Based on: https://github.com/zakroum-hicham/football-analysis-CV

This script:
1. Downloads the football-players-detection dataset from Roboflow
2. Fine-tunes a YOLOv8 model on it (ball, goalkeeper, player, referee)
3. Saves the best weights to ../models/best.pt

Usage:
    # First time: provide your Roboflow API key
    python training/train.py --api-key YOUR_KEY

    # Adjust for your hardware:
    python training/train.py --api-key YOUR_KEY --model yolov8n.pt --epochs 50 --imgsz 640
    python training/train.py --api-key YOUR_KEY --model yolov8x.pt --epochs 100 --imgsz 1280  # GPU recommended

The trained model will detect 4 classes: ball, goalkeeper, player, referee.
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train YOLO on football dataset")
    parser.add_argument("--api-key", required=True, help="Roboflow API key")
    parser.add_argument(
        "--model",
        default="yolov8s.pt",
        help="Base YOLO model to fine-tune (default: yolov8s.pt)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Training epochs (default: 50)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Image size (default: 640)"
    )
    parser.add_argument(
        "--batch", type=int, default=4, help="Batch size (default: 4)"
    )
    parser.add_argument(
        "--dataset-version",
        type=int,
        default=1,
        help="Roboflow dataset version (default: 1)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    # ---- Step 1: Download dataset ----
    print("=" * 60)
    print("Step 1: Downloading dataset from Roboflow...")
    print("=" * 60)

    from roboflow import Roboflow

    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace("roboflow-jvuqo").project(
        "football-players-detection-3zvbc"
    )
    version = project.version(args.dataset_version)
    dataset = version.download("yolov8")

    print(f"Dataset downloaded to: {dataset.location}")

    # ---- Step 2: Train ----
    print()
    print("=" * 60)
    print(f"Step 2: Training {args.model} for {args.epochs} epochs "
          f"at {args.imgsz}px (batch={args.batch})...")
    print("=" * 60)

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )

    # ---- Step 3: Copy best weights ----
    print()
    print("=" * 60)
    print("Step 3: Saving best weights...")
    print("=" * 60)

    # The best weights are saved by ultralytics at runs/detect/trainN/weights/best.pt
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    dest = models_dir / "best.pt"

    if best_pt.exists():
        shutil.copy2(best_pt, dest)
        print(f"Best weights saved to: {dest}")
    else:
        print(f"Warning: Could not find {best_pt}")
        # Try to find it manually
        runs_dir = Path("runs/detect")
        if runs_dir.exists():
            train_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
            if train_dirs:
                alt = train_dirs[-1] / "weights" / "best.pt"
                if alt.exists():
                    shutil.copy2(alt, dest)
                    print(f"Best weights saved to: {dest}")

    print()
    print("=" * 60)
    print("Training complete!")
    print(f"Use the model with:  --model-path {dest}")
    print("=" * 60)


if __name__ == "__main__":
    main()
