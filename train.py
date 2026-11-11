"""
train.py - Training Pipeline for PCB Defect Detection (Model Mayhem Challenge)
Target classes:
1. missing_hole
2. mouse_bite
3. open_circuit
4. short
5. spur
6. spurious_copper
"""

import argparse
import os
import sys
from pathlib import Path
import shutil
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model for PCB Defect Detection")
    parser.add_argument("--data", type=str, default="data/data.yaml", help="Path to data.yaml dataset config")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Base model weights (e.g., yolo11n.pt, yolo11s.pt, yolo11m.pt)")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training (e.g., 640, 800, 1024)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (8 is safe for 6GB VRAM with yolo11s)")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu", help="Device (0 for GPU, cpu)")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers (0 recommended on Windows)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience epochs")
    parser.add_argument("--project", type=str, default="runs/train", help="Directory to save runs")
    parser.add_argument("--name", type=str, default="pcb_defect_model", help="Experiment name")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("🚀 MODEL MAYHEM: PCB DEFECT DETECTION - TRAINING PIPELINE")
    print("=" * 60)
    print(f"Dataset Config : {args.data}")
    print(f"Base Model     : {args.model}")
    print(f"Epochs         : {args.epochs}")
    print(f"Image Size     : {args.imgsz}")
    print(f"Batch Size     : {args.batch}")
    print(f"Workers        : {args.workers}")
    print(f"Device         : {args.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() and args.device != 'cpu' else 'CPU'})")
    print("=" * 60)

    # Check data.yaml existence
    yaml_path = Path(args.data)
    if not yaml_path.exists():
        print(f"\n❌ Error: Dataset file '{yaml_path}' not found!")
        print("Please download and extract the Roboflow dataset into the 'data' folder.")
        print("Expected structure: data/data.yaml, data/train, data/valid, data/test")
        sys.exit(1)

    # Initialize YOLO Model
    print(f"\n📦 Loading model {args.model}...")
    model = YOLO(args.model)

    # Train model with PCB defect tuned augmentations
    print("\n🔥 Starting Training...")
    results = model.train(
        data=str(yaml_path.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=True,
        # Optimizer and scheduler settings
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        # Augmentation hyperparameters optimized for PCB micro-defects
        fliplr=0.5,
        flipud=0.5,
        degrees=10.0,
        scale=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        close_mosaic=10,  # Turn off mosaic for final 10 epochs for fine-grained boundary convergence
        # Performance
        amp=True,  # Automatic Mixed Precision for fast training & memory efficiency
        plots=False,  # Avoid matplotlib thread crash on Windows
        verbose=True,
    )

    print("\n✅ Training Complete!")
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(model.trainer.save_dir)
    best_weight = save_dir / "weights" / "best.pt"

    if best_weight.exists():
        print(f"\n🏆 Best model weights saved at: {best_weight.resolve()}")
        os.makedirs("models", exist_ok=True)
        shutil.copy(best_weight, "models/best.pt")
        print("📁 Copied to 'models/best.pt' for final submission.")
    else:
        last_weight = save_dir / "weights" / "last.pt"
        if last_weight.exists():
            os.makedirs("models", exist_ok=True)
            shutil.copy(last_weight, "models/best.pt")
            print(f"📁 Copied {last_weight} to 'models/best.pt'")

    # Run validation to output primary metric (mAP50-95) and recall
    print("\n📊 Running final validation evaluation on validation set...")
    val_model = YOLO("models/best.pt" if Path("models/best.pt").exists() else str(best_weight))
    metrics = val_model.val(data=str(yaml_path.resolve()), imgsz=args.imgsz, device=args.device, workers=args.workers, plots=False)

    print("\n" + "=" * 60)
    print("📈 FINAL VALIDATION RESULTS:")
    print(f"Primary Metric (mAP@50-95) : {metrics.box.map:.4f}")
    print(f"Tie-Breaker 1  (mAP@50)    : {metrics.box.map50:.4f}")
    print(f"Tie-Breaker 2  (Mean Recall): {metrics.box.mr:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
