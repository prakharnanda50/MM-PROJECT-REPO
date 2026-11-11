"""
inference.py - Generate predictions CSV for Model Mayhem Challenge
Prescribed CSV format:
image_name,class_name,x_min,y_min,x_max,y_max,confidence
"""

import argparse
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from ultralytics import YOLO
from tqdm import tqdm


# Prescribed target class labels
TARGET_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PCB Defect Detection Predictions CSV")
    parser.add_argument("--source", type=str, default="data/test/images", help="Path to test images folder or single image")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to trained model weights (.pt)")
    parser.add_argument("--output", type=str, default="output/predictions.csv", help="Output path for submission CSV")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=800, help="Inference image resolution")
    parser.add_argument("--save-visuals", action="store_true", help="Save annotated visualization images")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu", help="Device (0 or cpu)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("🔍 MODEL MAYHEM: PCB DEFECT DETECTION - INFERENCE PIPELINE")
    print("=" * 60)
    print(f"Source Images  : {args.source}")
    print(f"Model Weights  : {args.weights}")
    print(f"Output CSV     : {args.output}")
    print(f"Confidence     : {args.conf}")
    print(f"Device         : {args.device}")
    print("=" * 60)

    # Validate weights path
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"\n❌ Error: Weights file '{weights_path}' does not exist!")
        print("Please train the model first using train.py or place best.pt into 'models/'.")
        return

    # Validate source path
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"\n❌ Error: Source path '{source_path}' does not exist!")
        return

    # Collect images
    if source_path.is_file():
        image_files = [source_path]
    else:
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        image_files = [
            p for p in source_path.iterdir()
            if p.suffix.lower() in valid_extensions
        ]

    if not image_files:
        print(f"\n❌ No valid image files found in '{source_path}'")
        return

    print(f"Found {len(image_files)} image(s) for inference.")

    # Load YOLO model
    print(f"Loading weights from {weights_path}...")
    model = YOLO(str(weights_path))

    # Check class name mapping
    model_names = model.names
    print(f"Model detected classes: {model_names}")

    # Create output directories
    output_csv_path = Path(args.output)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    visuals_dir = output_csv_path.parent / "visual_predictions"
    if args.save_visuals:
        visuals_dir.mkdir(parents=True, exist_ok=True)

    records = []

    print("\n🚀 Running predictions...")
    for img_path in tqdm(image_files, desc="Processing images"):
        # Run inference
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )

        result = results[0]

        # Save annotated image if requested
        if args.save_visuals:
            save_path = visuals_dir / img_path.name
            result.save(filename=str(save_path))

        # Extract bounding boxes
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            # xyxy contains coordinates in original image pixel space [x1, y1, x2, y2]
            xyxy_list = boxes.xyxy.cpu().numpy()
            conf_list = boxes.conf.cpu().numpy()
            cls_list = boxes.cls.cpu().numpy().astype(int)

            for xyxy, conf, cls_id in zip(xyxy_list, conf_list, cls_list):
                x_min, y_min, x_max, y_max = xyxy
                class_name = model_names.get(cls_id, f"class_{cls_id}")

                # Ensure class name matches standard target format
                formatted_class = class_name.strip().lower().replace(" ", "_").replace("-", "_")

                records.append({
                    "image_name": img_path.name,
                    "class_name": formatted_class,
                    "x_min": round(float(x_min), 2),
                    "y_min": round(float(y_min), 2),
                    "x_max": round(float(x_max), 2),
                    "y_max": round(float(y_max), 2),
                    "confidence": round(float(conf), 4),
                })

    # Build DataFrame
    df = pd.DataFrame(records, columns=[
        "image_name",
        "class_name",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "confidence"
    ])

    # Save CSV
    df.to_csv(output_csv_path, index=False)
    print(f"\n✅ Predictions successfully written to: {output_csv_path.resolve()}")
    print(f"Total detections generated: {len(df)}")
    print("\nPreview of top 5 predictions:")
    print(df.head(5).to_string())

    if args.save_visuals:
        print(f"\n🖼️ Annotated visuals saved to: {visuals_dir.resolve()}")


if __name__ == "__main__":
    main()
