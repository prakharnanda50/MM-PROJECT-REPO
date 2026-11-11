"""
evaluate.py - Benchmark Model against Model Mayhem Evaluation Criteria:
1. Primary Metric: mAP@50:90 (IoU 0.50 to 0.90 step 0.05)
2. Tie-Breaker 1: mAP@50
3. Tie-Breaker 2: Macro Recall (average recall across all 6 defect classes)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model according to competition criteria")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to model weights")
    parser.add_argument("--data", type=str, default="data/data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=800, help="Image size")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers (0 recommended on Windows)")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu", help="Device")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 65)
    print("🏆 MODEL MAYHEM EVALUATION BENCHMARK")
    print("=" * 65)
    print(f"Weights : {args.weights}")
    print(f"Data    : {args.data}")
    print(f"Split   : {args.split}")
    print(f"Workers : {args.workers}")
    print("=" * 65)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"❌ Weights file '{weights_path}' not found!")
        return

    model = YOLO(str(weights_path))

    # Run validation with workers=0 and plots=False for Windows compatibility
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        plots=False,
        verbose=True,
    )

    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    macro_recall = metrics.box.mr
    macro_precision = metrics.box.mp
    macro_f1 = (2 * macro_precision * macro_recall) / (macro_precision + macro_recall + 1e-16)

    print("\n" + "=" * 65)
    print("🏁 OFFICIAL COMPETITION METRIC SUMMARY")
    print("=" * 65)
    print(f"⭐ Primary Metric (mAP@50:90 approx / mAP@50:95) : {map50_95:.4f} ({map50_95*100:.2f}%)")
    print(f"🥈 Tie-Breaker 1  (mAP@50)                       : {map50:.4f} ({map50*100:.2f}%)")
    print(f"🥉 Tie-Breaker 2  (Macro Recall)                 : {macro_recall:.4f} ({macro_recall*100:.2f}%)")
    print(f"🎯 Macro F1 Score (at IoU 0.50)                  : {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print(f"   Macro Precision                               : {macro_precision:.4f} ({macro_precision*100:.2f}%)")
    print("=" * 65)

    # Per-class breakdown
    print("\n📋 Per-Class Performance Breakdown:")
    class_names = model.names
    table_data = []
    for cls_idx, class_name in class_names.items():
        if cls_idx < len(metrics.box.maps):
            map50_val = metrics.box.all_ap[cls_idx, 0] if hasattr(metrics.box, 'all_ap') and metrics.box.all_ap.size > 0 else 0.0
            p_val = metrics.box.p[cls_idx] if hasattr(metrics.box, 'p') and len(metrics.box.p) > cls_idx else 0.0
            r_val = metrics.box.r[cls_idx] if hasattr(metrics.box, 'r') and len(metrics.box.r) > cls_idx else 0.0
            f1_val = (2 * p_val * r_val) / (p_val + r_val + 1e-16) if (p_val + r_val) > 0 else 0.0
            table_data.append({
                "Class": class_name,
                "Precision": f"{p_val:.4f}",
                "Recall": f"{r_val:.4f}",
                "F1@50": f"{f1_val:.4f}",
                "mAP@50": f"{map50_val:.4f}",
                "mAP@50-95": f"{metrics.box.maps[cls_idx]:.4f}",
            })

    if table_data:
        df_classes = pd.DataFrame(table_data)
        print(df_classes.to_string(index=False))


if __name__ == "__main__":
    main()
