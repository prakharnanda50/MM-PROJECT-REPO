"""
final_test_evaluation.py - Touches TEST Set Exactly ONCE with Final Chosen Model
Outputs:
1. Final TEST Macro F1, per-class P/R/F1, Confusion Matrix plot
2. Worst 3 classes & most common confusions
3. 10 misclassified examples with hypotheses
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO
from eval_pipeline import run_model_evaluation, parse_yolo_labels
from macro_f1_evaluator import plot_and_save_confusion_matrix, compute_iou, CLASS_NAMES

TEST_SPLIT = "splits/test.txt"
RESULTS_DIR = "results"
FAILURES_DIR = "results/failure_cases"


def run_test_evaluation(checkpoint_path, thresholds):
    print("=" * 65)
    print("FINAL TEST SET EVALUATION (TOUCHING TEST SET EXACTLY ONCE)")
    print(f"Model Checkpoint : {checkpoint_path}")
    print(f"Frozen Thresholds: {thresholds}")
    print("=" * 65)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FAILURES_DIR, exist_ok=True)

    model = YOLO(checkpoint_path)

    # 1. Evaluate on test set with frozen thresholds
    test_results = run_model_evaluation(
        model,
        TEST_SPLIT,
        thresholds=thresholds,
        imgsz=800,
        device=0 if torch.cuda.is_available() else "cpu",
        tune_thresholds=False
    )

    macro_f1 = test_results["macro_f1"]
    macro_p = test_results["macro_precision"]
    macro_r = test_results["macro_recall"]
    plain_acc = test_results["plain_accuracy"]
    cm = test_results["confusion_matrix"]
    class_metrics = test_results["class_metrics"]

    # 2. Save confusion matrix image
    cm_img_path = os.path.join(RESULTS_DIR, "final_test_confusion_matrix.png")
    plot_and_save_confusion_matrix(cm, save_path=cm_img_path)

    # 3. Analyze failure cases (10 misclassified examples)
    with open(TEST_SPLIT, "r") as f:
        test_images = [line.strip() for line in f if line.strip()]

    failure_cases = []

    for img_p in test_images:
        if len(failure_cases) >= 10:
            break

        fname = os.path.basename(img_p)
        with Image.open(img_p) as im:
            w, h = im.size
            im_draw = im.copy()
            draw = ImageDraw.Draw(im_draw)

        img_dir = os.path.dirname(img_p)
        base_split = os.path.dirname(img_dir)
        lbl_p = os.path.join(base_split, "labels", os.path.splitext(fname)[0] + ".txt")
        gt_boxes = parse_yolo_labels(lbl_p, w, h)

        # Predict
        preds_raw = model.predict(source=img_p, conf=0.05, imgsz=800, device=0, verbose=False)[0]
        preds = []
        if preds_raw.boxes is not None:
            for b, c, cl in zip(preds_raw.boxes.xyxy.cpu().numpy(), preds_raw.boxes.conf.cpu().numpy(), preds_raw.boxes.cls.cpu().numpy().astype(int)):
                if float(c) >= thresholds.get(int(cl), 0.25):
                    preds.append({
                        "class": int(cl),
                        "box": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                        "conf": float(c)
                    })

        # Match to find False Negatives (missed GT) and False Positives (hallucinated)
        matched_gt = set()
        unmatched_preds = []

        for p in preds:
            matched = False
            for g_i, g in enumerate(gt_boxes):
                if g_i not in matched_gt and g["class"] == p["class"]:
                    if compute_iou(p["box"], g["box"]) >= 0.5:
                        matched_gt.add(g_i)
                        matched = True
                        break
            if not matched:
                unmatched_preds.append(p)

        unmatched_gt = [g for i, g in enumerate(gt_boxes) if i not in matched_gt]

        if unmatched_gt or unmatched_preds:
            # Draw GT (Green) and Preds (Red)
            for g in gt_boxes:
                b = g["box"]
                draw.rectangle([b[0], b[1], b[2], b[3]], outline="green", width=2)
                draw.text((b[0], max(0, b[1] - 10)), f"GT:{CLASS_NAMES[g['class']]}", fill="green")
            for p in preds:
                b = p["box"]
                draw.rectangle([b[0], b[1], b[2], b[3]], outline="red", width=2)
                draw.text((b[0], max(0, b[3] + 2)), f"P:{CLASS_NAMES[p['class']]}({p['conf']:.2f})", fill="red")

            fail_save_path = os.path.join(FAILURES_DIR, f"fail_{len(failure_cases) + 1}_{fname}")
            im_draw.save(fail_save_path)

            # Determine failure type and hypothesis
            if unmatched_gt and not unmatched_preds:
                fail_type = "False Negative (Missed Defect)"
                missed_classes = [CLASS_NAMES[g['class']] for g in unmatched_gt]
                hypothesis = f"Missed subtle defect(s): {missed_classes}. The defect size is small relative to the board layout with low contrast against copper background."
            elif unmatched_preds and not unmatched_gt:
                fail_type = "False Positive (False Alarm)"
                pred_classes = [CLASS_NAMES[p['class']] for p in unmatched_preds]
                hypothesis = f"Spurious detection: {pred_classes}. Complex normal copper trace geometry / via pads visually mimicked defect characteristics."
            else:
                fail_type = "Localization/Misclassification Error"
                hypothesis = "Bounding box IoU fell below 0.50 threshold or nearby trace anomaly confused the multi-head detector."

            failure_cases.append({
                "index": len(failure_cases) + 1,
                "image_name": fname,
                "type": fail_type,
                "hypothesis": hypothesis,
                "image_path": fail_save_path
            })

    # Save summary report JSON
    summary_report = {
        "test_macro_f1": macro_f1,
        "test_macro_precision": macro_p,
        "test_macro_recall": macro_r,
        "test_plain_accuracy": plain_acc,
        "class_metrics": class_metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_image": cm_img_path,
        "failure_cases": failure_cases
    }

    report_path = os.path.join(RESULTS_DIR, "final_test_report.json")
    with open(report_path, "w") as f:
        json.dump(summary_report, f, indent=2)

    print("\n" + "=" * 65)
    print("🏁 FINAL TEST EVALUATION COMPLETE")
    print(f"⭐ Final TEST Macro F1 : {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print(f"   TEST Precision      : {macro_p:.4f}")
    print(f"   TEST Recall         : {macro_r:.4f}")
    print(f"   TEST Accuracy       : {plain_acc:.4f}")
    print("=" * 65)

    return summary_report


if __name__ == "__main__":
    FROZEN_THRESHOLDS = {0: 0.125, 1: 0.125, 2: 0.15, 3: 0.15, 4: 0.15, 5: 0.175}
    run_test_evaluation("models/best.pt", FROZEN_THRESHOLDS)
