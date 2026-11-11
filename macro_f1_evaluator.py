"""
macro_f1_evaluator.py - Official Macro F1 Evaluator for Object Detection
Per user specification:
- Macro F1 defined as the mean of per-class F1 at IoU 0.5.
- Match predictions to ground truth one-to-one, sorted by confidence.
- Thresholds selected on VALIDATION set to maximize each class's F1.
- Frozen thresholds applied to TEST set.
- Generates per-class P/R/F1, Confusion Matrix, and plain accuracy.
"""

import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CLASS_NAMES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
NUM_CLASSES = len(CLASS_NAMES)


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes in xyxy format [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def evaluate_detection_f1(all_gt, all_preds, thresholds=None, iou_thresh=0.5, tune_thresholds=True):
    """
    all_gt: dict of image_id -> list of {'class': int, 'box': [x1, y1, x2, y2]}
    all_preds: dict of image_id -> list of {'class': int, 'box': [x1, y1, x2, y2], 'conf': float}
    thresholds: dict of class_id -> float threshold (if None and tune_thresholds=True, will tune on input data)
    tune_thresholds: if True, grid-searches best threshold per class [0.05 ... 0.90] to maximize class F1
    """
    all_image_ids = sorted(list(set(list(all_gt.keys()) + list(all_preds.keys()))))

    if tune_thresholds or thresholds is None:
        best_thresholds = {}
        for c in range(NUM_CLASSES):
            best_f1 = -1.0
            best_t = 0.25
            # Search candidate thresholds
            for t in np.linspace(0.05, 0.85, 33):
                tp, fp, fn = 0, 0, 0
                for img_id in all_image_ids:
                    gt_boxes = [b['box'] for b in all_gt.get(img_id, []) if b['class'] == c]
                    preds = [p for p in all_preds.get(img_id, []) if p['class'] == c and p['conf'] >= t]
                    preds = sorted(preds, key=lambda x: x['conf'], reverse=True)

                    matched_gt = set()
                    for p in preds:
                        p_box = p['box']
                        best_iou = 0.0
                        best_gt_idx = -1
                        for g_idx, g_box in enumerate(gt_boxes):
                            if g_idx not in matched_gt:
                                iou = compute_iou(p_box, g_box)
                                if iou > best_iou:
                                    best_iou = iou
                                    best_gt_idx = g_idx
                        if best_iou >= iou_thresh:
                            matched_gt.add(best_gt_idx)
                            tp += 1
                        else:
                            fp += 1
                    fn += len(gt_boxes) - len(matched_gt)

                p_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_val = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0.0
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_t = float(t)
            best_thresholds[c] = round(best_t, 3)
    else:
        best_thresholds = {c: thresholds.get(c, 0.25) for c in range(NUM_CLASSES)}

    # Now run evaluation with best_thresholds
    class_metrics = {}
    # Confusion matrix: (NUM_CLASSES + 1) x (NUM_CLASSES + 1), where last row/col is background
    # rows: Ground Truth class (0..5, 6=background/FP)
    # cols: Predicted class (0..5, 6=background/FN)
    cm = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=int)

    total_tp, total_fp, total_fn = 0, 0, 0

    for c in range(NUM_CLASSES):
        t = best_thresholds[c]
        tp, fp, fn = 0, 0, 0

        for img_id in all_image_ids:
            gt_boxes = [b['box'] for b in all_gt.get(img_id, []) if b['class'] == c]
            preds = [p for p in all_preds.get(img_id, []) if p['class'] == c and p['conf'] >= t]
            preds = sorted(preds, key=lambda x: x['conf'], reverse=True)

            matched_gt = set()
            for p in preds:
                p_box = p['box']
                best_iou = 0.0
                best_gt_idx = -1
                for g_idx, g_box in enumerate(gt_boxes):
                    if g_idx not in matched_gt:
                        iou = compute_iou(p_box, g_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = g_idx
                if best_iou >= iou_thresh:
                    matched_gt.add(best_gt_idx)
                    tp += 1
                    cm[c, c] += 1
                else:
                    fp += 1
                    # Prediction on background
                    cm[NUM_CLASSES, c] += 1

            unmatched_fn = len(gt_boxes) - len(matched_gt)
            fn += unmatched_fn
            # Ground truth missed (predicted as background)
            cm[c, NUM_CLASSES] += unmatched_fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        class_metrics[c] = {
            "class_name": CLASS_NAMES[c],
            "threshold": t,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    macro_f1 = float(np.mean([class_metrics[c]["f1"] for c in range(NUM_CLASSES)]))
    macro_p = float(np.mean([class_metrics[c]["precision"] for c in range(NUM_CLASSES)]))
    macro_r = float(np.mean([class_metrics[c]["recall"] for c in range(NUM_CLASSES)]))
    plain_accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

    return {
        "macro_f1": macro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "plain_accuracy": plain_accuracy,
        "class_metrics": class_metrics,
        "thresholds": best_thresholds,
        "confusion_matrix": cm
    }


def plot_and_save_confusion_matrix(cm, save_path="output/confusion_matrix.png"):
    labels = CLASS_NAMES + ["background"]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel='True Label',
        xlabel='Predicted Label',
        title='Detection Confusion Matrix (IoU >= 0.50)'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"📊 Confusion matrix saved to: {save_path}")
