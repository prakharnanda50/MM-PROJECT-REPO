"""
eval_pipeline.py - Pipeline to evaluate YOLO models on split files using Macro F1
"""

import os
from PIL import Image
import torch
from ultralytics import YOLO
from tqdm import tqdm
from macro_f1_evaluator import evaluate_detection_f1, plot_and_save_confusion_matrix, CLASS_NAMES


def parse_yolo_labels(lbl_path, img_width, img_height):
    """
    Parse YOLO normalized bounding box labels into absolute pixel coordinates [x1, y1, x2, y2].
    """
    boxes = []
    if os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    x1 = (cx - w / 2.0) * img_width
                    y1 = (cy - h / 2.0) * img_height
                    x2 = (cx + w / 2.0) * img_width
                    y2 = (cy + h / 2.0) * img_height
                    boxes.append({
                        "class": cls_id,
                        "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                    })
    return boxes


def run_model_evaluation(model, split_file, thresholds=None, imgsz=640, device="0", tune_thresholds=True):
    """
    Evaluates model on split_file images.
    If tune_thresholds=True and thresholds is None, finds optimal per-class thresholds.
    If thresholds is provided, evaluates with those frozen thresholds.
    """
    with open(split_file, "r") as f:
        image_paths = [line.strip() for line in f if line.strip()]

    all_gt = {}
    all_preds = {}

    print(f"Running predictions on {len(image_paths)} images from {split_file}...")
    for img_p in tqdm(image_paths, desc="Predicting for F1 Eval"):
        if not os.path.exists(img_p):
            continue

        with Image.open(img_p) as im:
            w, h = im.size

        # Find label file
        # If image is in data/<split>/images/xxx.jpg, label is data/<split>/labels/xxx.txt
        img_dir = os.path.dirname(img_p)
        base_split = os.path.dirname(img_dir)
        fname = os.path.basename(img_p)
        lbl_p = os.path.join(base_split, "labels", os.path.splitext(fname)[0] + ".txt")

        gt_boxes = parse_yolo_labels(lbl_p, w, h)
        all_gt[fname] = gt_boxes

        # Predict with low confidence threshold to capture candidates for threshold sweep
        results = model.predict(source=img_p, conf=0.01, imgsz=imgsz, device=device, verbose=False)
        pred_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy_list = boxes.xyxy.cpu().numpy()
            conf_list = boxes.conf.cpu().numpy()
            cls_list = boxes.cls.cpu().numpy().astype(int)

            for b, c, cl in zip(xyxy_list, conf_list, cls_list):
                pred_boxes.append({
                    "class": int(cl),
                    "box": [round(float(b[0]), 2), round(float(b[1]), 2), round(float(b[2]), 2), round(float(b[3]), 2)],
                    "conf": float(c)
                })
        all_preds[fname] = pred_boxes

    # Compute Macro F1
    results = evaluate_detection_f1(all_gt, all_preds, thresholds=thresholds, tune_thresholds=tune_thresholds)
    return results
