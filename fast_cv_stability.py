"""
fast_cv_stability.py - Measure variance across 5 group-stratified folds
"""
import os
import json
import random
from collections import defaultdict
import numpy as np
import torch
from ultralytics import YOLO
from eval_pipeline import run_model_evaluation
from macro_f1_evaluator import CLASS_NAMES

def evaluate_folds(checkpoint="models/best.pt"):
    with open("splits/train.txt") as f:
        train_imgs = [line.strip() for line in f if line.strip()]
    with open("splits/val.txt") as f:
        val_imgs = [line.strip() for line in f if line.strip()]

    combined = train_imgs + val_imgs
    base_to_images = defaultdict(list)
    base_to_class = {}

    for img_p in combined:
        fname = os.path.basename(img_p)
        base_id = fname.split(".rf.")[0] if ".rf." in fname else os.path.splitext(fname)[0]
        base_to_images[base_id].append(img_p)

        if base_id not in base_to_class:
            img_dir = os.path.dirname(img_p)
            base_split = os.path.dirname(img_dir)
            lbl_p = os.path.join(base_split, "labels", os.path.splitext(fname)[0] + ".txt")
            if os.path.exists(lbl_p):
                with open(lbl_p) as f:
                    for line in f:
                        if line.strip():
                            base_to_class[base_id] = int(line.split()[0])
                            break

    class_to_bases = defaultdict(list)
    for b, c in base_to_class.items():
        class_to_bases[c].append(b)

    random.seed(42)
    folds_bases = [[] for _ in range(5)]
    for c in range(len(CLASS_NAMES)):
        bases = class_to_bases[c]
        random.shuffle(bases)
        for i, b in enumerate(bases):
            folds_bases[i % 5].append(b)

    fold_scores = []
    os.makedirs("splits/cv", exist_ok=True)
    frozen_thresholds = {0: 0.125, 1: 0.125, 2: 0.15, 3: 0.15, 4: 0.15, 5: 0.175}
    model = YOLO(checkpoint)

    for fold_idx in range(5):
        val_fold_bases = folds_bases[fold_idx]
        val_imgs = [img for b in val_fold_bases for img in base_to_images[b]]
        val_file = f"splits/cv/val_fold_{fold_idx}.txt"
        with open(val_file, "w") as f:
            f.write("\n".join(val_imgs))

        print(f"Evaluating Fold {fold_idx + 1}/5 ({len(val_imgs)} images)...")
        res = run_model_evaluation(
            model,
            val_file,
            thresholds=frozen_thresholds,
            imgsz=800,
            device=0 if torch.cuda.is_available() else "cpu",
            tune_thresholds=False
        )
        fold_scores.append(res["macro_f1"])
        print(f"  Fold {fold_idx + 1} Macro F1: {res['macro_f1']:.4f}")

    mean_f1 = float(np.mean(fold_scores))
    std_f1 = float(np.std(fold_scores))

    print("\n" + "=" * 65)
    print("5-FOLD CV GROUP STABILITY RESULTS:")
    for i, s in enumerate(fold_scores):
        print(f"  Fold {i + 1}: Macro F1 = {s:.4f}")
    print(f"  Mean Macro F1 : {mean_f1:.4f}")
    print(f"  Std Dev (sigma): +/- {std_f1:.4f}")
    print("=" * 65)

    os.makedirs("results", exist_ok=True)
    with open("results/5fold_cv_results.json", "w") as f:
        json.dump({
            "fold_scores": fold_scores,
            "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1
        }, f, indent=2)

if __name__ == "__main__":
    evaluate_folds()
