"""
cv_evaluation.py - 5-Fold Cross Validation on Train+Val for final candidate configuration
Reports mean +/- std of Macro F1 across folds to confirm stability.
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

DATA_DIR = "data"
CV_RESULTS_FILE = "results/5fold_cv_results.json"


def get_train_val_base_groups():
    # Read train and val splits
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

    return base_to_images, base_to_class


def run_5fold_cv(model_name="yolo11s.pt", epochs=15, imgsz=800, batch=8, opt_params=None):
    print("=" * 65)
    print("RUNNING 5-FOLD CROSS VALIDATION (GROUP-STRATIFIED BY PARENT BOARD)")
    print("=" * 65)

    base_to_images, base_to_class = get_train_val_base_groups()

    # Stratify by class into 5 folds
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

    for fold_idx in range(5):
        val_fold_bases = folds_bases[fold_idx]
        train_fold_bases = [b for f_i in range(5) if f_i != fold_idx for b in folds_bases[f_i]]

        train_imgs = [img for b in train_fold_bases for img in base_to_images[b]]
        val_imgs = [img for b in val_fold_bases for img in base_to_images[b]]

        train_file = f"splits/cv/train_fold_{fold_idx}.txt"
        val_file = f"splits/cv/val_fold_{fold_idx}.txt"

        with open(train_file, "w") as f:
            f.write("\n".join(train_imgs))
        with open(val_file, "w") as f:
            f.write("\n".join(val_imgs))

        yaml_file = f"splits/cv/data_fold_{fold_idx}.yaml"
        with open(yaml_file, "w") as f:
            f.write(f"path: {os.path.abspath('.').replace(os.sep, '/')}\ntrain: {train_file}\nval: {val_file}\nnc: 6\nnames: {CLASS_NAMES}\n")

        print(f"\n--- FOLD {fold_idx + 1}/5 (Train: {len(train_imgs)}, Val: {len(val_imgs)}) ---")
        model = YOLO(model_name)
        res = model.train(
            data=yaml_file,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=0 if torch.cuda.is_available() else "cpu",
            workers=0,
            optimizer="AdamW",
            cos_lr=True,
            fliplr=0.5,
            flipud=0.5,
            degrees=90.0,
            mixup=0.1,
            copy_paste=0.1,
            project="runs/cv",
            name=f"fold_{fold_idx}",
            exist_ok=True,
            plots=False,
            amp=True,
            verbose=False
        )

        save_dir = res.save_dir
        best_pt = os.path.join(save_dir, "weights", "best.pt")
        if not os.path.exists(best_pt):
            best_pt = os.path.join(save_dir, "weights", "last.pt")

        val_res = run_model_evaluation(YOLO(best_pt), val_file, imgsz=imgsz, device=0, tune_thresholds=True)
        fold_scores.append(val_res["macro_f1"])
        print(f"Fold {fold_idx + 1} VAL Macro F1: {val_res['macro_f1']:.4f}")

    mean_f1 = float(np.mean(fold_scores))
    std_f1 = float(np.std(fold_scores))

    print("\n" + "=" * 65)
    print("5-FOLD CROSS VALIDATION SUMMARY:")
    for i, s in enumerate(fold_scores):
        print(f"  Fold {i + 1}: Macro F1 = {s:.4f}")
    print(f"  Mean Macro F1 : {mean_f1:.4f}")
    print(f"  Std Dev       : +/- {std_f1:.4f}")
    print("=" * 65)

    os.makedirs(os.path.dirname(CV_RESULTS_FILE), exist_ok=True)
    with open(CV_RESULTS_FILE, "w") as f:
        json.dump({
            "fold_scores": fold_scores,
            "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1
        }, f, indent=2)


if __name__ == "__main__":
    run_5fold_cv()
