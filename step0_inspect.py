"""
step0_inspect.py - Comprehensive inspection of PCB Defect Dataset
"""

import os
import glob
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
import cv2

DATA_DIR = "data"
CLASS_NAMES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

def compute_dhash(image, hash_size=8):
    # Resize to (hash_size + 1, hash_size) and grayscale
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(resized, dtype=np.float32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    # Convert bool array to integer bitstring
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def hamming_distance(h1, h2):
    return bin(h1 ^ h2).count('1')

def main():
    print("=" * 60)
    print("STEP 0: COMPREHENSIVE DATASET INSPECTION")
    print("=" * 60)

    all_images = []
    all_labels = []

    for split in ['train', 'valid', 'test']:
        img_paths = glob.glob(os.path.join(DATA_DIR, split, "images", "*.*"))
        lbl_paths = glob.glob(os.path.join(DATA_DIR, split, "labels", "*.*"))
        all_images.extend(img_paths)
        all_labels.extend(lbl_paths)

    print(f"Total image files found: {len(all_images)}")
    print(f"Total label files found: {len(all_labels)}")

    # 1. Check for corrupt images & image resolutions
    resolutions = Counter()
    corrupt_images = []
    hashes = {}
    base_name_groups = defaultdict(list)
    image_metadata = {}

    for img_p in all_images:
        fname = os.path.basename(img_p)
        try:
            with Image.open(img_p) as img:
                w, h = img.size
                resolutions[(w, h)] += 1
                # Base name extraction before .rf.
                # e.g., '01_missing_hole_01_jpg.rf.0b8ba0226d351efd9727e8ae7f1f54bc.jpg' -> '01_missing_hole_01_jpg'
                if ".rf." in fname:
                    base_id = fname.split(".rf.")[0]
                else:
                    base_id = os.path.splitext(fname)[0]
                
                h_val = compute_dhash(img)
                hashes[fname] = h_val
                base_name_groups[base_id].append(img_p)
                image_metadata[fname] = {
                    "path": img_p,
                    "base_id": base_id,
                    "resolution": (w, h),
                    "hash": h_val
                }
        except Exception as e:
            corrupt_images.append((img_p, str(e)))

    print(f"Corrupt images: {len(corrupt_images)}")
    print(f"Image resolutions distribution: {dict(resolutions)}")
    print(f"Unique base parent image groups (by base filename): {len(base_name_groups)}")

    # 2. Annotation & Task Type Inspection
    # Check label contents
    box_counts = Counter()
    images_per_class = Counter()
    annotation_samples = []
    corrupt_labels = []
    empty_label_files = 0
    total_boxes = 0

    for img_p in all_images:
        fname = os.path.basename(img_p)
        base_no_ext = os.path.splitext(fname)[0]
        # Locate corresponding label
        # Label path is in same split but 'labels' dir
        split_dir = os.path.dirname(os.path.dirname(img_p))
        lbl_p = os.path.join(split_dir, "labels", base_no_ext + ".txt")

        classes_in_image = set()
        if os.path.exists(lbl_p):
            try:
                with open(lbl_p, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]
                if not lines:
                    empty_label_files += 1
                else:
                    if len(annotation_samples) < 3:
                        annotation_samples.append((fname, lines[:2]))
                    for line in lines:
                        parts = line.split()
                        cls_id = int(parts[0])
                        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"unknown_{cls_id}"
                        box_counts[cls_name] += 1
                        classes_in_image.add(cls_name)
                        total_boxes += 1
            except Exception as e:
                corrupt_labels.append((lbl_p, str(e)))
        else:
            empty_label_files += 1

        for c in classes_in_image:
            images_per_class[c] += 1

    print("\n--- TASK TYPE & ANNOTATIONS ---")
    print("Task Type: Object Detection (Bounding Box Localisation and Classification)")
    print("Annotation Format: Normalized YOLO format (class_idx center_x center_y width height)")
    print(f"Sample annotations:")
    for smp in annotation_samples:
        print(f"  {smp[0]}: {smp[1]}")
    print(f"Total bounding boxes across dataset: {total_boxes}")
    print("\nPer-Class Bounding Box Counts:")
    for c in CLASS_NAMES:
        print(f"  {c:<18}: {box_counts[c]:>5} boxes (appears in {images_per_class[c]:>5} images)")

    # 3. Leakage Analysis between Roboflow splits
    # Check if images from the same base board exist across train, valid, test
    split_bases = defaultdict(set)
    for img_p in all_images:
        fname = os.path.basename(img_p)
        split = img_p.split(os.sep)[1] if os.sep in img_p else img_p.split("/")[1]
        base_id = image_metadata[fname]["base_id"]
        split_bases[split].add(base_id)

    train_val_overlap = split_bases["train"].intersection(split_bases["valid"])
    train_test_overlap = split_bases["train"].intersection(split_bases["test"])
    val_test_overlap = split_bases["valid"].intersection(split_bases["test"])

    print("\n--- DATA LEAKAGE IN ROBOFLOW DEFAULT SPLIT ---")
    print(f"Unique base boards in train: {len(split_bases['train'])}")
    print(f"Unique base boards in valid: {len(split_bases['valid'])}")
    print(f"Unique base boards in test : {len(split_bases['test'])}")
    print(f"Base boards overlapping Train & Valid: {len(train_val_overlap)}")
    print(f"Base boards overlapping Train & Test : {len(train_test_overlap)}")
    print(f"Base boards overlapping Valid & Test : {len(val_test_overlap)}")

if __name__ == "__main__":
    main()
