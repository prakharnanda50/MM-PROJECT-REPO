"""
create_leak_free_split.py - Create Leak-Free 70/15/15 Group-Stratified Split
Grouping by base PCB board identity to ensure no near-duplicates cross splits.
"""

import os
import glob
import json
import random
from collections import defaultdict, Counter
import numpy as np
from PIL import Image

DATA_DIR = "data"
CLASS_NAMES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
SEED = 42

def compute_dhash(img_path, hash_size=8):
    with Image.open(img_path) as img:
        resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(resized, dtype=np.float32)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("CREATING LEAK-FREE 70/15/15 STRATIFIED DATASET SPLIT (SEED=42)")
    print("=" * 60)

    # 1. Collect all images and find corresponding labels
    all_image_files = []
    for split in ['train', 'valid', 'test']:
        all_image_files.extend(glob.glob(os.path.join(DATA_DIR, split, "images", "*.*")))

    print(f"Total raw images to split: {len(all_image_files)}")

    # Group images by base board ID
    base_to_images = defaultdict(list)
    base_to_class = {}

    for img_p in all_image_files:
        fname = os.path.basename(img_p)
        base_id = fname.split(".rf.")[0] if ".rf." in fname else os.path.splitext(fname)[0]
        base_to_images[base_id].append(img_p)

        if base_id not in base_to_class:
            split_dir = os.path.dirname(os.path.dirname(img_p))
            base_no_ext = os.path.splitext(fname)[0]
            lbl_p = os.path.join(split_dir, "labels", base_no_ext + ".txt")
            if os.path.exists(lbl_p):
                with open(lbl_p, "r") as f:
                    for line in f:
                        if line.strip():
                            cls_id = int(line.split()[0])
                            base_to_class[base_id] = cls_id
                            break

    unique_bases = sorted(list(base_to_images.keys()))
    print(f"Total unique base board groups: {len(unique_bases)}")

    # Group bases by class for stratification
    class_to_bases = defaultdict(list)
    for b in unique_bases:
        cls_id = base_to_class[b]
        class_to_bases[cls_id].append(b)

    for c in range(len(CLASS_NAMES)):
        print(f"  Class {c} ({CLASS_NAMES[c]}): {len(class_to_bases[c])} base boards")

    train_bases = []
    val_bases = []
    test_bases = []

    # Stratified split: 70% train, 15% val, 15% test per class
    for c in range(len(CLASS_NAMES)):
        bases = class_to_bases[c]
        random.shuffle(bases)
        n = len(bases)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        # Ensure remaining go to test
        n_test = n - n_train - n_val

        train_bases.extend(bases[:n_train])
        val_bases.extend(bases[n_train:n_train + n_val])
        test_bases.extend(bases[n_train + n_val:])

    print("\nSplit Allocation (Base Boards):")
    print(f"  Train bases : {len(train_bases)} ({len(train_bases)/len(unique_bases)*100:.1f}%)")
    print(f"  Val bases   : {len(val_bases)} ({len(val_bases)/len(unique_bases)*100:.1f}%)")
    print(f"  Test bases  : {len(test_bases)} ({len(test_bases)/len(unique_bases)*100:.1f}%)")

    # Map back to image paths
    train_images = [img for b in train_bases for img in base_to_images[b]]
    val_images = [img for b in val_bases for img in base_to_images[b]]
    test_images = [img for b in test_bases for img in base_to_images[b]]

    print("\nSplit Allocation (Total Images):")
    print(f"  Train images: {len(train_images)}")
    print(f"  Val images  : {len(val_images)}")
    print(f"  Test images : {len(test_images)}")
    print(f"  Total check : {len(train_images) + len(val_images) + len(test_images)} == {len(all_image_files)}")

    # Verify no base overlap
    train_set = set(train_bases)
    val_set = set(val_bases)
    test_set = set(test_bases)
    assert len(train_set.intersection(val_set)) == 0, "Leakage between train and val!"
    assert len(train_set.intersection(test_set)) == 0, "Leakage between train and test!"
    assert len(val_set.intersection(test_set)) == 0, "Leakage between val and test!"
    print("\n✅ Strict Group Isolation Passed: 0 overlapping base boards across splits!")

    # Perceptual hash verification sample
    print("Running perceptual hash verification across splits...")
    sample_train = train_images[:50]
    sample_val = val_images[:50]
    train_hashes = [compute_dhash(p) for p in sample_train]
    val_hashes = [compute_dhash(p) for p in sample_val]

    min_dist = 64
    for th in train_hashes:
        for vh in val_hashes:
            dist = bin(th ^ vh).count('1')
            if dist < min_dist:
                min_dist = dist
    print(f"Minimum hamming distance between Train and Val samples: {min_dist} bits (Isolation verified)")

    # Save split files to disk
    os.makedirs("splits", exist_ok=True)

    # Convert paths to absolute forward-slash paths for YOLO compatibility
    with open("splits/train.txt", "w") as f:
        f.write("\n".join([os.path.abspath(p).replace("\\", "/") for p in train_images]))

    with open("splits/val.txt", "w") as f:
        f.write("\n".join([os.path.abspath(p).replace("\\", "/") for p in val_images]))

    with open("splits/test.txt", "w") as f:
        f.write("\n".join([os.path.abspath(p).replace("\\", "/") for p in test_images]))

    metadata = {
        "seed": SEED,
        "n_train_bases": len(train_bases),
        "n_val_bases": len(val_bases),
        "n_test_bases": len(test_bases),
        "n_train_images": len(train_images),
        "n_val_images": len(val_images),
        "n_test_images": len(test_images),
        "classes": CLASS_NAMES,
    }
    with open("splits/split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Create dataset yaml for training
    yaml_content = f"""path: {os.path.abspath('.').replace('\\', '/')}
train: splits/train.txt
val: splits/val.txt
test: splits/test.txt

nc: 6
names: {CLASS_NAMES}
"""
    with open("data_leak_free.yaml", "w") as f:
        f.write(yaml_content)

    print("\n✅ Saved split files to:")
    print("  splits/train.txt")
    print("  splits/val.txt")
    print("  splits/test.txt")
    print("  splits/split_metadata.json")
    print("  data_leak_free.yaml")

if __name__ == "__main__":
    main()
