"""
download_dataset.py - Helper script to setup the Roboflow PCB Defect Dataset

Dataset Link:
https://universe.roboflow.com/uthm-project-visionbased-pcb-defect-detection/pcb-defect-detection-9ewqw-duvuw/dataset/2
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path


def extract_zip(zip_path: Path, target_dir: Path):
    print(f"📦 Extracting {zip_path} into {target_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print("✅ Extraction complete!")


def main():
    parser = argparse.ArgumentParser(description="Download or unpack Roboflow PCB dataset")
    parser.add_argument("--api-key", type=str, default=None, help="Roboflow API key (optional)")
    parser.add_argument("--zip-file", type=str, default=None, help="Path to downloaded dataset zip file")
    args = parser.parse_args()

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.zip_file:
        zip_path = Path(args.zip_file)
        if not zip_path.exists():
            print(f"❌ Zip file '{zip_path}' not found!")
            return
        extract_zip(zip_path, data_dir)
        return

    # Check if a zip file is already in the workspace or Downloads folder
    local_zips = list(Path(".").glob("*.zip"))
    downloads_zips = list(Path.home().glob("Downloads/*pcb*.zip")) + list(Path.home().glob("Downloads/*defect*.zip"))

    candidate_zips = local_zips + downloads_zips
    if candidate_zips:
        selected_zip = candidate_zips[0]
        print(f"🔍 Found candidate dataset zip file: {selected_zip}")
        extract_zip(selected_zip, data_dir)
        return

    if args.api_key:
        print("🌐 Downloading dataset via Roboflow API...")
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=args.api_key)
            project = rf.workspace("uthm-project-visionbased-pcb-defect-detection").project("pcb-defect-detection-9ewqw-duvuw")
            version = project.version(2)
            dataset = version.download("yolov8", location=str(data_dir.resolve()))
            print(f"✅ Downloaded to: {dataset.location}")
            return
        except Exception as e:
            print(f"❌ Roboflow download failed: {e}")

    print("\nℹ️  To prepare the dataset:")
    print("1. Go to: https://universe.roboflow.com/uthm-project-visionbased-pcb-defect-detection/pcb-defect-detection-9ewqw-duvuw/dataset/2")
    print("2. Click 'Download Dataset'")
    print("3. Choose Format: 'YOLOv8'")
    print("4. Either:")
    print("   a) Download the zip file and run:")
    print("      python download_dataset.py --zip-file <path_to_zip>")
    print("   b) Or copy your Roboflow API key from the download snippet and run:")
    print("      python download_dataset.py --api-key YOUR_API_KEY")


if __name__ == "__main__":
    main()
