"""
deploy_to_roboflow.py - Upload & Deploy Custom Trained YOLO11 Weights to Roboflow
"""

import argparse
import os
import sys
from pathlib import Path
from roboflow import Roboflow


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy custom trained PCB defect model to Roboflow")
    parser.add_argument("--api-key", type=str, default=os.getenv("ROBOFLOW_API_KEY"), help="Roboflow Private API Key")
    parser.add_argument("--workspace", type=str, required=True, help="Your Roboflow Workspace ID")
    parser.add_argument("--project", type=str, required=True, help="Your Roboflow Project ID")
    parser.add_argument("--version", type=int, default=1, help="Dataset version number in your project")
    parser.add_argument("--model-type", type=str, default="yolov11n", help="Roboflow model architecture type (e.g. yolov11n)")
    parser.add_argument("--model-path", type=str, default="runs/detect/runs/experiments/EXP02_Res800", help="Directory containing model weights folder")
    parser.add_argument("--filename", type=str, default="weights/best.pt", help="Relative path to best.pt from model-path")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("🚀 ROBOFLOW CUSTOM MODEL DEPLOYMENT")
    print("=" * 65)
    print(f"Workspace  : {args.workspace}")
    print(f"Project    : {args.project}")
    print(f"Version    : {args.version}")
    print(f"Model Type : {args.model-type if hasattr(args, 'model-type') else args.model_type}")
    print(f"Model Path : {args.model_path}")
    print("=" * 65)

    if not args.api_key:
        print("\n❌ Error: Missing Roboflow API Key!")
        print("Please provide it via --api-key <YOUR_KEY> or set the ROBOFLOW_API_KEY environment variable.")
        print("You can get your private API key from: https://app.roboflow.com/settings/api")
        sys.exit(1)

    # Verify model weights file
    weights_full_path = Path(args.model_path) / args.filename
    if not weights_full_path.exists():
        # Fallback check if models/best.pt exists
        if Path("models/best.pt").exists():
            print(f"Notice: '{weights_full_path}' not found, falling back to 'models/best.pt'...")
            # Create a temporary directory structure for Roboflow packaging
            deploy_dir = Path("runs/deploy_temp")
            (deploy_dir / "weights").mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy("models/best.pt", deploy_dir / "weights" / "best.pt")
            args.model_path = str(deploy_dir)
            args.filename = "weights/best.pt"
        else:
            print(f"\n❌ Error: Weights file not found at '{weights_full_path}' or 'models/best.pt'")
            sys.exit(1)

    print("\n🔑 Authenticating with Roboflow...")
    rf = Roboflow(api_key=args.api_key)

    try:
        workspace = rf.workspace(args.workspace)
        project = workspace.project(args.project)
        version = project.version(args.version)
    except Exception as e:
        print(f"\n❌ Failed to access project on Roboflow: {e}")
        print("Please check that your API key, workspace name, project name, and version number are correct.")
        sys.exit(1)

    print(f"📦 Packaging and uploading custom weights ({args.model_type}) to Roboflow...")
    try:
        version.deploy(
            model_type=args.model_type,
            model_path=args.model_path,
            filename=args.filename
        )
        print("\n" + "=" * 65)
        print("✅ SUCCESS: Model deployed to Roboflow!")
        print(f"Your hosted model is now accessible on Roboflow Universe / Dashboard at:")
        print(f"👉 https://universe.roboflow.com/{args.workspace}/{args.project}/{args.version}")
        print("=" * 65)
        print("\nYou can now:")
        print("1. Test live predictions via web browser / webcam in the 'Try This Model' tab.")
        print("2. Call the serverless Hosted API from Python, JavaScript, or cURL.")
        print("3. Deploy to edge devices (NVIDIA Jetson, Raspberry Pi, Docker) via Roboflow Inference.")
    except Exception as e:
        print(f"\n❌ Deployment error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
