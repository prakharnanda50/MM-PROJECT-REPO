"""
pack_git_repo.py - Pack Model Mayhem deliverables into a standard Git repository
"""

import os
import dulwich.porcelain as porcelain
from dulwich.repo import Repo

REPO_DIR = "."

def pack_repo():
    print("=" * 65)
    print("📦 INITIALIZING & PACKING GIT REPOSITORY")
    print("=" * 65)

    # 1. Initialize Git Repo if not exists
    if not os.path.exists(".git"):
        repo = porcelain.init(REPO_DIR)
        print("Initialized empty Git repository in D:/Model Mahyem/.git/")
    else:
        repo = Repo(REPO_DIR)
        print("Existing Git repository found.")

    # 2. Select files to track
    files_to_add = [
        ".gitignore",
        "README.md",
        "requirements.txt",
        "data_leak_free.yaml",
        "train.py",
        "evaluate.py",
        "inference.py",
        "macro_f1_evaluator.py",
        "eval_pipeline.py",
        "deploy_to_roboflow.py",
        "final_test_evaluation.py",
        "cv_evaluation.py",
        "fast_cv_stability.py",
        "create_leak_free_split.py",
        "step0_inspect.py",
        "download_dataset.py",
        "models/best.pt",
        "output/predictions.csv",
        "splits/train.txt",
        "splits/val.txt",
        "splits/test.txt",
        "results/experiments_log.csv",
        "results/final_test_report.json",
        "results/5fold_cv_results.json",
        "results/final_test_confusion_matrix.png",
    ]

    existing_files = [f for f in files_to_add if os.path.exists(f)]

    print(f"\nStaging {len(existing_files)} submission files:")
    for f in existing_files:
        size_kb = os.path.getsize(f) / 1024.0
        print(f"  + {f} ({size_kb:.1f} KB)")

    # Stage files
    porcelain.add(REPO_DIR, paths=existing_files)

    # Commit
    commit_msg = b"feat: Model Mayhem PCB Defect Detection submission - YOLO11 model, inference pipeline, evaluation scripts, and predictions\n"
    commit_id = porcelain.commit(
        REPO_DIR,
        message=commit_msg,
        committer=b"Antigravity ML Engineer <agent@model-mayhem.local>",
        author=b"Antigravity ML Engineer <agent@model-mayhem.local>"
    )

    print("\n" + "=" * 65)
    print("✅ GIT COMMIT CREATED SUCCESSFULLY")
    print(f"Commit ID : {commit_id.decode() if isinstance(commit_id, bytes) else commit_id}")
    print("Branch    : master (default)")
    print("=" * 65)

    # Create main branch pointing to HEAD
    try:
        repo.refs[b"refs/heads/main"] = repo.head()
        print("Branch 'main' created pointing to HEAD.")
    except Exception as e:
        print(f"Branch ref note: {e}")

if __name__ == "__main__":
    pack_repo()
