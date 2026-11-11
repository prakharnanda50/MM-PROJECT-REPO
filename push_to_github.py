"""
push_to_github.py - Push committed repository to GitHub
"""

import argparse
import sys
import dulwich.porcelain as porcelain
from dulwich.repo import Repo


def parse_args():
    parser = argparse.ArgumentParser(description="Push Model Mayhem Git repository to GitHub")
    parser.add_argument("--repo-url", type=str, required=True, help="GitHub repository URL (e.g. https://github.com/USER/REPO.git)")
    parser.add_argument("--branch", type=str, default="main", help="Branch name to push (default: main)")
    parser.add_argument("--token", type=str, default=None, help="GitHub Personal Access Token (if pushing over HTTPS to private or authenticated repo)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("🚀 PUSHING TO GITHUB")
    print("=" * 65)
    print(f"Target URL : {args.repo_url}")
    print(f"Branch     : {args.branch}")
    print("=" * 65)

    target_url = args.repo_url
    if args.token and "https://" in target_url and "@" not in target_url:
        # Inject token into HTTPS URL: https://TOKEN@github.com/USER/REPO.git
        target_url = target_url.replace("https://", f"https://{args.token}@")

    try:
        repo = Repo(".")
        refspec = f"refs/heads/{args.branch}:refs/heads/{args.branch}"
        print(f"Pushing refspec {refspec}...")
        porcelain.push(".", target_url, refspecs=[refspec.encode()])
        print("\n✅ Successfully pushed to GitHub!")
        print(f"👉 Check your repository at: {args.repo_url.replace('.git', '')}")
    except Exception as e:
        print(f"\n❌ Push failed: {e}")
        print("\nTips:")
        print("1. If using HTTPS with authentication, provide your GitHub Token:")
        print("   python push_to_github.py --repo-url <URL> --token <YOUR_GITHUB_TOKEN>")
        print("2. Or if you have Git installed, run in terminal:")
        print("   git remote add origin <URL>")
        print("   git branch -M main")
        print("   git push -u origin main")
        sys.exit(1)


if __name__ == "__main__":
    main()
