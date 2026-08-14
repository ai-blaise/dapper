"""
Upload output-datasets/ to HuggingFace Hub.

Uses upload_large_folder for robust handling of large files (~8GB) -
resumable, multi-threaded, and handles failures gracefully.

Usage:
    uv run python scripts/upload_to_hf.py

Requires: hf auth login (run beforehand)
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "BlaiseAI/blaise-sft-training-mix"
LOCAL_DIR = "output-datasets"


def main():
    api = HfApi()

    # Verify auth
    try:
        user = api.whoami()
        print(f"Authenticated as: {user['name']}")
    except Exception as e:
        print(f"Not authenticated. Run 'hf auth login' first.\n{e}")
        sys.exit(1)

    # Collect all files to upload (parquet + markdown)
    folder = Path(LOCAL_DIR)
    all_files = set()

    # Find all parquet files
    for f in folder.glob("**/*.parquet"):
        all_files.add(f)
    # Find all markdown files (README.md, etc.)
    for f in folder.glob("**/*.md"):
        all_files.add(f)

    files = sorted(all_files, key=lambda x: x.name)

    print(f"\nRepo: {REPO_ID}")
    print(f"Files to upload ({len(files)}):")
    for f in files:
        size_gb = f.stat().st_size / 1e9
        print(f"  {f.name} ({size_gb:.2f} GB)")
    print()

    # Upload all files in a single commit using upload_large_folder
    print(f"Uploading {len(files)} files to {REPO_ID}...")
    try:
        api.upload_large_folder(
            folder_path=str(folder),
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        print(
            f"\n✓ All {len(files)} files uploaded to https://huggingface.co/datasets/{REPO_ID}"
        )
    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
