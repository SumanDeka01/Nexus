"""
Expand the ISL dataset for the 8-word prototype.

This script:
- Keeps the existing ISL_DATASET folder untouched.
- Downloads additional videos from the public ISL500 Hugging Face dataset.
- Targets the same 8 words already used in the prototype.
- Downloads up to MAX_PER_CLASS videos per word.
- Creates a separate folder: ISL_DATASET_LARGE
- Writes a manifest CSV so we know exactly which files were downloaded.

Requirements:
    pip install huggingface_hub pandas

Run from the project root:
    python src/expand_isl_dataset.py

Optional:
    python src/expand_isl_dataset.py --per-class 50
"""

from pathlib import Path
import argparse
import csv
import re
import sys

from huggingface_hub import HfApi, hf_hub_download


# =========================================================
# SETTINGS
# =========================================================

REPO_ID = "ISL500/ISL-DATA"
REPO_TYPE = "dataset"

WORDS = [
    "eat",
    "go",
    "hello",
    "help",
    "no",
    "please",
    "water",
    "yes",
]

DEFAULT_PER_CLASS = 100

OUTPUT_DIR = Path("ISL_DATASET_LARGE")
VIDEO_DIR = OUTPUT_DIR / "videos"
MANIFEST_FILE = OUTPUT_DIR / "metadata.csv"


# =========================================================
# HELPERS
# =========================================================

def normalize_word(text):
    """Normalize text for matching folder/file names."""
    text = str(text).strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def path_matches_word(repo_path, word):
    """
    Match a target word against the directory/file path.

    Examples that can match 'eat':
        eat/...
        Eat/...
        .../eat__ISL500...
    """
    normalized = repo_path.replace("\\", "/")
    parts = normalized.split("/")

    target = normalize_word(word)

    # Folder match.
    for part in parts[:-1]:
        if normalize_word(part) == target:
            return True

    # Filename prefix match.
    filename = parts[-1]
    filename_normalized = normalize_word(filename)

    if filename_normalized.startswith(target + " "):
        return True

    # ISL500 filenames commonly contain:
    # eat__ISL500__...
    if filename.lower().startswith(target + "__"):
        return True

    return False


def choose_files(files, word, limit):
    """Select up to limit .mp4 files for one word."""
    candidates = []

    for path in files:
        lower = path.lower()

        if not lower.endswith(".mp4"):
            continue

        if path_matches_word(path, word):
            candidates.append(path)

    # Deterministic ordering makes the experiment reproducible.
    candidates = sorted(set(candidates))

    return candidates[:limit], len(candidates)


def load_existing_manifest():
    """Return already downloaded repo paths."""
    if not MANIFEST_FILE.exists():
        return set()

    existing = set()

    with MANIFEST_FILE.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            repo_path = row.get("repo_path")
            if repo_path:
                existing.add(repo_path)

    return existing


def append_manifest(rows):
    """Append download records to the manifest."""
    file_exists = MANIFEST_FILE.exists()

    with MANIFEST_FILE.open(
        "a",
        encoding="utf-8",
        newline=""
    ) as f:

        fieldnames = [
            "word",
            "repo_id",
            "repo_path",
            "local_path",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Expand the 8-word ISL prototype dataset."
    )

    parser.add_argument(
        "--per-class",
        type=int,
        default=DEFAULT_PER_CLASS,
        help="Maximum videos to download per word (default: 100)."
    )

    args = parser.parse_args()

    if args.per_class <= 0:
        raise ValueError("--per-class must be greater than 0.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("ISL DATASET EXPANSION")
    print("========================================")
    print("Source :", REPO_ID)
    print("Words  :", WORDS)
    print("Target :", args.per_class, "videos/class")
    print("Output :", OUTPUT_DIR)
    print()

    print("IMPORTANT:")
    print("- Existing ISL_DATASET will NOT be modified.")
    print("- New videos go into ISL_DATASET_LARGE.")
    print()

    # -----------------------------------------------------
    # Connect to Hugging Face
    # -----------------------------------------------------
    api = HfApi()

    print("Reading file list from Hugging Face...")
    print("This can take a little while for a large dataset.")

    try:
        files = api.list_repo_files(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE
        )
    except Exception as exc:
        print("\nCould not access the Hugging Face dataset.")
        print("Error:", exc)
        sys.exit(1)

    print(f"Files reported by repository: {len(files)}")

    existing_repo_paths = load_existing_manifest()

    total_downloaded = 0

    # -----------------------------------------------------
    # Process each word
    # -----------------------------------------------------
    for word in WORDS:

        selected, available = choose_files(
            files,
            word,
            args.per_class
        )

        print()
        print("----------------------------------------")
        print(f"{word.upper()}")
        print("----------------------------------------")
        print(f"Available matching videos: {available}")

        if available == 0:
            print("WARNING: No videos found for this word.")
            continue

        downloaded_rows = []
        class_downloaded = 0

        for index, repo_path in enumerate(selected, start=1):

            if repo_path in existing_repo_paths:
                continue

            print(
                f"[{index}/{len(selected)}] "
                f"Downloading {repo_path}"
            )

            try:
                downloaded_file = hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type=REPO_TYPE,
                    filename=repo_path,
                    local_dir=str(VIDEO_DIR / word)
                )

            except Exception as exc:
                print("  FAILED:", exc)
                continue

            # The local_dir preserves the repository path.
            downloaded_file = Path(downloaded_file)

            downloaded_rows.append({
                "word": word,
                "repo_id": REPO_ID,
                "repo_path": repo_path,
                "local_path": str(downloaded_file),
            })

            existing_repo_paths.add(repo_path)
            class_downloaded += 1
            total_downloaded += 1

        if downloaded_rows:
            append_manifest(downloaded_rows)

        print(
            f"{word}: downloaded {class_downloaded} new videos"
        )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------
    print()
    print("========================================")
    print("EXPANSION COMPLETE")
    print("========================================")
    print("New videos downloaded:", total_downloaded)
    print("Dataset folder:", OUTPUT_DIR)
    print("Manifest:", MANIFEST_FILE)

    if MANIFEST_FILE.exists():
        print("\nManifest created successfully.")
    else:
        print("\nWARNING: No manifest was created.")

    print()
    print("NEXT STEP:")
    print("Do NOT train yet.")
    print(
        "First inspect the number of videos per word, "
        "then we will extract 30-frame landmark sequences."
    )


if __name__ == "__main__":
    main()