"""Deterministic identity-aware train/test split for the Dataset folder.

Supports two dataset layouts:
  1. Flat layout  — Dataset/PersonName_1.jpg, Dataset/PersonName_2.jpg, …
  2. Subdirectory layout — Dataset/PersonName/1.jpg, Dataset/PersonName/2.jpg, …

Both layouts can coexist: files directly inside *dataset_dir* use the
filename-based naming rule; files inside a subfolder use the subfolder name.

CLI usage (when run directly):
    python dataset_split.py --dataset Dataset
    python dataset_split.py --list-identities
"""

import argparse
import os
import random
import re
from collections import defaultdict


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def person_name_from_filename(image_name):
    """Derive a canonical person name from a bare filename (no directory)."""
    stem = os.path.splitext(os.path.basename(image_name))[0]
    # Strip trailing copy markers like "(2)"
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    # If the stem looks like "First Last_3", strip the trailing _N
    if " " in stem:
        stem = re.sub(r"_\d+$", "", stem)
    return " ".join(stem.replace("_", " ").split()).casefold()


def _person_name_from_folder(folder_name):
    """Derive a canonical person name from a subfolder name."""
    return " ".join(folder_name.replace("_", " ").split()).casefold()


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def _collect_paths(dataset_dir):
    """Return a dict mapping person_name → [absolute_paths].

    Scans both flat files and one-level-deep subdirectories.
    Subdirectory name takes precedence over filename-based naming for files
    inside that subdirectory.
    """
    grouped = defaultdict(list)

    for entry in os.scandir(dataset_dir):
        if entry.is_file() and entry.name.lower().endswith(IMAGE_EXTENSIONS):
            # Flat file
            name = person_name_from_filename(entry.name)
            grouped[name].append(entry.path)

        elif entry.is_dir():
            # Subdirectory layout: use folder name as person name
            person = _person_name_from_folder(entry.name)
            for sub_entry in os.scandir(entry.path):
                if sub_entry.is_file() and sub_entry.name.lower().endswith(IMAGE_EXTENSIONS):
                    grouped[person].append(sub_entry.path)

    return grouped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_dataset(dataset_dir="Dataset", test_fraction=0.2, seed=42):
    """Return (train_paths, test_paths, singleton_count).

    Guarantees at least one training sample per identity.  Singleton
    identities (only one photo) are always assigned to training.
    """
    grouped = _collect_paths(dataset_dir)

    rng = random.Random(seed)
    train_paths, test_paths = [], []
    singleton_identities = 0

    for paths in grouped.values():
        paths = sorted(paths)
        rng.shuffle(paths)
        if len(paths) < 2:
            singleton_identities += 1
            train_paths.extend(paths)
            continue

        test_count = max(1, round(len(paths) * test_fraction))
        test_count = min(test_count, len(paths) - 1)
        test_paths.extend(paths[:test_count])
        train_paths.extend(paths[test_count:])

    return sorted(train_paths), sorted(test_paths), singleton_identities


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _list_identities(dataset_dir):
    grouped = _collect_paths(dataset_dir)
    if not grouped:
        print(f"No images found in '{dataset_dir}'.")
        return

    print(f"\nIdentities found in '{dataset_dir}':\n")
    total_photos = 0
    col = max(len(name) for name in grouped) + 2
    for name in sorted(grouped):
        count = len(grouped[name])
        total_photos += count
        print(f"  {name:<{col}}  {count} photo(s)")
    print(f"\n  Total: {len(grouped)} identities, {total_photos} photos")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Dataset",
                        help="Dataset folder (default: Dataset)")
    parser.add_argument("--test-fraction", type=float, default=0.2,
                        help="Fraction of photos held out per identity (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--list-identities", action="store_true",
                        help="Print all identity names and photo counts, then exit")
    args = parser.parse_args()

    if args.list_identities:
        _list_identities(args.dataset)
        return

    train, test, singletons = split_dataset(
        args.dataset, test_fraction=args.test_fraction, seed=args.seed
    )
    print(f"Dataset  : {args.dataset}")
    print(f"Train    : {len(train)} photos")
    print(f"Test     : {len(test)} photos")
    print(f"Singletons (train only): {singletons} identities")


if __name__ == "__main__":
    main()
