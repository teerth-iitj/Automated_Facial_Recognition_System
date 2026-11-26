"""Build the known-face gallery from Dataset.

This extracts embeddings from the pretrained face_recognition model; it does
not fit or fine-tune the underlying neural network.

New in this version
-------------------
* Max resize dimension raised 400 → 800 so small faces survive downscaling.
* number_of_times_to_upsample=1 for better detection of small faces.
* Multiple faces in one photo are handled gracefully: the largest bounding box
  is chosen instead of discarding the image.
* Zero-face images are skipped (not multi-face images).
* --jitters CLI arg for more robust encodings (default 1).
* --workers, --dataset, --output CLI args.
* Clear summary table printed at the end.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    project_python = os.path.join(project_dir, ".venv", "Scripts", "python.exe")
    current_python = os.path.normcase(os.path.abspath(sys.executable))
    project_python_normalized = os.path.normcase(os.path.abspath(project_python))
    if os.path.isfile(project_python) and current_python != project_python_normalized:
        raise SystemExit(
            subprocess.call(
                [project_python, os.path.abspath(__file__), *sys.argv[1:]],
                cwd=project_dir,
            )
        )

import face_recognition
import joblib
import numpy as np
from PIL import Image
from sklearn.svm import SVC

from dataset_split import person_name_from_filename, split_dataset


# ---------------------------------------------------------------------------
# Default constants (overridable via CLI)
# ---------------------------------------------------------------------------
_DEFAULT_DATASET = "Dataset"
_DEFAULT_OUTPUT  = "face_encodings.pkl"
_DEFAULT_WORKERS = min(4, os.cpu_count() or 1)
_MAX_DIM         = 800   # was 400 — larger value keeps small faces detectable


def _fit_identity_classifier(encodings, names):
    identities = set(names)
    if len(encodings) != len(names):
        raise ValueError("Each face encoding must have exactly one identity label.")
    if len(identities) < 2:
        raise ValueError("At least two identities are required to train the SVM classifier.")

    classifier = SVC(kernel="rbf", C=1.0, gamma="scale")
    classifier.fit(np.asarray(encodings), np.asarray(names))
    return classifier


# ---------------------------------------------------------------------------
# Worker function (must be module-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _encode_image(args_tuple):
    """Encode a single image.  Called in a worker process.

    Returns
    -------
    (image_name, encoding_or_None, issue_or_None, person_name)
    """
    image_path, jitters = args_tuple
    image_name = os.path.basename(image_path)

    # --- load & optionally downscale ---
    try:
        rgb = face_recognition.load_image_file(image_path)
        height, width = rgb.shape[:2]
        scale = min(1.0, _MAX_DIM / max(height, width))
        if scale < 1.0:
            new_size = (round(width * scale), round(height * scale))
            rgb = np.asarray(
                Image.fromarray(rgb).resize(new_size, Image.Resampling.LANCZOS)
            )
    except Exception as exc:
        return image_name, None, f"Could not read image: {exc}", None

    # --- locate faces (upsample=1 catches small faces) ---
    boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=1)

    if len(boxes) == 0:
        return image_name, None, "No faces found", None

    # --- if multiple faces, pick the largest bounding box ---
    if len(boxes) > 1:
        def _area(box):
            top, right, bottom, left = box
            return (bottom - top) * (right - left)
        boxes = [max(boxes, key=_area)]

    encodings = face_recognition.face_encodings(rgb, boxes, num_jitters=jitters)
    if not encodings:
        return image_name, None, "Encoding failed after location", None

    person = person_name_from_filename(image_name)
    return image_name, encodings[0], None, person


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default=_DEFAULT_DATASET,
        help=f"Folder containing labeled photos (default: {_DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output", default=_DEFAULT_OUTPUT,
        help=f"Output pickle file (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--workers", type=int, default=_DEFAULT_WORKERS,
        help=f"Parallel worker processes (default: {_DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--jitters", type=int, default=1,
        help="Number of jitters for face_encodings — higher = slower but more robust (default: 1)",
    )
    parser.add_argument(
        "--reuse-encodings", action="store_true",
        help="Train/update the SVM from encodings already in --output without processing dataset images",
    )
    args = parser.parse_args()

    if args.reuse_encodings:
        if not os.path.isfile(args.output):
            raise FileNotFoundError(f"Encoding gallery not found: {args.output}")
        data = joblib.load(args.output)
        known_encodings = data.get("encodings", [])
        known_names = data.get("names", [])
        if not known_encodings or not known_names:
            raise ValueError(f"{args.output} does not contain valid face encodings and names.")

        data["classifier"] = _fit_identity_classifier(known_encodings, known_names)
        data["classifier_type"] = "rbf_svm"
        joblib.dump(data, args.output)
        print(
            f"Trained RBF-SVM classifier for {len(set(known_names))} identities "
            f"using {len(known_encodings)} existing face encodings. Saved to {args.output}."
        )
        return

    # --- split dataset ---
    paths, test_paths, singleton_count = split_dataset(args.dataset)
    total_photos = len(paths)

    print(
        f"Training gallery: {total_photos} photos; "
        f"held-out test set: {len(test_paths)} photos; "
        f"single-photo identities: {singleton_count}."
    )
    print(
        f"Encoding with {args.workers} worker(s), "
        f"upsample=1, jitters={args.jitters}, max_dim={_MAX_DIM}…"
    )

    # --- encode in parallel ---
    known_encodings, known_names = [], []
    skipped = 0
    skipped_reasons: dict[str, int] = {}

    work_items = [(p, args.jitters) for p in paths]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for image_name, encoding, issue, person in pool.map(
            _encode_image, work_items, chunksize=4
        ):
            if issue:
                skipped += 1
                skipped_reasons[issue] = skipped_reasons.get(issue, 0) + 1
                print(f"  Skipped {image_name}: {issue}")
                continue
            known_encodings.append(encoding)
            known_names.append(person)
            print(f"  Encoded: {image_name}  →  {person}")

    if not known_encodings:
        print("\n✗ No valid face encodings found. Check the images in the dataset.")
        return

    # --- save gallery ---
    data = {
        "encodings":        known_encodings,
        "names":            known_names,
        "train_photo_count": len(paths),
        "train_files":      [os.path.basename(p) for p in paths],
        "classifier":       _fit_identity_classifier(known_encodings, known_names),
        "classifier_type":  "rbf_svm",
    }
    joblib.dump(data, args.output)

    # --- summary table ---
    encoded      = len(known_encodings)
    identities   = len(set(known_names))
    col_w        = 28
    print()
    print("─" * 50)
    print("  Training summary")
    print("─" * 50)
    print(f"  {'Total training photos':<{col_w}} {total_photos}")
    print(f"  {'Successfully encoded':<{col_w}} {encoded}")
    print(f"  {'Skipped':<{col_w}} {skipped}")
    for reason, cnt in skipped_reasons.items():
        print(f"    └─ {reason:<{col_w-4}} {cnt}")
    print(f"  {'Distinct identities':<{col_w}} {identities}")
    print(f"  {'Output file':<{col_w}} {args.output}")
    print("─" * 50)
    print(f"✓ Saved {encoded} encodings for {identities} identities to {args.output}")


if __name__ == "__main__":
    main()
