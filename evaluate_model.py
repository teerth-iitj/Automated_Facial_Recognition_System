"""Evaluate the deployed SVM face matcher on held-out photos.

New in this version
-------------------
* --top-k N  : report top-1 and top-N accuracy (default N=3).
* Threshold sweep: evaluates the configured SVM distance gate at 0.35–0.55.
* Simple text-based confusion matrix.
* --output-csv FILE: saves per-photo results to a CSV.
* Max resize dimension raised 400 → 800 (consistent with train_model.py).
* upsample=1 for test encoding (consistent with train_model.py).
"""

import argparse
import csv
import os
from collections import defaultdict

import face_recognition
import joblib
import numpy as np
from PIL import Image

from dataset_split import person_name_from_filename, split_dataset


UNKNOWN  = "<unknown>"
_MAX_DIM = 800   # keep consistent with train_model.py


# ---------------------------------------------------------------------------
# Encoding helper
# ---------------------------------------------------------------------------

def encode_test_image(path):
    """Return the largest face encoding, or an explicit skip reason."""
    try:
        rgb = face_recognition.load_image_file(path)
        height, width = rgb.shape[:2]
        scale = min(1.0, _MAX_DIM / max(height, width))
        if scale < 1.0:
            new_size = (round(width * scale), round(height * scale))
            rgb = np.asarray(
                Image.fromarray(rgb).resize(new_size, Image.Resampling.LANCZOS)
            )
        boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=1)
        if not boxes:
            return None, None
        boxes = [max(boxes, key=lambda box: (box[2] - box[0]) * (box[1] - box[3]))]
        encodings = face_recognition.face_encodings(rgb, boxes)
        return (encodings[0], None) if encodings else (None, "Face encoding failed")
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _rank_svm_classes(classifier, encoding, top_k):
    """Return class candidates ordered by the fitted SVM's decision scores."""
    classes = np.asarray(classifier.classes_)
    scores = np.asarray(
        classifier.decision_function(np.asarray(encoding).reshape(1, -1))
    ).reshape(-1)

    if len(classes) == 2 and scores.size == 1:
        order = [1, 0] if scores[0] >= 0 else [0, 1]
    elif scores.size == len(classes):
        order = np.argsort(scores)[::-1]
    else:
        raise ValueError(
            "The classifier's decision scores do not match its identity classes."
        )
    return [str(classes[index]) for index in order[:top_k]]


def score_predictions(y_true, y_pred):
    """Compute accuracy, macro and weighted precision/recall/F1."""
    labels = sorted(set(y_true) | set(y_pred))
    per_class = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall    = tp / (tp + fn) if tp + fn else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if precision + recall else 0.0)
        support   = sum(t == label for t in y_true)
        per_class.append((precision, recall, f1, support))

    count    = len(y_true)
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / count
    macro    = [sum(r[i] for r in per_class) / len(per_class) for i in range(3)]
    weighted = [sum(r[i] * r[3] for r in per_class) / count for i in range(3)]
    return accuracy, macro, weighted, labels, per_class


# ---------------------------------------------------------------------------
# Confusion matrix (text)
# ---------------------------------------------------------------------------

def print_confusion_matrix(y_true, y_pred, labels):
    """Print a compact text-based confusion matrix."""
    label_set = labels
    n = len(label_set)
    idx = {lbl: i for i, lbl in enumerate(label_set)}
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        matrix[idx[t]][idx[p]] += 1

    # Truncate labels for display
    max_lbl = 14
    short = [lbl[:max_lbl] for lbl in label_set]
    col_w = max(4, max(len(s) for s in short))

    header = f"{'':>{col_w}}  " + "  ".join(f"{s:>{col_w}}" for s in short)
    print("\nConfusion matrix  (rows=actual, cols=predicted):")
    print(header)
    print("─" * len(header))
    for i, row_lbl in enumerate(label_set):
        row = f"{short[i]:>{col_w}}  " + "  ".join(
            f"{matrix[i][j]:>{col_w}}" for j in range(n)
        )
        print(row)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

SWEEP_THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55]


def threshold_sweep(test_records):
    """Evaluate the SVM-predicted identity at several distance cutoffs."""
    results = []
    for thr in SWEEP_THRESHOLDS:
        y_t = [record["actual"] for record in test_records]
        y_p = [
            record["svm_prediction"]
            if record["distance"] < thr
            else UNKNOWN
            for record in test_records
        ]
        acc = sum(t == p for t, p in zip(y_t, y_p)) / len(y_t) if y_t else 0.0
        results.append((thr, acc))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",       default="Dataset",
                        help="Folder containing labeled photos")
    parser.add_argument("--encodings",     default="face_encodings.pkl",
                        help="Training gallery file")
    parser.add_argument("--threshold",     type=float, default=0.45,
                        help="Face-distance cutoff (default: 0.45)")
    parser.add_argument("--seed",          type=int, default=42,
                        help="Seed used to reproduce the split")
    parser.add_argument("--test-fraction", type=float, default=0.2,
                        help="Holdout share per identity")
    parser.add_argument("--top-k",         type=int, default=3,
                        help="Report top-K accuracy in addition to top-1 (default: 3)")
    parser.add_argument("--output-csv",    default=None,
                        help="Save per-photo results to this CSV file")
    args = parser.parse_args()

    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between 0 and 1")
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    # --- load the exact split and trained SVM gallery ---
    train_paths, test_paths, singleton_count = split_dataset(
        args.dataset, test_fraction=args.test_fraction, seed=args.seed
    )
    gallery = joblib.load(args.encodings)
    gallery_encs = gallery.get("encodings", [])
    gallery_names = gallery.get("names", [])
    training_files = set(gallery.get("train_files", []))
    expected_training_files = {os.path.basename(path) for path in train_paths}
    test_files = {os.path.basename(path) for path in test_paths}
    overlap = training_files & test_files

    if not gallery_encs:
        raise ValueError("Training gallery is empty. Run train_model.py first.")
    if not training_files:
        raise ValueError(
            "Gallery is missing train_files metadata; retrain it before evaluating "
            "so test-set leakage can be checked."
        )
    if training_files != expected_training_files:
        raise ValueError(
            "Gallery training files do not match the current deterministic split. "
            "Run train_model.py again before evaluating."
        )
    if overlap:
        raise ValueError(f"Train/test file overlap detected ({len(overlap)} photos)")
    if len(gallery_encs) != len(gallery_names):
        raise ValueError("The gallery has a different number of encodings and labels.")

    classifier = gallery.get("classifier")
    if classifier is None or not hasattr(classifier, "predict"):
        raise ValueError("Gallery has no trained classifier. Run train_model.py first.")
    classifier_classes = set(str(name) for name in classifier.classes_)
    enrolled_names = set(str(name) for name in gallery_names)
    if classifier_classes != enrolled_names:
        raise ValueError("SVM classes do not match the identities in the face gallery.")

    encodings_by_name = defaultdict(list)
    for name, encoding in zip(gallery_names, gallery_encs):
        encodings_by_name[str(name)].append(encoding)

    # --- encode test photos ---
    y_true, y_pred, y_topk = [], [], []
    test_records = []
    csv_rows = []
    skipped_no_gallery = 0
    skipped_no_face = 0
    encoding_errors = []

    for path in test_paths:
        actual = person_name_from_filename(path)
        if actual not in enrolled_names:
            skipped_no_gallery += 1
            continue
        encoding, error = encode_test_image(path)
        if error:
            encoding_errors.append((os.path.basename(path), error))
            continue
        if encoding is None:
            skipped_no_face += 1
            continue

        svm_prediction = str(
            classifier.predict(np.asarray(encoding).reshape(1, -1))[0]
        )
        identity_encodings = encodings_by_name.get(svm_prediction, [])
        if not identity_encodings:
            raise ValueError(
                f"SVM predicted '{svm_prediction}', which has no gallery encodings."
            )
        distance = float(
            np.min(face_recognition.face_distance(identity_encodings, encoding))
        )
        prediction = svm_prediction if distance < args.threshold else UNKNOWN
        topk_names = _rank_svm_classes(classifier, encoding, args.top_k)
        in_topk = actual in topk_names

        y_true.append(actual)
        y_pred.append(prediction)
        y_topk.append(in_topk)
        test_records.append(
            {
                "actual": actual,
                "svm_prediction": svm_prediction,
                "distance": distance,
            }
        )

        csv_rows.append({
            "file": os.path.basename(path),
            "actual": actual,
            "predicted": prediction,
            "svm_prediction": svm_prediction,
            "minimum_distance": f"{distance:.6f}",
            "correct_top1": actual == prediction,
            "correct_topk": in_topk,
            "top_k": args.top_k,
            "topk_names": "|".join(topk_names),
        })

    # --- guard ---
    identities = len(set(y_true))
    if identities < 2 or not y_true:
        raise ValueError(
            "Not enough valid held-out faces for metrics: "
            f"{len(test_paths)} test candidates, {skipped_no_gallery} without a "
            f"gallery identity, {skipped_no_face} without a face, "
            f"{len(encoding_errors)} image/encoding errors."
        )

    # --- main metrics ---
    accuracy, macro, weighted, labels, per_class = score_predictions(y_true, y_pred)
    topk_accuracy = sum(y_topk) / len(y_topk)

    accepted_count = sum(prediction != UNKNOWN for prediction in y_pred)
    print("\nFace recognition held-out evaluation")
    print(f"Classifier                : {gallery.get('classifier_type', 'unknown')}")
    print(f"Training split candidates : {len(train_paths)}")
    print(f"Valid gallery encodings   : {len(gallery_encs)}")
    print(f"Held-out test candidates  : {len(test_paths)}")
    print(f"Valid held-out faces      : {len(y_true)}")
    print(f"Train/test file overlap   : {len(overlap)}")
    print(f"Identities evaluated      : {identities}")
    print(f"Identities in gallery     : {len(enrolled_names)}")
    print(f"Singletons (train-only)   : {singleton_count}")
    print(f"Skipped — not in gallery  : {skipped_no_gallery}")
    print(f"Skipped — no face found   : {skipped_no_face}")
    print(f"Image/encoding errors     : {len(encoding_errors)}")
    print(
        f"Distance threshold        : {args.threshold:.2f}  |  "
        f"seed={args.seed}  |  test_fraction={args.test_fraction:.0%}"
    )
    print()
    print(f"Top-1 accuracy (with reject): {accuracy:.3f}")
    print(f"Top-{args.top_k} SVM class accuracy: {topk_accuracy:.3f}")
    print(f"Accepted by distance gate : {accepted_count}/{len(y_pred)}")
    print(
        f"Macro    precision/recall/F1 : "
        f"{macro[0]:.3f} / {macro[1]:.3f} / {macro[2]:.3f}"
    )
    print(
        f"Weighted precision/recall/F1 : "
        f"{weighted[0]:.3f} / {weighted[1]:.3f} / {weighted[2]:.3f}"
    )

    # --- per-photo results ---
    print("\nPer-test-photo results:")
    for row in csv_rows:
        mark = "PASS" if row["correct_top1"] else (
            "TOP-K" if row["correct_topk"] else "MISS"
        )
        print(
            f"  {mark:<5} actual={row['actual']:<20} "
            f"predicted={row['predicted']:<20} "
            f"svm={row['svm_prediction']:<20} "
            f"distance={row['minimum_distance']}"
        )

    if encoding_errors:
        print("\nImage/encoding errors:")
        for filename, error in encoding_errors:
            print(f"  {filename}: {error}")

    # --- confusion matrix ---
    print_confusion_matrix(y_true, y_pred, labels)

    # --- threshold sweep ---
    print("\nThreshold sweep (top-1 accuracy):")
    print(f"  {'Threshold':>10}  {'Accuracy':>10}")
    print("  " + "─" * 24)
    sweep = threshold_sweep(test_records)
    best_thr, best_acc = max(sweep, key=lambda x: x[1])
    for thr, acc in sweep:
        marker = " ←" if thr == best_thr else ""
        print(f"  {thr:>10.2f}  {acc:>10.3f}{marker}")
    print(f"\n  Best threshold: {best_thr:.2f}  (accuracy {best_acc:.3f})")

    # --- optional CSV ---
    if args.output_csv:
        fieldnames = [
            "file", "actual", "predicted", "svm_prediction",
            "minimum_distance", "correct_top1", "correct_topk", "top_k",
            "topk_names",
        ]
        with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nPer-photo CSV saved to: {args.output_csv}")

    print(
        "\nThe prediction and distance gate match app.py. The face-embedding "
        "network is pretrained; this evaluation does not train it."
    )
    print(
        "Threshold sweep is diagnostic only; choose a deployment threshold "
        "using a separate validation set."
    )


if __name__ == "__main__":
    main()
