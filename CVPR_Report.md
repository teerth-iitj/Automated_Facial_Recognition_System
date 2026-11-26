# Automated Face Recognition System

<div align="center">

| **Basab Bose** | **Teerth Bhardwaj** | **Ravikaran Singh** |
| :---: | :---: | :---: |
| M26CSE002 | M26CSE010 | M26CSE012 |
| `m26cse002@iitj.ac.in` | `m26cse010@iitj.ac.in` | `m26cse012@iitj.ac.in` |

Department of Computer Science and Engineering  
Indian Institute of Technology Jodhpur, India

</div>

---

## Abstract

Taking attendance manually in classrooms is slow, interrupts teaching, and leaves room for proxy entries or clerical mistakes. In this project, we built the **Automated Face Recognition System**, a practical web-based attendance application that identifies students from live webcam captures, uploaded photographs, and recorded classroom videos on standard CPU laptops without needing a GPU. Our recognition pipeline uses a pretrained ResNet-based deep face embedding model (via `dlib` and `face_recognition`) to convert each detected face into a compact 128-number representation. On top of these face encodings, we train a Support Vector Machine (SVM) with an RBF kernel to tell enrolled individuals apart, paired with a minimum face-distance check (threshold `0.45`) that rejects unfamiliar faces as `<unknown>`. The backend is built with Flask and SQLite, offering separate dashboards for administrators and faculty, course and semester management, automatic duplicate-attendance prevention, and CSV report downloads. We test our model on a dataset of 2,460 supported photos split deterministically into 1,983 training photos and 477 held-out test photos. Out of 471 valid detected test faces, the system achieves **98.3% Top-1 accuracy** (463 out of 471 photos correctly recognized and accepted), **99.6% Top-3 SVM accuracy**, and a **0.990 weighted F1-score**, with zero wrong identity swaps among all accepted predictions. All source code, environment files, prebuilt checkpoint, and reproducibility instructions are available at [https://github.com/teerth-iitj/Automated_Facial_Recognition_System](https://github.com/teerth-iitj/Automated_Facial_Recognition_System).

*(The formatted 7-page double-column CVPR PDF is available in [`CVPR_Report.pdf`](CVPR_Report.pdf).)*

---

## 1. Introduction

Recording student attendance is an everyday requirement across universities and colleges. Traditional methods like calling out names or passing around a paper sign-in sheet waste lecture time and are easy to forge via proxy signatures. Hardware options such as fingerprint scanners create queues and require physical contact, while RFID cards can easily be lent to friends.

Camera-based facial recognition provides a cleaner, contactless alternative. However, building a system that works smoothly in a real classroom requires handling uneven photo counts across students, running comfortably on normal CPU laptops without GPUs, rejecting unfamiliar faces safely, and integrating smoothly with semester and subject schedules.

---

## 2. Background and Related Work

Early face recognition methods like Eigenfaces, Fisherfaces, and LBPH relied on classical image projections and texture histograms, which break down under changes in lighting, pose, or expression. Modern systems such as DeepFace, FaceNet, and `dlib`'s ResNet-based embedding network map aligned faces into a 128-number representation where photos of the same person sit close together. Instead of retraining this large neural network on small classroom datasets, we extract the 128-number encodings once and train an RBF-kernel Support Vector Machine (`scikit-learn`) on top, combined with a distance cutoff to reject strangers.

---

## 3. How the Recognition Model Works

1. **Image Resizing & Face Detection:** Photos are resized so their longest dimension is at most **800 pixels** (`LANCZOS`), and `dlib`'s HOG face detector (`number_of_times_to_upsample=1`) locates faces. During training and evaluation, multi-face photos keep the largest bounding box by area, and zero-face photos are skipped. In live classroom recognition (`app.py`), *all* detected faces in a frame are processed so multiple students can be marked together.
2. **128-Dimensional Face Embeddings:** `face_recognition` aligns eye, nose, and mouth landmarks and converts each face into 128 numerical features using the frozen pretrained ResNet model (`num_jitters=1` by default).
3. **Parallel Gallery Building & RBF-SVM Training:** `train_model.py` uses `ProcessPoolExecutor` (up to 4 workers) to encode the training split in parallel, then trains an RBF-SVM (`SVC(kernel="rbf", C=1.0, gamma="scale")`, retaining 1,395 support vectors) and saves everything into `face_encodings.pkl` (~5.8 MB).
4. **Distance Safety Gate (`<unknown>` Rejection):** After the RBF-SVM predicts the closest enrolled name, the system checks the minimum face distance to that person's stored gallery photos (`face_recognition.face_distance`). Matches strictly below **0.45** are accepted; anything at or above `0.45` is rejected as `<unknown>`.

---

## 4. Web Application and System Design

`app.py` serves a Flask web portal backed by SQLite (`attendance.db` with `PRAGMA foreign_keys = ON`) across five tables: `users`, `semesters`, `subjects`, `faculty_subjects`, and `attendance`.

- **Admin Dashboard (`templates/admin_dashboard.html`):** Create/delete faculty accounts, manage semesters and subjects, assign subjects to faculty, and clear global records.
- **Faculty Dashboard (`templates/faculty_dashboard.html`):** Select an assigned semester and subject, then mark attendance via:
  - **Live Webcam Capture:** Browser WebRTC camera (`getUserMedia`) with device selector, capturing frames onto an HTML5 `<canvas>` and posting JPEG snapshots to `/recognize` (supports HTTPS via `FLASK_HTTPS=1` / `run_https_5443.ps1`).
  - **Photo Upload (`/recognize`):** Decodes uploaded images via OpenCV and marks all recognized students.
  - **Video Upload (`/recognize-video`):** Samples every 15th frame (up to 60 sampled frames max) and merges recognized students.
- **Duplicate Prevention & Export:** `already_marked_today()` prevents duplicate entries on the same day for the same student, semester, subject, and faculty member. Records can be inspected, deleted individually, cleared per class, or exported as CSV (`/attendance/download`).
- **Windows Bootstrap:** `start.bat` and `start.ps1` create `.venv`, pin `setuptools<81`, install `dlib-bin` and `requirements.txt`, fetch `face_recognition_models`, and launch Chrome.

---

## 5. Dataset and Evaluation Protocol

Using `dataset_split.py` (`seed=42`, `test_fraction=0.2`), the `Dataset/` folder (2,463 total files; **2,460** supported `.jpg`/`.jpeg`/`.png` images and 3 ignored `.heif`/`.webp` files) is partitioned deterministically with **0 file overlap**:

| Dataset & Split Metric | Observed Count |
|---|---|
| Total supported images in `Dataset/` | 2,460 photos (2,391 multi-photo + 69 single-photo) |
| Training split candidates | 1,983 photos |
| Valid gallery encodings saved in `face_encodings.pkl` | 1,964 encodings (19 skipped — no face found; 98 enrolled people) |
| Held-out test candidates | 477 photos |
| Valid held-out test faces evaluated | 471 photos (6 skipped — no face found; 0 errors) |
| Train / test file overlap | 0 photos |

---

## 6. Experimental Results

From `evaluation_split.log` at the deployed threshold `0.45`:

| Metric | Measured Result |
|---|---|
| Top-1 Accuracy (with 0.45 distance gate) | **98.3%** (`0.983`, 463 / 471) |
| Top-3 SVM Class Accuracy | **99.6%** (`0.996`, 469 / 471) |
| Wrong Identity Matches Among Accepted Photos | **0 / 463** (0.0%) |
| Weighted Precision / Recall / F1 | **0.998 / 0.983 / 0.990** |
| Macro Precision / Recall / F1 | `0.938 / 0.919 / 0.927` |

- **Distance Distribution (471 test photos):** min `0.0000`, mean `0.3112`, max `0.5314`. Nearly 90% (422 / 471) fell below `0.40`.
- **Rejected Photos (8 / 471):** Distances ranged from `0.4522` to `0.5314`. On 6 of those 8 (`TOP-K`), the RBF-SVM still guessed the correct person; only 2 (`MISS`) had a different top SVM guess, and both were safely rejected as `<unknown>`.
- **Threshold Sweep (`evaluation_split.log`):** `0.35` → `0.728`, `0.40` → `0.896`, `0.45` → `0.983` (deployed), `0.50` → `0.994`, `0.55` → `0.996`.

---

## 7. Limitations & Conclusion

Current limitations include lack of liveness anti-spoofing, student login disabled in `login()` (though `get_student_summary` exists in `app.py`), retraining required when adding new students, and skipping 3 `.heif`/`.webp` files. Overall, the system achieves **98.3% Top-1 accuracy** with zero false identity swaps among accepted predictions and provides a practical, CPU-friendly attendance web application.

---

## Appendix A — Commit History & Reproducibility

**Declaration Statement of Reproducibility:** All evaluation figures and software behaviors reported here are reproducible using the repository source code, `requirements.txt`, `environment.yml`, `dataset_split.py` (`seed=42`), `evaluate_model.py`, and the committed checkpoint `face_encodings.pkl` (~5.8 MB). Raw dataset photos (~1.05 GB) are excluded via `.gitignore`.

| Hash | Timestamp (IST) | Commit Message |
|---|---|---|
| `33e6fb4a` | 2026-11-26 12:30 | Initial project setup: Flask skeleton, SQLite schema, role-based login |
| `1f1803f2` | 2026-11-26 13:00 | Add train_model.py: parallel gallery builder with RBF-SVM classifier |
| `019e2610` | 2026-11-26 14:00 | Add evaluate_model.py: held-out metrics, confusion matrix, threshold sweep |
| `7b35ea04` | 2026-11-26 14:18 | Add req.py: dependency checker with auto-install fallback |
| `76fc31e9` | 2026-11-26 16:17 | Add Windows startup scripts: start.bat, start.ps1, HTTPS launcher |
| `80dcc8f9` | 2026-11-27 03:32 | Add run_all.py orchestrator; update requirements.txt with scikit-learn |
| `0ddd2073` | 2026-11-27 04:00 | Add pre-built face gallery checkpoint (face_encodings.pkl, ~5.8 MB) |
| `033a61bd` | 2026-11-27 04:26 | Repo cleanup: add .gitignore, README, REPRODUCIBILITY, LICENSE, environment.yml |
