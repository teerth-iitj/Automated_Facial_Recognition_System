# Reproducibility Declaration

## Automated Facial Recognition Attendance System

**Date:** November 27, 2026  
**Repository:** https://github.com/teerth-iitj/Automated_Facial_Recognition_System

---

## Declaration of Reproducibility

We, the authors of this project, declare that **all experimental results reported in the accompanying CVPR-format technical report are reproducible** using the artifacts and instructions provided in this repository. Specifically:

1. **Code** — All source code is committed and version-controlled in this repository.
2. **Checkpoint** — The pre-built face gallery and SVM classifier (`face_encodings.pkl`, ~5.8 MB) are committed to the repository and can be used directly for inference and evaluation without re-training.
3. **Environment** — Exact dependency versions are specified in `requirements.txt` and `environment.yml`.
4. **Evaluation script** — `evaluate_model.py` with `--seed 42` and `--test_fraction 0.2` reproduces the exact evaluation split reported in the paper.
5. **Dataset** — The dataset (~1 GB) is not included due to size, but the structure and naming convention are documented so any equivalent dataset can be used.

The main results (Top-1 accuracy: 98.3%, Weighted F1: 99.0%) were obtained with the checkpoint in this repository and can be reproduced using the steps below.

---

## Step-by-Step Reproduction Guide

### System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10/11 (tested), Ubuntu 20.04+ (should work) |
| Python | 3.8 – 3.12 (3.11 recommended) |
| RAM | ≥ 8 GB |
| CPU | Any modern multi-core CPU (no GPU required) |
| Disk | ≥ 2 GB free for dependencies + dataset |

### Step 1 — Clone the repository

```bash
git clone https://github.com/teerth-iitj/Automated_Facial_Recognition_System.git
cd Automated_Facial_Recognition_System
```

**Verify commit hash:**
```bash
git log --oneline -5
```

### Step 2 — Set up Python environment

**Option A: Conda**
```bash
conda env create -f environment.yml
conda activate facial_recognition
```

**Option B: pip + venv**
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

**Verify:**
```bash
python req.py --check
```
Expected output: `✔ All requirements satisfied`

### Step 3 — (Optional) Prepare Dataset

Place labeled face images in `Dataset/` following the naming convention:
```
Dataset/
├── John Doe.jpg          ← single image, person = "John Doe"
├── Jane Smith_0.jpg      ← numbered variants (underscore + digit)
├── Jane Smith_1.jpg
└── Brad Pitt (001).jpg   ← parenthesised numbering also supported
```
Skip this step if using the pre-built checkpoint.

### Step 4 — Use Pre-built Checkpoint (skip training)

The checkpoint `face_encodings.pkl` is already in the repository. Skip to Step 6 to run inference, or Step 5 to reproduce evaluation metrics.

### Step 5 — Reproduce Evaluation Metrics

```bash
python evaluate_model.py \
    --dataset Dataset \
    --encodings face_encodings.pkl \
    --threshold 0.45 \
    --seed 42 \
    --test_fraction 0.2
```

Expected output summary:
```
Top-1 accuracy (with reject): 0.983
Top-3 SVM class accuracy:     0.996
Accepted by distance gate:    463/471
Macro    precision/recall/F1: 0.938 / 0.919 / 0.927
Weighted precision/recall/F1: 0.998 / 0.983 / 0.990
```

### Step 6 — Re-train from scratch (optional)

```bash
python train_model.py \
    --dataset Dataset \
    --output face_encodings.pkl \
    --workers 4 \
    --jitters 1
```

Training time: ~10–20 minutes on a quad-core CPU for ~2000 images.

### Step 7 — Run the Web Application

```bash
python run_all.py --run --port 5000
```

Or simply double-click `start.bat` on Windows.

Open **http://localhost:5000** and log in with:
- Admin: `admin` / `admin123`
- Faculty: `faculty` / `faculty123`

---

## Artifact Inventory

| Artifact | Path | Description | Size |
|----------|------|-------------|------|
| Main app | `app.py` | Flask web server + SQLite attendance | ~45 KB |
| Training | `train_model.py` | Face gallery builder + SVM trainer | ~8.5 KB |
| Evaluation | `evaluate_model.py` | Held-out metrics, confusion matrix | ~15 KB |
| Dataset split | `dataset_split.py` | Reproducible train/test split | ~5.5 KB |
| Orchestrator | `run_all.py` | CLI to run all stages | ~10 KB |
| Checkpoint | `face_encodings.pkl` | Pre-built gallery + SVM model | ~5.8 MB |
| Requirements | `requirements.txt` | pip dependencies | <1 KB |
| Conda env | `environment.yml` | Conda environment spec | <1 KB |
| Templates | `templates/*.html` | Web UI (admin, faculty, login) | ~37 KB total |
| Startup | `start.ps1`, `start.bat` | Windows launcher | <3 KB |

---

## Known Limitations

- The face gallery includes only the identities enrolled at training time; unknown faces are rejected via a distance threshold (0.45).
- Recognition accuracy may degrade on faces with strong occlusions, extreme lighting, or significant age gap from training images.
- The SQLite database (`attendance.db`) is not committed to avoid personal data exposure; it is auto-created on first run.

---

## Contact

For reproducibility issues, please open a GitHub Issue at:  
https://github.com/teerth-iitj/Automated_Facial_Recognition_System/issues
