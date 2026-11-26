# Automated Facial Recognition Attendance System

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An end-to-end automated attendance management system that uses **face recognition** (dlib/face_recognition) with an **SVM classifier** to identify students and mark attendance through a web dashboard. Built for academic institutions.

---

## 📁 Repository Structure

```
facial_recognition/
├── app.py                    # Main Flask web application
├── train_model.py            # Build face gallery + train SVM classifier
├── evaluate_model.py         # Held-out evaluation with metrics & confusion matrix
├── dataset_split.py          # Train/test split utilities
├── run_all.py                # CLI orchestrator (train / evaluate / serve)
├── req.py                    # Dependency checker & auto-installer
├── requirements.txt          # pip requirements
├── environment.yml           # Conda environment definition
├── face_encodings.pkl        # Pre-built face gallery (checkpoint, ~5.8 MB)
├── start.ps1                 # One-click Windows startup script
├── start.bat                 # Batch wrapper for start.ps1
├── run_https_5443.ps1        # HTTPS server launcher
├── templates/
│   ├── base.html             # Shared layout
│   ├── login.html            # Login page
│   ├── admin_dashboard.html  # Admin panel (users, subjects, attendance)
│   └── faculty_dashboard.html# Faculty panel (take attendance, view reports)
├── Dataset/                  # ⚠ NOT tracked (see Dataset section below)
├── REPRODUCIBILITY.md        # Declaration + step-by-step reproducibility guide
└── .gitignore
```

---

## 🚀 Quick Start

### Prerequisites
- **Windows 10/11** (primary platform; Linux/macOS should work with minor tweaks)
- Python 3.8–3.12 (3.11 recommended)
- A webcam for live recognition, or image uploads

### Option A — One-click (Windows)
```bat
start.bat
```
This creates a `.venv`, installs all dependencies (including `dlib-bin`), and launches the Flask app in your browser.

### Option B — Manual (all platforms)

#### 1. Clone
```bash
git clone https://github.com/teerth-iitj/Automated_Facial_Recognition_System.git
cd Automated_Facial_Recognition_System
```

#### 2. Create environment
**Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate facial_recognition
```

**pip:**
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

#### 3. Obtain Dataset (see below)

#### 4. Build the face gallery (training)
```bash
python train_model.py --dataset Dataset --output face_encodings.pkl --jitters 1
```
Or use the pre-built checkpoint `face_encodings.pkl` already in the repo (skip this step).

#### 5. (Optional) Evaluate
```bash
python evaluate_model.py --dataset Dataset --encodings face_encodings.pkl --threshold 0.45
```

#### 6. Launch the web app
```bash
python run_all.py --run --port 5000
```
Open **http://localhost:5000** in your browser.

Default credentials:
| Role    | Username | Password    |
|---------|----------|-------------|
| Admin   | admin    | admin123    |
| Faculty | faculty  | faculty123  |

---

## 📂 Dataset

The dataset is **not included** in this repository because it totals ≈ 1 GB (2,463 images).

**Structure expected** (`Dataset/` folder in repo root):
```
Dataset/
├── Person Name.jpg              # single image → one identity
├── Person Name_0.jpg            # numbered variants of the same identity
├── Another Person (001).jpg     # parenthesised numbering also supported
└── ...
```

**Sources used in this project:**
- Publicly available celebrity images scraped from Google Images (used for academic, non-commercial research only)
- Local student images collected with informed consent

If you want to reproduce with **your own dataset**, simply populate `Dataset/` with sub-folders or flat files following the naming convention above, then re-run `train_model.py`.

---

## 🧠 Checkpoint

The pre-built face gallery and SVM classifier are stored in **`face_encodings.pkl`** (~5.8 MB) and are **committed to this repository**.

This checkpoint was produced by running:
```bash
python train_model.py --dataset Dataset --output face_encodings.pkl --workers 4 --jitters 1
```
on 1,964 images across 98 identities.  
Evaluation on 20 % held-out images:
- **Top-1 accuracy**: 98.3 %
- **Top-3 SVM accuracy**: 99.6 %
- **Distance threshold**: 0.45

For a larger model checkpoint (e.g., fine-tuned dlib models), please refer to the [HuggingFace model hub](https://huggingface.co/ageitgey/face_recognition) for the underlying `dlib` and `face_recognition_models` weights.

---

## 🔄 Reproducibility

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full declaration and step-by-step guide.

---

## 🏗️ Architecture

```
Webcam / Upload
      │
      ▼
face_recognition (dlib HOG/CNN)
   ● Face detection
   ● 128-d embedding
      │
      ▼
  SVM Classifier (RBF kernel)
      │
   Known?  ──No──▶  <unknown>
      │Yes
      ▼
Flask Web App (SQLite)
  ● Session management
  ● Attendance logging
  ● Admin / Faculty dashboards
```

---

## 📊 Evaluation Results

| Metric | Value |
|---|---|
| Top-1 Accuracy | 98.3 % |
| Top-3 Accuracy | 99.6 % |
| Macro F1 | 92.7 % |
| Weighted F1 | 99.0 % |
| Distance threshold | 0.45 |
| Test set size | 471 faces |
| Identities evaluated | 31 |

---

## 🛠️ CLI Reference

```bash
# Train
python run_all.py --train [--dataset Dataset] [--output face_encodings.pkl] [--workers 4]

# Evaluate
python run_all.py --evaluate [--threshold 0.45]

# Serve
python run_all.py --run [--port 5000] [--https]

# All-in-one
python run_all.py --all --port 5000
```

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📄 License

MIT © 2026 — IIT Jodhpur
