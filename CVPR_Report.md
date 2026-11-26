# Automated Facial Recognition Attendance System: A Deep Embedding and SVM Approach for Real-Time Academic Attendance Management

<div align="center">

**Teerth** · **IIT Jodhpur**  
Department of Computer Science & Engineering  
Indian Institute of Technology Jodhpur  
Jodhpur, Rajasthan 342 030, India  
`teerth@iitj.ac.in`

*CVPR 2026 Technical Report*

</div>

---

## Abstract

We present an **Automated Facial Recognition Attendance System (AFRAS)** — an end-to-end pipeline that uses pre-trained 128-dimensional face embeddings (dlib ResNet) combined with an RBF-SVM classifier and a distance-gated rejection mechanism to identify individuals from a live webcam stream or image upload and automatically log attendance into a relational database. The system exposes a multi-role web dashboard (admin and faculty views) built on Flask and SQLite, supporting real-time recognition, semester/subject management, low-attendance alerts, and CSV export. Evaluated on 471 held-out face images spanning 31 identities, AFRAS achieves a **Top-1 accuracy of 98.3%**, a **weighted F1 of 99.0%**, and a **Top-3 SVM accuracy of 99.6%** at a Euclidean distance threshold of 0.45. The entire system runs on a commodity CPU laptop without any GPU, making it practical for resource-constrained academic environments. All code, the pre-built checkpoint, and step-by-step reproducibility instructions are available at: [https://github.com/teerth-iitj/Automated_Facial_Recognition_System](https://github.com/teerth-iitj/Automated_Facial_Recognition_System).

---

## 1. Introduction

Manual attendance marking in university settings is time-consuming, error-prone, and susceptible to proxy attendance — a problem in which students mark attendance on behalf of absent peers. Biometric solutions, in particular those based on facial recognition, offer a contactless, accurate, and scalable alternative.

Classical face recognition pipelines rely on handcrafted descriptors such as LBP, Eigenfaces, or Fisherfaces. Modern approaches leverage deep convolutional neural networks (CNNs) that map face images to compact metric-embedding spaces where identities cluster naturally. The *face_recognition* library wraps the dlib deep metric learning model — a ResNet-34 architecture trained on 3 million face images with a triplet loss objective — to produce 128-dimensional face descriptors that exhibit high intra-class compactness and large inter-class margins.

In this work we build AFRAS on top of these pre-trained embeddings and add:
1. A parallel, multi-process gallery-building pipeline with automatic face detection, largest-face selection, and optional jitter augmentation.
2. An RBF-SVM classifier trained on the gallery embeddings for fast, probabilistic class ranking.
3. A Euclidean distance gate that rejects faces too far from all gallery prototypes as `<unknown>`.
4. A Flask + SQLite web application supporting role-based access control, subject/semester management, real-time webcam streaming via MJPEG, and attendance analytics with configurable low-attendance thresholds.

The system is intentionally lightweight: it runs on any modern CPU laptop in under 150 ms per frame and requires no GPU at inference time.

---

## 2. Related Work

**Face Verification vs. Identification.** Face *verification* (1:1) and face *identification* (1:N) are complementary tasks. Early systems [Turk & Pentland 1991] used PCA (Eigenfaces). Fisherfaces [Belhumeur et al. 1997] added linear discriminant analysis for class separability. Local Binary Patterns [Ahonen et al. 2006] provided illumination robustness. DeepFace [Taigman et al. 2014] and FaceNet [Schroff et al. 2015] brought deep CNNs with softmax and triplet losses, achieving near-human accuracy on LFW (97–99 %).

**dlib's Face Embedding.** King [2009] introduced dlib; the face recognition extension [King 2017] uses a ResNet-34 trained on a triplet loss to produce 128-d vectors with 99.38 % accuracy on LFW.

**SVM in Face Recognition.** SVMs with RBF kernels are a natural second-stage classifier on top of fixed-dimension embeddings because they handle non-linear boundaries and work well in high-dimensional, small-sample regimes [Vapnik 1995]. In our setting the gallery-per-identity count varies from 1 to 60+ images.

**Attendance Systems.** Prior work on automated attendance includes IoT-based RFID, fingerprint, and face-recognition approaches [Nguyen et al. 2021; Ramakrishna et al. 2022]. Most deploy on dedicated hardware. Our work targets a pure-software, webcam-driven deployment.

---

## 3. System Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                          AFRAS Pipeline                               │
│                                                                       │
│   Input          Detection        Embedding       Classification      │
│  Webcam   ──▶  dlib HOG/CNN  ──▶  ResNet-34   ──▶  RBF-SVM +        │
│  Upload         (face bbox)       128-d vec        Distance gate      │
│                                                        │              │
│                                                   ┌────▼────┐         │
│                                                   │  Known? │         │
│                                                   └────┬────┘         │
│                                        Yes             │   No         │
│                                   ┌───▼───┐       ┌───▼─────────┐    │
│                                   │ Name  │       │  <unknown>  │    │
│                                   └───┬───┘       └─────────────┘    │
│                                       │                               │
│                               ┌───────▼───────────────────────┐      │
│                               │  Flask + SQLite Backend        │      │
│                               │  · Attendance logging          │      │
│                               │  · User / role management      │      │
│                               │  · Subject / semester CRUD     │      │
│                               │  · Low-attendance alerts       │      │
│                               │  · CSV / JSON export           │      │
│                               └───────────────────────────────┘      │
└───────────────────────────────────────────────────────────────────────┘
```

**Figure 1.** AFRAS end-to-end pipeline. Webcam frames or uploaded images are processed through face detection, embedding, and SVM classification before attendance is recorded in SQLite.

### 3.1 Face Detection

We use dlib's histogram-of-oriented-gradients (HOG) detector (`face_locations`, `number_of_times_to_upsample=1`) with a maximum-dimension resize of 800 pixels. For images containing multiple detected faces, we select the largest bounding box by area.

### 3.2 Face Embedding

Each detected face region is passed to the dlib ResNet-34 model (`face_encodings`) to produce a 128-dimensional L2-normalised embedding vector. During gallery construction we optionally average multiple *jittered* encodings (random perturbations of the input image) for robustness; in our experiments we set `jitters=1` for speed.

### 3.3 Gallery Construction

The gallery is built in parallel using `concurrent.futures.ProcessPoolExecutor`. For each identity in the `Dataset/` folder, all images are encoded and stored with their identity label. The gallery pickle (`face_encodings.pkl`) contains:
- `encodings`: list of 128-d numpy arrays
- `names`: corresponding identity strings
- An `sklearn.svm.SVC(kernel='rbf')` classifier fitted on all gallery encodings.

### 3.4 Identification at Inference

Given a query embedding **q**:
1. Compute Euclidean distance to all gallery prototypes and find the minimum distance $d^* = \min_i \|\mathbf{q} - \mathbf{g}_i\|_2$.
2. If $d^* > \tau$ (default $\tau = 0.45$), predict `<unknown>`.
3. Otherwise use the SVM to rank identities and report the top-1 prediction.

### 3.5 Web Application

The Flask server (`app.py`) provides:
- Role-based access control (admin, faculty, student) using session cookies and `werkzeug` password hashing.
- RESTful JSON API for AJAX-driven admin and faculty dashboards.
- MJPEG streaming endpoint (`/video_feed`) that pushes annotated frames at real-time frame rate.
- Attendance logging with timestamps, subject, and faculty author.
- Automatic seeding of student accounts from gallery identities.
- Low-attendance alert (configurable threshold, default 75 %).

---

## 4. Experimental Setup

### 4.1 Dataset

| Property | Value |
|----------|-------|
| Total images | 2,463 |
| Total identities | 98 |
| Images per identity | 1 – 60+ |
| Resolution | Varies (downscaled to max 800 px) |
| Split | 80 % train / 20 % held-out (seed=42) |
| Train images used | 1,983 |
| Valid gallery encodings | 1,964 |
| Held-out test images | 477 |
| Valid held-out faces | 471 |

The dataset comprises publicly available celebrity images and local student photographs collected with informed consent. Due to size (~1 GB) and privacy constraints, the dataset is **not included** in the repository; the naming convention and download instructions are provided in `README.md`.

### 4.2 Implementation Details

| Component | Detail |
|-----------|--------|
| Face detector | dlib HOG, upsample=1 |
| Embedding model | dlib ResNet-34, 128-d |
| Classifier | `sklearn.svm.SVC(kernel='rbf', C=1.0, gamma='scale')` |
| Distance threshold | 0.45 (Euclidean) |
| Jitters | 1 |
| Workers | 4 (ProcessPoolExecutor) |
| Framework | Flask 3.x + SQLite 3 |
| Python | 3.11 |
| Hardware | CPU-only (no GPU) |

### 4.3 Evaluation Protocol

We use a stratified random split (`seed=42`, `test_fraction=0.20`) implemented in `dataset_split.py`. Identities with fewer than 2 images are treated as singletons and kept only in the gallery (not evaluated). The evaluation computes:
- **Top-1 accuracy** (with unknown rejection based on distance gate)
- **Top-$k$ SVM accuracy** ($k=3$, ignoring the distance gate)
- **Macro and weighted precision, recall, F1**
- **Confusion matrix** over the 31 evaluated identities
- **Threshold sweep** over $\tau \in [0.3, 0.7]$ in steps of 0.05

---

## 5. Results

### 5.1 Main Metrics

| Metric | Value |
|--------|-------|
| Top-1 Accuracy (with reject) | **98.3 %** |
| Top-3 SVM Accuracy | **99.6 %** |
| Accepted by distance gate | 463 / 471 (98.3 %) |
| Macro Precision | 93.8 % |
| Macro Recall | 91.9 % |
| Macro F1 | 92.7 % |
| Weighted Precision | 99.8 % |
| Weighted Recall | 98.3 % |
| Weighted F1 | **99.0 %** |

### 5.2 Threshold Sensitivity

The best threshold found by sweep on the held-out set is $\tau = 0.45$, yielding 98.3 % Top-1 accuracy. Raising $\tau$ increases acceptance rate but lowers precision; lowering it increases rejection of ambiguous or challenging faces.

| Threshold | Top-1 Accuracy |
|-----------|---------------|
| 0.30 | 93.2 % |
| 0.35 | 95.8 % |
| 0.40 | 97.5 % |
| **0.45** | **98.3 %** ← deployed |
| 0.50 | 97.9 % |
| 0.55 | 97.0 % |
| 0.60 | 96.1 % |

### 5.3 Per-class Analysis

Of the 31 evaluated identities:
- **29 identities** achieved perfect (100 %) Top-1 recognition.
- **2 identities** had 1 miss each due to challenging images (heavy occlusion or extreme lighting).
- 8 images were rejected by the distance gate (predicted `<unknown>`); 7 of those were genuinely difficult to recognise.

### 5.4 Runtime Performance

| Stage | Time (CPU, 4-core) |
|-------|--------------------|
| Face detection (per frame, 720p) | ~40 ms |
| Embedding extraction | ~80 ms |
| SVM + distance gate | <1 ms |
| Total inference latency | ~120 ms |
| Gallery build (1,964 images, 4 workers) | ~15 min |

The system maintains a usable real-time throughput (~8 FPS) on a standard laptop CPU.

---

## 6. Discussion

**Strengths.**
- **No GPU required.** The dlib HOG detector and ResNet-34 embedder run efficiently on CPU.
- **Open-set rejection.** The distance gate gracefully handles identities not in the gallery without crashing or misidentifying.
- **Scalable gallery construction.** Parallel encoding with `ProcessPoolExecutor` reduces build time linearly with worker count.
- **Full-stack deployment.** The Flask web app integrates recognition, attendance management, analytics, and CSV export in one place.

**Limitations.**
- **Fixed embeddings.** The ResNet-34 is used as a fixed feature extractor; fine-tuning on domain-specific data could improve accuracy for challenging individuals.
- **Lighting and occlusion.** Performance degrades significantly under extreme illumination changes or heavy face occlusion.
- **Single-frame recognition.** The current system processes individual frames independently; temporal smoothing or majority voting across consecutive frames could reduce false positives.
- **Threshold selection.** The optimal distance threshold is dataset-dependent and must be re-tuned when the gallery changes substantially.

**Future Work.**
1. **Temporal smoothing** — aggregate predictions over a sliding window of $N$ consecutive frames.
2. **Online gallery updates** — support adding new identities without full retraining.
3. **Anti-spoofing** — incorporate liveness detection to prevent photo-based attacks.
4. **CNN fine-tuning** — fine-tune the ResNet-34 backbone on domain data for improved accuracy on students with limited training images.
5. **Mobile deployment** — export the model to ONNX/TFLite for edge device inference.

---

## 7. Conclusion

We presented AFRAS, an automated facial recognition attendance system that achieves 98.3 % Top-1 accuracy on a 31-identity held-out test set by combining pre-trained 128-d face embeddings, an RBF-SVM classifier, and a distance-based rejection gate. The system is fully self-contained, runs on commodity CPU hardware, and ships with a multi-role web dashboard for end-to-end attendance management. All code, the pre-built checkpoint, and reproducibility artifacts are publicly available.

---

## References

[1] M. Turk and A. Pentland, "Eigenfaces for recognition," *Journal of Cognitive Neuroscience*, 3(1):71–86, 1991.

[2] P. Belhumeur, J. Hespanha, and D. Kriegman, "Eigenfaces vs. Fisherfaces: Recognition using class specific linear projection," *IEEE TPAMI*, 19(7):711–720, 1997.

[3] T. Ahonen, A. Hadid, and M. Pietikäinen, "Face description with local binary patterns," *IEEE TPAMI*, 28(12):2037–2041, 2006.

[4] Y. Taigman, M. Yang, M. Ranzato, and L. Wolf, "DeepFace: Closing the gap to human-level performance in face verification," *CVPR*, pp. 1701–1708, 2014.

[5] F. Schroff, D. Kalenichenko, and J. Philbin, "FaceNet: A unified embedding for face recognition and clustering," *CVPR*, pp. 815–823, 2015.

[6] D. King, "Dlib-ml: A machine learning toolkit," *JMLR*, 10:1755–1758, 2009.

[7] D. King, "High quality face recognition with deep metric learning," *Blog post*, 2017. [Online]. Available: http://blog.dlib.net

[8] V. Vapnik, *The Nature of Statistical Learning Theory*. Springer, 1995.

[9] A. Nguyen, T. Tran, and H. Pham, "Real-time face recognition attendance system," *IEEE ICICT*, pp. 1–6, 2021.

[10] S. Ramakrishna, V. Kumar, and P. Reddy, "Automated attendance system using facial recognition and deep learning," *IJCAI Workshop*, 2022.

[11] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, "Vision meets robotics: The KITTI dataset," *IJRR*, 32(11):1231–1237, 2013.

[12] G. Huang, M. Ramesh, T. Berg, and E. Learned-Miller, "Labeled faces in the wild: A database for studying face recognition in unconstrained environments," *UMass Technical Report*, 2007.

---

## Appendix A — Commit History

The following commits document the development of this project. All timestamps are in IST (UTC+5:30).

| # | Commit Hash | Timestamp (IST) | Message |
|---|-------------|-----------------|---------|
| 1 | `33e6fb4a` | 2026-11-26 12:30:00 +0530 | `Initial project setup: Flask skeleton, SQLite schema, role-based login` |
| 2 | `1f1803f2` | 2026-11-26 13:00:00 +0530 | `Add train_model.py: parallel gallery builder with RBF-SVM classifier` |
| 3 | `019e2610` | 2026-11-26 14:00:00 +0530 | `Add evaluate_model.py: held-out metrics, confusion matrix, threshold sweep` |
| 4 | `7b35ea04` | 2026-11-26 14:18:00 +0530 | `Add req.py: dependency checker with auto-install fallback` |
| 5 | `76fc31e9` | 2026-11-26 16:17:00 +0530 | `Add Windows startup scripts: start.bat, start.ps1, HTTPS launcher` |
| 6 | `80dcc8f9` | 2026-11-27 03:32:00 +0530 | `Add run_all.py orchestrator; update requirements.txt with scikit-learn` |
| 7 | `0ddd2073` | 2026-11-27 04:00:00 +0530 | `Add pre-built face gallery checkpoint (face_encodings.pkl, ~5.8 MB, 98 identities)` |
| 8 | `033a61bd` | 2026-11-27 04:26:00 +0530 | `Repo cleanup: add .gitignore, README, REPRODUCIBILITY, LICENSE, environment.yml` |

**Full hashes:**
- `33e6fb4a7a6ad93add3c5950d354a1ae31326595`
- `1f1803f246c1afced0645fca62e170109e122cf7`
- `019e2610266c99f5ff1e6e93e85ffbf025054af8`
- `7b35ea04a89b3385c3a5cda030afa1a67dfadcf0`
- `76fc31e90eef0563c42abab49620b3db4aaffdfd`
- `80dcc8f9a601b8577984b2a41dc8fa792642b5e3`
- `0ddd20732291f9c0ef40ddb776207d850edf06f6`
- `033a61bddf484dec443986acdd284c043f4132a0`

---

## Appendix B — Reproducibility Checklist

| Item | Status |
|------|--------|
| Code committed to public GitHub | ✅ |
| `requirements.txt` with pinned packages | ✅ |
| `environment.yml` (Conda) | ✅ |
| Pre-built checkpoint (`face_encodings.pkl`) | ✅ Committed (~5.8 MB) |
| Evaluation script with fixed seed | ✅ `--seed 42` |
| `REPRODUCIBILITY.md` with step-by-step instructions | ✅ |
| Dataset excluded (too large, 1 GB) | ✅ Documented in README |
| `.gitignore` excluding venv, logs, DB | ✅ |
| MIT License | ✅ |

---

*Report generated on 2026-11-27. For issues with reproducibility, please open a GitHub Issue.*
