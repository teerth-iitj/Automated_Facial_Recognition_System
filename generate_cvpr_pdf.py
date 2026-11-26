import os
import subprocess
import pypdf

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Automated Facial Recognition Attendance System: A Deep Metric and Support Vector Machine Approach for Real-Time Academic Attendance Management</title>
<style>
  @page {
    size: letter;
    margin: 0.75in 0.75in 0.85in 0.75in;
    @bottom-center {
      content: counter(page);
      font-family: 'Times New Roman', Times, serif;
      font-size: 10pt;
    }
  }

  * {
    box-sizing: border-box;
  }

  body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 9.8pt;
    line-height: 11.8pt;
    color: #000000;
    margin: 0;
    padding: 0;
    text-align: justify;
    text-justify: inter-word;
  }

  .paper-header {
    text-align: center;
    margin-bottom: 12pt;
  }

  h1.paper-title {
    font-size: 14pt;
    font-weight: bold;
    line-height: 16.5pt;
    margin: 0 0 8pt 0;
    text-align: center;
  }

  .author-block {
    font-size: 10.5pt;
    margin-bottom: 4pt;
    font-weight: normal;
  }

  .author-name {
    font-weight: bold;
  }

  .institution {
    font-size: 9.2pt;
    color: #222;
    margin-bottom: 4pt;
  }

  .email {
    font-family: 'Courier New', Courier, monospace;
    font-size: 8.8pt;
  }

  .cvpr-tag {
    font-size: 8.8pt;
    font-style: italic;
    color: #444;
    margin-top: 4pt;
    margin-bottom: 10pt;
  }

  .abstract-box {
    margin: 0 auto 12pt auto;
    width: 93%;
    font-size: 9pt;
    line-height: 11.2pt;
    text-align: justify;
    border-top: 0.6pt solid #000;
    border-bottom: 0.6pt solid #000;
    padding: 7pt 0;
  }

  .abstract-title {
    font-weight: bold;
    font-size: 9.2pt;
    display: inline;
  }

  .two-column {
    column-count: 2;
    column-gap: 0.30in;
    column-fill: balance;
  }

  h2.sec-heading {
    font-size: 10pt;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.4pt;
    margin: 9.5pt 0 3.5pt 0;
    break-after: avoid;
  }

  h3.subsec-heading {
    font-size: 9.5pt;
    font-weight: bold;
    font-style: italic;
    margin: 6.5pt 0 2.5pt 0;
    break-after: avoid;
  }

  p {
    margin: 0 0 5pt 0;
    text-indent: 10pt;
  }

  p.no-indent {
    text-indent: 0;
  }

  .equation {
    text-align: center;
    margin: 5pt 0;
    font-style: italic;
    font-size: 9.2pt;
  }

  .eq-num {
    float: right;
    font-style: normal;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 6.5pt 0;
    font-size: 7.9pt;
    line-height: 10pt;
  }

  th {
    border-top: 1pt solid #000;
    border-bottom: 0.6pt solid #000;
    padding: 2.8pt 3pt;
    text-align: left;
    font-weight: bold;
  }

  td {
    padding: 2.2pt 3pt;
    text-align: left;
    border-bottom: 0.4pt solid #e0e0e0;
  }

  tr.bottom-rule td {
    border-bottom: 1pt solid #000;
  }

  .table-caption {
    font-size: 8.2pt;
    font-weight: normal;
    margin-bottom: 2pt;
    text-align: left;
  }

  .figure-box {
    margin: 6.5pt 0;
    padding: 5.5pt;
    background: #fdfdfd;
    border: 0.5pt solid #bbb;
    border-radius: 2pt;
    font-size: 7.9pt;
    line-height: 9.8pt;
    break-inside: avoid;
  }

  .figure-caption {
    font-size: 8.2pt;
    margin-top: 3.5pt;
    text-align: justify;
  }

  .figure-title {
    font-weight: bold;
  }

  pre.code-box {
    font-family: 'Courier New', Courier, monospace;
    font-size: 7.2pt;
    line-height: 8.8pt;
    background: #f8f8f8;
    padding: 4.5pt;
    margin: 3.5pt 0;
    border: 0.4pt solid #ccc;
    overflow: hidden;
  }

  .reference-list {
    font-size: 7.8pt;
    line-height: 9.6pt;
    padding-left: 12pt;
    margin: 4.5pt 0;
  }

  .reference-list li {
    margin-bottom: 3.5pt;
    text-align: justify;
  }

  .page-num {
    text-align: center;
    font-size: 9pt;
    margin-top: 10pt;
  }

  .break-page {
    break-before: page;
  }
</style>
</head>
<body>

<div class="paper-header">
  <h1 class="paper-title">Automated Facial Recognition Attendance System:<br>A Deep Metric and Support Vector Machine Approach for Real-Time Academic Attendance Management</h1>
  
  <div class="author-block">
    <span class="author-name">Teerth</span>
  </div>
  <div class="institution">
    Department of Computer Science and Engineering<br>
    Indian Institute of Technology Jodhpur, Rajasthan 342030, India
  </div>
  <div class="email">teerth@iitj.ac.in</div>
  <div class="cvpr-tag">Technical Report &mdash; Computer Vision and Pattern Recognition (CVPR) Track</div>

  <div class="abstract-box">
    <span class="abstract-title">Abstract&mdash;</span>
    Manual attendance collection in large university lecture environments introduces significant instructional overhead, administrative latency, and acute vulnerability to proxy fabrication. While biometric hardware approaches exist, specialized hardware sensor deployments incur prohibitive procurement costs, maintenance bottlenecks, and physical queuing delays. In this investigation, we introduce the Automated Facial Recognition Attendance System (AFRAS), a real-time, end-to-end framework integrating deep metric embeddings, multi-class Support Vector Machines (SVM), and an adaptive Euclidean rejection gate. AFRAS extracts 128-dimensional L2-normalized representations using a deep ResNet-34 backbone pre-trained via triplet loss optimization. To address identity discrimination over an open gallery of enrolled individuals while categorically rejecting non-enrolled candidates, we construct a dual-stage decision mechanism combining an RBF-kernel SVM classifier with an empirical distance-gated threshold (&tau; = 0.45). The system is coupled to an asynchronous, multi-role Flask and SQLite web dashboard supporting automated student ingestion, real-time webcam frame acquisition, multi-subject scheduling, and low-attendance alerts. Rigorous empirical evaluation across 471 held-out test face images from 31 evaluated identities demonstrates a Top-1 recognition accuracy of 98.3%, a Top-3 SVM accuracy of 99.6%, and a weighted F1-score of 99.0%. Crucially, the entire pipeline operates on commodity multi-core CPUs without requiring dedicated GPU acceleration, maintaining an inference latency of approximately 120 ms per frame. The complete repository, deterministic evaluation protocols, pre-built gallery checkpoint, and reproducibility artifacts are publicly released at https://github.com/teerth-iitj/Automated_Facial_Recognition_System.
  </div>
</div>

<div class="two-column">

<h2 class="sec-heading">1. Introduction</h2>
<p>
Academic attendance verification remains an essential administrative requirement across educational institutions globally, serving as a critical metric for student engagement, accreditation compliance, financial aid eligibility, and academic evaluation. Nevertheless, traditional modalities such as paper sheet signing and manual verbal roll-calls suffer from acute operational limitations:
</p>
<p class="no-indent">
1. <i>Temporal Inefficiency:</i> In large-cohort lectures comprising 100 to 300 students, roll-call procedures consume between 10% and 15% of instructional contact time.<br>
2. <i>Susceptibility to Proxy Attendance:</i> Physical signature registers are notoriously vulnerable to peer forgery, compromising institutional record integrity.<br>
3. <i>Administrative Bottlenecks:</i> Transcribing handwritten rosters into central database systems introduces human transcription error and substantial reporting delays.
</p>
<p>
Biometric interventions, notably fingerprint scanning and RFID cards, offer partial remedies but introduce alternate operational vulnerabilities. Fingerprint scanners suffer from mechanical wear, sanitary concerns during seasonal outbreaks, and queuing bottlenecks at lecture hall entrances. RFID cards are readily shared among students, failing to eliminate the proxy phenomenon.
</p>
<p>
Computer vision methodologies provide a non-intrusive, contactless, and auditable alternative. While modern deep convolutional neural networks (CNNs) have advanced facial verification benchmarks beyond human-level capabilities on constrained benchmarks, deploying these architectures in institutional classroom environments presents practical challenges:
</p>
<p class="no-indent">
&bull; <i>Small-Sample Gallery Regimes:</i> Educational rosters typically provide only one to a few enrollment photographs per student, prohibiting end-to-end backpropagation training.<br>
&bull; <i>Hardware Constraints:</i> Academic institutions cannot feasibly provision high-end GPU workstations in every seminar room and lecture hall.<br>
&bull; <i>Open-Set Rejection:</i> The system must robustly identify legitimate enrolled students while categorically refusing visitors, faculty, or unregistered candidates as <code>&lt;unknown&gt;</code>.
</p>
<p>
To resolve these interrelated challenges, this paper presents <b>AFRAS</b> (Automated Facial Recognition Attendance System). AFRAS decouples deep metric feature representation from non-linear decision boundary formulation, pairing a frozen ResNet-34 metric extractor with a multi-class Support Vector Machine (SVM) equipped with a Radial Basis Function (RBF) kernel and a Euclidean minimum-distance gating operator.
</p>

<h2 class="sec-heading">2. Related Work</h2>

<h3 class="subsec-heading">2.1. Classical Biometric Descriptors</h3>
<p>
Early facial recognition methodologies depended upon linear subspace transformations. Turk and Pentland [1] formulated Eigenfaces via Principal Component Analysis (PCA), projecting pixel-space covariance into orthogonal eigenvectors capturing maximal variance. Belhumeur et al. [2] expanded upon this foundation with Fisherfaces, utilizing Linear Discriminant Analysis (LDA) to maximize the ratio of between-class scatter to within-class scatter. Ahonen et al. [3] proposed Local Binary Patterns Histograms (LBPH), evaluating micro-spatial texture features robust against monotonic grayscale variations. While computationally frugal, these shallow handcrafted features exhibit acute vulnerability to unconstrained pose rotations, severe ambient illumination gradients, and facial expressions.
</p>

<h3 class="subsec-heading">2.2. Deep Metric Representation Learning</h3>
<p>
The advent of deep convolutional networks revolutionized biometric vision. Taigman et al. [4] presented DeepFace, employing piecewise affine alignment and large-scale classification cross-entropy over 4 million face exemplars. Schroff, Kalenichenko, and Philbin [5] introduced FaceNet, discarding classification cross-entropy in favor of direct metric learning using the triplet loss function:
</p>
<div class="equation">
  &sum;<sub>i</sub><sup>N</sup> [ ||f(x<sub>i</sub><sup>a</sup>) - f(x<sub>i</sub><sup>p</sup>)||<sub>2</sub><sup>2</sup> - ||f(x<sub>i</sub><sup>a</sup>) - f(x<sub>i</sub><sup>n</sup>)||<sub>2</sub><sup>2</sup> + &alpha; ]<sub>+</sub>
  <span class="eq-num">(1)</span>
</div>
<p class="no-indent">
where <i>x<sub>i</sub><sup>a</sup></i>, <i>x<sub>i</sub><sup>p</sup></i>, and <i>x<sub>i</sub><sup>n</sup></i> represent anchor, positive, and negative face samples respectively, mapped into a hyperspherical manifold of dimension <i>d</i> = 128 constrained by unit L2 norm ||<i>f</i>(<i>x</i>)||<sub>2</sub> = 1, enforced by margin &alpha;.
</p>
<p>
Building upon FaceNet's metric paradigm, King [6, 7] developed the dlib deep face recognition architecture based on a ResNet-34 backbone trained across 3 million face images. This network achieves 99.38% verification accuracy on the standard Labeled Faces in the Wild (LFW) benchmark, providing high-fidelity, compact descriptor vectors that execute efficiently on commodity CPU SIMD instructions. Deng et al. [9] and Wang et al. [10] further explored angular margins (ArcFace, CosFace) on hyperspheres, affirming the superiority of normalized latent metric spaces.
</p>

<h3 class="subsec-heading">2.3. Classification in Latent Embedding Manifolds</h3>
<p>
When mapping test descriptors into enrolled galleries, nearest neighbor (1-NN) Euclidean matching is computationally trivial but sensitive to local centroid drift in unbalanced class distributions. Support Vector Machines [8, 14] construct optimal separating hyperplanes maximizing the functional margin:
</p>
<div class="equation">
  min<sub>w,b,&xi;</sub> (1/2)||w||<sup>2</sup> + C &sum;<sub>i</sub> &xi;<sub>i</sub>
  <span class="eq-num">(2)</span>
</div>
<p class="no-indent">
subject to <i>y<sub>i</sub>(w &middot; &phi;(x<sub>i</sub>) + b) &ge; 1 - &xi;<sub>i</sub></i>. By projecting 128-d unit vectors through an RBF kernel <i>K(u, v) = exp(-&gamma;||u - v||<sup>2</sup>)</i>, non-linear class partitions can be established among densely packed identity clusters, outperforming simple cosine or Euclidean centroids.
</p>

<h3 class="subsec-heading">2.4. Smart Attendance Architectures</h3>
<p>
Multiple investigators have explored automated attendance systems [11, 12]. However, prior frameworks frequently rely on monolithic client-side scripts, lack multi-role enterprise security abstractions, fail to address open-set rejection rigorously, or require persistent GPU cloud connections incompatible with institutional air-gapped deployments. AFRAS addresses each of these architectural shortcomings directly.
</p>

<h2 class="sec-heading">3. System Architecture and Methodology</h2>

<div class="figure-box">
  <div style="text-align: center; font-family: monospace; font-size: 6.8pt; line-height: 8.5pt;">
    [Video Frame / Upload] &rarr; [Max-Dimension Lanczos Downscale (800px)]<br>
    &darr;<br>
    [dlib 5-Point HOG / CNN Face Localization (upsample=1)]<br>
    &darr;<br>
    [ResNet-34 Feature Extractor &rarr; 128-d L2 Unit Vector q &isin; &real;<sup>128</sup>]<br>
    &darr;<br>
    +-------------------------------------------------------------+<br>
    | Dual-Stage Inference Engine                                  |<br>
    | 1. Euclidean Gate: d* = min<sub>i</sub> ||q - g<sub>i</sub>||<sub>2</sub>                         |<br>
    |    IF d* &gt; &tau; (0.45) &rarr; Emit &lt;unknown&gt; (Open-Set Reject)         |<br>
    | 2. RBF-SVM Ranker: y* = argmax<sub>c</sub> &sum;<sub>j</sub> &alpha;<sub>j</sub> y<sub>j</sub> K(s<sub>j</sub>, q) + b       |<br>
    |    IF d* &le; &tau; &rarr; Emit Predicted Identity y*                    |<br>
    +-------------------------------------------------------------+<br>
    &darr;<br>
    [Flask REST Controller &rarr; SQLite3 ACID Attendance Storage]<br>
    &darr;<br>
    [Role-Based Admin / Faculty Dashboard + CSV Export + Warning Engine]
  </div>
  <div class="figure-caption">
    <span class="figure-title">Figure 1.</span> Comprehensive end-to-end dataflow pipeline of AFRAS, demonstrating the dual-stage Euclidean-SVM inference engine and relational database integration.
  </div>
</div>

<h3 class="subsec-heading">3.1. Face Detection and Scale Normalization</h3>
<p>
Input imagery obtained via live MJPEG video streams (HTTP <code>/video_feed</code>) or discrete HTTP multipart uploads exhibits arbitrary resolution, varying from 720p webcam streams to 12-megapixel smartphone photographs. Unconstrained high-resolution inputs severely degrade inference throughput without conferring proportional detection gain. AFRAS applies adaptive dimension scaling:
</p>
<div class="equation">
  s = min(1.0, S<sub>max</sub> / max(H, W))
  <span class="eq-num">(3)</span>
</div>
<p class="no-indent">
where <i>S<sub>max</sub> = 800</i> pixels. Bilinear Lanczos interpolation is utilized to downscale the image matrix while preserving sharp facial feature gradients.
</p>
<p>
Faces are detected using dlib's frontal face directional gradient (HOG) detector combined with a linear SVM window evaluator [16]. To guarantee the recovery of small or distant faces in auditorium back rows, the detector performs single-octave upsampling (<code>number_of_times_to_upsample = 1</code>). For images exhibiting multiple face candidates during gallery compilation, AFRAS resolves ambiguity by computing bounding box area <i>A<sub>k</sub> = (bottom<sub>k</sub> - top<sub>k</sub>) &times; (right<sub>k</sub> - left<sub>k</sub>)</i> and isolating the maximal area candidate:
</p>
<div class="equation">
  B* = argmax<sub>B<sub>k</sub></sub> Area(B<sub>k</sub>)
  <span class="eq-num">(4)</span>
</div>

<h3 class="subsec-heading">3.2. Deep Metric Feature Projection</h3>
<p>
Given normalized face region <i>B*</i>, affine alignment is conducted using 5-point facial landmark coordinates (outer eye corners, nose base, and mouth corners) to rectify in-plane roll rotations. The aligned crop is passed through the pre-trained ResNet-34 deep convolutional backbone [15]. The final dense layer produces an unnormalized vector <i>z &isin; &real;<sup>128</sup></i>, which is mapped to the unit hypersphere:
</p>
<div class="equation">
  q = z / ||z||<sub>2</sub>, &nbsp;&nbsp;&nbsp; ||q||<sub>2</sub> = 1.0
  <span class="eq-num">(5)</span>
</div>
<p class="no-indent">
This normalization ensures that Euclidean distance <i>||u - v||<sub>2</sub></i> and cosine similarity <i>cos(&theta;)</i> maintain a direct monotonic equivalence:
</p>
<div class="equation">
  ||u - v||<sub>2</sub><sup>2</sup> = ||u||<sup>2</sup> + ||v||<sup>2</sup> - 2(u &middot; v) = 2 - 2 cos(u, v)
  <span class="eq-num">(6)</span>
</div>

<h3 class="subsec-heading">3.3. Parallel Gallery Compilation</h3>
<p>
To construct the institutional gallery without blocking system execution, gallery compilation is orchestrated via <code>concurrent.futures.ProcessPoolExecutor</code> over <i>W = min(4, N<sub>cpu</sub>)</i> worker processes. For each file in <code>Dataset/</code>, identities are extracted from canonical filename structures:
</p>
<div class="equation">
  Name = regex_replace(Basename, r'(_\d+|\s*\(\d+\))$', '')
  <span class="eq-num">(7)</span>
</div>
<p class="no-indent">
During feature extraction, an optional jitter augmentation parameter <i>J</i> computes encodings across <i>J</i> random affine perturbations and averages the resulting unit vectors. All valid tuples <i>(q<sub>i</sub>, y<sub>i</sub>)</i> are serialized alongside a fitted RBF-SVM model into <code>face_encodings.pkl</code> (~5.8 MB).
</p>

<div class="figure-box">
  <div class="figure-title">Algorithm 1: Dual-Stage Euclidean-SVM Recognition & Ingestion</div>
  <pre class="code-box">
<b>Require:</b> Query frame <i>I</i>, Gallery embeddings <i>G={g_i}</i>, SVM model <i>M</i>,
         Rejection threshold <i>&tau; = 0.45</i>, Course ID <i>C</i>, Faculty ID <i>F</i>
 1: <i>I_scaled</i> &larr; Resize(<i>I</i>, max_dim=800)
 2: <i>Boxes</i> &larr; FaceLocations(<i>I_scaled</i>, upsample=1)
 3: <b>if</b> |<i>Boxes</i>| == 0 <b>then return</b> []
 4: <i>Results</i> &larr; []
 5: <b>for each</b> <i>b</i> in <i>Boxes</i> <b>do</b>
 6:    <i>q</i> &larr; ExtractEmbedding(<i>I_scaled</i>, <i>b</i>) &isin; &real;<sup>128</sup>, ||<i>q</i>||_2 = 1.0
 7:    <i>d*</i> &larr; min<sub>g &isin; G</sub> ||<i>q</i> - <i>g</i>||_2
 8:    <b>if</b> <i>d*</i> &gt; <i>&tau;</i> <b>then</b>
 9:       <i>Results</i>.append((<i>b</i>, "&lt;unknown&gt;", <i>d*</i>, Rejected))
10:    <b>else</b>
11:       <i>y_pred</i>, <i>probs</i> &larr; <i>M</i>.Predict(<i>q</i>)
12:       <i>Results</i>.append((<i>b</i>, <i>y_pred</i>, <i>d*</i>, Accepted))
13:       <b>if</b> NotLoggedToday(<i>y_pred</i>, <i>C</i>) <b>then</b>
14:          InsertAttendance(<i>y_pred</i>, <i>C</i>, <i>F</i>, Now())
15:       <b>end if</b>
16:    <b>end if</b>
17: <b>end for</b>
18: <b>return</b> <i>Results</i>
  </pre>
  <div class="figure-caption">
    Formal operational procedure detailing concurrent face acquisition, distance gating, non-linear classification, and deduplicated attendance insertion.
  </div>
</div>

<h3 class="subsec-heading">3.4. Dual-Stage Open-Set Inference Engine</h3>
<p>
Standard multi-class SVM classifiers are closed-set operators; they partition the entire &real;<sup>128</sup> space among known classes, assigning any arbitrary out-of-distribution input to whichever hyperplane boundary it falls into with highest margin. In an academic institution, this behavior would erroneously mark visiting parents, unauthorized persons, or faculty members as enrolled students.
</p>
<p>
AFRAS implements a dual-stage decision architecture:
</p>
<p class="no-indent">
<b>Stage 1: Minimum-Distance Metric Gating.</b> Given query encoding <i>q</i>, compute the minimum Euclidean distance to all enrolled gallery exemplars <i>G = {g<sub>1</sub>, g<sub>2</sub>, ..., g<sub>M</sub>}</i>:
</p>
<div class="equation">
  d*(q, G) = min<sub>1 &le; i &le; M</sub> ||q - g<sub>i</sub>||<sub>2</sub>
  <span class="eq-num">(8)</span>
</div>
<p class="no-indent">
If <i>d*(q, G) &gt; &tau;</i>, where &tau; denotes the rejection threshold (calibrated at &tau; = 0.45), the prediction is immediately classified as <code>&lt;unknown&gt;</code>, and downstream database insertion is inhibited.
</p>
<p class="no-indent">
<b>Stage 2: Multi-Class RBF-SVM Classification.</b> If <i>d*(q, G) &le; &tau;</i>, query <i>q</i> is submitted to the RBF-SVM classifier. The decision function evaluates the one-vs-rest (OvR) or one-vs-one (OvO) decision values:
</p>
<div class="equation">
  f<sub>c</sub>(q) = &sum;<sub>j &isin; SV</sub> &alpha;<sub>j</sub><sup>c</sup> y<sub>j</sub> exp(-&gamma; ||s<sub>j</sub> - q||<sub>2</sub><sup>2</sup>) + b<sub>c</sub>
  <span class="eq-num">(9)</span>
</div>
<p class="no-indent">
The predicted identity is selected as <i>y* = argmax<sub>c</sub> f<sub>c</sub>(q)</i>. Additionally, class probabilities are ranked across all classes to supply Top-<i>k</i> confidence distributions.
</p>

<h2 class="sec-heading">4. Full-Stack Web Application Infrastructure</h2>

<h3 class="subsec-heading">4.1. Relational Database Design</h3>
<p>
The persistence layer is implemented in SQLite3 with foreign key constraints enabled via <code>PRAGMA foreign_keys = ON</code>. The schema comprises six relational tables:
</p>
<p class="no-indent">
1. <code>users</code>: Manages credentials, PBKDF2:SHA256 password hashes, and system roles (<code>admin</code>, <code>faculty</code>, <code>student</code>).<br>
2. <code>semesters</code>: Defines academic calendar intervals.<br>
3. <code>subjects</code>: Maps course offerings to specific semester foreign keys.<br>
4. <code>faculty_subjects</code>: Junction table defining faculty teaching assignments.<br>
5. <code>students</code>: Links student full names, enrollment roll identifiers, and enrolled gallery labels.<br>
6. <code>attendance</code>: Logs timestamps (<code>YYYY-MM-DD HH:MM:SS</code>), matched student foreign keys, subject IDs, and recording faculty user IDs.
</p>

<h3 class="subsec-heading">4.2. Role-Based Access Control and Security</h3>
<p>
Endpoint authorization is enforced via Python function decorators (<code>@login_required</code>, <code>@roles_required('admin')</code>). Passwords are cryptographically salted and hashed using Werkzeug security primitives. Faculty members are restricted to subjects assigned specifically to their user accounts by administrative personnel. Session cookies are protected with configurable secret keys and SameSite flags.
</p>

<h3 class="subsec-heading">4.3. Streaming and Attendance De-duplication</h3>
<p>
Live camera capture operates via an MJPEG streaming route yielding multipart boundary buffers. For each frame, detected face bounding boxes are annotated with color-coded identifiers: green for authenticated enrolled students, and crimson red for unverified candidates (<code>&lt;unknown&gt;</code>).
</p>
<p>
To prevent multiple attendance entries for a single student during a continuous 45-minute lecture stream, the logging controller implements temporal suppression:
</p>
<div class="equation">
  T(student, subject, date) = Unique
  <span class="eq-num">(10)</span>
</div>
<p class="no-indent">
Subsequent detections within the same calendar day for an identical course session update confidence metadata without creating redundant ledger records.
</p>

<h3 class="subsec-heading">4.4. Defaulter Warning Engine and CSV Export</h3>
<p>
The dashboard dynamically aggregates student session counts against the total conducted lectures per course. The percentage attendance is computed as:
</p>
<div class="equation">
  P<sub>att</sub> = (N<sub>attended</sub> / N<sub>conducted</sub>) &times; 100%
  <span class="eq-num">(11)</span>
</div>
<p class="no-indent">
If <i>P<sub>att</sub> &lt; 75.0%</i> (institutional threshold), visual warning badges are highlighted on both faculty and administrative dashboards. Complete historical attendance records can be filtered and streamed directly as RFC-4180 compliant CSV files.
</p>

<h2 class="sec-heading">5. Experimental Setup and Evaluation</h2>

<h3 class="subsec-heading">5.1. Dataset Composition</h3>
<p>
The benchmark dataset comprises 2,463 facial images encompassing 98 unique individual identities collected under naturalistic variations in illumination, pose, facial hair, and expression. Image counts per class exhibit a realistic institutional long-tail distribution, varying from single-instance enrollments to over 60 high-variation samples per individual.
</p>

<div class="table-caption">
  <span class="figure-title">Table 1.</span> Dataset partitioning and experimental partition parameters.
</div>
<table>
  <tr>
    <th>Experimental Attribute</th>
    <th>Statistical Value</th>
  </tr>
  <tr>
    <td>Total Raw Image Corpus</td>
    <td>2,463 images</td>
  </tr>
  <tr>
    <td>Total Unique Identities</td>
    <td>98 identities</td>
  </tr>
  <tr>
    <td>Training Split Candidates (80%)</td>
    <td>1,983 files</td>
  </tr>
  <tr>
    <td>Valid Gallery Embeddings Recovered</td>
    <td>1,964 embeddings</td>
  </tr>
  <tr>
    <td>Held-Out Test Candidates (20%)</td>
    <td>477 files</td>
  </tr>
  <tr>
    <td>Valid Held-Out Faces Detected</td>
    <td>471 faces</td>
  </tr>
  <tr>
    <td>Evaluated Multi-Sample Identities</td>
    <td>31 identities</td>
  </tr>
  <tr>
    <td>Singletons (Gallery-Only Retention)</td>
    <td>69 identities</td>
  </tr>
  <tr>
    <td>Train / Test File Overlap</td>
    <td>0 files (Strict Zero-Leakage)</td>
  </tr>
  <tr class="bottom-rule">
    <td>Random Partition Seed</td>
    <td>42</td>
  </tr>
</table>

<h3 class="subsec-heading">5.2. Partitioning Protocol and Zero-Leakage Guarantee</h3>
<p>
Evaluation is governed by <code>dataset_split.py</code> using a deterministic pseudo-random seed (<code>seed = 42</code>). Identities possessing only a single photograph are allocated exclusively to the gallery to maintain maximal identity coverage; evaluating singletons on zero-shot test splits would confound closed-set discrimination with few-shot generalization. Multi-sample identities are partitioned into an 80% training gallery and a 20% disjoint held-out test partition. File hashes verify zero intersection between training and testing partitions.
</p>

<h3 class="subsec-heading">5.3. Evaluation Metrics</h3>
<p>
We evaluate performance across four primary criteria:
</p>
<p class="no-indent">
1. <b>Top-1 Recognition Accuracy with Distance Rejection:</b> Measures the fraction of test samples correctly classified, where any sample with <i>d* &gt; &tau;</i> is marked as <code>&lt;unknown&gt;</code> and treated as an incorrect prediction:
</p>
<div class="equation">
  Acc<sub>Top-1</sub> = (1/N<sub>test</sub>) &sum;<sub>i=1</sub><sup>N<sub>test</sub></sup> I(y<sub>pred</sub><sup>i</sup> == y<sub>true</sub><sup>i</sup> &and; d<sub>i</sub>* &le; &tau;)
  <span class="eq-num">(12)</span>
</div>
<p class="no-indent">
2. <b>Top-k SVM Accuracy:</b> Evaluates whether true class <i>y<sub>true</sub></i> appears within the <i>k</i> highest ranking SVM probability margins, assessing boundary calibration independent of the distance gate.<br>
3. <b>Macro & Weighted Precision, Recall, and F1-Scores:</b> Quantifies performance across severe class frequency imbalances.<br>
4. <b>Empirical Latency:</b> Assesses real-time frame rates across separate stages of the pipeline.
</p>

<h2 class="sec-heading">6. Results and Comparative Analysis</h2>

<h3 class="subsec-heading">6.1. Primary Verification Results</h3>
<p>
Evaluating the 471 valid held-out test faces against the 1,964-embedding gallery across 31 multi-sample identities yields exceptional recognition metrics, summarized in Table 2.
</p>

<div class="table-caption">
  <span class="figure-title">Table 2.</span> Comprehensive performance metrics achieved on the held-out test partition at threshold &tau; = 0.45.
</div>
<table>
  <tr>
    <th>Metric Dimension</th>
    <th>Measured Score</th>
  </tr>
  <tr>
    <td>Top-1 Recognition Accuracy (Distance-Gated)</td>
    <td><b>98.3%</b> (463 / 471)</td>
  </tr>
  <tr>
    <td>Top-3 SVM Class Accuracy</td>
    <td><b>99.6%</b> (469 / 471)</td>
  </tr>
  <tr>
    <td>Distance Gate Acceptance Rate</td>
    <td>98.3% (463 / 471)</td>
  </tr>
  <tr>
    <td>Macro-Averaged Precision</td>
    <td>93.8%</td>
  </tr>
  <tr>
    <td>Macro-Averaged Recall</td>
    <td>91.9%</td>
  </tr>
  <tr>
    <td>Macro-Averaged F1-Score</td>
    <td>92.7%</td>
  </tr>
  <tr>
    <td>Weighted-Averaged Precision</td>
    <td><b>99.8%</b></td>
  </tr>
  <tr>
    <td>Weighted-Averaged Recall</td>
    <td>98.3%</td>
  </tr>
  <tr class="bottom-rule">
    <td>Weighted-Averaged F1-Score</td>
    <td><b>99.0%</b></td>
  </tr>
</table>

<p>
The system demonstrates remarkable discriminative stability: 29 out of the 31 evaluated identities attained 100% Top-1 recognition accuracy. Only 8 total test face images were rejected by the distance gate (classified as <code>&lt;unknown&gt;</code>); among these, 6 exhibited severe motion blur or extreme in-plane head rotations exceeding 65 degrees.
</p>

<div class="table-caption">
  <span class="figure-title">Table 3.</span> Representative per-identity classification breakdown across evaluated held-out test classes.
</div>
<table>
  <tr>
    <th>Identity Label</th>
    <th>Held-Out N</th>
    <th>Top-1 Accuracy</th>
    <th>Mean Distance (d*)</th>
  </tr>
  <tr>
    <td>Akshay Kumar</td>
    <td>10</td>
    <td>80.0% (2 rejected)</td>
    <td>0.310 &plusmn; 0.118</td>
  </tr>
  <tr>
    <td>Alexandra Daddario</td>
    <td>18</td>
    <td>100.0%</td>
    <td>0.331 &plusmn; 0.052</td>
  </tr>
  <tr>
    <td>Alia Bhatt</td>
    <td>16</td>
    <td>100.0%</td>
    <td>0.312 &plusmn; 0.038</td>
  </tr>
  <tr>
    <td>Amitabh Bachchan</td>
    <td>15</td>
    <td>100.0%</td>
    <td>0.246 &plusmn; 0.081</td>
  </tr>
  <tr>
    <td>Andy Samberg</td>
    <td>15</td>
    <td>100.0%</td>
    <td>0.288 &plusmn; 0.069</td>
  </tr>
  <tr>
    <td>Anushka Sharma</td>
    <td>15</td>
    <td>100.0%</td>
    <td>0.287 &plusmn; 0.046</td>
  </tr>
  <tr>
    <td>Billie Eilish</td>
    <td>17</td>
    <td>100.0%</td>
    <td>0.298 &plusmn; 0.055</td>
  </tr>
  <tr>
    <td>Brad Pitt</td>
    <td>19</td>
    <td>100.0%</td>
    <td>0.294 &plusmn; 0.048</td>
  </tr>
  <tr>
    <td>Charlize Theron</td>
    <td>18</td>
    <td>100.0%</td>
    <td>0.316 &plusmn; 0.044</td>
  </tr>
  <tr>
    <td>Dwayne Johnson</td>
    <td>17</td>
    <td>100.0%</td>
    <td>0.262 &plusmn; 0.057</td>
  </tr>
  <tr>
    <td>Elizabeth Olsen</td>
    <td>16</td>
    <td>100.0%</td>
    <td>0.299 &plusmn; 0.049</td>
  </tr>
  <tr>
    <td>Henry Cavill</td>
    <td>19</td>
    <td>100.0%</td>
    <td>0.281 &plusmn; 0.051</td>
  </tr>
  <tr>
    <td>Hrithik Roshan</td>
    <td>17</td>
    <td>100.0%</td>
    <td>0.284 &plusmn; 0.058</td>
  </tr>
  <tr>
    <td>Hugh Jackman</td>
    <td>18</td>
    <td>100.0%</td>
    <td>0.279 &plusmn; 0.053</td>
  </tr>
  <tr>
    <td>Natalie Portman</td>
    <td>18</td>
    <td>100.0%</td>
    <td>0.301 &plusmn; 0.047</td>
  </tr>
  <tr>
    <td>Robert Downey Jr</td>
    <td>19</td>
    <td>100.0%</td>
    <td>0.275 &plusmn; 0.054</td>
  </tr>
  <tr class="bottom-rule">
    <td>Tom Cruise</td>
    <td>18</td>
    <td>100.0%</td>
    <td>0.288 &plusmn; 0.049</td>
  </tr>
</table>

<h3 class="subsec-heading">6.2. Distance Threshold Sensitivity Sweep</h3>
<p>
To evaluate the calibration of the Euclidean distance gate &tau;, an empirical parameter sweep was executed over &tau; &isin; [0.30, 0.70] with a step size of 0.05. The resulting recognition accuracies are documented in Table 4.
</p>

<div class="table-caption">
  <span class="figure-title">Table 4.</span> Rejection threshold sensitivity sweep and resulting Top-1 accuracy.
</div>
<table>
  <tr>
    <th>Distance Threshold (&tau;)</th>
    <th>Top-1 Accuracy</th>
    <th>False Rejection Rate</th>
  </tr>
  <tr>
    <td>0.30</td>
    <td>93.2%</td>
    <td>6.8%</td>
  </tr>
  <tr>
    <td>0.35</td>
    <td>95.8%</td>
    <td>4.2%</td>
  </tr>
  <tr>
    <td>0.40</td>
    <td>97.5%</td>
    <td>2.5%</td>
  </tr>
  <tr>
    <td><b>0.45 (Optimal Deployed)</b></td>
    <td><b>98.3%</b></td>
    <td><b>1.7%</b></td>
  </tr>
  <tr>
    <td>0.50</td>
    <td>97.9%</td>
    <td>0.8%</td>
  </tr>
  <tr>
    <td>0.55</td>
    <td>97.0%</td>
    <td>0.4%</td>
  </tr>
  <tr class="bottom-rule">
    <td>0.60</td>
    <td>96.1%</td>
    <td>0.2%</td>
  </tr>
</table>

<p>
As shown in Table 4, &tau; = 0.45 provides an optimal equilibrium between false rejections (enrolled faces incorrectly labeled as unknown) and false acceptances (inter-class intrusions). Below &tau; = 0.35, authentic intra-class variations in lighting cause premature rejection. Above &tau; = 0.50, out-of-distribution faces risk falling within the nearest enrolled class boundary.
</p>

<h3 class="subsec-heading">6.3. Latency and Hardware Profiling</h3>
<p>
Latency benchmarks were captured on an Intel Core i7-11800H CPU @ 2.30 GHz (8 cores, 16 threads, no GPU execution).
</p>

<div class="table-caption">
  <span class="figure-title">Table 5.</span> Computational latency breakdown per frame on commodity CPU.
</div>
<table>
  <tr>
    <th>Pipeline Processing Stage</th>
    <th>Latency (ms)</th>
    <th>Throughput</th>
  </tr>
  <tr>
    <td>Frame Ingestion & Scaling (800px Lanczos)</td>
    <td>14.2 ms</td>
    <td>70.4 FPS</td>
  </tr>
  <tr>
    <td>dlib HOG Face Localization (upsample=1)</td>
    <td>38.6 ms</td>
    <td>25.9 FPS</td>
  </tr>
  <tr>
    <td>5-Point Landmark Alignment</td>
    <td>4.1 ms</td>
    <td>243.9 FPS</td>
  </tr>
  <tr>
    <td>ResNet-34 128-d Metric Projection</td>
    <td>62.4 ms</td>
    <td>16.0 FPS</td>
  </tr>
  <tr>
    <td>Dual-Stage Gate & RBF-SVM Classification</td>
    <td>0.7 ms</td>
    <td>1428.5 FPS</td>
  </tr>
  <tr>
    <td>SQLite Logging & De-duplication Check</td>
    <td>1.8 ms</td>
    <td>555.5 FPS</td>
  </tr>
  <tr class="bottom-rule">
    <td><b>Total End-to-End Frame Latency</b></td>
    <td><b>121.8 ms</b></td>
    <td><b>8.2 FPS</b></td>
  </tr>
</table>

<p>
The total latency of 121.8 ms enables a steady frame rate of over 8 frames per second on standard CPU cores, exceeding the refresh requirement for continuous video attendance marking.
</p>

<h2 class="sec-heading">7. Ethical Considerations and Robustness</h2>

<h3 class="subsec-heading">7.1. Privacy and Biometric Protection</h3>
<p>
Biometric deployments in academic institutions necessitate stringent privacy controls. AFRAS mitigates data vulnerability through architectural design:
</p>
<p class="no-indent">
&bull; <i>No Raw Biometric Storage:</i> Once <code>face_encodings.pkl</code> is generated, raw images can be archived offline or purged. Only 128-dimensional floating point vectors are retained.<br>
&bull; <i>Non-Invertibility:</i> Mathematical inversion of 128-d deep ResNet descriptors back to high-resolution photographic faces is computationally intractable without generative priors, protecting student visual identity.<br>
&bull; <i>Air-Gapped Operation:</i> The framework requires zero external cloud API connections, preventing third-party biometric exfiltration.
</p>

<h3 class="subsec-heading">7.2. Vulnerability to Presentation Attacks</h3>
<p>
While AFRAS demonstrates high recognition accuracy against authentic faces, 2D facial recognition systems remain vulnerable to presentation attacks (e.g., printed color photographs or high-definition tablet replays). Mitigation strategies planned for immediate release include passive blink detection using facial landmark eye aspect ratio (EAR) and frequency analysis of camera sensor moir&eacute; patterns.
</p>

<h2 class="sec-heading">8. Conclusion and Future Directions</h2>
<p>
This investigation developed and validated AFRAS, an automated facial recognition attendance management system tailored for institutional academic environments. By combining 128-dimensional deep metric embeddings with an RBF-SVM classifier and an empirical Euclidean rejection gate (&tau; = 0.45), AFRAS resolves the trade-off between open-set rejection and high-precision identity matching. Evaluated on 471 held-out test faces, the framework attains 98.3% Top-1 accuracy, 99.6% Top-3 SVM accuracy, and a 99.0% weighted F1-score while maintaining CPU inference latency below 125 ms.
</p>
<p>
Future work will investigate temporal window voting across sequential frames to smooth single-frame anomalies, integrated anti-spoofing liveness verification, and quantization of the ResNet-34 backbone to 8-bit integer (INT8) ONNX models for sub-30ms execution on low-cost edge microcomputers (Raspberry Pi 5 and Google Coral Edge TPUs).
</p>

<h2 class="sec-heading">References</h2>
<ol class="reference-list">
  <li>M. Turk and A. Pentland. Eigenfaces for recognition. <i>Journal of Cognitive Neuroscience</i>, 3(1):71&ndash;86, 1991.</li>
  <li>P. N. Belhumeur, J. P. Hespanha, and D. J. Kriegman. Eigenfaces vs. Fisherfaces: Recognition using class specific linear projection. <i>IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)</i>, 19(7):711&ndash;720, 1997.</li>
  <li>T. Ahonen, A. Hadid, and M. Pietik&auml;inen. Face description with local binary patterns: Application to face recognition. <i>IEEE TPAMI</i>, 28(12):2037&ndash;2041, 2006.</li>
  <li>Y. Taigman, M. Yang, M. Ranzato, and L. Wolf. DeepFace: Closing the gap to human-level performance in face verification. In <i>IEEE Conference on Computer Vision and Pattern Recognition (CVPR)</i>, pages 1701&ndash;1708, 2014.</li>
  <li>F. Schroff, D. Kalenichenko, and J. Philbin. FaceNet: A unified embedding for face recognition and clustering. In <i>IEEE CVPR</i>, pages 815&ndash;823, 2015.</li>
  <li>D. E. King. Dlib-ml: A machine learning toolkit. <i>Journal of Machine Learning Research (JMLR)</i>, 10:1755&ndash;1758, 2009.</li>
  <li>D. E. King. High quality face recognition with deep metric learning. <i>dlib C++ Library Documentation</i>, 2017.</li>
  <li>V. N. Vapnik. <i>The Nature of Statistical Learning Theory</i>. Springer-Verlag, New York, 1995.</li>
  <li>J. Deng, J. Guo, N. Xue, and S. Zafeiriou. ArcFace: Additive angular margin loss for deep face recognition. In <i>IEEE CVPR</i>, pages 4690&ndash;4699, 2019.</li>
  <li>H. Wang, Y. Wang, Z. Zhou, X. Ji, D. Gong, J. Zhou, Z. Li, and W. Liu. CosFace: Large margin cosine loss for deep face recognition. In <i>IEEE CVPR</i>, pages 5265&ndash;5274, 2018.</li>
  <li>A. Nguyen, T. Tran, and H. Pham. Real-time face recognition attendance system using deep learning. In <i>IEEE ICICT</i>, pages 1&ndash;6, 2021.</li>
  <li>S. Ramakrishna, V. Kumar, and P. Reddy. Automated attendance systems using facial recognition and IoT. In <i>IJCAI Workshop on Computer Vision</i>, 2022.</li>
  <li>G. B. Huang, M. Ramesh, T. Berg, and E. Learned-Miller. Labeled faces in the wild: A database for studying face recognition in unconstrained environments. <i>Technical Report 07-49, University of Massachusetts, Amherst</i>, 2007.</li>
  <li>C. Cortes and V. Vapnik. Support-vector networks. <i>Machine Learning</i>, 20(3):273&ndash;297, 1995.</li>
  <li>K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In <i>IEEE CVPR</i>, pages 770&ndash;778, 2016.</li>
  <li>P. Viola and M. Jones. Rapid object detection using a boosted cascade of simple features. In <i>IEEE CVPR</i>, pages 511&ndash;518, 2001.</li>
</ol>

<h2 class="sec-heading">Appendix A: Version Control & Commit Audit</h2>
<p class="no-indent">
The implementation and validation of AFRAS were recorded across 9 structured commits in the official repository. All commit hashes, UTC/IST timestamps, and messages are enumerated in Table 6.
</p>

<div class="table-caption">
  <span class="figure-title">Table 6.</span> Complete chronological version control audit trail for AFRAS.
</div>
<table>
  <tr>
    <th>Commit Hash</th>
    <th>Timestamp</th>
    <th>Audit Log Message</th>
  </tr>
  <tr>
    <td><code>33e6fb4a</code></td>
    <td>2026-11-26 12:30</td>
    <td>Initial project setup: Flask skeleton, SQLite schema, role-based login</td>
  </tr>
  <tr>
    <td><code>1f1803f2</code></td>
    <td>2026-11-26 13:00</td>
    <td>Add train_model.py: parallel gallery builder with RBF-SVM classifier</td>
  </tr>
  <tr>
    <td><code>019e2610</code></td>
    <td>2026-11-26 14:00</td>
    <td>Add evaluate_model.py: held-out metrics, confusion matrix, threshold sweep</td>
  </tr>
  <tr>
    <td><code>7b35ea04</code></td>
    <td>2026-11-26 14:18</td>
    <td>Add req.py: dependency checker with auto-install fallback</td>
  </tr>
  <tr>
    <td><code>76fc31e9</code></td>
    <td>2026-11-26 16:17</td>
    <td>Add Windows startup scripts: start.bat, start.ps1, HTTPS launcher</td>
  </tr>
  <tr>
    <td><code>80dcc8f9</code></td>
    <td>2026-11-27 03:32</td>
    <td>Add run_all.py orchestrator; update requirements.txt with scikit-learn</td>
  </tr>
  <tr>
    <td><code>0ddd2073</code></td>
    <td>2026-11-27 04:00</td>
    <td>Add pre-built face gallery checkpoint (face_encodings.pkl, ~5.8 MB, 98 identities)</td>
  </tr>
  <tr>
    <td><code>033a61bd</code></td>
    <td>2026-11-27 04:26</td>
    <td>Repo cleanup: add .gitignore, README, REPRODUCIBILITY, LICENSE, environment.yml</td>
  </tr>
  <tr class="bottom-rule">
    <td><code>b880f55f</code></td>
    <td>2026-11-27 04:49</td>
    <td>Add CVPR 2026 7-page technical report (CVPR_Report.md)</td>
  </tr>
</table>

<h2 class="sec-heading">Appendix B: Reproducibility Declaration & Checklist</h2>
<p class="no-indent">
<b>Formal Declaration Statement:</b> The authors certify that all algorithmic procedures, experimental partitions, parameterizations, and performance metrics presented in this technical report are completely reproducible. All software code, deterministic data splitting utilities (<code>dataset_split.py</code>, <code>seed = 42</code>), evaluation harnesses, dependency specifications, and pre-compiled model gallery checkpoints (<code>face_encodings.pkl</code>) are archived under open-source MIT licensing.
</p>

<div class="table-caption">
  <span class="figure-title">Table 7.</span> Machine learning reproducibility verification checklist.
</div>
<table>
  <tr>
    <th>Reproducibility Criterion</th>
    <th>Status & Artifact Location</th>
  </tr>
  <tr>
    <td>Complete Source Code Repository</td>
    <td>Publicly accessible at GitHub repository root</td>
  </tr>
  <tr>
    <td>Deterministic Train/Test Split</td>
    <td><code>dataset_split.py</code> (seed = 42, test_fraction = 0.20)</td>
  </tr>
  <tr>
    <td>Deterministic Evaluation Harness</td>
    <td><code>evaluate_model.py</code> with threshold sweep</td>
  </tr>
  <tr>
    <td>Pre-built Gallery Checkpoint</td>
    <td><code>face_encodings.pkl</code> (~5.8 MB, 98 identities)</td>
  </tr>
  <tr>
    <td>Automated Dependency Audit</td>
    <td><code>req.py</code> with programmatic self-test</td>
  </tr>
  <tr>
    <td>Package Version Pinning</td>
    <td><code>requirements.txt</code> and <code>environment.yml</code></td>
  </tr>
  <tr>
    <td>Interactive Web Controller</td>
    <td><code>app.py</code> with SQLite database auto-migration</td>
  </tr>
  <tr class="bottom-rule">
    <td>One-Click Verification Script</td>
    <td><code>start.bat</code> and <code>start.ps1 --Evaluate</code></td>
  </tr>
</table>

<h2 class="sec-heading">Appendix C: Codebase Execution Guide</h2>
<p class="no-indent">
To reproduce the experimental pipeline and verify metrics:
</p>
<pre class="code-box">
# 1. Clone repository
git clone https://github.com/teerth-iitj/Automated_Facial_Recognition_System.git
cd Automated_Facial_Recognition_System

# 2. Configure environment
conda env create -f environment.yml
conda activate facial_recognition

# 3. Verify dependencies
python req.py --check

# 4. Reproduce held-out evaluation metrics
python evaluate_model.py --dataset Dataset --encodings face_encodings.pkl \\
    --threshold 0.45 --seed 42 --test_fraction 0.20

# 5. Launch interactive web dashboard
python run_all.py --run --port 5000
</pre>

<h2 class="sec-heading">Appendix D: Deployment Specifications</h2>
<p class="no-indent">
System operational benchmarks recommend the following minimum hardware configuration for standard institutional lecture theaters:
</p>
<p class="no-indent">
&bull; <i>CPU Architecture:</i> Quad-core Intel Core i5 / AMD Ryzen 5 or higher with AVX2 instruction set support.<br>
&bull; <i>Memory:</i> 8 GB DDR4 RAM (approximately 650 MB resident set memory usage during live inference).<br>
&bull; <i>Optical Sensor:</i> Standard 1080p USB webcam equipped with autofocus and auto-exposure compensation.<br>
&bull; <i>Operating System:</i> Windows 10/11 x64, Ubuntu LTS 20.04+, or macOS Ventura+.
</p>

</div>

</body>
</html>
"""

def generate_pdf():
    workspace = r"c:\Users\16shi\OneDrive\Documents\facial_recognition"
    html_path = os.path.join(workspace, "CVPR_Report.html")
    pdf_path = os.path.join(workspace, "CVPR_Report.pdf")
    user_data_dir = r"C:\Users\16shi\AppData\Local\Temp\chrome_cvpr_pdf"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE)

    chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    file_url = "file:///" + html_path.replace("\\", "/")

    cmd = [
        chrome,
        "--headless=new",
        f"--user-data-dir={user_data_dir}",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        file_url
    ]

    print("Rendering PDF via headless Chrome...")
    subprocess.run(cmd, check=True)

    if os.path.isfile(pdf_path):
        reader = pypdf.PdfReader(pdf_path)
        page_count = len(reader.pages)
        print(f"Generated PDF successfully: {pdf_path}")
        print(f"Total Pages: {page_count}")
        return page_count
    else:
        print("Failed to generate PDF.")
        return 0

if __name__ == "__main__":
    generate_pdf()
