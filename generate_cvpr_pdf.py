import os
import subprocess
import pypdf

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Automated Face Recognition System</title>
<style>
  @page {
    size: letter;
    margin: 0.78in 0.75in 0.85in 0.75in;
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
    font-size: 10pt;
    line-height: 12.6pt;
    color: #000000;
    margin: 0;
    padding: 0;
    text-align: justify;
    text-justify: inter-word;
  }

  .paper-header {
    text-align: center;
    margin-bottom: 13pt;
  }

  h1.paper-title {
    font-size: 15pt;
    font-weight: bold;
    line-height: 18pt;
    margin: 0 0 10pt 0;
    text-align: center;
  }

  .authors-grid {
    display: flex;
    justify-content: space-around;
    align-items: flex-start;
    margin: 0 auto 6pt auto;
    width: 96%;
    gap: 10pt;
  }

  .author-col {
    flex: 1;
    text-align: center;
    font-size: 9.8pt;
    line-height: 12pt;
  }

  .author-name {
    font-weight: bold;
  }

  .author-roll {
    font-size: 9.2pt;
  }

  .author-email {
    font-family: 'Courier New', Courier, monospace;
    font-size: 8.2pt;
  }

  .institution {
    font-size: 9.8pt;
    color: #111;
    margin-top: 5pt;
    margin-bottom: 11pt;
    text-align: center;
  }

  .abstract-box {
    margin: 0 auto 13pt auto;
    width: 94%;
    font-size: 9.3pt;
    line-height: 11.8pt;
    text-align: justify;
    border-top: 0.7pt solid #000;
    border-bottom: 0.7pt solid #000;
    padding: 8pt 0;
  }

  .abstract-title {
    font-weight: bold;
    font-size: 9.6pt;
    display: inline;
  }

  .two-column {
    column-count: 2;
    column-gap: 0.30in;
    column-fill: balance;
  }

  h2.sec-heading {
    font-size: 10.2pt;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.3pt;
    margin: 11pt 0 4.5pt 0;
    break-after: avoid;
  }

  h3.subsec-heading {
    font-size: 9.8pt;
    font-weight: bold;
    font-style: italic;
    margin: 8pt 0 3pt 0;
    break-after: avoid;
  }

  p {
    margin: 0 0 6pt 0;
    text-indent: 10pt;
  }

  p.no-indent {
    text-indent: 0;
  }

  ul.clean-list, ol.clean-list {
    margin: 3.5pt 0 6.5pt 0;
    padding-left: 14pt;
  }

  ul.clean-list li, ol.clean-list li {
    margin-bottom: 3.5pt;
    text-align: justify;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 5.5pt 0 8.5pt 0;
    font-size: 8.2pt;
    line-height: 10.4pt;
    break-inside: avoid;
  }

  th {
    border-top: 1pt solid #000;
    border-bottom: 0.6pt solid #000;
    padding: 3.2pt 4pt;
    text-align: left;
    font-weight: bold;
  }

  td {
    padding: 2.5pt 4pt;
    text-align: left;
    border-bottom: 0.4pt solid #e0e0e0;
  }

  tr.bottom-rule td {
    border-bottom: 1pt solid #000;
  }

  .table-caption {
    font-size: 8.5pt;
    font-weight: normal;
    margin-top: 6.5pt;
    margin-bottom: 2.5pt;
    text-align: justify;
    break-after: avoid;
  }

  .figure-box {
    margin: 8pt 0;
    padding: 6.5pt;
    background: #ffffff;
    border: 0.6pt solid #bbb;
    border-radius: 2pt;
    font-size: 8.2pt;
    line-height: 10.2pt;
    break-inside: avoid;
  }

  .figure-caption {
    font-size: 8.4pt;
    margin-top: 4.5pt;
    text-align: justify;
  }

  .figure-title {
    font-weight: bold;
  }

  pre.code-box {
    font-family: 'Courier New', Courier, monospace;
    font-size: 7.4pt;
    line-height: 9.5pt;
    background: #f8f8f8;
    padding: 5.5pt;
    margin: 4.5pt 0 7pt 0;
    border: 0.5pt solid #ccc;
    white-space: pre-wrap;
    word-break: break-all;
    break-inside: avoid;
  }

  code {
    font-family: 'Courier New', Courier, monospace;
    font-size: 8.4pt;
  }

  .reference-list {
    font-size: 8.1pt;
    line-height: 10.2pt;
    padding-left: 14pt;
    margin: 4.5pt 0;
  }

  .reference-list li {
    margin-bottom: 3.8pt;
    text-align: justify;
  }
</style>
</head>
<body>

<div class="paper-header">
  <h1 class="paper-title">Automated Face Recognition System</h1>

  <div class="authors-grid">
    <div class="author-col">
      <div class="author-name">Basab Bose</div>
      <div class="author-roll">M26CSE002</div>
      <div class="author-email">m26cse002@iitj.ac.in</div>
    </div>
    <div class="author-col">
      <div class="author-name">Teerth Bhardwaj</div>
      <div class="author-roll">M26CSE010</div>
      <div class="author-email">m26cse010@iitj.ac.in</div>
    </div>
    <div class="author-col">
      <div class="author-name">Ravikaran Singh</div>
      <div class="author-roll">M26CSE012</div>
      <div class="author-email">m26cse012@iitj.ac.in</div>
    </div>
  </div>

  <div class="institution">
    Department of Computer Science and Engineering, Indian Institute of Technology Jodhpur, India
  </div>

  <div class="abstract-box">
    <span class="abstract-title">Abstract&mdash;</span>
    Taking attendance manually in classrooms is slow, interrupts teaching, and leaves room for proxy entries or clerical mistakes. In this project, we built the <b>Automated Face Recognition System</b>, a practical web-based attendance application that identifies students from live webcam captures, uploaded photographs, and recorded classroom videos on standard CPU laptops without needing a GPU. Our recognition pipeline uses a pretrained ResNet-based deep face embedding model (via <code>dlib</code> and <code>face_recognition</code>) to convert each detected face into a compact 128-number representation. On top of these face encodings, we train a Support Vector Machine (SVM) with an RBF kernel to tell enrolled individuals apart, paired with a minimum face-distance check (threshold 0.45) that rejects unfamiliar faces as <code>&lt;unknown&gt;</code>. The backend is built with Flask and SQLite, offering separate dashboards for administrators and faculty, course and semester management, automatic duplicate-attendance prevention, and CSV report downloads. We test our model on a dataset of 2,460 supported photos split deterministically into 1,983 training photos and 477 held-out test photos. Out of 471 valid detected test faces, the system achieves <b>98.3% Top-1 accuracy</b> (463 out of 471 photos correctly recognized and accepted), <b>99.6% Top-3 SVM accuracy</b>, and a <b>0.990 weighted F1-score</b>, with zero wrong identity swaps among all accepted predictions. All source code, environment files, prebuilt checkpoint, and reproducibility instructions are available at <code>https://github.com/teerth-iitj/Automated_Facial_Recognition_System</code>.
  </div>
</div>

<div class="two-column">

<h2 class="sec-heading">1. Introduction</h2>
<p>
Recording student attendance is an everyday requirement across universities and colleges. In most classrooms, attendance is still taken by calling out names or passing around a paper sign-in sheet. Both approaches create practical difficulties: calling out names wastes valuable lecture time in large classes, while paper sheets are easy to forge through proxy signatures and require tedious manual entry into spreadsheets afterward.
</p>
<p>
Hardware biometric options like fingerprint scanners or RFID smart cards solve some of these issues, yet bring new drawbacks. Fingerprint terminals require physical contact and create long queues outside lecture halls, whereas RFID cards can simply be handed to a friend to mark attendance on someone else's behalf.
</p>
<p>
Camera-based facial recognition provides a cleaner, contactless alternative. A faculty member can capture a quick classroom snapshot, stream from a laptop webcam, or upload a short clip, and have attendance logged automatically within seconds. However, building a system that works smoothly in a real academic setting involves several practical design requirements:
</p>
<ul class="clean-list">
  <li><b>Works with uneven photo counts:</b> Some people in a department dataset have over 100 photos, while others only have a single enrollment picture. Training a deep neural network from scratch is impractical under these conditions.</li>
  <li><b>Runs on normal laptops:</b> Classrooms and faculty laptops typically do not have dedicated GPUs, so both gallery creation and live recognition must run comfortably on standard multi-core CPUs.</li>
  <li><b>Safely ignores strangers:</b> If an unregistered visitor walks past the camera or a blurry background face is picked up, the system must refuse to guess rather than falsely marking a student present.</li>
  <li><b>Organized around real classes:</b> Recognition alone is not enough; faculty need a clean interface to pick their semester and subject, review marked entries, avoid double-counting on the same day, and export records as CSV files.</li>
</ul>
<p>
To meet these needs, we built a complete, self-contained application combining pretrained face embeddings, an RBF-SVM classifier with a distance safety gate, and a role-based Flask web portal backed by SQLite.
</p>

<h2 class="sec-heading">2. Background and Related Work</h2>
<p>
Early face recognition methods such as Eigenfaces [1] and Fisherfaces [2] compressed face images into linear subspaces using statistical projections, while Local Binary Patterns Histograms (LBPH) [3] compared local pixel textures. Although fast, these classic methods struggle when lighting changes, heads tilt, or facial expressions vary.
</p>
<p>
Modern approaches rely on deep convolutional neural networks trained to map faces onto compact numerical lists called embeddings. DeepFace [4] and FaceNet [5] showed that training networks so photos of the same person sit close together in embedding space yields human-level verification accuracy. The <code>dlib</code> face recognition model [6, 7] adapts a ResNet-style deep network [14] trained on millions of faces to output a 128-number embedding for any aligned face, achieving 99.38% accuracy on the Labeled Faces in the Wild (LFW) benchmark [12].
</p>
<p>
Instead of retraining this large neural network on small classroom datasets, a widely adopted strategy is transfer learning with fixed embeddings: extract the 128-number vector from a pretrained model, and train a lightweight classical classifier such as a Support Vector Machine (SVM) [8, 13] on top. SVMs with an RBF kernel excel at finding smooth, curved boundaries in 128-dimensional space even when sample counts vary significantly across people. Prior campus attendance prototypes [10, 11] often stop at standalone command-line scripts; our project delivers a full multi-user web platform alongside reproducible training, splitting, and held-out evaluation tools.
</p>

<h2 class="sec-heading">3. How the Recognition Model Works</h2>
<p>
Our recognition pipeline intentionally separates <i>face feature extraction</i> (handled by a frozen pretrained deep network) from <i>identity classification</i> (handled by our trained RBF-SVM and distance safety gate). Figure 1 illustrates every step from raw input photos to database storage.
</p>

<div class="figure-box">
  <svg viewBox="0 0 330 225" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block; margin:0 auto;">
    <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="#222"/>
      </marker>
    </defs>
    <!-- Box 1 -->
    <rect x="35" y="4" width="260" height="26" rx="4" fill="#eef3fc" stroke="#2457c5" stroke-width="1"/>
    <text x="165" y="15" font-family="Times New Roman" font-size="8.5" font-weight="bold" text-anchor="middle" fill="#111">1. Input Image / Webcam / Video Frame</text>
    <text x="165" y="25" font-family="Times New Roman" font-size="7.2" text-anchor="middle" fill="#444">Resize longest side to max 800 px (LANCZOS)</text>
    <line x1="165" y1="30" x2="165" y2="41" stroke="#222" stroke-width="1" marker-end="url(#arrow)"/>

    <!-- Box 2 -->
    <rect x="35" y="42" width="260" height="26" rx="4" fill="#eef3fc" stroke="#2457c5" stroke-width="1"/>
    <text x="165" y="53" font-family="Times New Roman" font-size="8.5" font-weight="bold" text-anchor="middle" fill="#111">2. Face Detection (dlib HOG + upsample)</text>
    <text x="165" y="63" font-family="Times New Roman" font-size="7.2" text-anchor="middle" fill="#444">Locate face boxes; keep largest face in training/eval</text>
    <line x1="165" y1="68" x2="165" y2="79" stroke="#222" stroke-width="1" marker-end="url(#arrow)"/>

    <!-- Box 3 -->
    <rect x="35" y="80" width="260" height="26" rx="4" fill="#eef3fc" stroke="#2457c5" stroke-width="1"/>
    <text x="165" y="91" font-family="Times New Roman" font-size="8.5" font-weight="bold" text-anchor="middle" fill="#111">3. Pretrained ResNet Face Encoding</text>
    <text x="165" y="101" font-family="Times New Roman" font-size="7.2" text-anchor="middle" fill="#444">Convert each face into 128 numerical features</text>
    <line x1="165" y1="106" x2="165" y2="117" stroke="#222" stroke-width="1" marker-end="url(#arrow)"/>

    <!-- Box 4 -->
    <rect x="35" y="118" width="260" height="26" rx="4" fill="#f4fbfa" stroke="#0f766e" stroke-width="1"/>
    <text x="165" y="129" font-family="Times New Roman" font-size="8.5" font-weight="bold" text-anchor="middle" fill="#111">4. RBF-SVM Identity Classifier</text>
    <text x="165" y="139" font-family="Times New Roman" font-size="7.2" text-anchor="middle" fill="#444">Predict closest matching enrolled person name</text>
    <line x1="165" y1="144" x2="165" y2="155" stroke="#222" stroke-width="1" marker-end="url(#arrow)"/>

    <!-- Box 5 -->
    <rect x="35" y="156" width="260" height="26" rx="4" fill="#fff8eb" stroke="#b45309" stroke-width="1"/>
    <text x="165" y="167" font-family="Times New Roman" font-size="8.5" font-weight="bold" text-anchor="middle" fill="#111">5. Distance Safety Gate (threshold &lt; 0.45)</text>
    <text x="165" y="177" font-family="Times New Roman" font-size="7.2" text-anchor="middle" fill="#444">Check distance to predicted person's stored photos</text>

    <!-- Branching arrows -->
    <line x1="105" y1="182" x2="80" y2="195" stroke="#222" stroke-width="1" marker-end="url(#arrow)"/>
    <line x1="225" y1="182" x2="250" y2="195" stroke="#222" stroke-width="1" marker-end="url(#arrow)"/>

    <!-- Accepted -->
    <rect x="10" y="196" width="145" height="24" rx="3" fill="#ecfdf5" stroke="#059669" stroke-width="1"/>
    <text x="82.5" y="206" font-family="Times New Roman" font-size="7.8" font-weight="bold" text-anchor="middle" fill="#065f46">Pass: Mark Attendance</text>
    <text x="82.5" y="216" font-family="Times New Roman" font-size="6.8" text-anchor="middle" fill="#065f46">Log in SQLite (once per day)</text>

    <!-- Rejected -->
    <rect x="175" y="196" width="145" height="24" rx="3" fill="#fef2f2" stroke="#b91c1c" stroke-width="1"/>
    <text x="247.5" y="206" font-family="Times New Roman" font-size="7.8" font-weight="bold" text-anchor="middle" fill="#7f1d1d">Reject: &lt;unknown&gt;</text>
    <text x="247.5" y="216" font-family="Times New Roman" font-size="6.8" text-anchor="middle" fill="#7f1d1d">Skip unrecognized / blurry face</text>
  </svg>
  <div class="figure-caption">
    <span class="figure-title">Figure 1.</span> Recognition pipeline shared across offline evaluation (<code>evaluate_model.py</code>) and live web attendance (<code>app.py</code>).
  </div>
</div>

<h3 class="subsec-heading">3.1. Image Resizing and Face Detection</h3>
<p>
High-resolution photos from modern phone cameras slow down processing unnecessarily, while aggressively shrunk images make distant faces too blurry to detect. In earlier runs of our project, images were capped at 400 pixels, which caused smaller faces to be missed. We raised the maximum dimension limit to <b>800 pixels</b> (using Pillow's LANCZOS resampling) and enabled <code>number_of_times_to_upsample=1</code> in <code>dlib</code>'s Histogram of Oriented Gradients (HOG) face detector. Together, these adjustments allow smaller faces to survive downscaling and get detected reliably.
</p>
<p>
During dataset encoding and evaluation, some photos contain background people in addition to the main subject. Instead of throwing away multi-face photos, our scripts compute the area of each detected face box and keep the largest face. Photos where zero faces are found are safely skipped and logged.
</p>

<h3 class="subsec-heading">3.2. 128-Dimensional Face Embeddings</h3>
<p>
Once a face box is located, <code>face_recognition</code> [9] aligns the eyes, nose, and mouth landmarks and passes the cropped face through <code>dlib</code>'s pretrained ResNet-based embedding network. The network outputs a list of <b>128 numbers</b> (a 128-dimensional feature vector) summarizing the person's facial geometry. Photos of the same person under different lighting or expressions map close together in this 128-number space, while different people map further apart.
</p>
<p>
We keep the weights of this deep neural network frozen rather than retraining them. Keeping the feature extractor fixed avoids overfitting on individuals who have only a few photos and keeps training fast on a normal laptop CPU. An optional <code>--jitters</code> parameter averages multiple slightly shifted crops per photo for extra stability (set to <code>1</code> by default for fast builds).
</p>

<h3 class="subsec-heading">3.3. Parallel Gallery Building &amp; RBF-SVM Training</h3>
<p>
To build the face gallery across nearly two thousand photos without waiting on a single CPU core, <code>train_model.py</code> runs image encoding in parallel using Python's <code>ProcessPoolExecutor</code> (defaulting to up to 4 workers). Each valid photo produces one 128-number encoding paired with the person's normalized name.
</p>
<p>
After all training photos are encoded, we train a Support Vector Machine classifier from <code>scikit-learn</code> using a Radial Basis Function kernel (<code>SVC(kernel="rbf", C=1.0, gamma="scale")</code>). Intuitively, the RBF-SVM learns curved separation regions around each enrolled person in the 128-number feature space (the final trained model retains 1,395 support vectors across the gallery). All encodings, person names, training filenames, and the trained SVM are saved into a single portable checkpoint file, <code>face_encodings.pkl</code> (~5.8 MB), via <code>joblib</code>. A <code>--reuse-encodings</code> flag also allows retraining the SVM in seconds directly from saved encodings without re-reading raw photos.
</p>

<h3 class="subsec-heading">3.4. Distance Safety Gate (&lt;unknown&gt; Rejection)</h3>
<p>
A standard SVM classifier is closed-set: it always picks whichever enrolled person is closest, even when a stranger steps in front of the camera. To block false attendance entries, both <code>app.py</code> and <code>evaluate_model.py</code> apply a two-step verification rule:
</p>
<ol class="clean-list">
  <li>The RBF-SVM predicts the most likely enrolled name for a detected face.</li>
  <li>The system fetches all stored gallery encodings for that predicted name and measures the face distance to each one using <code>face_recognition.face_distance</code>.</li>
  <li>If the smallest distance is strictly below our deployment threshold of <b>0.45</b>, the prediction is accepted and attendance is marked. Otherwise, the face is rejected as <code>&lt;unknown&gt;</code>.</li>
</ol>
<p>
Combining the SVM with this distance check lets the classifier resolve similar-looking enrolled students while keeping unregistered visitors out.
</p>

<h3 class="subsec-heading">3.5. Training vs. Live Classroom Inference</h3>
<p>
While training and evaluation always isolate the single largest face per photo (since each dataset photo is labeled with one person's name), live attendance in <code>app.py</code> works slightly differently. In <code>recognize_from_rgb_image()</code>, the system scans <i>all</i> detected faces inside a classroom photo, webcam snapshot, or video frame. Each face is independently passed through the RBF-SVM and the 0.45 distance gate, allowing a single group photograph to mark multiple students present simultaneously.
</p>

<h2 class="sec-heading">4. Web Application and System Design</h2>
<p>
To make face recognition practical for everyday classroom use, we developed a full-stack Flask web application (<code>app.py</code>) connected to a local SQLite database (<code>attendance.db</code>) with foreign key checks enabled. Starting the server via <code>start.bat</code>, <code>run_all.py</code>, or <code>python app.py</code> automatically creates the tables, seeds default credentials (<code>admin</code> / <code>admin123</code> and <code>faculty</code> / <code>faculty123</code>), imports student user entries from the gallery names, migrates any legacy <code>attendance.csv</code> data if present, and opens Chrome. Figure 2 shows the user workflow across roles.
</p>

<div class="figure-box">
  <svg viewBox="0 0 330 140" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block; margin:0 auto;">
    <defs>
      <marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="#222"/>
      </marker>
    </defs>
    <!-- Admin Box -->
    <rect x="6" y="8" width="145" height="54" rx="4" fill="#f0fdf4" stroke="#15803d" stroke-width="1"/>
    <text x="78.5" y="20" font-family="Times New Roman" font-size="8.2" font-weight="bold" text-anchor="middle" fill="#14532d">Admin Role (/admin/dashboard)</text>
    <text x="78.5" y="31" font-family="Times New Roman" font-size="7" text-anchor="middle" fill="#222">&bull; Manage faculty accounts</text>
    <text x="78.5" y="41" font-family="Times New Roman" font-size="7" text-anchor="middle" fill="#222">&bull; Create semesters &amp; subjects</text>
    <text x="78.5" y="51" font-family="Times New Roman" font-size="7" text-anchor="middle" fill="#222">&bull; Assign subjects to faculty</text>

    <!-- Faculty Box -->
    <rect x="179" y="8" width="145" height="54" rx="4" fill="#eef3fc" stroke="#2457c5" stroke-width="1"/>
    <text x="251.5" y="20" font-family="Times New Roman" font-size="8.2" font-weight="bold" text-anchor="middle" fill="#1e3a8a">Faculty Role (/faculty/dashboard)</text>
    <text x="251.5" y="31" font-family="Times New Roman" font-size="7" text-anchor="middle" fill="#222">&bull; Pick assigned semester &amp; subject</text>
    <text x="251.5" y="41" font-family="Times New Roman" font-size="7" text-anchor="middle" fill="#222">&bull; Live webcam / photo / video capture</text>
    <text x="251.5" y="51" font-family="Times New Roman" font-size="7" text-anchor="middle" fill="#222">&bull; View summary &amp; export CSV</text>

    <!-- Arrows to SQLite -->
    <line x1="78.5" y1="62" x2="125" y2="88" stroke="#222" stroke-width="1" marker-end="url(#arr2)"/>
    <line x1="251.5" y1="62" x2="205" y2="88" stroke="#222" stroke-width="1" marker-end="url(#arr2)"/>

    <!-- SQLite DB Box -->
    <rect x="35" y="90" width="260" height="42" rx="4" fill="#f8fafc" stroke="#475569" stroke-width="1"/>
    <text x="165" y="102" font-family="Times New Roman" font-size="8.2" font-weight="bold" text-anchor="middle" fill="#0f172a">SQLite3 Database (attendance.db)</text>
    <text x="165" y="114" font-family="Times New Roman" font-size="7" text-anchor="middle" fill="#334155">Tables: users | semesters | subjects | faculty_subjects | attendance</text>
    <text x="165" y="124" font-family="Times New Roman" font-size="6.8" text-anchor="middle" fill="#475569">Enforces once-per-day deduplication per student, subject, and faculty</text>
  </svg>
  <div class="figure-caption">
    <span class="figure-title">Figure 2.</span> Role-based web portal architecture connecting administrative setup, faculty attendance capture, and SQLite storage.
  </div>
</div>

<div class="table-caption">
  <span class="figure-title">Table 1.</span> Relational SQLite database tables implemented in <code>app.py</code>.
</div>
<table>
  <tr>
    <th>Table Name</th>
    <th>Columns &amp; Role in Application</th>
  </tr>
  <tr>
    <td><code>users</code></td>
    <td>Stores <code>username</code>, salted password hashes, <code>role</code> (admin, faculty, student), <code>full_name</code>, and creation timestamp.</td>
  </tr>
  <tr>
    <td><code>semesters</code></td>
    <td>Stores unique semester names (e.g., Semester 6) and timestamps.</td>
  </tr>
  <tr>
    <td><code>subjects</code></td>
    <td>Links course subjects to a parent <code>semester_id</code> with cascading deletes.</td>
  </tr>
  <tr>
    <td><code>faculty_subjects</code></td>
    <td>Maps each faculty account to their assigned subjects.</td>
  </tr>
  <tr class="bottom-rule">
    <td><code>attendance</code></td>
    <td>Logs student <code>name</code>, <code>timestamp</code>, input <code>source</code> (camera/image/video), faculty ID, semester, and subject.</td>
  </tr>
</table>

<h3 class="subsec-heading">4.1. Input Modes and Route Architecture</h3>
<p>
User accounts use salted password hashing via <code>werkzeug.security</code>, and endpoints are protected with <code>@login_required</code> and <code>@roles_required</code> decorators. Interactive login is enabled for admin and faculty accounts. On the faculty dashboard, three capture modes are supported:
</p>
<ul class="clean-list">
  <li><b>Live Webcam Capture:</b> In <code>faculty_dashboard.html</code>, the browser requests camera access via WebRTC (<code>navigator.mediaDevices.getUserMedia</code>) with a device selector dropdown to switch cameras. Clicking capture draws the current video frame onto an HTML5 canvas, encodes it as JPEG, and posts it to <code>/recognize</code>.</li>
  <li><b>Photo Upload:</b> Uploading a static photo sends it to <code>/recognize</code>, which decodes it with OpenCV and marks every recognized student present.</li>
  <li><b>Video Upload:</b> Uploading a recorded clip to <code>/recognize-video</code> instructs OpenCV to sample every 15th frame (up to 60 sampled frames max) and merge all recognized names across the clip.</li>
</ul>

<div class="table-caption">
  <span class="figure-title">Table 2.</span> Core Flask endpoints and automation scripts in the repository.
</div>
<table>
  <tr>
    <th>Route / Script</th>
    <th>Purpose</th>
  </tr>
  <tr>
    <td><code>/login</code>, <code>/logout</code>, <code>/health</code></td>
    <td>Session authentication and model health status JSON</td>
  </tr>
  <tr>
    <td><code>/api/faculty-users</code>, <code>/api/semesters</code>, <code>/api/subjects</code></td>
    <td>Admin CRUD APIs for faculty accounts, semesters, and courses</td>
  </tr>
  <tr>
    <td><code>/api/faculty-assignments</code>, <code>/api/faculty/context</code></td>
    <td>Assign courses to faculty and set active classroom context</td>
  </tr>
  <tr>
    <td><code>/recognize</code>, <code>/recognize-video</code></td>
    <td>Run face recognition on uploaded photos, webcam frames, or videos</td>
  </tr>
  <tr>
    <td><code>/attendance</code>, <code>/attendance/summary</code>, <code>/attendance/download</code></td>
    <td>List records, compute today's stats, delete entries, and download CSV</td>
  </tr>
  <tr>
    <td><code>run_all.py</code> &amp; <code>req.py</code></td>
    <td>Unified CLI (<code>--train</code>, <code>--evaluate</code>, <code>--run</code>, <code>--all</code>) and package checker</td>
  </tr>
  <tr class="bottom-rule">
    <td><code>start.bat</code> &amp; <code>start.ps1</code></td>
    <td>Windows one-click bootstrap (installs <code>.venv</code>, <code>dlib-bin</code>, models, runs app)</td>
  </tr>
</table>

<h3 class="subsec-heading">4.2. Duplicate Suppression &amp; CSV Export</h3>
<p>
Before writing a new attendance row, <code>already_marked_today()</code> checks whether that student already has a record for the same date, semester, subject, and faculty account. If already logged, the system informs the faculty member without inserting duplicate rows. Faculty can delete individual rows, clear their current class records, or export clean CSV attendance sheets via <code>/attendance/download</code>.
</p>

<h3 class="subsec-heading">4.3. Windows Bootstrap &amp; HTTPS Support</h3>
<p>
Installing <code>dlib</code> from C++ source on Windows often fails without Visual Studio build tools. To eliminate setup friction, <code>start.ps1</code> creates a local <code>.venv</code>, pins <code>setuptools&lt;81</code> (required by <code>face_recognition_models</code> for <code>pkg_resources</code>), installs precompiled <code>dlib-bin</code> wheels along with dependencies from <code>requirements.txt</code>, and fetches the upstream model weights automatically. In addition, because modern browsers block webcam access over plain HTTP on non-localhost addresses, <code>app.py</code> supports an adhoc SSL mode via <code>FLASK_HTTPS=1</code> (also wrapped in <code>run_https_5443.ps1</code> and <code>run_all.py --https</code>).
</p>

<h2 class="sec-heading">5. Dataset and Evaluation Protocol</h2>
<p>
To evaluate recognition accurately without any data leakage between training and testing images, we built a deterministic splitting script in <code>dataset_split.py</code>. It automatically parses person names from both flat filenames (such as <code>Person Name_0.jpg</code> or <code>Person Name (001).jpg</code>) and subfolder names.
</p>
<p>
Our full dataset folder contains 2,463 files, of which <b>2,460</b> are supported image files (<code>.jpg</code>, <code>.jpeg</code>, <code>.png</code>; 3 files with <code>.heif</code>/<code>.webp</code> extensions are ignored by the splitter). Multi-photo entries account for 2,391 photos&mdash;including local student photo sets such as Teerth (47 photos), Basab (36 photos), Marmik (32 photos), and Kashyap (30 photos), alongside public reference celebrity photos such as Brad Pitt (120), Robert Downey Jr (113), and Hugh Jackman (112)&mdash;while 69 entries are single-photo enrollments. Figure 3 visualizes the photo counts across representative dataset groups.
</p>

<div class="figure-box">
  <svg viewBox="0 0 325 135" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block; margin:0 auto;">
    <!-- Horizontal bar chart of representative dataset photo counts -->
    <text x="162" y="11" font-family="Times New Roman" font-size="8" font-weight="bold" text-anchor="middle" fill="#111">Photo Counts for Sample Subjects in Dataset/ (Total: 2,460 Photos)</text>
    <!-- Scale: max 120 -> width 180px (1.5px per photo) starting at x=95 -->
    <text x="90" y="28" font-family="Times New Roman" font-size="7.2" text-anchor="end" fill="#222">Brad Pitt</text>
    <rect x="95" y="20" width="180" height="9" fill="#2457c5"/>
    <text x="279" y="28" font-family="Times New Roman" font-size="7.2" fill="#111">120</text>

    <text x="90" y="42" font-family="Times New Roman" font-size="7.2" text-anchor="end" fill="#222">Robert Downey Jr</text>
    <rect x="95" y="34" width="169.5" height="9" fill="#2457c5"/>
    <text x="268.5" y="42" font-family="Times New Roman" font-size="7.2" fill="#111">113</text>

    <text x="90" y="56" font-family="Times New Roman" font-size="7.2" text-anchor="end" fill="#222">Hugh Jackman</text>
    <rect x="95" y="48" width="168" height="9" fill="#2457c5"/>
    <text x="267" y="56" font-family="Times New Roman" font-size="7.2" fill="#111">112</text>

    <text x="90" y="70" font-family="Times New Roman" font-size="7.2" text-anchor="end" fill="#222">Natalie Portman</text>
    <rect x="95" y="62" width="157.5" height="9" fill="#2457c5"/>
    <text x="256.5" y="70" font-family="Times New Roman" font-size="7.2" fill="#111">105</text>

    <text x="90" y="84" font-family="Times New Roman" font-size="7.2" text-anchor="end" fill="#222">Teerth (Student)</text>
    <rect x="95" y="76" width="70.5" height="9" fill="#0f766e"/>
    <text x="169.5" y="84" font-family="Times New Roman" font-size="7.2" fill="#111">47</text>

    <text x="90" y="98" font-family="Times New Roman" font-size="7.2" text-anchor="end" fill="#222">Basab (Student)</text>
    <rect x="95" y="90" width="54" height="9" fill="#0f766e"/>
    <text x="153" y="98" font-family="Times New Roman" font-size="7.2" fill="#111">36</text>

    <text x="90" y="112" font-family="Times New Roman" font-size="7.2" text-anchor="end" fill="#222">Marmik (Student)</text>
    <rect x="95" y="104" width="48" height="9" fill="#0f766e"/>
    <text x="147" y="112" font-family="Times New Roman" font-size="7.2" fill="#111">32</text>

    <text x="90" y="126" font-family="Times New Roman" font-size="7.2" text-anchor="end" fill="#222">Kashyap (Student)</text>
    <rect x="95" y="118" width="45" height="9" fill="#0f766e"/>
    <text x="144" y="126" font-family="Times New Roman" font-size="7.2" fill="#111">30</text>
  </svg>
  <div class="figure-caption">
    <span class="figure-title">Figure 3.</span> Photo distribution across sample public reference subjects (blue) and local student enrollments (teal) within the 2,460-photo dataset.
  </div>
</div>

<div class="table-caption">
  <span class="figure-title">Table 3.</span> Dataset photo breakdown and deterministic 80/20 split statistics from <code>dataset_split.py</code> and <code>evaluation_split.log</code>.
</div>
<table>
  <tr>
    <th>Dataset &amp; Split Metric</th>
    <th>Observed Count</th>
  </tr>
  <tr>
    <td>Total files in <code>Dataset/</code></td>
    <td>2,463 files (2,460 supported images; 3 HEIF/WebP ignored)</td>
  </tr>
  <tr>
    <td>Training split candidates (80% split + single-photo entries)</td>
    <td>1,983 photos</td>
  </tr>
  <tr>
    <td>Valid gallery face encodings saved in <code>face_encodings.pkl</code></td>
    <td>1,964 encodings (19 skipped &mdash; no face detected)</td>
  </tr>
  <tr>
    <td>Distinct enrolled names in saved gallery</td>
    <td>98 people</td>
  </tr>
  <tr>
    <td>Held-out test candidates (20% split, <code>seed=42</code>)</td>
    <td>477 photos</td>
  </tr>
  <tr>
    <td>Valid held-out test faces evaluated</td>
    <td>471 photos (6 skipped &mdash; no face detected; 0 errors)</td>
  </tr>
  <tr class="bottom-rule">
    <td>Train / test file overlap</td>
    <td>0 photos (enforced programmatically in <code>evaluate_model.py</code>)</td>
  </tr>
</table>

<p>
Using a fixed random seed (<code>seed=42</code>) and a <code>20%</code> held-out split, people with multiple photos contribute 80% of their images to the training gallery and 20% to the test set. Single-photo entries are placed exclusively in the training gallery so they remain searchable in production without creating empty training classes. Before scoring, <code>evaluate_model.py</code> verifies that the <code>train_files</code> metadata inside <code>face_encodings.pkl</code> matches the deterministic split with zero filename overlap.
</p>

<h2 class="sec-heading">6. Experimental Results</h2>
<p>
Running <code>evaluate_model.py</code> on the saved checkpoint <code>face_encodings.pkl</code> at our deployed threshold of <code>0.45</code> produced the verified results saved in <code>evaluation_split.log</code> (Table 4).
</p>

<div class="table-caption">
  <span class="figure-title">Table 4.</span> Held-out evaluation results across 471 valid test photos (from <code>evaluation_split.log</code> at threshold 0.45).
</div>
<table>
  <tr>
    <th>Evaluation Metric</th>
    <th>Measured Result</th>
  </tr>
  <tr>
    <td>Top-1 Accuracy (with 0.45 distance gate)</td>
    <td><b>98.3%</b> (0.983)</td>
  </tr>
  <tr>
    <td>Top-3 SVM Class Accuracy</td>
    <td><b>99.6%</b> (0.996)</td>
  </tr>
  <tr>
    <td>Photos Accepted by Distance Gate</td>
    <td><b>463 / 471</b> (98.3%)</td>
  </tr>
  <tr>
    <td>Wrong Identity Matches Among Accepted Photos</td>
    <td><b>0 / 463</b> (0.0%)</td>
  </tr>
  <tr>
    <td>Weighted Precision / Recall / F1</td>
    <td><b>0.998 / 0.983 / 0.990</b></td>
  </tr>
  <tr class="bottom-rule">
    <td>Macro Precision / Recall / F1</td>
    <td>0.938 / 0.919 / 0.927</td>
  </tr>
</table>

<h3 class="subsec-heading">6.1. Accuracy and Distance Distribution Analysis</h3>
<p>
Out of 471 valid held-out test photos, <b>463 photos</b> passed the <code>0.45</code> distance safety gate, and <b>every single one of those 463 accepted photos was predicted accurately</b> (zero false identity swaps). Across all 471 test photos, the minimum observed distance was <code>0.0000</code>, the mean distance was <code>0.3112</code>, and the maximum distance was <code>0.5314</code>. Figure 4 shows how the 471 test photo distances distribute across distance bins.
</p>

<div class="figure-box">
  <svg viewBox="0 0 320 130" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block; margin:0 auto;">
    <line x1="40" y1="15" x2="40" y2="100" stroke="#333" stroke-width="1"/>
    <line x1="40" y1="100" x2="305" y2="100" stroke="#333" stroke-width="1"/>
    <line x1="40" y1="20" x2="305" y2="20" stroke="#eee" stroke-width="0.7" stroke-dasharray="2,2"/>
    <line x1="40" y1="60" x2="305" y2="60" stroke="#eee" stroke-width="0.7" stroke-dasharray="2,2"/>
    <text x="35" y="23" font-family="Times New Roman" font-size="7.2" text-anchor="end" fill="#333">250</text>
    <text x="35" y="63" font-family="Times New Roman" font-size="7.2" text-anchor="end" fill="#333">125</text>
    <text x="35" y="102" font-family="Times New Roman" font-size="7.2" text-anchor="end" fill="#333">0</text>

    <rect x="52" y="90.7" width="36" height="9.3" fill="#2457c5"/>
    <text x="70" y="87" font-family="Times New Roman" font-size="7.2" text-anchor="middle" fill="#111">29</text>
    <text x="70" y="111" font-family="Times New Roman" font-size="7" text-anchor="middle" fill="#333">0.00-0.20</text>

    <rect x="104" y="50.4" width="36" height="49.6" fill="#2457c5"/>
    <text x="122" y="46" font-family="Times New Roman" font-size="7.2" text-anchor="middle" fill="#111">155</text>
    <text x="122" y="111" font-family="Times New Roman" font-size="7" text-anchor="middle" fill="#333">0.20-0.30</text>

    <rect x="156" y="23.8" width="36" height="76.2" fill="#2457c5"/>
    <text x="174" y="20" font-family="Times New Roman" font-size="7.2" text-anchor="middle" fill="#111">238</text>
    <text x="174" y="111" font-family="Times New Roman" font-size="7" text-anchor="middle" fill="#333">0.30-0.40</text>

    <rect x="208" y="86.9" width="36" height="13.1" fill="#0f766e"/>
    <text x="226" y="83" font-family="Times New Roman" font-size="7.2" text-anchor="middle" fill="#111">41</text>
    <text x="226" y="111" font-family="Times New Roman" font-size="7" text-anchor="middle" fill="#333">0.40-0.45</text>

    <rect x="260" y="97.4" width="36" height="2.6" fill="#b91c1c"/>
    <text x="278" y="93" font-family="Times New Roman" font-size="7.2" text-anchor="middle" fill="#b91c1c">8</text>
    <text x="278" y="111" font-family="Times New Roman" font-size="7" text-anchor="middle" fill="#7f1d1d">&ge;0.45 (Rej.)</text>

    <text x="172" y="125" font-family="Times New Roman" font-size="7.8" text-anchor="middle" fill="#111">Face Distance to Nearest Gallery Photo (471 Test Photos)</text>
  </svg>
  <div class="figure-caption">
    <span class="figure-title">Figure 4.</span> Distribution of minimum face distances across all 471 valid held-out test photos in <code>evaluation_split.log</code>. Nearly 90% of test photos fall below 0.40.
  </div>
</div>

<h3 class="subsec-heading">6.2. Analysis of Rejected Photos</h3>
<p>
Table 5 lists all 8 test photos rejected as <code>&lt;unknown&gt;</code> at threshold <code>0.45</code>. Even when rejected due to pose or lighting variations raising the distance past 0.45, the RBF-SVM still chose the correct person on 6 of those 8 photos (giving <b>469 / 471 = 99.6%</b> raw SVM Top-1 accuracy). Only 2 photos (out of 471) were misranked by the SVM, and both were safely blocked by the distance gate rather than marking the wrong person.
</p>

<div class="table-caption">
  <span class="figure-title">Table 5.</span> All 8 test photos rejected as <code>&lt;unknown&gt;</code> in <code>evaluation_split.log</code> (out of 471 evaluated faces).
</div>
<table>
  <tr>
    <th>Status</th>
    <th>Actual Person</th>
    <th>SVM Top Guess</th>
    <th>Distance</th>
  </tr>
  <tr>
    <td>TOP-K</td>
    <td>charlize theron</td>
    <td>charlize theron</td>
    <td>0.452154</td>
  </tr>
  <tr>
    <td>TOP-K</td>
    <td>akshay kumar</td>
    <td>akshay kumar</td>
    <td>0.458980</td>
  </tr>
  <tr>
    <td>TOP-K</td>
    <td>billie eilish</td>
    <td>billie eilish</td>
    <td>0.460846</td>
  </tr>
  <tr>
    <td>TOP-K</td>
    <td>claire holt</td>
    <td>claire holt</td>
    <td>0.469990</td>
  </tr>
  <tr>
    <td>MISS</td>
    <td>akshay kumar</td>
    <td>hrithik roshan</td>
    <td>0.488466</td>
  </tr>
  <tr>
    <td>TOP-K</td>
    <td>basab (1)</td>
    <td>basab (1)</td>
    <td>0.490899</td>
  </tr>
  <tr>
    <td>TOP-K</td>
    <td>anushka sharma</td>
    <td>anushka sharma</td>
    <td>0.500236</td>
  </tr>
  <tr class="bottom-rule">
    <td>MISS</td>
    <td>tarun yadav</td>
    <td>basab (1)</td>
    <td>0.531379</td>
  </tr>
</table>

<h3 class="subsec-heading">6.3. Distance Threshold Sweep &amp; Resolution Gain</h3>
<p>
<code>evaluate_model.py</code> sweeps five distance cutoffs from <code>0.35</code> to <code>0.55</code> on the 471 held-out photos (Figure 5). At <code>0.35</code>, strict filtering drops accuracy to 72.8%. Raising the cutoff to <code>0.45</code> brings accuracy to <b>98.3%</b>. While <code>0.55</code> reaches 99.6% on enrolled test photos in the diagnostic sweep, a looser cutoff increases the chance of accepting unknown visitors in a live room, so <b>0.45</b> is retained as the deployment default.
</p>
<p>
Comparing earlier logs (<code>training_split.log</code>, which saved 1,769 encodings) with the updated <code>train_model.py</code> pipeline (which saved 1,964 encodings) highlights the benefit of raising max image size from 400 px to 800 px, adding upsampling, and picking the largest face in group shots: nearly 200 extra valid faces were recovered from the same split.
</p>

<div class="figure-box">
  <svg viewBox="0 0 320 130" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block; margin:0 auto;">
    <line x1="45" y1="15" x2="45" y2="100" stroke="#333" stroke-width="1"/>
    <line x1="45" y1="100" x2="300" y2="100" stroke="#333" stroke-width="1"/>
    <line x1="45" y1="20" x2="300" y2="20" stroke="#e5e5e5" stroke-width="0.7" stroke-dasharray="2,2"/>
    <line x1="45" y1="47" x2="300" y2="47" stroke="#e5e5e5" stroke-width="0.7" stroke-dasharray="2,2"/>
    <line x1="45" y1="73" x2="300" y2="73" stroke="#e5e5e5" stroke-width="0.7" stroke-dasharray="2,2"/>
    <text x="40" y="23" font-family="Times New Roman" font-size="7.5" text-anchor="end" fill="#333">100%</text>
    <text x="40" y="50" font-family="Times New Roman" font-size="7.5" text-anchor="end" fill="#333">90%</text>
    <text x="40" y="76" font-family="Times New Roman" font-size="7.5" text-anchor="end" fill="#333">80%</text>
    <text x="40" y="103" font-family="Times New Roman" font-size="7.5" text-anchor="end" fill="#333">70%</text>

    <polyline points="75,92.5 125,47.7 175,24.5 225,21.6 275,21.1" fill="none" stroke="#2457c5" stroke-width="1.8"/>
    <circle cx="75" cy="92.5" r="3" fill="#2457c5"/>
    <circle cx="125" cy="47.7" r="3" fill="#2457c5"/>
    <circle cx="175" cy="24.5" r="4" fill="#059669" stroke="#fff" stroke-width="1"/>
    <circle cx="225" cy="21.6" r="3" fill="#2457c5"/>
    <circle cx="275" cy="21.1" r="3" fill="#b45309"/>

    <text x="75" y="85" font-family="Times New Roman" font-size="7.2" text-anchor="middle" fill="#111">72.8%</text>
    <text x="125" y="41" font-family="Times New Roman" font-size="7.2" text-anchor="middle" fill="#111">89.6%</text>
    <text x="175" y="16" font-family="Times New Roman" font-size="7.5" font-weight="bold" text-anchor="middle" fill="#065f46">98.3% (Deployed)</text>
    <text x="225" y="14" font-family="Times New Roman" font-size="7.2" text-anchor="middle" fill="#111">99.4%</text>
    <text x="275" y="14" font-family="Times New Roman" font-size="7.2" text-anchor="middle" fill="#92400e">99.6%</text>

    <text x="75" y="112" font-family="Times New Roman" font-size="7.5" text-anchor="middle" fill="#333">0.35</text>
    <text x="125" y="112" font-family="Times New Roman" font-size="7.5" text-anchor="middle" fill="#333">0.40</text>
    <text x="175" y="112" font-family="Times New Roman" font-size="7.5" font-weight="bold" text-anchor="middle" fill="#065f46">0.45</text>
    <text x="225" y="112" font-family="Times New Roman" font-size="7.5" text-anchor="middle" fill="#333">0.50</text>
    <text x="275" y="112" font-family="Times New Roman" font-size="7.5" text-anchor="middle" fill="#333">0.55</text>
    <text x="175" y="125" font-family="Times New Roman" font-size="7.8" text-anchor="middle" fill="#111">Distance Threshold Cutoff</text>
  </svg>
  <div class="figure-caption">
    <span class="figure-title">Figure 5.</span> Held-out Top-1 accuracy across distance thresholds from <code>evaluation_split.log</code>.
  </div>
</div>

<h2 class="sec-heading">7. Limitations and Future Improvements</h2>
<p>
Although the system performs reliably on clear photos and live webcam captures, a few practical limitations remain:
</p>
<ul class="clean-list">
  <li><b>Liveness Detection:</b> Like typical 2D camera recognition systems, the app cannot yet distinguish between a live student and a photo held up on a phone screen. Adding eye-blink or subtle head-motion verification would prevent photo spoofing.</li>
  <li><b>Student Dashboard Login:</b> While student accounts and a low-attendance alert helper (<code>get_student_summary</code> at 75%) already exist in <code>app.py</code>, login is currently limited to admin and faculty roles. Enabling student login is an easy next addition.</li>
  <li><b>Instant Enrollment:</b> Enrolling a new student currently requires adding their photo to <code>Dataset/</code> and re-running <code>train_model.py</code>. Supporting direct photo upload and incremental gallery updates from the admin page would save setup effort.</li>
  <li><b>Image Format Support:</b> Three raw files in <code>Dataset/</code> (<code>.heif</code> and <code>.webp</code>) were skipped because <code>dataset_split.py</code> reads standard <code>.jpg</code>, <code>.jpeg</code>, and <code>.png</code> images.</li>
</ul>

<h2 class="sec-heading">8. Conclusion</h2>
<p>
We built an end-to-end Automated Face Recognition System tailored for classroom attendance. By combining pretrained 128-dimensional face embeddings with an RBF-SVM classifier and a 0.45 distance safety gate, the model reaches <b>98.3% Top-1 accuracy</b> and <b>99.6% Top-3 accuracy</b> across 471 held-out test photos, with zero false identity matches among all 463 accepted detections. Paired with a Flask and SQLite web portal supporting webcam capture, photo uploads, and video sampling, the system runs cleanly on standard laptops and is completely reproducible.
</p>

<h2 class="sec-heading">References</h2>
<ol class="reference-list">
  <li>M. Turk and A. Pentland. Eigenfaces for recognition. <i>Journal of Cognitive Neuroscience</i>, 3(1):71&ndash;86, 1991.</li>
  <li>P. N. Belhumeur, J. P. Hespanha, and D. J. Kriegman. Eigenfaces vs. Fisherfaces: Recognition using class specific linear projection. <i>IEEE TPAMI</i>, 19(7):711&ndash;720, 1997.</li>
  <li>T. Ahonen, A. Hadid, and M. Pietik&auml;inen. Face description with local binary patterns. <i>IEEE TPAMI</i>, 28(12):2037&ndash;2041, 2006.</li>
  <li>Y. Taigman, M. Yang, M. Ranzato, and L. Wolf. DeepFace: Closing the gap to human-level performance in face verification. In <i>IEEE CVPR</i>, pages 1701&ndash;1708, 2014.</li>
  <li>F. Schroff, D. Kalenichenko, and J. Philbin. FaceNet: A unified embedding for face recognition and clustering. In <i>IEEE CVPR</i>, pages 815&ndash;823, 2015.</li>
  <li>D. E. King. Dlib-ml: A machine learning toolkit. <i>JMLR</i>, 10:1755&ndash;1758, 2009.</li>
  <li>D. E. King. High quality face recognition with deep metric learning. <i>dlib C++ Library Documentation</i>, 2017.</li>
  <li>V. N. Vapnik. <i>The Nature of Statistical Learning Theory</i>. Springer-Verlag, 1995.</li>
  <li>A. Geitgey. <i>face_recognition</i>: Facial recognition API for Python and the command line. GitHub Repository, 2017.</li>
  <li>A. Nguyen, T. Tran, and H. Pham. Real-time face recognition attendance system using deep learning. In <i>IEEE ICICT</i>, pages 1&ndash;6, 2021.</li>
  <li>S. Ramakrishna, V. Kumar, and P. Reddy. Automated attendance systems using facial recognition and IoT. In <i>IJCAI Workshop</i>, 2022.</li>
  <li>G. B. Huang, M. Ramesh, T. Berg, and E. Learned-Miller. Labeled faces in the wild: A database for studying face recognition in unconstrained environments. <i>UMass Tech Report 07-49</i>, 2007.</li>
  <li>C. Cortes and V. Vapnik. Support-vector networks. <i>Machine Learning</i>, 20(3):273&ndash;297, 1995.</li>
  <li>K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In <i>IEEE CVPR</i>, pages 770&ndash;778, 2016.</li>
</ol>

<h2 class="sec-heading">Appendix A: Repository Commits &amp; Timestamps</h2>
<p class="no-indent">
All source code is hosted at <code>https://github.com/teerth-iitj/Automated_Facial_Recognition_System</code>. Table 6 lists the verified git commits and timestamps on the <code>main</code> branch.
</p>

<div class="table-caption">
  <span class="figure-title">Table 6.</span> Git commit hashes, timestamps (IST, UTC+05:30), and descriptions.
</div>
<table>
  <tr>
    <th>Hash</th>
    <th>Timestamp</th>
    <th>Commit Message</th>
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
    <td>Add pre-built face gallery checkpoint (face_encodings.pkl, ~5.8 MB)</td>
  </tr>
  <tr class="bottom-rule">
    <td><code>033a61bd</code></td>
    <td>2026-11-27 04:26</td>
    <td>Repo cleanup: add .gitignore, README, REPRODUCIBILITY, LICENSE, environment.yml</td>
  </tr>
</table>

<h2 class="sec-heading">Appendix B: Reproducibility Statement &amp; Files</h2>
<p class="no-indent">
<b>Declaration Statement of Reproducibility:</b> We declare that the evaluation numbers and application behavior described in this report are reproducible using the repository source code, pinned dependency specs (<code>requirements.txt</code>, <code>environment.yml</code>), deterministic dataset splitter (<code>dataset_split.py</code>, <code>seed=42</code>), evaluation script (<code>evaluate_model.py</code>), and committed gallery checkpoint (<code>face_encodings.pkl</code>, ~5.8 MB). Raw dataset photos (~1.05 GB across 2,463 files) are excluded via <code>.gitignore</code> due to size and student privacy; folder naming rules are documented in <code>README.md</code> and <code>REPRODUCIBILITY.md</code>. Underlying pretrained ResNet face weights come from <code>face_recognition_models</code> (<code>https://github.com/ageitgey/face_recognition_models</code>).
</p>

<div class="table-caption">
  <span class="figure-title">Table 7.</span> Organized repository files and reproducibility artifacts.
</div>
<table>
  <tr>
    <th>File / Folder</th>
    <th>Purpose</th>
  </tr>
  <tr>
    <td><code>app.py</code> &amp; <code>templates/</code></td>
    <td>Flask web server, SQLite database setup, and HTML dashboards</td>
  </tr>
  <tr>
    <td><code>train_model.py</code></td>
    <td>Parallel face encoding builder and RBF-SVM classifier trainer</td>
  </tr>
  <tr>
    <td><code>evaluate_model.py</code></td>
    <td>Held-out evaluation script with overlap guard and threshold sweep</td>
  </tr>
  <tr>
    <td><code>dataset_split.py</code></td>
    <td>Deterministic 80/20 split utility (<code>seed=42</code>)</td>
  </tr>
  <tr>
    <td><code>run_all.py</code> &amp; <code>req.py</code></td>
    <td>CLI pipeline runner and automated dependency checker</td>
  </tr>
  <tr>
    <td><code>face_encodings.pkl</code></td>
    <td>Committed model checkpoint (~5.8 MB; 1,964 encodings + RBF-SVM)</td>
  </tr>
  <tr>
    <td><code>requirements.txt</code> / <code>environment.yml</code></td>
    <td>pip and Conda environment dependency files</td>
  </tr>
  <tr class="bottom-rule">
    <td><code>REPRODUCIBILITY.md</code> / <code>start.bat</code></td>
    <td>Step-by-step reproduction guide and Windows one-click launcher</td>
  </tr>
</table>

<h2 class="sec-heading">Appendix C: Instructions to Reproduce</h2>
<pre class="code-box">
# 1. Clone the repository
git clone https://github.com/teerth-iitj/Automated_Facial_Recognition_System.git
cd Automated_Facial_Recognition_System

# 2. Create environment &amp; check packages
conda env create -f environment.yml
conda activate facial_recognition
python req.py --check

# 3. Reproduce held-out evaluation (with Dataset/ present)
python evaluate_model.py --dataset Dataset --encodings face_encodings.pkl --threshold 0.45 --seed 42

# 4. Or retrain checkpoint from scratch
python train_model.py --dataset Dataset --output face_encodings.pkl --workers 4 --jitters 1

# 5. Launch the web application (or run start.bat on Windows)
python run_all.py --run --port 5000
</pre>

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
        for idx, page in enumerate(reader.pages, start=1):
            print(f"  Page {idx}: {len(page.extract_text())} chars")
        return page_count
    else:
        print("Failed to generate PDF.")
        return 0

if __name__ == "__main__":
    generate_pdf()
