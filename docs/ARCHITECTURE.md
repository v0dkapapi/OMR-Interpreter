# System Architecture & Technical Specification

The **OMR Interpreter** is an end-to-end computer vision and machine learning pipeline designed to automatically align, read, and grade standard multiple-choice answer sheets (specifically the UCD S3231 examination form format).

---

## 1. High-Level Architecture Overview

```mermaid
flowchart TD
    A["Raw Answer Sheet Image\n(Scan, Phone Photo, Hard Capture)"] --> B["1. Preprocessing & Alignment\n(src/preprocess.py)"]
    
    subgraph Preprocessing ["Phase 1: Preprocessing & Normalization"]
        B --> B1["Portrait Normalization"]
        B1 --> B2["SIFT Matching + RANSAC Homography\n(Canonical: 1654 x 2338)"]
        B2 --> B3["Quadrilateral Contour Fallback"]
        B3 --> B4["Morphological Illumination Flattening\n+ CLAHE Contrast Enhancement"]
        B4 --> B5["Hough Deskewing Consensus"]
        B5 --> B6["Multi-Factor Alignment Telemetry\n(Flags LOW_ALIGNMENT_CONFIDENCE)"]
    end
    
    B6 --> C["Canonical Aligned Image (1654 x 2338)"]
    
    C --> D["Phase 2: Bubble Mark Classification\n(src/train_bubble_model.py, src/grader.py)"]
    C --> E["Phase 3: Handwritten Metadata OCR\n(src/handwriting_ocr.py)"]
    
    subgraph MarkInference ["Phase 2: Dual-Channel Bubble Inference"]
        D --> D1["Crop 48x48 Bubble Patches\n(120 Qs x 5 Options = 600 Patches)"]
        D1 --> D2["Channel A: Classical Circular Fill Ratio\n(Otsu Threshold + Circular Mask)"]
        D1 --> D3["Channel B: MobileNetV3-Small CNN\n(3 Classes: unfilled, filled, ambiguous)"]
        D2 --> D4["Decision Fusion & Disagreement Detection"]
        D3 --> D4
    end
    
    subgraph MetadataExtraction ["Phase 3: Student Details OCR"]
        E --> E1["Crop Metadata ROIs\n(Name, Student ID, Module, Seat, Date)"]
        E1 --> E2["TrOCR Primary / EasyOCR Fallback"]
        E2 --> E3["8-Column Digit Grid Bubble Read"]
        E3 --> E4["Format Validation & Roster Cross-Check"]
    end
    
    D4 --> F["Phase 4: Deterministic Grading Engine\n(src/grader.py)"]
    E4 --> F
    
    subgraph GradingEngine ["Phase 4: Rule-Based Evaluation"]
        F --> F1["CORRECT (+marks)"]
        F --> F2["INCORRECT (-marks or 0)"]
        F --> F3["MULTIPLE_MARK (Rule 0, never single-option)"]
        F --> F4["UNANSWERED (0 marks)"]
        F --> F5["REVIEW (Ambiguity / Disagreement Flag)"]
    end
    
    F --> G["Phase 5: FastAPI Backend & Web Dashboard\n(src/app.py, frontend/)"]
    
    subgraph Presentation ["Phase 5: Telemetry & Delivery"]
        G --> G1["REST API: /api/grade, /api/results/{id}"]
        G --> G2["Export Endpoints: CSV, JSON"]
        G --> G3["Annotated Sheet Preview Canvas\n(Red: Flagged, Gray: Conforming)"]
        G --> G4["Pure White & Crimson Red Dashboard UI"]
    end
```

---

## 2. Component Pipeline Details

### 2.1 Phase 1: Preprocessing & Canonical Alignment (`src/preprocess.py`)
1. **Orientation Correction:** Checks dimensions and aspect ratio; rotates landscape images by 90 or 270 degrees into portrait orientation.
2. **Feature-Based Homography:** Computes SIFT features on the input image and matches them against precomputed template keypoints. RANSAC computes a perspective transform matrix $H \in \mathbb{R}^{3 \times 3}$, warping the image to the canonical frame ($1654 \times 2338$ pixels, corresponding to A4 at 200 DPI).
3. **Geometric Fallback:** If SIFT inliers fall below acceptable thresholds, an adaptive contour detector locates the largest 4-sided convex polygon representing the page boundary and performs a quadrilateral perspective transform.
4. **Illumination Correction:** Normalizes severe lighting gradients and smartphone cast shadows using morphological background division (elliptical structuring element of size $51 \times 51$) followed by Contrast Limited Adaptive Histogram Equalization (CLAHE).
5. **Consensus Deskewing:** Applies probabilistic Hough line transform to detect dominant horizontal and vertical lines, correcting residual rotation skew within $\pm 2^\circ$.
6. **Alignment Confidence Telemetry:** Calculates a multi-factor score (0.0 to 1.0) incorporating SIFT inlier ratio, homography determinant condition, and edge gradient correlation. If the score falls below `0.65`, the sheet is tagged with `LOW_ALIGNMENT_CONFIDENCE` to warn examiners.

### 2.2 Phase 2: Bubble Mark Extraction & Inference
- **ROI Projection:** Based on `template.json`, coordinates are projected for 120 questions $\times$ 5 options (`A` through `E`) = 600 bubble locations, plus 8 columns $\times$ 10 digit bubbles for student numbers.
- **Dual Inference Channels:**
  - **Classical Baseline:** Extracts circular ROI, computes Otsu binarization, and measures fill ratio against a circular mask. Fill ratios $\ge 0.38$ indicate filled bubbles; ratios between $0.20$ and $0.42$ indicate ambiguous marks.
  - **MobileNetV3-Small Classifier:** 48x48 patches are evaluated by a fine-tuned MobileNetV3-Small network outputting probabilities across 3 classes: `unfilled`, `filled`, and `ambiguous`.
- **Decision Fusion Implementation (`src/fusion.py`, shared by `src/app.py`, `src/evaluate.py` and `src/batch_grade.py`):**
  - All 600 patches on a sheet are classified in a single CNN batch. Thresholds are named constants at the top of `src/fusion.py`, tuned on validation contributors only (Experiment 4).
  - **Multiple mark:** two or more options CNN-`filled` with confidence $> 0.80$. Confident CNN evidence on every option keeps shadows and printed letters from invalidating a correct answer.
  - **Fill-margin single answer:** darkest option fill $\ge 0.26$ and either $\ge 0.05$ ahead of the runner-up or CNN-`filled` with confidence $> 0.50$. The CNN vetoes it when it says `unfilled` with confidence $> 0.50$, which removes most false marks on blank rows. A vetoed bubble with fill $\ge 0.80$ is a genuine disagreement and goes to `REVIEW`.
  - **CNN single answer:** if nothing was selected and exactly one option is CNN-`filled` with confidence $> 0.80$, it is accepted (faint pencil marks). If its fill ratio is below $0.10$, the disagreement goes to `REVIEW`.
  - **Fill-margin-only mode:** without trained weights in `models/bubble_mobilenet_v3.pth`, every option gets CNN class `none`, so only the fill-margin rule fires (no veto and no multi-mark detection; 71.43% on the stress test). `/api/health` reports the active `bubble_reader_mode`.

### 2.3 Phase 3: Handwritten Metadata OCR (`src/handwriting_ocr.py`)
- **Metadata Fields:** Name, Student Number (written), Module Code, Seat Number, Examination Date.
- **Localization:** Each field has a search cell (`FIELD_CELLS`). Ink is detected against a local background estimate, printed label glyphs are erased (components wholly inside `LABEL_BOXES`), printed rules removed, and the handwriting cropped tightly; entries written over two lines are split and read line by line.
- **Constrained n-best decoding:** TrOCR (`microsoft/trocr-base-handwritten`) decodes all field crops in one batch with 5 beams. A per-field token mask blocks tokens that cannot occur in the field (module code: uppercase and digits; date: digits, separators and month-name pieces; name: letters), which stops the language model rewriting codes into English words.
- **Validated selection (`src/field_parsing.py`):** The first beam that parses as a valid value wins: names need two or three words without function words; module codes are a known subject prefix (`OCR_CONFIG["module_prefixes"]`, snapped by edit distance) plus 4-5 digits; dates are normalized to ISO `YYYY-MM-DD` from any common layout; seat numbers are an optional letter plus 1-4 digits. EasyOCR is the fallback reader. If nothing validates, the top reading is returned with `is_valid: false`, and the dashboard shows a "Check" badge.
- **Written student number:** read from the 8 handwritten boxes (TrOCR, EasyOCR and per-cell digits). The bubble grid below the boxes is not yet decoded (see `docs/KNOWN_ISSUES_AND_ROADMAP.md`).
- **Privacy Isolation:** Optional validation cross-checks student ID digits against `roster.csv` without ever logging, printing, or transmitting student names.

### 2.4 Phase 4: Deterministic Grader (`src/grader.py`)
Grading executes purely deterministic rules:
- **`CORRECT`:** Exactly one bubble marked matching the answer key. Awarded full marks (default: +1.0).
- **`INCORRECT`:** Exactly one bubble marked differing from the answer key. Deducted negative marks (if configured) or awarded 0.0.
- **`MULTIPLE_MARK`:** Two or more bubbles detected as filled for a single question. Strictly awarded 0.0 marks; never attributed to a single option.
- **`UNANSWERED`:** Zero bubbles detected as filled. Awarded 0.0 marks.
- **`REVIEW`:** Disagreements between classical fill ratio and neural classifier, faint pencil marks, or ambiguous symbols. Flags the question for human verification while preserving intermediate metrics.

### 2.5 Phase 5: FastAPI Backend & Web UI (`src/app.py`, `frontend/`)
- **REST Endpoints:**
  - `POST /api/grade`: Upload sheet image with optional scoring parameters and custom answer key (JSON or CSV).
  - `GET /api/results/{id}`: Retrieve structured grading report and question statuses.
  - `GET /api/export/{id}.csv` and `GET /api/export/{id}.json`: Download evaluation reports.
  - `GET /api/health`: Health status and device telemetry.
- **UI Design System:**
  - Pure white background (`#FFFFFF`) with crimson red accent palette (`#7F1D1D`, `#B91C1C`, `#DC2626`).
  - Zero emojis across all interface copy, status badges, and notifications.
  - High-resolution interactive canvas overlay highlighting flagged questions in red and conforming questions in neutral gray.
