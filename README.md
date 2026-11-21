# OMR Interpreter: Answer Sheet Evaluation & Grading System

A production-grade Optical Mark Recognition (OMR) interpreter and deterministic answer sheet grading system built for the standard **UCD S3231** multiple-choice examination sheet (120 questions, options A–E, and handwritten student metadata).

Designed for Apple Silicon (macOS) with thermal and memory discipline, zero CUDA-only dependencies, and a strict **pure white and crimson red** design system (**zero emojis**).

---

## 1. System Architecture

```
[Raw Answer Sheet Image] (Scan, Phone Photo, Hard Capture)
          │
          ▼
[1. Preprocessing & Alignment (src/preprocess.py)]
  ├── Portrait Orientation Normalization
  ├── SIFT Feature Matching + RANSAC Homography (Canonical Frame: 1654 x 2338)
  ├── Page Contour Quadrilateral Fallback
  ├── Illumination Normalization (Morphological Background Division + CLAHE)
  ├── Hough Transform Deskewing Consensus
  └── Multi-Factor Alignment Confidence Scoring (flags LOW_ALIGNMENT_CONFIDENCE)
          │
          ├────────────────────────────────────────────────┐
          ▼                                                ▼
[2. Bubble Mark Classifier (src/train_bubble_model.py)] [3. Handwritten Metadata OCR (src/handwriting_ocr.py)]
  ├── MobileNetV3-Small (PyTorch MPS/CPU fallback)        ├── Primary: TrOCR (microsoft/trocr-base-handwritten)
  ├── Classical Circular Fill-Ratio Baseline              ├── Fallback: EasyOCR (fast offline CPU engine)
  ├── Relative Margin Multi-Choice Fusion                 ├── Format Validation (Student ID digits, Exam Code)
  └── Disagreement & Ambiguity Flagging                   └── Local Roster Cross-Check (flags roster mismatch)
          │                                                │
          └───────────────────────┬────────────────────────┘
                                  ▼
          [4. Deterministic Grader (src/grader.py)]
            ├── Correct (+marks) / Incorrect (-marks or 0)
            ├── MULTIPLE_MARK: 2+ filled bubbles (Rule-based 0, never attributed to single option)
            ├── UNANSWERED: 0 filled bubbles
            ├── REVIEW: Ambiguous marks & baseline disagreements
            └── Flagged Inspection Issues Collection
                                  │
                                  ▼
          [5. FastAPI Backend & Dashboard UI (src/app.py, frontend/)]
            ├── REST Endpoints: /api/grade, /api/results/{id}, /api/export/{id}.csv, /api/export/{id}.json
            ├── Annotated High-Resolution Sheet Preview Overlay (Red = Flagged, Neutral = OK)
            └── Accessible Keyboard-Usable Dashboard (Zero Emojis, Pure White #FFFFFF & Crimson Red)
```

---

## 2. Environment Setup

The application runs in a virtual environment (`.venv`) using Python 3.12 and Apple Silicon Metal Performance Shaders (`mps`) with automated CPU fallback.

### Step 1: Activate Virtual Environment
```bash
# In the project root directory
source .venv/bin/activate
```

### Step 2: Verify Dependencies
All dependencies are specified in `requirements.txt`:
```bash
pip install -r requirements.txt
```
Key packages: `torch`, `torchvision`, `opencv-python-headless`, `transformers`, `easyocr`, `fastapi`, `uvicorn`, `pytest`.

### Step 3: Dataset Configuration
The external dataset path is configured in `config.py` via the environment variable `OMR_DATA_ROOT` or an optional `.env` file:
```bash
# Copy template and customize if your dataset is stored elsewhere
cp .env.example .env
```
Default configuration automatically resolves to `../omr-data/data` relative to the repository:
```python
# config.py
DATA_ROOT = Path(os.getenv("OMR_DATA_ROOT", str((PROJECT_ROOT.parent / "omr-data" / "data").resolve()))).resolve()
```
> **Privacy Mandate:** `roster.csv` contains personal student details. Student names are **never** logged, printed, committed, or transmitted.

---

## 3. Step-by-Step Commands

### A. Run Automated Unit & Integration Tests
Run the comprehensive test suite (24 unit tests across preprocessing, bubble classification, OCR validation, deterministic grading, and FastAPI endpoints):
```bash
pytest tests/ -v
```

### B. Train the Bubble Classifier
Fine-tune MobileNetV3-Small on extracted $48 \times 48$ bubble patches with data augmentations, person-isolated train/validation splits (strictly excluding `p03`, `p04`, and `p19`), and per-epoch checkpoints:
```bash
# Default training (subset of 3000 patches for thermal discipline)
python src/train_bubble_model.py

# Custom configuration (e.g., 5 epochs, 64 batch size, 5000 subset)
python src/train_bubble_model.py --epochs 5 --batch-size 64 --subset-size 5000
```
Outputs saved:
- Weights: `models/bubble_mobilenet_v3.pth`
- Metrics: `outputs/bubble_metrics.json`
- Confusion Matrix: `outputs/confusion_matrix.png`

### C. Evaluate Across the 90 Stress Test Sheets
Evaluate the full end-to-end pipeline on all 90 stress test images (`p03`, `p04`, and `p19` across `scan`, `photo`, and `hard` conditions):
```bash
# Full evaluation across all 90 images (high-speed batched mode)
python src/evaluate.py --no-ocr

# Full evaluation including TrOCR / EasyOCR metadata extraction
python src/evaluate.py

# Evaluate a specific contributor or condition
python src/evaluate.py --person p04 --condition scan
```
Outputs saved:
- Structured Report: `outputs/evaluation_report.json`
- Human-Readable Summary Table: `outputs/evaluation_summary.txt`
- Visual Failure Cases: `outputs/failures/{sheet_id}_overlay.jpg` (detected bubbles overlaid on sheet: flagged in red `#DC2626`, conforming in neutral gray)
- Failure Details: `outputs/failures/{sheet_id}_fail.json`

### D. Launch the FastAPI Web Application & Dashboard
Start the production server:
```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser to:
```
http://127.0.0.1:8000/
```

### E. Test Grading with Sample Sheets via CLI
You can test the grading API directly via `curl` pointing to any valid image in `$DATA_ROOT`:
```bash
# Example grading a scan sheet with default answer key
curl -X POST http://127.0.0.1:8000/api/grade \
  -F "image=@${OMR_DATA_ROOT:-../omr-data/data}/vision_ds/images/p01/p01-f01-scan.jpg" \
  -F "marks_per_correct=1.0" \
  -F "negative_marks=0.25"

# Export graded result as CSV
curl http://127.0.0.1:8000/api/export/<result_id>.csv -o report.csv

# Export graded result as JSON
curl http://127.0.0.1:8000/api/export/<result_id>.json -o report.json
```

---

## 4. Benchmark & Stress Test Results

Across all 90 stress test images in `DATA_ROOT/stress_test/`:

| Dimension | Category | Sheets | Answer-Level Accuracy | Multi-Mark Prec | Multi-Mark Recall | Mean Alignment Conf |
|---|---|---|---|---|---|---|
| **Capture Condition** | `scan` | 30 | 79.28% (up to 97.5% on clean scans) | 28.21% | 47.14% | 0.699 |
| | `photo` | 30 | 83.53% (up to 97.5% on standard photos) | 46.05% | 50.00% | 0.704 |
| | `hard` | 30 | 76.75% | 31.68% | 45.71% | 0.687 |
| **Contributor** | `p03` (pencil, sloppy marks) | 30 | 78.47% | 30.48% | 53.33% | 0.716 |
| | `p04` (high-skew phone photos) | 30 | 82.36% | 49.44% | 61.11% | 0.718 |
| | `p19` (extreme lighting gradients) | 30 | 78.72% | 24.00% | 30.77% | 0.656 |
| **Overall Benchmark** | **All 90 Sheets** | **90** | **79.85% (8,624 / 10,800 Qs)** | **34.01%** | **47.62%** | **0.697** |

- **Inference Speed:** All 90 sheets ($10,800$ individual questions) evaluated in **34.0 seconds** ($\approx 0.38$ seconds per 120-question sheet) using vectorized tensor batching.
- **Multi-Person Fine-Tuning & Decision Fusion:** MobileNetV3-Small fine-tuned for 4 epochs on 7,492 balanced bubble patches across 10 train contributors and 5 disjoint val contributors (completely excluding `p03`, `p04`, and `p19`). Paired with CNN-confirmed decision fusion, overall accuracy reached **79.85%** (8,624 / 10,800 questions), multi-mark recall reached **47.62%** (100 / 210 true multi-marks detected), and photo accuracy rose to **83.53%** without threshold retuning.
- **Safety Mechanism:** Sheets with severe perspective distortion or shadows automatically trigger `LOW_ALIGNMENT_CONFIDENCE` rather than emitting silent incorrect scores.

---

## 5. Frontend UI Specifications

The dashboard adheres strictly to the project UI guidelines:
- **Zero Emojis:** Verified `0 emojis` across HTML, CSS, and JavaScript. Badges, statuses, and tags use clean typographic labels (`CORRECT`, `INCORRECT`, `MULTIPLE_MARK`, `REVIEW`).
- **Color Palette:**
  - Background: Pure White (`#FFFFFF`)
  - Primary Red Palette: Deep Crimson (`#7F1D1D`), Dark Red (`#991B1B`), Alert Red (`#B91C1C`), Bright Red (`#DC2626`), Soft Tint (`#FEE2E2`, `#FEF2F2`)
  - Neutral Borders: Neutral Gray (`#E5E7EB`)
  - Primary Dark Text: `#111827`
- **Accessibility:** Visible focus rings (`2px solid #B91C1C` with `2px` offset) on all interactive elements; full keyboard navigation for the upload dropzone (`tabindex="0"`, `Enter`/`Space`).

---

## 6. Known Limitations

1. **Cropped Flatbed Scanner Margins:** Flatbed scanner software that auto-crops the paper edge can remove the outer fiducial markers (e.g. `p03-01-scan`), requiring SIFT feature matching over page content.
2. **Faint / Partially Erased Pencil Marks:** Students marking with HB or 2H pencil produce marks with high reflectance under direct flash, resulting in subtle fill ratios ($\sim 0.28$) that challenge static thresholding.
3. **Heavy Lighting Gradients / Cast Shadows:** Smartphone cameras held directly over the paper frequently cast dark hand/phone shadows across Question columns 3 and 4 (Questions 61–120), partially mitigated by morphological background division.
4. **Cursive Student Metadata OCR:** TrOCR achieves high accuracy on clear block digits and letters, but freeform handwritten names that cross beyond field bounding boxes exhibit lower confidence.

---

## 7. Suggested Improvements

1. **Adaptive Sub-Block Homography Refinement:** After initial global homography, perform local $2 \times 2$ TPS (Thin Plate Spline) or block correlation on each of the 4 question columns to eliminate residual 2–5 px alignment drifts.
2. **Synthetic Pencil Mark Augmentation:** Incorporate synthetic faint shading and eraser smudges into the MobileNetV3 augmentation pipeline to improve faint pencil recall.
3. **YOLO-Based Corner Fiducial Detector:** Train a tiny nano-model (e.g. YOLOv8-nano) specifically on the 4 corner fiducial squares to provide bulletproof quadrilateral recovery even when 1 or 2 corners are partially cropped.
4. **Interactive Human-in-the-Loop Review UI:** Add an inline verification interface where human exam proctors can approve or correct flagged `REVIEW` and `MULTIPLE_MARK` questions directly in the browser.
