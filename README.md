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
            ├── Marked Sheet Preview (Green = correct answer, Red = wrong or invalid choice)
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

### 0. Set Up the Dataset (optional for tests, required for training and evaluation)
Google Drive downloads the dataset as several zip parts. Merge them into `OMR_DATA_ROOT` (default `../omr-data/data`):
```bash
python scripts/setup_dataset.py ~/Downloads/AI-for-Vision-Dataset-*.zip
```
This creates `vision_ds/` and a `stress_test/` folder of symlinks to the `p03`, `p04` and `p19` images.

### A. Run Automated Unit & Integration Tests
Run the test suite (preprocessing, decision fusion, OCR validation, deterministic grading, batch CLI, and FastAPI endpoints):
```bash
pytest tests/ -v
```
The suite does not need the private dataset: integration tests generate synthetic answer sheets from `template.json` (see `tests/synthetic.py`), including perspective-warped "phone" captures that exercise SIFT alignment. Three tests skip automatically when `OMR_DATA_ROOT` or the local `roster.csv` is absent.

### B. Train the Bubble Classifier
Fine-tune MobileNetV3-Small on extracted $48 \times 48$ bubble patches with data augmentations, person-isolated train/validation splits (strictly excluding `p03`, `p04`, and `p19`), and per-epoch checkpoints:
```bash
# Benchmark configuration (Run 4: 90 forms across 71 contributors, about 1 minute on an M-series Mac)
python src/train_bubble_model.py --epochs 5 --subset 20000 --max-forms 90 --force-extract

# Quick smoke run (defaults: 4 epochs, 5000-patch subset, 15 forms)
python src/train_bubble_model.py
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
- Visual Failure Cases: `outputs/failures/{sheet_id}_overlay.jpg` (mismatched choices in red `#DC2626`, missed ground-truth answers outlined in green `#16A34A`)
- Failure Details: `outputs/failures/{sheet_id}_fail.json`

### D. Batch-Grade a Folder of Sheets
Grade every image in a folder with the same pipeline as the web API, using several worker processes:
```bash
python -m src.batch_grade path/to/sheets --out outputs/batch --workers 4 \
    --answer-key answer_key.json --negative-marks 0.25
```
Outputs: `outputs/batch/batch_summary.csv` (one row per sheet) and `outputs/batch/sheets/{stem}.json` (full report). Add `--ocr` to extract handwritten fields; the name field is always redacted in batch outputs. A sheet that fails to load is reported as `ERROR` without stopping the batch.

### E. Launch the FastAPI Web Application & Dashboard
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

Across all 90 stress test images in `DATA_ROOT/stress_test/` (Run 4, see `docs/EVALUATION_PROTOCOL.md`):

| Dimension | Slice | Sheets | Questions | Accuracy (%) | Invalid Prec | Invalid Rec | Invalid F1 | REVIEW rate | Mean Align Conf |
|---|---|---|---|---|---|---|---|---|---|
| **Condition** | `scan` | 30 | 3,600 | 93.92% | 75.56% | 48.57% | 59.13% | 1.78% | 0.712 |
| | `photo` | 30 | 3,600 | 93.86% | 75.00% | 47.14% | 57.89% | 2.83% | 0.697 |
| | `hard` | 30 | 3,600 | 92.06% | 68.89% | 44.29% | 53.91% | 2.64% | 0.699 |
| **Contributor** | `p03` | 30 | 3,600 | 92.97% | 100.00% | 35.00% | 51.85% | 3.92% | 0.754 |
| | `p04` | 30 | 3,600 | 98.42% | 92.16% | 65.28% | 76.42% | 0.39% | 0.714 |
| | `p19` | 30 | 3,600 | 88.44% | 48.39% | 38.46% | 42.86% | 2.94% | 0.641 |
| **Overall** | **All Sheets** | **90** | **10,800** | **93.28%** | **73.13%** | **46.67%** | **56.98%** | **2.42%** | **0.703** |

- **Inference Speed:** All 90 sheets (10,800 questions) evaluated in about 23.5 seconds (about 0.26 seconds per sheet, `--no-ocr`), with all 600 bubbles per sheet classified in one batch.
- **Training:** MobileNetV3-Small fine-tuned for 5 epochs on 14,400 patches from 42 training contributors, validated on 5,600 patches from 29 disjoint contributors (92.84% patch accuracy). `p03`, `p04` and `p19` are never used for training or tuning.
- **Decision Fusion:** CNN-veto fusion (`src/fusion.py`) tuned on validation contributors only. The previous rules scored 75.20% on the same stress set; most of the gain comes from stopping blank questions being read as answered.
- **Invalid answers:** ground truth `?` covers both multiple marks and ticks, circles, crosses and faint marks, so the precision and recall columns measure invalid-answer detection.
- **Safety Mechanism:** Sheets with severe perspective distortion or shadows automatically trigger `LOW_ALIGNMENT_CONFIDENCE` rather than emitting silent incorrect scores.

---

## 5. Frontend UI Specifications

The dashboard adheres strictly to the project UI guidelines:
- **Zero Emojis:** Verified `0 emojis` across HTML, CSS, and JavaScript. Badges, statuses, and tags use clean typographic labels (`CORRECT`, `INCORRECT`, `MULTIPLE_MARK`, `REVIEW`).
- **Color Palette:**
  - Background: Pure White (`#FFFFFF`)
  - Primary Red Palette: Deep Crimson (`#7F1D1D`), Dark Red (`#991B1B`), Alert Red (`#B91C1C`), Bright Red (`#DC2626`), Soft Tint (`#FEE2E2`, `#FEF2F2`)
  - Neutral Borders: Neutral Gray (`#E5E7EB`)
  - Answer Marking (sheet overlay only): Green (`#16A34A`) for correct answers, Bright Red (`#DC2626`) for wrong or invalid choices
  - Primary Dark Text: `#111827`
- **Accessibility:** Visible focus rings (`2px solid #B91C1C` with `2px` offset) on all interactive elements; full keyboard navigation for the upload dropzone (`tabindex="0"`, `Enter`/`Space`).

---

## 6. Known Limitations

1. **Cropped Flatbed Scanner Margins:** Flatbed scanner software that auto-crops the paper edge can remove the outer fiducial markers (e.g. `p03-01-scan`), requiring SIFT feature matching over page content.
2. **Faint / Partially Erased Pencil Marks:** Students marking with HB or 2H pencil produce marks with high reflectance under direct flash, resulting in subtle fill ratios ($\sim 0.28$) that challenge static thresholding.
3. **Heavy Lighting Gradients / Cast Shadows:** Smartphone cameras held directly over the paper frequently cast dark hand/phone shadows across Question columns 3 and 4 (Questions 61–120), partially mitigated by morphological background division.
4. **Handwritten Field OCR:** Held-out exact match is about 67% for names, 61% for module codes, 54% for dates and 53% for seat numbers (Experiment 5). Readings that fail format validation are flagged with a "Check" badge. The written student number is still unreliable (about 16%). Benchmark with `python scripts/evaluate_ocr.py --forms 20 --kinds scan photo hard`.

---

## 7. Suggested Improvements

1. **Adaptive Sub-Block Homography Refinement:** After initial global homography, perform local $2 \times 2$ TPS (Thin Plate Spline) or block correlation on each of the 4 question columns to eliminate residual 2–5 px alignment drifts.
2. **Synthetic Pencil Mark Augmentation:** Incorporate synthetic faint shading and eraser smudges into the MobileNetV3 augmentation pipeline to improve faint pencil recall.
3. **YOLO-Based Corner Fiducial Detector:** Train a tiny nano-model (e.g. YOLOv8-nano) specifically on the 4 corner fiducial squares to provide bulletproof quadrilateral recovery even when 1 or 2 corners are partially cropped.
4. **Interactive Human-in-the-Loop Review UI (still open):** Add an inline verification interface where human exam proctors can approve or correct flagged `REVIEW` and `MULTIPLE_MARK` questions directly in the browser.
