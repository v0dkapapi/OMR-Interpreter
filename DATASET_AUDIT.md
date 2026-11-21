# OMR Interpreter: Dataset Audit Report

**Date of Audit:** November 17, 2026  
**Auditor:** Antigravity AI  
**Dataset Name:** S3231 Answer Sheet Dataset (v1.0) & Stress Test Evaluation Suite  
**Location:** `$DATA_ROOT` (default: `../omr-data/data/`, configured via `config.py`)  
**Compliance & Privacy:** Roster inspected for schema/headers only. Zero student names logged, printed, or exported.

---

## 1. Executive Summary

A comprehensive, non-destructive audit of the S3231 Answer Sheet Dataset was conducted to verify file integrity, metadata schemas, image characteristics, label coverage, and stress test properties.

Key Findings:
1. **Dataset Scale:** 740 filled forms across 74 contributors (10 forms per contributor). 2,184 images totaling ~5.22 GB.
2. **Image Conditions:** Forms were captured across three primary conditions: `scan` (733), `photo` (719), and `hard` (732). Average 2.95 images per form.
3. **Absence of Bounding Box Labels:** Ground truth labels (`labels/labels.csv` and `labels/json/`) contain sheet-level and question-level answers (`q1` to `q120`), student ID digits, and handwritten metadata. **No bubble bounding boxes or pixel coordinates exist in the ground truth.** Bubble patches for model training must be derived via geometric template ROIs on aligned sheets.
4. **Stress Test Overlap & Leakage Risk:** `DATA_ROOT/stress_test/` contains 90 raw images across contributors `p03`, `p04`, and `p19` (10 forms each, 3 conditions each). In `vision_ds/roster.csv` and `labels/labels.csv`, `p03` and `p19` are assigned to the `train` split, while `p04` is assigned to `test`. **`p03`, `p04`, and `p19` must be strictly excluded from all training and validation splits** in `vision_ds` to prevent train/test leakage.
5. **Labels for Stress Test Exist:** Because `p03`, `p04`, and `p19` have complete ground truth entries in `labels/labels.csv` and `labels/json/`, full quantitative accuracy and multi-mark precision/recall evaluation can be executed on the stress test suite.
6. **Minor File Discrepancy:** Exactly one image naming mismatch exists in `p10`: `images/p10/p10-f07-photo.jpg` exists on disk but is absent from `images.csv`, while `images/p10/p10-f08-photo.jpg` is in `images.csv` but absent from disk. All other 2,183 images match 1-to-1. There are 0 duplicate image files.

---

## 2. Dataset Structure and Inventory

```
$DATA_ROOT/ (e.g. ../omr-data/data/)
├── vision_ds/
│   ├── README.md               # Authoritative dataset documentation (v1.0)
│   ├── roster.csv              # Contributor-to-person mapping (74 rows)
│   ├── images/                 # 74 contributor directories (p01 - p74), 2,184 JPG files (~5.22 GB)
│   └── labels/
│       ├── labels.csv          # 740 form label rows (139 columns)
│       ├── images.csv          # 2,184 image inventory rows
│       └── json/               # 740 per-form ground truth JSON files
└── stress_test/
    ├── p03/                    # 30 raw evaluation images (p03-f01 to p03-f10)
    ├── p04/                    # 30 raw evaluation images (p04-f01 to p04-f10)
    └── p19/                    # 30 raw evaluation images (p19-f01 to p19-f10)
```

---

## 3. Privacy & Personal Data Audit (`roster.csv`)

- **File Header:** `contributor,person,split`
- **Rows:** 74 data rows (1 header + 74 contributor entries).
- **Split Distribution in Roster:**
  - `train`: 44 contributors (~59.5%)
  - `val`: 15 contributors (~20.3%)
  - `test`: 15 contributors (~20.3%)
- **Privacy Protocol:**
  - Column `person` contains real student identity strings.
  - In accordance with specification rules, `roster.csv` remains strictly outside the workspace, is never committed, never logged, and student names are never displayed.
  - In the pipeline, `roster.csv` is accessed only for local cross-validation of OCR student IDs.

---

## 4. Ground Truth Labels Inspection

### 4.1 `labels/labels.csv` Schema
- **Rows:** 740 rows (1 row per form; exactly 10 forms per contributor).
- **Columns:** 139 total columns:
  - Form & Contributor: `form_id`, `contributor`, `form_no`, `split`
  - Student Identifier: `student_number` (8-char bubble representation, `-` = empty, `?` = invalid), `student_number_written` (handwritten digits)
  - Questions: `q1` to `q120`
  - Annotations: `changed` (erased markings, e.g. `q11:E>C`), `sloppy` (irregular marks, e.g. `q12:circle B; q21:multiple CD`)
  - Handwritten Metadata: `name`, `module_code`, `seat_number`, `date`
  - Capture & Form Metadata: `tool` (pencil/pen), `style` (fill/line), `n_answered` (10 to 120), `hard_photo` (condition description), `notes`, `n_images`, `image_kinds`

### 4.2 Answer Value Statistics (`q1` – `q120`)
Across 740 forms $\times$ 120 questions = 88,800 question cells:
- **Single Marked Option (`A`, `B`, `C`, `D`, `E`):** 42,610 instances (48.0%)
- **Blank / Unanswered (`-`):** 44,078 instances (49.6%)
- **Symbol / Non-standard Mark (`?`):** 2,112 instances (2.4%)
- **Multi-mark combinations (e.g. `BD`, captured under `sloppy` or raw):** Present in `sloppy` and raw strings.

### 4.3 `labels/json/` Schema
- **Count:** 740 JSON files matching `<form_id>.json`.
- **Top-level Keys:**
  - `form_id`: string (e.g. `"p01-f01"`)
  - `student_number`: string (e.g. `"74228162"`) or `null` if invalid
  - `answers`: dictionary with string keys `"1"` through `"120"`. Marked values are `"A"`..`"E"`; blank (`-`) and invalid (`?`) are normalized to `null`.
  - `raw`: contains exact un-normalized symbols (`"A"`-`"E"`, `"-"`, `"`?`"`)
  - `meta`: dictionary of capture and form attributes (`contributor`, `split`, `tool`, `style`, `n_answered`, `hard_photo`, `changed`, `sloppy`, `notes`, `n_images`, `image_kinds`)
  - Handwritten metadata: `name`, `module_code`, `seat_number`, `date`

---

## 5. Vision Dataset Images Inspection (`vision_ds/images/`)

### 5.1 Verification of ~3 Images per Form Hypothesis
- Forms with 3 images: **712** (96.2%)
- Forms with 2 images: **17** (2.3%)
- Forms with 1 image: **10** (1.4%)
- Forms with 4 images: **1** (`p70-f01` has `scan`, `hard`, `photo`, `photo2`)
- **Total Images:** 2,184
- **Average:** 2.951 images per form. Hypothesis confirmed.

### 5.2 Contributor Breakdown
- 65 contributors have exactly 30 images.
- 9 contributors have fewer or different image totals:
  - `p05`: 29 images
  - `p10`: 29 images
  - `p27`: 29 images
  - `p35`: 29 images
  - `p40`: 24 images
  - `p48`: 10 images
  - `p50`: 28 images
  - `p58`: 28 images
  - `p62`: 28 images

### 5.3 Image Resolutions & Formats
- 1,024 distinct resolutions detected across 2,184 images.
- Most frequent resolutions:
  - `4032 x 3024`: 235 images
  - `4284 x 5712`: 109 images
  - `3024 x 4032`: 106 images
  - `3072 x 4096`: 64 images
  - `1200 x 1600`: 57 images
  - `1536 x 2048`: 40 images
  - `960 x 1280`: 37 images
- Aspect ratios and orientations vary (landscape vs portrait phone shots). Multi-Picture Object (MPO) format metadata is present in some smartphone captures but decodable as standard JPEG.

### 5.4 Orphan & Integrity Audit
- **Duplicates:** Zero duplicate files (SHA-256 hash comparison across all 2,184 images).
- **Inventory Mismatch in `p10`:**
  - `vision_ds/images/p10/p10-f07-photo.jpg` exists on disk but is omitted from `images.csv`.
  - `images/p10/p10-f08-photo.jpg` is recorded in `images.csv` but missing from disk.
  - In `labels.csv`: `p10-f07` specifies `n_images=2, kinds=scan hard`; `p10-f08` specifies `n_images=3, kinds=scan photo hard`.
  - Implication: Handled gracefully in dataset loading by querying physical disk existence rather than blindly trusting `images.csv`.

---

## 6. Stress Test Evaluation Suite (`stress_test/`)

### 6.1 Directory & File Contents
- **Total files:** 90 images located in subdirectories `stress_test/p03/`, `stress_test/p04/`, `stress_test/p19/`.
- **Form count:** Exactly 10 forms per person (`f01` to `f10`).
- **Condition breakdown per person:**
  - `p03`: 10 hard, 10 scan, 10 photo
  - `p04`: 10 hard, 10 scan, 10 photo
  - `p19`: 10 hard, 10 scan, 10 photo

### 6.2 Naming Inconsistencies & Regex Verification
The stress test files exhibit diverse naming conventions:
- Form notation omitting 'f': `p03-01-scan.jpg`, `p19-04-hard.jpg`
- Condition naming variations: `p03-f01-hardphoto.jpg`
- File extensions: `.jpeg` on `p04` (`p04-f01-hard.jpeg`) vs `.jpg`

**Regex Pattern Implemented & Tested:**
```python
r'^(?P<person>p\d+)[-_](?:f?(?P<form>\d+))[-_](?P<kind>scan\d*|photo\d*|hardphoto\d*|hard\d*)\.(?P<ext>jpg|jpeg)$'
```
- **Parsing Rate:** 90/90 (100% matched, 0 unparsed).
- **Normalization:** `hardphoto` $\rightarrow$ `hard`; `.jpeg` $\rightarrow$ `.jpg`; form `01` $\rightarrow$ `f01`.

### 6.3 Overlap Analysis & Train/Test Isolation
- `p03` and `p19` are marked `train` in `vision_ds/labels/labels.csv` and have corresponding directories in `vision_ds/images/`.
- `p04` is marked `test` in `vision_ds/labels/labels.csv`.
- Hash check demonstrates that `stress_test` contains raw, unscaled/unconverted uploads whereas `vision_ds` underwent clean-up.
- **Mandate:** `p03`, `p04`, and `p19` must be explicitly excluded from all training and validation splits in `vision_ds` to prevent any data leakage.
- **Ground Truth Availability:** All 30 forms for `p03`, `p04`, and `p19` exist in `labels/labels.csv` and `labels/json/`, allowing full evaluation of answer-level accuracy and multi-mark detection.

---

## 7. Bubble Patch Derivation Strategy

Because the dataset does not provide bounding box annotations for individual bubbles, we establish an automated patch extraction and labeling strategy:

```
[Input Image (Scan/Photo/Hard)]
             │
             ▼
[Preprocessing & Perspective Correction]
   • Fiducial marker / corner detection
   • Homography transform to Canonical Sheet Frame (e.g. 1654 x 2338)
             │
             ▼
[Template ROI Projection]
   • Project fixed ROI coordinates (x, y, w, h) for every Question (1..120) × Option (A..E)
   • Project fixed ROI coordinates for 8 Student ID columns × 10 digits (0..9)
             │
             ▼
[Ground Truth Mapping]
   • If labels.csv Question answer == Option:
       Assign label -> "filled"
   • If Question answer == "-" or (answer in 'ABCDE' and answer != Option):
       Assign label -> "unfilled"
   • If Question has multiple marks (e.g., 'BD') or appears in `sloppy` / `changed`:
       Assign label -> "ambiguous" (or "filled" for marked options, "ambiguous" for erased/faint)
             │
             ▼
[Cropped Patch (e.g. 32x32 or 48x48)] -> Cached in cache/patches/
```

### Classical Baseline & Sanity Check
Alongside the neural network bubble classifier (MobileNetV3-small / ResNet-18), a classical fill-ratio method (adaptive thresholding, morphological opening, circular mask fill ratio) will be implemented as a baseline and sanity check. Disagreements between the neural model and classical baseline will be flagged for manual review (`REVIEW`).

---

## 8. Explicit Assumptions

1. **Sheet Geometry & Canonical Template:** All 740 forms in this dataset follow the UCD S3231 layout:
   - 120 multiple-choice questions arranged in 4 columns of 30 questions each.
   - 5 options per question (`A`, `B`, `C`, `D`, `E`).
   - Student number grid: 8 columns $\times$ 10 digits (`0` to `9`), with handwritten boxes above.
   - Handwritten header fields: Name, Module Code, Seat Number, Date.
   - Corner/boundary fiducial alignment marks to guide homography.
2. **Train/Val Split Strategy:**
   - Person-based splitting: all forms of a person stay in the same split.
   - Persons `p03`, `p04`, and `p19` are excluded from training and validation.
   - Remaining contributors in `train` split are used for model training; remaining `val` contributors are used for validation.
3. **Hardware & Compute Constraints:**
   - Environment: MacBook Air Apple Silicon (M5 fanless).
   - PyTorch device: `"mps"` with `PYTORCH_ENABLE_MPS_FALLBACK=1`, falling back to CPU.
   - Training runs on resized/downscaled cached patches in `cache/` (git-ignored) with configurable subset size, epochs, and batch size.
4. **Offline Resilience:**
   - TrOCR (`microsoft/trocr-base-handwritten`) downloads are cached locally; if offline or unavailable, fallback to EasyOCR executes gracefully.
