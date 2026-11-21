# Dataset Specification & Data Privacy Protocol

This document provides the authoritative inventory, schema specification, splitting rules, and privacy mandates for the S3231 Answer Sheet Dataset used in the OMR Interpreter project.

---

## 1. Dataset Inventory & Physical Characteristics

The dataset consists of standard examination answer sheets for the UCD S3231 examination form format.

- **Dataset Name:** S3231 Answer Sheet Dataset (v1.0)
- **Total Physical Forms:** 740 filled forms.
- **Total Contributors:** 74 distinct students (`p01` through `p74`), each completing exactly 10 forms (`f01` through `f10`).
- **Total Captured Images:** 2,184 images (~5.22 GB total).
- **Average Captures per Form:** 2.951 images per form.
  - Forms with 3 images: 712 (96.2%)
  - Forms with 2 images: 17 (2.3%)
  - Forms with 1 image: 10 (1.4%)
  - Forms with 4 images: 1 (`p70-f01` has `scan`, `hard`, `photo`, `photo2`)

### 1.1 Capture Conditions
Each form was captured under up to three distinct photographic conditions:
1. **`scan` (733 images):** Flatbed document scanners or flat scanning apps. High contrast, uniform lighting, minimal perspective distortion, pure white paper background.
2. **`photo` (719 images):** Handheld smartphone photographs under standard indoor room lighting. Moderate perspective skew, minor barrel distortion, ambient illumination.
3. **`hard` (732 images):** Severe capture stress tests. Highly angled perspective skew, low or uneven lighting, harsh specular glare, cast phone/hand shadows across question columns.

### 1.2 File Inventory Note (`p10` Discrepancy)
During dataset audit, exactly one inventory discrepancy was documented in contributor `p10`:
- `images/p10/p10-f07-photo.jpg` physically exists on disk but is absent from `labels/images.csv`.
- `images/p10/p10-f08-photo.jpg` is listed in `labels/images.csv` but is absent from physical storage.
- **Implementation Handling:** The codebase queries actual file existence on disk (`Path.exists()`) rather than indexing `images.csv` blindly.

---

## 2. Ground Truth Labels & Mark Conventions

Ground truth annotations are stored in `labels/labels.csv` (139 columns) and individual per-form JSON files in `labels/json/<form_id>.json`.

> **Critical Note:** The dataset does **not** contain bounding box coordinates for individual bubbles. Bubble locations are computed geometrically via canonical template projection (`template.json`).

### 2.1 Question Cell Symbol Conventions (`q1` to `q120`)
Across 740 forms $\times$ 120 questions = 88,800 question cells:
- **`A`, `B`, `C`, `D`, `E` (48.0% / 42,610 cells):** Unambiguous single filled bubble.
- **`-` (49.6% / 44,078 cells):** Unanswered / blank question.
- **`?` (2.4% / 2,112 cells):** Non-standard marking (tick, cross, circle around bubble letter, or faint smudge).
- **`changed` (column):** Corrected marks (e.g., `q11:E>C` indicates option E was erased or crossed out and replaced by C).
- **`sloppy` (column):** Irregular annotations (e.g., `q12:circle B; q21:multiple CD`).

---

## 3. Person-Based Train, Validation, and Test Splitting

To ensure true generalization across different handwriting styles, pencil grades, and marking habits, data splits are strictly **person-isolated**:

- **Total Contributors:** 74 persons.
- **Train Split:** 44 contributors (~59.5% of dataset).
- **Validation Split:** 15 contributors (~20.3% of dataset).
- **Test Split:** 15 contributors (~20.3% of dataset).

### 3.1 Strict Stress Test Isolation Rule
The evaluation suite in `DATA_ROOT/stress_test/` consists of 90 raw, unscaled images:
- Contributors: `p03` (30 images), `p04` (30 images), `p19` (30 images).
- Conditions per contributor: 10 `scan`, 10 `photo`, 10 `hard`.

**Data Leakage Prevention Mandate:**
- Contributors `p03`, `p04`, and `p19` are permanently excluded from all training and validation patch extraction (`EXCLUDED_PERSONS = ["p03", "p04", "p19"]` in `config.py`).
- Models must never be trained or hyperparameter-tuned on images from these three contributors.

---

## 4. Personal Data Privacy Protocol (`roster.csv`)

`DATA_ROOT/vision_ds/roster.csv` maps contributor IDs (`p01` through `p74`) to real student names, roll numbers, and examination metadata.

### Non-Negotiable Privacy Rules
1. **Never Commit:** `roster.csv` is explicitly excluded in `.gitignore` and must never be committed to git.
2. **Never Log:** Terminal logs, training scripts, and automated test runners must never print real student names.
3. **Never Export:** Evaluation reports, JSON exports, and failure manifests must reference only anonymous IDs (e.g., `p04-f01`, `Student ID 17385347`) and never student names.
4. **Isolate in Code:** In `src/handwriting_ocr.py`, `roster.csv` is accessed only for local mathematical ID validation. If an extracted student ID matches a roster entry, confirmation flags are returned without exposing student identity strings in application responses.
