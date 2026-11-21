# Known Technical Issues & Engineering Roadmap

This document catalogs identified technical limitations, architectural redundancies, and concrete planned improvements across the OMR Interpreter codebase.

---

## 1. Known Technical Issues & Code Inconsistencies

### 1.1 Duplicated Decision Fusion Logic
- **Locations:** `src/evaluate.py` (lines 284–301) vs `src/app.py` (lines 336–350).
- **Issue:** Decision fusion logic is implemented in two distinct places with slight behavioral differences:
  - `src/evaluate.py` uses vectorized tensor batching across all 120 questions at once and checks answers directly against ground truth labels without calling `OMRGrader`.
  - `src/app.py` processes questions individually and routes disagreements (e.g. CNN says `unfilled` but fill ratio $> 0.38$) to `OMRGrader.grade_sheet()`, producing `REVIEW` status flags.
- **Impact:** Benchmark evaluation script does not emit `REVIEW` telemetry or run through the exact same grading code path as the live API.
- **Remediation:** Extract decision fusion into a shared module (`src/fusion.py`) imported by both `evaluate.py` and `app.py`.

### 1.2 Sandbox Cache Restrictions in EasyOCR
- **Location:** `src/handwriting_ocr.py` (lines 166–172).
- **Issue:** When EasyOCR is invoked for the first time in a restricted environment, it attempts to download detection and recognition model weights to `~/.EasyOCR/model/`. In sandboxed or offline runners, this network call or directory write can fail if not pre-cached.
- **Remediation:** Provide pre-downloaded model weight caching in `models/ocr_cache/` or configure EasyOCR model directory to point strictly inside the project root.

### 1.3 Static Coordinates in `template.json` Under Skew
- **Location:** `template.json`.
- **Issue:** Bubble bounding boxes are defined at fixed pixel coordinates assuming a perfectly planar homography. On mobile photos with perspective curl (e.g., paper bent away from camera), outer columns (Questions 91–120) experience 2–5 px alignment drift.
- **Remediation:** Implement sub-block local affine refinement or fiducial cross-alignment across the 4 question column headers.

### 1.4 Dataset Inventory Discrepancy in `p10`
- **Location:** `vision_ds/labels/images.csv`.
- **Issue:** `p10-f07-photo.jpg` is present on disk but absent from `images.csv`, whereas `p10-f08-photo.jpg` is recorded in `images.csv` but absent from disk.
- **Remediation:** Ensure dataset loaders verify physical disk existence via `Path.exists()` rather than relying solely on CSV indices.

---

## 2. Prioritized Engineering Roadmap

### Priority 1: High Impact & Core Stability
1. **Unify Decision Fusion Module:**
   - Create `src/fusion.py` containing standardized fusion rules, fill-ratio thresholds, and review classification logic.
   - Refactor both `src/evaluate.py` and `src/app.py` to use `src/fusion.py`.
2. **Sub-Block Perspective Refinement:**
   - Add local corner detection or feature matching on the 4 column boundary lines to eliminate the 2–5 px shift on tilted phone captures.
3. **Automated Synthetic Sheet Generation in Tests:**
   - Add a pytest fixture that generates a minimal synthetic sheet from `template.json` to replace skipped integration tests when external dataset is absent.

### Priority 2: Accuracy & Robustness
1. **Segmented Character Classifier for Student Numbers:**
   - Replace open-vocabulary sequence-to-sequence OCR (TrOCR) on the 8-digit student number boxes with a constrained digit classifier trained on segmented digit crops.
2. **Synthetic Pencil Mark Augmentation:**
   - Add synthetic faint pencil strokes, eraser smudges, and low-contrast marks to the training patch pipeline to improve recall on `p03`-style sheets.
3. **Column Shadow Equalization:**
   - Refine morphological background division in `src/preprocess.py` using adaptive window sizes tailored to vertical column widths.

### Priority 3: User Experience & Proctor Workflow
1. **Interactive Human-in-the-Loop Review Dashboard:**
   - Enable exam proctors to click flagged `REVIEW` and `MULTIPLE_MARK` questions in the web preview canvas and toggle the resolved answer with single keystrokes.
2. **Batch Processing CLI & Progress Telemetry:**
   - Add a CLI utility to grade batches of hundreds of examination sheets with multi-core parallelism and progress reporting.
