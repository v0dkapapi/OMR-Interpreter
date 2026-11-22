# Known Technical Issues & Engineering Roadmap

This document catalogs identified technical limitations, architectural redundancies, and concrete planned improvements across the OMR Interpreter codebase.

---

## 1. Known Technical Issues & Code Inconsistencies

### 1.1 Duplicated Decision Fusion Logic (RESOLVED)
- **Resolution:** Fusion now lives in `src/fusion.py` (`BubbleReader`, `fuse_question`). `src/app.py`, `src/evaluate.py` and `src/batch_grade.py` all call it; `evaluate.py` also runs each sheet through `OMRGrader` and reports a REVIEW rate. The old disagreement check in `app.py` was unreachable (it required fill $< 0.20$ inside a branch requiring fill $\ge 0.26$); disagreements are now recorded for the CNN-only acceptance branch and routed to REVIEW. The fusion rules were then retuned on validation contributors (Experiment 4): stress test accuracy 93.28%.
- **Locations:** `src/evaluate.py` (lines 284–301) vs `src/app.py` (lines 336–350).
- **Issue:** Decision fusion logic is implemented in two distinct places with slight behavioral differences:
  - `src/evaluate.py` uses vectorized tensor batching across all 120 questions at once and checks answers directly against ground truth labels without calling `OMRGrader`.
  - `src/app.py` processes questions individually and routes disagreements (e.g. CNN says `unfilled` but fill ratio $> 0.38$) to `OMRGrader.grade_sheet()`, producing `REVIEW` status flags.
- **Impact:** Benchmark evaluation script does not emit `REVIEW` telemetry or run through the exact same grading code path as the live API.
- **Remediation:** Extract decision fusion into a shared module (`src/fusion.py`) imported by both `evaluate.py` and `app.py`.

### 1.2 Sandbox Cache Restrictions in EasyOCR (RESOLVED)
- **Resolution:** EasyOCR weights are stored in `models/ocr_cache/easyocr/` (git-ignored). Copy `craft_mlt_25k.pth` and `english_g2.pth` there to pre-seed offline runners and set `OMR_OFFLINE=1` to forbid downloads.
- **Location:** `src/handwriting_ocr.py` (lines 166–172).
- **Issue:** When EasyOCR is invoked for the first time in a restricted environment, it attempts to download detection and recognition model weights to `~/.EasyOCR/model/`. In sandboxed or offline runners, this network call or directory write can fail if not pre-cached.
- **Remediation:** Provide pre-downloaded model weight caching in `models/ocr_cache/` or configure EasyOCR model directory to point strictly inside the project root.

### 1.3 Static Coordinates in `template.json` Under Skew
- **Location:** `template.json`.
- **Issue:** Bubble bounding boxes are defined at fixed pixel coordinates assuming a perfectly planar homography. On mobile photos with perspective curl (e.g., paper bent away from camera), outer columns (Questions 91–120) experience 2–5 px alignment drift.
- **Remediation:** Implement sub-block local affine refinement or fiducial cross-alignment across the 4 question column headers.

### 1.4 Dataset Inventory Discrepancy in `p10` (MITIGATED)
- **Status:** `src/dataset.py` builds its patch cache from `labels.csv` and checks `Path.exists()` for every image (with a `.jpeg` fallback), so the missing `p10-f08-photo.jpg` is skipped rather than crashing. `images.csv` is not read by any loader.
- **Location:** `vision_ds/labels/images.csv`.
- **Issue:** `p10-f07-photo.jpg` is present on disk but absent from `images.csv`, whereas `p10-f08-photo.jpg` is recorded in `images.csv` but absent from disk.
- **Remediation:** Ensure dataset loaders verify physical disk existence via `Path.exists()` rather than relying solely on CSV indices.

### 1.5 Answered Questions Without a Key Entry Reported as CORRECT (RESOLVED)
- **Location:** `src/grader.py`.
- **Issue:** A marked question with no answer key entry was given status `CORRECT` (0 marks), so a partial custom key showed every other answered question as correct.
- **Resolution:** New `NOT_GRADED` status with the `NO_ANSWER_KEY_ENTRY` flag.

### 1.6 False Roster Flag Without Local roster.csv (RESOLVED)
- **Location:** `src/handwriting_ocr.py` `cross_check_roster`.
- **Issue:** When `roster.csv` was absent, every contributor was flagged `ROSTER_CONTRIBUTOR_NOT_FOUND`, and the roster unit test failed on any machine without the private file.
- **Resolution:** The contributor check only runs when the roster was loaded; the roster test skips otherwise.

### 1.7 Patch Extraction Collapsed to One Contributor (RESOLVED)
- **Location:** `src/dataset.py` `build_cache`.
- **Issue:** `max_forms` took the first N rows of `labels.csv` (sorted by contributor), so small budgets trained and validated on one person each.
- **Resolution:** Round-robin selection across contributors.

### 1.8 Invalid-Answer Labels Mix Multi-Marks and Sloppy Marks
- **Location:** `vision_ds/labels/labels.csv`, consumed by `src/evaluate.py`.
- **Issue:** Every invalid answer is labelled `?`; the `sloppy` column distinguishes `multiple` from `circle`, `tick`, `cross` and `faint`. "Multi-mark" metrics currently measure all invalid answers together.
- **Remediation:** Parse the `sloppy` column in `evaluate.py` to report true multiple-mark precision and recall separately.

### 1.10 Handwritten Field OCR (IMPROVED, Experiment 5)
- **Status:** Held-out exact match: name 66.7%, module code 61.4%, date 54.4%, seat 52.6% (was 19-37%). Remaining errors are mostly single-character digit confusions, years misread as other plausible years, and very faint pencil.
- **Next steps:** Fine-tune TrOCR on labelled header crops from the training split (labels exist for every form); decode the student-number bubble grid, which should be far more reliable than reading the written digits (15.8%).

### 1.9 Remaining Error Profile (Run 4)
- `p19` (strong lighting gradients) is the weakest contributor at 88.44%, and errors still increase in question columns 3 and 4 (see 1.3). Column shadow equalization and sub-block alignment remain the most promising next steps.

---

## 2. Prioritized Engineering Roadmap

### Priority 1: High Impact & Core Stability
1. **Unify Decision Fusion Module (DONE):**
   - Create `src/fusion.py` containing standardized fusion rules, fill-ratio thresholds, and review classification logic.
   - Refactor both `src/evaluate.py` and `src/app.py` to use `src/fusion.py`.
2. **Sub-Block Perspective Refinement:**
   - Add local corner detection or feature matching on the 4 column boundary lines to eliminate the 2–5 px shift on tilted phone captures.
3. **Automated Synthetic Sheet Generation in Tests (DONE, `tests/synthetic.py`):**
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
2. **Batch Processing CLI & Progress Telemetry (DONE, `src/batch_grade.py`):**
   - Add a CLI utility to grade batches of hundreds of examination sheets with multi-core parallelism and progress reporting.
