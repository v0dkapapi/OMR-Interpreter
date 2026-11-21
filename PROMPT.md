# OMR Interpreter: full build specification

Build a complete OMR Interpreter web application (Python backend, clean frontend dashboard) in this workspace. Work in phases and stop after each step for my confirmation.

## Environment
- Dataset lives OUTSIDE the workspace at DATA_ROOT (e.g. ../omr-data/data). Read it through a configurable DATA_ROOT in config.py. Never copy raw data into the workspace. Do not recursively index that folder beyond what the audit needs.
- A Python venv already exists at .venv (activate with: source .venv/bin/activate). Install packages into it.
- Target: MacBook Air, Apple Silicon (M5), fanless, so sustained loads throttle. Use PyTorch with device "mps" when available, else CPU, and set PYTORCH_ENABLE_MPS_FALLBACK=1. Use opencv-python-headless. Prefer EasyOCR over PaddleOCR. No CUDA-only dependencies.
- Do not train on the full 5.1 GB at full resolution. Resize on load, cache downscaled copies in cache/ (git-ignored), and run a small subset first. Make epochs, subset size and batch size configurable, save a checkpoint each epoch so training can resume.

## Data
DATA_ROOT/vision_ds/
- images/: about 74 person folders (p01, p02, ...), 2,184 files, 5.1 GB
- labels/: images.csv, labels.csv, json/ (742 files). Added by a different contributor, so naming or format may differ from images/.
- README.md: authoritative dataset description. READ FIRST.
- roster.csv: personal data. Keep local, never print student names, never log, never commit, never send anywhere.

DATA_ROOT/stress_test/ (p03, p04, p19)
- Names like pXX-fYY-{scan|photo|hardphoto|hard}.{jpg|jpeg}, plus extras such as pXX-01-scan.jpg and p19-04-hard.jpg.
- Likely: scan = flatbed scan, photo = phone photo, hard/hardphoto = difficult phone photo (shadow, skew, perspective, blur). fYY = form number. VERIFY against README.md and the images.
- Naming is inconsistent. Parse with a tolerant case-insensitive regex, normalize hard/hardphoto to "hard", accept .jpg/.jpeg, and report non-matching files.
- Use stress_test ONLY for evaluation, never for training or validation.

## UI requirements
- Strict white background (#FFFFFF) with red shades (#7F1D1D, #991B1B, #B91C1C, #DC2626, #FEE2E2, #FEF2F2), neutral borders (#E5E7EB), dark text (#111827 / #7F1D1D).
- NO emojis anywhere in the frontend (labels, headings, statuses, buttons, tables, placeholders).
- Responsive, accessible, visible focus states, keyboard-usable dropzone.
- Features: upload dropzone; answer key selector (default or upload JSON/CSV); extracted handwritten student details with confidence indicators (text bars or percentages, no icons); score summary (marks obtained, total, percentage, answered, unanswered, multi-mark count); question-by-question table (selected option(s), expected answer, status, marks awarded); flagged issues panel; CSV and JSON export buttons; image preview with detected bubbles overlaid (red = flagged, neutral = ok); loading and error states.

## Pipeline requirements
1. Template: template.json with page size, alignment anchors, bubble ROIs (question, option, x, y, w, h), handwritten ROIs (student name, roll number, exam ID), and scoring config (marks per correct, negative marking, multi-mark policy). Derive it from the real sheets. Provide a helper script that overlays the template on a sample sheet for visual checking.
2. src/preprocess.py: handle scan, photo and hard conditions. Perspective correction (page contour or fiducials, with feature-matching/ECC fallback against a reference sheet), deskew, shadow normalization (CLAHE or background division), denoise, adaptive threshold. Return an alignment quality score; below threshold, flag LOW_ALIGNMENT_CONFIDENCE rather than producing silent wrong answers.
3. src/train_bubble_model.py: MobileNetV3-small or ResNet-18 (pretrained, fine-tuned) on cropped bubble patches. Classes: unfilled, filled, ambiguous. Augment with rotation, brightness/contrast, noise, blur, small perspective jitter, shadow gradients. Split train/val by PERSON. Exclude stress_test people (also detect them inside vision_ds/images by filename or hash and exclude). Handle class imbalance. Save weights, metrics report, confusion matrix. Also implement a classical fill-ratio baseline as fallback and sanity check, and flag disagreements.
4. src/handwriting_ocr.py: primary TrOCR (microsoft/trocr-base-handwritten) on MPS, fallback EasyOCR. Return text plus confidence per field. Validate fields (roll number digits only, exam ID pattern). Cross-check against roster.csv locally and flag mismatches without altering the extracted value. Cache model downloads; fail gracefully offline.
5. src/grader.py: one option filled and correct = award marks; one filled and wrong = configured negative or zero marking; 2+ filled = MULTIPLE_MARK, scored per rule (default 0), never attributed to one answer, never crashes; none filled = UNANSWERED; ambiguous = REVIEW and listed in flagged issues. Deterministic, with unit tests for every case.
6. src/app.py (FastAPI): POST /api/grade (image plus optional answer key), GET /api/results/{id}, GET /api/export/{id}.csv and .json, GET /api/health. Validate file type and size. Store uploads only in a temp folder with cleanup. Serve the frontend from the same server.
7. src/evaluate.py: run the full pipeline on stress_test, report accuracy per condition (scan/photo/hard) and per person, save failure cases under outputs/failures/. Where labels exist, compute answer-level accuracy and multi-mark precision and recall.

## Execution order (stop after each step, run checks, summarize, wait for me to say "continue")
- Step 0: Dataset audit. Read README.md, inspect roster.csv headers only, parse labels/images.csv and labels.csv, check labels/json/ schema. Verify the hypothesis of about 3 images per label entry. Report images per person, resolutions, label coverage, orphans, duplicates, and what labels actually contain (answers, bubble boxes, student details). Write DATASET_AUDIT.md with no student names. If labels lack bubble-level annotations, derive patches from template ROIs and document it. State assumptions explicitly.
- Step 1: Project structure, config.py with DATA_ROOT, requirements.txt, .gitignore (models/, outputs/, cache/, roster.csv, .venv/), default template.json, sample answer_key.json. Install requirements into .venv.
- Step 2: src/preprocess.py with a visual check on one scan, one photo and one hard image.
- Step 3: dataset loader, src/train_bubble_model.py, src/handwriting_ocr.py. Run a small subset first.
- Step 4: src/grader.py with unit tests.
- Step 5: src/app.py and frontend/index.html, styles.css, app.js.
- Step 6: src/evaluate.py and a README with exact commands (activate venv, train, evaluate, run uvicorn, test with sample sheets).

Finish with known limitations and suggested improvements.
