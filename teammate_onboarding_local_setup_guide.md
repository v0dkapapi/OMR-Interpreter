# OMR Interpreter: Teammate Onboarding & Local Setup Guide

This guide covers cloning the repository, configuring your local Python environment, verifying the test suite, running the web dashboard, and initializing your AI coding assistant with the project's guardrails.

---

## 1. Clone the Repository

```bash
git clone https://github.com/Dhyanii/omr-interpreter.git
cd omr-interpreter
```

---

## 2. Set Up Virtual Environment & Dependencies

Python 3.12 is recommended (Python 3.10+ compatible):

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install exact pinned dependencies
pip install -r requirements.txt
```

*(On Windows PowerShell, activate the environment using `.venv\Scripts\Activate.ps1`.)*

---

## 3. Configure Local Environment

Copy the environment template to create your local `.env` file:

```bash
cp .env.example .env
```

- **With local dataset access:** Download the dataset folder from Google Drive (it arrives as several zip parts) and merge it:
  ```bash
  python scripts/setup_dataset.py ~/Downloads/AI-for-Vision-Dataset-*.zip
  ```
  This builds `../omr-data/data/vision_ds/` and a `stress_test/` folder of symlinks for `p03`, `p04` and `p19`. To use a different location, pass `--data-root /path/to/omr-data/data` and set `OMR_DATA_ROOT` to the same path in `.env`.
- **Without local dataset access:** You do not need the dataset to run unit tests or start the web application—the default test suite runs against synthetic sheets generated from the template.

---

## 4. Verify the Installation

Run the automated test suite to confirm your environment is configured properly:

```bash
pytest tests/ -v
```

All tests should pass with 0 failures: 45 passed with the dataset in place, or 42 passed and 3 skipped without it (two tests need sample sheets under `OMR_DATA_ROOT` and one needs the local `roster.csv`). Integration tests use synthetic sheets generated from `template.json`, so they run anywhere.

To reproduce the benchmark, train the classifier and evaluate:

```bash
python src/train_bubble_model.py --epochs 5 --subset 20000 --max-forms 90
python src/evaluate.py --no-ocr
```

Expect about 93% answer-level accuracy (Run 4 in `docs/EVALUATION_PROTOCOL.md`).

---

## 5. Start the Web Dashboard

Launch the FastAPI server using Uvicorn:

```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser to: `http://127.0.0.1:8000/`

---

## 6. AI Assistant Onboarding Prompt

If you use an AI coding assistant (Claude Code, Cursor, Antigravity, Gemini CLI, or ChatGPT), paste the prompt below as your first message after opening the repository so it follows the project's privacy, grading, and evaluation rules:

```markdown
You are an expert AI/ML software engineer assisting me with the OMR Interpreter codebase.
This project is an Optical Mark Recognition (OMR) system and deterministic grading engine for standard UCD S3231 120-question multiple-choice examination forms.

### 1. Non-Negotiable System Guardrails & Rules
1. Strict Data Privacy: roster.csv contains real personal student details. Student names must NEVER be committed to git, logged in terminal output, printed in JSON reports, or exposed in the UI.
2. Deterministic Grading: Grading logic is strictly rule-based:
   - Single mark matching key = CORRECT (+marks).
   - Single mark differing from key = INCORRECT (-negative_marks or 0).
   - 2+ options filled = MULTIPLE_MARK (strictly 0 marks, never single-option attributed).
   - 0 options filled = UNANSWERED (0 marks).
   - Ambiguities & classifier disagreements = REVIEW (must never be silently guessed).
3. Zero Emojis: The codebase and UI strictly adhere to a pure white (#FFFFFF) and crimson red design system. Maintain zero emojis across all code, logs, API responses, and frontend text.
4. No Test Set Leakage: Contributors p03, p04, and p19 are strictly reserved for stress testing and must NEVER be used for training or validation patch extraction.

### 2. Key Source-of-Truth Files
- Configuration: config.py (reads OMR_DATA_ROOT from .env or defaults to ../omr-data/data)
- Preprocessing & Alignment: src/preprocess.py (SIFT homography to 1654x2338 canonical frame, Hough deskew, CLAHE)
- Bubble Mark Classifier: src/train_bubble_model.py (MobileNetV3-Small on 48x48 patches)
- Decision Fusion: src/fusion.py (single shared implementation used by the API, evaluation and batch CLI)
- Evaluation: src/evaluate.py (stress test benchmark over 90 sheets, reports REVIEW rate)
- Batch Grading CLI: src/batch_grade.py (multi-process folder grading)
- Deterministic Grader: src/grader.py (rule-based scoring)
- Handwriting OCR: src/handwriting_ocr.py (TrOCR primary, EasyOCR fallback)
- Web Application: src/app.py & frontend/ (FastAPI backend and pure HTML/JS dashboard)
- Documentation:
  - docs/ARCHITECTURE.md: Detailed pipeline phases and Mermaid diagrams
  - docs/DATASET_AND_PRIVACY.md: Dataset inventory, labels schema, and privacy rules
  - docs/EVALUATION_PROTOCOL.md: Blind evaluation benchmark standards (93.28% baseline, Run 4)
  - docs/EXPERIMENT_LOG.md: Experimental runs and metric history
  - docs/KNOWN_ISSUES_AND_ROADMAP.md: Known issues with line references and prioritized tasks

### 3. Local Environment Verification Steps
Before proposing or making any changes:
1. Verify the Python virtual environment (.venv) and dependencies from requirements.txt.
2. Verify config.py and .env paths.
3. Run the automated test suite with `pytest tests/ -v` to ensure all tests pass cleanly.
4. Check `git status` to ensure that model weights (*.pth), cache files, and outputs remain properly ignored.

Please acknowledge that you have read and understood these guardrails, and summarize your understanding of the architecture before proceeding.
```