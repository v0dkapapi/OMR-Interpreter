# Contributing to OMR Interpreter

Thank you for contributing to the OMR Interpreter project. This guide outlines development practices, testing standards, and non-negotiable data privacy rules.

---

## 1. Core Principles

1. **Deterministic Accuracy:** Grading logic must remain completely deterministic. Ambiguities and disagreements must be flagged for `REVIEW`, never guessed.
2. **Strict Privacy:** Personal student data (`roster.csv`) must never be committed, logged, or printed.
3. **No Overfitting:** Never tune thresholds or train models on reserved evaluation persons (`p03`, `p04`, `p19`).
4. **Clean Design System:** The user interface adheres strictly to a pure white (`#FFFFFF`) and crimson red palette (`#7F1D1D`, `#B91C1C`). Zero emojis are permitted across backend logs, API responses, or frontend components.

---

## 2. Development Setup

```bash
# Clone the repository
git clone <repository_url>
cd omr-interpreter

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env to set OMR_DATA_ROOT pointing to your local dataset directory
```

---

## 3. Testing Requirements

Before submitting any code changes, all automated tests must pass:

```bash
# Run the complete test suite
pytest tests/ -v
```

If your changes modify API endpoints or grading logic, ensure new unit tests are added under `tests/`.

---

## 4. Branching & Commit Guidelines

- **Main Branch:** `main` must always remain stable and deployable.
- **Feature Branches:** Use descriptive branch names:
  - `feature/<description>` for new capabilities
  - `fix/<description>` for bug fixes
  - `docs/<description>` for documentation updates
- **Commit Messages:** Use clear, imperative messages (e.g., `Refactor decision fusion into shared module`, `Add sub-block alignment check`).
- **No Binaries:** Never commit `.pth`, `.pt`, `.onnx` model weights or raw dataset images. Ensure `.gitignore` remains intact.

---

## 5. Pull Request Checklist

When opening a Pull Request, confirm that:
- [ ] `pytest tests/ -v` passes with 0 failures.
- [ ] No personal student names or raw dataset images are included in the diff.
- [ ] No hardcoded absolute local file paths are introduced.
- [ ] All new functions include type annotations and docstrings.
- [ ] Zero emojis are added to code, logs, or UI text.
