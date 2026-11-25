import os
import sys
from pathlib import Path

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# Set PyTorch MPS fallback before any torch import
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# External Dataset Root (Configurable via environment variable OMR_DATA_ROOT)
# Defaults to ../omr-data/data relative to PROJECT_ROOT for team portability
_DEFAULT_DATA_ROOT = str((PROJECT_ROOT.parent / "omr-data" / "data").resolve())
DATA_ROOT = Path(os.getenv("OMR_DATA_ROOT", _DEFAULT_DATA_ROOT)).resolve()

# Dataset subdirectories
VISION_DS_DIR = DATA_ROOT / "vision_ds"
VISION_IMAGES_DIR = VISION_DS_DIR / "images"
VISION_LABELS_DIR = VISION_DS_DIR / "labels"
VISION_ROSTER_PATH = VISION_DS_DIR / "roster.csv"
STRESS_TEST_DIR = DATA_ROOT / "stress_test"

# Internal Project Directories (all git-ignored for runtime outputs)
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "cache"
PATCH_CACHE_DIR = CACHE_DIR / "patches"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FAILURES_DIR = OUTPUTS_DIR / "failures"
TEMP_UPLOADS_DIR = PROJECT_ROOT / "temp_uploads"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Templates & Keys
TEMPLATE_PATH = PROJECT_ROOT / "template.json"
DEFAULT_ANSWER_KEY_PATH = PROJECT_ROOT / "answer_key.json"

# Leakage Prevention: Stress test contributors MUST be strictly excluded from train & val
EXCLUDED_PERSONS = ["p03", "p04", "p19"]

# Canonical Sheet Dimensions for Preprocessing (A4 at ~200 DPI: 1654 x 2338)
CANONICAL_WIDTH = 1654
CANONICAL_HEIGHT = 2338

# Alignment Thresholds
ALIGNMENT_CONFIDENCE_THRESHOLD = 0.65  # Below this, flag LOW_ALIGNMENT_CONFIDENCE

# Bubble Classifier Settings
BUBBLE_CLASSES = ["unfilled", "filled", "ambiguous"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(BUBBLE_CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(BUBBLE_CLASSES)}
BUBBLE_PATCH_SIZE = (48, 48)  # Height x Width for CNN patch inputs

# Classical Fill-Ratio Baseline Settings
CLASSICAL_FILL_THRESHOLD = 0.38      # Above this ratio is considered filled
CLASSICAL_AMBIGUOUS_LOWER = 0.20    # Ambiguity lower bound
CLASSICAL_AMBIGUOUS_UPPER = 0.42    # Ambiguity upper bound

# Training Hyperparameters (Configurable for thermal discipline on fanless M5)
TRAIN_CONFIG = {
    "model_name": "mobilenet_v3_small",
    "epochs": int(os.getenv("OMR_EPOCHS", "5")),
    "batch_size": int(os.getenv("OMR_BATCH_SIZE", "64")),
    "subset_size": int(os.getenv("OMR_SUBSET_SIZE", "3000")),
    "learning_rate": float(os.getenv("OMR_LR", "1e-3")),
    "checkpoint_dir": MODELS_DIR / "checkpoints",
    "final_model_path": MODELS_DIR / "bubble_mobilenet_v3.pth",
}

# OCR Settings
OCR_CONFIG = {
    "trocr_model": "microsoft/trocr-base-handwritten",
    "model_cache_dir": MODELS_DIR / "ocr_cache",
    # EasyOCR detector/recognizer weights (pre-seed this folder for offline runners)
    "easyocr_model_dir": MODELS_DIR / "ocr_cache" / "easyocr",
    # Set OMR_OFFLINE=1 to forbid model downloads once weights are cached
    "offline": os.getenv("OMR_OFFLINE", "0") == "1",
    "roll_digits": 8,  # Student number digit count
    # Subject prefixes of S3231 module codes (all 740 labelled forms use these).
    # OCR'd prefixes are snapped to the nearest entry. Extend for other cohorts.
    "module_prefixes": ["ACM", "BIOL", "BMGT", "CHEM", "COMP", "ECON", "EEEN", "MATH", "PHYC", "STAT"],
    # Plausible exam years; dates outside this range are rejected as misreads
    # (handwritten "2024" is often read as 2004/2014/2016). All labelled dates
    # fall in 2019-2026; widen this for older or future cohorts.
    "date_year_range": (2019, 2030),
    # TrOCR beam search width; the n-best list is filtered by field validators.
    "trocr_num_beams": 5,
    # TrOCR fine-tuned on S3231 header crops (src/train_trocr.py). Used first
    # when present; the base model remains the fallback. Git-ignored.
    "trocr_finetuned_dir": MODELS_DIR / "trocr_finetuned",
}

# Scoring Policy Defaults
DEFAULT_SCORING = {
    "marks_per_correct": 1.0,
    "negative_marks": 0.0,
    "multi_mark_marks": 0.0,
    "unanswered_marks": 0.0,
    "total_questions": 120,
    "options_per_question": 5,  # A, B, C, D, E
}

def get_device():
    """Detect and return optimal PyTorch device (MPS for Apple Silicon, CPU fallback)."""
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")

def ensure_directories():
    """Ensure required runtime and cache directories exist."""
    for d in [MODELS_DIR, CACHE_DIR, PATCH_CACHE_DIR, OUTPUTS_DIR, FAILURES_DIR, TEMP_UPLOADS_DIR, TRAIN_CONFIG["checkpoint_dir"]]:
        d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Root: {DATA_ROOT} (exists: {DATA_ROOT.exists()})")
    print(f"Device: {get_device()}")
    ensure_directories()
    print("Runtime directories created and verified.")
