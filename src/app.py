"""FastAPI backend application for OMR Interpreter.

Endpoints:
- POST /api/grade: Accepts OMR sheet image and optional answer key (JSON/CSV)
- GET /api/results/{id}: Returns cached grading results
- GET /api/export/{id}.csv: Returns CSV export of results
- GET /api/export/{id}.json: Returns JSON export of results
- GET /api/health: Service health and environment status
- Serves static frontend from /
"""

import os
import sys
import csv
import json
import uuid
import base64
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Ensure MPS fallback before torch import
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config
from src.preprocess import SheetPreprocessor, PreprocessResult
from src.fusion import BubbleReader, load_bubble_model
from src.handwriting_ocr import HandwritingOCR
from src.grader import OMRGrader, ScoringConfig, GradeResult, QuestionStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("omr.app")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_components()
    yield

app = FastAPI(
    title="OMR Interpreter API",
    version="1.0.0",
    description="High-accuracy OMR Answer Sheet Interpreter & Grading Backend",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory results cache: result_id -> GradeResult
RESULTS_CACHE: Dict[str, GradeResult] = {}
OVERLAY_IMAGE_CACHE: Dict[str, str] = {} # result_id -> base64 data URI

# Global components (lazy loaded on startup)
preprocessor: Optional[SheetPreprocessor] = None
ocr_engine: Optional[HandwritingOCR] = None
bubble_reader: Optional[BubbleReader] = None
grader: Optional[OMRGrader] = None
template_data: Dict[str, Any] = {}
default_answer_key: Dict[str, str] = {}


def load_components():
    """Initialize models and configurations."""
    global preprocessor, ocr_engine, bubble_reader, grader, template_data, default_answer_key

    config.ensure_directories()

    # Load template.json
    if config.TEMPLATE_PATH.exists():
        with open(config.TEMPLATE_PATH, "r", encoding="utf-8") as f:
            template_data = json.load(f)

    # Load default answer_key.json
    if config.DEFAULT_ANSWER_KEY_PATH.exists():
        with open(config.DEFAULT_ANSWER_KEY_PATH, "r", encoding="utf-8") as f:
            ak_data = json.load(f)
            default_answer_key = ak_data.get("answers", {})

    preprocessor = SheetPreprocessor(template_path=config.TEMPLATE_PATH)
    ocr_engine = HandwritingOCR()
    grader = OMRGrader()

    # Load MobileNetV3 bubble classifier (fill-margin-only fallback if weights are absent)
    device = config.get_device()
    model = None
    try:
        model = load_bubble_model(device)
    except Exception as e:
        logger.warning(f"Could not load bubble model weights: {e}. Using classical fill-ratio baseline only.")
    if model is None:
        logger.warning("Bubble model weights not available. Running in fill-margin-only mode (no multi-mark detection).")
    bubble_reader = BubbleReader(template_data.get("bubble_rois", {}), model=model, device=device)
    logger.info(f"Bubble reader initialized in {bubble_reader.mode} mode.")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    device = config.get_device()
    return {
        "status": "ok",
        "device": str(device),
        "mps_built": torch.backends.mps.is_built(),
        "models": {
            "bubble_classifier": "MobileNetV3-Small",
            "weights_present": config.TRAIN_CONFIG["final_model_path"].exists(),
            "trocr_primary": "microsoft/trocr-base-handwritten",
            "easyocr_fallback": True,
            "bubble_reader_mode": bubble_reader.mode if bubble_reader else "uninitialized",
        },
        "template_questions": len(template_data.get("bubble_rois", {})),
        "default_key_entries": len(default_answer_key),
    }


def parse_uploaded_answer_key(file_bytes: bytes, filename: str) -> Dict[str, str]:
    """Parse custom answer key from JSON or CSV file."""
    text = file_bytes.decode("utf-8", errors="ignore").strip()
    result = {}

    if filename.endswith(".json") or text.startswith("{"):
        try:
            data = json.loads(text)
            if "answers" in data and isinstance(data["answers"], dict):
                return {str(k): str(v).strip().upper() for k, v in data["answers"].items()}
            return {str(k): str(v).strip().upper() for k, v in data.items() if str(k).isdigit()}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON answer key: {e}")

    # Parse as CSV
    try:
        reader = csv.reader(text.splitlines())
        for row in reader:
            if len(row) >= 2:
                q_clean = "".join(filter(str.isdigit, row[0]))
                if q_clean:
                    ans = row[1].strip().upper()
                    if ans in "ABCDE":
                        result[q_clean] = ans
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV answer key: {e}")


# Overlay colours (BGR)
OVERLAY_GREEN = (74, 163, 22)    # #16A34A: correct answer
OVERLAY_RED = (38, 38, 220)      # #DC2626: chosen answer that is wrong or invalid
OVERLAY_GRAY = (128, 128, 128)   # chosen answer that could not be graded (no key entry)
OVERLAY_FAINT = (190, 190, 190)  # every other bubble
OVERLAY_FILL_ALPHA = 0.55        # translucent fills keep the student's mark visible


def render_overlay(
    canonical_bgr: np.ndarray,
    grade_result: GradeResult,
    template: Dict[str, Any]
) -> np.ndarray:
    """Draw the marking overlay on the canonical sheet.

    - Correct answer chosen: the chosen bubble is green.
    - Wrong answer chosen: the chosen bubble is red and the correct answer green.
    - Multiple marks: each chosen bubble is red; the correct answer is green
      (an outer green ring if the student also marked it).
    - Unanswered or under review: the correct answer gets a green outline only,
      so it is not mistaken for a mark the student made.
    """
    bubble_rois = template.get("bubble_rois", {})
    fills = canonical_bgr.copy()
    rings: List[tuple] = []   # (center, radius, colour, thickness)

    for q in grade_result.questions:
        opts = bubble_rois.get(str(q.question))
        if not opts:
            continue
        selected = set(q.selected_options)
        expected = q.expected_answer if q.expected_answer in opts else None
        status = q.status

        for opt, roi in opts.items():
            center = (int(roi["x"] + roi["w"] / 2), int(roi["y"] + roi["h"] / 2))
            r = int(min(roi["w"], roi["h"]) / 2)

            if opt in selected:
                if status == QuestionStatus.CORRECT.value:
                    colour = OVERLAY_GREEN
                elif status == QuestionStatus.NOT_GRADED.value:
                    colour = OVERLAY_GRAY
                else:  # INCORRECT, MULTIPLE_MARK, REVIEW
                    colour = OVERLAY_RED
                cv2.circle(fills, center, r - 1, colour, -1)
                rings.append((center, r + 2, colour, 3))
                if opt == expected and status != QuestionStatus.CORRECT.value:
                    rings.append((center, r + 6, OVERLAY_GREEN, 3))
            elif opt == expected:
                if status in (QuestionStatus.INCORRECT.value, QuestionStatus.MULTIPLE_MARK.value):
                    cv2.circle(fills, center, r - 1, OVERLAY_GREEN, -1)
                rings.append((center, r + 2, OVERLAY_GREEN, 3))
            else:
                rings.append((center, r, OVERLAY_FAINT, 1))

    overlay = cv2.addWeighted(fills, OVERLAY_FILL_ALPHA, canonical_bgr, 1.0 - OVERLAY_FILL_ALPHA, 0)
    for center, radius, colour, thickness in rings:
        cv2.circle(overlay, center, radius, colour, thickness)
    return overlay


def generate_overlay_preview(
    canonical_bgr: np.ndarray,
    grade_result: GradeResult,
    template: Dict[str, Any]
) -> str:
    """Render the marking overlay and return it as a JPEG data URI."""
    overlay = render_overlay(canonical_bgr, grade_result, template)
    success, buffer = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


@app.post("/api/grade")
async def grade_sheet(
    image: UploadFile = File(...),
    answer_key_file: Optional[UploadFile] = File(None),
    answer_key_json: Optional[str] = Form(None),
    marks_per_correct: float = Form(1.0),
    negative_marks: float = Form(0.0),
    multi_mark_marks: float = Form(0.0)
):
    """Process and grade an uploaded OMR answer sheet."""
    global preprocessor, ocr_engine, bubble_reader, grader, template_data, default_answer_key
    if preprocessor is None or ocr_engine is None or bubble_reader is None or grader is None:
        load_components()

    # 1. Validate file extension and MIME type
    allowed_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    ext = Path(image.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(allowed_exts)}"
        )

    # 2. Enforce file size limit (25 MB)
    max_size = 25 * 1024 * 1024
    image_bytes = await image.read()
    if len(image_bytes) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of 25MB (Received {len(image_bytes)/(1024*1024):.1f}MB)"
        )

    # 3. Determine Answer Key
    target_answer_key = dict(default_answer_key)
    if answer_key_file is not None and answer_key_file.filename:
        ak_bytes = await answer_key_file.read()
        target_answer_key = parse_uploaded_answer_key(ak_bytes, answer_key_file.filename)
    elif answer_key_json and answer_key_json.strip():
        try:
            parsed_json = json.loads(answer_key_json)
            if "answers" in parsed_json:
                target_answer_key = {str(k): str(v).upper() for k, v in parsed_json["answers"].items()}
            else:
                target_answer_key = {str(k): str(v).upper() for k, v in parsed_json.items()}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid answer_key_json: {e}")

    # 4. Safe temporary file storage with guaranteed cleanup
    temp_dir = tempfile.mkdtemp(prefix="omr_upload_")
    temp_image_path = Path(temp_dir) / f"upload{ext}"

    try:
        with open(temp_image_path, "wb") as f:
            f.write(image_bytes)

        # 5. Preprocessing & Alignment
        res: PreprocessResult = preprocessor.process(temp_image_path)
        canonical_bgr = res.canonical_bgr
        norm_gray = res.normalized_gray

        # 6. Bubble Classification and decision fusion (shared with src/evaluate.py)
        bubble_rois = template_data.get("bubble_rois", {})
        question_predictions = bubble_reader.read(norm_gray)

        # 7. Handwritten Student Metadata Extraction
        student_details = {}
        if ocr_engine is not None:
            student_details = ocr_engine.extract_metadata(
                canonical_bgr, template=template_data
            )
        else:
            logger.warning("OCR engine not initialized; skipping metadata extraction.")

        # 8. Deterministic Grading
        scoring_cfg = ScoringConfig(
            marks_per_correct=marks_per_correct,
            negative_marks=negative_marks,
            multi_mark_marks=multi_mark_marks,
            total_questions=len(bubble_rois)
        )

        grade_result = grader.grade(
            question_predictions=question_predictions,
            answer_key=target_answer_key,
            scoring_config=scoring_cfg,
            student_details=student_details,
            alignment_confidence=res.confidence,
            alignment_flags=res.flags
        )

        # 9. Generate Annotated Overlay Image
        overlay_b64 = generate_overlay_preview(canonical_bgr, grade_result, template_data)

        # 10. Cache result for export endpoints
        result_id = str(uuid.uuid4())
        RESULTS_CACHE[result_id] = grade_result
        OVERLAY_IMAGE_CACHE[result_id] = overlay_b64

        response_dict = grade_result.to_dict()
        response_dict["result_id"] = result_id
        response_dict["overlay_preview"] = overlay_b64

        return JSONResponse(status_code=200, content=response_dict)

    finally:
        # Guaranteed cleanup of uploaded temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/api/results/{result_id}")
async def get_result(result_id: str):
    """Retrieve cached grading results by result_id."""
    if result_id not in RESULTS_CACHE:
        raise HTTPException(status_code=404, detail="Result not found or expired.")
    res = RESULTS_CACHE[result_id]
    data = res.to_dict()
    data["result_id"] = result_id
    data["overlay_preview"] = OVERLAY_IMAGE_CACHE.get(result_id, "")
    return JSONResponse(status_code=200, content=data)


@app.get("/api/export/{result_id}.csv")
async def export_csv(result_id: str):
    """Export graded results as downloadable CSV."""
    if result_id not in RESULTS_CACHE:
        raise HTTPException(status_code=404, detail="Result not found.")
    res = RESULTS_CACHE[result_id]
    csv_content = res.to_csv()
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="omr_report_{result_id[:8]}.csv"'}
    )


@app.get("/api/export/{result_id}.json")
async def export_json(result_id: str):
    """Export graded results as downloadable JSON."""
    if result_id not in RESULTS_CACHE:
        raise HTTPException(status_code=404, detail="Result not found.")
    res = RESULTS_CACHE[result_id]
    json_content = res.to_json(indent=2)
    return Response(
        content=json_content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="omr_report_{result_id[:8]}.json"'}
    )


# Serve Static Frontend Files
FRONTEND_PATH = config.FRONTEND_DIR
if FRONTEND_PATH.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_PATH)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(
            str(FRONTEND_PATH / "index.html"),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
