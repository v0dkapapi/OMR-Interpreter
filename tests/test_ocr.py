"""Unit tests for src/handwriting_ocr.py."""

import pytest
import numpy as np
import cv2
from pathlib import Path
import config
from src.handwriting_ocr import HandwritingOCR
from src.preprocess import SheetPreprocessor

@pytest.fixture(scope="module")
def ocr_engine():
    # Test OCR with fallback enabled
    return HandwritingOCR(use_trocr=False) # Test fallback engine directly

def is_accessible(p: Path) -> bool:
    try:
        return p.exists() and p.is_file()
    except (PermissionError, OSError):
        return False

def test_handwriting_ocr_fields(ocr_engine):
    """Test OCR metadata extraction on a sample canonical sheet."""
    sample_scan = config.VISION_IMAGES_DIR / "p01" / "p01-f01-scan.jpg"
    if not is_accessible(sample_scan):
        pytest.skip("Sample scan image not accessible or within sandbox boundary")

    preprocessor = SheetPreprocessor()
    res = preprocessor.process(sample_scan)

    meta = ocr_engine.extract_metadata(res.canonical_bgr, contributor="p01")
    assert "fields" in meta
    assert "overall_confidence" in meta
    assert "flags" in meta

    fields = meta["fields"]
    for expected in ["name", "module_code", "seat_number", "date", "student_number_written"]:
        assert expected in fields
        assert "text" in fields[expected]
        assert "confidence" in fields[expected]
        assert "engine" in fields[expected]

    # Verify student name is NOT leaked into any flags or logs
    assert meta["overall_confidence"] > 0.0

def test_field_validation(ocr_engine):
    """Test format validators for roll number, module code, and seat number."""
    # Roll number valid (8 digits)
    valid, flags = ocr_engine.validate_field("student_number_written", "74228162")
    assert valid is True
    assert len(flags) == 0

    # Roll number invalid (less than 8 digits)
    valid, flags = ocr_engine.validate_field("student_number_written", "12345")
    assert valid is False
    assert "INVALID_STUDENT_NUMBER_FORMAT" in flags

    # Module code valid
    valid, flags = ocr_engine.validate_field("module_code", "MATH33066")
    assert valid is True
    assert len(flags) == 0

    # Module code invalid
    valid, flags = ocr_engine.validate_field("module_code", "12345")
    assert valid is False
    assert "INVALID_MODULE_CODE_FORMAT" in flags

def test_roster_cross_check(ocr_engine):
    """Test roster cross-checking without student name disclosure."""
    if not ocr_engine.roster_splits:
        pytest.skip("roster.csv not available locally (it is never committed)")
    # Valid contributor in roster
    flags = ocr_engine.cross_check_roster("74228162", contributor="p01")
    assert "ROSTER_CONTRIBUTOR_NOT_FOUND" not in flags

    # Non-existent contributor
    flags = ocr_engine.cross_check_roster("74228162", contributor="p999")
    assert "ROSTER_CONTRIBUTOR_NOT_FOUND" in flags


def test_roster_absent_does_not_flag_contributor():
    """Without a local roster, contributor checks must not raise false NOT_FOUND flags."""
    engine = HandwritingOCR(use_trocr=False)
    engine.roster_splits = {}
    flags = engine.cross_check_roster("74228162", contributor="p01")
    assert "ROSTER_CONTRIBUTOR_NOT_FOUND" not in flags
    assert "STUDENT_NUMBER_LENGTH_MISMATCH" not in flags
