"""Unit tests for src/preprocess.py."""

import pytest
import numpy as np
import cv2
from pathlib import Path
import config
from src.preprocess import SheetPreprocessor, PreprocessResult

@pytest.fixture(scope="module")
def preprocessor():
    return SheetPreprocessor()

def is_accessible(p: Path) -> bool:
    try:
        return p.exists() and p.is_file()
    except (PermissionError, OSError):
        return False

def test_preprocess_dimensions(preprocessor):
    """Verify preprocessor outputs canonical dimensions."""
    sample_scan = config.VISION_IMAGES_DIR / "p01" / "p01-f01-scan.jpg"
    if not is_accessible(sample_scan):
        pytest.skip("Sample scan image not accessible or within sandbox boundary")

    res = preprocessor.process(sample_scan)
    assert isinstance(res, PreprocessResult)
    assert res.canonical_bgr.shape == (config.CANONICAL_HEIGHT, config.CANONICAL_WIDTH, 3)
    assert res.canonical_gray.shape == (config.CANONICAL_HEIGHT, config.CANONICAL_WIDTH)
    assert res.normalized_gray.shape == (config.CANONICAL_HEIGHT, config.CANONICAL_WIDTH)
    assert res.thresholded.shape == (config.CANONICAL_HEIGHT, config.CANONICAL_WIDTH)
    assert res.confidence >= config.ALIGNMENT_CONFIDENCE_THRESHOLD
    assert "LOW_ALIGNMENT_CONFIDENCE" not in res.flags

def test_preprocess_photo_and_hard(preprocessor):
    """Verify preprocessing on photo and hard conditions."""
    photo_path = config.VISION_IMAGES_DIR / "p01" / "p01-f01-photo.jpg"
    hard_path = config.VISION_IMAGES_DIR / "p01" / "p01-f01-hard.jpg"

    if is_accessible(photo_path):
        res_photo = preprocessor.process(photo_path)
        assert res_photo.confidence >= config.ALIGNMENT_CONFIDENCE_THRESHOLD
        assert res_photo.is_reliable

    if is_accessible(hard_path):
        res_hard = preprocessor.process(hard_path)
        assert res_hard.confidence >= config.ALIGNMENT_CONFIDENCE_THRESHOLD
        assert res_hard.is_reliable

def test_low_alignment_confidence_flag(preprocessor):
    """Verify that a non-OMR image correctly flags LOW_ALIGNMENT_CONFIDENCE."""
    # Create random noise image (not a sheet)
    noise_img = np.random.randint(0, 256, (800, 600, 3), dtype=np.uint8)
    res = preprocessor.process(noise_img)

    assert "LOW_ALIGNMENT_CONFIDENCE" in res.flags
    assert res.confidence < config.ALIGNMENT_CONFIDENCE_THRESHOLD
    assert not res.is_reliable
