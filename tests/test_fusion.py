"""Tests for the shared decision fusion module (src/fusion.py).

Rule-level tests use hand-built observations. Integration tests run synthetic
sheets generated from template.json through the real preprocessor, so they
need no private dataset.
"""

import json

import numpy as np
import pytest
import torch

import config
from src.fusion import (
    BubbleReader,
    OptionObservation,
    answer_symbol,
    extract_bubble_patch,
    fuse_question,
    patches_to_tensor,
)
from src.grader import OMRGrader, QuestionStatus, ScoringConfig
from src.preprocess import SheetPreprocessor
from tests.synthetic import (
    FillOracleModel,
    expected_symbols,
    make_answer_pattern,
    render_canonical_sheet,
    simulate_phone_capture,
)


def obs(option, nn_class, fill, nn_conf=0.95, classical=None):
    if classical is None:
        classical = "filled" if fill >= config.CLASSICAL_FILL_THRESHOLD else (
            "ambiguous" if fill >= config.CLASSICAL_AMBIGUOUS_LOWER else "unfilled"
        )
    return OptionObservation(option, nn_class, nn_conf, fill, classical)


def blank(option, fill=0.05):
    return obs(option, "unfilled", fill)


# ---------------------------------------------------------------- rule tests

def test_fuse_single_clear_answer():
    pred = fuse_question([blank("A"), obs("B", "filled", 0.85), blank("C"), blank("D"), blank("E")])
    assert pred["filled_options"] == ["B"]
    assert pred["disagreements"] == []
    assert answer_symbol(pred) == "B"


def test_fuse_multiple_mark_requires_two_options():
    pred = fuse_question([obs("A", "filled", 0.80), blank("B"), obs("C", "filled", 0.78), blank("D"), blank("E")])
    assert sorted(pred["filled_options"]) == ["A", "C"]
    assert answer_symbol(pred) == "?"


def test_fuse_all_blank_is_unanswered():
    pred = fuse_question([blank(o) for o in "ABCDE"])
    assert pred["filled_options"] == []
    assert answer_symbol(pred) == "-"


def test_fuse_uniform_shadow_without_cnn_fill_is_unanswered():
    """A shadow darkening all bubbles equally must not be read as a mark."""
    pred = fuse_question([obs(o, "unfilled", 0.30) for o in "ABCDE"])
    assert pred["filled_options"] == []


def test_fuse_cnn_veto_removes_false_mark_on_blank_question():
    """A darker printed letter on a blank row must not become an answer."""
    pred = fuse_question([obs("A", "unfilled", 0.40, nn_conf=0.9), blank("B", 0.20), blank("C"), blank("D"), blank("E")])
    assert pred["filled_options"] == []
    assert pred["disagreements"] == []


def test_fuse_veto_on_very_dark_bubble_goes_to_review():
    """CNN says empty but the bubble is almost fully dark: a genuine disagreement."""
    pred = fuse_question([obs("A", "unfilled", 0.90, nn_conf=0.9), blank("B", 0.20), blank("C"), blank("D"), blank("E")])
    assert pred["filled_options"] == []
    assert pred["disagreements"] == ["A"]

    result = OMRGrader().grade({1: pred}, answer_key={"1": "A"}, scoring_config=ScoringConfig(total_questions=1))
    assert result.questions[0].status == QuestionStatus.REVIEW
    assert result.questions[0].marks_awarded == 0.0


def test_fuse_cnn_accepts_faint_pencil_mark():
    """A faint mark the fill ratio misses is accepted on confident CNN evidence."""
    pred = fuse_question([obs("A", "unfilled", 0.18), obs("B", "filled", 0.22, nn_conf=0.95), blank("C", 0.15), blank("D"), blank("E")])
    assert pred["filled_options"] == ["B"]
    assert pred["disagreements"] == []


def test_fuse_cnn_accept_without_ink_goes_to_review():
    """CNN-filled but the fill ratio shows almost no ink: route to REVIEW."""
    pred = fuse_question([blank("A"), obs("B", "filled", 0.05, nn_conf=0.95), blank("C"), blank("D"), blank("E")])
    assert pred["filled_options"] == ["B"]
    assert pred["disagreements"] == ["B"]


def test_fuse_low_confidence_second_mark_is_not_multi():
    """One confident mark plus one uncertain CNN-filled option stays a single answer."""
    pred = fuse_question([obs("A", "filled", 0.85, nn_conf=0.95), obs("B", "filled", 0.30, nn_conf=0.6), blank("C"), blank("D"), blank("E")])
    assert pred["filled_options"] == ["A"]


def test_patches_to_tensor_matches_torchvision():
    """Vectorized normalization must equal ToTensor + Normalize on RGB patches."""
    from PIL import Image
    from torchvision import transforms as T
    import cv2

    rng = np.random.default_rng(0)
    patch = rng.integers(0, 256, size=config.BUBBLE_PATCH_SIZE, dtype=np.uint8)
    ref = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)))
    ours = patches_to_tensor([patch])[0]
    assert torch.allclose(ref, ours, atol=1e-5)


def test_extract_patch_out_of_bounds_is_blank():
    img = np.zeros((100, 100), dtype=np.uint8)
    patch = extract_bubble_patch(img, {"x": 500, "y": 500, "w": 28, "h": 24})
    assert patch.shape == config.BUBBLE_PATCH_SIZE
    assert patch.min() == 255


# ----------------------------------------------------- synthetic integration

@pytest.fixture(scope="module")
def template():
    with open(config.TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def preprocessor():
    # Synthetic sheets must align against the synthetic reference, even when the
    # real dataset (and therefore the real p01 reference scan) is present.
    return SheetPreprocessor(template_path=config.TEMPLATE_PATH, reference="synthetic")


@pytest.fixture(scope="module")
def synthetic(template, preprocessor):
    pattern = make_answer_pattern()
    sheet = render_canonical_sheet(pattern, template, preprocessor)
    return pattern, sheet


@pytest.fixture(scope="module")
def oracle_reader(template):
    return BubbleReader(template["bubble_rois"], model=FillOracleModel(), device=torch.device("cpu"))


def _accuracy(reader, norm_gray, pattern):
    expected = expected_symbols(pattern)
    preds = reader.read(norm_gray)
    return sum(answer_symbol(preds[q]) == expected[q] for q in expected) / len(expected)


def test_synthetic_scan_reads_every_answer(synthetic, preprocessor, oracle_reader):
    pattern, sheet = synthetic
    res = preprocessor.process(sheet)
    assert res.is_reliable
    assert _accuracy(oracle_reader, res.normalized_gray, pattern) == 1.0


@pytest.mark.parametrize("seed,shift", [(3, 0.03), (11, 0.05)])
def test_synthetic_phone_capture_reads_every_answer(synthetic, preprocessor, oracle_reader, seed, shift):
    pattern, sheet = synthetic
    photo = simulate_phone_capture(sheet, seed=seed, max_corner_shift=shift)
    res = preprocessor.process(photo)
    assert res.method_used == "sift_features"
    assert res.is_reliable
    assert _accuracy(oracle_reader, res.normalized_gray, pattern) == 1.0


def test_fill_margin_only_mode_without_weights(synthetic, preprocessor, template):
    """Without weights, single answers are read by fill margin alone and double
    marks are never attributed to one option."""
    pattern, sheet = synthetic
    reader = BubbleReader(template["bubble_rois"], model=None, device=torch.device("cpu"))
    assert reader.mode == "fill_margin_only"

    preds = reader.read(preprocessor.process(sheet).normalized_gray)
    for q, opts in pattern.items():
        got = preds[q]["filled_options"]
        if len(opts) == 1:
            assert got == opts
        elif len(opts) > 1:
            assert len(got) != 1
        else:
            assert got == []
