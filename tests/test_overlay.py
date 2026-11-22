"""Tests for the answer-marking overlay colours (src/app.py render_overlay)."""

import json

import numpy as np
import pytest

import config
from src.app import render_overlay
from src.grader import OMRGrader, ScoringConfig


@pytest.fixture(scope="module")
def template():
    with open(config.TEMPLATE_PATH, encoding="utf-8") as f:
        return json.load(f)


def center(template, q, opt):
    roi = template["bubble_rois"][str(q)][opt]
    return int(roi["y"] + roi["h"] / 2), int(roi["x"] + roi["w"] / 2)


def colour_at(img, yx):
    """Classify a pixel by its dominant channel (translucent fills are pale)."""
    b, g, r = (int(v) for v in img[yx])
    if g - max(r, b) >= 25:
        return "green"
    if r - max(g, b) >= 25:
        return "red"
    return "neutral"


@pytest.fixture(scope="module")
def overlay(template):
    predictions = {
        1: {"filled_options": ["B"]},            # correct
        2: {"filled_options": ["A"]},            # wrong, key is C
        3: {"filled_options": ["A", "D"]},       # multi-mark, key is D (among marks)
        4: {"filled_options": ["B", "E"]},       # multi-mark, key is C (not marked)
        5: {"filled_options": []},               # unanswered, key is E
    }
    key = {"1": "B", "2": "C", "3": "D", "4": "C", "5": "E"}
    result = OMRGrader().grade(predictions, answer_key=key, scoring_config=ScoringConfig(total_questions=5))
    sheet = np.full((config.CANONICAL_HEIGHT, config.CANONICAL_WIDTH, 3), 255, dtype=np.uint8)
    return render_overlay(sheet, result, template)


def test_correct_answer_is_green(overlay, template):
    assert colour_at(overlay, center(template, 1, "B")) == "green"
    assert colour_at(overlay, center(template, 1, "A")) == "neutral"


def test_wrong_answer_red_and_correct_answer_green(overlay, template):
    assert colour_at(overlay, center(template, 2, "A")) == "red"
    assert colour_at(overlay, center(template, 2, "C")) == "green"
    assert colour_at(overlay, center(template, 2, "B")) == "neutral"


def test_multi_mark_choices_red_and_key_green(overlay, template):
    assert colour_at(overlay, center(template, 3, "A")) == "red"
    assert colour_at(overlay, center(template, 3, "D")) == "red"   # invalid mark, outer green ring
    assert colour_at(overlay, center(template, 4, "B")) == "red"
    assert colour_at(overlay, center(template, 4, "E")) == "red"
    assert colour_at(overlay, center(template, 4, "C")) == "green"


def test_multi_mark_key_among_marks_gets_outer_green_ring(overlay, template):
    cy, cx = center(template, 3, "D")
    roi = template["bubble_rois"]["3"]["D"]
    r = int(min(roi["w"], roi["h"]) / 2)
    assert colour_at(overlay, (cy, cx + r + 6)) == "green"


def test_unanswered_key_is_green_outline_only(overlay, template):
    cy, cx = center(template, 5, "E")
    roi = template["bubble_rois"]["5"]["E"]
    r = int(min(roi["w"], roi["h"]) / 2)
    assert colour_at(overlay, (cy, cx)) == "neutral"          # not filled
    assert colour_at(overlay, (cy, cx + r + 2)) == "green"    # outline
