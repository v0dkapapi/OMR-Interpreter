"""Model-free tests for handwriting localization in header field cells."""

import cv2
import numpy as np

from src.handwriting_ocr import FIELD_CELLS, LABEL_BOXES, HandwritingOCR

H, W = 2338, 1654


def blank_sheet():
    return np.full((H, W, 3), 235, dtype=np.uint8)


def draw_label(img, field):
    lab = LABEL_BOXES[field]
    cv2.putText(img, "Label", (lab["x"] + 2, lab["y"] + lab["h"] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)


def draw_text(img, text, x, y, scale=1.1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, scale, (40, 40, 40), 2)


def test_empty_cell_returns_no_crops():
    img = blank_sheet()
    draw_label(img, "module_code")
    assert HandwritingOCR.locate_handwriting(img, "module_code") == []


def test_label_is_erased_and_writing_located():
    img = blank_sheet()
    draw_label(img, "module_code")
    cell = FIELD_CELLS["module_code"]
    draw_text(img, "ECON12964", cell["x"] + 200, cell["y"] + 75)
    crops = HandwritingOCR.locate_handwriting(img, "module_code")
    assert len(crops) == 1
    crop = crops[0]
    # Tight crop around the writing, not the whole cell
    assert crop.shape[1] < cell["w"] - 100
    assert crop.shape[0] < cell["h"]


def test_two_line_entry_is_split():
    img = blank_sheet()
    draw_label(img, "date")
    cell = FIELD_CELLS["date"]
    draw_text(img, "5 August", cell["x"] + 110, cell["y"] + 40, scale=0.9)
    draw_text(img, "2024", cell["x"] + 110, cell["y"] + 85, scale=0.9)
    crops = HandwritingOCR.locate_handwriting(img, "date")
    assert len(crops) == 2
    assert crops[0].shape[1] > crops[1].shape[1]  # "5 August" wider than "2024"
