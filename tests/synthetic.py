"""Synthetic OMR sheet generator for dataset-free integration tests.

Builds a canonical sheet from template.json (the same synthetic reference the
preprocessor falls back to when the private dataset is absent), fills chosen
bubbles with pencil-like marks, and optionally simulates a phone capture with
a perspective warp, background margin, illumination gradient and noise.

No real sheets or personal data are used.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

import config
from src.preprocess import SheetPreprocessor

OPTIONS = ["A", "B", "C", "D", "E"]


def make_answer_pattern(seed: int = 7, n_questions: int = 120) -> Dict[int, List[str]]:
    """Deterministic mix of single answers, multi-marks and blanks.

    Every 10th question is left blank and every 15th has two marks, so the
    pattern exercises all grading branches.
    """
    rng = np.random.default_rng(seed)
    pattern: Dict[int, List[str]] = {}
    for q in range(1, n_questions + 1):
        if q % 10 == 0:
            pattern[q] = []
        elif q % 15 == 0:
            pattern[q] = sorted(rng.choice(OPTIONS, size=2, replace=False).tolist())
        else:
            pattern[q] = [str(rng.choice(OPTIONS))]
    return pattern


def render_canonical_sheet(
    marks: Dict[int, List[str]],
    template: Dict,
    preprocessor: Optional[SheetPreprocessor] = None,
    ink_level: int = 55,
) -> np.ndarray:
    """Render a canonical-resolution grayscale sheet with the given bubbles filled."""
    pre = preprocessor or SheetPreprocessor(template_path=config.TEMPLATE_PATH)
    sheet = pre._synthesize_reference_sheet().copy()
    rois = template["bubble_rois"]
    for q, opts in marks.items():
        for opt in opts:
            roi = rois[str(q)][opt]
            cx = int(roi["x"] + roi["w"] / 2)
            cy = int(roi["y"] + roi["h"] / 2)
            r = int(min(roi["w"], roi["h"]) / 2) - 1
            cv2.circle(sheet, (cx, cy), r, ink_level, -1)
    return sheet


def simulate_phone_capture(
    canonical_gray: np.ndarray,
    seed: int = 3,
    max_corner_shift: float = 0.03,
    margin: int = 120,
) -> np.ndarray:
    """Warp the sheet onto a larger dark background with mild perspective and shading."""
    rng = np.random.default_rng(seed)
    h, w = canonical_gray.shape[:2]
    out_w, out_h = w + 2 * margin, h + 2 * margin

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-max_corner_shift, max_corner_shift, size=(4, 2)) * np.float32([w, h])
    dst = src + np.float32([margin, margin]) + jitter.astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(
        canonical_gray, H, (out_w, out_h), borderMode=cv2.BORDER_CONSTANT, borderValue=60
    ).astype(np.float32)

    # Horizontal illumination gradient (window-side lighting)
    gradient = np.linspace(1.0, 0.82, out_w, dtype=np.float32)[None, :]
    warped *= gradient
    warped += rng.normal(0, 3.0, size=warped.shape).astype(np.float32)
    warped = np.clip(warped, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(warped, (3, 3), 0)


def encode_jpeg(gray: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


def expected_symbols(pattern: Dict[int, List[str]]) -> Dict[int, str]:
    """Map a mark pattern to the ground-truth alphabet used by evaluate.py."""
    out = {}
    for q, opts in pattern.items():
        out[q] = opts[0] if len(opts) == 1 else ("?" if len(opts) > 1 else "-")
    return out


def answer_key_for(pattern: Dict[int, List[str]], wrong_every: int = 7) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Build an answer key that matches most single answers and contradicts some.

    Returns the key and the expected grader status counts.
    """
    key: Dict[str, str] = {}
    counts = {"CORRECT": 0, "INCORRECT": 0, "MULTIPLE_MARK": 0, "UNANSWERED": 0}
    for q, opts in pattern.items():
        if len(opts) == 0:
            key[str(q)] = "A"
            counts["UNANSWERED"] += 1
        elif len(opts) > 1:
            key[str(q)] = opts[0]
            counts["MULTIPLE_MARK"] += 1
        elif q % wrong_every == 0:
            key[str(q)] = OPTIONS[(OPTIONS.index(opts[0]) + 1) % len(OPTIONS)]
            counts["INCORRECT"] += 1
        else:
            key[str(q)] = opts[0]
            counts["CORRECT"] += 1
    return key, counts


class FillOracleModel(torch.nn.Module):
    """Deterministic stand-in for the bubble CNN, for synthetic-sheet tests.

    Classifies a normalized (N, 3, H, W) patch batch as "filled" when the dark
    fraction inside the central bubble disc exceeds 0.5, otherwise "unfilled".
    This exercises the CNN-dependent fusion rules (for example multi-mark
    detection) without depending on whichever trained weights are on disk.
    """

    def __init__(self, dark_threshold: float = 0.5, radius: int = 12):
        super().__init__()
        h, w = config.BUBBLE_PATCH_SIZE
        yy, xx = np.ogrid[:h, :w]
        disc = ((yy - h // 2) ** 2 + (xx - w // 2) ** 2) <= radius ** 2
        self.register_buffer("disc", torch.from_numpy(disc.astype(np.float32)))
        self.dark_threshold = dark_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x[:, 0] * 0.229 + 0.485                      # undo ImageNet normalization
        dark = ((gray < 0.5).float() * self.disc).sum(dim=(1, 2)) / self.disc.sum()
        filled = dark > self.dark_threshold
        logits = torch.full((x.shape[0], len(config.BUBBLE_CLASSES)), -5.0, device=x.device)
        logits[:, config.CLASS_TO_IDX["filled"]] = torch.where(filled, 5.0, -5.0)
        logits[:, config.CLASS_TO_IDX["unfilled"]] = torch.where(filled, -5.0, 5.0)
        return logits
