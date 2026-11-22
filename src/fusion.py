"""Shared bubble reading and decision fusion for the OMR pipeline.

This module is the single source of truth for turning a preprocessed sheet into
per-question predictions. Both the live API (src/app.py) and the stress test
benchmark (src/evaluate.py) call it, so the benchmark measures exactly the code
path that serves users.

Pipeline per sheet:
1. Crop every bubble ROI from the illumination-normalized canonical sheet.
2. Score each patch with the classical fill-ratio baseline.
3. Score all patches with the MobileNetV3 classifier in a single batch.
4. Fuse both signals per question using relative fill margins.

When no trained classifier weights are available, the reader runs in
fill-margin-only mode: every option gets the CNN class "none" with confidence
0.0, so only the relative fill-margin rules can select an answer. Multiple
marks and CNN-confirmed weak marks cannot be detected in this mode. The
fill-ratio class is deliberately NOT used as a stand-in for the CNN: on real
sheets printed letters, shadows and small alignment drift push empty bubbles
to fill ratios of 0.25 to 0.40, and treating those as CNN evidence floods the
multi-mark rules (36% stress test accuracy versus 71% for margin-only).

Fusion output per question (consumed by src/grader.py):
    {
        "filled_options": [...],     # 0, 1 or 2 options
        "ambiguous_options": [...],  # reserved; currently always empty
        "disagreements": [...],      # options accepted on CNN evidence alone
        "confidences": {opt: float}, # CNN (or classical) confidence per option
    }
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

import config
from src.train_bubble_model import classical_classify_bubble

# Margin (px) added around each template ROI before resizing to the patch size.
# Must match the margin used when the training patches were extracted.
PATCH_MARGIN = 6

# ImageNet normalization used by the MobileNetV3 classifier.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# Decision fusion thresholds. Calibrated on held-out vision_ds validation
# contributors (never on stress test persons); see docs/EXPERIMENT_LOG.md,
# Experiment 4. Changing these changes the benchmark baseline.
MULTI_CNN_CONF = 0.80        # >= 2 options CNN-filled above this -> multiple mark
SINGLE_MIN_FILL = 0.26       # fill-margin candidate: darkest option at least this dark ...
SINGLE_MIN_GAP = 0.05        # ... and this far ahead of the runner-up
SINGLE_CNN_CONF = 0.50       # ... (or CNN-filled above this instead of the gap)
VETO_CNN_CONF = 0.50         # CNN says unfilled above this -> drop the fill-margin candidate
VETO_REVIEW_FILL = 0.80      # vetoed candidate this dark -> genuine disagreement (REVIEW)
CNN_ACCEPT_CONF = 0.80       # nothing selected, exactly one CNN-filled above this -> accept
CNN_ACCEPT_REVIEW_FILL = 0.10  # accepted on CNN alone with fill below this -> REVIEW


@dataclass
class OptionObservation:
    """Classifier evidence for one bubble option."""
    option: str
    nn_class: str
    nn_conf: float
    fill_ratio: float
    classical_class: str


def extract_bubble_patch(norm_gray: np.ndarray, roi: Dict[str, int]) -> np.ndarray:
    """Crop one bubble ROI (plus margin) and resize it to the classifier patch size."""
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    y1 = max(0, y - PATCH_MARGIN)
    y2 = min(norm_gray.shape[0], y + h + PATCH_MARGIN)
    x1 = max(0, x - PATCH_MARGIN)
    x2 = min(norm_gray.shape[1], x + w + PATCH_MARGIN)
    patch = norm_gray[y1:y2, x1:x2]
    if patch.size == 0:
        return np.full(config.BUBBLE_PATCH_SIZE, 255, dtype=np.uint8)
    return cv2.resize(patch, config.BUBBLE_PATCH_SIZE, interpolation=cv2.INTER_AREA)


def patches_to_tensor(patches: List[np.ndarray]) -> torch.Tensor:
    """Convert grayscale uint8 patches to a normalized (N, 3, H, W) float tensor.

    Equivalent to torchvision ToTensor + Normalize on a grayscale-to-RGB image.
    """
    arr = np.stack(patches).astype(np.float32) / 255.0          # (N, H, W)
    arr = np.repeat(arr[:, None, :, :], 3, axis=1)              # (N, 3, H, W)
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(arr)


def fuse_question(observations: List[OptionObservation]) -> Dict[str, Any]:
    """Fuse CNN and fill-ratio evidence for one question.

    Rules, in order:
    1. MULTIPLE MARK: two or more options CNN-filled with confidence above
       MULTI_CNN_CONF. Requiring confident CNN evidence on every option keeps
       shadows and printed letters from invalidating a correct single answer.
    2. FILL-MARGIN SINGLE: the darkest option is dark enough and clearly ahead
       of the runner-up. The CNN can veto it (confident "unfilled"), which
       removes most false marks on blank questions. A veto on a very dark
       bubble is a genuine disagreement and is routed to REVIEW.
    3. CNN SINGLE: nothing selected yet and exactly one option is confidently
       CNN-filled (faint pencil marks the fill ratio misses). If the fill ratio
       shows almost no ink, the disagreement is routed to REVIEW.
    4. Otherwise UNANSWERED.

    Without a trained CNN (class "none"), only rule 2 can fire, without veto.
    """
    result: Dict[str, Any] = {
        "filled_options": [],
        "ambiguous_options": [],
        "disagreements": [],
        "confidences": {o.option: round(o.nn_conf, 4) for o in observations},
    }
    if not observations:
        return result

    ranked = sorted(observations, key=lambda o: o.fill_ratio, reverse=True)
    cnn_filled = [o for o in ranked if o.nn_class == "filled" and o.nn_conf > MULTI_CNN_CONF]

    # 1. Multiple mark
    if len(cnn_filled) >= 2:
        result["filled_options"] = [o.option for o in cnn_filled]
        return result

    # 2. Fill-margin single answer, subject to CNN veto
    top = ranked[0]
    sec_fill = ranked[1].fill_ratio if len(ranked) > 1 else 0.0
    if top.fill_ratio >= SINGLE_MIN_FILL and (
        top.fill_ratio - sec_fill >= SINGLE_MIN_GAP
        or (top.nn_class == "filled" and top.nn_conf > SINGLE_CNN_CONF)
    ):
        if top.nn_class == "unfilled" and top.nn_conf > VETO_CNN_CONF:
            if top.fill_ratio >= VETO_REVIEW_FILL:
                result["disagreements"] = [top.option]
                return result
        else:
            result["filled_options"] = [top.option]
            return result

    # 3. CNN-confirmed single answer
    if len(cnn_filled) == 1:
        accepted = cnn_filled[0]
        result["filled_options"] = [accepted.option]
        if accepted.fill_ratio < CNN_ACCEPT_REVIEW_FILL:
            result["disagreements"] = [accepted.option]

    return result


def answer_symbol(prediction: Dict[str, Any]) -> str:
    """Collapse a fused prediction to the ground-truth label alphabet.

    Returns the single option letter, "?" for a multiple mark, "-" for unanswered.
    """
    filled = prediction.get("filled_options", [])
    if len(filled) == 1:
        return filled[0]
    if len(filled) > 1:
        return "?"
    return "-"


class BubbleReader:
    """Reads all bubbles on a canonical sheet and returns fused predictions."""

    def __init__(
        self,
        bubble_rois: Dict[str, Dict[str, Dict[str, int]]],
        model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        self.bubble_rois = bubble_rois
        self.model = model
        self.device = device if device is not None else config.get_device()
        if self.model is not None:
            self.model.to(self.device).eval()

    @property
    def mode(self) -> str:
        return "cnn+classical" if self.model is not None else "fill_margin_only"

    def _classify(self, patches: List[np.ndarray]) -> Tuple[List[str], List[float]]:
        """Run the CNN on all patches in one batch."""
        batch = patches_to_tensor(patches).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(batch), dim=-1).cpu().numpy()
        idx = probs.argmax(axis=-1)
        classes = [config.IDX_TO_CLASS[int(i)] for i in idx]
        confs = [float(probs[n, i]) for n, i in enumerate(idx)]
        return classes, confs

    def observe(self, norm_gray: np.ndarray) -> Dict[int, List[OptionObservation]]:
        """Collect per-option classifier evidence for every question."""
        keys: List[Tuple[int, str]] = []
        patches: List[np.ndarray] = []
        classical: List[Tuple[str, float]] = []

        for q_str, options in self.bubble_rois.items():
            for opt, roi in options.items():
                patch = extract_bubble_patch(norm_gray, roi)
                keys.append((int(q_str), opt))
                patches.append(patch)
                classical.append(classical_classify_bubble(patch))

        if not patches:
            return {}

        if self.model is not None:
            nn_classes, nn_confs = self._classify(patches)
        else:
            # No CNN evidence: fusion falls back to the fill-margin rules only.
            nn_classes = ["none"] * len(classical)
            nn_confs = [0.0] * len(classical)

        by_question: Dict[int, List[OptionObservation]] = {}
        for (q_num, opt), (c_class, fill), nn_class, nn_conf in zip(keys, classical, nn_classes, nn_confs):
            by_question.setdefault(q_num, []).append(
                OptionObservation(opt, nn_class, nn_conf, fill, c_class)
            )
        return by_question

    def read(self, norm_gray: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Return fused predictions for every question on the sheet."""
        return {q: fuse_question(obs) for q, obs in self.observe(norm_gray).items()}


def load_bubble_model(device: Optional[torch.device] = None) -> Optional[torch.nn.Module]:
    """Load trained classifier weights, or return None for fill-margin-only mode."""
    from src.train_bubble_model import build_model

    model_path = config.TRAIN_CONFIG["final_model_path"]
    if not model_path.exists():
        return None
    device = device if device is not None else config.get_device()
    model = build_model(num_classes=len(config.BUBBLE_CLASSES), pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()
