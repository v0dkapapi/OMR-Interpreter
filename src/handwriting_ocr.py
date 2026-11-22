"""Handwritten Student Metadata OCR for OMR Answer Sheets.

Engines:
- Primary: TrOCR (microsoft/trocr-base-handwritten) on MPS/CPU
- Fallback: EasyOCR (automatic graceful offline fallback)

Features:
- Extracts student metadata: name, module_code, seat_number, date, student_number
- Per-sheet handwriting localization inside each field cell (printed labels blanked,
  multi-line entries split), so text written above, beside or across lines is not cut off
- TrOCR n-best decoding filtered by field validators (src/field_parsing.py): the first
  reading that is a valid name / module code / date / seat number wins, which rejects
  language-model hallucinations
- Dates normalized to ISO YYYY-MM-DD; module code prefixes snapped to known subjects
- Structural grid line suppression and per-cell digit slicing for segmented ID boxes
- Field format validation (8-digit roll number, module code regex, seat number format, date parsing)
- Local roster.csv cross-checking (flags mismatches without modifying text)
- Strict privacy: NEVER logs, prints, or exports student names
"""

import os
import re
import csv
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

import config
from src.field_parsing import MONTHS, PARSERS, parse_date

logger = logging.getLogger(__name__)

# Handwriting bands below the printed field labels, measured from the mean ink
# map of 79 aligned training sheets (see docs/EXPERIMENT_LOG.md, Experiment 5).
# Must match handwritten_rois in template.json.
CANONICAL_FIELD_ROIS = {
    "name": {"x": 650, "y": 248, "w": 540, "h": 61},
    "module_code": {"x": 650, "y": 345, "w": 540, "h": 63},
    "seat_number": {"x": 650, "y": 455, "w": 275, "h": 54},
    "date": {"x": 935, "y": 458, "w": 295, "h": 52},
    "student_number_written": {"x": 1238, "y": 194, "w": 324, "h": 64},
}


# Search cells for handwriting localization (canonical pixels). Each cell spans
# the printed label and all space a student might write in, up to the next field.
FIELD_CELLS = {
    "name": {"x": 645, "y": 216, "w": 550, "h": 104},
    "module_code": {"x": 645, "y": 316, "w": 550, "h": 112},
    "seat_number": {"x": 645, "y": 430, "w": 280, "h": 92},
    "date": {"x": 920, "y": 430, "w": 312, "h": 98},
}

# Printed field labels inside each cell ("Name", "Module Code", ...). Ink blobs
# lying entirely inside a label box (plus LABEL_MARGIN) are printed glyphs and
# are erased; handwriting that crosses into the box extends beyond it and stays.
LABEL_BOXES = {
    "name": {"x": 648, "y": 219, "w": 94, "h": 31},
    "module_code": {"x": 648, "y": 317, "w": 160, "h": 30},
    "seat_number": {"x": 646, "y": 435, "w": 106, "h": 30},
    "date": {"x": 938, "y": 435, "w": 74, "h": 32},
}

# EasyOCR character allowlists used as a fallback reader per field.
LABEL_MARGIN = 8

# Constrained decoding: tokens TrOCR may emit per field. Blocking everything
# else stops the decoder's English language model from turning "ECON12964"
# into "economology" or a date into "management".
_ORDINAL_SUFFIXES = {"st", "nd", "rd", "th"}


def _allowed_token(field_name: str, token_text: str) -> bool:
    core = token_text.strip()
    if not core:
        return True  # bare space
    if field_name == "module_code":
        return bool(re.fullmatch(r"[A-Z0-9]+", core))
    # seat_number is left unconstrained: short values give the language model
    # little room to wander, and constraining it measured worse (Experiment 5).
    if field_name == "date":
        if re.fullmatch(r"[0-9/.\-,]+", core):
            return True
        if core.isalpha():
            low = core.lower()
            return low in _ORDINAL_SUFFIXES or any(low in m for m in MONTHS)
        return False
    if field_name == "name":
        return bool(re.fullmatch(r"[A-Za-z'\-]+", core))
    return True

# Ink blocks at least this tall (canonical px) are checked for two touching lines.
SPLIT_MIN_BLOCK_HEIGHT = 62

EASYOCR_ALLOWLISTS = {
    "module_code": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
    "seat_number": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
    "date": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/.- ",
}


class HandwritingOCR:
    """Production handwriting OCR engine with TrOCR and EasyOCR fallback."""

    def __init__(
        self,
        trocr_model_name: str = config.OCR_CONFIG["trocr_model"],
        device: Optional[torch.device] = None,
        use_trocr: bool = True
    ):
        self.device = device if device else config.get_device()
        self.trocr_model_name = trocr_model_name
        self.use_trocr = use_trocr

        # Lazy loaded engines
        self._trocr_processor = None
        self._trocr_model = None
        self._easyocr_reader = None
        self._trocr_available = None

        # Preload roster mapping (student_id / contributor -> split) WITHOUT student names
        self.roster_splits: Dict[str, str] = {}
        self._load_roster_metadata()

    def _load_roster_metadata(self):
        """Load contributor/split mapping from roster.csv without student names."""
        roster_path = config.VISION_ROSTER_PATH
        try:
            if roster_path.exists():
                with open(roster_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header: contributor,person,split
                    for row in reader:
                        if len(row) >= 3:
                            contributor = row[0].strip()
                            split = row[2].strip()
                            self.roster_splits[contributor] = split
        except (PermissionError, OSError) as e:
            logger.warning(f"Could not read roster.csv metadata: {e}")

    def _init_trocr(self) -> bool:
        """Attempt to load TrOCR model; return True if successful, False if offline/unavailable."""
        if self._trocr_available is not None:
            return self._trocr_available

        if not self.use_trocr:
            self._trocr_available = False
            return False

        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            logger.info(f"Loading TrOCR model '{self.trocr_model_name}' on {self.device}...")
            try:
                # Fast local load from cache without remote hub queries
                self._trocr_processor = TrOCRProcessor.from_pretrained(
                    self.trocr_model_name, local_files_only=True
                )
                self._trocr_model = VisionEncoderDecoderModel.from_pretrained(
                    self.trocr_model_name, local_files_only=True
                ).to(self.device)
            except Exception:
                self._trocr_processor = TrOCRProcessor.from_pretrained(self.trocr_model_name)
                self._trocr_model = VisionEncoderDecoderModel.from_pretrained(
                    self.trocr_model_name
                ).to(self.device)
            self._trocr_model.eval()
            self._trocr_available = True
            logger.info("TrOCR model loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"TrOCR initialization failed (offline or network error): {e}. Falling back to EasyOCR.")
            self._trocr_available = False
            return False

    def _get_easyocr(self):
        """Lazy loader for EasyOCR reader."""
        if self._easyocr_reader is None:
            import easyocr
            # EasyOCR GPU is CUDA only; on Mac use gpu=False.
            # Keep model weights inside the project (git-ignored models/) instead of
            # ~/.EasyOCR so sandboxed or offline runners can use a pre-seeded cache.
            model_dir = Path(config.OCR_CONFIG["easyocr_model_dir"])
            model_dir.mkdir(parents=True, exist_ok=True)
            self._easyocr_reader = easyocr.Reader(
                ["en"],
                gpu=False,
                model_storage_directory=str(model_dir),
                user_network_directory=str(model_dir / "user_network"),
                download_enabled=not config.OCR_CONFIG["offline"],
            )
        return self._easyocr_reader

    def _enhance_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Apply contrast-limited adaptive histogram equalization to enhance faint/pencil handwriting."""
        if crop_bgr.size == 0:
            return crop_bgr
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def _suppress_grid_lines(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Suppress printed grid/box borders for segmented digit fields using morphological subtraction."""
        if crop_bgr.size == 0:
            return crop_bgr
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Structural kernel for vertical dividing lines (height >= 16)
        vert_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 16))
        vlines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_k)

        # Structural kernel for horizontal borders (width >= 20)
        horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        hlines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_k)

        lines = cv2.add(vlines, hlines)
        cleaned_thresh = cv2.subtract(thresh, lines)
        return cv2.cvtColor(255 - cleaned_thresh, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _clean_field_text(field_name: str, text: str) -> str:
        """Strip printed form field prompt labels and common OCR prompt artifacts."""
        clean = text.strip()
        field_prefixes = {
            "name": r"^(?:name|student\s*name|warne|haine|haine\s*name|student)[:\s.-]*",
            "module_code": r"^(?:module\s*code|course\s*code|module|moqule\s*gode|moqule|aouuic\s*coc|code|course)[:\s.-]*",
            "seat_number": r"^(?:seat\s*no\.?|seat\s*number|seat|ocal\s*n[o0]\.?|ealfio|peal\s*n[o0]\.?|oeat\s*n[o0]\.?|oga\s*mo)[:\s.-]*",
            "date": r"^(?:date|exam\s*date|dalte|dale)[:\s.-]*",
            "student_number_written": r"^(?:student\s*no\.?|student\s*number|student\s*id|id)[:\s.-]*"
        }
        prefix = field_prefixes.get(field_name)
        if prefix:
            stripped = re.sub(prefix, "", clean, flags=re.IGNORECASE).strip()
            if stripped:
                clean = stripped

        # Clean stray leading/trailing quotes and noise punctuation
        clean = re.sub(r"^[\"'`.,;:?!\-_/\\|]+\s*", "", clean)
        clean = re.sub(r"\s*[\"'`.,;:?!\-_/\\|]+$", "", clean)

        if field_name == "name":
            # Strip prompt artifacts like leading single letters, 'none', 's.'
            clean = re.sub(r"^(?:s\.|none|s|is)\s+", "", clean, flags=re.IGNORECASE).strip()
            clean = re.sub(r"\b(?:name|student|student\s*name)\b", "", clean, flags=re.IGNORECASE).strip()
            clean = re.sub(r"^[\"'`.,;:?!\-_/\\|]+\s*", "", clean)
            clean = re.sub(r"\s*[\"'`.,;:?!\-_/\\|]+$", "", clean)
            # Title-case words nicely while preserving Scottish/Irish names (e.g. McLoughlin, O'Connor)
            words = []
            for w in clean.split():
                if w.startswith("Mc") and len(w) > 2:
                    words.append("Mc" + w[2:].capitalize())
                elif w.startswith("O'") and len(w) > 2:
                    words.append("O'" + w[2:].capitalize())
                elif w.isupper() or w.islower():
                    words.append(w.capitalize())
                else:
                    words.append(w)
            clean = " ".join(words)

        return clean.strip()

    def _run_trocr(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Run TrOCR generation on a BGR image crop."""
        pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        pixel_values = self._trocr_processor(
            pil_img, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self._trocr_model.generate(
                pixel_values,
                max_new_tokens=32,
                return_dict_in_generate=True,
                output_scores=True
            )

        text = self._trocr_processor.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )[0].strip()

        if outputs.scores:
            logits = torch.stack(outputs.scores, dim=1)
            probs = torch.softmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1).values
            conf = float(torch.mean(max_probs).item())
        else:
            conf = 0.85

        return text, conf

    def _run_easyocr(self, crop_bgr: np.ndarray, allowlist: Optional[str] = None) -> Tuple[str, float]:
        """Run EasyOCR detection and recognition on a BGR image crop."""
        try:
            reader = self._get_easyocr()
            if allowlist:
                results = reader.readtext(crop_bgr, allowlist=allowlist)
            else:
                results = reader.readtext(crop_bgr)

            if not results:
                return "", 0.0

            # Sort detected boxes top-to-bottom (grouped by vertical buckets), then left-to-right
            sorted_boxes = sorted(results, key=lambda r: (r[0][0][1] // 20, r[0][0][0]))
            text_parts = [r[1] for r in sorted_boxes]
            confs = [r[2] for r in sorted_boxes]
            raw_text = " ".join(text_parts).strip()
            conf = float(np.mean(confs)) if confs else 0.5
            return raw_text, conf
        except Exception as e:
            logger.debug(f"EasyOCR run error: {e}")
            return "", 0.0

    def _score_student_number(self, text: str, raw_conf: float) -> Tuple[float, str, float]:
        """Schema scoring for the written 8-digit student number.

        Returns (composite_score, digits, adjusted_confidence).
        """
        clean = self._clean_field_text("student_number_written", text)
        if not clean:
            return -1.0, "", 0.0
        # Map leading 'a'/'A' to '8' (common optical character confusion)
        if re.match(r"^[aA]\s*\d{6,8}", clean):
            clean = "8" + clean[1:]
        digits = re.sub(r"\D", "", clean)
        alpha_penalty = min(0.4, len(re.findall(r"[A-Za-z]", clean)) * 0.05)
        if len(digits) == config.OCR_CONFIG["roll_digits"]:
            return (1.0 * 0.70 + raw_conf * 0.30 - alpha_penalty), digits, max(raw_conf, 0.85)
        if 6 <= len(digits) <= 9:
            return (0.5 * 0.70 + raw_conf * 0.30 - alpha_penalty), digits, raw_conf
        if digits:
            return (0.2 * 0.70 + raw_conf * 0.30 - alpha_penalty), digits, raw_conf
        return -0.5, clean, 0.0

    # ------------------------------------------------------------------
    # Handwriting localization
    # ------------------------------------------------------------------

    @staticmethod
    def locate_handwriting(canonical_bgr: np.ndarray, field_name: str) -> List[np.ndarray]:
        """Find the handwriting inside a field cell and return one crop per text line.

        The printed label is blanked, long printed rules are removed, and the
        remaining ink is cropped tightly (with padding). Entries written over
        two lines (e.g. "5 August" / "2024") are split so each line can be read
        separately. Returns an empty list when the cell holds no handwriting.
        """
        cell = FIELD_CELLS[field_name]
        cx, cy, cw, ch = cell["x"], cell["y"], cell["w"], cell["h"]
        img_h, img_w = canonical_bgr.shape[:2]
        x1, y1 = max(0, cx), max(0, cy)
        x2, y2 = min(img_w, cx + cw), min(img_h, cy + ch)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return []
        gray = cv2.cvtColor(canonical_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

        # Local background estimate (strokes removed) and ink = darker than background
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
        background = cv2.GaussianBlur(background, (0, 0), 5)

        diff = cv2.subtract(background, gray)
        threshold = max(18.0, 0.35 * float(np.percentile(diff, 99.5)))
        ink = (diff > threshold).astype(np.uint8) * 255

        # Remove long printed rules and cell borders
        h_lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1)))
        v_lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45)))
        ink = cv2.subtract(ink, cv2.add(h_lines, v_lines))

        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        if n <= 1:
            return []

        # Erase printed label glyphs: components wholly inside the label box
        lab = LABEL_BOXES[field_name]
        lx1 = lab["x"] - x1 - LABEL_MARGIN
        ly1 = lab["y"] - y1 - LABEL_MARGIN
        lx2 = lab["x"] + lab["w"] - x1 + LABEL_MARGIN
        ly2 = lab["y"] + lab["h"] - y1 + LABEL_MARGIN
        label_ids = [
            i for i in range(1, n)
            if stats[i, cv2.CC_STAT_LEFT] >= lx1 and stats[i, cv2.CC_STAT_TOP] >= ly1
            and stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] <= lx2
            and stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] <= ly2
        ]
        clean = gray.copy()
        if label_ids:
            label_mask = cv2.dilate(np.isin(labels, label_ids).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            clean[label_mask] = background[label_mask]

        candidates = [i for i in range(1, n) if i not in set(label_ids)]
        if not candidates:
            return []
        areas = {i: int(stats[i, cv2.CC_STAT_AREA]) for i in candidates}
        total = float(sum(areas.values()))
        keep = [i for i, a in areas.items() if a >= max(12.0, 0.01 * total)]
        if not keep or sum(areas[i] for i in keep) < 60:
            return []
        mask = np.isin(labels, keep)

        ys, xs = np.where(mask)
        bx1, bx2, by1, by2 = xs.min(), xs.max(), ys.min(), ys.max()

        # Split into text lines on empty row gaps
        profile = mask[by1:by2 + 1, bx1:bx2 + 1].sum(axis=1)
        empty = profile <= max(1, 0.02 * profile.max())
        segments, start = [], None
        for i, is_empty in enumerate(empty):
            if not is_empty and start is None:
                start = i
            elif is_empty and start is not None:
                segments.append([start, i - 1])
                start = None
        if start is not None:
            segments.append([start, len(empty) - 1])
        # Merge segments separated by small gaps or too short to be a line
        merged: List[List[int]] = []
        for seg in segments:
            if merged and (seg[0] - merged[-1][1] < 6 or (seg[1] - seg[0]) < 12 or (merged[-1][1] - merged[-1][0]) < 12):
                merged[-1][1] = seg[1]
            else:
                merged.append(seg)

        # A single block about twice a line's height is two lines whose strokes
        # touch (descenders, a slanted "2024"): split at the lightest middle row.
        if len(merged) == 1 and (merged[0][1] - merged[0][0]) >= SPLIT_MIN_BLOCK_HEIGHT:
            s0, s1 = merged[0]
            smooth = np.convolve(profile[s0:s1 + 1].astype(float), np.ones(5) / 5, mode="same")
            lo, hi = int(len(smooth) * 0.35), int(len(smooth) * 0.65)
            if hi > lo:
                cut = lo + int(np.argmin(smooth[lo:hi]))
                if smooth[cut] <= 0.15 * smooth.max():
                    merged = [[s0, s0 + cut], [s0 + cut + 1, s1]]

        if len(merged) > 1:
            line_ink = [int(mask[by1 + s0:by1 + s1 + 1].sum()) for s0, s1 in merged]
            total_ink = float(sum(line_ink))
            merged = [seg for seg, ink_px in zip(merged, line_ink) if ink_px >= 0.12 * total_ink]

        pad = 8
        clean_bgr = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
        crops = []
        for s0, s1 in merged:
            line_mask = mask[by1 + s0:by1 + s1 + 1, :]
            cols = np.where(line_mask.any(axis=0))[0]
            if cols.size == 0:
                continue
            ty1 = max(0, by1 + s0 - pad)
            ty2 = min(clean.shape[0], by1 + s1 + 1 + pad)
            tx1 = max(0, cols.min() - pad)
            tx2 = min(clean.shape[1], cols.max() + 1 + pad)
            crops.append(clean_bgr[ty1:ty2, tx1:tx2])
        return crops

    # ------------------------------------------------------------------
    # TrOCR n-best decoding and validated field reading
    # ------------------------------------------------------------------

    def _field_token_mask(self, field_name: Optional[str], vocab_size: int) -> torch.Tensor:
        """Boolean vocabulary mask of tokens allowed for a field (cached)."""
        if not hasattr(self, "_token_masks"):
            self._token_masks: Dict[Tuple[Optional[str], int], torch.Tensor] = {}
        key = (field_name, vocab_size)
        if key not in self._token_masks:
            tok = self._trocr_processor.tokenizer
            if field_name is None:
                mask = torch.ones(vocab_size, dtype=torch.bool)
            else:
                mask = torch.zeros(vocab_size, dtype=torch.bool)
                pieces = tok.convert_ids_to_tokens(list(range(min(vocab_size, len(tok)))))
                for i, piece in enumerate(pieces):
                    if piece is not None and _allowed_token(field_name, tok.convert_tokens_to_string([piece])):
                        mask[i] = True
                mask[tok.eos_token_id] = True
            self._token_masks[key] = mask
        return self._token_masks[key]

    def _trocr_nbest(
        self, crops: List[np.ndarray], fields: Optional[List[Optional[str]]] = None
    ) -> List[List[Tuple[str, float]]]:
        """Decode all crops in one batch; return the n-best (text, confidence) per crop.

        When ``fields`` is given, decoding for each crop is constrained to the
        tokens valid for its field. Confidence is the exponentiated
        length-normalized beam log-probability.
        """
        if not crops or not self._init_trocr():
            return [[] for _ in crops]
        from transformers import LogitsProcessor, LogitsProcessorList

        k = int(config.OCR_CONFIG.get("trocr_num_beams", 5))
        fields = fields or [None] * len(crops)
        images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
        pixel_values = self._trocr_processor(images, return_tensors="pt").pixel_values.to(self.device)
        owner = self

        class FieldConstraint(LogitsProcessor):
            def __call__(self, input_ids, scores):
                vocab = scores.shape[-1]
                if not hasattr(self, "row_mask"):
                    per_crop = torch.stack([owner._field_token_mask(f, vocab) for f in fields])
                    self.row_mask = per_crop.repeat_interleave(k, dim=0).to(scores.device)
                return scores.masked_fill(~self.row_mask[: scores.shape[0]], float("-inf"))

        with torch.no_grad():
            out = self._trocr_model.generate(
                pixel_values,
                num_beams=k,
                num_return_sequences=k,
                max_new_tokens=24,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True,
                logits_processor=LogitsProcessorList([FieldConstraint()]),
            )
        texts = self._trocr_processor.batch_decode(out.sequences, skip_special_tokens=True)
        scores = out.sequences_scores.detach().cpu().numpy() if out.sequences_scores is not None else np.zeros(len(texts))
        results = []
        for i in range(len(crops)):
            beams = []
            for j in range(i * k, (i + 1) * k):
                text = texts[j].strip()
                if text:
                    beams.append((text, float(np.exp(scores[j]))))
            results.append(beams)
        return results

    @staticmethod
    def _line_candidates(line_beams: List[List[Tuple[str, float]]], limit: int = 8) -> List[Tuple[str, float]]:
        """Combine per-line n-best lists into whole-field candidates, best first."""
        line_beams = [b for b in line_beams if b]
        if not line_beams:
            return []
        if len(line_beams) == 1:
            return line_beams[0][:limit]
        top = [b[0] for b in line_beams]
        candidates = [(" ".join(t for t, _ in top), float(np.mean([c for _, c in top])))]
        # Vary one line at a time through its alternatives
        for i, beams in enumerate(line_beams):
            for alt in beams[1:3]:
                combo = list(top)
                combo[i] = alt
                candidates.append((" ".join(t for t, _ in combo), float(np.mean([c for _, c in combo]))))
        return candidates[:limit]

    def _select_reading(
        self, field_name: str, trocr_candidates: List[Tuple[str, float]], crop_for_easyocr: Optional[np.ndarray]
    ) -> Tuple[str, float, str, bool]:
        """Pick the first candidate that parses as a valid field value.

        Order: TrOCR n-best, then EasyOCR. If nothing validates, the top TrOCR
        reading is returned unparsed with reduced confidence and is_valid False.
        Returns (text, confidence, engine, parsed_ok).
        """
        parser = PARSERS[field_name]
        for text, conf in trocr_candidates:
            value = parser(text)
            if value:
                return value, conf, "trocr", True
        if field_name == "date":
            # Second pass: allow repairing a misread year digit ("2004" -> "2024")
            for text, conf in trocr_candidates:
                value = parse_date(text, repair_year=True)
                if value:
                    return value, conf * 0.9, "trocr", True

        easy_text, easy_conf = "", 0.0
        if crop_for_easyocr is not None and crop_for_easyocr.size:
            easy_text, easy_conf = self._run_easyocr(crop_for_easyocr, allowlist=EASYOCR_ALLOWLISTS.get(field_name))
            value = parser(easy_text) if easy_text else None
            if value:
                return value, easy_conf, "easyocr", True

        if trocr_candidates:
            text, conf = trocr_candidates[0]
            return self._clean_field_text(field_name, text), conf * 0.5, "trocr", False
        if easy_text:
            return self._clean_field_text(field_name, easy_text), easy_conf * 0.5, "easyocr", False
        return "", 0.0, "none", False

    def read_fields(self, canonical_bgr: np.ndarray, field_names: List[str]) -> Dict[str, Tuple[str, float, str, bool]]:
        """Locate and read several text fields, batching all TrOCR work into one call."""
        lines_by_field: Dict[str, List[np.ndarray]] = {}
        for f in field_names:
            lines = self.locate_handwriting(canonical_bgr, f)
            if not lines:
                # No ink found: fall back to the fixed handwriting band so faint
                # writing below the detection threshold still gets a chance.
                roi = CANONICAL_FIELD_ROIS[f]
                band = canonical_bgr[roi["y"]:roi["y"] + roi["h"], roi["x"]:roi["x"] + roi["w"]]
                lines = [band] if band.size else []
            lines_by_field[f] = lines

        flat = [(f, crop) for f in field_names for crop in lines_by_field[f]]
        try:
            beams = self._trocr_nbest([crop for _, crop in flat], fields=[f for f, _ in flat])
        except Exception as e:
            logger.warning(f"TrOCR batch decoding failed: {e}")
            beams = [[] for _ in flat]

        results = {}
        idx = 0
        for f in field_names:
            n_lines = len(lines_by_field[f])
            field_beams = beams[idx: idx + n_lines]
            idx += n_lines
            candidates = self._line_candidates(field_beams)
            joined = self._stack_lines(lines_by_field[f])
            results[f] = self._select_reading(f, candidates, joined)
        return results

    @staticmethod
    def _stack_lines(lines: List[np.ndarray]) -> Optional[np.ndarray]:
        """Lay line crops side by side so EasyOCR reads them as one text line."""
        if not lines:
            return None
        h = max(l.shape[0] for l in lines)
        padded = [cv2.copyMakeBorder(l, 0, h - l.shape[0], 0, 12, cv2.BORDER_REPLICATE) for l in lines]
        return np.hstack(padded)

    def _read_student_number(self, crop_bgr: np.ndarray) -> Tuple[str, float, str]:
        """Read the 8 written student number boxes (TrOCR, EasyOCR and per-cell)."""
        candidates: List[Tuple[str, float, str]] = []
        cleaned_grid = self._suppress_grid_lines(crop_bgr)

        if self._init_trocr():
            try:
                for variant in (crop_bgr, cleaned_grid):
                    t_text, t_conf = self._run_trocr(variant)
                    if t_text:
                        candidates.append((t_text, t_conf, "trocr"))
            except Exception as e:
                logger.warning(f"TrOCR candidate generation failed for student number: {e}")

        try:
            e_digits, ed_conf = self._run_easyocr(cleaned_grid, allowlist="0123456789")
            if e_digits:
                candidates.append((e_digits, ed_conf, "easyocr"))

            # Per-cell digit slicing
            w = crop_bgr.shape[1]
            cell_digits, cell_confs = [], []
            for i in range(8):
                cell = crop_bgr[:, int(i * w / 8):int((i + 1) * w / 8)]
                trimmed = cell[4:-4, 4:-4] if cell.shape[0] > 8 and cell.shape[1] > 8 else cell
                clahe_cell = self._enhance_crop(trimmed)
                r_c = self._get_easyocr().readtext(clahe_cell, allowlist="0123456789")
                if r_c and re.sub(r"\D", "", r_c[0][1]):
                    cell_digits.append(re.sub(r"\D", "", r_c[0][1])[-1])
                    cell_confs.append(float(r_c[0][2]))
                elif self._init_trocr():
                    tc, tc_conf = self._run_trocr(clahe_cell)
                    d = re.sub(r"\D", "", tc.translate(str.maketrans("oOlIisSbaAzZ", "001115568822")))
                    cell_digits.append(d[-1] if d else "")
                    cell_confs.append(tc_conf if d else 0.0)
                else:
                    cell_digits.append("")
                    cell_confs.append(0.0)
            if all(len(d) == 1 for d in cell_digits):
                candidates.append(("".join(cell_digits), float(np.mean(cell_confs)), "per_cell"))
        except Exception as e:
            logger.warning(f"EasyOCR candidate generation failed for student number: {e}")

        if not candidates:
            return "", 0.0, "none"
        scored = sorted(
            ((*self._score_student_number(t, c), eng) for t, c, eng in candidates),
            key=lambda item: item[0], reverse=True,
        )
        best_score, best_text, best_conf, best_engine = scored[0]
        if best_score < 0.15 or not best_text.strip():
            return "", 0.0, best_engine
        return best_text.strip(), round(best_conf, 4), best_engine

    def ocr_crop(self, crop_bgr: np.ndarray, field_name: str = "") -> Tuple[str, float, str]:
        """Read a single pre-cropped field. Returns (text, confidence, engine_used)."""
        if crop_bgr.size == 0:
            return "", 0.0, "none"
        if field_name == "student_number_written":
            return self._read_student_number(crop_bgr)
        if field_name in PARSERS:
            beams = self._trocr_nbest([crop_bgr], fields=[field_name])[0]
            text, conf, engine, _ = self._select_reading(field_name, beams, crop_bgr)
            return text, round(conf, 4), engine
        text, conf = self._run_easyocr(crop_bgr)
        return text, round(conf, 4), "easyocr" if text else "none"

    def validate_field(self, field_name: str, text: str) -> Tuple[bool, List[str]]:
        """Validate field formats and generate flags."""
        flags = []
        is_valid = True
        clean = text.strip()

        if not clean:
            flags.append("FIELD_EMPTY_OR_UNREADABLE")
            return False, flags

        if field_name == "student_number_written":
            digits_only = re.sub(r"\D", "", clean)
            if len(digits_only) != config.OCR_CONFIG["roll_digits"]:
                flags.append("INVALID_STUDENT_NUMBER_FORMAT")
                is_valid = False

        elif field_name == "module_code":
            clean_code = re.sub(r"[\s.-]+", "", clean.upper())
            if not re.fullmatch(r"[A-Z]{2,4}\d{4,5}", clean_code):
                flags.append("INVALID_MODULE_CODE_FORMAT")
                is_valid = False

        elif field_name == "seat_number":
            digits_only = re.sub(r"\D", "", clean)
            clean_stripped = re.sub(r"\s+", "", clean).upper()
            is_seat_fmt = bool(re.fullmatch(r"[A-Z]?[-.]?\d{1,4}", clean_stripped))
            if not digits_only or len(digits_only) > 4 or not is_seat_fmt:
                flags.append("INVALID_SEAT_NUMBER_FORMAT")
                is_valid = False

        elif field_name == "date":
            if not re.search(r"\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}|\d{1,2}\s+[A-Za-z]+(?:\s+\d{2,4})?|[A-Za-z]+\s+\d{2,4}", clean):
                flags.append("UNRECOGNIZED_DATE_FORMAT")
                is_valid = False

        return is_valid, flags

    def cross_check_roster(self, extracted_student_id: str, contributor: str = "") -> List[str]:
        """Cross-check extracted student number or contributor against roster.csv locally."""
        flags = []
        # Only check contributors when the local roster was actually loaded;
        # an absent roster must not produce a false NOT_FOUND flag.
        if contributor and self.roster_splits:
            if contributor not in self.roster_splits:
                flags.append("ROSTER_CONTRIBUTOR_NOT_FOUND")

        digits = re.sub(r"\D", "", extracted_student_id)
        if len(digits) != 8:
            flags.append("STUDENT_NUMBER_LENGTH_MISMATCH")

        return flags

    def extract_metadata(
        self,
        canonical_bgr: np.ndarray,
        template: Optional[Dict[str, Any]] = None,
        contributor: str = ""
    ) -> Dict[str, Any]:
        """Extract all handwritten metadata fields from a preprocessed canonical sheet."""
        if template is None:
            with open(config.TEMPLATE_PATH, "r", encoding="utf-8") as f:
                template = json.load(f)
        handwritten_rois = template.get("handwritten_rois", CANONICAL_FIELD_ROIS)

        text_fields = ["name", "module_code", "seat_number", "date"]
        readings = self.read_fields(canonical_bgr, text_fields)

        roi = handwritten_rois.get("student_number_written", CANONICAL_FIELD_ROIS["student_number_written"])
        sn_crop = canonical_bgr[roi["y"]:roi["y"] + roi["h"], roi["x"]:roi["x"] + roi["w"]]
        sn_text, sn_conf, sn_engine = self._read_student_number(sn_crop) if sn_crop.size else ("", 0.0, "none")
        readings["student_number_written"] = (sn_text, sn_conf, sn_engine, bool(sn_text))

        extracted_fields = {}
        all_flags: List[str] = []
        confidences = []
        for field_name in CANONICAL_FIELD_ROIS:
            text, conf, engine, _ = readings[field_name]
            is_valid, field_flags = self.validate_field(field_name, text)
            extracted_fields[field_name] = {
                "text": text,
                "extracted_text": text,
                "confidence": round(float(conf), 4),
                "engine": engine,
                "is_valid": is_valid,
                "flags": field_flags,
            }
            all_flags.extend(field_flags)
            confidences.append(float(conf))

        roster_flags = self.cross_check_roster(sn_text, contributor=contributor)
        all_flags.extend(roster_flags)
        mean_confidence = float(np.mean(confidences)) if confidences else 0.0

        return {
            "fields": extracted_fields,
            "overall_confidence": round(mean_confidence, 4),
            "flags": sorted(set(all_flags)),
            "student_id": extracted_fields["student_number_written"],
            "student_number_written": extracted_fields["student_number_written"],
            "name": extracted_fields["name"],
            "module_code": extracted_fields["module_code"],
            "seat_number": extracted_fields["seat_number"],
            "date": extracted_fields["date"],
            "exam_date": extracted_fields["date"],
        }
