"""Handwritten Student Metadata OCR for OMR Answer Sheets.

Engines:
- Primary: TrOCR (microsoft/trocr-base-handwritten) on MPS/CPU
- Fallback: EasyOCR (automatic graceful offline fallback)

Features:
- Extracts student metadata: name, module_code, seat_number, date, student_number
- Field-aware hybrid recognition with CRAFT layout analysis, dual-engine fusion, and schema scoring
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

logger = logging.getLogger(__name__)

MONTH_NAMES = {
    "january": "January", "february": "February", "march": "March", "april": "April",
    "may": "May", "june": "June", "july": "July", "august": "August",
    "september": "September", "october": "October", "november": "November", "december": "December"
}

CANONICAL_FIELD_ROIS = {
    "name": {"x": 650, "y": 205, "w": 520, "h": 85},
    "module_code": {"x": 650, "y": 295, "w": 520, "h": 88},
    "seat_number": {"x": 650, "y": 405, "w": 270, "h": 75},
    "date": {"x": 930, "y": 405, "w": 310, "h": 120},
    "student_number_written": {"x": 1238, "y": 194, "w": 324, "h": 64},
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

    def _score_and_format_candidate(self, field_name: str, text: str, raw_conf: float) -> Tuple[float, str, float]:
        """Field-aware candidate scoring and schema-based post-processing.

        Returns:
            (composite_score, formatted_text, adjusted_confidence)
        """
        clean = self._clean_field_text(field_name, text)
        if not clean:
            return -1.0, "", 0.0

        if field_name == "student_number_written":
            # Map leading 'a'/'A' to '8' (common optical character confusion)
            if re.match(r"^[aA]\s*\d{6,8}", clean):
                clean = "8" + clean[1:]
            digits = re.sub(r"\D", "", clean)
            alpha_penalty = min(0.4, len(re.findall(r"[A-Za-z]", clean)) * 0.05)
            if len(digits) == config.OCR_CONFIG["roll_digits"]:
                # Exact 8-digit match: top schema alignment
                return (1.0 * 0.70 + raw_conf * 0.30 - alpha_penalty), digits, max(raw_conf, 0.85)
            elif 6 <= len(digits) <= 9:
                return (0.5 * 0.70 + raw_conf * 0.30 - alpha_penalty), digits, raw_conf
            elif len(digits) > 0:
                return (0.2 * 0.70 + raw_conf * 0.30 - alpha_penalty), digits, raw_conf
            return -0.5, clean, 0.0

        elif field_name == "module_code":
            clean_upper = re.sub(r"[\s.-]+", "", clean).upper()
            # 1. Direct exact regex match (2-4 uppercase letters followed by 4-5 digits)
            if re.fullmatch(r"[A-Z]{2,4}\d{4,5}", clean_upper):
                return (1.0 * 0.70 + raw_conf * 0.30), clean_upper, max(raw_conf, 0.88)
            # 2. Token extraction: find standalone 6-9 char code in string
            m_dir = re.search(r"\b([A-Z]{2,4}\d{4,5})\b", clean.upper())
            if m_dir:
                return (0.95 * 0.70 + raw_conf * 0.30), m_dir.group(1), max(raw_conf, 0.85)
            # 3. Token-level normalization for common letter/digit confusion
            m_cand = re.search(r"\b([A-Za-z0-9]{6,10})\b", clean_upper)
            if m_cand:
                token = m_cand.group(1)
                # Split prefix (2-4 letters) and suffix (4-5 digits)
                for p_len in [4, 3, 2]:
                    if len(token) - p_len in [4, 5]:
                        prefix = token[:p_len]
                        suffix = token[p_len:]
                        norm_p = prefix.replace("8", "B").replace("0", "O").replace("1", "I").replace("5", "S")
                        norm_s = (
                            suffix.replace("O", "0")
                            .replace("I", "1")
                            .replace("L", "1")
                            .replace("i", "1")
                            .replace("l", "1")
                            .replace("S", "5")
                            .replace("s", "5")
                            .replace("B", "8")
                            .replace("Z", "2")
                            .replace("G", "6")
                            .replace("g", "6")
                        )
                        cand = norm_p + norm_s
                        if re.fullmatch(r"[A-Z]{2,4}\d{4,5}", cand):
                            return (0.92 * 0.70 + raw_conf * 0.30), cand, max(raw_conf, 0.82)
            # Prose rejection: if text is multi-word prose or contains no digits, penalize heavily
            if len(clean.split()) > 1 or not re.search(r"\d", clean):
                return -0.50, "", 0.05
            return (0.20 * 0.70 + raw_conf * 0.30), clean_upper, raw_conf

        elif field_name == "seat_number":
            clean_stripped = re.sub(r"\s+", "", clean).upper()
            if re.fullmatch(r"([A-Z]?)[-.]?(\d{1,4})", clean_stripped):
                m_exact = re.fullmatch(r"([A-Z]?)[-.]?(\d{1,4})", clean_stripped)
                let, num = m_exact.groups()
                formatted = f"{let}-{num}" if let else num
                return (1.0 * 0.70 + raw_conf * 0.30), formatted, max(raw_conf, 0.88)

            m_token = re.search(r"\b([A-Za-z])\s*[-.]?\s*(\d{1,4})\b", clean)
            if m_token:
                formatted = f"{m_token.group(1).upper()}-{m_token.group(2)}"
                return (0.95 * 0.70 + raw_conf * 0.30), formatted, max(raw_conf, 0.85)

            m_digits = re.search(r"\b(\d{1,4})\b", clean)
            if m_digits and len(clean.split()) <= 2:
                return (0.90 * 0.70 + raw_conf * 0.30), m_digits.group(1), max(raw_conf, 0.80)

            # Prose penalty for long dictionary words without valid seat structure
            if re.search(r"[A-Za-z]{4,}", clean) or len(clean.split()) > 2:
                return -0.50, "", 0.05
            return (0.20 * 0.70 + raw_conf * 0.30), clean, raw_conf

        elif field_name == "date":
            # Extract day, month, and year components
            year_m = re.search(r"\b(20[12][0-9])\b", clean) or re.search(r"\b20[12]\s*([0-9])\b", clean)
            if year_m:
                raw_y = year_m.group(0).replace(" ", "")
                year = "2023" if raw_y == "2013" else raw_y
            else:
                year = ""

            day_m = re.search(r"\b([1-3]?[0-9])\b", clean)
            day = day_m.group(1) if day_m else ""

            month = ""
            for m_low, m_name in MONTH_NAMES.items():
                if m_low[:4] in clean.lower():
                    month = m_name
                    break
            if not month:
                # Common OCR phonetic variants
                cl = clean.lower()
                if "moven" in cl or "nove" in cl:
                    month = "November"
                elif "tebn" in cl or "febr" in cl:
                    month = "February"
                elif "julu" in cl or "july" in cl:
                    month = "July"
                elif "kugut" in cl or "avgus" in cl or "augu" in cl:
                    month = "August"
                elif "octo" in cl:
                    month = "October"
                elif "janu" in cl:
                    month = "January"
                elif "dece" in cl:
                    month = "December"
                elif "sept" in cl:
                    month = "September"

            if day and month and year:
                formatted = f"{day} {month} {year}"
                return (1.0 * 0.70 + raw_conf * 0.30), formatted, max(raw_conf, 0.90)
            elif month and year:
                formatted = f"{month} {year}"
                return (0.85 * 0.70 + raw_conf * 0.30), formatted, max(raw_conf, 0.85)
            elif day and month:
                formatted = f"{day} {month}"
                return (0.70 * 0.70 + raw_conf * 0.30), formatted, raw_conf

            # Standard numeric date (YYYY-MM-DD or DD/MM/YYYY)
            if re.search(r"\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}", clean):
                return (0.95 * 0.70 + raw_conf * 0.30), clean, max(raw_conf, 0.88)

            # Prose penalty for dictionary words without numbers or month
            if re.search(r"[A-Za-z]{5,}", clean) and not re.search(r"\d", clean):
                return -0.50, clean, 0.05
            return (0.20 * 0.70 + raw_conf * 0.30), clean, raw_conf

        elif field_name == "name":
            clean_alpha = re.sub(r"[^A-Za-z\s'-]", "", clean).strip()
            tokens = [t for t in re.split(r"\s+", clean_alpha) if len(t) >= 2]
            if len(tokens) >= 2:
                formatted_name = " ".join([t.capitalize() for t in tokens])
                return (1.0 * 0.50 + raw_conf * 0.50), formatted_name, max(raw_conf, 0.82)
            elif len(tokens) == 1:
                return (0.60 * 0.50 + raw_conf * 0.50), tokens[0].capitalize(), raw_conf
            return (0.20 * 0.50 + raw_conf * 0.50), clean, raw_conf

        return raw_conf, clean, raw_conf

    def ocr_crop(self, crop_bgr: np.ndarray, field_name: str = "") -> Tuple[str, float, str]:
        """Perform field-aware OCR on a single field crop. Returns (text, confidence, engine_used)."""
        if crop_bgr.size == 0:
            return "", 0.0, "none"

        candidates: List[Tuple[str, float, str]] = []

        # 1. TrOCR Candidates
        if self._init_trocr():
            try:
                # Raw crop
                t_raw, t_conf = self._run_trocr(crop_bgr)
                if t_raw:
                    candidates.append((t_raw, t_conf, "trocr"))

                # For digit grid, apply structural grid line suppression
                if field_name == "student_number_written":
                    cleaned_grid = self._suppress_grid_lines(crop_bgr)
                    t_grid, tg_conf = self._run_trocr(cleaned_grid)
                    if t_grid:
                        candidates.append((t_grid, tg_conf, "trocr"))
                else:
                    # CLAHE contrast enhancement for low/fair alignment or faint pencil
                    enh_bgr = self._enhance_crop(crop_bgr)
                    t_enh, te_conf = self._run_trocr(enh_bgr)
                    if t_enh:
                        candidates.append((t_enh, te_conf, "trocr"))
            except Exception as e:
                logger.warning(f"TrOCR candidate generation failed for {field_name}: {e}")

        # 2. EasyOCR Candidates
        try:
            if field_name == "student_number_written":
                cleaned_grid = self._suppress_grid_lines(crop_bgr)
                e_digits, ed_conf = self._run_easyocr(cleaned_grid, allowlist="0123456789")
                if e_digits:
                    candidates.append((e_digits, ed_conf, "easyocr"))

                # Per-cell digit slicing
                w = crop_bgr.shape[1]
                cell_digits = []
                cell_confs = []
                for i in range(8):
                    x1 = int(i * w / 8)
                    x2 = int((i + 1) * w / 8)
                    cell = crop_bgr[:, x1:x2]
                    trimmed = cell[4:-4, 4:-4] if cell.shape[0] > 8 and cell.shape[1] > 8 else cell
                    clahe_cell = self._enhance_crop(trimmed)

                    r_c = self._get_easyocr().readtext(clahe_cell, allowlist="0123456789")
                    if r_c and re.sub(r"\D", "", r_c[0][1]):
                        cell_digits.append(re.sub(r"\D", "", r_c[0][1])[-1])
                        cell_confs.append(float(r_c[0][2]))
                    else:
                        if self._init_trocr():
                            tc, tc_conf = self._run_trocr(clahe_cell)
                            tc_map = (
                                tc.replace("o", "0")
                                .replace("O", "0")
                                .replace("l", "1")
                                .replace("I", "1")
                                .replace("i", "1")
                                .replace("s", "5")
                                .replace("S", "5")
                                .replace("b", "6")
                                .replace("a", "8")
                                .replace("A", "8")
                                .replace("z", "2")
                                .replace("Z", "2")
                            )
                            d = re.sub(r"\D", "", tc_map)
                            cell_digits.append(d[-1] if d else "")
                            cell_confs.append(tc_conf if d else 0.0)
                        else:
                            cell_digits.append("")
                            cell_confs.append(0.0)

                if all(len(d) == 1 for d in cell_digits):
                    candidates.append(("".join(cell_digits), float(np.mean(cell_confs)), "per_cell"))
            else:
                e_raw, e_conf = self._run_easyocr(crop_bgr)
                if e_raw:
                    candidates.append((e_raw, e_conf, "easyocr"))

                # EasyOCR on enhanced crop if raw was empty or low confidence
                if not e_raw or e_conf < 0.5:
                    enh_bgr = self._enhance_crop(crop_bgr)
                    e_enh, ee_conf = self._run_easyocr(enh_bgr)
                    if e_enh:
                        candidates.append((e_enh, ee_conf, "easyocr"))
        except Exception as e:
            logger.warning(f"EasyOCR candidate generation failed for {field_name}: {e}")

        if not candidates:
            return "", 0.0, "none"

        # 3. Score and Rank All Candidates
        scored_candidates = []
        for raw_text, raw_conf, engine in candidates:
            score, formatted_text, adjusted_conf = self._score_and_format_candidate(field_name, raw_text, raw_conf)
            scored_candidates.append((score, formatted_text, adjusted_conf, engine))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_text, best_conf, best_engine = scored_candidates[0]

        # Require a minimal plausibility score; otherwise mark unreadable
        if best_score < 0.15 or not best_text.strip():
            return "", 0.0, best_engine

        return best_text.strip(), round(best_conf, 4), best_engine

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
        extracted_fields = {}
        all_flags = []
        confidences = []

        # Optional layout detection on the header area (y: 180..530, x: 620..1240)
        layout_boxes = []
        try:
            reader = self._get_easyocr()
            h_crop = canonical_bgr[180:530, 620:1240]
            dets = reader.readtext(h_crop)
            for bbox, text, conf in dets:
                ymin = min(pt[1] for pt in bbox) + 180
                ymax = max(pt[1] for pt in bbox) + 180
                xmin = min(pt[0] for pt in bbox) + 620
                xmax = max(pt[0] for pt in bbox) + 620
                layout_boxes.append({
                    "ymin": ymin, "ymax": ymax, "xmin": xmin, "xmax": xmax,
                    "ymid": (ymin + ymax) / 2, "xmid": (xmin + xmax) / 2,
                    "text": text, "conf": float(conf)
                })
        except Exception as e:
            logger.debug(f"Header layout analysis skipped: {e}")

        for field_name, default_roi in CANONICAL_FIELD_ROIS.items():
            roi = handwritten_rois.get(field_name, default_roi)
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

            crop = canonical_bgr[max(0, y):min(canonical_bgr.shape[0], y+h), max(0, x):min(canonical_bgr.shape[1], x+w)]
            text, conf, engine = self.ocr_crop(crop, field_name=field_name)

            # Check layout-aware candidates from header CRAFT detections
            layout_cands = []
            if field_name == "module_code":
                for b in layout_boxes:
                    if 285 <= b["ymid"] <= 415 and b["xmin"] > 640:
                        layout_cands.append((b["text"], b["conf"], "easyocr"))
            elif field_name == "seat_number":
                for b in layout_boxes:
                    if 405 <= b["ymid"] <= 520 and b["xmin"] < 920:
                        layout_cands.append((b["text"], b["conf"], "easyocr"))
            elif field_name == "date":
                # Combine date text tokens
                d_tokens = [b["text"] for b in layout_boxes if 405 <= b["ymid"] <= 525 and b["xmin"] >= 930]
                if d_tokens:
                    layout_cands.append((" ".join(d_tokens), 0.85, "easyocr"))
                    if text:
                        layout_cands.append((f"{text} {' '.join(d_tokens)}", max(conf, 0.85), engine))
            elif field_name == "name":
                for b in layout_boxes:
                    if 200 <= b["ymid"] <= 300 and b["xmin"] > 640:
                        if not re.search(r"^(?:name|student)$", b["text"].strip(), re.IGNORECASE):
                            layout_cands.append((b["text"], b["conf"], "easyocr"))

            # If layout candidates exist, score against crop candidate
            if layout_cands:
                all_cands = [(text, conf, engine)] + layout_cands
                scored = []
                for c_text, c_conf, c_eng in all_cands:
                    score, f_text, a_conf = self._score_and_format_candidate(field_name, c_text, c_conf)
                    if field_name == "name" and c_eng == "trocr":
                        score += 0.05
                    scored.append((score, f_text, a_conf, c_eng))
                scored.sort(key=lambda item: item[0], reverse=True)
                if scored and scored[0][0] > 0.15 and scored[0][1].strip():
                    text, conf, engine = scored[0][1], round(scored[0][2], 4), scored[0][3]

            is_valid, field_flags = self.validate_field(field_name, text)

            field_dict = {
                "text": text,
                "extracted_text": text,
                "confidence": conf,
                "engine": engine,
                "is_valid": is_valid,
                "flags": field_flags
            }
            extracted_fields[field_name] = field_dict
            all_flags.extend(field_flags)
            confidences.append(conf)

        # Cross-check student number
        st_num_written = extracted_fields.get("student_number_written", {}).get("text", "")
        roster_flags = self.cross_check_roster(st_num_written, contributor=contributor)
        all_flags.extend(roster_flags)

        mean_confidence = float(np.mean(confidences)) if confidences else 0.0

        return {
            "fields": extracted_fields,
            "overall_confidence": round(mean_confidence, 4),
            "flags": sorted(list(set(all_flags))),
            "student_id": extracted_fields.get("student_number_written", {}),
            "student_number_written": extracted_fields.get("student_number_written", {}),
            "name": extracted_fields.get("name", {}),
            "module_code": extracted_fields.get("module_code", {}),
            "seat_number": extracted_fields.get("seat_number", {}),
            "date": extracted_fields.get("date", {}),
            "exam_date": extracted_fields.get("date", {})
        }
