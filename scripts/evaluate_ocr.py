"""Benchmark handwritten field OCR against labels.csv.

Reports, per field, the exact-match rate and character error rate (CER) after
light normalization, broken down by capture condition. Only aggregate numbers
are printed: extracted or ground-truth names are NEVER printed or saved.

Forms are taken from the val/test split (never the stress test contributors),
spread across contributors.

Usage:
    python scripts/evaluate_ocr.py --forms 30
    python scripts/evaluate_ocr.py --forms 30 --kinds scan photo hard --no-trocr
"""

import argparse
import csv
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402

FIELDS = ["name", "module_code", "seat_number", "date", "student_number_written"]


def normalize(field: str, text: str) -> str:
    text = (text or "").strip()
    if field == "name":
        return re.sub(r"\s+", " ", text.lower())
    if field in ("module_code", "seat_number"):
        return re.sub(r"[\s\-.]", "", text.upper())
    if field == "student_number_written":
        return re.sub(r"\D", "", text)
    # date: compare as ISO YYYY-MM-DD; parse other spellings leniently so the
    # benchmark scores what was read, not how it was formatted
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text) or not text:
        return text
    try:
        from dateutil import parser as date_parser
        return date_parser.parse(text, dayfirst=True, fuzzy=True).strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        return text


def cer(pred: str, truth: str) -> float:
    """Levenshtein distance divided by ground-truth length."""
    if not truth:
        return 0.0 if not pred else 1.0
    prev = list(range(len(truth) + 1))
    for i, pc in enumerate(pred, 1):
        cur = [i]
        for j, tc in enumerate(truth, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (pc != tc)))
        prev = cur
    return min(1.0, prev[-1] / len(truth))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark handwritten field OCR.")
    parser.add_argument("--forms", type=int, default=30, help="Number of forms to evaluate.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many forms (round-robin order).")
    parser.add_argument("--kinds", nargs="+", default=["scan", "photo"], choices=["scan", "photo", "hard"])
    parser.add_argument("--no-trocr", action="store_true", help="EasyOCR only.")
    parser.add_argument("--json-out", type=Path, default=None, help="Save aggregate metrics as JSON.")
    args = parser.parse_args()

    logging.disable(logging.WARNING)
    from src.dataset import _round_robin_by_contributor
    from src.handwriting_ocr import HandwritingOCR
    from src.preprocess import SheetPreprocessor

    with open(config.VISION_LABELS_DIR / "labels.csv", encoding="utf-8") as f:
        rows = [
            r for r in csv.DictReader(f)
            if r["split"] in ("val", "test") and r["contributor"] not in config.EXCLUDED_PERSONS
        ]
    rows = _round_robin_by_contributor(rows)[args.offset: args.offset + args.forms]

    with open(config.TEMPLATE_PATH, encoding="utf-8") as f:
        template = json.load(f)
    pre = SheetPreprocessor()
    ocr = HandwritingOCR(use_trocr=not args.no_trocr)

    exact = defaultdict(lambda: defaultdict(list))
    errors = defaultdict(lambda: defaultdict(list))
    t0 = time.time()
    n_images = 0
    for r in rows:
        for kind in args.kinds:
            path = config.VISION_IMAGES_DIR / r["contributor"] / f'{r["form_id"]}-{kind}.jpg'
            if not path.exists():
                continue
            canonical = pre.process(path).canonical_bgr
            meta = ocr.extract_metadata(canonical, template=template)
            n_images += 1
            for field in FIELDS:
                pred = normalize(field, meta["fields"][field]["text"])
                truth = normalize(field, r[field])
                for bucket in (kind, "all"):
                    exact[field][bucket].append(pred == truth)
                    errors[field][bucket].append(cer(pred, truth))
            print(f"\r{n_images} images processed ({time.time() - t0:.0f}s)", end="", flush=True)
    print()

    buckets = args.kinds + ["all"]
    header = f"{'field':<24}" + "".join(f"{b + ' exact':>12}{b + ' CER':>10}" for b in buckets)
    print(header)
    print("-" * len(header))
    report = {}
    for field in FIELDS:
        line = f"{field:<24}"
        report[field] = {}
        for b in buckets:
            ex = 100.0 * sum(exact[field][b]) / max(1, len(exact[field][b]))
            ce = 100.0 * sum(errors[field][b]) / max(1, len(errors[field][b]))
            report[field][b] = {"exact_percent": round(ex, 1), "cer_percent": round(ce, 1)}
            line += f"{ex:>11.1f}%{ce:>9.1f}%"
        print(line)
    print(f"{n_images} images from {len(rows)} forms, {time.time() - t0:.0f}s "
          f"({'EasyOCR only' if args.no_trocr else 'TrOCR + EasyOCR'})")

    if args.json_out:
        args.json_out.write_text(json.dumps({"images": n_images, "fields": report}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
