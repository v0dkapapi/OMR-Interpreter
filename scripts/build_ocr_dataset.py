"""Build the handwriting crop dataset used to fine-tune TrOCR.

For every labelled image, the header fields are cropped with the same code the
app uses at inference (HandwritingOCR.field_image), so training and inference
see identical inputs. Targets come from labels.csv; dates use the label's ISO
form (YYYY-MM-DD), so the model learns to output normalized dates directly.

Output (git-ignored): cache/ocr_crops/<split>/<field>/<image-stem>.png and
cache/ocr_crops/manifest.csv. The manifest holds label text, including the
(fake) names, so it stays local; nothing is printed except counts.

Stress test contributors (p03, p04, p19) are never included.

Usage:
    python scripts/build_ocr_dataset.py                 # train + val/test splits
    python scripts/build_ocr_dataset.py --splits train
"""

import argparse
import csv
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402

FIELDS = ["name", "module_code", "seat_number", "date", "student_number_written"]
OUT_DIR = config.CACHE_DIR / "ocr_crops"

_STATE = {}


def _init():
    logging.disable(logging.WARNING)
    from src.handwriting_ocr import HandwritingOCR
    from src.preprocess import SheetPreprocessor
    _STATE["pre"] = SheetPreprocessor()
    _STATE["field_image"] = HandwritingOCR.field_image


def _process(job):
    import cv2
    path, row, split = job
    out = []
    try:
        canonical = _STATE["pre"].process(Path(path)).canonical_bgr
    except Exception:
        return out
    stem = Path(path).stem
    for field in FIELDS:
        target = row[field].strip()
        if not target:
            continue
        img = _STATE["field_image"](canonical, field)
        if img is None or img.size == 0:
            continue
        dest = OUT_DIR / split / field / f"{stem}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), img)
        out.append({
            "path": str(dest.relative_to(OUT_DIR)), "field": field, "target": target,
            "split": split, "contributor": row["contributor"], "form_id": row["form_id"],
            "kind": stem.rsplit("-", 1)[-1],
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build TrOCR fine-tuning crops.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    with open(config.VISION_LABELS_DIR / "labels.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["split"] in args.splits and r["contributor"] not in config.EXCLUDED_PERSONS]

    jobs = []
    for r in rows:
        person_dir = config.VISION_IMAGES_DIR / r["contributor"]
        if not person_dir.is_dir():
            continue
        for img in sorted(person_dir.glob(f'{r["form_id"]}-*')):
            if img.suffix.lower() in {".jpg", ".jpeg"}:
                jobs.append((str(img), r, r["split"]))

    print(f"{len(rows)} forms, {len(jobs)} images")
    t0 = time.time()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as pool:
        for i, recs in enumerate(pool.map(_process, jobs, chunksize=4), 1):
            records.extend(recs)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)} images ({time.time() - t0:.0f}s)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "field", "target", "split", "contributor", "form_id", "kind"])
        writer.writeheader()
        writer.writerows(records)
    by = {}
    for r in records:
        by.setdefault((r["split"], r["field"]), 0)
        by[(r["split"], r["field"])] += 1
    for k in sorted(by):
        print(f"  {k[0]:<6} {k[1]:<24} {by[k]}")
    print(f"{len(records)} crops in {time.time() - t0:.0f}s -> {OUT_DIR / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
