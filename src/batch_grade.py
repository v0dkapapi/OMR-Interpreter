"""Batch grading CLI for folders of OMR answer sheets.

Runs the same pipeline as POST /api/grade (preprocess -> shared bubble fusion
-> deterministic grader) over every image in a folder, using several worker
processes, and writes:

- <out>/sheets/<stem>.json : full per-sheet grading report
- <out>/batch_summary.csv  : one row per sheet (scores, counts, flags)

Handwriting OCR is off by default for speed. With --ocr, extracted student
numbers are included, but the handwritten name field is always redacted from
batch outputs.

Usage:
    python -m src.batch_grade path/to/sheets --out outputs/batch --workers 4
    python -m src.batch_grade path/to/sheets --answer-key answer_key.json --negative-marks 0.25
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402

logger = logging.getLogger("omr.batch")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
REDACTED_FIELDS = ("name",)

SUMMARY_COLUMNS = [
    "sheet", "status", "marks_obtained", "total_marks", "percentage",
    "correct", "incorrect", "multi_mark", "unanswered", "review",
    "alignment_confidence", "alignment_method", "alignment_flags",
    "student_number", "seconds", "error",
]

# Per-process components, created once by the worker initializer.
_WORKER: Dict[str, Any] = {}


def load_answer_key(path: Optional[Path]) -> Dict[str, str]:
    """Load an answer key from JSON ({"answers": {...}} or flat) or CSV (question,answer)."""
    path = path or config.DEFAULT_ANSWER_KEY_PATH
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        answers = data.get("answers", data)
        return {str(k): str(v).strip().upper() for k, v in answers.items() if str(k).isdigit()}
    key = {}
    for row in csv.reader(text.splitlines()):
        if len(row) >= 2:
            q = "".join(filter(str.isdigit, row[0]))
            a = row[1].strip().upper()
            if q and a in {"A", "B", "C", "D", "E"}:
                key[q] = a
    return key


def _init_worker(use_ocr: bool, device_name: str) -> None:
    """Build pipeline components once per worker process."""
    import torch
    from src.fusion import BubbleReader, load_bubble_model
    from src.grader import OMRGrader
    from src.preprocess import SheetPreprocessor

    torch.set_num_threads(1)
    logging.basicConfig(level=logging.WARNING)
    device = torch.device(device_name)

    with open(config.TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = json.load(f)

    _WORKER["template"] = template
    _WORKER["preprocessor"] = SheetPreprocessor(template_path=config.TEMPLATE_PATH)
    _WORKER["reader"] = BubbleReader(
        template.get("bubble_rois", {}), model=load_bubble_model(device), device=device
    )
    _WORKER["grader"] = OMRGrader()
    _WORKER["ocr"] = None
    if use_ocr:
        from src.handwriting_ocr import HandwritingOCR
        _WORKER["ocr"] = HandwritingOCR(use_trocr=False, device=device)


def _redact(student_details: Dict[str, Any]) -> Dict[str, Any]:
    details = json.loads(json.dumps(student_details))
    fields = details.get("fields", {})
    for name in REDACTED_FIELDS:
        if name in fields:
            fields[name]["text"] = "[REDACTED]"
    return details


def grade_one(image_path: str, answer_key: Dict[str, str], scoring: Dict[str, float], out_dir: str) -> Dict[str, Any]:
    """Grade a single sheet inside a worker and write its JSON report."""
    from src.grader import ScoringConfig

    t0 = time.time()
    path = Path(image_path)
    row: Dict[str, Any] = {c: "" for c in SUMMARY_COLUMNS}
    row["sheet"] = path.name
    try:
        template = _WORKER["template"]
        res = _WORKER["preprocessor"].process(path)
        predictions = _WORKER["reader"].read(res.normalized_gray)

        student_details: Dict[str, Any] = {}
        if _WORKER["ocr"] is not None:
            student_details = _redact(_WORKER["ocr"].extract_metadata(res.canonical_bgr, template=template))

        result = _WORKER["grader"].grade(
            question_predictions=predictions,
            answer_key=answer_key,
            scoring_config=ScoringConfig(total_questions=len(template.get("bubble_rois", {})), **scoring),
            student_details=student_details,
            alignment_confidence=res.confidence,
            alignment_flags=res.flags,
        )

        report = result.to_dict()
        report["sheet"] = path.name
        report["alignment_method"] = res.method_used
        report["bubble_reader_mode"] = _WORKER["reader"].mode
        (Path(out_dir) / "sheets" / f"{path.stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        s = result.summary
        row.update({
            "status": "OK",
            "marks_obtained": s.marks_obtained,
            "total_marks": s.total_marks,
            "percentage": s.percentage,
            "correct": s.correct_count,
            "incorrect": s.incorrect_count,
            "multi_mark": s.multi_mark_count,
            "unanswered": s.unanswered_count,
            "review": s.review_count,
            "alignment_confidence": round(res.confidence, 4),
            "alignment_method": res.method_used,
            "alignment_flags": ";".join(res.flags),
            "student_number": student_details.get("fields", {}).get("student_number_written", {}).get("text", ""),
        })
    except Exception as e:  # one bad sheet must never stop the batch
        row["status"] = "ERROR"
        row["error"] = f"{type(e).__name__}: {e}"
    row["seconds"] = round(time.time() - t0, 2)
    return row


def discover_images(input_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in input_dir.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def run_batch(
    input_dir: Path,
    out_dir: Path,
    answer_key: Dict[str, str],
    scoring: Dict[str, float],
    workers: int = 2,
    use_ocr: bool = False,
    recursive: bool = False,
    device_name: str = "cpu",
) -> List[Dict[str, Any]]:
    """Grade every image in input_dir; returns the summary rows."""
    images = discover_images(input_dir, recursive)
    (out_dir / "sheets").mkdir(parents=True, exist_ok=True)
    if not images:
        logger.warning(f"No images found in {input_dir}")
        return []

    rows: List[Dict[str, Any]] = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(use_ocr, device_name)) as pool:
        futures = [pool.submit(grade_one, str(p), answer_key, scoring, str(out_dir)) for p in images]
        for done, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            elapsed = time.time() - start
            eta = elapsed / done * (len(images) - done)
            print(
                f"[{done}/{len(images)}] {row['sheet']}: {row['status']} "
                f"{row['percentage'] if row['status'] == 'OK' else row['error']} | "
                f"elapsed {elapsed:.1f}s, eta {eta:.1f}s",
                flush=True,
            )

    rows.sort(key=lambda r: r["sheet"])
    with open(out_dir / "batch_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(r["status"] == "OK" for r in rows)
    flagged = sum(1 for r in rows if r["status"] == "OK" and (r["review"] or r["alignment_flags"]))
    print(
        f"Graded {ok}/{len(rows)} sheets in {time.time() - start:.1f}s. "
        f"{flagged} need proctor attention. Summary: {out_dir / 'batch_summary.csv'}"
    )
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Grade a folder of OMR answer sheets.")
    parser.add_argument("input_dir", type=Path, help="Folder containing sheet images.")
    parser.add_argument("--out", type=Path, default=config.OUTPUTS_DIR / "batch", help="Output folder.")
    parser.add_argument("--answer-key", type=Path, default=None, help="JSON or CSV answer key (default: answer_key.json).")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)),
                        help="Worker processes (default keeps a fanless laptop cool).")
    parser.add_argument("--marks-per-correct", type=float, default=1.0)
    parser.add_argument("--negative-marks", type=float, default=0.0)
    parser.add_argument("--multi-mark-marks", type=float, default=0.0)
    parser.add_argument("--ocr", action="store_true", help="Extract handwritten fields (slower; name is redacted).")
    parser.add_argument("--recursive", action="store_true", help="Include images in subfolders.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"],
                        help="Torch device per worker (cpu is safest with multiple workers).")
    args = parser.parse_args(argv)

    if not args.input_dir.is_dir():
        parser.error(f"{args.input_dir} is not a directory")

    rows = run_batch(
        input_dir=args.input_dir,
        out_dir=args.out,
        answer_key=load_answer_key(args.answer_key),
        scoring={
            "marks_per_correct": args.marks_per_correct,
            "negative_marks": args.negative_marks,
            "multi_mark_marks": args.multi_mark_marks,
        },
        workers=args.workers,
        use_ocr=args.ocr,
        recursive=args.recursive,
        device_name=args.device,
    )
    return 0 if rows and all(r["status"] == "OK" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
