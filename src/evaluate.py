"""Comprehensive Stress Test Evaluation Suite for OMR Interpreter.

Runs the complete OMR interpretation pipeline on all 90 stress test images
(p03, p04, p19 across scan, photo, and hard conditions).

Reports:
- Answer-level accuracy per condition (scan, photo, hard)
- Answer-level accuracy per person (p03, p04, p19)
- Multi-mark precision, recall, and F1 score where labels exist
- Sheet alignment quality and LOW_ALIGNMENT_CONFIDENCE rates
- Saves visual failure overlay artifacts under outputs/failures/
  (Flagged bubbles in red, conforming in neutral; no roster.csv names)
- Saves evaluation_report.json and evaluation_summary.txt under outputs/
"""

import os
import sys
import re
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np

# Ensure MPS fallback before torch import
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config
from src.preprocess import SheetPreprocessor, PreprocessResult
from src.fusion import BubbleReader, answer_symbol, load_bubble_model
from src.handwriting_ocr import HandwritingOCR
from src.grader import OMRGrader, ScoringConfig, QuestionStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("omr.evaluate")

# Tolerant regex for stress test image naming
STRESS_FILENAME_REGEX = re.compile(
    r"^(?P<person>p\d+)[-_](?:f?(?P<form>\d+))[-_](?P<kind>scan\d*|photo\d*|hardphoto\d*|hard\d*)\.(?P<ext>jpg|jpeg)$",
    re.IGNORECASE
)


def parse_stress_filename(path: Path) -> Optional[Tuple[str, str, str]]:
    """Parse person, form_id, and condition from stress test filename.

    Returns:
        (person, form_id, condition) e.g. ('p03', 'p03-f01', 'scan')
    """
    m = STRESS_FILENAME_REGEX.match(path.name)
    if not m:
        return None
    person = m.group("person").lower()
    form_num = int(m.group("form"))
    form_id = f"{person}-f{form_num:02d}"
    kind = m.group("kind").lower()

    if "hard" in kind:
        condition = "hard"
    elif "photo" in kind:
        condition = "photo"
    else:
        condition = "scan"

    return person, form_id, condition


def generate_failure_overlay(
    canonical_bgr: np.ndarray,
    bubble_rois: Dict[str, Any],
    pred_answers: Dict[int, str],
    gt_answers: Dict[str, str],
    pred_options: Dict[int, List[str]]
) -> np.ndarray:
    """Generate annotated failure overlay image.

    Bubbles flagged or mismatched are highlighted in Red (#DC2626: BGR 38, 38, 220).
    Conforming bubbles are marked in neutral dark gray (BGR 75, 85, 99) or faint gray.
    Never prints student names.
    """
    overlay = canonical_bgr.copy()

    for q_str, opts in bubble_rois.items():
        q_num = int(q_str)
        gt_ans = gt_answers.get(q_str, "-")
        pred_ans = pred_answers.get(q_num, "-")
        is_mismatch = (pred_ans != gt_ans)
        chosen_opts = pred_options.get(q_num, [])

        for opt, roi in opts.items():
            cx = int(roi["x"] + roi["w"] / 2)
            cy = int(roi["y"] + roi["h"] / 2)
            r = int(min(roi["w"], roi["h"]) / 2)

            is_chosen = (opt in chosen_opts)
            is_gt_bubble = (opt == gt_ans)

            if is_chosen:
                if is_mismatch:
                    # Red circle and inner fill for incorrect or multi-mark bubble
                    cv2.circle(overlay, (cx, cy), r + 2, (38, 38, 220), 3)
                    cv2.circle(overlay, (cx, cy), r - 2, (38, 38, 220), -1)
                else:
                    # Conforming correct selection in neutral gray
                    cv2.circle(overlay, (cx, cy), r + 2, (75, 85, 99), 3)
                    cv2.circle(overlay, (cx, cy), r - 2, (107, 114, 128), -1)
            else:
                if is_gt_bubble and is_mismatch:
                    # Expected bubble that was missed (dashed blue/red outline)
                    cv2.circle(overlay, (cx, cy), r + 1, (200, 100, 0), 2)
                elif is_mismatch:
                    # Faint red ring for options of a mismatched question
                    cv2.circle(overlay, (cx, cy), r, (120, 120, 220), 1)
                else:
                    # Neutral faint gray circle
                    cv2.circle(overlay, (cx, cy), r, (180, 180, 180), 1)

    return overlay


class StressEvaluator:
    """Evaluates the full OMR pipeline across stress test sheets."""

    def __init__(
        self,
        data_root: Path = config.DATA_ROOT,
        failures_dir: Path = config.OUTPUTS_DIR / "failures",
        use_ocr: bool = True
    ):
        self.data_root = data_root
        self.stress_dir = data_root / "stress_test"
        self.labels_json_dir = data_root / "vision_ds" / "labels" / "json"
        self.failures_dir = failures_dir
        self.failures_dir.mkdir(parents=True, exist_ok=True)
        self.use_ocr = use_ocr

        # Load Template
        with open(config.TEMPLATE_PATH, "r", encoding="utf-8") as f:
            self.template = json.load(f)
        self.bubble_rois = self.template.get("bubble_rois", {})

        # Load Components
        self.preprocessor = SheetPreprocessor(template_path=config.TEMPLATE_PATH)
        self.device = config.get_device()

        # Bubble reader (same fusion code path as the live API)
        model = load_bubble_model(self.device)
        if model is None:
            logger.warning("Bubble model weights not found. Evaluating fill-margin-only baseline.")
        self.bubble_reader = BubbleReader(self.bubble_rois, model=model, device=self.device)
        logger.info(f"Bubble reader mode: {self.bubble_reader.mode}")

        # OCR Engine (Fast fallback for evaluation speed)
        self.ocr_engine = HandwritingOCR(use_trocr=False) if use_ocr else None
        self.grader = OMRGrader()

    def discover_images(
        self,
        person_filter: Optional[str] = None,
        condition_filter: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Tuple[Path, str, str, str]]:
        """Collect stress test images matching filters."""
        all_files = list(self.stress_dir.glob("**/*.*"))
        candidates = []

        for p in all_files:
            if p.suffix.lower() not in [".jpg", ".jpeg"]:
                continue
            parsed = parse_stress_filename(p)
            if not parsed:
                continue
            person, form_id, condition = parsed

            if person_filter and person != person_filter.lower():
                continue
            if condition_filter and condition != condition_filter.lower():
                continue

            candidates.append((p, person, form_id, condition))

        candidates.sort(key=lambda x: (x[1], x[2], x[3]))
        if limit:
            candidates = candidates[:limit]
        return candidates

    def load_ground_truth(self, form_id: str) -> Optional[Dict[str, Any]]:
        """Load ground truth JSON for a given form_id."""
        json_path = self.labels_json_dir / f"{form_id}.json"
        if not json_path.exists():
            return None
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read ground truth {json_path}: {e}")
            return None

    def evaluate_sheet_bubbles(
        self,
        norm_gray: np.ndarray
    ) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, str]]:
        """Run batched bubble inference and fusion for all questions on one sheet."""
        question_preds = self.bubble_reader.read(norm_gray)
        single_answer_preds = {q: answer_symbol(p) for q, p in question_preds.items()}
        return question_preds, single_answer_preds

    def run(
        self,
        person_filter: Optional[str] = None,
        condition_filter: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute evaluation across the stress test dataset."""
        candidates = self.discover_images(person_filter, condition_filter, limit)
        total_sheets = len(candidates)
        logger.info(f"Discovered {total_sheets} stress test images to evaluate.")

        # Metric Accumulators
        per_condition: Dict[str, Dict[str, Any]] = {
            "scan": {"total_q": 0, "correct_q": 0, "tp": 0, "fp": 0, "fn": 0, "conf_sum": 0.0, "sheets": 0, "low_conf": 0, "review_q": 0},
            "photo": {"total_q": 0, "correct_q": 0, "tp": 0, "fp": 0, "fn": 0, "conf_sum": 0.0, "sheets": 0, "low_conf": 0, "review_q": 0},
            "hard": {"total_q": 0, "correct_q": 0, "tp": 0, "fp": 0, "fn": 0, "conf_sum": 0.0, "sheets": 0, "low_conf": 0, "review_q": 0},
        }

        per_person: Dict[str, Dict[str, Any]] = {
            "p03": {"total_q": 0, "correct_q": 0, "tp": 0, "fp": 0, "fn": 0, "conf_sum": 0.0, "sheets": 0, "low_conf": 0, "review_q": 0},
            "p04": {"total_q": 0, "correct_q": 0, "tp": 0, "fp": 0, "fn": 0, "conf_sum": 0.0, "sheets": 0, "low_conf": 0, "review_q": 0},
            "p19": {"total_q": 0, "correct_q": 0, "tp": 0, "fp": 0, "fn": 0, "conf_sum": 0.0, "sheets": 0, "low_conf": 0, "review_q": 0},
        }

        overall = {"total_q": 0, "correct_q": 0, "tp": 0, "fp": 0, "fn": 0, "sheets": 0, "review_q": 0, "conf_sum": 0.0, "low_conf": 0}
        failure_records = []

        start_time = time.time()

        for idx, (img_path, person, form_id, condition) in enumerate(candidates, 1):
            sheet_name = img_path.name
            t0 = time.time()

            # 1. Preprocess
            res: PreprocessResult = self.preprocessor.process(img_path)
            align_conf = res.confidence
            is_low_conf = "LOW_ALIGNMENT_CONFIDENCE" in res.flags

            # 2. Bubble Inference
            q_preds, single_preds = self.evaluate_sheet_bubbles(res.normalized_gray)

            # 2b. Route through the deterministic grader to count REVIEW questions,
            # exactly as the live API would flag them for a proctor.
            graded = self.grader.grade(
                q_preds,
                answer_key=None,
                scoring_config=ScoringConfig(total_questions=len(self.bubble_rois)),
                alignment_confidence=res.confidence,
                alignment_flags=res.flags,
            )
            sheet_review = graded.summary.review_count

            # 3. OCR (if enabled)
            ocr_meta = {}
            if self.ocr_engine:
                ocr_meta = self.ocr_engine.extract_metadata(
                    res.canonical_bgr, template=self.template, contributor=person
                )

            # 4. Compare with Ground Truth
            gt_data = self.load_ground_truth(form_id)
            if not gt_data:
                logger.warning(f"No ground truth found for {form_id}, skipping metric computation.")
                continue

            raw_gt_answers = gt_data.get("raw", {}).get("answers", {})

            sheet_total = 0
            sheet_correct = 0
            mismatched_questions = []

            for q_num in range(1, 121):
                q_str = str(q_num)
                gt_ans = raw_gt_answers.get(q_str, "-")
                pred_ans = single_preds.get(q_num, "-")

                sheet_total += 1
                is_correct = (pred_ans == gt_ans)

                if is_correct:
                    sheet_correct += 1
                else:
                    mismatched_questions.append({
                        "question": q_num,
                        "ground_truth": gt_ans,
                        "predicted": pred_ans,
                        "filled_options": q_preds[q_num]["filled_options"],
                        "ambiguous_options": q_preds[q_num]["ambiguous_options"]
                    })

                # Multi-mark detection metrics (GT Positive: gt_ans == '?')
                is_gt_multimark = (gt_ans == "?")
                is_pred_multimark = (pred_ans == "?")

                tp = int(is_gt_multimark and is_pred_multimark)
                fp = int(not is_gt_multimark and is_pred_multimark)
                fn = int(is_gt_multimark and not is_pred_multimark)

                # Accumulate condition
                per_condition[condition]["total_q"] += 1
                per_condition[condition]["correct_q"] += int(is_correct)
                per_condition[condition]["tp"] += tp
                per_condition[condition]["fp"] += fp
                per_condition[condition]["fn"] += fn

                # Accumulate person
                per_person[person]["total_q"] += 1
                per_person[person]["correct_q"] += int(is_correct)
                per_person[person]["tp"] += tp
                per_person[person]["fp"] += fp
                per_person[person]["fn"] += fn

                # Overall
                overall["total_q"] += 1
                overall["correct_q"] += int(is_correct)
                overall["tp"] += tp
                overall["fp"] += fp
                overall["fn"] += fn

            per_condition[condition]["sheets"] += 1
            per_condition[condition]["review_q"] += sheet_review
            per_person[person]["review_q"] += sheet_review
            overall["review_q"] += sheet_review
            per_condition[condition]["conf_sum"] += align_conf
            if is_low_conf:
                per_condition[condition]["low_conf"] += 1

            per_person[person]["sheets"] += 1
            per_person[person]["conf_sum"] += align_conf
            if is_low_conf:
                per_person[person]["low_conf"] += 1

            overall["sheets"] += 1
            overall["conf_sum"] += align_conf
            overall["low_conf"] += int(is_low_conf)

            sheet_acc = (sheet_correct / sheet_total) * 100.0 if sheet_total > 0 else 0.0
            dt = time.time() - t0
            logger.info(
                f"[{idx:02d}/{total_sheets:02d}] {sheet_name} ({person}/{condition}): "
                f"Acc={sheet_acc:.1f}% ({sheet_correct}/{sheet_total}) | AlignConf={align_conf:.2f} ({res.method_used}) | {dt:.2f}s"
            )

            # Save Failure Cases (below 85% accuracy or flagged alignment)
            if sheet_acc < 85.0 or is_low_conf:
                overlay_img = generate_failure_overlay(
                    res.canonical_bgr,
                    self.bubble_rois,
                    single_preds,
                    raw_gt_answers,
                    {q: q_preds[q]["filled_options"] for q in q_preds}
                )
                failure_stem = Path(sheet_name).stem
                overlay_path = self.failures_dir / f"{failure_stem}_overlay.jpg"
                cv2.imwrite(str(overlay_path), overlay_img)

                fail_json_path = self.failures_dir / f"{failure_stem}_fail.json"
                fail_record = {
                    "sheet": sheet_name,
                    "person": person,
                    "form_id": form_id,
                    "condition": condition,
                    "accuracy": round(sheet_acc, 2),
                    "correct": sheet_correct,
                    "total": sheet_total,
                    "alignment_confidence": round(align_conf, 4),
                    "alignment_method": res.method_used,
                    "flags": res.flags,
                    "mismatched_questions_count": len(mismatched_questions),
                    "mismatched_questions": mismatched_questions
                }
                with open(fail_json_path, "w", encoding="utf-8") as f:
                    json.dump(fail_record, f, indent=2)
                failure_records.append(fail_record)

        total_elapsed = time.time() - start_time

        # Compute Aggregated Rates
        def finalize_metrics(d: Dict[str, Any]) -> Dict[str, Any]:
            tot_q = d.get("total_q", 0)
            cor_q = d.get("correct_q", 0)
            acc = (cor_q / float(tot_q)) * 100.0 if tot_q > 0 else 0.0

            tp = d.get("tp", 0)
            fp = d.get("fp", 0)
            fn = d.get("fn", 0)

            prec = (tp / float(tp + fp)) * 100.0 if (tp + fp) > 0 else 100.0
            rec = (tp / float(tp + fn)) * 100.0 if (tp + fn) > 0 else 100.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            sheets = d.get("sheets", 0)
            conf_avg = (d.get("conf_sum", 0.0) / float(sheets)) if sheets > 0 else 0.0
            low_conf_pct = (d.get("low_conf", 0) / float(sheets)) * 100.0 if sheets > 0 else 0.0

            return {
                "total_questions": tot_q,
                "correct_questions": cor_q,
                "accuracy_percent": round(acc, 2),
                "multi_mark_precision": round(prec, 2),
                "multi_mark_recall": round(rec, 2),
                "multi_mark_f1": round(f1, 2),
                "mean_alignment_confidence": round(conf_avg, 3),
                "low_alignment_rate_percent": round(low_conf_pct, 1),
                "review_questions": d.get("review_q", 0),
                "review_rate_percent": round((d.get("review_q", 0) / float(tot_q)) * 100.0, 2) if tot_q > 0 else 0.0,
                "sheets_evaluated": sheets
            }

        report = {
            "evaluation_time_seconds": round(total_elapsed, 2),
            "total_sheets_evaluated": overall["sheets"],
            "overall": finalize_metrics(overall),
            "by_condition": {c: finalize_metrics(per_condition[c]) for c in per_condition},
            "by_person": {p: finalize_metrics(per_person[p]) for p in per_person},
            "failure_cases_saved": len(failure_records)
        }

        # Save Report JSON
        report_json_path = config.OUTPUTS_DIR / "evaluation_report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Generate Plain Text Summary
        summary_txt_path = config.OUTPUTS_DIR / "evaluation_summary.txt"
        summary_txt = self.format_summary_text(report)
        with open(summary_txt_path, "w", encoding="utf-8") as f:
            f.write(summary_txt)

        print("\n" + summary_txt)
        return report

    def format_summary_text(self, report: Dict[str, Any]) -> str:
        """Create clean Markdown summary table with zero emojis."""
        lines = [
            "================================================================================",
            "                   OMR INTERPRETER STRESS TEST EVALUATION REPORT                ",
            "================================================================================",
            f"Total Sheets Evaluated: {report['total_sheets_evaluated']} | Elapsed Time: {report['evaluation_time_seconds']:.1f}s",
            f"Failures Saved Under: outputs/failures/ ({report['failure_cases_saved']} cases)",
            "",
            "1. OVERALL PERFORMANCE:",
            f"   - Answer-Level Accuracy:    {report['overall']['accuracy_percent']}% ({report['overall']['correct_questions']}/{report['overall']['total_questions']})",
            f"   - Multi-Mark Precision:     {report['overall']['multi_mark_precision']}%",
            f"   - Multi-Mark Recall:        {report['overall']['multi_mark_recall']}%",
            f"   - Multi-Mark F1 Score:      {report['overall']['multi_mark_f1']}%",
            f"   - Routed to REVIEW:         {report['overall']['review_rate_percent']}% ({report['overall']['review_questions']} questions)",
            "",
            "2. PERFORMANCE BY CAPTURE CONDITION:",
            "   Condition | Sheets | Accuracy | Multi-Mark Prec | Multi-Mark Rec | F1     | Align Conf",
            "   ----------+--------+----------+-----------------+----------------+--------+-----------",
        ]

        for c, m in report["by_condition"].items():
            lines.append(
                f"   {c:<9} | {m['sheets_evaluated']:<6} | {m['accuracy_percent']:>7.2f}% | "
                f"{m['multi_mark_precision']:>14.1f}% | {m['multi_mark_recall']:>13.1f}% | {m['multi_mark_f1']:>5.1f}% | "
                f"{m['mean_alignment_confidence']:>9.3f}"
            )

        lines.extend([
            "",
            "3. PERFORMANCE BY PERSON:",
            "   Person    | Sheets | Accuracy | Multi-Mark Prec | Multi-Mark Rec | F1     | Align Conf",
            "   ----------+--------+----------+-----------------+----------------+--------+-----------",
        ])

        for p, m in report["by_person"].items():
            lines.append(
                f"   {p:<9} | {m['sheets_evaluated']:<6} | {m['accuracy_percent']:>7.2f}% | "
                f"{m['multi_mark_precision']:>14.1f}% | {m['multi_mark_recall']:>13.1f}% | {m['multi_mark_f1']:>5.1f}% | "
                f"{m['mean_alignment_confidence']:>9.3f}"
            )

        lines.append("================================================================================")
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate OMR interpreter on stress_test dataset.")
    parser.add_argument("--person", type=str, choices=["p03", "p04", "p19"], help="Filter by person.")
    parser.add_argument("--condition", type=str, choices=["scan", "photo", "hard"], help="Filter by condition.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sheets to evaluate.")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR to speed up evaluation.")
    args = parser.parse_args()

    evaluator = StressEvaluator(use_ocr=not args.no_ocr)
    evaluator.run(
        person_filter=args.person,
        condition_filter=args.condition,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
