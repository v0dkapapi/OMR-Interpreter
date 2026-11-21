"""Deterministic grading engine for OMR answer sheets.

Rules implemented:
- One option filled and correct -> award marks_per_correct (e.g. +1.0)
- One option filled and wrong -> award configured negative or zero marking (e.g. -0.25 or 0.0)
- 2+ options filled on same question -> MULTIPLE_MARK, scored per rule (default 0.0), never attributed to single option, never crashes
- No options filled -> UNANSWERED, scored per rule (default 0.0)
- Ambiguous bubbles or model vs fill-ratio disagreements -> REVIEW, routed to flagged issues panel
- Deterministic calculation of total score, marks obtained, percentage, and summary counts
- CSV and JSON export serialization helpers
"""

import csv
import io
import json
import dataclasses
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Dict, Any, List, Union

import numpy as np
import config


class QuestionStatus(str, Enum):
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    MULTIPLE_MARK = "MULTIPLE_MARK"
    UNANSWERED = "UNANSWERED"
    REVIEW = "REVIEW"


@dataclass
class ScoringConfig:
    """Scoring rules configuration."""
    marks_per_correct: float = 1.0
    negative_marks: float = 0.0       # Penalty magnitude (e.g. 0.25 or 0.0)
    multi_mark_marks: float = 0.0     # Default 0.0
    unanswered_marks: float = 0.0     # Default 0.0
    total_questions: int = 120

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "ScoringConfig":
        if not data:
            return cls()
        return cls(
            marks_per_correct=float(data.get("marks_per_correct", 1.0)),
            negative_marks=float(data.get("negative_marks", 0.0)),
            multi_mark_marks=float(data.get("multi_mark_marks", 0.0)),
            unanswered_marks=float(data.get("unanswered_marks", 0.0)),
            total_questions=int(data.get("total_questions", 120))
        )


@dataclass
class QuestionResult:
    """Evaluation result for an individual question."""
    question: int
    selected_options: List[str]
    expected_answer: Optional[str]
    status: str
    marks_awarded: float
    confidence: float = 1.0
    flags: List[str] = dataclasses.field(default_factory=list)


@dataclass
class ScoreSummary:
    """Aggregate score summary for the answer sheet."""
    marks_obtained: float
    total_marks: float
    percentage: float
    answered_count: int
    unanswered_count: int
    correct_count: int
    incorrect_count: int
    multi_mark_count: int
    review_count: int


@dataclass
class FlaggedIssue:
    """Structured issue entry for the UI flagged issues panel."""
    category: str        # e.g., "ALIGNMENT", "MULTIPLE_MARK", "AMBIGUOUS", "OCR_VALIDATION"
    severity: str        # "CRITICAL", "WARNING", "INFO"
    message: str
    question: Optional[int] = None
    target_field: Optional[str] = None
    details: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclass
class GradeResult:
    """Complete grading report for an answer sheet."""
    summary: ScoreSummary
    questions: List[QuestionResult]
    flagged_issues: List[FlaggedIssue]
    student_details: Dict[str, Any] = dataclasses.field(default_factory=dict)
    alignment_confidence: float = 1.0
    alignment_flags: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to JSON-serializable dictionary."""
        return {
            "summary": asdict(self.summary),
            "questions": [asdict(q) for q in self.questions],
            "flagged_issues": [asdict(f) for f in self.flagged_issues],
            "student_details": self.student_details,
            "alignment_confidence": self.alignment_confidence,
            "alignment_flags": self.alignment_flags,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize result to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_csv(self) -> str:
        """Export question-by-question table to CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header summary rows
        writer.writerow(["# OMR Graded Report"])
        writer.writerow(["Marks Obtained", self.summary.marks_obtained])
        writer.writerow(["Total Marks", self.summary.total_marks])
        writer.writerow(["Percentage", f"{self.summary.percentage:.2f}%"])
        writer.writerow(["Correct", self.summary.correct_count])
        writer.writerow(["Incorrect", self.summary.incorrect_count])
        writer.writerow(["Multiple Marks", self.summary.multi_mark_count])
        writer.writerow(["Unanswered", self.summary.unanswered_count])
        writer.writerow(["Under Review", self.summary.review_count])
        writer.writerow([])

        # Table header
        writer.writerow([
            "Question", "Selected Option(s)", "Expected Answer", "Status",
            "Marks Awarded", "Confidence", "Flags"
        ])

        for q in self.questions:
            writer.writerow([
                q.question,
                " ".join(q.selected_options) if q.selected_options else "None",
                q.expected_answer or "N/A",
                q.status,
                f"{q.marks_awarded:+.2f}",
                f"{q.confidence:.2f}",
                "; ".join(q.flags) if q.flags else ""
            ])

        return output.getvalue()


class OMRGrader:
    """Deterministic, robust grading engine."""

    def __init__(self, default_scoring: Optional[ScoringConfig] = None):
        self.default_scoring = default_scoring or ScoringConfig()

    def grade(
        self,
        question_predictions: Dict[int, Dict[str, Any]],
        answer_key: Optional[Union[Dict[str, str], Dict[int, str]]] = None,
        scoring_config: Optional[ScoringConfig] = None,
        student_details: Optional[Dict[str, Any]] = None,
        alignment_confidence: float = 1.0,
        alignment_flags: Optional[List[str]] = None
    ) -> GradeResult:
        """Grade questions deterministically.

        Parameters:
            question_predictions: mapping of q_num -> {
                "filled_options": List[str],      # e.g. ["B"] or ["A", "C"]
                "ambiguous_options": List[str],   # e.g. ["D"]
                "disagreements": List[str],       # model vs baseline disagreements
                "confidences": Dict[str, float]   # option -> float
            }
            answer_key: dict mapping question str/int -> expected option (e.g. {"1": "A"})
            scoring_config: ScoringConfig instance
            student_details: dict of extracted handwritten student metadata
            alignment_confidence: float [0.0, 1.0] from preprocessor
            alignment_flags: list of flags from preprocessor (e.g. LOW_ALIGNMENT_CONFIDENCE)
        """
        cfg = scoring_config or self.default_scoring
        alignment_flags = alignment_flags or []
        student_details = student_details or {}

        # Normalize answer key mapping to int -> str
        norm_key: Dict[int, str] = {}
        if answer_key:
            for k, v in answer_key.items():
                try:
                    norm_key[int(k)] = str(v).strip().upper()
                except (ValueError, TypeError):
                    continue

        question_results: List[QuestionResult] = []
        flagged_issues: List[FlaggedIssue] = []

        # Check alignment flags first
        if "LOW_ALIGNMENT_CONFIDENCE" in alignment_flags:
            flagged_issues.append(FlaggedIssue(
                category="ALIGNMENT",
                severity="CRITICAL",
                message=f"Sheet alignment quality ({alignment_confidence:.2f}) is below confidence threshold ({config.ALIGNMENT_CONFIDENCE_THRESHOLD}). Bubble positions may be slightly misaligned.",
                details={"confidence": alignment_confidence}
            ))

        # Check handwritten student metadata flags
        if "flags" in student_details:
            for flag in student_details["flags"]:
                flagged_issues.append(FlaggedIssue(
                    category="STUDENT_METADATA",
                    severity="WARNING",
                    message=f"Handwritten metadata issue flagged: {flag}",
                    target_field="student_metadata",
                    details={"flag": flag}
                ))

        total_marks_obtained = 0.0
        correct_count = 0
        incorrect_count = 0
        multi_mark_count = 0
        unanswered_count = 0
        review_count = 0
        answered_count = 0

        total_questions = cfg.total_questions

        for q_num in range(1, total_questions + 1):
            q_data = question_predictions.get(q_num, {})
            filled_opts = sorted(list(q_data.get("filled_options", [])))
            ambiguous_opts = sorted(list(q_data.get("ambiguous_options", [])))
            disagreements = sorted(list(q_data.get("disagreements", [])))
            confidences = q_data.get("confidences", {})

            expected_opt = norm_key.get(q_num)
            q_flags: List[str] = []

            # Determine confidence (mean over options or minimum)
            q_conf = float(np.mean(list(confidences.values()))) if confidences else 1.0

            # 1. Check for Ambiguous Marking or Disagreements -> REVIEW
            if ambiguous_opts or disagreements:
                status = QuestionStatus.REVIEW
                marks = 0.0
                review_count += 1
                if ambiguous_opts:
                    q_flags.append(f"AMBIGUOUS_MARKING({','.join(ambiguous_opts)})")
                    flagged_issues.append(FlaggedIssue(
                        category="AMBIGUOUS_MARKING",
                        severity="WARNING",
                        message=f"Question {q_num}: Ambiguous bubble markings detected on option(s): {', '.join(ambiguous_opts)}.",
                        question=q_num,
                        details={"ambiguous_options": ambiguous_opts}
                    ))
                if disagreements:
                    q_flags.append(f"MODEL_BASELINE_DISAGREEMENT({','.join(disagreements)})")
                    flagged_issues.append(FlaggedIssue(
                        category="MODEL_BASELINE_DISAGREEMENT",
                        severity="INFO",
                        message=f"Question {q_num}: MobileNetV3 and classical fill-ratio baseline disagreed on option(s): {', '.join(disagreements)}.",
                        question=q_num,
                        details={"disagreements": disagreements}
                    ))

            # 2. Check for Multiple Marks (2+ filled options) -> MULTIPLE_MARK
            elif len(filled_opts) >= 2:
                status = QuestionStatus.MULTIPLE_MARK
                marks = cfg.multi_mark_marks  # Default 0.0
                multi_mark_count += 1
                answered_count += 1
                q_flags.append("MULTIPLE_MARK_DETECTED")
                flagged_issues.append(FlaggedIssue(
                    category="MULTIPLE_MARK",
                    severity="WARNING",
                    message=f"Question {q_num}: Multiple bubbles marked ({', '.join(filled_opts)}). Invalid per single-choice policy.",
                    question=q_num,
                    details={"filled_options": filled_opts}
                ))

            # 3. Check for Unanswered (0 filled options) -> UNANSWERED
            elif len(filled_opts) == 0:
                status = QuestionStatus.UNANSWERED
                marks = cfg.unanswered_marks  # Default 0.0
                unanswered_count += 1

            # 4. Exactly One Option Filled -> Check Correct / Incorrect
            else:
                chosen = filled_opts[0]
                answered_count += 1

                if expected_opt is None:
                    # No expected answer in answer key
                    status = QuestionStatus.CORRECT
                    marks = 0.0
                    q_flags.append("NO_ANSWER_KEY_ENTRY")
                elif chosen == expected_opt:
                    status = QuestionStatus.CORRECT
                    marks = cfg.marks_per_correct
                    correct_count += 1
                else:
                    status = QuestionStatus.INCORRECT
                    # Subtract negative marks if configured
                    marks = -abs(cfg.negative_marks)
                    incorrect_count += 1

            total_marks_obtained += marks

            question_results.append(QuestionResult(
                question=q_num,
                selected_options=filled_opts,
                expected_answer=expected_opt,
                status=status.value if isinstance(status, QuestionStatus) else status,
                marks_awarded=round(marks, 2),
                confidence=round(q_conf, 4),
                flags=q_flags
            ))

        # Theoretical maximum score
        max_possible_marks = total_questions * cfg.marks_per_correct
        percentage = (
            (total_marks_obtained / max_possible_marks * 100.0)
            if max_possible_marks > 0 else 0.0
        )
        # Clamp percentage to minimum 0.0 if negative marking pulls score negative
        clamped_percentage = max(0.0, percentage)

        summary = ScoreSummary(
            marks_obtained=round(total_marks_obtained, 2),
            total_marks=round(max_possible_marks, 2),
            percentage=round(clamped_percentage, 2),
            answered_count=answered_count,
            unanswered_count=unanswered_count,
            correct_count=correct_count,
            incorrect_count=incorrect_count,
            multi_mark_count=multi_mark_count,
            review_count=review_count
        )

        return GradeResult(
            summary=summary,
            questions=question_results,
            flagged_issues=flagged_issues,
            student_details=student_details,
            alignment_confidence=alignment_confidence,
            alignment_flags=alignment_flags
        )
