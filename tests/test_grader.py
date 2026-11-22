"""Unit tests for src/grader.py covering all grading cases and policies."""

import pytest
import json
import csv
from src.grader import (
    OMRGrader, ScoringConfig, QuestionStatus, GradeResult, ScoreSummary
)


@pytest.fixture
def grader():
    return OMRGrader()


def test_one_option_filled_and_correct(grader):
    """Test one option filled and matching expected answer awards full marks."""
    predictions = {
        1: {"filled_options": ["B"], "confidences": {"B": 0.98}}
    }
    answer_key = {"1": "B"}
    cfg = ScoringConfig(marks_per_correct=1.0, total_questions=1)

    result = grader.grade(predictions, answer_key=answer_key, scoring_config=cfg)

    q1 = result.questions[0]
    assert q1.status == QuestionStatus.CORRECT
    assert q1.selected_options == ["B"]
    assert q1.marks_awarded == 1.0
    assert result.summary.correct_count == 1
    assert result.summary.incorrect_count == 0
    assert result.summary.marks_obtained == 1.0
    assert result.summary.percentage == 100.0


def test_one_option_filled_and_wrong_zero_marking(grader):
    """Test one option filled and wrong awards 0.0 when negative marking is 0.0."""
    predictions = {
        1: {"filled_options": ["A"], "confidences": {"A": 0.95}}
    }
    answer_key = {"1": "B"}
    cfg = ScoringConfig(marks_per_correct=1.0, negative_marks=0.0, total_questions=1)

    result = grader.grade(predictions, answer_key=answer_key, scoring_config=cfg)

    q1 = result.questions[0]
    assert q1.status == QuestionStatus.INCORRECT
    assert q1.selected_options == ["A"]
    assert q1.marks_awarded == 0.0
    assert result.summary.correct_count == 0
    assert result.summary.incorrect_count == 1
    assert result.summary.marks_obtained == 0.0
    assert result.summary.percentage == 0.0


def test_one_option_filled_and_wrong_negative_marking(grader):
    """Test one option filled and wrong penalizes configured negative marks."""
    predictions = {
        1: {"filled_options": ["A"], "confidences": {"A": 0.95}}
    }
    answer_key = {"1": "C"}
    cfg = ScoringConfig(marks_per_correct=1.0, negative_marks=0.25, total_questions=1)

    result = grader.grade(predictions, answer_key=answer_key, scoring_config=cfg)

    q1 = result.questions[0]
    assert q1.status == QuestionStatus.INCORRECT
    assert q1.selected_options == ["A"]
    assert q1.marks_awarded == -0.25
    assert result.summary.marks_obtained == -0.25
    assert result.summary.percentage == 0.0  # Percentage clamped at 0%


def test_two_options_filled_multiple_mark(grader):
    """Test two options filled on same question treated as MULTIPLE_MARK."""
    predictions = {
        1: {"filled_options": ["A", "C"], "confidences": {"A": 0.92, "C": 0.88}}
    }
    # Notice expected answer is C; even though C was marked, multiple marks must invalidate!
    answer_key = {"1": "C"}
    cfg = ScoringConfig(marks_per_correct=1.0, multi_mark_marks=0.0, total_questions=1)

    result = grader.grade(predictions, answer_key=answer_key, scoring_config=cfg)

    q1 = result.questions[0]
    assert q1.status == QuestionStatus.MULTIPLE_MARK
    assert q1.selected_options == ["A", "C"]
    assert q1.marks_awarded == 0.0
    # Must NOT be attributed to single answer C
    assert q1.selected_options != ["C"]
    assert result.summary.multi_mark_count == 1
    assert result.summary.correct_count == 0
    assert len(result.flagged_issues) == 1
    assert result.flagged_issues[0].category == "MULTIPLE_MARK"


def test_three_or_more_options_filled_no_crash(grader):
    """Test 3, 4, or 5 options filled handled gracefully without crashing."""
    predictions = {
        1: {"filled_options": ["A", "B", "C", "D", "E"], "confidences": {}}
    }
    answer_key = {"1": "B"}
    cfg = ScoringConfig(marks_per_correct=1.0, total_questions=1)

    result = grader.grade(predictions, answer_key=answer_key, scoring_config=cfg)

    q1 = result.questions[0]
    assert q1.status == QuestionStatus.MULTIPLE_MARK
    assert q1.selected_options == ["A", "B", "C", "D", "E"]
    assert q1.marks_awarded == 0.0
    assert result.summary.multi_mark_count == 1


def test_no_options_filled_unanswered(grader):
    """Test no options filled classified as UNANSWERED with 0 marks."""
    predictions = {
        1: {"filled_options": [], "confidences": {}}
    }
    answer_key = {"1": "D"}
    cfg = ScoringConfig(marks_per_correct=1.0, total_questions=1)

    result = grader.grade(predictions, answer_key=answer_key, scoring_config=cfg)

    q1 = result.questions[0]
    assert q1.status == QuestionStatus.UNANSWERED
    assert q1.selected_options == []
    assert q1.marks_awarded == 0.0
    assert result.summary.unanswered_count == 1
    assert result.summary.answered_count == 0


def test_ambiguous_bubble_triggers_review(grader):
    """Test ambiguous bubble routed to REVIEW and listed in flagged issues."""
    predictions = {
        1: {
            "filled_options": [],
            "ambiguous_options": ["C"],
            "confidences": {"C": 0.55}
        }
    }
    answer_key = {"1": "C"}
    cfg = ScoringConfig(total_questions=1)

    result = grader.grade(predictions, answer_key=answer_key, scoring_config=cfg)

    q1 = result.questions[0]
    assert q1.status == QuestionStatus.REVIEW
    assert q1.marks_awarded == 0.0
    assert result.summary.review_count == 1
    assert any(f.category == "AMBIGUOUS_MARKING" for f in result.flagged_issues)


def test_model_baseline_disagreement_triggers_review(grader):
    """Test model vs classical fill-ratio disagreement routed to REVIEW."""
    predictions = {
        1: {
            "filled_options": ["A"],
            "disagreements": ["A"],
            "confidences": {"A": 0.70}
        }
    }
    answer_key = {"1": "A"}
    cfg = ScoringConfig(total_questions=1)

    result = grader.grade(predictions, answer_key=answer_key, scoring_config=cfg)

    q1 = result.questions[0]
    assert q1.status == QuestionStatus.REVIEW
    assert result.summary.review_count == 1
    assert any(f.category == "MODEL_BASELINE_DISAGREEMENT" for f in result.flagged_issues)


def test_low_alignment_confidence_flagged(grader):
    """Test low alignment confidence from preprocessor flagged as critical issue."""
    predictions = {1: {"filled_options": ["A"]}}
    answer_key = {"1": "A"}
    cfg = ScoringConfig(total_questions=1)

    result = grader.grade(
        predictions,
        answer_key=answer_key,
        scoring_config=cfg,
        alignment_confidence=0.52,
        alignment_flags=["LOW_ALIGNMENT_CONFIDENCE"]
    )

    assert result.alignment_confidence == 0.52
    assert "LOW_ALIGNMENT_CONFIDENCE" in result.alignment_flags
    assert any(f.category == "ALIGNMENT" and f.severity == "CRITICAL" for f in result.flagged_issues)


def test_full_120_question_grading_and_serialization(grader):
    """Test grading full 120 questions, JSON serialization, and CSV export."""
    cfg = ScoringConfig(marks_per_correct=1.0, negative_marks=0.25, total_questions=120)

    # 100 correct, 10 incorrect, 5 multiple marks, 5 unanswered
    predictions = {}
    answer_key = {}
    for i in range(1, 101):
        predictions[i] = {"filled_options": ["B"]}
        answer_key[str(i)] = "B"

    for i in range(101, 111):
        predictions[i] = {"filled_options": ["A"]}
        answer_key[str(i)] = "C"

    for i in range(111, 116):
        predictions[i] = {"filled_options": ["A", "D"]}
        answer_key[str(i)] = "A"

    for i in range(116, 121):
        predictions[i] = {"filled_options": []}
        answer_key[str(i)] = "E"

    result = grader.grade(
        predictions,
        answer_key=answer_key,
        scoring_config=cfg,
        student_details={"name": "Test Student", "flags": ["UNRECOGNIZED_DATE_FORMAT"]}
    )

    # 100 correct * 1.0 = 100.0
    # 10 incorrect * -0.25 = -2.50
    # 5 multi-mark * 0.0 = 0.0
    # 5 unanswered * 0.0 = 0.0
    # Total marks obtained = 97.50 / 120.0 = 81.25%
    assert result.summary.correct_count == 100
    assert result.summary.incorrect_count == 10
    assert result.summary.multi_mark_count == 5
    assert result.summary.unanswered_count == 5
    assert result.summary.marks_obtained == 97.50
    assert result.summary.total_marks == 120.0
    assert result.summary.percentage == 81.25

    # Test JSON serialization
    json_str = result.to_json()
    parsed = json.loads(json_str)
    assert parsed["summary"]["marks_obtained"] == 97.50
    assert len(parsed["questions"]) == 120

    # Test CSV export
    csv_str = result.to_csv()
    assert "Marks Obtained,97.5" in csv_str
    assert "Percentage,81.25%" in csv_str
    assert "Question,Selected Option(s),Expected Answer" in csv_str


def test_missing_answer_key_entry_not_reported_correct(grader):
    """A marked question with no answer key entry must not be reported as CORRECT."""
    predictions = {
        1: {"filled_options": ["A"]},
        2: {"filled_options": ["B"]},
    }
    answer_key = {"1": "A"}  # question 2 has no key entry
    cfg = ScoringConfig(marks_per_correct=1.0, total_questions=2)

    result = grader.grade(predictions, answer_key=answer_key, scoring_config=cfg)

    q2 = result.questions[1]
    assert q2.status == QuestionStatus.NOT_GRADED
    assert q2.marks_awarded == 0.0
    assert "NO_ANSWER_KEY_ENTRY" in q2.flags
    assert result.summary.correct_count == 1
    assert result.summary.answered_count == 2
