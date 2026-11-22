"""Tests for the batch grading CLI (src/batch_grade.py) on synthetic sheets."""

import csv
import json

import pytest

import config
from src.batch_grade import load_answer_key, run_batch
from src.preprocess import SheetPreprocessor
from tests.synthetic import (
    answer_key_for, encode_jpeg, make_answer_pattern,
    render_canonical_sheet, simulate_phone_capture,
)


@pytest.fixture(scope="module")
def template():
    with open(config.TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_batch_grades_folder_and_survives_bad_files(tmp_path, template, monkeypatch):
    # Worker processes inherit this and align against the synthetic reference.
    monkeypatch.setenv("OMR_REFERENCE_SHEET", "synthetic")
    pre = SheetPreprocessor(template_path=config.TEMPLATE_PATH, reference="synthetic")
    pattern = make_answer_pattern(seed=4)
    key, expected = answer_key_for(pattern)
    sheet = render_canonical_sheet(pattern, template, pre)

    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "s01-scan.jpg").write_bytes(encode_jpeg(sheet))
    (in_dir / "s02-photo.jpg").write_bytes(encode_jpeg(simulate_phone_capture(sheet, seed=9)))
    (in_dir / "s03-corrupt.jpg").write_bytes(b"not an image")
    (in_dir / "notes.txt").write_text("ignored")

    out_dir = tmp_path / "out"
    rows = run_batch(in_dir, out_dir, answer_key=key, scoring={"negative_marks": 0.25}, workers=2)

    by_sheet = {r["sheet"]: r for r in rows}
    assert set(by_sheet) == {"s01-scan.jpg", "s02-photo.jpg", "s03-corrupt.jpg"}
    assert by_sheet["s03-corrupt.jpg"]["status"] == "ERROR"

    # Workers use whatever bubble model is on disk (trained CNN or fill-margin
    # fallback), so this test checks the CLI mechanics, not classifier accuracy;
    # exact grading on synthetic sheets is covered in test_api and test_fusion.
    for name in ("s01-scan.jpg", "s02-photo.jpg"):
        r = by_sheet[name]
        assert r["status"] == "OK"
        assert r["correct"] + r["incorrect"] + r["multi_mark"] + r["unanswered"] + r["review"] <= 120
        assert r["correct"] >= 0.8 * expected["CORRECT"]

    with open(out_dir / "batch_summary.csv", newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 3
    report = json.loads((out_dir / "sheets" / "s02-photo.json").read_text())
    assert len(report["questions"]) == 120


def test_load_answer_key_json_and_csv(tmp_path):
    j = tmp_path / "k.json"
    j.write_text(json.dumps({"answers": {"1": "a", "2": "C"}}))
    assert load_answer_key(j) == {"1": "A", "2": "C"}

    c = tmp_path / "k.csv"
    c.write_text("Question,Answer\nQ1,b\n2,E\n3,Z\n")
    assert load_answer_key(c) == {"1": "B", "2": "E"}
