"""Tests for handwritten field parsers (src/field_parsing.py).

Inputs are real raw OCR outputs observed on the dataset. Names used here are
invented placeholders, not dataset values.
"""

import pytest

from src.field_parsing import parse_date, parse_module_code, parse_name, parse_seat_number


@pytest.mark.parametrize("raw,expected", [
    ("MATH33066", "MATH33066"),
    ("M41H33066", "MATH33066"),      # digits misread as letters in the prefix
    ("CNEM19397", "CHEM19397"),      # prefix snapped to a known subject
    ("EEEN90019", "EEEN90019"),
    ("tco N40s46", "ECON40546"),     # mixed case, spaces, letter-for-digit
    ("Math-3066", "MATH3066"),
])
def test_module_code_parses_common_misreads(raw, expected):
    assert parse_module_code(raw) == expected


@pytest.mark.parametrize("raw", ["economology", "the most important factors", "greenroom", "ECON ^( 96 4", ""])
def test_module_code_rejects_hallucinations(raw):
    assert parse_module_code(raw) is None


def test_module_code_unknown_prefix_rejected():
    assert parse_module_code("ZZZZ12345") is None
    assert parse_module_code("ZZZZ12345", prefixes=[]) == "ZZZZ12345"


@pytest.mark.parametrize("raw,expected", [
    ("19 April 2025", "2025-04-19"),
    ("15 MAY 2024", "2024-05-15"),
    ("15 Mny 9o24", "2024-05-15"),         # fuzzy month, letter-for-digit year
    ("20 OCTOBER2025", "2025-10-20"),      # glued month and year
    ("5 Avqust 2024", "2024-08-05"),
    ("29.02.2024", "2024-02-29"),          # leap day, dotted
    ("02/09/25", "2025-09-02"),            # day first, two-digit year
    ("2024-05-15", "2024-05-15"),          # already ISO
    ("5 August 2024", "2024-08-05"),       # two lines joined
])
def test_date_parses_to_iso(raw, expected):
    assert parse_date(raw) == expected


@pytest.mark.parametrize("raw", ["management", "Date", "the most important factors", "15 May 2004", "31/02/2024", ""])
def test_date_rejects_invalid(raw):
    # 2004 is outside the plausible exam-year range; 31 February does not exist
    assert parse_date(raw) is None


@pytest.mark.parametrize("raw,expected", [
    ("A-245", "A-245"), ("C130", "C130"), ("G- 303", "G-303"), ("12", "12"), ("H-12", "H-12"),
])
def test_seat_number(raw, expected):
    assert parse_seat_number(raw) == expected


@pytest.mark.parametrize("raw", ["Date", "psychotherapist", "", "12345"])
def test_seat_number_rejects(raw):
    assert parse_seat_number(raw) is None


def test_name_requires_two_or_three_words_and_no_function_words():
    assert parse_name("jane doe") == "Jane Doe"
    assert parse_name("JANE O'BRIEN-SMITH") == "Jane O'Brien-Smith"
    assert parse_name("Jane") is None
    assert parse_name("the most important factors") is None
    assert parse_name("Name Jane Doe") is None


def test_date_slashes_read_as_ones():
    assert parse_date("0610112023") == "2023-01-06"


def test_date_year_repair_only_when_requested():
    assert parse_date("12 March 2004") is None
    assert parse_date("12 March 2004", repair_year=True) == "2024-03-12"
    assert parse_date("5 September 2016") is None  # outside plausible exam years


def test_module_code_four_read_as_letter():
    assert parse_module_code("CHGMA 0345") == "CHEM40345"
    assert parse_module_code("STATU 1036") == "STAT41036"
