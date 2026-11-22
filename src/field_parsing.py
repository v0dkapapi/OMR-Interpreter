"""Parsers and validators for handwritten header fields.

Each parser takes raw OCR text and returns the normalized field value, or None
when the text cannot be a valid value for that field. The OCR engine uses them
to pick the first valid reading from TrOCR's n-best list, which stops the
language model's fluent-English hallucinations ("the most important factors")
from being reported as a module code or date.

Pure functions, no models: safe to unit-test anywhere.
"""

import datetime
import re
from typing import Iterable, Optional, Sequence

import config

MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]
_MONTH_ALIASES = {m: i + 1 for i, m in enumerate(MONTHS)}
_MONTH_ALIASES.update({m[:3]: i + 1 for i, m in enumerate(MONTHS)})
_MONTH_ALIASES["sept"] = 9

# Letter <-> digit confusions typical of handwriting OCR. Handwritten 4s are
# often read as A, U or Y; these mappings apply only where digits are expected.
_TO_DIGIT = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "i": "1", "|": "1",
                           "L": "1", "Z": "2", "z": "2", "S": "5", "s": "5", "G": "6", "b": "6",
                           "T": "7", "B": "8", "g": "9", "q": "9",
                           "A": "4", "U": "4", "Y": "4", "y": "4"})
_TO_LETTER = str.maketrans({"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"})

# Function words that never occur in a person's name; their presence means the
# reading is a language-model hallucination rather than a name.
_NAME_STOPWORDS = {
    "the", "of", "and", "a", "an", "to", "in", "is", "for", "on", "with", "at", "by",
    "from", "that", "this", "it", "as", "be", "are", "was", "or", "not", "name",
    "student", "module", "code", "date", "seat",
}


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _closest(word: str, vocabulary: Iterable[str], max_dist: int) -> Optional[str]:
    best, best_d = None, max_dist + 1
    for v in vocabulary:
        d = levenshtein(word, v)
        if d < best_d:
            best, best_d = v, d
    return best if best_d <= max_dist else None


# ----------------------------------------------------------------- name

def parse_name(text: str) -> Optional[str]:
    """Two or three alphabetic words (hyphens and apostrophes allowed)."""
    cleaned = re.sub(r"[^A-Za-z\s'\-]", " ", text or "")
    tokens = [t.strip("'-") for t in cleaned.split()]
    tokens = [t for t in tokens if len(t) >= 2]
    if not 2 <= len(tokens) <= 3:
        return None
    if any(t.lower() in _NAME_STOPWORDS for t in tokens):
        return None
    return " ".join(_title_case(t) for t in tokens)


def _title_case(word: str) -> str:
    parts = re.split(r"([-'])", word)
    out = []
    for p in parts:
        if p in "-'" or not p:
            out.append(p)
        elif p.lower().startswith("mc") and len(p) > 2:
            out.append("Mc" + p[2:].capitalize())
        else:
            out.append(p.capitalize())
    return "".join(out)


# ----------------------------------------------------------- module code

def parse_module_code(text: str, prefixes: Optional[Sequence[str]] = None) -> Optional[str]:
    """Subject prefix (snapped to a known prefix) followed by 4-5 digits."""
    prefixes = list(prefixes if prefixes is not None else config.OCR_CONFIG["module_prefixes"])
    compact = re.sub(r"[^A-Za-z0-9]", "", text or "")
    if len(compact) < 6:
        return None

    best = None
    for p_len in (4, 3):
        for d_len in (5, 4):
            if len(compact) < p_len + d_len:
                continue
            # Try every window so stray leading/trailing marks are ignored.
            for start in range(0, len(compact) - p_len - d_len + 1):
                raw_p = compact[start:start + p_len].upper().translate(_TO_LETTER)
                raw_d = compact[start + p_len:start + p_len + d_len].translate(_TO_DIGIT)
                if not raw_p.isalpha() or not raw_d.isdigit():
                    continue
                if prefixes:
                    snapped = _closest(raw_p, prefixes, max_dist=1)
                    if snapped is None or len(snapped) != p_len:
                        continue
                    dist = levenshtein(raw_p, snapped)
                else:
                    snapped, dist = raw_p, 0
                extra = len(compact) - p_len - d_len
                # Prefer exact prefixes, longer digit runs, and fewer leftovers.
                key = (dist, -d_len, extra, start)
                if best is None or key < best[0]:
                    best = (key, snapped + raw_d)
    return best[1] if best else None


# ------------------------------------------------------------------- date

def _month_from_word(word: str) -> Optional[int]:
    w = word.lower()
    if len(w) < 3:
        return None
    if w in _MONTH_ALIASES:
        return _MONTH_ALIASES[w]
    full = _closest(w, MONTHS, max_dist=max(1, len(w) // 3))
    if full:
        return _MONTH_ALIASES[full]
    if len(w) >= 3:
        abbr = _closest(w[:3], [m[:3] for m in MONTHS], max_dist=0)
        if abbr and levenshtein(w, MONTHS[_MONTH_ALIASES[abbr] - 1][: len(w)]) <= 1:
            return _MONTH_ALIASES[abbr]
    return None


def _year(token: str) -> Optional[int]:
    if len(token) == 4:
        return int(token)
    if len(token) == 2:
        return 2000 + int(token)
    return None


def _valid(y: Optional[int], m: Optional[int], d: Optional[int]) -> Optional[str]:
    lo, hi = config.OCR_CONFIG["date_year_range"]
    if y is None or m is None or d is None or not lo <= y <= hi:
        return None
    try:
        return datetime.date(y, m, d).isoformat()
    except ValueError:
        return None


def parse_date(text: str, repair_year: bool = False) -> Optional[str]:
    """Parse a handwritten date in any common layout into ISO YYYY-MM-DD.

    Handles "19 April 2025", "15 MAY 2024", "5 August\\n2024", "29.02.2024",
    "02/09/25" (day first, as written in Ireland), "2024-05-15", and slashes
    misread as 1s ("0610112023" for 06/01/2023).

    With repair_year=True, an out-of-range year whose third digit was likely
    misread ("2004" for "2024") is repaired. Callers use this only after no
    candidate parsed without repair.
    """
    if not text:
        return None
    if repair_year:
        def _fix(m: re.Match) -> str:
            y = m.group(0)
            lo, hi = config.OCR_CONFIG["date_year_range"]
            if lo <= int(y) <= hi:
                return y
            fixed = y[:2] + "2" + y[3]
            return fixed if lo <= int(fixed) <= hi else y
        text = re.sub(r"(?<!\d)20\d\d(?!\d)", _fix, text)
    raw_tokens = re.findall(r"[A-Za-z]+|\d+|[A-Za-z0-9]+", text)
    tokens = []
    for t in raw_tokens:
        # Repair digit tokens with letter confusions (e.g. "2O24", "l5").
        if re.search(r"\d", t) and re.search(r"[A-Za-z]", t):
            fixed = t.translate(_TO_DIGIT)
            tokens.append(fixed if fixed.isdigit() else t)
        else:
            tokens.append(t)
    # Split glued day/month ("20OCTOBER") and month/year ("OCTOBER2025") runs.
    split_tokens = []
    for t in tokens:
        split_tokens.extend(re.findall(r"\d+|[A-Za-z]+", t))
    words = [t for t in split_tokens if t.isalpha()]
    numbers = [t for t in split_tokens if t.isdigit()]

    month = None
    for w in words:
        month = _month_from_word(w)
        if month:
            break

    if month:
        year = next((int(n) for n in numbers if len(n) == 4), None)
        day = next((int(n) for n in numbers if len(n) <= 2 and 1 <= int(n) <= 31), None)
        if year is None:
            two = [n for n in numbers if len(n) == 2]
            if len(two) >= 2:
                year = 2000 + int(two[-1])
        return _valid(year, month, day)

    if len(numbers) >= 3:
        a, b, c = numbers[0], numbers[1], numbers[2]
        if len(a) == 4:                                   # YYYY-MM-DD
            return _valid(int(a), int(b), int(c))
        year = _year(c)
        day_first = _valid(year, int(b), int(a))           # DD/MM/YY(YY)
        return day_first or _valid(year, int(a), int(b))   # MM/DD fallback
    if len(numbers) == 1 and len(numbers[0]) == 8:        # DDMMYYYY
        n = numbers[0]
        return _valid(int(n[4:]), int(n[2:4]), int(n[:2]))
    if len(numbers) == 1 and len(numbers[0]) == 10:       # DD/MM/YYYY, slashes read as 1
        n = numbers[0]
        if n[2] == "1" and n[5] == "1":
            return _valid(int(n[6:]), int(n[3:5]), int(n[:2]))
    return None


# ------------------------------------------------------------ seat number

def parse_seat_number(text: str) -> Optional[str]:
    """Optional row letter, optional hyphen, 1-4 digits (e.g. "A-245", "H333", "12")."""
    compact = re.sub(r"[\s.]", "", text or "").upper()
    m = re.fullmatch(r"([A-Z]?)(-?)([0-9OISZBG]{1,4})", compact)
    if not m:
        m = re.search(r"\b([A-Z]?)\s*(-?)\s*(\d{1,4})\b", (text or "").upper())
        if not m:
            return None
    letter, hyphen, digits = m.groups()
    digits = digits.translate(_TO_DIGIT)
    if not digits.isdigit():
        return None
    return f"{letter}{hyphen}{digits}" if letter else digits


PARSERS = {
    "name": parse_name,
    "module_code": parse_module_code,
    "date": parse_date,
    "seat_number": parse_seat_number,
}
