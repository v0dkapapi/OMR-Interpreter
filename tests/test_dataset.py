"""Tests for dataset form selection (no private data needed)."""

from src.dataset import _round_robin_by_contributor


def test_round_robin_spreads_forms_across_contributors():
    rows = [{"contributor": p, "form_id": f"{p}-f{i:02d}"} for p in ("p07", "p01", "p02") for i in range(1, 11)]
    ordered = _round_robin_by_contributor(rows)

    assert len(ordered) == len(rows)
    assert {r["contributor"] for r in ordered[:3]} == {"p01", "p02", "p07"}
    # A small budget must not collapse to a single contributor.
    assert len({r["contributor"] for r in ordered[:5]}) == 3
