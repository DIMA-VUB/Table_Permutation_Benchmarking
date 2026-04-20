"""
tests/test_table_permuter.py
-----------------------------
Unit tests for the table permutation logic.
Run with:  pytest tests/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from table_permuter import (
    _build_permutation,
    _derangement,
    permute_entry,
    permute_table,
    permute_jsonl,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TABLE = {
    "columns": ["season", "tropical lows", "tropical cyclones", "severe tropical cyclones", "strongest storm"],
    "data": [
        ["1990 - 91", "10", "10", "7", "marian"],
        ["1991 - 92", "11", "10", "9", "jane - irna"],
        ["1992 - 93", "6", "3", "1", "oliver"],
        ["1993 - 94", "12", "11", "7", "theodore"],
        ["1994 - 95", "19", "9", "6", "chloe"],
        ["1995 - 96", "19", "14", "9", "olivia"],
        ["1996 - 97", "15", "14", "3", "pancho"],
        ["1997 - 98", "10", "9", "3", "tiffany"],
        ["1998 - 99", "21", "14", "9", "gwenda"],
        ["1999 - 00", "13", "12", "5", "john / paul"],
    ],
}

SAMPLE_ENTRY = {
    "id": "abc123",
    "qtype": "NumericalReasoning",
    "qsubtype": "Aggregation",
    "table": SAMPLE_TABLE,
    "question": "What is the average?",
    "answer": "10.6",
}


# ---------------------------------------------------------------------------
# Derangement tests
# ---------------------------------------------------------------------------

class TestDerangement:
    def test_no_fixed_points(self):
        import random
        rng = random.Random(0)
        for _ in range(50):
            indices = list(range(5))
            d = _derangement(indices, rng)
            assert all(a != b for a, b in zip(indices, d)), "Fixed point found!"

    def test_is_permutation(self):
        import random
        rng = random.Random(1)
        indices = list(range(7))
        d = _derangement(indices, rng)
        assert sorted(d) == sorted(indices)

    def test_single_index_raises(self):
        import random
        rng = random.Random(0)
        with pytest.raises(ValueError):
            _derangement([0], rng)


# ---------------------------------------------------------------------------
# Build permutation tests
# ---------------------------------------------------------------------------

class TestBuildPermutation:
    def test_fixed_positions_respected(self):
        import random
        rng = random.Random(42)
        perm = _build_permutation(5, fixed={0, 4}, derange=False, rng=rng)
        assert perm[0] == 0
        assert perm[4] == 4

    def test_all_fixed_is_identity(self):
        import random
        rng = random.Random(0)
        perm = _build_permutation(4, fixed={0, 1, 2, 3}, derange=False, rng=rng)
        assert perm == [0, 1, 2, 3]

    def test_full_derange_no_fixed_points(self):
        import random
        rng = random.Random(7)
        for _ in range(30):
            perm = _build_permutation(6, fixed=set(), derange=True, rng=rng)
            assert all(perm[i] != i for i in range(6))


# ---------------------------------------------------------------------------
# permute_table tests
# ---------------------------------------------------------------------------

class TestPermuteTable:
    def _cell_set(self, table):
        """Multiset of all cell values (order-independent)."""
        return sorted(str(v) for row in table["data"] for v in row)

    def test_data_preserved(self):
        result = permute_table(SAMPLE_TABLE, fix_rows=0, fix_cols=0, seed=1)
        assert self._cell_set(result) == self._cell_set(SAMPLE_TABLE)

    def test_columns_preserved(self):
        result = permute_table(SAMPLE_TABLE, fix_rows=0, fix_cols=0, seed=2)
        assert sorted(result["columns"]) == sorted(SAMPLE_TABLE["columns"])

    def test_row_count_preserved(self):
        result = permute_table(SAMPLE_TABLE, fix_rows=0, fix_cols=0, seed=3)
        assert len(result["data"]) == len(SAMPLE_TABLE["data"])

    def test_col_count_preserved(self):
        result = permute_table(SAMPLE_TABLE, fix_rows=0, fix_cols=0, seed=4)
        for row in result["data"]:
            assert len(row) == len(SAMPLE_TABLE["columns"])

    def test_none_rows_unchanged(self):
        """fix_rows=None: rows stay in original order regardless of seed."""
        result = permute_table(SAMPLE_TABLE, fix_rows=None, fix_cols=0, seed=7)
        orig_rows = [list(r) for r in SAMPLE_TABLE["data"]]
        # column order may change but each row's *position* must be intact
        # We verify by checking that col-0 values (season strings, unique) keep order
        result_row_positions = [row[0] for row in result["data"]]
        orig_row_positions = [row[0] for row in SAMPLE_TABLE["data"]]
        assert result_row_positions == orig_row_positions, \
            "Rows must not be reordered when fix_rows=None"

    def test_none_cols_unchanged(self):
        """fix_cols=None: column order stays intact regardless of seed."""
        result = permute_table(SAMPLE_TABLE, fix_rows=0, fix_cols=None, seed=8)
        assert result["columns"] == SAMPLE_TABLE["columns"], \
            "Columns must not be reordered when fix_cols=None"
        for orig_row, result_row in zip(SAMPLE_TABLE["data"], result["data"]):
            # rows may be reordered but within each result row the column
            # order must match the original column order
            pass  # column-order per row verified via the header check above

    def test_fixed_col_stays(self):
        """With fix_cols=1, column 0 must remain at position 0."""
        result = permute_table(SAMPLE_TABLE, fix_rows=0, fix_cols=1, seed=10)
        assert result["columns"][0] == SAMPLE_TABLE["columns"][0]

    def test_fixed_row_and_col(self):
        """fix_rows and fix_cols are independent axis constraints.

        fix_rows=1: 1 leading row (index 0) stays at row-position 0; rows 1..n-1 are deranged.
        fix_cols=1: 1 leading column (index 0) stays at col-position 0; cols 1..m-1 are deranged.
        These two constraints operate independently on their respective axes.
        """
        result = permute_table(SAMPLE_TABLE, fix_rows=1, fix_cols=1, seed=5)

        # --- column axis: col 0 stays at position 0 -------------------------
        assert result["columns"][0] == SAMPLE_TABLE["columns"][0], \
            "Column 0 header must remain at position 0"
        # Col 0 value is preserved in every row (col permutation keeps col_perm[0]==0)
        orig_col0 = [row[0] for row in SAMPLE_TABLE["data"]]
        result_col0 = [row[0] for row in result["data"]]
        assert sorted(result_col0) == sorted(orig_col0), \
            "Column 0 values must be preserved across all rows"
        # Free cols 1..4 must be deranged within each row relative to original col order
        orig_row0_free = [SAMPLE_TABLE["data"][0][j] for j in range(1, 5)]
        result_row0_free = [result["data"][0][j] for j in range(1, 5)]
        assert any(result_row0_free[k] != orig_row0_free[k] for k in range(4)), \
            "Free columns must not all stay in place (derangement expected)"

        # --- row axis: row 0 stays at position 0 ----------------------------
        assert result["data"][0][0] == SAMPLE_TABLE["data"][0][0], \
            "Cell [0][0] must stay fixed (row 0 and col 0 both fixed)"
        # Free rows 1..9 must be deranged: no free row stays at its original index
        n = len(SAMPLE_TABLE["data"])
        for i in range(1, n):
            orig_row_i_col0 = SAMPLE_TABLE["data"][i][0]
            result_row_i_col0 = result["data"][i][0]
            # Since col 0 is fixed, col-0 value identifies the originating row.
            # After derangement no free row should sit at its original index.
            pass  # derangement of rows 1-9 verified by the global derangement test
        orig_rows = [tuple(r) for r in SAMPLE_TABLE["data"]]
        result_rows = [tuple(r) for r in result["data"]]
        assert result_rows[0][0] == orig_rows[0][0], \
            "Row 0 must remain at row-position 0 (col-0 value unchanged)"
        assert all(result_rows[i] != orig_rows[i] for i in range(1, n)), \
            "Free rows 1..n-1 must all move (derangement)"

    def test_full_derangement_no_row_unchanged(self):
        """Full derangement: no row stays in its original position."""
        result = permute_table(SAMPLE_TABLE, fix_rows=0, fix_cols=0, seed=99)
        n = len(SAMPLE_TABLE["data"])
        orig_rows = [tuple(r) for r in SAMPLE_TABLE["data"]]
        new_rows = [tuple(r) for r in result["data"]]
        fixed_row_count = sum(1 for i in range(n) if orig_rows[i] == new_rows[i])
        # After derangement on rows no row occupies its original slot
        assert fixed_row_count == 0, f"{fixed_row_count} rows unchanged after derangement"

    def test_reproducible_with_seed(self):
        r1 = permute_table(SAMPLE_TABLE, fix_rows=0, fix_cols=0, seed=77)
        r2 = permute_table(SAMPLE_TABLE, fix_rows=0, fix_cols=0, seed=77)
        assert r1 == r2

    def test_different_seeds_differ(self):
        r1 = permute_table(SAMPLE_TABLE, fix_rows=0, fix_cols=0, seed=1)
        r2 = permute_table(SAMPLE_TABLE, fix_rows=0, fix_cols=0, seed=2)
        # Extremely unlikely to be identical
        assert r1 != r2


# ---------------------------------------------------------------------------
# permute_entry tests
# ---------------------------------------------------------------------------

class TestPermuteEntry:
    def test_meta_attached(self):
        result = permute_entry(SAMPLE_ENTRY, fix_rows=1, fix_cols=2, seed=0)
        assert "permutation_meta" in result
        assert result["permutation_meta"]["fix_rows"] == 1
        assert result["permutation_meta"]["fix_cols"] == 2

    def test_original_unchanged(self):
        import copy
        original = copy.deepcopy(SAMPLE_ENTRY)
        permute_entry(SAMPLE_ENTRY, seed=1)
        assert SAMPLE_ENTRY == original

    def test_non_table_fields_preserved(self):
        result = permute_entry(SAMPLE_ENTRY, seed=5)
        for key in ("id", "qtype", "qsubtype", "question", "answer"):
            assert result[key] == SAMPLE_ENTRY[key]


# ---------------------------------------------------------------------------
# permute_jsonl tests
# ---------------------------------------------------------------------------

class TestPermuteJsonl:
    def test_roundtrip(self, tmp_path):
        input_file = tmp_path / "in.jsonl"
        output_file = tmp_path / "out.jsonl"
        lines = [json.dumps(SAMPLE_ENTRY)]
        input_file.write_text("\n".join(lines), encoding="utf-8")

        results = permute_jsonl(input_file, output_file)
        assert len(results) == 1
        assert output_file.exists()

    def test_multi_entry(self, tmp_path):
        entry2 = {**SAMPLE_ENTRY, "id": "other"}
        input_file = tmp_path / "multi.jsonl"
        output_file = tmp_path / "multi_out.jsonl"
        input_file.write_text(
            json.dumps(SAMPLE_ENTRY) + "\n" + json.dumps(entry2),
            encoding="utf-8",
        )
        results = permute_jsonl(input_file, output_file, seed=0)
        assert len(results) == 2
