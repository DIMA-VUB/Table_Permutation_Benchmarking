"""
table_permuter.py
-----------------
Core logic for row/column permutation of tabular data entries.

Permutation semantics
---------------------
fix_rows  : int | list[int] | set[int] | None
    None          → rows are NOT permuted (axis skipped).
    0             → all rows are free (full derangement).
    k  (int > 0)  → rows 0 … k-1 stay fixed; remaining rows form
                    a derangement among themselves.
    {i, j, …}    → exactly the listed indices stay fixed; every
                    other row index forms a derangement.
fix_cols  : int | list[int] | set[int] | None  – same semantics applied to columns.

Both axes are independent.  Passing None for one axis leaves that axis
untouched while the other axis is still permuted.

"""

from __future__ import annotations

import ast
import copy
import json
import random
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derangement(indices: list[int], rng: random.Random, max_attempts: int = 10_000) -> list[int]:
    """Return a derangement of *indices* (no fixed points).

    Uses repeated Fisher-Yates until a derangement is found.
    Raises RuntimeError after *max_attempts* failed tries (practically
    impossible for n >= 3, but guards against n == 1).
    """
    if len(indices) <= 1:
        raise ValueError(
            f"Cannot derange a single index: {indices}. "
            "Use at least 2 free rows/columns for full derangement."
        )
    for _ in range(max_attempts):
        candidate = indices[:]
        rng.shuffle(candidate)
        if all(a != b for a, b in zip(indices, candidate)):
            return candidate
    raise RuntimeError(f"Could not find a derangement after {max_attempts} attempts.")  # pragma: no cover


def _build_permutation(
    n: int,
    fixed: set[int],
    derange: bool,
    rng: random.Random,
) -> list[int]:
    """Build a permutation of range(n) respecting *fixed* positions.

    Parameters
    ----------
    n       : total number of positions
    fixed   : indices that must map to themselves
    derange : if True, free indices form a derangement
    rng     : seeded RNG
    """
    perm = list(range(n))
    free_positions = [i for i in range(n) if i not in fixed]

    if not free_positions:
        return perm  # nothing to shuffle

    free_values = free_positions[:]

    if derange and len(free_values) > 1:
        free_values = _derangement(free_values, rng)
    else:
        rng.shuffle(free_values)

    for pos, val in zip(free_positions, free_values):
        perm[pos] = val

    return perm


# ---------------------------------------------------------------------------
# Table permutation
# ---------------------------------------------------------------------------

# Type alias used throughout the module
_FixArg = int | list[int] | set[int] | None


def _resolve_fixed(fix: _FixArg, n: int) -> set[int] | None:
    """Convert a fix_rows / fix_cols argument to a set of fixed indices.

    Returns
    -------
    None          – the axis should not be permuted at all.
    set[int]      – the indices that must stay in place; all others are free
                    to be deranged.

    Accepted forms for *fix*
    ------------------------
    None                 → return None  (axis skipped)
    int k                → return set(range(k))  (first k indices fixed)
    list[int] / set[int] → return set(fix)  (exactly those indices fixed)
    """
    if fix is None:
        return None
    if isinstance(fix, (list, set, frozenset)):
        return set(fix)
    # plain int: first `fix` indices are fixed
    return set(range(fix))


def _compute_permutations(
    n_rows: int,
    n_cols: int,
    fix_rows: _FixArg,
    fix_cols: _FixArg,
    seed: int | None,
) -> tuple[list[int], list[int]]:
    """Return (row_perm, col_perm) without applying them."""
    rng = random.Random(seed)
    derange = True

    fixed_cols = _resolve_fixed(fix_cols, n_cols)
    col_perm = (
        list(range(n_cols))
        if fixed_cols is None
        else _build_permutation(n_cols, fixed_cols, derange=derange, rng=rng)
    )

    fixed_rows = _resolve_fixed(fix_rows, n_rows)
    row_perm = (
        list(range(n_rows))
        if fixed_rows is None
        else _build_permutation(n_rows, fixed_rows, derange=derange, rng=rng)
    )
    return row_perm, col_perm


def _apply_permutations(
    table: dict[str, Any],
    row_perm: list[int],
    col_perm: list[int],
) -> dict[str, Any]:
    """Apply pre-computed row_perm and col_perm to *table*, preserving cell types."""
    columns = table["columns"]
    data = table["data"]
    n_cols = len(columns)
    new_columns = [columns[col_perm[j]] for j in range(n_cols)]
    col_permuted = [[row[col_perm[j]] for j in range(n_cols)] for row in data]
    new_data = [col_permuted[row_perm[i]] for i in range(len(data))]
    return {"columns": new_columns, "data": new_data}


def permute_table(
    table: dict[str, Any],
    fix_rows: _FixArg = None,
    fix_cols: _FixArg = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Return a *new* table dict with rows and/or columns permuted.

    Parameters
    ----------
    table    : dict with keys ``columns`` (list[str]) and ``data``
               (list[list[Any]]).
    fix_rows : None            → rows are not permuted.
               0               → full derangement on all rows.
               k  (int > 0)    → first k rows stay fixed; rest are deranged.
               list / set[int] → exactly those indices stay fixed; all others
                                  are deranged.
    fix_cols : same semantics as fix_rows, applied to columns.
    seed     : RNG seed for reproducibility.

    Returns
    -------
    New table dict (same structure) with permuted rows/columns.
    """
    n_rows = len(table["data"])
    n_cols = len(table["columns"])
    row_perm, col_perm = _compute_permutations(
        n_rows, n_cols, fix_rows, fix_cols, seed
    )
    return _apply_permutations(table, row_perm, col_perm)


# ---------------------------------------------------------------------------
# Instruction-field helpers
# ---------------------------------------------------------------------------

_TABLE_MARKER = "[TABLE] \n"


def _parse_instruction_table(instruction: str) -> tuple[dict[str, Any] | None, int, int]:
    """Find and parse the [TABLE] block in the instruction string.

    Returns (parsed_table, block_start, block_end) where block_start/block_end
    are the character indices of the table dict string inside *instruction*.
    Returns (None, -1, -1) if the marker is absent or the block cannot be parsed.
    """
    start = instruction.find(_TABLE_MARKER)
    if start == -1:
        return None, -1, -1
    block_start = start + len(_TABLE_MARKER)
    block_end = instruction.find("\n", block_start)
    if block_end == -1:
        block_end = len(instruction)
    table_str = instruction[block_start:block_end]
    try:
        parsed = ast.literal_eval(table_str)
    except (ValueError, SyntaxError):
        return None, -1, -1
    return parsed, block_start, block_end


def _replace_instruction_table(
    instruction: str,
    new_table: dict[str, Any],
    block_start: int,
    block_end: int,
) -> str:
    """Splice the permuted table dict back into the instruction string."""
    new_table_str = str({"columns": new_table["columns"], "data": new_table["data"]})
    return instruction[:block_start] + new_table_str + instruction[block_end:]


# ---------------------------------------------------------------------------
# Entry-level functions
# ---------------------------------------------------------------------------

def permute_entry(
    entry: dict[str, Any],
    fix_rows: _FixArg = None,
    fix_cols: _FixArg = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Permute a single JSONL entry and return a new dict.

    fix_rows / fix_cols accept the same values as :func:`permute_table`.
    When a list or set of indices is passed, exactly those positions are
    kept fixed and all remaining positions are deranged.
    """
    new_entry = copy.deepcopy(entry)

    n_rows = len(entry["table"]["data"])
    n_cols = len(entry["table"]["columns"])

    # Compute permutation vectors once – reused for both the table and the
    # instruction's embedded [TABLE] block so both stay in sync.
    row_perm, col_perm = _compute_permutations(
        n_rows, n_cols, fix_rows, fix_cols, seed
    )

    new_entry["table"] = _apply_permutations(entry["table"], row_perm, col_perm)

    # If the entry has an instruction field, parse its [TABLE] block and
    # apply the SAME permutation vectors to it, preserving original cell types.
    if "instruction" in new_entry:
        instr_table, block_start, block_end = _parse_instruction_table(
            new_entry["instruction"]
        )
        if instr_table is not None:
            permuted_instr_table = _apply_permutations(instr_table, row_perm, col_perm)
            new_entry["instruction"] = _replace_instruction_table(
                new_entry["instruction"], permuted_instr_table, block_start, block_end
            )

    # sets are not JSON-serialisable; convert to sorted lists for the meta field
    def _ser(v: _FixArg):
        return sorted(v) if isinstance(v, (set, frozenset)) else v

    new_entry["permutation_meta"] = {
        "fix_rows": _ser(fix_rows),
        "fix_cols": _ser(fix_cols),
        "seed": seed,
    }
    return new_entry


def permute_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    fix_rows: _FixArg = None,
    fix_cols: _FixArg = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Read *input_path*, permute every entry, write to *output_path*.

    fix_rows / fix_cols accept the same values as :func:`permute_table`:
    ``None``, an ``int``, or a ``list`` / ``set`` of indices to keep fixed.

    Returns the list of permuted entries.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    with input_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    # Use a base seed, offset per entry for diversity
    permuted = []
    for i, entry in enumerate(entries):
        entry_seed = (seed + i) if seed is not None else None
        permuted.append(
            permute_entry(
                entry,
                fix_rows=fix_rows,
                fix_cols=fix_cols,
                seed=entry_seed,
            )
        )

    with output_path.open("w", encoding="utf-8") as fh:
        for p in permuted:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    return permuted
