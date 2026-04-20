"""
batch_generator.py
------------------
Generate a configurable set of permuted JSONL files from one input file,
covering different fix_rows / fix_cols combinations.

This is useful for ablation studies: you can sweep across all combinations
and later compare LLM answers across permutations.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import sys
sys.path.insert(0, str(Path(__file__).parent))
from table_permuter import permute_jsonl


@dataclass
class PermutationSpec:
    """One permutation variant to generate."""
    fix_rows: set[int] = field(default_factory=set)
    fix_cols: set[int] = field(default_factory=set)
    label: str = ""
    seed: int | None = None

    def __post_init__(self):
        if not self.label:
            r = "R" + ("".join(str(i) for i in sorted(self.fix_rows)) or "x")
            c = "C" + ("".join(str(i) for i in sorted(self.fix_cols)) or "x")
            self.label = f"fix_{r}_{c}"


def generate_all_specs(
    n_rows: int,
    n_cols: int,
    max_fixed_rows: int = 1,
    max_fixed_cols: int = 1,
    seed: int | None = 42,
) -> list[PermutationSpec]:
    """
    Generate specs for:
    - Full derangement (no fixed rows/cols)
    - Fix up to *max_fixed_rows* rows × *max_fixed_cols* cols combinations

    Parameters
    ----------
    n_rows / n_cols       : table dimensions
    max_fixed_rows/cols   : maximum number of rows/cols to fix simultaneously
    """
    specs: list[PermutationSpec] = []

    # Full derangement
    specs.append(PermutationSpec(fix_rows=set(), fix_cols=set(), label="full_derangement", seed=seed))

    # Enumerate fixed-row subsets × fixed-col subsets
    row_subsets: list[frozenset[int]] = [frozenset()]
    for r in range(1, max_fixed_rows + 1):
        row_subsets += [frozenset(s) for s in itertools.combinations(range(n_rows), r)]

    col_subsets: list[frozenset[int]] = [frozenset()]
    for c in range(1, max_fixed_cols + 1):
        col_subsets += [frozenset(s) for s in itertools.combinations(range(n_cols), c)]

    for rs, cs in itertools.product(row_subsets, col_subsets):
        if not rs and not cs:
            continue  # already added as full_derangement
        specs.append(PermutationSpec(fix_rows=set(rs), fix_cols=set(cs), seed=seed))

    return specs


def run_batch(
    input_path: str | Path,
    output_dir: str | Path,
    specs: Iterable[PermutationSpec],
) -> dict[str, Path]:
    """Run all specs and return {label: output_path}."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for spec in specs:
        out = output_dir / f"{spec.label}.jsonl"
        permute_jsonl(
            input_path=input_path,
            output_path=out,
            fix_rows=spec.fix_rows,
            fix_cols=spec.fix_cols,
            seed=spec.seed,
        )
        results[spec.label] = out
        print(f"  ✓  {spec.label:40s} → {out}")

    return results


# ---------------------------------------------------------------------------
# Quick summary helper
# ---------------------------------------------------------------------------

def summarise_batch(output_dir: str | Path) -> None:
    """Print a brief summary of all .jsonl files in *output_dir*."""
    output_dir = Path(output_dir)
    files = sorted(output_dir.glob("*.jsonl"))
    print(f"\n{'Variant':<45} {'Entries':>7}  {'Columns (first entry)'}")
    print("-" * 90)
    for f in files:
        lines = f.read_text(encoding="utf-8").strip().splitlines()
        n = len(lines)
        first_cols = json.loads(lines[0])["table"]["columns"] if lines else []
        print(f"{f.stem:<45} {n:>7}  {first_cols}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Batch permutation generator.")
    p.add_argument("--input", "-i", default="data/input.jsonl")
    p.add_argument("--output-dir", "-o", default="output/batch")
    p.add_argument("--n-rows", type=int, default=10, help="Number of rows in the table.")
    p.add_argument("--n-cols", type=int, default=5, help="Number of columns in the table.")
    p.add_argument("--max-fixed-rows", type=int, default=1)
    p.add_argument("--max-fixed-cols", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    specs = generate_all_specs(
        n_rows=args.n_rows,
        n_cols=args.n_cols,
        max_fixed_rows=args.max_fixed_rows,
        max_fixed_cols=args.max_fixed_cols,
        seed=args.seed,
    )
    print(f"Generating {len(specs)} permutation variants …\n")
    run_batch(args.input, args.output_dir, specs)
    summarise_batch(args.output_dir)
