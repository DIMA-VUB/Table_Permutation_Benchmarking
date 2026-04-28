"""
cli.py
------
Command-line interface for the table permutation tool.

Usage examples
--------------
# Permute nothing (default: both axes omitted) → output/input_frNone_fcNone.jsonl
python cli.py --input data/input.jsonl --output output/

# Full derangement on both axes → output/input_fr0_fc0.jsonl
python cli.py --input data/input.jsonl --output output/ \
    --fix-rows 0 --fix-cols 0

# Permute only rows; columns unchanged → output/input_fr0_fcNone.jsonl
python cli.py --input data/input.jsonl --output output/ \
    --fix-rows 0

# Keep first 2 rows and first 3 columns fixed → output/input_fr2_fc3.jsonl
python cli.py --input data/input.jsonl --output output/ \
    --fix-rows 2 --fix-cols 3

# Fix specific row indices 0 and 3; derange all others → output/input_fr0-3_fcNone.jsonl
python cli.py --input data/input.jsonl --output output/ \
    --fix-rows 0,3

# Fix specific column indices 1 and 2; derange all other columns
python cli.py --input data/input.jsonl --output output/ \
    --fix-cols 1,2

# Reproducible run → output/input_fr0_fc0.jsonl
python cli.py --input data/input.jsonl --output output/ \
    --fix-rows 0 --fix-cols 0 --seed 42

# Pretty-print the first permuted entry to stdout
python cli.py --input data/input.jsonl --output output/ \
    --fix-rows 0 --fix-cols 0 --preview
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Support running from repo root or from src/
sys.path.insert(0, str(Path(__file__).parent))
from table_permuter import permute_jsonl


def _parse_fix_arg(value: str | None) -> int | list[int] | None:
    """Convert a CLI string for --fix-rows / --fix-cols to the appropriate type.

    Accepted forms
    --------------
    Omitted / None  → None        (axis not permuted)
    "0"             → 0           (full derangement – int 0 means fix first 0 positions)
    "3"             → 3           (first 3 positions fixed)
    "0,"            → [0]         (fix only index 0; trailing comma forces list mode)
    "0,3,5"         → [0, 3, 5]  (fix exactly those indices; rest deranged)
    """
    if value is None:
        return None
    # Trailing comma → treat as explicit index list even for a single element
    parts = [p.strip() for p in value.split(",") if p.strip() != ""]
    if "," in value or len(parts) > 1:
        return [int(p) for p in parts]
    return int(parts[0])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Permute rows/columns of tabular JSONL data for LLM invariance testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", "-i", required=True, help="Path to input JSONL file.")
    p.add_argument("--output", "-o", required=True, help="Output folder (created if absent).")
    p.add_argument(
        "--fix-rows",
        type=str,
        default=None,
        metavar="K_or_INDICES",
        help="Rows to keep fixed. "
             "Single int k: fix first k rows (0 = full derangement). "
             "Comma-separated list e.g. '0,3': fix exactly those row indices, derange the rest. "
             "Omit to leave rows unchanged.",
    )
    p.add_argument(
        "--fix-cols",
        type=str,
        default=None,
        metavar="K_or_INDICES",
        help="Columns to keep fixed. "
             "Single int k: fix first k columns (0 = full derangement). "
             "Comma-separated list e.g. '1,2': fix exactly those column indices, derange the rest. "
             "Omit to leave columns unchanged.",
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility.")
    p.add_argument(
        "--preview",
        action="store_true",
        help="Pretty-print the first permuted entry to stdout after processing.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Generate all axis-permutation variants: rows+cols, rows-only, cols-only. "
             "Ignores --fix-rows and --fix-cols.",
    )
    args = p.parse_args(argv)
    # Convert raw strings to int | list[int] | None
    args.fix_rows = _parse_fix_arg(args.fix_rows)
    args.fix_cols = _parse_fix_arg(args.fix_cols)
    return args


def _fix_label(v: int | list[int] | None) -> str:
    """Compact string representation of a fix_rows/fix_cols value for filenames.

    None        → 'None'
    0           → '0'
    3           → '3'
    [0]         → '[0]'
    [0, 3, 5]   → '[0,3,5]'
    """
    if v is None:
        return "None"
    if isinstance(v, list):
        return "[" + ",".join(str(i) for i in v) + "]"
    return str(v)


def _build_output_path(
    output_dir: str,
    input_path: str,
    fix_rows: int | list[int] | None,
    fix_cols: int | list[int] | None,
) -> Path:
    """Build output path: <output_dir>/<input_stem>_fr{fix_rows}_fc{fix_cols}.jsonl

    Examples
    --------
    fix_rows=2,   fix_cols=3       → input_fr2_fc3.jsonl
    fix_rows=0,   fix_cols=None    → input_fr0_fcNone.jsonl
    fix_rows=[0,3], fix_cols=None  → input_fr[0,3]_fcNone.jsonl
    fix_rows=0,   fix_cols=[0]     → input_fr0_fc[0].jsonl
    """
    stem = Path(input_path).stem
    suffix = f"_fr{_fix_label(fix_rows)}_fc{_fix_label(fix_cols)}"
    return Path(output_dir) / (stem + suffix + ".jsonl")


# All axis-permutation variants to generate when --all is used.
_ALL_VARIANTS: list[tuple[int | None, int | None]] = [
    (0, 0),     # rows + cols fully deranged
    (0, None),  # rows only
    (None, 0),  # cols only
]


def generate_all(
    input_path: str,
    output_dir: str,
    seed: int | None = None,
    preview: bool = False,
) -> None:
    """Generate all axis-permutation variants for *input_path*.

    Produces one output file per variant (rows+cols, rows-only, cols-only).
    """
    for fix_rows, fix_cols in _ALL_VARIANTS:
        output_path = _build_output_path(output_dir, input_path, fix_rows, fix_cols)
        print(f"[table_permutation] Input : {input_path}")
        print(f"[table_permutation] Output: {output_path}")
        print(f"[table_permutation] fix_rows={fix_rows}  fix_cols={fix_cols}  seed={seed}")
        results = permute_jsonl(
            input_path=input_path,
            output_path=output_path,
            fix_rows=fix_rows,
            fix_cols=fix_cols,
            seed=seed,
        )
        print(f"[table_permutation] Done – {len(results)} entries written to {output_path}\n")
        if preview and results:
            print("--- First permuted entry (preview) ---")
            print(json.dumps(results[0], indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.all:
        generate_all(args.input, args.output, seed=args.seed, preview=args.preview)
        return

    fix_rows = args.fix_rows
    fix_cols = args.fix_cols

    output_path = _build_output_path(args.output, args.input, fix_rows, fix_cols)

    print(f"[table_permutation] Input : {args.input}")
    print(f"[table_permutation] Output: {output_path}")
    print(f"[table_permutation] fix_rows={fix_rows}  fix_cols={fix_cols}  seed={args.seed}")

    results = permute_jsonl(
        input_path=args.input,
        output_path=output_path,
        fix_rows=fix_rows,
        fix_cols=fix_cols,
        seed=args.seed,
    )

    print(f"[table_permutation] Done – {len(results)} entries written to {output_path}")

    if args.preview and results:
        print("\n--- First permuted entry (preview) ---")
        print(json.dumps(results[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    #  input filenames 
    Folder_path = r"D:\TABLE_DATASET\TableBench\FactChecking_MatchBased"
    # input_filename = ["TableBench.jsonl", "TableBench_PoT.jsonl", "TableBench_SCoT.jsonl","TableBench_TCoT.jsonl","TableBench_DP.jsonl"]
    input_filename = ["TableBench_DP_FactChecking_MatchBased_edited.jsonl"]
    for input_file in input_filename:
        # Fix first column (index 0), derange all other columns + full row derangement
        # --fix-rows 0   → int 0 → full row derangement (fix first 0 rows = none fixed)
        # --fix-cols 0,  → [0]  → pin col index 0; derange all remaining columns
        main([
            "--input",  r"{folder}\{input_file}".format(folder=Folder_path, input_file=input_file),
            "--output", Folder_path,
            "--seed", "42",
            "--fix-cols", "0,",   # fix col index 0 only; derange all other columns
        ])

        main([
            "--input",  r"{folder}\{input_file}".format(folder=Folder_path, input_file=input_file),
            "--output", Folder_path,
            "--seed", "42",
            "--fix-rows", "0",    # full row derangement
        ])

        # main([
        #     "--input",  r"{folder}\{input_file}".format(folder=Folder_path, input_file=input_file),
        #     "--output", Folder_path,
        #     "--seed", "42",
        #     "--fix-rows", "0",    # full row derangement
        #     "--fix-cols", "0,",   # fix col index 0 only; derange all other columns
        # ])

    # main()
