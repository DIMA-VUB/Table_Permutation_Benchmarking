# Table Permutation – LLM Invariance Testing

A VS Code Python project for generating row/column permutations of tabular JSONL data, designed to test **permutation invariance** of LLMs when processing tables.

---

## Project Structure

```
table_permutation/
├── data/
│   └── input.jsonl          # Your source JSONL data
├── output/                  # Generated permuted files (created at runtime)
├── src/
│   ├── table_permuter.py    # Core permutation logic
│   ├── cli.py               # Single-file CLI
│   └── batch_generator.py   # Batch sweep over all fix_rows×fix_cols combos
├── tests/
│   └── test_table_permuter.py
├── .vscode/
│   ├── launch.json          # Pre-configured debug/run configurations
│   └── settings.json
├── requirements.txt
└── README.md
```

---

## Permutation Semantics

| `fix_rows` | `fix_cols` | Behaviour |
|-----------|-----------|-----------|
| `{}` (empty) | `{}` (empty) | **Full derangement** – every row AND every column moves; no element stays on its original diagonal. |
| `{0}` | `{}` | Row 0 stays; all other rows deranged. Columns fully shuffled. |
| `{}` | `{0}` | Column 0 stays; all other columns deranged. Rows fully shuffled. |
| `{0}` | `{0}` | Row 0 and column 0 stay fixed; remaining rows/cols shuffled. |
| **Single entry in file** | any | Plain shuffle (no derangement enforced). |

> **Derangement**: a permutation where no element appears in its original position.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Full derangement (default)
python src/cli.py --input data/input.jsonl --output output/deranged.jsonl --preview

# 3. Fix first row and first column
python src/cli.py \
    --input data/input.jsonl \
    --output output/fix_r0_c0.jsonl \
    --fix-rows 0 \
    --fix-cols 0 \
    --seed 42 \
    --preview

# 4. Fix multiple rows/cols
python src/cli.py \
    --input data/input.jsonl \
    --output output/fix_multi.jsonl \
    --fix-rows 0 1 \
    --fix-cols 0 2

# 5. Batch sweep (all combinations up to 1 fixed row, 1 fixed col)
python src/batch_generator.py \
    --input data/input.jsonl \
    --output-dir output/batch \
    --n-rows 10 --n-cols 5 \
    --max-fixed-rows 1 --max-fixed-cols 1 \
    --seed 42

# 6. Run tests
pytest tests/ -v
```

---

## VS Code Configurations (F5)

| Configuration | Description |
|--------------|-------------|
| CLI – Full Derangement | Runs full derangement on input.jsonl |
| CLI – Fix Row 0 + Col 0 | Fixes first row and first column |
| CLI – Fix Row 0 only | Fixes only the first row |
| Batch Generator | Sweeps all fix_rows×fix_cols combos |
| Run Pytest | Runs the full test suite |

---

## Output Format

Each output JSONL entry is identical to the input entry with two changes:
1. `table.columns` and `table.data` are permuted.
2. A `permutation_meta` key is added:

```json
{
  "permutation_meta": {
    "fix_rows": [0],
    "fix_cols": [],
    "seed": 42,
    "single_example": false
  }
}
```

---

## Python API

```python
from src.table_permuter import permute_table, permute_entry, permute_jsonl

# Permute a single table dict
new_table = permute_table(table, fix_rows={0}, fix_cols={0}, seed=42)

# Permute a full entry dict
new_entry = permute_entry(entry, fix_rows={0}, fix_cols={0}, seed=42)

# Permute an entire JSONL file
results = permute_jsonl("data/input.jsonl", "output/out.jsonl", seed=42)
```
