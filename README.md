# Table Permutation – LLM Invariance Testing

A VS Code Python project for generating row/column permutations of tabular JSONL data, designed to test **permutation invariance** of LLMs when processing tables.

---

## Project Structure

```
table_permutation/
├── data/
│   └── input.jsonl              # Your source JSONL data
├── output/                      # Generated permuted files (created at runtime)
├── results/                     # Ollama inference results (created at runtime)
├── logs/                        # nohup run logs (created at runtime)
├── scripts/
│   └── run_ollama.sh            # nohup launcher for ollama_runner.py
├── src/
│   ├── table_permuter.py        # Core permutation logic
│   ├── cli.py                   # Permutation CLI (single file or --all variants)
│   ├── ollama_runner.py         # Ollama inference runner (file or folder)
│   └── batch_generator.py       # Batch sweep over all fix_rows×fix_cols combos
├── tests/
│   └── test_table_permuter.py
├── .env                         # Local config (gitignored) – copy from .env.example
├── .env.example                 # Template for environment variables
├── .vscode/
│   ├── launch.json
│   └── settings.json
├── requirements.txt
└── README.md
```

---

## Permutation Semantics

| `fix_rows` | `fix_cols` | Behaviour |
|-----------|-----------|-----------|
| `None` | `None` | No permutation on either axis. |
| `0` | `0` | **Full derangement** – every row AND column moves. |
| `k > 0` | `None` | First *k* rows stay fixed; remaining rows deranged. Columns untouched. |
| `None` | `k > 0` | First *k* columns stay fixed; remaining columns deranged. Rows untouched. |

> **Derangement**: a permutation where no element appears in its original position.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy environment template
cp .env.example .env          # then edit OLLAMA_HOST / OLLAMA_PORT if needed

# 3. Full derangement on both axes
python src/cli.py --input data/input.jsonl --output output/ --fix-rows 0 --fix-cols 0

# 4. Generate all axis-permutation variants at once (rows+cols, rows-only, cols-only)
python src/cli.py --input data/input.jsonl --output output/ --seed 42 --all

# 5. Run tests
pytest tests/ -v
```

---

## Ollama Inference

Run a permuted JSONL file (or an entire folder) through a locally-running [Ollama](https://ollama.com) model.  
Entries without an `instruction` key are skipped. Already-processed entries are skipped on resume.

### Environment variables (`.env`)

```
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
```

### Single file

```bash
python src/ollama_runner.py \
    --input  output/input_fr0_fc0.jsonl \
    --output results/ \
    --model  llama3
```

### Entire folder

```bash
python src/ollama_runner.py \
    --input  output/ \
    --output results/ \
    --model  llama3
```

Output path per file: `results/<model>/<input_stem>.jsonl`  
Each line: `{ "id": "...", "raw_output": "...", "pred_answer": "..." }`

### Background run with `nohup` (recommended for large datasets)

```bash
chmod +x scripts/run_ollama.sh

# Single file
./scripts/run_ollama.sh output/input_fr0_fc0.jsonl results/ llama3

# Entire folder
./scripts/run_ollama.sh output/ results/ llama3

# With extra flags (workers, timeout)
./scripts/run_ollama.sh output/ results/ llama3 --workers 4 --timeout 180
```

The script:
- Launches the runner with `nohup` so it survives terminal disconnection.
- Writes a timestamped log to `logs/<model>_<timestamp>.log`.
- Saves a `.pid` file alongside the log for easy monitoring/stopping.

```bash
# Follow progress
tail -f logs/llama3_20260420_143000.log

# Stop the run
kill $(cat logs/llama3_20260420_143000.pid)
```

**Resume**: re-running the same command skips entries already present in the output file.

---

## Python API

```python
from src.table_permuter import permute_table, permute_entry, permute_jsonl

# Permute a single table dict
new_table = permute_table(table, fix_rows=0, fix_cols=0, seed=42)

# Permute a full entry dict (also updates the instruction [TABLE] block)
new_entry = permute_entry(entry, fix_rows=0, fix_cols=None, seed=42)

# Permute an entire JSONL file
results = permute_jsonl("data/input.jsonl", "output/out.jsonl", seed=42)
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

Each permuted JSONL entry keeps all original fields plus:
1. `table.columns` and `table.data` are permuted.
2. If `instruction` is present, its embedded `[TABLE]` block is updated to match.
3. A `permutation_meta` key is added:

```json
{
  "permutation_meta": {
    "fix_rows": 0,
    "fix_cols": 0,
    "seed": 42
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
