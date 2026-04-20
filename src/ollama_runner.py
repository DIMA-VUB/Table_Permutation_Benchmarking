"""
ollama_runner.py
----------------
Send JSONL entries to a locally-running Ollama model and collect responses.

Each input entry must have:
  • ``id``          – identifier kept in the output.
  • ``instruction`` – full prompt text sent to the model.

Output is written to ``<output_root>/<model_name>/<input_stem>.jsonl``.
Each output line contains:
  • ``id``           – copied from the input entry.
  • ``raw_output``   – the model's literal response string.
  • ``pred_answer``  – text extracted after the last "Final Answer:" token.

Usage (CLI)
-----------
# Single file
python src/ollama_runner.py \\
    --input  output/input_fr0_fc0.jsonl \\
    --output results/ \\
    --model  llama3

# Entire folder (all *.jsonl files inside)
python src/ollama_runner.py \\
    --input  output/ \\
    --output results/ \\
    --model  mistral \\
    --host   http://localhost:11434 \\
    --timeout 120 \\
    --workers 4

Entries that lack an ``instruction`` key are silently skipped.
Already-processed entries are skipped on resume (matched by ``id``).

Usage (Python API)
------------------
from ollama_runner import run_jsonl

run_jsonl(
    input_path="output/input_fr0_fc0.jsonl",
    output_root="results/",
    model="llama3",
)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError as exc:
    raise ImportError(
        "The 'requests' library is required. Install it with: pip install requests"
    ) from exc

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # python-dotenv is optional; fall back to os.environ / defaults


def _default_host() -> str:
    """Build the Ollama base URL from OLLAMA_HOST / OLLAMA_PORT env vars."""
    host = os.environ.get("OLLAMA_HOST", "localhost").strip()
    port = os.environ.get("OLLAMA_PORT", "11434").strip()
    return f"http://{host}:{port}"


# ---------------------------------------------------------------------------
# Ollama HTTP client
# ---------------------------------------------------------------------------

_FINAL_ANSWER_RE = re.compile(r"Final Answer\s*:\s*(.+)", re.IGNORECASE)


def _extract_answer(text: str) -> str:
    """Return the last 'Final Answer: …' value found in *text*, or empty string."""
    matches = _FINAL_ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else ""


def _query_ollama(
    prompt: str,
    model: str,
    host: str,
    timeout: int,
) -> str:
    """POST a single prompt to the Ollama /api/generate endpoint.

    Returns the model's response string.
    Raises ``requests.HTTPError`` on non-2xx responses.
    """
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["response"]


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _process_entry(
    entry: dict[str, Any],
    model: str,
    host: str,
    timeout: int,
) -> dict[str, Any]:
    """Prompt the model for a single entry and return the result dict."""
    prompt = str(entry["instruction"])
    raw = _query_ollama(prompt, model=model, host=host, timeout=timeout)
    return {
        "id": entry["id"],
        "raw_output": raw,
        "pred_answer": _extract_answer(raw),
    }


def _load_done_ids(output_path: Path) -> set[str]:
    """Return the set of ids already present in *output_path* (for resume support)."""
    if not output_path.exists():
        return set()
    done: set[str] = set()
    with output_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def run_jsonl(
    input_path: str | Path,
    output_root: str | Path,
    model: str,
    host: str | None = None,
    timeout: int = 120,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Run all entries in *input_path* through an Ollama model.

    Already-processed entries (matched by ``id``) are skipped so the run can
    be resumed after an interruption.  Results are appended to the output file
    as each entry completes (live progress).

    Parameters
    ----------
    input_path  : JSONL file to read.
    output_root : Root folder; results go to ``<output_root>/<model>/<stem>.jsonl``.
    model       : Ollama model name (e.g. ``"llama3"``, ``"mistral"``).
    host        : Ollama server URL.  Defaults to OLLAMA_HOST:OLLAMA_PORT from .env.
    timeout     : Per-request timeout in seconds.
    workers     : Number of parallel HTTP workers (keep low for large models).

    Returns
    -------
    List of result dicts written during this run (excludes already-done entries).
    """
    if host is None:
        host = _default_host()

    input_path = Path(input_path)

    # Build output path: <output_root>/<model>/<stem>.jsonl
    out_dir = Path(output_root) / model
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / input_path.name

    # Load all input entries, skipping those without an 'instruction' key
    entries: list[dict[str, Any]] = []
    skipped = 0
    with input_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entry = json.loads(line)
                if "instruction" not in entry:
                    skipped += 1
                    continue
                entries.append(entry)

    # Resume: skip entries whose id is already in the output file
    done_ids = _load_done_ids(output_path)
    pending = [e for e in entries if e["id"] not in done_ids]

    print(f"[ollama_runner] model   : {model}")
    print(f"[ollama_runner] input   : {input_path}  ({len(entries)} with instruction, {skipped} skipped)")
    print(f"[ollama_runner] output  : {output_path}")
    print(f"[ollama_runner] host    : {host}  workers={workers}  timeout={timeout}s")
    if done_ids:
        print(f"[ollama_runner] resume  : {len(done_ids)} already done, {len(pending)} remaining")

    if not pending:
        print("[ollama_runner] Nothing to do – all entries already processed.")
        return []

    results: list[dict[str, Any]] = []
    total = len(entries)
    done_count = len(done_ids)

    def _task(entry: dict[str, Any]) -> dict[str, Any]:
        try:
            return _process_entry(entry, model=model, host=host, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            return {"id": entry.get("id", ""), "raw_output": "", "pred_answer": "", "error": str(exc)}

    # Open output file in append mode so each result is written immediately
    with output_path.open("a", encoding="utf-8") as out_fh:
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_entry = {pool.submit(_task, e): e for e in pending}
                for future in as_completed(future_to_entry):
                    result = future.result()
                    out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_fh.flush()
                    results.append(result)
                    done_count += 1
                    print(f"  [{done_count}/{total}] id={result['id']}  pred={result['pred_answer']!r}")
        else:
            for entry in pending:
                result = _task(entry)
                out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_fh.flush()
                results.append(result)
                done_count += 1
                print(f"  [{done_count}/{total}] id={result['id']}  pred={result['pred_answer']!r}")

    print(f"[ollama_runner] Done – {len(results)} new results → {output_path}")
    return results


def run_folder(
    input_folder: str | Path,
    output_root: str | Path,
    model: str,
    host: str | None = None,
    timeout: int = 120,
    workers: int = 1,
) -> None:
    """Run every ``*.jsonl`` file in *input_folder* through an Ollama model.

    Each file is processed independently via :func:`run_jsonl`.  The folder
    is scanned non-recursively.  Files without any entry that has an
    ``instruction`` key are skipped automatically inside :func:`run_jsonl`.
    """
    input_folder = Path(input_folder)
    jsonl_files = sorted(input_folder.glob("*.jsonl"))
    if not jsonl_files:
        print(f"[ollama_runner] No *.jsonl files found in {input_folder}")
        return
    print(f"[ollama_runner] Folder scan: {len(jsonl_files)} file(s) in {input_folder}")
    for jsonl_file in jsonl_files:
        print(f"\n[ollama_runner] === {jsonl_file.name} ===")
        run_jsonl(
            input_path=jsonl_file,
            output_root=output_root,
            model=model,
            host=host,
            timeout=timeout,
            workers=workers,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a JSONL file through a local Ollama model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",   "-i", required=True,
                   help="Path to a single JSONL file or a folder of JSONL files.")
    p.add_argument("--output",  "-o", required=True, help="Output root folder.")
    p.add_argument("--model",   "-m", required=True, help="Ollama model name (e.g. llama3).")
    p.add_argument("--host",    default=None,
                   help="Ollama server URL. Defaults to OLLAMA_HOST:OLLAMA_PORT from .env.")
    p.add_argument("--timeout", type=int, default=180, help="Per-request timeout in seconds.")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers. Keep at 1 for large single-GPU models.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    input_path = Path(args.input)
    kwargs = dict(
        output_root=args.output,
        model=args.model,
        host=args.host,
        timeout=args.timeout,
        workers=args.workers,
    )
    if input_path.is_dir():
        run_folder(input_path, **kwargs)
    else:
        run_jsonl(input_path, **kwargs)


if __name__ == "__main__":
    main()
