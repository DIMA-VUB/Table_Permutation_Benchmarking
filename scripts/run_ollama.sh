#!/usr/bin/env bash
# run_ollama.sh
# -------------
# Launch ollama_runner.py in the background with nohup so the process
# survives terminal disconnection.
#
# Usage:
#   chmod +x scripts/run_ollama.sh
#   ./scripts/run_ollama.sh <input> <output_root> <model> [extra args...]
#
# Examples:
#   # Single file
#   ./scripts/run_ollama.sh output/input_fr0_fc0.jsonl results/ llama3
#
#   # Entire folder  (all *.jsonl inside)
#   ./scripts/run_ollama.sh output/ results/ mistral
#
#   # With extra flags
#   ./scripts/run_ollama.sh output/ results/ llama3 --workers 4 --timeout 180
#
# Logs are written to logs/<model>_<timestamp>.log
# PID is printed so you can monitor or kill the process:
#   kill <PID>
#   tail -f logs/<model>_<timestamp>.log

set -euo pipefail

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <input_path_or_folder> <output_root> <model> [extra args...]"
    exit 1
fi

INPUT="$1"
OUTPUT="$2"
MODEL="$3"
shift 3
EXTRA_ARGS=("$@")

# ---------------------------------------------------------------------------
# Resolve project root (one level above scripts/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Python interpreter: prefer .venv, fall back to system python
# ---------------------------------------------------------------------------
if [[ -f "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
    elif [[ -f "$PROJECT_ROOT/venv/bin/python" ]]; then
    PYTHON="$PROJECT_ROOT/venv/bin/python"
else
    PYTHON="$(command -v python3 || command -v python)"
fi

# ---------------------------------------------------------------------------
# Log file
# ---------------------------------------------------------------------------
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${MODEL}_${TIMESTAMP}.log"

# ---------------------------------------------------------------------------
# Load .env if present (makes OLLAMA_HOST / OLLAMA_PORT available to Python)
# ---------------------------------------------------------------------------
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$PROJECT_ROOT/.env"
    set +a
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "[run_ollama] Starting ollama_runner in background"
echo "[run_ollama] input   : $INPUT"
echo "[run_ollama] output  : $OUTPUT"
echo "[run_ollama] model   : $MODEL"
echo "[run_ollama] log     : $LOG_FILE"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "[run_ollama] extra   : ${EXTRA_ARGS[*]}"

nohup "$PYTHON" "$PROJECT_ROOT/src/ollama_runner.py" \
--input  "$INPUT" \
--output "$OUTPUT" \
--model  "$MODEL" \
"${EXTRA_ARGS[@]}" \
>> "$LOG_FILE" 2>&1 &

PID=$!
echo "[run_ollama] PID     : $PID"
echo "[run_ollama] Monitor : tail -f $LOG_FILE"
echo "[run_ollama] Stop    : kill $PID"

# Write PID file next to the log for easy reference
echo "$PID" > "${LOG_FILE%.log}.pid"
