"""
Per-entry evaluation helpers that dispatch to the correct TableBench metric
depending on qtype/qsubtype, and extract the 'Final Answer:' from raw model output.

Metric dispatch mirrors eval_tablebench_script.py:
    FactChecking                                → EM
    NumericalReasoning                          → EM
    DataAnalysis / CorrelationAnalysis,
                   TrendForecasting,
                   StatisticalAnalysis          → EM_with_error_10
    DataAnalysis / ImpactAnalysis               → EM
    DataAnalysis / (all other subtypes)         → ROUGE-L
    Visualization                               → Pass@1 (0 – requires code exec)
"""

import re
from typing import Tuple

from evaluation.qa_metrics import QAMetric

# Lazily initialised shared instance so the rouge model is loaded only once
_METRIC_ENGINE: QAMetric | None = None


def get_metric_engine() -> QAMetric:
    global _METRIC_ENGINE
    if _METRIC_ENGINE is None:
        _METRIC_ENGINE = QAMetric()
    return _METRIC_ENGINE


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
_FINAL_ANS_RE = re.compile(r"Final Answer:\s*(.+)", re.IGNORECASE)


def parse_prediction(raw_output: str) -> str:
    """
    Extract the answer from a model's raw output.

    Looks for the last occurrence of 'Final Answer: <text>' (case-insensitive).
    Falls back to the last non-empty line of the output if the marker is absent.
    """
    if not raw_output:
        return ""
    matches = _FINAL_ANS_RE.findall(raw_output)
    if matches:
        return matches[-1].strip()
    # fallback: last non-empty line
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    return lines[-1] if lines else ""


# ---------------------------------------------------------------------------
# DataAnalysis subtype sets
# ---------------------------------------------------------------------------
_DA_EM_WITH_ERROR_10 = {"CorrelationAnalysis", "TrendForecasting", "StatisticalAnalysis"}
_DA_EM               = {"ImpactAnalysis"}
# all other DataAnalysis subtypes → ROUGE-L


# ---------------------------------------------------------------------------
# Per-entry scoring
# ---------------------------------------------------------------------------
def tablebench_score(
    pred: str,
    gold: str,
    qtype: str,
    qsubtype: str,
    engine: QAMetric | None = None,
) -> Tuple[float, str]:
    """
    Compute the TableBench-appropriate score for a single (pred, gold) pair.

    Returns
    -------
    score : float
        Value in [0, 1].  (QAMetric returns 0-100 for batches; we normalise here.)
    metric_name : str
        One of 'EM', 'EM_with_error_10', 'ROUGE-L', 'Pass@1'.
    """
    if engine is None:
        engine = get_metric_engine()

    refs  = [gold]
    preds = [pred]

    if qtype in ("FactChecking", "NumericalReasoning"):
        return engine.compute_em_only(refs, preds) / 100.0, "EM"

    elif qtype == "DataAnalysis":
        if qsubtype in _DA_EM_WITH_ERROR_10:
            return engine.compute_em_with_error_only(refs, preds, error_range=10) / 100.0, "EM_with_error_10"
        elif qsubtype in _DA_EM:
            return engine.compute_em_only(refs, preds) / 100.0, "EM"
        else:
            return engine.compute_rougel_only(refs, preds) / 100.0, "ROUGE-L"

    elif qtype == "Visualization":
        # Pass@1 requires code execution – not available in this pipeline
        return 0.0, "Pass@1"

    else:
        return engine.compute_em_only(refs, preds) / 100.0, "EM"
