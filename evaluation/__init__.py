"""
TableBench-compatible evaluation package.

Exposes:
    QAMetric       – computes EM, EM_with_error_10, ROUGE-L (mirrors TableBench)
    tablebench_score – per-row dispatch that picks the right metric by qtype/qsubtype
    parse_prediction – extracts 'Final Answer:' from raw model output
"""

from evaluation.qa_metrics import QAMetric
from evaluation.evaluator import tablebench_score, parse_prediction

__all__ = ["QAMetric", "tablebench_score", "parse_prediction"]
