"""
QAMetric – computes EM, EM_with_error_*, ROUGE-L for a batch of predictions.
Mirrors: https://github.com/TableBench/TableBench/blob/main/metrics/qa_metrics.py

Dependencies:
    pip install evaluate rouge_score
"""

import re
import string

import evaluate

from evaluation.base_metric import BaseMetric
from evaluation.custom_em_metric import compute_em, compute_em_with_tolerance


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class QAMetric(BaseMetric):
    """
    Batch metric engine identical to TableBench's QAMetric.

    Usage::

        metric = QAMetric()
        scores = metric.compute(references=["10.6"], predictions=["10.6"])
        # {'EM': 100.0, 'EM_with_error_2': 100.0, ..., 'ROUGE-L': 100.0}
    """

    def __init__(self, **kwargs):
        import sys
        sys.setrecursionlimit(8735 * 2080 + 10)  # set once, not per compute() call
        self.rouge = evaluate.load("rouge")
        # Keep a fast scorer for single-pair ROUGE-L (avoids evaluate overhead)
        try:
            from rouge_score import rouge_scorer as _rs
            self._fast_rouge = _rs.RougeScorer(["rougeL"], use_stemmer=False)
        except ImportError:
            self._fast_rouge = None

    def preprocess(self, references, predictions):
        """Normalize both lists in place."""
        processed_predictions = []
        processed_references  = []
        for prediction, reference in zip(predictions, references):
            processed_predictions.append(normalize_answer(prediction))
            processed_references.append(normalize_answer(reference))
        return processed_references, processed_predictions

    # keep the original typo as an alias so any code referencing it still works
    def prepsocess(self, references, predictions):
        return self.preprocess(references, predictions)

    def compute_em_only(self, references, predictions) -> float:
        """Fast path: returns EM score (0–100) without computing ROUGE-L."""
        refs, preds = self.preprocess(references, predictions)
        return round(compute_em(references=refs, predictions=preds) * 100, 2)

    def compute_em_with_error_only(self, references, predictions, error_range=10) -> float:
        """Fast path: returns EM_with_error score (0–100) without computing ROUGE-L."""
        refs, preds = self.preprocess(references, predictions)
        return round(compute_em_with_tolerance(references=refs, predictions=preds, error_range=error_range) * 100, 2)

    def compute_rougel_only(self, references, predictions) -> float:
        """Fast path: returns ROUGE-L (0–100) without computing EM variants."""
        refs, preds = self.preprocess(references, predictions)
        if self._fast_rouge is not None and len(refs) == 1:
            score = self._fast_rouge.score(refs[0], preds[0])
            return round(score["rougeL"].fmeasure * 100, 2)
        rouge_score = self.rouge.compute(references=refs, predictions=preds)
        return round(rouge_score["rougeL"] * 100, 2)

    def compute(self, references, predictions):
        """
        Compute all supported metrics for a batch.

        Returns a dict with keys:
            EM, EM_with_error_2, EM_with_error_5, EM_with_error_10, ROUGE-L
        All values are percentages (0–100), rounded to 2 decimal places.
        """
        references, predictions = self.preprocess(references, predictions)

        em_score             = compute_em(references=references, predictions=predictions)
        em_score_with_error_2  = compute_em_with_tolerance(references=references, predictions=predictions, error_range=2)
        em_score_with_error_5  = compute_em_with_tolerance(references=references, predictions=predictions, error_range=5)
        em_score_with_error_10 = compute_em_with_tolerance(references=references, predictions=predictions, error_range=10)

        rouge_score = self.rouge.compute(references=references, predictions=predictions)

        return {
            "EM":                round(em_score             * 100, 2),
            "EM_with_error_2":   round(em_score_with_error_2  * 100, 2),
            "EM_with_error_5":   round(em_score_with_error_5  * 100, 2),
            "EM_with_error_10":  round(em_score_with_error_10 * 100, 2),
            "ROUGE-L":           round(rouge_score["rougeL"]  * 100, 2),
        }
