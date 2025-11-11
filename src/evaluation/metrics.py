"""Model evaluation metrics utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn import metrics as sk_metrics


@dataclass
class EvaluationResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    confusion_matrix: np.ndarray

    def as_dict(self) -> Dict[str, float]:
        return {
            "model": self.model_name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc or float("nan"),
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> EvaluationResult:
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    accuracy = sk_metrics.accuracy_score(y_true_labels, y_pred_labels)
    precision = sk_metrics.precision_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)
    recall = sk_metrics.recall_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)
    f1 = sk_metrics.f1_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)

    try:
        roc_auc = sk_metrics.roc_auc_score(y_true, y_pred, multi_class="ovr")
    except ValueError:
        roc_auc = None

    confusion = sk_metrics.confusion_matrix(y_true_labels, y_pred_labels)

    return EvaluationResult(
        model_name=model_name,
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=float(roc_auc) if roc_auc is not None else None,
        confusion_matrix=confusion,
    )


def aggregate_results(results: List[EvaluationResult]) -> Dict[str, float]:
    return {result.model_name: result.as_dict() for result in results}
