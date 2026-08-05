import json
from pathlib import Path

from src.utils.config import METRIC_DIR


def save_metrics(
    model_name,
    accuracy,
    precision,
    recall,
    f1,
    roc_auc,
    latency,
    false_reversal_rate,
    confusion_matrix,

    true_positive,
    true_negative,
    false_positive,
    false_negative,

    true_positive_rate,
    true_negative_rate,
    false_positive_rate,
    false_negative_rate,

    loss_history=None,
    accuracy_history=None,
):
    metrics = {

        "model": model_name,

        # ----------------------------------
        # Performance Metrics
        # ----------------------------------

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,

        # ----------------------------------
        # Operational Metrics
        # ----------------------------------

        "latency": latency,
        "false_reversal_rate": false_reversal_rate,

        # ----------------------------------
        # Confusion Matrix
        # ----------------------------------

        "confusion_matrix": confusion_matrix,

        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,

        # ----------------------------------
        # Failure Analysis Rates
        # ----------------------------------

        "true_positive_rate": true_positive_rate,
        "true_negative_rate": true_negative_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,

        # ----------------------------------
        # Training History
        # ----------------------------------

        "loss_history": loss_history or [],
        "accuracy_history": accuracy_history or [],
    }

    with open(
        METRIC_DIR / f"{model_name}.json",
        "w",
    ) as f:
        json.dump(metrics, f, indent=4)