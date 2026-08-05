import json
from pathlib import Path

METRIC_DIR = Path("metrics")
METRIC_DIR.mkdir(exist_ok=True)


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
):
    metrics = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "latency_ms": latency,
        "false_reversal_rate": false_reversal_rate,
        "confusion_matrix": confusion_matrix,
    }

    with open(
        METRIC_DIR / f"{model_name}.json",
        "w",
    ) as f:
        json.dump(metrics, f, indent=4)