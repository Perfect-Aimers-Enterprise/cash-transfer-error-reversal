import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

metric_dir = Path("metrics")

models = []

for file in metric_dir.glob("*.json"):
    with open(file) as f:
        models.append(json.load(f))

names = [m["model"] for m in models]


def plot_bar(metric_key, title, ylabel):
    values = [m[metric_key] for m in models]

    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Models")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_confusion_counts():

    tp = [m["true_positive"] for m in models]
    tn = [m["true_negative"] for m in models]
    fp = [m["false_positive"] for m in models]
    fn = [m["false_negative"] for m in models]

    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(10, 6))

    plt.bar(x - 1.5 * width, tp, width, label="TP")
    plt.bar(x - 0.5 * width, tn, width, label="TN")
    plt.bar(x + 0.5 * width, fp, width, label="FP")
    plt.bar(x + 1.5 * width, fn, width, label="FN")

    plt.xticks(x, names)
    plt.ylabel("Count")
    plt.xlabel("Models")
    plt.title("Confusion Matrix Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Performance Metrics
# --------------------------------------------------

plot_bar(
    "accuracy",
    "Accuracy Comparison",
    "Accuracy",
)

plot_bar(
    "precision",
    "Precision Comparison",
    "Precision",
)

plot_bar(
    "recall",
    "Recall Comparison",
    "Recall",
)

plot_bar(
    "f1",
    "F1 Score Comparison",
    "F1 Score",
)

plot_bar(
    "roc_auc",
    "ROC-AUC Comparison",
    "ROC-AUC",
)

plot_bar(
    "latency",
    "Inference Latency",
    "Milliseconds",
)

plot_bar(
    "false_reversal_rate",
    "False Reversal Rate",
    "Rate",
)


# --------------------------------------------------
# Confusion Matrix Counts
# --------------------------------------------------

plot_bar(
    "true_positive",
    "True Positives",
    "Count",
)

plot_bar(
    "true_negative",
    "True Negatives",
    "Count",
)

plot_bar(
    "false_positive",
    "False Positives",
    "Count",
)

plot_bar(
    "false_negative",
    "False Negatives",
    "Count",
)


# --------------------------------------------------
# Failure Analysis Rates
# --------------------------------------------------

plot_bar(
    "true_positive_rate",
    "True Positive Rate (Recall)",
    "Rate",
)

plot_bar(
    "true_negative_rate",
    "True Negative Rate (Specificity)",
    "Rate",
)

plot_bar(
    "false_positive_rate",
    "False Positive Rate",
    "Rate",
)

plot_bar(
    "false_negative_rate",
    "False Negative Rate",
    "Rate",
)


plot_confusion_counts()