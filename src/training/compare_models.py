import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

# --------------------------------------------------
# Configuration & Directory Setup
# --------------------------------------------------
METRIC_DIR = Path("src/metrics")
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

models = []
for file in METRIC_DIR.glob("*.json"):
    with open(file, "r", encoding="utf-8") as f:
        models.append(json.load(f))

names = [m.get("model", f"Model {i+1}") for i, m in enumerate(models)]
print(f"Models loaded: {len(models)}\n")


# --------------------------------------------------
# Save & Show Helper
# --------------------------------------------------
def save_and_show(filename: str) -> None:
    """Saves the current figure with high DPI and displays it."""
    filepath = PLOT_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# --------------------------------------------------
# High-Level Metrics Bar Plotter
# --------------------------------------------------
def plot_bar(metric_key: str, title: str, ylabel: str) -> None:
    """Plots and saves a comparative bar chart for a specified metric key."""
    values = [m.get(metric_key, 0) for m in models]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color="#2b5c8f", alpha=0.85)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Models", fontsize=12)
    plt.grid(axis="y", alpha=0.3)

    # Value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            f"{yval:.4f}" if isinstance(yval, float) else f"{yval}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    save_and_show(f"{metric_key}_bar_comparison.png")


# --------------------------------------------------
# Confusion Matrix & Failure Breakdown
# --------------------------------------------------
def plot_confusion_counts() -> None:
    """Plots grouped bar charts for TP, TN, FP, and FN across models."""
    tp = [m.get("true_positive", 0) for m in models]
    tn = [m.get("true_negative", 0) for m in models]
    fp = [m.get("false_positive", 0) for m in models]
    fn = [m.get("false_negative", 0) for m in models]

    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(10, 6))
    plt.bar(x - 1.5 * width, tp, width, label="TP")
    plt.bar(x - 0.5 * width, tn, width, label="TN")
    plt.bar(x + 0.5 * width, fp, width, label="FP")
    plt.bar(x + 1.5 * width, fn, width, label="FN")

    plt.xticks(x, names)
    plt.ylabel("Count", fontsize=12)
    plt.xlabel("Models", fontsize=12)
    plt.title("Confusion Matrix Counts Comparison", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_and_show("confusion_counts_comparison.png")


def plot_failure_breakdown() -> None:
    """Plots stacked failure analysis highlighting TP, TN, FP, FN distribution."""
    tp = np.array([m.get("true_positive", 0) for m in models])
    tn = np.array([m.get("true_negative", 0) for m in models])
    fp = np.array([m.get("false_positive", 0) for m in models])
    fn = np.array([m.get("false_negative", 0) for m in models])

    plt.figure(figsize=(10, 6))
    plt.bar(names, tn, label="True Negatives (Correct)", color="#2ca02c")
    plt.bar(names, tp, bottom=tn, label="True Positives (Correct)", color="#1f77b4")
    plt.bar(
        names,
        fp,
        bottom=tn + tp,
        label="False Positives (Error)",
        color="#ff7f0e",
    )
    plt.bar(
        names,
        fn,
        bottom=tn + tp + fp,
        label="False Negatives (Error)",
        color="#d62728",
    )

    plt.ylabel("Total Count", fontsize=12)
    plt.xlabel("Models", fontsize=12)
    plt.title("Model Prediction Composition & Failure Breakdown", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_and_show("failure_breakdown_stacked.png")


def plot_confusion_heatmap(model: dict) -> None:
    """Plots individual model confusion matrix with adaptive text contrast."""
    cm = model.get("confusion_matrix")
    if cm is None:
        return

    cm = np.array(cm)
    model_name = model.get("model", "Model")
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()

    labels = ["No Error", "Error"]
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("Actual Label", fontsize=12)
    plt.title(f"{model_name} Confusion Matrix", fontsize=14, fontweight="bold")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text_color = "white" if val > thresh else "black"
            plt.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                color=text_color,
                fontsize=13,
                fontweight="bold",
            )

    plt.tight_layout()
    save_and_show(f"confusion_matrix_{model_name.replace(' ', '_')}.png")


# --------------------------------------------------
# Epoch Curve Plotters (Single & Combined)
# --------------------------------------------------
def plot_combined_history(metric_key: str, title: str, ylabel: str) -> None:
    """Plots metric progression over epochs for all models that contain the key."""
    plt.figure(figsize=(8, 5))
    has_data = False

    for model in models:
        history = model.get(metric_key)
        if history:
            has_data = True
            plt.plot(
                range(1, len(history) + 1),
                history,
                linewidth=2,
                label=model.get("model", "Model"),
            )

    if has_data:
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_and_show(f"combined_{metric_key}.png")
    else:
        plt.close()


# --------------------------------------------------
# ROC & Precision-Recall (Individual & Combined)
# --------------------------------------------------
def plot_roc_curves(individual: bool = True) -> None:
    """Plots combined ROC curves, and optionally individual ones for each model."""
    plt.figure(figsize=(7, 6))
    has_data = False

    for model in models:
        actuals = model.get("actuals")
        probs = model.get("probabilities")
        m_name = model.get("model", "Model")

        if actuals is not None and probs is not None:
            has_data = True
            fpr, tpr, _ = roc_curve(actuals, probs)

            # Add to combined plot
            plt.plot(fpr, tpr, linewidth=2, label=m_name)

            # Individual Plot option
            if individual:
                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, linewidth=2, color="#1f77b4", label=m_name)
                plt.plot([0, 1], [0, 1], "--", color="grey", label="Random")
                plt.xlabel("False Positive Rate", fontsize=12)
                plt.ylabel("True Positive Rate", fontsize=12)
                plt.title(f"{m_name} ROC Curve", fontsize=14, fontweight="bold")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                save_and_show(f"roc_curve_{m_name.replace(' ', '_')}.png")

    if has_data:
        plt.plot([0, 1], [0, 1], "--", color="grey", label="Random Guess")
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curves Comparison", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_and_show("roc_curves_combined.png")
    else:
        plt.close()


def plot_pr_curves(individual: bool = True) -> None:
    """Plots combined PR curves, and optionally individual ones for each model."""
    plt.figure(figsize=(7, 6))
    has_data = False

    for model in models:
        actuals = model.get("actuals")
        probs = model.get("probabilities")
        m_name = model.get("model", "Model")

        if actuals is not None and probs is not None:
            has_data = True
            precision, recall, _ = precision_recall_curve(actuals, probs)

            # Combined plot
            plt.plot(recall, precision, linewidth=2, label=m_name)

            # Individual Plot option
            if individual:
                plt.figure(figsize=(6, 5))
                plt.plot(recall, precision, linewidth=2, color="#ff7f0e", label=m_name)
                plt.xlabel("Recall", fontsize=12)
                plt.ylabel("Precision", fontsize=12)
                plt.title(f"{m_name} Precision-Recall Curve", fontsize=14, fontweight="bold")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                save_and_show(f"pr_curve_{m_name.replace(' ', '_')}.png")

    if has_data:
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("Precision-Recall Curves Comparison", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_and_show("pr_curves_combined.png")
    else:
        plt.close()


# --------------------------------------------------
# Printed Console Summaries
# --------------------------------------------------
def print_summary_table() -> None:
    """Prints a structured Markdown/Console comparison table of all model metrics."""
    print("=" * 95)
    print(
        f"{'Model':<20}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1 Score':<12}"
        f"{'ROC-AUC':<12}"
        f"{'Latency(ms)':<15}"
    )
    print("=" * 95)

    for m in models:
        name = m.get("model", "Unknown")
        acc = f"{m.get('accuracy', 0):.4f}"
        prec = f"{m.get('precision', 0):.4f}"
        rec = f"{m.get('recall', 0):.4f}"
        f1 = f"{m.get('f1', 0):.4f}"
        auc = f"{m.get('roc_auc', 0):.4f}"
        lat = f"{m.get('latency', 0):.4f}"
        print(f"{name:<20}{acc:<12}{prec:<12}{rec:<12}{f1:<12}{auc:<12}{lat:<15}")

    print("=" * 95 + "\n")


def print_top_performers() -> None:
    """Prints top performers across primary evaluation metrics."""
    if not models:
        return

    best_acc = max(models, key=lambda x: x.get("accuracy", 0))
    best_auc = max(models, key=lambda x: x.get("roc_auc", 0))
    best_prec = max(models, key=lambda x: x.get("precision", 0))
    best_rec = max(models, key=lambda x: x.get("recall", 0))
    lowest_lat = min(models, key=lambda x: x.get("latency", float("inf")))
    lowest_frr = min(models, key=lambda x: x.get("false_reversal_rate", float("inf")))

    print("======================================================")
    print("                BEST PERFORMERS SUMMARY               ")
    print("======================================================")
    print(f"Best Accuracy            : {best_acc.get('model')} ({best_acc.get('accuracy', 0)*100:.2f}%)")
    print(f"Best ROC-AUC             : {best_auc.get('model')} ({best_auc.get('roc_auc', 0):.4f})")
    print(f"Highest Precision        : {best_prec.get('model')} ({best_prec.get('precision', 0):.4f})")
    print(f"Highest Recall           : {best_rec.get('model')} ({best_rec.get('recall', 0):.4f})")
    print(f"Lowest Latency           : {lowest_lat.get('model')} ({lowest_lat.get('latency', 0):.4f} ms)")
    print(f"Lowest False Reversal    : {lowest_frr.get('model')} ({lowest_frr.get('false_reversal_rate', 0):.4f})")
    print("======================================================\n")


# --------------------------------------------------
# Main Pipeline
# --------------------------------------------------
if __name__ == "__main__":
    if not models:
        print("No metrics JSON files found in directory. Exiting.")
        exit()

    # 1. Print Text Tables & Best Performers
    print_summary_table()
    print_top_performers()

    # 2. Bar Plot Comparisons
    metrics_to_plot = [
        ("accuracy", "Accuracy Comparison", "Accuracy"),
        ("precision", "Precision Comparison", "Precision"),
        ("recall", "Recall Comparison", "Recall"),
        ("f1", "F1 Score Comparison", "F1 Score"),
        ("roc_auc", "ROC-AUC Comparison", "ROC-AUC"),
        ("latency", "Inference Latency", "Milliseconds"),
        ("false_reversal_rate", "False Reversal Rate", "Rate"),
        ("true_positive_rate", "True Positive Rate (Recall)", "Rate"),
        ("true_negative_rate", "True Negative Rate (Specificity)", "Rate"),
        ("false_positive_rate", "False Positive Rate", "Rate"),
        ("false_negative_rate", "False Negative Rate", "Rate"),
    ]

    for key, title, ylabel in metrics_to_plot:
        plot_bar(key, title, ylabel)

    # 3. Confusion Matrix Count & Breakdown Plots
    plot_confusion_counts()
    plot_failure_breakdown()

    # 4. Individual Model Confusion Heatmaps
    for model in models:
        plot_confusion_heatmap(model)

    # 5. Combined Epoch Histories
    plot_combined_history("loss_history", "Training Loss over Epochs", "Loss")
    plot_combined_history("accuracy_history", "Training Accuracy over Epochs", "Accuracy")
    plot_combined_history("roc_auc_history", "ROC-AUC over Epochs", "ROC-AUC")
    plot_combined_history("precision_history", "Precision over Epochs", "Precision")
    plot_combined_history("recall_history", "Recall over Epochs", "Recall")
    plot_combined_history("f1_history", "F1 Score over Epochs", "F1 Score")

    # 6. ROC and PR Curves (Individual + Combined)
    plot_roc_curves(individual=True)
    plot_pr_curves(individual=True)

    print(f"All evaluation plots successfully saved to: {PLOT_DIR.resolve()}")