import time

import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.metrics import roc_auc_score

def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    epochs,
    device,
):
    print("\nTraining Started...\n")

    model.to(device)

    loss_history = []
    accuracy_history = []
    roc_auc_history = []

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        epoch_labels = []
        epoch_probs = []

        for features, labels in train_loader:

            features = features.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()

            outputs = model(features)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            # -----------------------------
            # Accuracy
            # -----------------------------

            probs = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()

            total += labels.size(0)

            # -----------------------------
            # Save data for ROC-AUC
            # -----------------------------

            epoch_labels.extend(
                labels.cpu().numpy()
            )

            epoch_probs.extend(
                probs[:, 1].detach().cpu().numpy()
            )

        avg_loss = running_loss / len(train_loader)

        train_accuracy = correct / total

        train_auc = roc_auc_score(
            epoch_labels,
            epoch_probs,
        )

        loss_history.append(avg_loss)
        accuracy_history.append(train_accuracy)
        roc_auc_history.append(train_auc)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {avg_loss:.4f} "
            f"Accuracy: {train_accuracy * 100:.2f}% "
            f"ROC-AUC: {train_auc:.4f}"
        )

    print("\nTraining Finished!\n")

    return (
        loss_history,
        accuracy_history,
        roc_auc_history,
    )

# def evaluate_model(
#     model,
#     test_loader,
#     device,
#     class_names=None,
#     threshold=0.50,
# ):

#     model.eval()

#     predictions = []
#     probabilities = []
#     actuals = []

#     total_latency = 0.0
#     total_samples = 0

#     with torch.no_grad():

#         for features, labels in test_loader:

#             features = features.to(device)
#             labels = labels.to(device)

#             start = time.perf_counter()

#             outputs = model(features)

#             end = time.perf_counter()

#             total_latency += (end - start)
#             total_samples += features.size(0)

#             probs = torch.softmax(
#                 outputs,
#                 dim=1,
#             )

#             # ----------------------------------------
#             # Binary Classification
#             # ----------------------------------------

#             if outputs.shape[1] == 2:

#                 positive_probs = probs[:, 1]

#                 predicted = (
#                     positive_probs >= threshold
#                 ).long()

#                 probabilities.extend(
#                     positive_probs.cpu().numpy().tolist()
#                 )

#             # ----------------------------------------
#             # Multi-Class Classification
#             # ----------------------------------------

#             else:

#                 predicted = torch.argmax(
#                     probs,
#                     dim=1,
#                 )

#                 probabilities.extend(
#                     probs.max(dim=1).values.cpu().numpy().tolist()
#                 )

#             predictions.extend(
#                 predicted.cpu().numpy().tolist()
#             )

#             actuals.extend(
#                 labels.cpu().numpy().tolist()
#             )

#     # ===================================================
#     # Accuracy
#     # ===================================================

#     accuracy = accuracy_score(
#         actuals,
#         predictions,
#     )

#     print("=" * 60)
#     print(f"Accuracy : {accuracy * 100:.2f}%")
#     print("=" * 60)

#     # ===================================================
#     # Classification Report
#     # ===================================================

#     report = classification_report(
#         actuals,
#         predictions,
#         target_names=class_names,
#         zero_division=0,
#         output_dict=True,
#     )

#     print("\nClassification Report\n")

#     print(
#         classification_report(
#             actuals,
#             predictions,
#             target_names=class_names,
#             zero_division=0,
#         )
#     )

#     # ===================================================
#     # Confusion Matrix
#     # ===================================================

#     cm = confusion_matrix(
#         actuals,
#         predictions,
#     )

#     print("\nConfusion Matrix\n")
#     print(cm)

#     tn, fp, fn, tp = cm.ravel()

#     # ===================================================
#     # ROC-AUC + False Reversal Rate
#     # ===================================================

#     roc_auc = None
#     false_reversal_rate = None

#     if len(set(actuals)) == 2:

#         roc_auc = roc_auc_score(
#             actuals,
#             probabilities,
#         )

#         print(f"\nROC-AUC : {roc_auc:.4f}")

#         tn, fp, fn, tp = cm.ravel()

#         tnr = tn / (tn + fp)      # Specificity
#         fpr = fp / (fp + tn)
#         fnr = fn / (fn + tp)
#         tpr = tp / (tp + fn)      # Recall

#         false_reversal_rate = (
#             fp / (fp + tn)
#         )

#         print(
#             f"False Reversal Rate : "
#             f"{false_reversal_rate * 100:.2f}%"
#         )

#     # ===================================================
#     # Latency
#     # ===================================================

#     avg_latency = (
#         total_latency / total_samples
#     ) * 1000

#     print(
#         f"Average Latency : "
#         f"{avg_latency:.4f} ms/transaction"
#     )

#     print("=" * 60)

#     # ===================================================
#     # Return Metrics
#     # ===================================================

#     return {

#         "accuracy": accuracy,

#         "precision": report[
#             "weighted avg"
#         ]["precision"],

#         "recall": report[
#             "weighted avg"
#         ]["recall"],

#         "f1": report[
#             "weighted avg"
#         ]["f1-score"],

#         "roc_auc": roc_auc,

#         "latency": avg_latency,

#         "false_reversal_rate": false_reversal_rate,

#         "true_negative": int(tn),
#         "false_positive": int(fp),
#         "false_negative": int(fn),
#         "true_positive": int(tp),


#         "true_negative_rate": tnr,
#         "false_positive_rate": fpr,
#         "false_negative_rate": fnr,
#         "true_positive_rate": tpr,

#         "confusion_matrix": cm.tolist(),

#         # "classification_report": report,

#     }


def evaluate_model(
    model,
    test_loader,
    device,
    class_names=None,
    threshold=0.50,
    save_curve_data=False,      # <- only True for GRU
):

    model.eval()

    predictions = []
    probabilities = []
    actuals = []

    total_latency = 0.0
    total_samples = 0

    with torch.no_grad():

        for features, labels in test_loader:

            features = features.to(device)
            labels = labels.to(device)

            start = time.perf_counter()

            outputs = model(features)

            end = time.perf_counter()

            total_latency += (end - start)
            total_samples += features.size(0)

            probs = torch.softmax(
                outputs,
                dim=1,
            )

            # ----------------------------------------
            # Binary Classification
            # ----------------------------------------

            if outputs.shape[1] == 2:

                positive_probs = probs[:, 1]

                predicted = (
                    positive_probs >= threshold
                ).long()

                probabilities.extend(
                    positive_probs.cpu().numpy().tolist()
                )

            # ----------------------------------------
            # Multi-Class Classification
            # ----------------------------------------

            else:

                predicted = torch.argmax(
                    probs,
                    dim=1,
                )

                probabilities.extend(
                    probs.max(dim=1).values.cpu().numpy().tolist()
                )

            predictions.extend(
                predicted.cpu().numpy().tolist()
            )

            actuals.extend(
                labels.cpu().numpy().tolist()
            )

    # ===================================================
    # Accuracy
    # ===================================================

    accuracy = accuracy_score(
        actuals,
        predictions,
    )

    print("=" * 60)
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print("=" * 60)

    # ===================================================
    # Classification Report
    # ===================================================

    report = classification_report(
        actuals,
        predictions,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    print("\nClassification Report\n")

    print(
        classification_report(
            actuals,
            predictions,
            target_names=class_names,
            zero_division=0,
        )
    )

    # ===================================================
    # Confusion Matrix
    # ===================================================

    cm = confusion_matrix(
        actuals,
        predictions,
    )

    print("\nConfusion Matrix\n")
    print(cm)

    tn, fp, fn, tp = cm.ravel()

    # ===================================================
    # ROC Metrics
    # ===================================================

    roc_auc = None
    false_reversal_rate = None

    tnr = None
    fpr = None
    fnr = None
    tpr = None

    if len(set(actuals)) == 2:

        roc_auc = roc_auc_score(
            actuals,
            probabilities,
        )

        print(f"\nROC-AUC : {roc_auc:.4f}")

        tnr = tn / (tn + fp)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        tpr = tp / (tp + fn)

        false_reversal_rate = fpr

        print(
            f"False Reversal Rate : "
            f"{false_reversal_rate*100:.2f}%"
        )

    # ===================================================
    # Latency
    # ===================================================

    avg_latency = (
        total_latency / total_samples
    ) * 1000

    print(
        f"Average Latency : "
        f"{avg_latency:.4f} ms/transaction"
    )

    print("=" * 60)

    # ===================================================
    # Metrics
    # ===================================================

    metrics = {

        "accuracy": accuracy,

        "precision": report["weighted avg"]["precision"],

        "recall": report["weighted avg"]["recall"],

        "f1": report["weighted avg"]["f1-score"],

        "roc_auc": roc_auc,

        "latency": avg_latency,

        "false_reversal_rate": false_reversal_rate,

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),

        "true_negative_rate": tnr,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "true_positive_rate": tpr,

        "confusion_matrix": cm.tolist(),
    }

    # ---------------------------------------------------
    # Optional curve data (GRU only)
    # ---------------------------------------------------

    if save_curve_data:

        metrics["actuals"] = actuals
        metrics["probabilities"] = probabilities
        metrics["predictions"] = predictions

    return metrics