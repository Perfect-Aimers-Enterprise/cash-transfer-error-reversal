import time

import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


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

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for features, labels in train_loader:

            features = features.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()

            outputs = model(features)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {avg_loss:.4f}"
        )

    print("\nTraining Finished!\n")


def evaluate_model(
    model,
    test_loader,
    device,
    class_names=None,
    threshold=0.50,
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

            # -----------------------------
            # Binary Classification
            # -----------------------------
            if outputs.shape[1] == 2:

                positive_probs = probs[:, 1]

                predicted = (
                    positive_probs >= threshold
                ).long()

                probabilities.extend(
                    positive_probs.cpu().numpy().tolist()
                )

            # -----------------------------
            # Multi-class Classification
            # -----------------------------
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

    accuracy = accuracy_score(
        actuals,
        predictions,
    )

    print("=" * 60)
    print(f"Accuracy : {accuracy*100:.2f}%")
    print("=" * 60)

    print("\nClassification Report\n")

    print(
        classification_report(
            actuals,
            predictions,
            target_names=class_names,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        actuals,
        predictions,
    )

    print("\nConfusion Matrix\n")
    print(cm)

    # ------------------------------------
    # ROC-AUC
    # ------------------------------------

    if len(set(actuals)) == 2:

        roc = roc_auc_score(
            actuals,
            probabilities,
        )

        print(f"\nROC-AUC : {roc:.4f}")

        tn, fp, fn, tp = cm.ravel()

        false_reversal_rate = (
            fp / (fp + tn)
        )

        print(
            f"False Reversal Rate : "
            f"{false_reversal_rate*100:.2f}%"
        )

    # ------------------------------------
    # Average Latency
    # ------------------------------------

    avg_latency = (
        total_latency / total_samples
    ) * 1000

    print(
        f"Average Latency : "
        f"{avg_latency:.4f} ms/transaction"
    )

    print("=" * 60)

    return accuracy