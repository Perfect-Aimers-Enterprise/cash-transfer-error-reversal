import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    epochs,
    device,
):
    print("Training Started...\n")

    model.to(device)

    for epoch in range(epochs):

        model.train()

        running_loss = 0

        for features, labels in train_loader:

            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(features)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {avg_loss:.4f}"
        )

    print("\nTraining Finished!")


def evaluate_model(
    model,
    test_loader,
    device,
):

    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():

        for features, labels in test_loader:

            features = features.to(device)

            outputs = model(features)

            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy())

            actuals.extend(labels.numpy())

    accuracy = accuracy_score(
        actuals,
        predictions,
    )

    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report\n")

    print(
        classification_report(
            actuals,
            predictions,
        )
    )

    print("\nConfusion Matrix\n")

    print(
        confusion_matrix(
            actuals,
            predictions,
        )
    )

    return accuracy