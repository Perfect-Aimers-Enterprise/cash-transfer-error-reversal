import torch

from src.inference.preprocess import preprocess_transaction

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def run_model(
    model,
    checkpoint,
    scaler,
    encoders,
    transaction,
):
    X = preprocess_transaction(
        transaction,
        scaler,
        encoders,
    )

    print("Inference shape:", X.shape)

    x = torch.tensor(
        X,
        dtype=torch.float32,
    ).to(device)

    with torch.no_grad():

        outputs = model(x)

        probs = torch.softmax(
            outputs,
            dim=1,
        )

    confidence, pred = torch.max(
        probs,
        dim=1,
    )

    return (
        checkpoint["classes"][pred.item()],
        round(confidence.item() * 100, 2),
    )