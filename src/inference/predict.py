import torch
import pandas as pd

from src.preprocessing.preprocess import load_encoders
from src.utils.model_utils import load_model
from src.utils.scaler import load_scaler
from src.utils.error_rules import get_error_details
from src.utils.config import (
    MODEL_PATH,
    ENCODER_PATH,
    SCALER_PATH,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model, checkpoint = load_model(
    MODEL_PATH,
    device,
)

scaler = load_scaler(SCALER_PATH)
encoders = load_encoders(
    ENCODER_PATH,
)


def predict(transaction):

    df = pd.DataFrame([transaction])

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["timestamp"] = df["timestamp"].apply(lambda x: x.timestamp())

    df["account_valid"] = (
        df["account_valid"]
        .astype(int)
    )

    df["beneficiary_match"] = (
        df["beneficiary_match"]
        .astype(int)
    )

    categorical = [
        "debit_status",
        "credit_status",
        "transaction_type",
    ]

    for col in categorical:

        df[col] = encoders[col].transform(
            df[col]
        )

    # Drop transaction_id because the model wasn't trained on it
    if "transaction_id" in df.columns:
        df = df.drop(columns=["transaction_id"])
    
    if "sender_account" in df.columns:
        df = df.drop(columns=["sender_account"])

    if "receiver_account" in df.columns:
        df = df.drop(columns=["receiver_account"])

    print(df)
    print(df.dtypes)
    print(df.columns.tolist())
    print(df.iloc[0].to_dict())
    X = scaler.transform(df.values)
    print(X[0])

    x = torch.tensor(
        X,
        dtype=torch.float32,
    ).to(device)

    with torch.no_grad():

        outputs = model(x)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1,
        )

    # label = checkpoint["classes"][
    #     prediction.item()
    # ]
    print("Prediction index:", prediction.item())
    print("Classes:", checkpoint["classes"])

    label = checkpoint["classes"][prediction.item()]

    details = get_error_details(label)


    return {
        "prediction": label,
        "confidence": round(
            confidence.item() * 100,
            2,
        ),
        **details,
    }

# if __name__ == "__main__":

#     sample = {
#         "transaction_id": "TXN900003",
#         "sender_account": 12345678,
#         "receiver_account": 87654321,
#         "amount": 900000,
#         "timestamp": "2026-07-21 12:50:00",
#         "debit_status": "Failed",
#         "credit_status": "Failed",
#         "account_valid": True,
#         "beneficiary_match": True,
#         "retry_count": 0,
#         "transaction_type": "Bank Transfer",
#     }

# print(predict(sample))