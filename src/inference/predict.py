import torch
import pandas as pd
import numpy as np

from src.preprocessing.preprocess import load_encoders
from src.utils.model_utils import load_model
from src.utils.scaler import load_scaler
from src.utils.error_rules import get_error_details
from src.utils.config import (
    ERROR_MODEL_PATH,
    ERROR_ENCODER_PATH,
    ERROR_SCALER_PATH,
    REASON_MODEL_PATH,
    REASON_ENCODER_PATH,
    REASON_SCALER_PATH,
)
from src.inference.run_model import run_model

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------
# Error Detector
# -----------------------------

error_model, error_checkpoint = load_model(
    ERROR_MODEL_PATH,
    device,
)

error_scaler = load_scaler(
    ERROR_SCALER_PATH,
)

error_encoders = load_encoders(
    ERROR_ENCODER_PATH,
)

# -----------------------------
# Reason Classifier
# -----------------------------

reason_model, reason_checkpoint = load_model(
    REASON_MODEL_PATH,
    device,
)

reason_scaler = load_scaler(
    REASON_SCALER_PATH,
)

reason_encoders = load_encoders(
    REASON_ENCODER_PATH,
)


def predict(transaction):

    error_label, error_conf = run_model(
        error_model,
        error_checkpoint,
        error_scaler,
        error_encoders,
        transaction,
    )

    if error_label == "No Error":
        return {
            "error_detected": False,
            "prediction": "No Error",
            "confidence": error_conf,
            **get_error_details("No Error"),
        }

    reason, reason_conf = run_model(
        reason_model,
        reason_checkpoint,
        reason_scaler,
        reason_encoders,
        transaction,
    )

    return {
        "error_detected": True,
        "prediction": reason,
        "confidence": reason_conf,
        **get_error_details(reason),
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