import numpy as np
import pandas as pd


def preprocess_transaction(
    transaction,
    scaler,
    encoders,
    sequence_length=10,
):
    """
    Preprocess a transaction for GRU inference.

    Returns
    -------
    ndarray
        Shape: (1, sequence_length, num_features)
    """

    df = pd.DataFrame([transaction])

    # --------------------------
    # Timestamp
    # --------------------------

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["timestamp"] = (
        df["timestamp"].astype("int64") // 10**9
    )

    # --------------------------
    # Encode categorical fields
    # --------------------------

    for col in ["channel", "location"]:

        df[col] = encoders[col].transform(
            df[col].astype(str)
        )

    # --------------------------
    # Remove fields not used
    # --------------------------

    df.drop(
        columns=[
            "transfer_id",
            "sender_id",
            "beneficiary_id",
            "device_id",
            "session_id",
            "error_flag",
            "reversal_reason",
            "reversal_executed",
        ],
        errors="ignore",
        inplace=True,
    )

    # --------------------------
    # Scale
    # --------------------------

    X = scaler.transform(df)

    # --------------------------
    # Build GRU sequence
    # --------------------------

    X = np.repeat(
        X,
        sequence_length,
        axis=0,
    )

    X = np.expand_dims(
        X,
        axis=0,
    )

    return X