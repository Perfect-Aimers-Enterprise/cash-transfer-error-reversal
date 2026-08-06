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

    # for col in ["channel", "location"]:

    #     if col in df.columns and col in encoders:

    #         df[col] = encoders[col].transform(
    #             df[col].astype(str)
    #         )
    # --------------------------
    # Encode categorical fields
    # --------------------------

    # Default values if missing
    if "channel" not in df.columns:
        df["channel"] = "mobile"

    if "location" not in df.columns:
        df["location"] = "Abuja"

    for col in ["channel", "location"]:

        if col in encoders:

            # Handle unseen categories
            value = str(df[col].iloc[0])

            if value not in encoders[col].classes_:
                value = encoders[col].classes_[0]

            df[col] = encoders[col].transform([value])
        else:
            # Encoder missing completely
            df[col] = 0


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