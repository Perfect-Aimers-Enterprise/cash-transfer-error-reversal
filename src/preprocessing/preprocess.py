import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder

from src.utils.config import DATASET_PATH


def load_data():
    return pd.read_csv(DATASET_PATH)


# ----------------------------------
# Feature Engineering
# ----------------------------------

def engineer_features(df):

    df = df.copy()

    # ----------------------------
    # Timestamp
    # ----------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%d/%m/%Y %H:%M",
    )

    df = df.sort_values("timestamp")

    df["timestamp"] = (
        df["timestamp"].astype("int64")
        // 10**9
    )

    return df


# ----------------------------------
# Encoding
# ----------------------------------

def encode_features(df):

    df = df.copy()

    encoders = {}

    categorical_columns = [
        "channel",
        "location",
        "reversal_reason",
    ]

    for column in categorical_columns:

        encoder = LabelEncoder()

        df[column] = encoder.fit_transform(
            df[column].astype(str)
        )

        encoders[column] = encoder

    return df, encoders


# ----------------------------------
# Main preprocessing
# ----------------------------------

def preprocess(df):

    df = df.copy()

    # Feature engineering
    df = engineer_features(df)

    # ----------------------------
    # Drop ID columns
    # ----------------------------

    df.drop(
        columns=[
            "transfer_id",
            # "sender_id",
            "beneficiary_id",
            "device_id",
            "session_id",
        ],
        inplace=True,
    )

    # ----------------------------
    # Convert booleans
    # ----------------------------

    df["error_flag"] = df["error_flag"].astype(int)
    df["reversal_executed"] = (
        df["reversal_executed"].astype(int)
    )

    # Encoding
    df, encoders = encode_features(df)

    return df, encoders


def save_encoders(encoders, path):
    joblib.dump(encoders, path)


def load_encoders(path):
    return joblib.load(path)