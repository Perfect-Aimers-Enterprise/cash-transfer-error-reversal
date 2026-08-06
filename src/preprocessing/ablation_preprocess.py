import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from src.utils.config import DATASET_PATH
from src.preprocessing.ablation_feature_engineering import (
    engineer_features,
)


# ==========================================================
# Load Dataset
# ==========================================================

def load_data():

    return pd.read_csv(DATASET_PATH)


# ==========================================================
# Encode Remaining Categorical Features
# ==========================================================

def encode_features(df):

    df = df.copy()

    encoders = {}

    categorical_columns = [
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


# ==========================================================
# Main Preprocessing
# ==========================================================

def preprocess(df):

    df = df.copy()

    # ----------------------------------------
    # Feature Engineering
    # ----------------------------------------

    df = engineer_features(df)

    # ----------------------------------------
    # Convert Labels
    # ----------------------------------------

    df["error_flag"] = (
        df["error_flag"]
        .astype(int)
    )

    df["reversal_executed"] = (
        df["reversal_executed"]
        .astype(int)
    )

    # ----------------------------------------
    # Encode Remaining Categoricals
    # ----------------------------------------

    df, encoders = encode_features(df)

    # ----------------------------------------
    # Remove Original Channel
    # One-hot version already exists
    # ----------------------------------------

    df.drop(
        columns=[
            "channel",
        ],
        inplace=True,
    )

    # ----------------------------------------
    # Remove IDs
    # They are only needed during
    # feature engineering
    # ----------------------------------------

    df.drop(

        columns=[

            "transfer_id",
            "sender_id",
            "beneficiary_id",
            "device_id",
            "session_id",

        ],

        inplace=True,

    )

    return df, encoders


# ==========================================================
# Save Encoders
# ==========================================================

def save_encoders(
    encoders,
    path,
):

    joblib.dump(
        encoders,
        path,
    )


# ==========================================================
# Load Encoders
# ==========================================================

def load_encoders(path):

    return joblib.load(path)