import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder

from src.utils.config import DATASET_PATH


def load_data():
    return pd.read_csv(DATASET_PATH)


def preprocess(df):

    df = df.copy()

    # Drop transaction id
    # df.drop(columns=["transaction_id"], inplace=True)
    df.drop(
        columns=[
            "transaction_id",
            "sender_account",
            "receiver_account",
        ],
        inplace=True,
    )

    # Convert timestamp to Unix timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp"] = df["timestamp"].apply(lambda x: x.timestamp())

    # Convert boolean columns
    df["account_valid"] = df["account_valid"].astype(int)
    df["beneficiary_match"] = df["beneficiary_match"].astype(int)

    encoders = {}

    categorical_columns = [
        "debit_status",
        "credit_status",
        "transaction_type",
        "error_label",
    ]

    for column in categorical_columns:

        encoder = LabelEncoder()

        df[column] = encoder.fit_transform(df[column])

        encoders[column] = encoder

    return df, encoders


def save_encoders(encoders, path):

    joblib.dump(encoders, path)


def load_encoders(path):

    return joblib.load(path)