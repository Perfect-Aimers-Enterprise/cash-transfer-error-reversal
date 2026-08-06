import numpy as np
import pandas as pd


# ==========================================================
# Feature Engineering for Ablation Study
# ==========================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates engineered features for the ablation study.

    The dataset MUST already contain:

    transfer_id
    sender_id
    beneficiary_id
    amount
    timestamp
    channel
    location
    device_id
    session_id
    error_flag

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe.
    """

    df = df.copy()

    # =====================================================
    # Timestamp
    # =====================================================

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%d/%m/%Y %H:%M",
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    # -----------------------------------------------------
    # Hour
    # -----------------------------------------------------

    df["hour"] = df["timestamp"].dt.hour

    # -----------------------------------------------------
    # Night Transaction
    # 8PM - 6AM
    # -----------------------------------------------------

    df["is_night"] = (
        (df["hour"] >= 20)
        | (df["hour"] <= 6)
    ).astype(int)

    # -----------------------------------------------------
    # Weekend
    # -----------------------------------------------------

    df["is_weekend"] = (
        df["timestamp"].dt.dayofweek >= 5
    ).astype(int)

    # =====================================================
    # Amount Features
    # =====================================================

    df["amount_log"] = np.log1p(
        df["amount"]
    )

    # =====================================================
    # Sender Transaction Count
    # =====================================================

    df["sender_txn_count"] = (
        df.groupby("sender_id")
        .cumcount()
        + 1
    )

    # =====================================================
    # Sender Running Mean
    # =====================================================

    df["sender_avg_amount_so_far"] = (
        df.groupby("sender_id")["amount_log"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    # =====================================================
    # Sender Running Std
    # =====================================================

    df["sender_std_amount_so_far"] = (
        df.groupby("sender_id")["amount_log"]
        .expanding()
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    # =====================================================
    # Seconds Since Last Transaction
    # =====================================================

    df["sender_seconds_since_last"] = (

        df.groupby("sender_id")["timestamp"]

        .diff()

        .dt.total_seconds()

        .fillna(0)

    )

    # =====================================================
    # Running Error Rate
    # =====================================================

    sender_errors = (

        df.groupby("sender_id")["error_flag"]

        .cumsum()

        - df["error_flag"]

    )

    sender_previous = (

        df.groupby("sender_id")

        .cumcount()

    )

    df["sender_error_rate_so_far"] = (

        sender_errors

        /

        sender_previous.replace(0, np.nan)

    ).fillna(0)

    # =====================================================
    # Beneficiary Transaction Count
    # =====================================================

    df["beneficiary_txn_count_so_far"] = (

        df.groupby("beneficiary_id")

        .cumcount()

        + 1

    )

    # =====================================================
    # Distinct Senders Per Beneficiary
    # =====================================================

    distinct_counts = []

    seen = {}

    for sender, beneficiary in zip(

        df["sender_id"],

        df["beneficiary_id"],

    ):

        if beneficiary not in seen:

            seen[beneficiary] = set()

        seen[beneficiary].add(sender)

        distinct_counts.append(

            len(seen[beneficiary])

        )

    df[

        "beneficiary_distinct_senders_so_far"

    ] = distinct_counts

    # =====================================================
    # New Beneficiary?
    # =====================================================

    first_pair = set()

    values = []

    for sender, beneficiary in zip(

        df["sender_id"],

        df["beneficiary_id"],

    ):

        key = (

            sender,

            beneficiary,

        )

        if key in first_pair:

            values.append(0)

        else:

            values.append(1)

            first_pair.add(key)

    df["is_new_beneficiary_for_sender"] = values

    # =====================================================
    # New Device?
    # =====================================================

    sender_devices = {}

    values = []

    for sender, device in zip(

        df["sender_id"],

        df["device_id"],

    ):

        sender_devices.setdefault(

            sender,

            set(),

        )

        if device in sender_devices[sender]:

            values.append(0)

        else:

            values.append(1)

            sender_devices[sender].add(device)

    df["is_new_device_for_sender"] = values

    # =====================================================
    # New Location?
    # =====================================================

    sender_locations = {}

    values = []

    for sender, location in zip(

        df["sender_id"],

        df["location"],

    ):

        sender_locations.setdefault(

            sender,

            set(),

        )

        if location in sender_locations[sender]:

            values.append(0)

        else:

            values.append(1)

            sender_locations[sender].add(location)

    df["is_new_location_for_sender"] = values

    # =====================================================
    # Duplicate-like
    # Same sender
    # Same beneficiary
    # Same amount
    # Within 5 minutes
    # =====================================================

    duplicate = np.zeros(len(df))

    previous = {}

    for i, row in df.iterrows():

        key = (

            row["sender_id"],

            row["beneficiary_id"],

            row["amount"],

        )

        if key in previous:

            delta = (

                row["timestamp"]

                -

                previous[key]

            ).total_seconds()

            if delta <= 300:

                duplicate[i] = 1

        previous[key] = row["timestamp"]

    df["is_duplicate_like"] = duplicate.astype(int)

    # =====================================================
    # Channel One-Hot
    # =====================================================

    channels = pd.get_dummies(

        df["channel"],

        prefix="channel",

    )

    df = pd.concat(

        [

            df,

            channels,

        ],

        axis=1,

    )

    # Ensure every expected channel exists

    for col in [

        "channel_mobile",

        "channel_web",

        "channel_ussd",

    ]:

        if col not in df.columns:

            df[col] = 0

    # =====================================================
    # Unix Timestamp
    # =====================================================

    df["timestamp"] = (

        df["timestamp"]

        .astype("int64")

        //

        10**9

    )

    return df