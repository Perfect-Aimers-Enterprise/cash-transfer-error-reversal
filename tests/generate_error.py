import pandas as pd
import random
import uuid

# -----------------------
# Load Dataset
# -----------------------

df = pd.read_csv("dataset/raw/transactions.csv")

# -----------------------
# Error Classes
# -----------------------

ERROR_TYPES = [
    "No Error",
    "Wrong Beneficiary",
    "Technical Glitch / Timeout",
    "Incorrect Amount Entry",
    "Pending Review",
    "Duplicate Transaction",
    "Unauthorized Reversal Request",
    "Reversed in Error (False Positive)",
]

error_df = df[df["reversal_reason"].isin(ERROR_TYPES)].copy()
no_error_df = df[df["reversal_reason"] == "No Error"]

cities = df["location"].unique().tolist()
channels = df["channel"].unique().tolist()

# -----------------------
# IDs
# -----------------------

next_tx = (
    df["transfer_id"]
    .str.replace("TXN", "", regex=False)
    .astype(int)
    .max()
    + 1
)

synthetic = []

# =====================================================
# Target:
# Total errors == Total No Error
# =====================================================

target_total_errors = len(no_error_df)

target_per_class = target_total_errors // len(ERROR_TYPES)

print(f"Target per error class: {target_per_class:,}\n")

# =====================================================
# Generate synthetic samples PER CLASS
# =====================================================
print(sorted(df["reversal_reason"].unique()))
for error in ERROR_TYPES:

    current = error_df[
        error_df["reversal_reason"] == error
    ]

    current_count = len(current)

    needed = target_per_class - current_count

    print(
        f"{error:<25}"
        f"{current_count:>6} -> "
        f"{target_per_class:>6}"
        f"   (+{needed})"
    )

    if needed <= 0:
        continue

    for _ in range(needed):

        row = current.sample(1).iloc[0].copy()

        # ---------------------------------
        # IDs
        # ---------------------------------

        row["transfer_id"] = f"TXN{next_tx}"
        next_tx += 1

        row["sender_id"] = (
            f"SND{random.randint(100000,999999)}"
        )

        row["beneficiary_id"] = (
            f"BEN{random.randint(200000,999999)}"
        )

        row["device_id"] = (
            f"DEV{random.randint(10000000,99999999)}"
        )

        row["session_id"] = (
            uuid.uuid4().hex[:16]
        )

        # ---------------------------------
        # Amount ±2%
        # ---------------------------------

        row["amount"] = round(
            row["amount"]
            * random.uniform(0.98, 1.02),
            2,
        )

        # ---------------------------------
        # Timestamp ±10 min
        # ---------------------------------

        dt = pd.to_datetime(
            row["timestamp"],
            dayfirst=True,
        )

        dt += pd.Timedelta(
            minutes=random.randint(-10, 10)
        )

        row["timestamp"] = dt.strftime(
            "%d/%m/%Y %H:%M"
        )

        # ---------------------------------
        # Random city (15%)
        # ---------------------------------

        if random.random() < 0.15:
            row["location"] = random.choice(cities)

        # ---------------------------------
        # Random channel (10%)
        # ---------------------------------

        if random.random() < 0.10:
            row["channel"] = random.choice(channels)

        synthetic.append(row)

# =====================================================
# Merge
# =====================================================

synthetic_df = pd.DataFrame(synthetic)

balanced = pd.concat(
    [df, synthetic_df],
    ignore_index=True,
)

balanced = balanced.sample(
    frac=1,
    random_state=42,
).reset_index(drop=True)

balanced.to_csv(
    "transactions_balanced.csv",
    index=False,
)

# =====================================================
# Summary
# =====================================================

print("\n" + "=" * 60)

print(f"Original Dataset : {len(df):,}")
print(f"No Error         : {len(no_error_df):,}")
print(f"Original Errors  : {len(error_df):,}")
print(f"Synthetic Errors : {len(synthetic_df):,}")
print(f"Final Dataset    : {len(balanced):,}")

print("\nFinal Error Distribution")

print(
    balanced[
        balanced["reversal_reason"].isin(ERROR_TYPES)
    ]["reversal_reason"]
    .value_counts()
)

print("=" * 60)