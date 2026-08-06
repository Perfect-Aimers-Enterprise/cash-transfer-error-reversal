import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.utils.config import DATASET_PATH


# ----------------------------------------
# Load Dataset
# ----------------------------------------

df = pd.read_csv(DATASET_PATH)


# ----------------------------------------
# 1. Error Flag Distribution
# ----------------------------------------

plt.figure(figsize=(7,5))

counts = df["error_flag"].value_counts().sort_index()

plt.bar(
    ["No Error (0)", "Error (1)"],
    counts.values,
)

plt.ylabel("Number of Transactions")
plt.xlabel("Error Flag")
plt.title("Distribution of Error Labels")

for i, v in enumerate(counts.values):
    plt.text(i, v, f"{v:,}", ha="center")

plt.tight_layout()
plt.show()


# ----------------------------------------
# 2. Transaction Amount Distribution
# log(1 + amount)
# ----------------------------------------

plt.figure(figsize=(8,5))

amount_log = np.log1p(df["amount"])

plt.hist(
    amount_log,
    bins=40,
)

plt.xlabel("log(1 + Amount)")
plt.ylabel("Transaction Count")
plt.title("Distribution of Transaction Amounts")

plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ----------------------------------------
# 3. Error Rate by Channel
# ----------------------------------------

error_rate = (
    df.groupby("channel")["error_flag"]
      .mean()
      .sort_values()
)

plt.figure(figsize=(8,5))

plt.barh(
    error_rate.index,
    error_rate.values,
)

plt.xlabel("Error Rate")
plt.ylabel("Channel")
plt.title("Error Rate by Transaction Channel")

plt.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.show()