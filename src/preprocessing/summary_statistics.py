import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATASET = Path(
    "dataset/raw/ablation_dataset.csv"
)

OUTPUT = Path(
    "dataset/raw"
)

OUTPUT.mkdir(exist_ok=True)


# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv(DATASET)

print("=" * 60)
print("Dataset Loaded")
print("=" * 60)

print(df.shape)

# =====================================================
# Summary Statistics
# =====================================================

summary = df.describe(
    include="all"
).transpose()

summary.to_csv(
    OUTPUT / "summary_statistics.csv"
)

print()

print("=" * 60)
print("Summary Statistics")
print("=" * 60)

print(summary)

# =====================================================
# Correlation Matrix
# =====================================================

corr = df.corr(
    numeric_only=True
)

corr.to_csv(
    OUTPUT / "correlation_matrix.csv"
)

# =====================================================
# Correlation Heatmap
# =====================================================

plt.figure(
    figsize=(18,15)
)

plt.imshow(
    corr,
    cmap="coolwarm",
    aspect="auto",
)

plt.colorbar()

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90,
    fontsize=8,
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns,
    fontsize=8,
)

plt.title(
    "Feature Correlation Matrix"
)

plt.tight_layout()

plt.savefig(
    OUTPUT /
    "correlation_matrix.png",
    dpi=300,
)

plt.show()

# =====================================================
# Feature Distribution
# =====================================================

numeric = df.select_dtypes(
    include="number"
).columns

for column in numeric:

    plt.figure(
        figsize=(6,4)
    )

    plt.hist(
        df[column],
        bins=30,
    )

    plt.title(column)

    plt.xlabel(column)

    plt.ylabel("Count")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT /
        f"{column}.png"
    )

    plt.close()

print()

print("=" * 60)
print("Summary Statistics Saved")
print("=" * 60)