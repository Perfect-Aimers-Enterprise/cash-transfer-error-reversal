from pathlib import Path

from src.preprocessing.ablation_preprocess import (
    load_data,
    preprocess,
    save_encoders,
)

from src.utils.config import (
    ERROR_ENCODER_PATH,
)

# ==========================================================
# Output Paths
# ==========================================================

OUTPUT_DIR = Path("dataset/raw")

OUTPUT_DIR.mkdir(
    exist_ok=True,
)

OUTPUT_DATASET = (
    OUTPUT_DIR /
    "ablation_dataset.csv"
)

# ==========================================================
# Build Dataset
# ==========================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = load_data()

print(
    f"Original Shape : {df.shape}"
)

print("=" * 60)
print("Engineering Features...")
print("=" * 60)

df, encoders = preprocess(df)

save_encoders(
    encoders,
    ERROR_ENCODER_PATH,
)

# ==========================================================
# Save
# ==========================================================

df.to_csv(
    OUTPUT_DATASET,
    index=False,
)

print("=" * 60)
print("Ablation Dataset Saved")
print("=" * 60)

print(
    f"Location : {OUTPUT_DATASET}"
)

print(
    f"Shape : {df.shape}"
)

print()

print(df.head())

print()

print("=" * 60)
print("Columns")
print("=" * 60)

for column in df.columns:

    print(column)

print()

print("=" * 60)
print("Missing Values")
print("=" * 60)

print(
    df.isnull().sum()
)

print()

print("=" * 60)
print("Class Distribution")
print("=" * 60)

print(
    df["error_flag"]
    .value_counts()
)

print()

print("=" * 60)
print("Finished.")
print("=" * 60)