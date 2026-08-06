import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.preprocessing.ablation_preprocess import (
    load_data,
    preprocess,
    save_encoders,
)
from src.preprocessing.feature_groups import (
    ABLATION_GROUPS,
)

from src.preprocessing.sequence import create_sequences
from src.preprocessing.dataset import TransactionDataset

from src.models.gru import GRUClassifier

from src.training.trainer import (
    train_model,
    evaluate_model,
)

from src.utils.model_utils import save_model
from src.utils.metrics import save_metrics

from src.utils.scaler import (
    fit_scaler,
    save_scaler,
)

from src.utils.config import (
    ABLATION_MODEL_PATH,
    ABLATION_ENCODER_PATH,
    ABLATION_SCALER_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    TEST_SIZE,
    RANDOM_STATE,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

print("Loading ablation dataset...")

# ----------------------------------------
# Load + preprocess
# ----------------------------------------

df = load_data()

df, encoders = preprocess(df)

save_encoders(
    encoders,
    ABLATION_ENCODER_PATH,
)

# ----------------------------------------
# Features used by GRU
# sender_id is ONLY for grouping
# ----------------------------------------

FEATURE_SET = "all_features"

feature_columns = ABLATION_GROUPS[
    FEATURE_SET
]

print("=" * 60)
print("Ablation Feature Set")
print("=" * 60)

print(f"Number of Features : {len(feature_columns)}")

for feature in feature_columns:
    print(feature)

print("=" * 60)

# ----------------------------------------
# Build sequences
# ----------------------------------------
print(df["error_flag"].value_counts())
print(df["error_flag"].value_counts(normalize=True) * 100)

X, y = create_sequences(
    df=df,
    feature_columns=feature_columns,
    target_column="error_flag",
    sequence_length=10,
)

print("Sequence Shape:", X.shape)

# ----------------------------------------
# Scale features
# ----------------------------------------

samples, seq_len, features = X.shape

X_flat = X.reshape(-1, features)

X_flat, scaler = fit_scaler(X_flat)

save_scaler(
    scaler,
    ABLATION_SCALER_PATH,
)

X = X_flat.reshape(
    samples,
    seq_len,
    features,
)

# ----------------------------------------
# Split
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

train_dataset = TransactionDataset(
    X_train,
    y_train,
)

test_dataset = TransactionDataset(
    X_test,
    y_test,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
)

# ----------------------------------------
# Model
# ----------------------------------------

model = GRUClassifier(
    input_size=features,
    hidden_size=64,
    num_layers=2,
    num_classes=2,
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)

# ----------------------------------------
# Train
# ----------------------------------------

loss_history, accuracy_history, roc_auc_history = train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    EPOCHS,
    device,
)

# ----------------------------------------
# Evaluate
# ----------------------------------------

metrics = evaluate_model(
    model,
    test_loader,
    device,
    class_names=[
        "No Error",
        "Error",
    ],
    save_curve_data=True,
)


save_metrics(
    model_name=f"gru_{FEATURE_SET}",
    **metrics,
    loss_history=loss_history,
    accuracy_history=accuracy_history,
    roc_auc_history=roc_auc_history
)

# ----------------------------------------
# Save
# ----------------------------------------


save_model(
    model=model,
    model_path=ABLATION_MODEL_PATH,
    input_size=X_train.shape[2],
    hidden_size=64,
    num_layers=2,
    num_classes=2,
    classes=["No Error", "Error"],
)

print("\nGRU Ablation Model Saved.")