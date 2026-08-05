import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.preprocessing.preprocess import (
    load_data,
    preprocess,
    save_encoders,
)

from src.preprocessing.dataset import TransactionDataset

from src.models.logistic_regression import LogisticRegressionClassifier

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
    LOGISTICS_MODEL_PATH,
    LOGISTICS_ENCODER_PATH,
    LOGISTICS_SCALER_PATH,
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

print("Loading dataset...")

# ----------------------------------------
# Load + preprocess
# ----------------------------------------

df = load_data()

df, encoders = preprocess(df)

save_encoders(
    encoders,
    LOGISTICS_ENCODER_PATH,
)

# ----------------------------------------
# Features used by GRU
# sender_id is ONLY for grouping
# ----------------------------------------

feature_columns = [
    "amount",
    "timestamp",
    "channel",
    "location",
]

# ----------------------------------------
# Build sequences
# ----------------------------------------
print(df["error_flag"].value_counts())
print(df["error_flag"].value_counts(normalize=True) * 100)

feature_columns = [
    "amount",
    "timestamp",
    "channel",
    "location",
]

X = df[feature_columns].values

y = df["error_flag"].values


# ----------------------------------------
# Scale features
# ----------------------------------------


X, scaler = fit_scaler(X)

save_scaler(
    scaler,
    LOGISTICS_SCALER_PATH,
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

model = LogisticRegressionClassifier(
    input_size=X_train.shape[1],
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

train_model(
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
)

save_metrics(
    model_name="Logistic Regression",
    **metrics,
)

# ----------------------------------------
# Save
# ----------------------------------------

save_model(
    model=model,
    model_path=LOGISTICS_MODEL_PATH,
    input_size=X_train.shape[1],
    num_classes=2,
    classes=[
        "No Error",
        "Error",
    ],
    hidden_size=None,
    num_layers=None,
)

print("\nError Detection Model Saved.")