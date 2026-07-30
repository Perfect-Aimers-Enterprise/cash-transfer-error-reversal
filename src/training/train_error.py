# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.utils.class_weight import compute_class_weight
# import numpy as np

# from torch.utils.data import DataLoader
# from sklearn.model_selection import train_test_split

# from src.preprocessing.preprocess import (
#     load_data,
#     preprocess,
#     save_encoders,
# )

# from src.preprocessing.sequence import create_sequences
# from src.preprocessing.dataset import TransactionDataset
# from src.models.gru import GRUClassifier

# from src.training.trainer import (
#     train_model,
#     evaluate_model,
# )

# from src.utils.model_utils import save_model

# from src.utils.scaler import (
#     fit_scaler,
#     save_scaler,
# )

# from src.utils.focal_loss import FocalLoss

# from src.utils.config import (
#     ERROR_MODEL_PATH,
#     ERROR_ENCODER_PATH,
#     ERROR_SCALER_PATH,
#     BATCH_SIZE,
#     LEARNING_RATE,
#     EPOCHS,
#     TEST_SIZE,
#     RANDOM_STATE,
# )

# device = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )

# print("Using device:", device)

# print("Loading dataset...")

# df = load_data()

# df, encoders = preprocess(df)

# save_encoders(
#     encoders,
#     ERROR_ENCODER_PATH,
# )

# # -------------------------
# # Features
# # -------------------------

# X = df.drop(
#     columns=[
#         "error_flag",
#         "reversal_reason",
#         "reversal_executed",
#     ]
# ).values

# y = df["error_flag"].values

# X, scaler = fit_scaler(X)

# save_scaler(
#     scaler,
#     ERROR_SCALER_PATH,
# )

# feature_columns = [
#     c
#     for c in df.columns
#     if c
#     not in [
#         "error_flag",
#         "reversal_reason",
#         "reversal_executed",
#     ]
# ]

# X, y = create_sequences(
#     df=df,
#     feature_columns=feature_columns,
#     target_column="error_flag",
#     sequence_length=10,
# )

# # -------------------------
# # Split
# # -------------------------

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=TEST_SIZE,
#     random_state=RANDOM_STATE,
#     stratify=y,
# )

# train_dataset = TransactionDataset(
#     X_train,
#     y_train,
# )

# test_dataset = TransactionDataset(
#     X_test,
#     y_test,
# )

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
# )

# test_loader = DataLoader(
#     test_dataset,
#     batch_size=BATCH_SIZE,
# )

# # -------------------------
# # Model
# # -------------------------

# model = GRUClassifier(
#     input_size=X_train.shape[2],
#     hidden_size=64,
#     num_layers=2,
#     num_classes=2,
# ).to(device)



# criterion = nn.CrossEntropyLoss()

# optimizer = optim.AdamW(
#     model.parameters(),
#     lr=1e-3,
#     weight_decay=1e-4,
# )

# train_model(
#     model,
#     train_loader,
#     criterion,
#     optimizer,
#     EPOCHS,
#     device,
# )

# evaluate_model(
#     model,
#     test_loader,
#     device,
#     class_names=["No Error", "Error"],
# )

# save_model(
#     model=model,
#     model_path=ERROR_MODEL_PATH,
#     input_size=X_train.shape[2],
#     num_classes=2,
#     classes=["No Error", "Error"],
# )

# print("\nError Detection Model Saved.")


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

from src.preprocessing.sequence import create_sequences
from src.preprocessing.dataset import TransactionDataset

from src.models.gru import GRUClassifier

from src.training.trainer import (
    train_model,
    evaluate_model,
)

from src.utils.model_utils import save_model

from src.utils.scaler import (
    fit_scaler,
    save_scaler,
)

from src.utils.config import (
    ERROR_MODEL_PATH,
    ERROR_ENCODER_PATH,
    ERROR_SCALER_PATH,
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
    ERROR_ENCODER_PATH,
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
    ERROR_SCALER_PATH,
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

evaluate_model(
    model,
    test_loader,
    device,
    class_names=[
        "No Error",
        "Error",
    ],
)

# ----------------------------------------
# Save
# ----------------------------------------

# save_model(
#     model=model,
#     model_path=ERROR_MODEL_PATH,
#     input_size=features,
#     num_classes=2,
#     classes=[
#         "No Error",
#         "Error",
#     ],
# )

save_model(
    model=model,
    model_path=ERROR_MODEL_PATH,
    input_size=X_train.shape[2],
    hidden_size=64,
    num_layers=2,
    num_classes=2,
    classes=["No Error", "Error"],
)

print("\nError Detection Model Saved.")