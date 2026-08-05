# import torch
# import torch.nn as nn
# import torch.optim as optim

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

# from src.utils.config import (
#     REASON_MODEL_PATH,
#     REASON_ENCODER_PATH,
#     REASON_SCALER_PATH,
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

# # -------------------------
# # Keep only error rows
# # -------------------------

# df = df[
#     df["error_flag"] == 1
# ].copy()

# save_encoders(
#     encoders,
#     REASON_ENCODER_PATH,
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

# y = df["reversal_reason"].values

# X, scaler = fit_scaler(X)

# save_scaler(
#     scaler,
#     REASON_SCALER_PATH,
# )

# X, y = create_sequences(
#     X,
#     y,
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

# num_classes = len(
#     encoders["reversal_reason"].classes_
# )

# model = GRUClassifier(
#     input_size=X_train.shape[2],
#     hidden_size=64,
#     num_layers=2,
#     num_classes=num_classes,
# ).to(device)

# criterion = nn.CrossEntropyLoss()

# optimizer = optim.Adam(
#     model.parameters(),
#     lr=LEARNING_RATE,
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
#     class_names=list(
#         encoders["reversal_reason"].classes_
#     ),
# )

# save_model(
#     model=model,
#     model_path=REASON_MODEL_PATH,
#     input_size=X_train.shape[2],
#     num_classes=num_classes,
#     classes=list(
#         encoders["reversal_reason"].classes_
#     ),
# )

# print("\nReason Classification Model Saved.")

import torch
import torch.nn as nn
import torch.optim as optim

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
    REASON_MODEL_PATH,
    REASON_ENCODER_PATH,
    REASON_SCALER_PATH,
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

# -----------------------------------
# Load & preprocess
# -----------------------------------

df = load_data()

df, encoders = preprocess(df)

# -----------------------------------
# Keep only transactions with errors
# -----------------------------------

df = df[
    df["error_flag"] == 1
].copy()

print(df["reversal_reason"].value_counts())

save_encoders(
    encoders,
    REASON_ENCODER_PATH,
)

# -----------------------------------
# Feature columns
# -----------------------------------

feature_columns = [
    c
    for c in df.columns
    if c not in [
        "transfer_id",
        "sender_id",
        "beneficiary_id",
        "device_id",
        "session_id",
        "error_flag",
        "reversal_reason",
        "reversal_executed",
    ]
]

# -----------------------------------
# Scale ONLY feature columns
# -----------------------------------

print("\nFeatures used:")
print(feature_columns)

df[feature_columns], scaler = fit_scaler(
    df[feature_columns]
)

save_scaler(
    scaler,
    REASON_SCALER_PATH,
)

# -----------------------------------
# Create GRU sequences
# -----------------------------------

X, y = create_sequences(
    df=df,
    feature_columns=feature_columns,
    target_column="reversal_reason",
    sequence_length=10,
)

print(f"\nGenerated sequences: {len(X):,}")

import numpy as np

print("Unique labels:", np.unique(y))
print("Number of labels:", len(np.unique(y)))

for label in np.unique(y):
    print(
        encoders["reversal_reason"].inverse_transform([label])[0]
    )

# -----------------------------------
# Train/Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

# -----------------------------------
# Datasets
# -----------------------------------

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

# -----------------------------------
# Model
# -----------------------------------

num_classes = len(
    encoders["reversal_reason"].classes_
)

model = GRUClassifier(
    input_size=X_train.shape[2],
    hidden_size=64,
    num_layers=2,
    num_classes=num_classes,
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)

# -----------------------------------
# Train
# -----------------------------------

loss_history, accuracy_history, roc_auc_history = train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    EPOCHS,
    device,
)

# -----------------------------------
# Evaluate
# -----------------------------------

# evaluate_model(
#     model,
#     test_loader,
#     device,
#     class_names=list(
#         encoders["reversal_reason"].classes_
#     ),
# )
classes_present = sorted(set(y))

class_names = [
    encoders["reversal_reason"].inverse_transform([c])[0]
    for c in classes_present
]

metrics = evaluate_model(
    model,
    test_loader,
    device,
    class_names=class_names,
)

# -----------------------------------
# Save model
# -----------------------------------

# save_model(
#     model=model,
#     model_path=REASON_MODEL_PATH,
#     input_size=X_train.shape[2],
#     num_classes=num_classes,
#     classes=list(
#         encoders["reversal_reason"].classes_
#     ),
# )
save_model(
    model=model,
    model_path=REASON_MODEL_PATH,
    input_size=X_train.shape[2],
    hidden_size=64,
    num_layers=2,
    num_classes=num_classes,
    classes=list(encoders["reversal_reason"].classes_),
    **metrics,
    loss_history=loss_history,
    accuracy_history=accuracy_history,
    roc_auc_history=roc_auc_history
)
print("\nReason Classification Model Saved.")