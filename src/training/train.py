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

from src.preprocessing.dataset import TransactionDataset

from src.models.mlp import TransactionClassifier

from src.training.trainer import (
    train_model,
    evaluate_model,
)

from src.utils.model_utils import save_model

from src.utils.config import (
    MODEL_PATH,
    ENCODER_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    TEST_SIZE,
    RANDOM_STATE,
    SCALER_PATH
)

from src.utils.scaler import (
    fit_scaler,
    save_scaler,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

# ====================================
# Load Dataset
# ====================================

print("Loading dataset...")

df = load_data()

df, encoders = preprocess(df)

save_encoders(
    encoders,
    ENCODER_PATH,
)

# ====================================
# Features & Labels
# ====================================

X = df.drop(columns=["error_label"]).values

X, scaler = fit_scaler(X)

save_scaler(
    scaler,
    SCALER_PATH,
)

y = df["error_label"].values

# ====================================
# Train/Test Split
# ====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

# ====================================
# Datasets
# ====================================

train_dataset = TransactionDataset(X_train, y_train)
test_dataset = TransactionDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# ====================================
# Model
# ====================================

input_size = X_train.shape[1]
num_classes = len(set(y))


model = TransactionClassifier(
    input_size=input_size,
    num_classes=num_classes,
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

# ====================================
# Train
# ====================================

train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    EPOCHS,
    device,
)

# ====================================
# Evaluate
# ====================================

evaluate_model(
    model,
    test_loader,
    device,
)

# ====================================
# Save Model
# ====================================

save_model(
    model=model,
    model_path=MODEL_PATH,
    input_size=input_size,
    num_classes=num_classes,
    classes=list(encoders["error_label"].classes_),
)

print(f"\nModel saved to:\n{MODEL_PATH}")