from pathlib import Path

# ===========================
# Paths
# ===========================

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = BASE_DIR / "dataset/raw/transactions.csv"

MODEL_PATH = BASE_DIR / "saved_models/cash_transfer_reversal_model.pth"

ENCODER_PATH = BASE_DIR / "saved_models/encoders.pkl"

SCALER_PATH = BASE_DIR / "saved_models/scaler.pkl"

METADATA_PATH = BASE_DIR / "saved_models/model_metadata.json"

ERROR_MODEL_PATH = BASE_DIR / "saved_models/error_detector.pth"
LOGISTICS_MODEL_PATH = BASE_DIR / "saved_models/logistics_error_detector.pth"
ANN_MODEL_PATH = BASE_DIR / "saved_models/ann_error_detector.pth"
REASON_MODEL_PATH = BASE_DIR / "saved_models/reason_classifier.pth"

ERROR_ENCODER_PATH = BASE_DIR / "saved_models/error_encoders.pkl"
LOGISTICS_ENCODER_PATH =  BASE_DIR / "saved_models/logistics_error_encoders.pkl"
ANN_ENCODER_PATH = BASE_DIR / "saved_models/ann_error_encoders.pkl"
REASON_ENCODER_PATH = BASE_DIR / "saved_models/reason_encoders.pkl"

ERROR_SCALER_PATH = BASE_DIR / "saved_models/error_scaler.pkl"
LOGISTICS_SCALER_PATH = BASE_DIR / "saved_models/logistics_error_scaler.pkl"
ANN_SCALER_PATH = BASE_DIR / "saved_models/ann_error_scaler.pkl"
REASON_SCALER_PATH = BASE_DIR / "saved_models/reason_scaler.pkl"

METRIC_DIR = BASE_DIR / "src/metrics"

# ===========================
# Training
# ===========================

BATCH_SIZE = 32

LEARNING_RATE = 0.001

EPOCHS = 50

TEST_SIZE = 0.2

RANDOM_STATE = 42