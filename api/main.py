from fastapi import FastAPI
from pydantic import BaseModel

from src.inference.predict import predict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Cash Transfer Error Reversal API",
    version="1.0.0",
    description="AI-powered transaction error classification system",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cash-transfer-error-reversal.onrender.com",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Request Model
# ==========================================

class Transaction(BaseModel):
    transaction_id: str
    sender_account: int
    receiver_account: int
    amount: float
    timestamp: str
    debit_status: str
    credit_status: str
    account_valid: bool
    beneficiary_match: bool
    retry_count: int
    transaction_type: str


# ==========================================
# Root
# ==========================================

@app.get("/")
def root():
    return {
        "message": "Cash Transfer Error Reversal API",
        "version": "1.0.0",
    }


# ==========================================
# Health Check
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded",
    }


# ==========================================
# Prediction Endpoint
# ==========================================

@app.post("/predict")
def predict_transaction(transaction: Transaction):

    result = predict(transaction.model_dump())

    return result