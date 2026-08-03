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
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Request Model
# ==========================================

class Transaction(BaseModel):
    transfer_id: str
    sender_id: str
    beneficiary_id: str
    amount: float
    timestamp: str
    channel: str
    location: str
    device_id: str
    session_id: str


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