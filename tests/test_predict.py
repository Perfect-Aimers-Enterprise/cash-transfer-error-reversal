from src.inference.predict import predict

sample = {
    "transfer_id": "TXN999999",
    "sender_id": "SND123456",
    "beneficiary_id": "BEN654321",
    "amount": 45000.50,
    "timestamp": "2026-07-30 14:20:00",
    "channel": "mobile",
    "location": "Abuja",
    "device_id": "DEV12345678",
    "session_id": "abc123def4567890",
}

print(predict(sample))