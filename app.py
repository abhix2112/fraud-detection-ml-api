import torch
import torch.nn as nn
import numpy as np
import joblib

from fastapi import FastAPI
from pydantic import BaseModel

# --------------------------
# Define Model Architecture
# --------------------------

class FraudNet(nn.Module):

    def __init__(self, input_size):
        super(FraudNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


# --------------------------
# Load Model + Scaler
# --------------------------

INPUT_SIZE = 30  # features in creditcard dataset

model = FraudNet(INPUT_SIZE)
model.load_state_dict(torch.load("fraud_model.pth"))
model.eval()

scaler = joblib.load("scaler.pkl")

print("Model and scaler loaded successfully.")

# --------------------------
# FastAPI App
# --------------------------

app = FastAPI(title="Fraud Detection API")


# --------------------------
# Input Schema
# --------------------------

class Transaction(BaseModel):
    features: list


# --------------------------
# Prediction Endpoint
# --------------------------

@app.post("/predict")
def predict_fraud(data: Transaction):

    transaction = np.array(data.features).reshape(1, -1)

    scaled = scaler.transform(transaction)

    tensor_input = torch.FloatTensor(scaled)

    with torch.no_grad():

        logits = model(tensor_input)

        prob = torch.sigmoid(logits).item()

    return {
    "fraud_probability": float(prob),
    "fraud": prob > 0.02,
    "risk_level": (
        "low" if prob < 0.01
        else "medium" if prob < 0.05
        else "high"
    )
}