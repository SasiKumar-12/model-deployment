from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np

# Load the bundle
bundle = joblib.load("model/model_bundle.joblib")
model = bundle["model"]
threshold = bundle["threshold"]
feature_cols = bundle["feature_cols"]

app = FastAPI(title="Breast Cancer Binary Classification API")

class InputData(BaseModel):
    features: List[float] = Field(
        ..., 
        description=f"List of {len(feature_cols)} numeric features in this exact order: {feature_cols}"
    )

@app.get("/")
def home():
    return {
        "message": "Model API is running",
        "expected_features": len(feature_cols),
        "feature_order": feature_cols
    }

@app.post("/predict")
def predict(data: InputData):
    if len(data.features) != len(feature_cols):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(feature_cols)} features, got {len(data.features)}"
        )

    input_array = np.array([data.features])
    probability = model.predict_proba(input_array)[0][1]
    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "probability": float(probability),
        "threshold_used": threshold
    }