# Model Deployment & Serving — Binary Classification API

A production-ready REST API that serves a trained binary classification model (breast cancer diagnosis) using **FastAPI** and **scikit-learn**.

**Live API:** https://model-deployment-v6fc.onrender.com
**Interactive Docs:** https://model-deployment-v6fc.onrender.com/docs

## Overview

This project packages a trained `RandomForestClassifier` (wrapped in a scikit-learn `Pipeline` with a `StandardScaler`) as a live REST API endpoint. The model predicts whether a tumor is malignant or benign based on 30 numeric features derived from cell nuclei measurements.

A key detail: the model uses a **custom decision threshold of 0.15** (instead of the default 0.5), deliberately tuned to minimize false negatives — since missing a malignant case is far costlier than a false alarm in a clinical context.

## Tech Stack

- **FastAPI** — web framework for the REST API
- **Uvicorn** — ASGI server
- **scikit-learn** — model pipeline (StandardScaler + RandomForestClassifier)
- **joblib** — model serialization
- **Render** — cloud hosting

## Project Structure

```
model-deployment/
├── main.py                 # FastAPI application
├── model/
│   └── model_bundle.joblib # Trained model + threshold + feature metadata
├── requirements.txt        # Python dependencies
└── README.md
```

## API Endpoints

### `GET /`
Returns basic API info and the expected feature order.

**Response:**
```json
{
  "message": "Model API is running",
  "expected_features": 30,
  "feature_order": ["mean radius", "mean texture", "..."]
}
```

### `POST /predict`
Runs a prediction on 30 input features (in the exact order returned by `GET /`).

**Request body:**
```json
{
  "features": [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.9075,
  "threshold_used": 0.15
}
```

- `prediction`: `1` = malignant, `0` = benign
- `probability`: model's raw predicted probability of the positive class
- `threshold_used`: the decision threshold applied to convert probability into a class label

## Running Locally

```bash
# Clone the repo
git clone https://github.com/SasiKumar-12/model-deployment.git
cd model-deployment

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the interactive API documentation.

## Deployment

This API is deployed on **Render** as a free-tier web service:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

> Note: the free tier spins down after inactivity, so the first request after idle time may take 30–50 seconds to respond.

## Model Details

- **Algorithm:** Random Forest Classifier (400 estimators)
- **Preprocessing:** StandardScaler
- **Test ROC-AUC:** 0.9947
- **Test F1 Score:** 0.8913
- **Decision threshold:** 0.15 (tuned for false-negative sensitivity)
