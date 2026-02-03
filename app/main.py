from fastapi import FastAPI
import numpy as np
from ecg_anomaly_detection.models.inference import ECGAnomalyDetector

app = FastAPI(title="ECG Anomaly Detection API")

# Load model ONCE at startup
detector = ECGAnomalyDetector(
    model_path="artifacts/model.pth",
    threshold_path="artifacts/threshold.json",
    seq_len=140,
    n_features=1,
    hidden_dim=64
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(ecg: list[float]):
    sequence = np.array(ecg, dtype=np.float32)
    result = detector.predict(sequence)
    return result
