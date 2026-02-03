import torch
import json
import numpy as np
from ecg_anomaly_detection.models.lstm_autoencoder import LSTMAutoencoder


class ECGAnomalyDetector:
    def __init__(self, model_path: str, threshold_path: str, seq_len: int, n_features: int, hidden_dim: int):
        self.model = LSTMAutoencoder(seq_len, n_features, hidden_dim)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        with open(threshold_path) as f:
            self.threshold = json.load(f)["threshold"]

    def predict(self, sequence: np.ndarray):
        with torch.no_grad():
            x = torch.tensor(sequence).unsqueeze(0).unsqueeze(-1)
            recon = self.model(x)
            error = torch.mean((recon - x) ** 2).item()

        return {
            "anomaly": error > self.threshold,
            "score": error,
            "threshold": self.threshold
        }
