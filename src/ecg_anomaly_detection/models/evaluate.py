import torch
import numpy as np
import json
from ecg_anomaly_detection.models.lstm_autoencoder import LSTMAutoencoder


def compute_threshold(model, sequences, percentile: float, threshold_path: str):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(sequences).unsqueeze(-1)
        outputs = model(inputs)
        errors = torch.mean((outputs - inputs) ** 2, dim=(1, 2)).numpy()

    threshold = np.percentile(errors, percentile)

    with open(threshold_path, "w") as f:
        json.dump({"threshold": float(threshold)}, f)

    return threshold
