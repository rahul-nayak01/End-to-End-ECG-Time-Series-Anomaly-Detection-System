import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def scale_signal(signal: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize ECG signal.
    """
    scaler = StandardScaler()
    signal_scaled = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
    return signal_scaled, scaler
