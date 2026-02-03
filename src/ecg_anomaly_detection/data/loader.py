import pandas as pd
import numpy as np
from typing import Tuple


def load_ecg_csv(file_path: str) -> np.ndarray:
    """
    Load ECG signal from CSV file.
    Assumes a single-column ECG signal.
    """
    df = pd.read_csv(file_path)
    signal = df.iloc[:, 0].values
    return signal.astype(np.float32)
