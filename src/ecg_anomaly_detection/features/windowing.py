import numpy as np


def create_sequences(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    Convert 1D ECG signal into overlapping sequences.
    """
    sequences = []
    for i in range(len(signal) - window_size):
        sequences.append(signal[i : i + window_size])
    return np.array(sequences)
