import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden_dim: int):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=n_features,
            batch_first=True
        )

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded
