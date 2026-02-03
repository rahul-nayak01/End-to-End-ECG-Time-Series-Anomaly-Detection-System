import torch
from torch.utils.data import DataLoader, TensorDataset
from ecg_anomaly_detection.models.lstm_autoencoder import LSTMAutoencoder


def train_model(
    train_sequences,
    seq_len: int,
    n_features: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    model_path: str
):
    model = LSTMAutoencoder(seq_len, n_features, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    dataset = TensorDataset(torch.tensor(train_sequences).unsqueeze(-1))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), model_path)
