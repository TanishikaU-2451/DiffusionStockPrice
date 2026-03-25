from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import Settings, settings


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(inputs)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


class AutoencoderTrainer:
    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_matrix: np.ndarray) -> DenoisingAutoencoder:
        torch.manual_seed(self.config.random_state)
        model = DenoisingAutoencoder(
            input_dim=train_matrix.shape[1],
            latent_dim=self.config.latent_dim,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()

        tensor_data = torch.from_numpy(train_matrix).float().to(self.device)
        dataset = torch.utils.data.TensorDataset(tensor_data)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        model.train()
        for _ in range(self.config.epochs):
            for (batch,) in loader:
                noisy_batch = batch + torch.randn_like(batch) * self.config.noise_std
                reconstruction, _ = model(noisy_batch)
                loss = criterion(reconstruction, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model.eval()

    def reconstruct_and_embed(
        self,
        model: DenoisingAutoencoder,
        data_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        tensor_data = torch.from_numpy(data_matrix).float().to(self.device)
        with torch.no_grad():
            reconstruction, latent = model(tensor_data)
        return (
            reconstruction.cpu().numpy().astype(np.float32),
            latent.cpu().numpy().astype(np.float32),
        )
