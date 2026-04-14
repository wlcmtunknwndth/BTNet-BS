"""Training loop for BTNet models."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


def train_BTNet(model, K_train, prices_train, epochs=200, lr=0.01, log_every: int = 50):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    if isinstance(K_train, np.ndarray):
        K_train = torch.from_numpy(K_train).float()
    if isinstance(prices_train, np.ndarray):
        prices_train = torch.from_numpy(prices_train).float()

    if K_train.dim() == 1:
        K_train = K_train.unsqueeze(1)
    if prices_train.dim() == 1:
        prices_train = prices_train.unsqueeze(1)

    loss_history = []

    for epoch in tqdm(range(epochs), desc="train_BTNet"):
        model.train()
        optimizer.zero_grad()

        predictions = model(K_train)

        if predictions.shape != prices_train.shape:
            predictions = predictions.view_as(prices_train)

        loss = loss_fn(predictions, prices_train)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if log_every and (epoch + 1) % log_every == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

    return loss_history
