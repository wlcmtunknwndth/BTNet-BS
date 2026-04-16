"""BTNet for American options via ReLU transfer from a trained European network.

Experimental: copy weights from trained BTNetEuropean, add ReLU after each
hidden ConvLayer. No retraining on American prices is performed.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from btnn_bs.model_european import BTNetEuropean


class BTNetAmericanReLU(nn.Module):
    def __init__(self, european_model: BTNetEuropean) -> None:
        super().__init__()

        self._n_dim = european_model._n_dim
        self._initial_layer = copy.deepcopy(european_model._initial_layer)
        self._conv_layer = copy.deepcopy(european_model._conv_layer)

        for p in self.parameters():
            p.requires_grad_(False)

        self.eval()

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        if k.dim() == 1:
            k = k.unsqueeze(1)
        elif k.dim() == 2 and k.size(1) != 1:
            k = k[:, :1]

        k = k.float()
        x = self._initial_layer(k)

        for _ in range(self._n_dim):
            x = torch.relu(self._conv_layer(x))

        if x.dim() == 2 and x.size(1) == 1:
            return x.squeeze(1)
        if x.dim() == 2 and x.size(1) > 1:
            return x.mean(dim=1, keepdim=True)
        return x.unsqueeze(1) if x.dim() == 1 else x

    def predict(self, k) -> np.ndarray:
        with torch.no_grad():
            if isinstance(k, np.ndarray):
                k = torch.from_numpy(k).float()
            elif not isinstance(k, torch.Tensor):
                k = torch.tensor(k, dtype=torch.float32)

            if k.dim() == 1:
                k = k.unsqueeze(1)
            elif k.dim() == 2 and k.size(1) != 1:
                k = k[:, :1]

            return self.forward(k).cpu().numpy()
