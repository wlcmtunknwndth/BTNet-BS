"""BTNet architectures: European (ReLU + conv)"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from btnn_bs.layers import ConvLayer, DenseLayer

class BTNetEuropean(nn.Module):
    def __init__(self, n_dim, S0, sig, T, t0, r=None):
        super().__init__()

        self._n_dim = n_dim
        self._S0 = S0
        self._sig = sig
        self._T = T
        self._t0 = t0
        self._r = r if r is not None else 0.05

        dt = (T - t0) / n_dim
        sqrt_dt = np.sqrt(dt)

        u = np.exp(sig * sqrt_dt)
        d = np.exp(-sig * sqrt_dt)

        w = np.array(
            [
                (np.exp(-self._r * dt) * u - 1) / (u - d),
                (1 - np.exp(-self._r * dt) * d) / (u - d),
            ],
            dtype=np.float32,
        )

        w_init = np.ones((1, n_dim + 1), dtype=np.float32)
        b_init = np.array(
            [-S0 * np.exp(sig * sqrt_dt * (2 * j - n_dim)) for j in range(n_dim + 1)],
            dtype=np.float32,
        )

        self._initial_layer = DenseLayer(
            input_dim=1,
            output_dim=n_dim + 1,
            W=w_init,
            bias=b_init,
            transformation=torch.relu,
        )

        self._conv_layer = ConvLayer(filter_weight=w)

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        if k.dim() == 1:
            k = k.unsqueeze(1)
        elif k.dim() == 2 and k.size(1) != 1:
            k = k[:, :1]

        k = k.float()

        x = self._initial_layer(k)

        for _ in range(self._n_dim):
            x = self._conv_layer(x)

        if x.dim() == 2 and x.size(1) == 1:
            return x.squeeze(1)
        if x.dim() == 2 and x.size(1) > 1:
            return x.mean(dim=1, keepdim=True)
        return x.unsqueeze(1) if x.dim() == 1 else x

    def predict(self, k):
        self.eval()

        with torch.no_grad():
            if isinstance(k, np.ndarray):
                k = torch.from_numpy(k).float()
            elif not isinstance(k, torch.Tensor):
                k = torch.tensor(k, dtype=torch.float32)

            if k.dim() == 1:
                k = k.unsqueeze(1)
            elif k.dim() == 2 and k.size(1) != 1:
                k = k[:, :1]

            outputs = self.forward(k)

        return outputs.cpu().numpy()
