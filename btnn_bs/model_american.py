"""BTNet architectures: American (maxout layers)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from btnn_bs.layers import DenseLayer, MaxoutLayer

class BTNetAmerican(nn.Module):
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

        self._W = np.array(
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

        self._maxout_layers = nn.ModuleList()
        for i in range(n_dim):
            input_dim = n_dim + 1 - i
            output_dim = n_dim - i

            w_linear = np.ones((1, output_dim), dtype=np.float32)
            b_linear = np.array(
                [
                    -S0 * np.exp(sig * sqrt_dt * (2 * j - (n_dim - 1 - i)))
                    for j in range(output_dim)
                ],
                dtype=np.float32,
            )

            self._maxout_layers.append(
                MaxoutLayer(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    filter_weight=self._W,
                    W_linear=w_linear,
                    bias=b_linear,
                )
            )

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        if k.dim() == 1:
            k = k.unsqueeze(1)
        elif k.dim() == 2 and k.size(1) != 1:
            k = k[:, :1]

        k = k.float()
        batch_size = k.size(0)

        v = self._initial_layer(k)

        for maxout_layer in self._maxout_layers:
            v = maxout_layer(v, k)

        if v.dim() == 2 and v.size(1) == 1:
            return v
        return v.view(batch_size, -1)[:, :1]

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

            return self.forward(k).cpu().numpy()
