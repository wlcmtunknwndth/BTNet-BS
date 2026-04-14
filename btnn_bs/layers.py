"""Fixed-weight layers used by BTNet (CRR-style backward induction as convolutions)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, filter_weight=None):
        super().__init__()

        self._conv_1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=2,
            bias=False,
        )

        if filter_weight is not None:
            with torch.no_grad():
                self._conv_1d.weight.data = torch.as_tensor(
                    filter_weight.reshape(1, 1, 2),
                    dtype=torch.float32,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self._conv_1d(x)
        return x.squeeze(1)


class DenseLayer(nn.Module):
    def __init__(self, input_dim, output_dim, W=None, bias=None, transformation=None):
        super().__init__()

        self._linear = nn.Linear(
            in_features=input_dim,
            out_features=output_dim,
            bias=True,
        )

        if W is not None:
            with torch.no_grad():
                if isinstance(W, np.ndarray):
                    W = torch.as_tensor(data=W, dtype=torch.float32)
                if W.dim() == 1:
                    W = W.unsqueeze(0)

                self._linear.weight.data = W.T if W.shape[0] == input_dim else W

        if bias is not None:
            with torch.no_grad():
                if isinstance(bias, np.ndarray):
                    bias = torch.as_tensor(data=bias, dtype=torch.float32)
                self._linear.bias.data = bias.flatten()

        self._transformation = transformation if transformation is not None else (lambda t: t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._linear(x)
        return self._transformation(out)


class MaxoutLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        filter_weight=None,
        W_linear=None,
        bias=None,
    ):
        super().__init__()

        self._conv_1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=2,
            bias=False,
        )

        self._linear = nn.Linear(
            in_features=1,
            out_features=output_dim,
            bias=True,
        )

        if filter_weight is not None:
            with torch.no_grad():
                self._conv_1d.weight.data = torch.as_tensor(
                    data=filter_weight.reshape(1, 1, 2),
                    dtype=torch.float32,
                )

        if W_linear is not None:
            with torch.no_grad():
                if isinstance(W_linear, np.ndarray):
                    W_linear = torch.as_tensor(W_linear, dtype=torch.float32)
                if W_linear.dim() == 1:
                    W_linear = W_linear.unsqueeze(0)

                self._linear.weight.data = W_linear.T

        if bias is not None:
            with torch.no_grad():
                if isinstance(bias, np.ndarray):
                    bias = torch.as_tensor(bias, dtype=torch.float32)

                self._linear.bias.data = bias.flatten()

    def forward(self, v_prev: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        v_conv = v_prev.unsqueeze(1)
        v_conv = self._conv_1d(v_conv)
        v_conv = v_conv.squeeze(1)

        v_linear = self._linear(k)

        return torch.max(v_conv, v_linear)
