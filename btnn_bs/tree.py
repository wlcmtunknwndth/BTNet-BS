"""Implied binomial tree (CRR-style) helper used in the original notebook."""

from __future__ import annotations

import numpy as np


class MyIBT_CRR:
    def __init__(self, n):
        self.n = n
        tri_size = int((n + 1) * (n + 2) / 2)
        self.S_ = np.zeros(shape=(tri_size,), dtype=np.float32)
        self.q_ = np.zeros(shape=(tri_size,), dtype=np.float32)
        self.V_ = np.zeros(shape=(tri_size,), dtype=np.float32)

    def _set_S(self, i, j, val):
        self.S_[int(i * (i + 1) / 2) + j] = val

    def _get_S(self, i, j):
        return self.S_[int(i * (i + 1) / 2) + j]

    def _set_q(self, i, j, val):
        self.q_[int(i * (i + 1) / 2) + j] = val

    def _get_q(self, i, j):
        return self.q_[int(i * (i + 1) / 2) + j]

    def _set_V(self, i, j, val):
        self.V_[int(i * (i + 1) / 2) + j] = val

    def _get_V(self, i, j):
        return self.V_[int(i * (i + 1) / 2) + j]

    def build(self, S0, t0, T, r, sigma, teta):
        self.S0 = S0
        self.t0 = t0
        self.T = T
        self.r = r
        self.sigma = sigma

        dt = (T - t0) / self.n
        sqdt = np.sqrt(dt)

        for k in range((self.n // 2) + 1):
            self._set_S(2 * k, k, S0)

            for j in range(1, self.n - 2 * k + 1):
                t_ = t0 + (2 * k + j - 1) * dt
                self._set_S(
                    2 * k + j,
                    k,
                    self._get_S(2 * k + j - 1, k)
                    * np.exp(-sigma(self._get_S(2 * k + j - 1, k), t_, teta) * sqdt),
                )
                self._set_S(
                    2 * k + j,
                    k + j,
                    self._get_S(2 * k + j - 1, k + j - 1)
                    * np.exp(sigma(self._get_S(2 * k + j - 1, k + j - 1), t_, teta) * sqdt),
                )

        for i in range(self.n):
            for j in range(i + 1):
                self._set_q(
                    i,
                    j,
                    (self._get_S(i, j) * np.exp(r * dt) - self._get_S(i + 1, j))
                    / (self._get_S(i + 1, j + 1) - self._get_S(i + 1, j)),
                )

    def eval_european_call(self, K):
        dt = (self.T - self.t0) / self.n

        for j in range(self.n + 1):
            self._set_V(self.n, j, max(self._get_S(self.n, j) - K, 0))

        for i in range(self.n - 1, -1, -1):
            for j in range(i + 1):
                self._set_V(
                    i,
                    j,
                    np.exp(-self.r * dt)
                    * (
                        self._get_V(i + 1, j + 1) * self._get_q(i, j)
                        + self._get_V(i + 1, j) * (1 - self._get_q(i, j))
                    ),
                )

        return self._get_V(0, 0)

    def eval_americal_call(self, K):
        dt = (self.T - self.t0) / self.n

        for j in range(self.n + 1):
            self._set_V(self.n, j, max(K - self._get_S(self.n, j), 0))

        for i in range(self.n - 1, -1, -1):
            for j in range(i + 1):
                self._set_V(
                    i,
                    j,
                    max(
                        np.exp(-self.r * dt)
                        * (
                            self._get_V(i + 1, j + 1) * self._get_q(i, j)
                            + self._get_V(i + 1, j) * (1 - self._get_q(i, j))
                        ),
                        self._get_S(i, j) - K,
                    ),
                )

        return self._get_V(0, 0)

    def eval_european_put(self, K):
        dt = (self.T - self.t0) / self.n

        for j in range(self.n + 1):
            self._set_V(self.n, j, max(K - self._get_S(self.n, j), 0))

        for i in range(self.n - 1, -1, -1):
            for j in range(i + 1):
                self._set_V(
                    i,
                    j,
                    np.exp(-self.r * dt)
                    * (
                        self._get_V(i + 1, j + 1) * self._get_q(i, j)
                        + self._get_V(i + 1, j) * (1 - self._get_q(i, j))
                    ),
                )

        return self._get_V(0, 0)

    def eval_american_put(self, K):
        dt = (self.T - self.t0) / self.n

        for j in range(self.n + 1):
            self._set_V(self.n, j, max(K - self._get_S(self.n, j), 0))

        for i in range(self.n - 1, -1, -1):
            for j in range(i + 1):
                self._set_V(
                    i,
                    j,
                    max(
                        np.exp(-self.r * dt)
                        * (
                            self._get_V(i + 1, j + 1) * self._get_q(i, j)
                            + self._get_V(i + 1, j) * (1 - self._get_q(i, j))
                        ),
                        K - self._get_S(i, j),
                    ),
                )

        return self._get_V(0, 0)

    def plot(self):
        import matplotlib.pyplot as plt

        x_lst = []
        y_lst = []
        dt = (self.T - self.t0) / self.n
        for i in range(self.n + 1):
            t_ = self.t0 + i * dt
            for j in range(i + 1):
                x_lst.append(t_)
                y_lst.append(self._get_S(i, j))

        plt.figure(figsize=(8, 6))
        plt.scatter(x_lst, y_lst, c="red")
        plt.title("IBT Nodes", fontsize=20)
        plt.ylabel(r"Price $S$", fontsize=15, labelpad=10)
        plt.xlabel(r"Time $t$", fontsize=15, labelpad=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

    @property
    def min(self):
        return self._get_S(self.n, 0)

    @property
    def max(self):
        return self._get_S(self.n, self.n)
