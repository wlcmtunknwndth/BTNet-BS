"""Reference option prices (Black–Scholes and one-step binomial backward induction)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bs_put_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def american_put_prices_binomial(S0, K, T, r, sigma, n=100):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    S = np.zeros(n + 1)
    for j in range(n + 1):
        S[j] = S0 * (u**j) * (d ** (n - j))

    V = np.maximum(K - S, 0)

    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            S_ij = S0 * (u**j) * (d ** (i - j))
            V[j] = max(
                K - S_ij,
                discount * (p * V[j + 1] + (1 - p) * V[j]),
            )

    return V[0]
