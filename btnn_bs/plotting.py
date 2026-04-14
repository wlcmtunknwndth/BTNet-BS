"""Matplotlib helpers for BTNet experiments."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_comparison(K, bs_prices, nn_prices, crr_prices, title="Option Price Comparison"):
    plt.figure(figsize=(10, 6))

    K = np.array(K).flatten()
    bs_prices = np.array(bs_prices).flatten()
    nn_prices = np.array(nn_prices).flatten()
    crr_prices = np.array(crr_prices).flatten()

    plt.plot(K, bs_prices, "b--", label="Black-Scholes")
    plt.plot(K, nn_prices, "r--", label="BTNet")
    plt.plot(K, crr_prices, "g--", label="Cox-Ross-Rubinstein")

    plt.xlabel("Strike Price K")
    plt.ylabel("Option Price")
    plt.legend()
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_errors(K, bs_prices, nn_prices, crr_prices, title="Prediction Errors"):
    plt.figure(figsize=(12, 8))

    K = np.array(K).flatten()
    bs_prices = np.array(bs_prices).flatten()
    nn_prices = np.array(nn_prices).flatten()
    crr_prices = np.array(crr_prices).flatten()

    errors_nn_vs_bs = nn_prices - bs_prices
    errors_crr_vs_bs = crr_prices - bs_prices

    plt.subplot(2, 1, 1)
    plt.plot(K, errors_nn_vs_bs, "r-", linewidth=2, label="NN - BS")
    plt.plot(K, errors_crr_vs_bs, "g--", linewidth=2, label="CRR - BS")

    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.xlabel("Strike Price K")
    plt.ylabel("Error vs Black Scholes")
    plt.title(f"{title} - Black-Scholes")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    errors_nn_vs_crr = nn_prices - crr_prices
    plt.plot(K, errors_nn_vs_crr, "purple", linewidth=2, label="NN - CRR")

    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.fill_between(K, 0, errors_nn_vs_crr, alpha=0.3, color="purple")
    plt.xlabel("Strike Price K")
    plt.ylabel("Error (NN - CRR)")
    plt.title(f"{title} - CRR Tree")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_training_losses(loss_hist_euro, loss_hist_amer, title="Training Losses"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_hist_euro, "b-", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("European Model Training Loss")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    plt.subplot(1, 2, 2)
    plt.plot(loss_hist_amer, "r-", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("American Model Training Loss")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_prices_with_quantlib(
    K,
    bs_prices,
    nn_prices,
    crr_prices,
    ql_prices,
    title="Option prices vs QuantLib",
):
    plt.figure(figsize=(10, 6))

    K = np.array(K).flatten()
    bs_prices = np.array(bs_prices).flatten()
    nn_prices = np.array(nn_prices).flatten()
    crr_prices = np.array(crr_prices).flatten()
    ql_prices = np.array(ql_prices).flatten()

    plt.plot(K, bs_prices, "b--", label="Black–Scholes")
    plt.plot(K, nn_prices, "r-", label="BTNet")
    plt.plot(K, crr_prices, "g--", label="CRR / IBT")
    plt.plot(K, ql_prices, "k.", markersize=4, label="QuantLib")

    plt.xlabel("Strike K")
    plt.ylabel("Put price")
    plt.legend()
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_errors_vs_quantlib(
    K,
    ql_ref,
    *,
    nn_prices=None,
    bs_prices=None,
    crr_prices=None,
    title="Errors vs QuantLib",
):
    K = np.array(K).flatten()
    ql_ref = np.array(ql_ref).flatten()

    panels = []
    if nn_prices is not None:
        panels.append(("NN − QL", np.array(nn_prices).flatten()))
    if bs_prices is not None:
        panels.append(("BS − QL", np.array(bs_prices).flatten()))
    if crr_prices is not None:
        panels.append(("CRR − QL", np.array(crr_prices).flatten()))

    if not panels:
        return

    n = len(panels)
    plt.figure(figsize=(10, 3.2 * n))
    for i, (name, y) in enumerate(panels):
        plt.subplot(n, 1, i + 1)
        err = y - ql_ref
        plt.plot(K, err, linewidth=2, label=name)
        plt.axhline(0, color="k", linestyle="-", alpha=0.25)
        plt.fill_between(K, 0, err, alpha=0.2)
        plt.xlabel("Strike K")
        plt.ylabel("Price diff")
        plt.title(f"{title} — {name}")
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
