"""BTNet-style networks and helpers for binomial-tree option pricing (Black–Scholes setting)."""

import importlib

from btnn_bs.analytics import american_put_prices_binomial, bs_put_price
from btnn_bs.layers import ConvLayer, DenseLayer, MaxoutLayer
from btnn_bs.model_american import BTNetAmerican
from btnn_bs.model_american_relu import BTNetAmericanReLU
from btnn_bs.model_european import BTNetEuropean

# Import as submodule then bind names so we can reload if the kernel cached an old plotting.py
# (missing QuantLib plot helpers) before the file was updated.
_plotting = importlib.import_module("btnn_bs.plotting")
if not hasattr(_plotting, "plot_prices_with_quantlib"):
    _plotting = importlib.reload(_plotting)
for _n in ("plot_prices_with_quantlib", "plot_errors_vs_quantlib"):
    if not hasattr(_plotting, _n):
        raise ImportError(
            f"btnn_bs.plotting is missing {_n}; restart the Jupyter kernel or reinstall editable btnn-bs."
        )

plot_comparison = _plotting.plot_comparison
plot_errors = _plotting.plot_errors
plot_training_losses = _plotting.plot_training_losses
plot_prices_with_quantlib = _plotting.plot_prices_with_quantlib
plot_errors_vs_quantlib = _plotting.plot_errors_vs_quantlib
from btnn_bs.quantlib import (
    QuantLibBenchmarkResult,
    american_put_grid_baw,
    american_put_grid_crr,
    error_stats,
    european_put_grid,
    print_comparison_table,
    run_quantlib_benchmark,
)
from btnn_bs.training import train_BTNet
from btnn_bs.tree import MyIBT_CRR
from btnn_bs.greeks import (
    btnet_greeks,
    btnet_american_greeks,
    btnet_american_greeks_fixed_W,
    american_greeks_fd,
    bs_greeks,
    greeks_error_table,
    plot_greeks,
    plot_greeks_errors,
    plot_american_greeks,
    plot_american_greeks_errors,
    plot_american_greeks_transfer,
    plot_american_greeks_transfer_errors,
)

__all__ = [
    "MyIBT_CRR",
    "ConvLayer",
    "DenseLayer",
    "MaxoutLayer",
    "BTNetEuropean",
    "BTNetAmerican",
    "BTNetAmericanReLU",
    "train_BTNet",
    "bs_put_price",
    "american_put_prices_binomial",
    "plot_comparison",
    "plot_errors",
    "plot_training_losses",
    "plot_prices_with_quantlib",
    "plot_errors_vs_quantlib",
    "run_quantlib_benchmark",
    "QuantLibBenchmarkResult",
    "european_put_grid",
    "american_put_grid_crr",
    "american_put_grid_baw",
    "error_stats",
    "print_comparison_table",
    # Greeks — European
    "btnet_greeks",
    "bs_greeks",
    "greeks_error_table",
    "plot_greeks",
    "plot_greeks_errors",
    # Greeks — American
    "btnet_american_greeks",
    "american_greeks_fd",
    "plot_american_greeks",
    "plot_american_greeks_errors",
    # Greeks — weight-transfer experiment
    "btnet_american_greeks_fixed_W",
    "plot_american_greeks_transfer",
    "plot_american_greeks_transfer_errors",
]
