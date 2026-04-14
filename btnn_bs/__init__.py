"""BTNet-style networks and helpers for binomial-tree option pricing (Black–Scholes setting)."""

from btnn_bs.analytics import american_put_prices_binomial, bs_put_price
from btnn_bs.layers import ConvLayer, DenseLayer, MaxoutLayer
from btnn_bs.model_american import BTNetAmerican
from btnn_bs.model_european import BTNetEuropean
from btnn_bs.plotting import (
    plot_comparison,
    plot_errors,
    plot_errors_vs_quantlib,
    plot_prices_with_quantlib,
    plot_training_losses,
)
from btnn_bs.quantlib_ref import (
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

__all__ = [
    "MyIBT_CRR",
    "ConvLayer",
    "DenseLayer",
    "MaxoutLayer",
    "BTNetEuropean",
    "BTNetAmerican",
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
]
