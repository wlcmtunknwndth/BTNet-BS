"""BTNet-style networks and helpers for binomial-tree option pricing (Black–Scholes setting)."""

from btnn_bs.analytics import american_put_prices_binomial, bs_put_price
from btnn_bs.layers import ConvLayer, DenseLayer, MaxoutLayer
from btnn_bs.models import BTNetAmerican, BTNetEuropean
from btnn_bs.plotting import plot_comparison, plot_errors, plot_training_losses
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
]
