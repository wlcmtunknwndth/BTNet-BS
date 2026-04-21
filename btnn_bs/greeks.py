"""Greeks for BTNet via PyTorch autograd + Black-Scholes analytical formulas.

Supported Greeks (European put):
    Delta  = dV/dS
    Gamma  = d²V/dS²
    Vega   = dV/dsigma
    Theta  = dV/dt  (= -dV/dT, sign convention: positive when option loses value per day)

American put Greeks are also supported. Since there is no closed-form formula,
they are verified against central finite differences on the same functional pass.

The key idea: instead of baking S0/sigma/r/T into the network weights at construction
time, we write a *functional* CRR/BTNet forward pass that treats all market parameters
as differentiable PyTorch tensors. autograd then propagates gradients through the
full computation graph — terminal payoffs, convolutional discounting steps,
and the early-exercise max() operations.
"""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm


# ---------------------------------------------------------------------------
# 1.  Functional (differentiable) BTNet forward pass
# ---------------------------------------------------------------------------

def _crr_european_put(
    K: torch.Tensor,
    S0: torch.Tensor,
    sigma: torch.Tensor,
    r: torch.Tensor,
    T: torch.Tensor,
    n: int = 9,
) -> torch.Tensor:
    dt = T / n
    sqrt_dt = torch.sqrt(dt)

    u = torch.exp(sigma * sqrt_dt)
    d = torch.exp(-sigma * sqrt_dt)

    # Discounted risk-neutral weights (= filter of the conv layer)
    W0 = (torch.exp(-r * dt) * u - 1.0) / (u - d)   # exp(-r*dt) * pi_d
    W1 = (1.0 - torch.exp(-r * dt) * d) / (u - d)   # exp(-r*dt) * pi_u

    # Terminal stock prices S_j^n = S0 * exp(sigma*sqrt_dt*(2j - n))
    j = torch.arange(n + 1, dtype=torch.float32)
    S_T = S0 * torch.exp(sigma * sqrt_dt * (2.0 * j - n))

    # Terminal put payoffs V_j = max(K - S_j, 0)
    V = torch.relu(K - S_T)          # shape (n+1,)

    # Backward induction  (n applications of the 1-D convolution with [W0, W1])
    for _ in range(n):
        V = W0 * V[:-1] + W1 * V[1:]   # (L,) -> (L-1,)

    return V.squeeze()


def _crr_american_put(
    K: torch.Tensor,
    S0: torch.Tensor,
    sigma: torch.Tensor,
    r: torch.Tensor,
    T: torch.Tensor,
    n: int = 9,
) -> torch.Tensor:
    """CRR/BTNet American put price as a pure PyTorch expression.

    Implements backward induction with early-exercise via ``torch.max``,
    which is differentiable via subgradients — autograd routes the gradient
    through whichever branch (continuation vs intrinsic) achieves the max.

    Note on time indexing: loop index ``i`` (0-based) corresponds to time
    step ``n-1-i`` counting from 0 at the origin.  The stock price at node
    ``(time_idx, j)`` is  S0 * exp(sigma * sqrt_dt * (2j - time_idx)).
    This is the corrected formula vs the BTNetAmerican class initialisation.
    """
    dt = T / n
    sqrt_dt = torch.sqrt(dt)

    u = torch.exp(sigma * sqrt_dt)
    d = torch.exp(-sigma * sqrt_dt)

    W0 = (torch.exp(-r * dt) * u - 1.0) / (u - d)
    W1 = (1.0 - torch.exp(-r * dt) * d) / (u - d)

    # Terminal payoffs
    j = torch.arange(n + 1, dtype=torch.float32)
    S_T = S0 * torch.exp(sigma * sqrt_dt * (2.0 * j - n))
    V = torch.relu(K - S_T)                           # shape (n+1,)

    # Backward induction with early exercise
    for i in range(n):
        # Correct time index for this backward step (fixes the bias-index bug
        # present in BTNetAmerican's __init__ — see greeks.py docstring)
        time_idx = n - 1 - i

        # Discounted continuation value
        V_cont = W0 * V[:-1] + W1 * V[1:]             # shape (n-i,)

        # Intrinsic value K - S(time_idx, j)  for j = 0 … n-1-i
        j_curr = torch.arange(n - i, dtype=torch.float32)
        S_curr = S0 * torch.exp(sigma * sqrt_dt * (2.0 * j_curr - time_idx))
        V_intr = torch.relu(K - S_curr)

        # Early exercise decision
        V = torch.max(V_cont, V_intr)

    return V.squeeze()


# ---------------------------------------------------------------------------
# 2.  Greeks via autograd
# ---------------------------------------------------------------------------

def _scalar(x: float) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)


def btnet_greeks(
    K_values: np.ndarray,
    S0: float,
    sigma: float,
    r: float,
    T: float,
    n: int = 9,
) -> dict[str, np.ndarray]:
    """Compute Delta, Gamma, Vega, Theta for BTNet European put via autograd"""
    K_flat = K_values.flatten()
    deltas, gammas, vegas, thetas = [], [], [], []

    for K_val in K_flat:
        K = _scalar(K_val)

        # ---- Delta & Gamma  (derivatives w.r.t. S0) ----
        S0_t = torch.tensor(S0, dtype=torch.float32, requires_grad=True)
        V = _crr_european_put(K, S0_t, _scalar(sigma), _scalar(r), _scalar(T), n)

        # Delta = dV/dS  (1st derivative, keep graph for 2nd)
        (delta,) = torch.autograd.grad(V, S0_t, create_graph=True)
        deltas.append(delta.item())

        # Gamma = d²V/dS²
        (gamma,) = torch.autograd.grad(delta, S0_t)
        gammas.append(gamma.item())

        # ---- Vega  (dV/dsigma) ----
        sigma_t = torch.tensor(sigma, dtype=torch.float32, requires_grad=True)
        V2 = _crr_european_put(K, _scalar(S0), sigma_t, _scalar(r), _scalar(T), n)
        (vega,) = torch.autograd.grad(V2, sigma_t)
        vegas.append(vega.item())

        # ---- Theta  (dV/dt = -dV/dT) ----
        T_t = torch.tensor(T, dtype=torch.float32, requires_grad=True)
        V3 = _crr_european_put(K, _scalar(S0), _scalar(sigma), _scalar(r), T_t, n)
        (dV_dT,) = torch.autograd.grad(V3, T_t)
        # theta = dV/dt, and t = T_maturity - tau, so dV/dt = -dV/dT
        thetas.append(-dV_dT.item())

    return {
        "delta": np.array(deltas),
        "gamma": np.array(gammas),
        "vega":  np.array(vegas),
        "theta": np.array(thetas),
    }


def btnet_american_greeks(
    K_values: np.ndarray,
    S0: float,
    sigma: float,
    r: float,
    T: float,
    n: int = 9,
) -> dict[str, np.ndarray]:
    """Compute Delta, Gamma, Vega, Theta for BTNet American put via autograd.

    Uses the differentiable ``_crr_american_put`` pass (with the corrected
    time-index formula).  There is no closed-form reference for American Greeks;
    use ``american_greeks_fd`` to verify against central finite differences.
    """
    K_flat = K_values.flatten()
    deltas, gammas, vegas, thetas = [], [], [], []

    for K_val in K_flat:
        K = _scalar(K_val)

        # ---- Delta & Gamma  (w.r.t. S0) ----
        S0_t = torch.tensor(S0, dtype=torch.float32, requires_grad=True)
        V = _crr_american_put(K, S0_t, _scalar(sigma), _scalar(r), _scalar(T), n)

        (delta,) = torch.autograd.grad(V, S0_t, create_graph=True)
        deltas.append(delta.item())

        (gamma,) = torch.autograd.grad(delta, S0_t)
        gammas.append(gamma.item())

        # ---- Vega  (w.r.t. sigma) ----
        sigma_t = torch.tensor(sigma, dtype=torch.float32, requires_grad=True)
        V2 = _crr_american_put(K, _scalar(S0), sigma_t, _scalar(r), _scalar(T), n)
        (vega,) = torch.autograd.grad(V2, sigma_t)
        vegas.append(vega.item())

        # ---- Theta  (dV/dt = -dV/dT) ----
        T_t = torch.tensor(T, dtype=torch.float32, requires_grad=True)
        V3 = _crr_american_put(K, _scalar(S0), _scalar(sigma), _scalar(r), T_t, n)
        (dV_dT,) = torch.autograd.grad(V3, T_t)
        thetas.append(-dV_dT.item())

    return {
        "delta": np.array(deltas),
        "gamma": np.array(gammas),
        "vega":  np.array(vegas),
        "theta": np.array(thetas),
    }


def american_greeks_fd(
    K_values: np.ndarray,
    S0: float,
    sigma: float,
    r: float,
    T: float,
    n: int = 9,
    eps_S: float = 1e-3,
    eps_sigma: float = 1e-4,
    eps_T: float = 1e-4,
) -> dict[str, np.ndarray]:
    """Central finite differences for American put Greeks (used as reference).

    All bumps are applied to the same ``_crr_american_put`` functional pass,
    so this is a pure numerical verification of the autograd results.

    Args:
        eps_S:     bump size for Delta/Gamma (fraction of S0).
        eps_sigma: bump size for Vega.
        eps_T:     bump size for Theta.
    """
    def price(K_val, s, vol, rate, mat):
        with torch.no_grad():
            return _crr_american_put(
                _scalar(K_val), _scalar(s), _scalar(vol), _scalar(rate), _scalar(mat), n
            ).item()

    K_flat = K_values.flatten()
    h_S = eps_S * S0

    deltas = [(price(K, S0 + h_S, sigma, r, T) - price(K, S0 - h_S, sigma, r, T)) / (2 * h_S)
              for K in K_flat]
    gammas = [(price(K, S0 + h_S, sigma, r, T) - 2 * price(K, S0, sigma, r, T) + price(K, S0 - h_S, sigma, r, T)) / h_S**2
              for K in K_flat]
    vegas  = [(price(K, S0, sigma + eps_sigma, r, T) - price(K, S0, sigma - eps_sigma, r, T)) / (2 * eps_sigma)
              for K in K_flat]
    # Theta = -dV/dT  (longer maturity → higher value → dV/dT > 0 → theta < 0)
    thetas = [-(price(K, S0, sigma, r, T + eps_T) - price(K, S0, sigma, r, T - eps_T)) / (2 * eps_T)
              for K in K_flat]

    return {
        "delta": np.array(deltas),
        "gamma": np.array(gammas),
        "vega":  np.array(vegas),
        "theta": np.array(thetas),
    }


# ---------------------------------------------------------------------------
# 3.  Black-Scholes analytical Greeks (reference, European only)
# ---------------------------------------------------------------------------

def bs_greeks(
    K_values: np.ndarray,
    S0: float,
    sigma: float,
    r: float,
    T: float,
) -> dict[str, np.ndarray]:
    """
    Formulas:
        d1 = [ln(S/K) + (r + sigma²/2)*T] / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)

        Delta = N(d1) - 1                              in (-1, 0)
        Gamma = N'(d1) / (S * sigma * sqrt(T))         > 0
        Vega  = S * sqrt(T) * N'(d1)                   > 0
        Theta = -[S*N'(d1)*sigma/(2*sqrt(T)) - r*K*exp(-rT)*N(-d2)]
    """
    K = K_values.flatten()
    sqT = np.sqrt(T)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT

    Nprime_d1 = norm.pdf(d1)

    delta = norm.cdf(d1) - 1.0
    gamma = Nprime_d1 / (S0 * sigma * sqT)
    vega  = S0 * sqT * Nprime_d1
    theta = -(S0 * Nprime_d1 * sigma / (2.0 * sqT) - r * K * np.exp(-r * T) * norm.cdf(-d2))

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


# ---------------------------------------------------------------------------
# 4.  Error table
# ---------------------------------------------------------------------------

def greeks_error_table(
    btnet: dict[str, np.ndarray],
    bs: dict[str, np.ndarray],
) -> None:
    """Print MAE and max absolute error for each Greek."""
    print(f"{'Greek':<8} {'MAE':>12} {'Max |err|':>12}")
    print("-" * 34)
    for key in ("delta", "gamma", "vega", "theta"):
        err = np.abs(btnet[key] - bs[key])
        print(f"{key.capitalize():<8} {err.mean():>12.6f} {err.max():>12.6f}")


# ---------------------------------------------------------------------------
# 5.  Plotting
# ---------------------------------------------------------------------------

def plot_greeks(
    K_values: np.ndarray,
    btnet: dict[str, np.ndarray],
    bs: dict[str, np.ndarray],
    S0: float,
) -> None:
    K = K_values.flatten()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"European Put Greeks  (BTNet autograd  vs  Black-Scholes)\n"
        f"S₀ = {S0},  σ = const,  verified per strike K ∈ [{K.min():.2f}, {K.max():.2f}]",
        fontsize=13,
    )

    specs = [
        ("delta", "Delta  (∂V/∂S)",     "upper right"),
        ("gamma", "Gamma  (∂²V/∂S²)",   "upper right"),
        ("vega",  "Vega  (∂V/∂σ)",      "upper left"),
        ("theta", "Theta  (∂V/∂t)",     "lower right"),
    ]

    for ax, (key, title, loc) in zip(axes.flat, specs):
        ax.plot(K, bs[key],     "b-",  linewidth=2.0, label="Black-Scholes (analytical)")
        ax.plot(K, btnet[key],  "r--", linewidth=1.8, label="BTNet (autograd)")
        ax.axvline(S0, color="grey", linestyle=":", linewidth=1.0, label=f"ATM  K = S₀ = {S0}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Strike K")
        ax.legend(loc=loc, fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_greeks_errors(
    K_values: np.ndarray,
    btnet: dict[str, np.ndarray],
    bs: dict[str, np.ndarray],
) -> None:
    K = K_values.flatten()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Greeks Errors  |BTNet autograd − Black-Scholes analytical|", fontsize=13)

    colors = {"delta": "steelblue", "gamma": "darkorange", "vega": "seagreen", "theta": "crimson"}
    titles = {
        "delta": "Delta error",
        "gamma": "Gamma error",
        "vega":  "Vega error",
        "theta": "Theta error",
    }

    for ax, key in zip(axes.flat, ("delta", "gamma", "vega", "theta")):
        err = np.abs(btnet[key] - bs[key])
        ax.plot(K, err, color=colors[key], linewidth=2.0)
        ax.fill_between(K, 0, err, alpha=0.2, color=colors[key])
        ax.set_title(f"{titles[key]}   (MAE = {err.mean():.2e})", fontsize=11)
        ax.set_xlabel("Strike K")
        ax.set_ylabel("|BTNet − BS|")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_american_greeks(
    K_values: np.ndarray,
    autograd: dict[str, np.ndarray],
    fd: dict[str, np.ndarray],
    european_bs: dict[str, np.ndarray],
    S0: float,
) -> None:
    """3-curve comparison for American Greeks:
        - BTNet autograd (American)
        - Finite differences (American, same CRR pass — numerical verification)
        - Black-Scholes (European, for early-exercise premium visibility)
    """
    K = K_values.flatten()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"American Put Greeks  (BTNet autograd  vs  FD  vs  BS European)\n"
        f"S₀ = {S0},  ATM line shown",
        fontsize=13,
    )

    specs = [
        ("delta", "Delta  (∂V/∂S)",   "upper right"),
        ("gamma", "Gamma  (∂²V/∂S²)", "upper right"),
        ("vega",  "Vega  (∂V/∂σ)",    "upper left"),
        ("theta", "Theta  (∂V/∂t)",   "lower right"),
    ]

    for ax, (key, title, loc) in zip(axes.flat, specs):
        ax.plot(K, european_bs[key], "b:",  linewidth=1.5, label="BS European (reference)")
        ax.plot(K, fd[key],          "g-",  linewidth=2.0, label="FD American (CRR, bump-reprice)")
        ax.plot(K, autograd[key],    "r--", linewidth=1.8, label="BTNet autograd (American)")
        ax.axvline(S0, color="grey", linestyle=":", linewidth=1.0, label=f"ATM K=S₀={S0}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Strike K")
        ax.legend(loc=loc, fontsize=7.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_american_greeks_errors(
    K_values: np.ndarray,
    autograd: dict[str, np.ndarray],
    fd: dict[str, np.ndarray],
) -> None:
    """Absolute error |autograd − FD| per Greek for American put."""
    K = K_values.flatten()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("|BTNet autograd − Finite Differences|  (American put, CRR n=9)", fontsize=13)

    colors = {"delta": "steelblue", "gamma": "darkorange", "vega": "seagreen", "theta": "crimson"}

    for ax, key in zip(axes.flat, ("delta", "gamma", "vega", "theta")):
        err = np.abs(autograd[key] - fd[key])
        ax.plot(K, err, color=colors[key], linewidth=2.0)
        ax.fill_between(K, 0, err, alpha=0.2, color=colors[key])
        ax.set_title(f"{key.capitalize()}   MAE = {err.mean():.2e}", fontsize=11)
        ax.set_xlabel("Strike K")
        ax.set_ylabel("|autograd − FD|")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 6.  Weight-transfer experiment Greeks
# ---------------------------------------------------------------------------

def _crr_american_put_fixed_W(
    K: torch.Tensor,
    S0: torch.Tensor,
    sigma: torch.Tensor,
    T: torch.Tensor,
    W0_fixed: torch.Tensor,
    W1_fixed: torch.Tensor,
    n: int = 9,
) -> torch.Tensor:
    """CRR American put with externally specified (frozen) W0, W1.

    Unlike ``_crr_american_put``, the discounting weights are *not* derived
    from sigma/r/T — they are passed in directly.  This lets autograd
    differentiate w.r.t. S0, sigma, T while keeping W frozen, matching the
    behaviour of a transferred-weight model whose conv filter was fixed during
    the European-training step.

    Gradients flow only through the stock-price geometry
    (terminal payoffs and intrinsic values), not through W.
    """
    dt = T / n
    sqrt_dt = torch.sqrt(dt)

    # Terminal payoffs
    j = torch.arange(n + 1, dtype=torch.float32)
    S_T = S0 * torch.exp(sigma * sqrt_dt * (2.0 * j - n))
    V = torch.relu(K - S_T)

    # Backward induction — W is constant (no gradient flows through it)
    for i in range(n):
        time_idx = n - 1 - i
        V_cont = W0_fixed * V[:-1] + W1_fixed * V[1:]
        j_curr = torch.arange(n - i, dtype=torch.float32)
        S_curr = S0 * torch.exp(sigma * sqrt_dt * (2.0 * j_curr - time_idx))
        V_intr = torch.relu(K - S_curr)
        V = torch.max(V_cont, V_intr)

    return V.squeeze()


def btnet_american_greeks_fixed_W(
    K_values: np.ndarray,
    S0: float,
    sigma: float,
    T: float,
    W0_val: float,
    W1_val: float,
    n: int = 9,
) -> dict[str, np.ndarray]:
    """Greeks for an American put model whose conv filter W is externally fixed.

    Used in the weight-transfer experiment (v4 notebook): W comes from a
    trained European model rather than from the analytical CRR formula.
    Because W is frozen, Vega and Theta only reflect the sensitivity of the
    stock-price tree — they do *not* include the indirect channel through the
    risk-neutral probabilities (which is present in the fully-analytical model).

    Args:
        W0_val, W1_val: The two elements of the conv filter [W0, W1].
                        Extract from the model with:
                        ``model._maxout_layers[0]._conv_1d.weight.data.squeeze().tolist()``
    """
    W0_t = torch.tensor(W0_val, dtype=torch.float32)   # no requires_grad -> frozen
    W1_t = torch.tensor(W1_val, dtype=torch.float32)

    K_flat = K_values.flatten()
    deltas, gammas, vegas, thetas = [], [], [], []

    for K_val in K_flat:
        K = _scalar(K_val)

        # ---- Delta & Gamma (w.r.t. S0) ----
        S0_t = torch.tensor(S0, dtype=torch.float32, requires_grad=True)
        V = _crr_american_put_fixed_W(K, S0_t, _scalar(sigma), _scalar(T), W0_t, W1_t, n)
        (delta,) = torch.autograd.grad(V, S0_t, create_graph=True)
        deltas.append(delta.item())
        (gamma,) = torch.autograd.grad(delta, S0_t)
        gammas.append(gamma.item())

        # ---- Vega (w.r.t. sigma) ----
        sigma_t = torch.tensor(sigma, dtype=torch.float32, requires_grad=True)
        V2 = _crr_american_put_fixed_W(K, _scalar(S0), sigma_t, _scalar(T), W0_t, W1_t, n)
        (vega,) = torch.autograd.grad(V2, sigma_t)
        vegas.append(vega.item())

        # ---- Theta (dV/dt = -dV/dT) ----
        T_t = torch.tensor(T, dtype=torch.float32, requires_grad=True)
        V3 = _crr_american_put_fixed_W(K, _scalar(S0), _scalar(sigma), T_t, W0_t, W1_t, n)
        (dV_dT,) = torch.autograd.grad(V3, T_t)
        thetas.append(-dV_dT.item())

    return {
        "delta": np.array(deltas),
        "gamma": np.array(gammas),
        "vega":  np.array(vegas),
        "theta": np.array(thetas),
    }


def plot_american_greeks_transfer(
    K_values: np.ndarray,
    greeks_analytical: dict[str, np.ndarray],
    greeks_transfer: dict[str, np.ndarray],
    european_bs: dict[str, np.ndarray],
    S0: float,
    W_analytical: list,
    W_transfer: list,
) -> None:
    """4-panel plot comparing Greeks of analytical vs transferred-W American model.

    Three curves per panel:
        - Black-Scholes European (dotted, reference)
        - Analytical W American (solid blue)
        - Transferred W American (dashed orange)
    """
    K = K_values.flatten()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"American Put Greeks: Analytical W vs Transferred W\n"
        f"Analytical W = {[round(w,4) for w in W_analytical]}  |  "
        f"Transferred W = {[round(w,4) for w in W_transfer]}",
        fontsize=12,
    )

    specs = [
        ("delta", "Delta  (∂V/∂S)",   "upper right"),
        ("gamma", "Gamma  (∂²V/∂S²)", "upper right"),
        ("vega",  "Vega  (∂V/∂σ)",    "upper left"),
        ("theta", "Theta  (∂V/∂t)",   "lower right"),
    ]

    for ax, (key, title, loc) in zip(axes.flat, specs):
        ax.plot(K, european_bs[key],       "b:",  linewidth=1.5, label="BS European (reference)")
        ax.plot(K, greeks_analytical[key], "b-",  linewidth=2.0, label="Analytical W (Theorem 2)")
        ax.plot(K, greeks_transfer[key],   "r--", linewidth=1.8, label="Transferred W (Euro-trained)")
        ax.axvline(S0, color="grey", linestyle=":", linewidth=1.0, label=f"ATM K=S\u2080={S0}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Strike K")
        ax.legend(loc=loc, fontsize=7.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_american_greeks_transfer_errors(
    K_values: np.ndarray,
    greeks_analytical: dict[str, np.ndarray],
    greeks_transfer: dict[str, np.ndarray],
) -> None:
    """Absolute difference |analytical W - transferred W| per Greek."""
    K = K_values.flatten()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "|Greeks(analytical W) - Greeks(transferred W)|  (American put)",
        fontsize=13,
    )

    colors = {"delta": "steelblue", "gamma": "darkorange", "vega": "seagreen", "theta": "crimson"}

    for ax, key in zip(axes.flat, ("delta", "gamma", "vega", "theta")):
        err = np.abs(greeks_analytical[key] - greeks_transfer[key])
        ax.plot(K, err, color=colors[key], linewidth=2.0)
        ax.fill_between(K, 0, err, alpha=0.2, color=colors[key])
        ax.set_title(f"{key.capitalize()}   MAE = {err.mean():.2e}", fontsize=11)
        ax.set_xlabel("Strike K")
        ax.set_ylabel("|analytical - transferred|")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
