"""QuantLib reference pricers and benchmark metrics (optional dependency).

Install: ``pip install QuantLib-Python`` or ``pip install 'btnn-bs[quantlib]'`` (see pyproject optional deps).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

try:
    import QuantLib as ql
except ImportError as exc:  # pragma: no cover - optional in dev envs without wheels
    ql = None  # type: ignore[assignment]
    _QL_IMPORT_ERROR = exc
else:
    _QL_IMPORT_ERROR = None


def require_quantlib() -> None:
    if ql is None:
        raise ImportError(
            "QuantLib is not installed. Use: pip install QuantLib-Python "
            "or conda install -c conda-forge quantlib-python"
        ) from _QL_IMPORT_ERROR


def _as_ql_date(ref_date: Any | None) -> Any:
    require_quantlib()
    if ref_date is None:
        return ql.Date(1, ql.January, 2025)
    if isinstance(ref_date, ql.Date):
        return ref_date
    raise TypeError("ref_date must be None or a QuantLib.Date")


def maturity_date(ref_date: Any, T: float) -> Any:
    """Expiry as ``ref_date + round(T * 365.25)`` calendar days (NullCalendar)."""
    require_quantlib()
    days = int(max(1, round(float(T) * 365.25)))
    return ref_date + ql.Period(days, ql.Days)


def build_bsm_process(
    S0: float,
    r: float,
    sigma: float,
    ref_date: Any,
    q: float = 0.0,
):
    require_quantlib()
    calendar = ql.NullCalendar()
    dc = ql.Actual365Fixed()
    spot = ql.QuoteHandle(ql.SimpleQuote(float(S0)))
    rf = ql.YieldTermStructureHandle(ql.FlatForward(ref_date, float(r), dc))
    div = ql.YieldTermStructureHandle(ql.FlatForward(ref_date, float(q), dc))
    vol = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(ref_date, calendar, float(sigma), dc)
    )
    return ql.BlackScholesMertonProcess(spot, div, rf, vol)


def price_european_put_ql(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    *,
    q: float = 0.0,
    ref_date: Any | None = None,
) -> float:
    """European put, analytic Black–Scholes engine."""
    require_quantlib()
    ref_date = _as_ql_date(ref_date)
    ql.Settings.instance().evaluationDate = ref_date
    maturity = maturity_date(ref_date, T)
    process = build_bsm_process(S0, r, sigma, ref_date, q)
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, float(K))
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(exercise, payoff)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return float(option.NPV())


def price_american_put_binomial_ql(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    *,
    steps: int = 500,
    q: float = 0.0,
    ref_date: Any | None = None,
) -> float:
    """American put, CRR binomial tree (``BinomialVanillaEngine``)."""
    require_quantlib()
    ref_date = _as_ql_date(ref_date)
    ql.Settings.instance().evaluationDate = ref_date
    maturity = maturity_date(ref_date, T)
    process = build_bsm_process(S0, r, sigma, ref_date, q)
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, float(K))
    exercise = ql.AmericanExercise(ref_date, maturity)
    option = ql.VanillaOption(exercise, payoff)
    option.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", int(steps)))
    return float(option.NPV())


def price_american_put_baw_ql(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    *,
    q: float = 0.0,
    ref_date: Any | None = None,
) -> float:
    """American put, Barone–Adesi & Whaley approximation (if available in your QuantLib build)."""
    require_quantlib()
    ref_date = _as_ql_date(ref_date)
    ql.Settings.instance().evaluationDate = ref_date
    maturity = maturity_date(ref_date, T)
    process = build_bsm_process(S0, r, sigma, ref_date, q)
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, float(K))
    exercise = ql.AmericanExercise(ref_date, maturity)
    option = ql.VanillaOption(exercise, payoff)
    if not hasattr(ql, "BaroneAdesiWhaleApproximationEngine"):
        raise AttributeError(
            "BaroneAdesiWhaleApproximationEngine not found in this QuantLib build"
        )
    option.setPricingEngine(ql.BaroneAdesiWhaleApproximationEngine(process))
    return float(option.NPV())


def european_put_grid(
    S0: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    sigma: float,
    *,
    q: float = 0.0,
    ref_date: Any | None = None,
) -> np.ndarray:
    return np.array(
        [
            price_european_put_ql(S0, float(k), T, r, sigma, q=q, ref_date=ref_date)
            for k in np.asarray(strikes).flatten()
        ],
        dtype=np.float64,
    )


def american_put_grid_crr(
    S0: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    sigma: float,
    *,
    steps: int = 500,
    q: float = 0.0,
    ref_date: Any | None = None,
) -> np.ndarray:
    return np.array(
        [
            price_american_put_binomial_ql(
                S0, float(k), T, r, sigma, steps=steps, q=q, ref_date=ref_date
            )
            for k in np.asarray(strikes).flatten()
        ],
        dtype=np.float64,
    )


def american_put_grid_baw(
    S0: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    sigma: float,
    *,
    q: float = 0.0,
    ref_date: Any | None = None,
) -> np.ndarray:
    return np.array(
        [
            price_american_put_baw_ql(S0, float(k), T, r, sigma, q=q, ref_date=ref_date)
            for k in np.asarray(strikes).flatten()
        ],
        dtype=np.float64,
    )


def error_stats(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    p = np.asarray(pred, dtype=np.float64).flatten()
    r = np.asarray(ref, dtype=np.float64).flatten()
    d = p - r
    return {
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "max_abs": float(np.max(np.abs(d))),
    }


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, float(time.perf_counter() - t0)


@dataclass
class QuantLibBenchmarkResult:
    """All reference curves on ``K_test`` plus timing of one full grid pass."""

    K: np.ndarray
    ql_european_analytic: np.ndarray
    ql_american_crr: np.ndarray
    ql_american_baw: np.ndarray | None
    ref_date: Any
    seconds_european_grid: float
    seconds_american_crr_grid: float
    seconds_american_baw_grid: float | None
    amer_crr_steps: int


def run_quantlib_benchmark(
    S0: float,
    K_test: np.ndarray,
    T: float,
    r: float,
    sigma: float,
    *,
    amer_crr_steps: int = 500,
    q: float = 0.0,
    ref_date: Any | None = None,
    include_baw: bool = True,
) -> QuantLibBenchmarkResult:
    """Price European (analytic) and American (CRR + optional BAW) on the same strike grid."""
    require_quantlib()
    ref_date = _as_ql_date(ref_date)
    K = np.asarray(K_test, dtype=np.float64).flatten()

    euro, t_euro = _timed(
        lambda: european_put_grid(S0, K, T, r, sigma, q=q, ref_date=ref_date)
    )

    amer_crr, t_amer = _timed(
        lambda: american_put_grid_crr(
            S0, K, T, r, sigma, steps=amer_crr_steps, q=q, ref_date=ref_date
        )
    )

    baw: np.ndarray | None = None
    t_baw: float | None = None
    if include_baw:
        try:
            baw, t_baw = _timed(
                lambda: american_put_grid_baw(S0, K, T, r, sigma, q=q, ref_date=ref_date)
            )
        except (AttributeError, RuntimeError):
            baw = None
            t_baw = None

    return QuantLibBenchmarkResult(
        K=K,
        ql_european_analytic=euro,
        ql_american_crr=amer_crr,
        ql_american_baw=baw,
        ref_date=ref_date,
        seconds_european_grid=t_euro,
        seconds_american_crr_grid=t_amer,
        seconds_american_baw_grid=t_baw,
        amer_crr_steps=amer_crr_steps,
    )


def print_comparison_table(
    *,
    european: dict[str, dict[str, float]] | None = None,
    american: dict[str, dict[str, float]] | None = None,
) -> None:
    """Pretty-print MAE / RMSE / max|err| blocks (keys = row label, values = error_stats dict)."""
    lines = []
    if european:
        lines.append("European put — errors vs QuantLib (analytic)")
        for name, stats in european.items():
            lines.append(
                f"  {name:22s}  MAE={stats['mae']:.2e}  RMSE={stats['rmse']:.2e}  max|.|={stats['max_abs']:.2e}"
            )
    if american:
        lines.append("American put — errors vs QuantLib (CRR tree)")
        for name, stats in american.items():
            lines.append(
                f"  {name:22s}  MAE={stats['mae']:.2e}  RMSE={stats['rmse']:.2e}  max|.|={stats['max_abs']:.2e}"
            )
    print("\n".join(lines))
