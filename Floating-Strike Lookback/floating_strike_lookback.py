"""
European Floating-Strike (Standard) Lookback Options — Black–Scholes closed-form.

From lookback.txt:
- Call: strike floats to the minimum; payoff S_T - min(S_t) = S_T - Y_T. L = historically realized minimum.
- Put:  strike floats to the maximum; payoff max(S_t) - S_T = Z_T - S_T. L = historically realized maximum.
- Notation: S_0 spot, T maturity, r rate, δ (delta) dividend yield, σ vol, L current extremum (from inception to today).
- Auxiliary d_1, d_2, d_3 as in lookback.txt (with δ). Supports δ ≠ 0 and optional L.
"""

from math import log, sqrt, exp, erf
from typing import Callable, Optional
from dataclasses import dataclass

_BGK_BETA1 = 0.5826


@dataclass
class OptionGreeks:
    delta: float
    gamma: float
    vega: float


def norm_cdf(x: float) -> float:
    """Cumulative distribution function of the standard normal N(d)."""
    try:
        from scipy.stats import norm
        return float(norm.cdf(x))
    except ImportError:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Probability density function of the standard normal N(d)."""
    return exp(-x * x / 2.0) / sqrt(2.0 * 3.141592653589793)


def _validate_monitoring_points(monitoring_points: int) -> None:
    if monitoring_points <= 0:
        raise ValueError("monitoring_points must be a positive integer")


def _finite_difference_greeks(
    pricer: Callable[[float, float], float],
    S0: float,
    sigma: float,
    dS: Optional[float] = None,
    dSigma: Optional[float] = None,
) -> OptionGreeks:
    if S0 <= 0:
        raise ValueError("S0 must be positive for finite-difference Greeks")
    if sigma <= 0:
        raise ValueError("sigma must be positive for finite-difference Greeks")
    bump_s = max(1e-4 * S0, 1e-6) if dS is None else dS
    if bump_s <= 0:
        raise ValueError("dS must be positive")
    bump_sigma = max(1e-4 * sigma, 1e-6) if dSigma is None else dSigma
    if bump_sigma <= 0:
        raise ValueError("dSigma must be positive")
    if bump_sigma >= sigma:
        bump_sigma = 0.5 * sigma
    if bump_sigma <= 0:
        raise ValueError("dSigma is too large relative to sigma")

    v0 = pricer(S0, sigma)
    v_up = pricer(S0 + bump_s, sigma)
    v_dn = pricer(S0 - bump_s, sigma)
    delta = (v_up - v_dn) / (2.0 * bump_s)
    gamma = (v_up - 2.0 * v0 + v_dn) / (bump_s * bump_s)

    vega_up = pricer(S0, sigma + bump_sigma)
    vega_dn = pricer(S0, sigma - bump_sigma)
    vega = (vega_up - vega_dn) / (2.0 * bump_sigma)
    return OptionGreeks(delta=delta, gamma=gamma, vega=vega)


# -----------------------------------------------------------------------------
# Auxiliary distance functions (from lookback.txt)
# d_{1,K} = (ln(S_0/K) + (r - δ + σ²/2)T) / (σ√T), d_2 = d_1 - σ√T, d_3 = d_1 - 2(r-δ)√T/σ
# -----------------------------------------------------------------------------

def d1(S: float, K: float, T: float, r: float, delta: float, sigma: float) -> float:
    return (log(S / K) + (r - delta + 0.5 * sigma**2) * T) / (sigma * sqrt(T))


def d2(S: float, K: float, T: float, r: float, delta: float, sigma: float) -> float:
    return d1(S, K, T, r, delta, sigma) - sigma * sqrt(T)


def d3(S: float, K: float, T: float, r: float, delta: float, sigma: float) -> float:
    return d1(S, K, T, r, delta, sigma) - 2 * (r - delta) * sqrt(T) / sigma


# -----------------------------------------------------------------------------
# European floating-strike lookback call (strike = minimum)
# Payoff: S_T - min(S_t) = S_T - Y_T. L = current realized minimum (from inception to today).
# At inception L = None -> use L = S_0. Supports δ ≠ 0.
# -----------------------------------------------------------------------------

def floating_strike_lookback_call(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    delta: float = 0.0,
    L: Optional[float] = None,
) -> float:
    """
    European floating-strike lookback call (buy at historical minimum).
    L: current realized minimum from inception to today; if None, pricing at inception (L = S_0).
    delta: continuous dividend yield δ (optional, default 0).
    """
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive")
    if L is not None and L > S0:
        raise ValueError("For floating-strike call, running minimum L must satisfy L <= S0")
    if T <= 0:
        return max(S0 - (L if L is not None else S0), 0.0)
    L_use = S0 if L is None else L
    b = r - delta
    expo = 2.0 * b / (sigma**2)
    log_ratio = log(L_use / S0)
    sqrt_T = sqrt(T)
    _d1 = (log(S0 / L_use) + (b + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    _d2 = _d1 - sigma * sqrt_T
    _d3 = _d1 - (2.0 * b / sigma) * sqrt_T

    disc_div = exp(-delta * T)
    disc_rf = exp(-r * T)

    if abs(b) >= 1e-12:
        call = (
            S0 * disc_div * norm_cdf(_d1)
            - L_use * disc_rf * norm_cdf(_d2)
            + (sigma**2 / (2.0 * b))
            * S0
            * (
                disc_div * norm_cdf(-_d1)
                - disc_rf * exp(expo * log_ratio) * norm_cdf(-_d3)
            )
        )
    else:
        call = (
            S0 * disc_div * norm_cdf(_d1)
            - L_use * disc_rf * norm_cdf(_d2)
            + S0
            * disc_div
            * (
                -sigma * sqrt_T * norm_pdf(_d1)
                + (0.5 * sigma**2 * T + log(S0 / L_use)) * norm_cdf(-_d1)
            )
        )
    return call


# -----------------------------------------------------------------------------
# European floating-strike lookback put (strike = maximum)
# Payoff: max(S_t) - S_T = Z_T - S_T. L = current realized maximum (from inception to today).
# At inception L = None -> use L = S_0. Supports δ ≠ 0.
# -----------------------------------------------------------------------------

def floating_strike_lookback_put(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    delta: float = 0.0,
    L: Optional[float] = None,
) -> float:
    """
    European floating-strike lookback put (sell at historical maximum).
    L: current realized maximum from inception to today; if None, pricing at inception (L = S_0).
    delta: continuous dividend yield δ (optional, default 0).
    """
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive")
    if L is not None and L < S0:
        raise ValueError("For floating-strike put, running maximum L must satisfy L >= S0")
    if T <= 0:
        return max((L if L is not None else S0) - S0, 0.0)
    L_use = S0 if L is None else L
    b = r - delta
    expo = 2.0 * b / (sigma**2)
    log_ratio = log(L_use / S0)
    sqrt_T = sqrt(T)
    _d1 = (log(S0 / L_use) + (b + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    _d2 = _d1 - sigma * sqrt_T
    _d3 = _d1 - (2.0 * b / sigma) * sqrt_T

    disc_div = exp(-delta * T)
    disc_rf = exp(-r * T)

    if abs(b) >= 1e-12:
        put = (
            -S0 * disc_div * norm_cdf(-_d1)
            + L_use * disc_rf * norm_cdf(-_d2)
            + (sigma**2 / (2.0 * b))
            * S0
            * (
                disc_div * norm_cdf(_d1)
                - disc_rf * exp(expo * log_ratio) * norm_cdf(_d3)
            )
        )
    else:
        put = (
            -S0 * disc_div * norm_cdf(-_d1)
            + L_use * disc_rf * norm_cdf(-_d2)
            + S0
            * disc_div
            * (
                sigma * sqrt_T * norm_pdf(_d1)
                + (0.5 * sigma**2 * T + log(S0 / L_use)) * norm_cdf(_d1)
            )
        )
    return put


# -----------------------------------------------------------------------------
# BGK continuity correction (discrete-monitoring proxy for continuous formulas)
# -----------------------------------------------------------------------------

def floating_strike_lookback_call_bgk(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    monitoring_points: int,
    delta: float = 0.0,
    L: Optional[float] = None,
) -> float:
    """
    BGK-corrected floating-strike call approximation (Broadie-Glasserman-Kou, 1999).
    Uses Theorem 2 at inception (L is None), and Theorem 3 for in-progress options.
    """
    _validate_monitoring_points(monitoring_points)
    if T <= 0:
        return floating_strike_lookback_call(S0, T, r, sigma, delta=delta, L=L)
    corr = _BGK_BETA1 * sigma * sqrt(T / monitoring_points)
    eta = exp(corr)
    if L is None:
        v_cont = floating_strike_lookback_call(S0, T, r, sigma, delta=delta, L=None)
        return (v_cont - S0) * (1.0 + corr) + S0
    l_adjusted = L / eta
    v_cont = floating_strike_lookback_call(S0, T, r, sigma, delta=delta, L=l_adjusted)
    return eta * v_cont + (1.0 - eta) * S0


def floating_strike_lookback_put_bgk(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    monitoring_points: int,
    delta: float = 0.0,
    L: Optional[float] = None,
) -> float:
    """
    BGK-corrected floating-strike put approximation (Broadie-Glasserman-Kou, 1999).
    Uses Theorem 2 at inception (L is None), and Theorem 3 for in-progress options.
    """
    _validate_monitoring_points(monitoring_points)
    if T <= 0:
        return floating_strike_lookback_put(S0, T, r, sigma, delta=delta, L=L)
    corr = _BGK_BETA1 * sigma * sqrt(T / monitoring_points)
    eta = exp(corr)
    if L is None:
        v_cont = floating_strike_lookback_put(S0, T, r, sigma, delta=delta, L=None)
        return (v_cont + S0) * (1.0 - corr) - S0
    l_adjusted = L * eta
    v_cont = floating_strike_lookback_put(S0, T, r, sigma, delta=delta, L=l_adjusted)
    eta_inv = 1.0 / eta
    return eta_inv * v_cont + (eta_inv - 1.0) * S0


def floating_strike_lookback_call_greeks(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    delta: float = 0.0,
    L: Optional[float] = None,
    dS: Optional[float] = None,
    dSigma: Optional[float] = None,
) -> OptionGreeks:
    """Finite-difference Delta/Gamma/Vega for floating_strike_lookback_call."""
    pricer = lambda spot, vol: floating_strike_lookback_call(spot, T, r, vol, delta=delta, L=L)
    return _finite_difference_greeks(pricer, S0, sigma, dS=dS, dSigma=dSigma)


def floating_strike_lookback_put_greeks(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    delta: float = 0.0,
    L: Optional[float] = None,
    dS: Optional[float] = None,
    dSigma: Optional[float] = None,
) -> OptionGreeks:
    """Finite-difference Delta/Gamma/Vega for floating_strike_lookback_put."""
    pricer = lambda spot, vol: floating_strike_lookback_put(spot, T, r, vol, delta=delta, L=L)
    return _finite_difference_greeks(pricer, S0, sigma, dS=dS, dSigma=dSigma)


# -----------------------------------------------------------------------------
# Payoffs (for Monte Carlo or backtesting)
# -----------------------------------------------------------------------------

def payoff_floating_strike_call(S_T: float, Y_T: float) -> float:
    """Payoff at expiry: S_T - min(S_t) = S_T - Y_T."""
    return max(S_T - Y_T, 0.0)


def payoff_floating_strike_put(Z_T: float, S_T: float) -> float:
    """Payoff at expiry: max(S_t) - S_T = Z_T - S_T."""
    return max(Z_T - S_T, 0.0)


# -----------------------------------------------------------------------------
# Example / CLI
# -----------------------------------------------------------------------------

@dataclass
class Params:
    S0: float = 100.0
    T: float = 1.0
    r: float = 0.05
    sigma: float = 0.2
    delta: float = 0.0


if __name__ == "__main__":
    p = Params()
    c = floating_strike_lookback_call(p.S0, p.T, p.r, p.sigma, delta=p.delta)
    pt = floating_strike_lookback_put(p.S0, p.T, p.r, p.sigma, delta=p.delta)
    print("Floating-strike lookback (from lookback.txt formulas)")
    print(f"  S0={p.S0}, T={p.T}, r={p.r}, σ={p.sigma}, δ={p.delta}")
    print(f"  Call (value): {c:.4f}")
    print(f"  Put (value):  {pt:.4f}")
    # With non-zero dividend
    p_div = Params(delta=0.02)
    c_div = floating_strike_lookback_call(p_div.S0, p_div.T, p_div.r, p_div.sigma, delta=p_div.delta)
    pt_div = floating_strike_lookback_put(p_div.S0, p_div.T, p_div.r, p_div.sigma, delta=p_div.delta)
    print(f"  Call (δ=0.02): {c_div:.4f}")
    print(f"  Put (δ=0.02):  {pt_div:.4f}")
    # With optional current extremum L
    c_L = floating_strike_lookback_call(p.S0, p.T, p.r, p.sigma, delta=p.delta, L=95.0)
    pt_L = floating_strike_lookback_put(p.S0, p.T, p.r, p.sigma, delta=p.delta, L=108.0)
    print(f"  Call (L_min=95):  {c_L:.4f}")
    print(f"  Put (L_max=108):  {pt_L:.4f}")
    print(f"  Payoff call (S_T=105, Y_T=90): {payoff_floating_strike_call(105.0, 90.0):.4f}")
    print(f"  Payoff put (Z_T=110, S_T=100): {payoff_floating_strike_put(110.0, 100.0):.4f}")
