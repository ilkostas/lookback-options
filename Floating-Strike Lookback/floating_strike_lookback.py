"""
European Floating-Strike (Standard) Lookback Options — Black–Scholes closed-form.

From lookback.txt:
- Call: strike floats to the minimum; payoff S_T - min(S_t) = S_T - Y_T. L = historically realized minimum.
- Put:  strike floats to the maximum; payoff max(S_t) - S_T = Z_T - S_T. L = historically realized maximum.
- Notation: S_0 spot, T maturity, r rate, δ (delta) dividend yield, σ vol, L current extremum (from inception to today).
- Auxiliary d_1, d_2, d_3 as in lookback.txt (with δ). Supports δ ≠ 0 and optional L.
"""

from math import log, sqrt, exp
from typing import Optional
from dataclasses import dataclass


def norm_cdf(x: float) -> float:
    """Cumulative distribution function of the standard normal N(d)."""
    try:
        from scipy.stats import norm
        return float(norm.cdf(x))
    except ImportError:
        a1, a2, a3, a4, a5 = 0.31938153, -0.356563782, 1.781477937, -1.821255978, 1.330274429
        t = 1.0 / (1.0 + 0.2316419 * abs(x))
        n = (a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5) * exp(-x * x / 2) / sqrt(2 * 3.141592653589793)
        return 1 - n if x < 0 else n


def norm_pdf(x: float) -> float:
    """Probability density function of the standard normal N(d)."""
    return exp(-x * x / 2.0) / sqrt(2.0 * 3.141592653589793)


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
    if T <= 0:
        return max(S0 - (L if L is not None else S0), 0.0)
    L_use = S0 if L is None else L
    b = r - delta
    expo = 2.0 * b / (sigma**2)
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
                - disc_rf * (L_use / S0) ** expo * norm_cdf(-_d3)
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
    if T <= 0:
        return max((L if L is not None else S0) - S0, 0.0)
    L_use = S0 if L is None else L
    b = r - delta
    expo = 2.0 * b / (sigma**2)
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
                - disc_rf * (L_use / S0) ** expo * norm_cdf(_d3)
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
