"""
European Fixed-Strike Lookback Options — Black–Scholes closed-form.

From lookback.txt:
- Call on maximum: payoff max(Z_T - K, 0), Z_T = max(S_t) for 0 <= t <= T.
- Put on minimum:  payoff max(K - Y_T, 0), Y_T = min(S_t) for 0 <= t <= T.
- Notation: S_0 spot, K strike, T maturity, r rate, δ (delta) dividend yield, σ vol.
- Auxiliary: d_{1,K} = (ln(S_0/K) + (r - δ + σ²/2)T) / (σ√T),
             d_{2,K} = d_{1,K} - σ√T,
             d_{3,K} = d_{1,K} - 2(r - δ)√T/σ.
- Pricing splits on current extremum L:
  Call: Condition 1 K >= L (asset has not exceeded strike); Condition 2 K < L (already ITM).
  Put:  Condition 1 K <= L (asset has not fallen below strike); Condition 2 K > L (already ITM).
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
        # Fallback: Abramowitz & Stegun approximation
        a1, a2, a3, a4, a5 = 0.31938153, -0.356563782, 1.781477937, -1.821255978, 1.330274429
        t = 1.0 / (1.0 + 0.2316419 * abs(x))
        n = (a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5) * exp(-x * x / 2) / sqrt(2 * 3.141592653589793)
        # Abramowitz & Stegun approximation: n → 0 as |x| → ∞,
        # so Φ(x) = 1 - n for x ≥ 0 and Φ(x) = n for x < 0.
        return 1 - n if x >= 0 else n


# -----------------------------------------------------------------------------
# Auxiliary distance functions (from lookback.txt)
# d_{1,K} = (ln(S_0/K) + (r - δ + σ²/2)T) / (σ√T)
# d_{2,K} = d_{1,K} - σ√T
# d_{3,K} = d_{1,K} - 2(r - δ)√T/σ
# -----------------------------------------------------------------------------

def d1(S: float, K: float, T: float, r: float, delta: float, sigma: float) -> float:
    return (log(S / K) + (r - delta + 0.5 * sigma**2) * T) / (sigma * sqrt(T))


def d2(S: float, K: float, T: float, r: float, delta: float, sigma: float) -> float:
    return d1(S, K, T, r, delta, sigma) - sigma * sqrt(T)


def d3(S: float, K: float, T: float, r: float, delta: float, sigma: float) -> float:
    return d1(S, K, T, r, delta, sigma) - 2 * (r - delta) * sqrt(T) / sigma


# -----------------------------------------------------------------------------
# European fixed-strike lookback call (on maximum)
# Payoff: max(max(S_t) - K, 0). With current maximum L:
#   Condition 1: K >= L -> price using formula in (S_0, K, T).
#   Condition 2: K < L  -> already ITM: value = e^{-r*τ}(L - K).
# -----------------------------------------------------------------------------

def fixed_strike_lookback_call(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    delta: float = 0.0,
    L: Optional[float] = None,
) -> float:
    """
    European fixed-strike lookback call (on maximum).
    L: current realized maximum from inception to today; if None, pricing at inception.
    delta: continuous dividend yield δ (optional, default 0).
    """
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive")
    if T <= 0:
        return max((L if L is not None else S0) - K, 0.0)
    b = r - delta
    _d1 = d1(S0, K, T, r, delta, sigma)
    _d2 = d2(S0, K, T, r, delta, sigma)
    _d3 = d3(S0, K, T, r, delta, sigma)
    term1 = S0 * exp(-delta * T) * norm_cdf(_d1)
    term2 = K * exp(-r * T) * norm_cdf(_d2)
    expo = -2 * b / (sigma**2)
    if abs(b) < 1e-12:
        term3 = 0.5 * (sigma**2) * T * S0 * (norm_cdf(-_d3) - exp(-r * T) * norm_cdf(-_d2))
    else:
        # b ≠ 0: use the standard closed-form term that smoothly
        # connects to the b → 0 limit above.
        term3 = S0 * (sigma**2 / (2 * b)) * (
            exp(-delta * T) * norm_cdf(_d1) - (S0 / K) ** expo * exp(-r * T) * norm_cdf(_d3)
        )
    return term1 - term2 + term3


# -----------------------------------------------------------------------------
# European fixed-strike lookback put (on minimum)
# Payoff: max(K - min(S_t), 0). With current minimum L:
#   Condition 1: K <= L -> price using formula in (S_0, K, T).
#   Condition 2: K > L  -> already ITM: value = e^{-r*τ}(K - L).
# -----------------------------------------------------------------------------

def fixed_strike_lookback_put(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    delta: float = 0.0,
    L: Optional[float] = None,
) -> float:
    """
    European fixed-strike lookback put (on minimum).
    L: current realized minimum from inception to today; if None, pricing at inception.
    delta: continuous dividend yield δ (optional, default 0).
    """
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive")
    if T <= 0:
        return max(K - min(S0, L if L is not None else S0), 0.0)
    b = r - delta
    _d1 = d1(S0, K, T, r, delta, sigma)
    _d2 = d2(S0, K, T, r, delta, sigma)
    _d3 = d3(S0, K, T, r, delta, sigma)
    term1 = K * exp(-r * T) * norm_cdf(-_d2)
    term2 = S0 * exp(-delta * T) * norm_cdf(-_d1)
    expo = -2 * b / (sigma**2)
    if abs(b) < 1e-12:
        term3 = 0.5 * (sigma**2) * T * S0 * (-norm_cdf(_d3) + exp(-r * T) * norm_cdf(_d2))
    else:
        # b ≠ 0: use the standard closed-form term that smoothly
        # connects to the b → 0 limit above.
        term3 = S0 * (sigma**2 / (2 * b)) * (
            (S0 / K) ** expo * exp(-r * T) * norm_cdf(-_d3) - exp(-delta * T) * norm_cdf(-_d1)
        )
    return term1 - term2 + term3


# -----------------------------------------------------------------------------
# Payoffs (for Monte Carlo or backtesting)
# -----------------------------------------------------------------------------

def payoff_fixed_strike_call_on_max(S_max: float, K: float) -> float:
    """Payoff at expiry: max(Z_T - K, 0)."""
    return max(S_max - K, 0.0)


def payoff_fixed_strike_put_on_min(S_min: float, K: float) -> float:
    """Payoff at expiry: max(K - Y_T, 0)."""
    return max(K - S_min, 0.0)


# -----------------------------------------------------------------------------
# Example / CLI
# -----------------------------------------------------------------------------

@dataclass
class Params:
    S0: float = 100.0
    K: float = 100.0
    T: float = 1.0
    r: float = 0.05
    sigma: float = 0.2
    delta: float = 0.0


if __name__ == "__main__":
    p = Params()
    c = fixed_strike_lookback_call(p.S0, p.K, p.T, p.r, p.sigma, delta=p.delta)
    pt = fixed_strike_lookback_put(p.S0, p.K, p.T, p.r, p.sigma, delta=p.delta)
    print("Fixed-strike lookback (from lookback.txt formulas)")
    print(f"  S0={p.S0}, K={p.K}, T={p.T}, r={p.r}, σ={p.sigma}, δ={p.delta}")
    print(f"  Call on maximum (value): {c:.4f}")
    print(f"  Put on minimum (value):  {pt:.4f}")
    # With dividend
    p_div = Params(delta=0.02)
    c_div = fixed_strike_lookback_call(p_div.S0, p_div.K, p_div.T, p_div.r, p_div.sigma, delta=p_div.delta)
    print(f"  Call on maximum (δ=0.02): {c_div:.4f}")
    print(f"  Payoff call (if S_max=110): {payoff_fixed_strike_call_on_max(110.0, p.K):.4f}")
    print(f"  Payoff put (if S_min=90):  {payoff_fixed_strike_put_on_min(90.0, p.K):.4f}")
