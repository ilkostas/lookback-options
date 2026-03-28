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
# d_{1,K} = (ln(S_0/K) + (r - δ + σ²/2)T) / (σ√T)
# d_{2,K} = d_{1,K} - σ√T
# d_{3,K} = d_{1,K} - 2(r - δ)√T/σ
# -----------------------------------------------------------------------------

def d1(S: float, K: float, T: float, r: float, delta: float, sigma: float) -> float:
    return (log(S / K) + (r - delta + 0.5 * sigma**2) * T) / (sigma * sqrt(T))


def d2(S: float, K: float, T: float, r: float, delta: float, sigma: float) -> float:
    sqrt_T = sqrt(T)
    return (log(S / K) + (r - delta - 0.5 * sigma**2) * T) / (sigma * sqrt_T)


def d3(S: float, K: float, T: float, r: float, delta: float, sigma: float) -> float:
    sqrt_T = sqrt(T)
    return (log(S / K) + (-r + delta + 0.5 * sigma**2) * T) / (sigma * sqrt_T)


def _d123(S: float, K: float, T: float, r: float, delta: float, sigma: float) -> tuple[float, float, float]:
    sqrt_T = sqrt(T)
    log_ratio = log(S / K)
    sigma_sq = sigma**2
    d1_val = (log_ratio + (r - delta + 0.5 * sigma_sq) * T) / (sigma * sqrt_T)
    d2_val = d1_val - sigma * sqrt_T
    d3_val = d1_val - 2 * (r - delta) * sqrt_T / sigma
    return d1_val, d2_val, d3_val


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
    _d1, _d2, _d3 = _d123(S0, K, T, r, delta, sigma)
    disc_div = exp(-delta * T)
    disc_rf = exp(-r * T)
    term1 = S0 * disc_div * norm_cdf(_d1)
    term2 = K * disc_rf * norm_cdf(_d2)
    expo = -2 * b / (sigma**2)
    log_ratio = log(S0 / K)
    if abs(b) < 1e-12:
        term3 = 0.5 * (sigma**2) * T * S0 * (norm_cdf(-_d3) - disc_rf * norm_cdf(-_d2))
    else:
        # b ≠ 0: use the standard closed-form term that smoothly
        # connects to the b → 0 limit above.
        term3 = S0 * (sigma**2 / (2 * b)) * (
            disc_div * norm_cdf(_d1) - exp(expo * log_ratio) * disc_rf * norm_cdf(_d3)
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
    _d1, _d2, _d3 = _d123(S0, K, T, r, delta, sigma)
    disc_div = exp(-delta * T)
    disc_rf = exp(-r * T)
    term1 = K * disc_rf * norm_cdf(-_d2)
    term2 = S0 * disc_div * norm_cdf(-_d1)
    expo = -2 * b / (sigma**2)
    log_ratio = log(S0 / K)
    if abs(b) < 1e-12:
        term3 = 0.5 * (sigma**2) * T * S0 * (-norm_cdf(_d3) + disc_rf * norm_cdf(_d2))
    else:
        # b ≠ 0: use the standard closed-form term that smoothly
        # connects to the b → 0 limit above.
        term3 = S0 * (sigma**2 / (2 * b)) * (
            exp(expo * log_ratio) * disc_rf * norm_cdf(-_d3) - disc_div * norm_cdf(-_d1)
        )
    return term1 - term2 + term3


# -----------------------------------------------------------------------------
# BGK continuity correction (discrete-monitoring proxy for continuous formulas)
# -----------------------------------------------------------------------------

def fixed_strike_lookback_call_bgk(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    monitoring_points: int,
    delta: float = 0.0,
    L: Optional[float] = None,
) -> float:
    """
    BGK-corrected fixed-strike call approximation (Broadie-Glasserman-Kou, Theorem 4).
    """
    _validate_monitoring_points(monitoring_points)
    if T <= 0:
        return fixed_strike_lookback_call(S0, K, T, r, sigma, delta=delta, L=L)
    corr = _BGK_BETA1 * sigma * sqrt(T / monitoring_points)
    eta = exp(corr)
    l_adjusted = None if L is None else L * eta
    v_cont = fixed_strike_lookback_call(S0 * eta, K * eta, T, r, sigma, delta=delta, L=l_adjusted)
    return exp(-corr) * v_cont


def fixed_strike_lookback_put_bgk(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    monitoring_points: int,
    delta: float = 0.0,
    L: Optional[float] = None,
) -> float:
    """
    BGK-corrected fixed-strike put approximation (Broadie-Glasserman-Kou, Theorem 4).
    """
    _validate_monitoring_points(monitoring_points)
    if T <= 0:
        return fixed_strike_lookback_put(S0, K, T, r, sigma, delta=delta, L=L)
    corr = _BGK_BETA1 * sigma * sqrt(T / monitoring_points)
    eta = exp(-corr)
    l_adjusted = None if L is None else L * eta
    v_cont = fixed_strike_lookback_put(S0 * eta, K * eta, T, r, sigma, delta=delta, L=l_adjusted)
    return exp(corr) * v_cont


def fixed_strike_lookback_call_greeks(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    delta: float = 0.0,
    L: Optional[float] = None,
    dS: Optional[float] = None,
    dSigma: Optional[float] = None,
) -> OptionGreeks:
    """Finite-difference Delta/Gamma/Vega for fixed_strike_lookback_call."""
    pricer = lambda spot, vol: fixed_strike_lookback_call(spot, K, T, r, vol, delta=delta, L=L)
    return _finite_difference_greeks(pricer, S0, sigma, dS=dS, dSigma=dSigma)


def fixed_strike_lookback_put_greeks(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    delta: float = 0.0,
    L: Optional[float] = None,
    dS: Optional[float] = None,
    dSigma: Optional[float] = None,
) -> OptionGreeks:
    """Finite-difference Delta/Gamma/Vega for fixed_strike_lookback_put."""
    pricer = lambda spot, vol: fixed_strike_lookback_put(spot, K, T, r, vol, delta=delta, L=L)
    return _finite_difference_greeks(pricer, S0, sigma, dS=dS, dSigma=dSigma)


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
