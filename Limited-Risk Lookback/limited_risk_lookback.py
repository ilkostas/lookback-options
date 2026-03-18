"""
European Limited Risk Lookback Options — Black–Scholes closed-form with dividend yield.

Limited risk call (Up-and-Out Call, Conze–Viswanathan 1991):
- Payoff (S_T - K) if the maximum asset price remains strictly below the barrier m; else 0.
- Equivalent to standard Up-and-Out Call. Price is positive and bounded above by the
  standard Black–Scholes call. Formula uses bear-spread terms [N(d_1)-N(x_1)], etc.,
  and reflection terms with lambda = (r - δ + σ²/2)/σ².

Limited risk put (Down-and-Out Put):
- Payoff max(K - S_T, 0) unless the historical minimum falls below the barrier; else 0.

Notation: S_0 spot, K strike, T maturity, r rate, δ (delta) continuous dividend yield, σ vol,
barrier = cut-off level (upper for call, lower for put). Optional M_0 (call) / m_0 (put).
BGK continuity correction is intentionally not applied here: limited-risk/barrier lookbacks
need a product-specific discrete-to-continuous correction, not a blanket extremum shift.
"""

from math import log, sqrt, exp, erf
from typing import Callable, Optional
from dataclasses import dataclass


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


# -----------------------------------------------------------------------------
# Auxiliary distance functions (dividend-adjusted from lookback.txt)
# d   = [ln(S_0/K) + (r - δ + σ²/2)T] / (σ√T)
# d_m = [ln(S_0/m) + (r - δ + σ²/2)T] / (σ√T)
# d'  = [ln(S_0/m) + (r - δ - σ²/2)T] / (σ√T)
# -----------------------------------------------------------------------------

def _d(S: float, K: float, T: float, r: float, delta: float, sigma: float) -> float:
    return (log(S / K) + (r - delta + 0.5 * sigma**2) * T) / (sigma * sqrt(T))


def _d_m(S: float, m: float, T: float, r: float, delta: float, sigma: float) -> float:
    return (log(S / m) + (r - delta + 0.5 * sigma**2) * T) / (sigma * sqrt(T))


# -----------------------------------------------------------------------------
# Limited risk call (Up-and-Out Call): (S_T - K) if max stays < barrier else 0
# Equivalent to Conze–Viswanathan (1991) UOC / Buchen–Konstandatos Method-of-Images form.
# Formula: bear-spread + reflection penalty. m > K, m > S_0.
# lambda = (r - δ + σ²/2)/σ²; d_1,d_2; x_1,x_2; y_1,y_2.
# Note: y_2 is defined with (r - δ + σ²/2) drift; bond term uses N(y_2 - σ√T), which yields d'-style drift (r - δ - σ²/2).
# -----------------------------------------------------------------------------

def _lambda_uoc(r: float, delta: float, sigma: float) -> float:
    """Barrier reflection exponent for UOC: (r - δ + σ²/2) / σ²."""
    return (r - delta + 0.5 * sigma**2) / (sigma**2)


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


def limited_risk_lookback_call(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    barrier: float,
    delta: float = 0.0,
    M_0: Optional[float] = None,
) -> float:
    """
    Limited risk lookback call (Up-and-Out Call). Payoff = (S_T - K) if the
    running maximum stays strictly below the barrier; else 0. Equivalent to
    standard UOC. Price is positive and bounded above by the Black–Scholes call.
    barrier: upper cut-off (maximum allowed). M_0: current realized max; None = S_0.
    delta: continuous dividend yield δ (optional, default 0).
    """
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive")
    if T <= 0:
        cur_max = S0 if M_0 is None else M_0
        return (max(S0 - K, 0.0) if cur_max < barrier else 0.0)
    cur_max = S0 if M_0 is None else M_0
    if cur_max >= barrier:
        return 0.0
    m = barrier
    sig_sqrt_T = sigma * sqrt(T)
    lam = _lambda_uoc(r, delta, sigma)
    log_m_over_s0 = log(m / S0)
    # d_1, d_2 (strike K); x_1, x_2 (barrier m); y_1, y_2 (reflection; y_2 - σ√T gives d' drift)
    d_1 = (log(S0 / K) + (r - delta + 0.5 * sigma**2) * T) / sig_sqrt_T
    d_2 = d_1 - sig_sqrt_T
    x_1 = (log(S0 / m) + (r - delta + 0.5 * sigma**2) * T) / sig_sqrt_T
    x_2 = x_1 - sig_sqrt_T
    y_1 = (log(m * m / (S0 * K)) + (r - delta + 0.5 * sigma**2) * T) / sig_sqrt_T
    y_2 = (log(m / S0) + (r - delta + 0.5 * sigma**2) * T) / sig_sqrt_T
    # C = S_0*exp(-δ*T)*[N(d_1)-N(x_1)] - exp(-r*T)*K*[N(d_2)-N(x_2)]
    #   - S_0*exp(-δ*T)*(m/S_0)^(2*λ)*[N(y_1)-N(y_2)]
    #   + exp(-r*T)*K*(m/S_0)^(2*λ-2)*[N(y_1 - σ√T) - N(y_2 - σ√T)]
    base_spread_call = (
        S0 * exp(-delta * T) * (norm_cdf(d_1) - norm_cdf(x_1))
        - exp(-r * T) * K * (norm_cdf(d_2) - norm_cdf(x_2))
    )
    reflection_penalty = (
        S0 * exp(-delta * T) * exp((2 * lam) * log_m_over_s0) * (norm_cdf(y_1) - norm_cdf(y_2))
        - exp(-r * T) * K * exp((2 * lam - 2) * log_m_over_s0)
        * (norm_cdf(y_1 - sig_sqrt_T) - norm_cdf(y_2 - sig_sqrt_T))
    )
    return base_spread_call - reflection_penalty


# -----------------------------------------------------------------------------
# Limited risk put: max(K - S_T, 0) unless running minimum <= barrier → 0
# barrier = lower cut-off (minimum allowed). m_0 = current realized minimum.
# Reflection term follows same conventions as the call (Conze–Viswanathan / Buchen–Konstandatos).
# -----------------------------------------------------------------------------

def limited_risk_lookback_put(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    barrier: float,
    delta: float = 0.0,
    m_0: Optional[float] = None,
) -> float:
    """
    Limited risk lookback put. Payoff = standard put max(K - S_T, 0) unless the
    historical minimum falls below the barrier (cut-off), in which case payoff = 0.
    barrier: lower cut-off level (minimum allowed for the asset path).
    m_0: current realized minimum from inception to today; if None, pricing at inception (m_0 = S_0).
    delta: continuous dividend yield δ (optional, default 0).
    """
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive")
    if T <= 0:
        cur_min = S0 if m_0 is None else m_0
        return (max(K - S0, 0.0) if cur_min > barrier else 0.0)
    cur_min = S0 if m_0 is None else m_0
    if cur_min <= barrier:
        return 0.0
    m = barrier
    sig_sqrt_T = sigma * sqrt(T)
    lam = _lambda_uoc(r, delta, sigma)
    log_m_over_s0 = log(m / S0)
    # Use the same distance structure as the call: d_1, d_2; x_1, x_2; y_1, y_2.
    d_1 = (log(S0 / K) + (r - delta + 0.5 * sigma**2) * T) / sig_sqrt_T
    d_2 = d_1 - sig_sqrt_T
    x_1 = (log(S0 / m) + (r - delta + 0.5 * sigma**2) * T) / sig_sqrt_T
    x_2 = x_1 - sig_sqrt_T
    y_1 = (log(m * m / (S0 * K)) + (r - delta + 0.5 * sigma**2) * T) / sig_sqrt_T
    y_2 = (log(m / S0) + (r - delta + 0.5 * sigma**2) * T) / sig_sqrt_T
    # P = -S_0*exp(-δ*T)*[N(-d_1)-N(-x_1)] + exp(-r*T)*K*[N(-d_2)-N(-x_2)]
    #   + S_0*exp(-δ*T)*(m/S_0)^(2*λ)*[N(y_1)-N(y_2)]
    #   - exp(-r*T)*K*(m/S_0)^(2*λ-2)*[N(y_1 - σ√T) - N(y_2 - σ√T)]
    base_spread_put = (
        -S0 * exp(-delta * T) * (norm_cdf(-d_1) - norm_cdf(-x_1))
        + exp(-r * T) * K * (norm_cdf(-d_2) - norm_cdf(-x_2))
    )
    reflection_penalty = (
        S0 * exp(-delta * T) * exp((2 * lam) * log_m_over_s0) * (norm_cdf(y_1) - norm_cdf(y_2))
        - exp(-r * T) * K * exp((2 * lam - 2) * log_m_over_s0)
        * (norm_cdf(y_1 - sig_sqrt_T) - norm_cdf(y_2 - sig_sqrt_T))
    )
    return base_spread_put + reflection_penalty


def limited_risk_lookback_call_greeks(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    barrier: float,
    delta: float = 0.0,
    M_0: Optional[float] = None,
    dS: Optional[float] = None,
    dSigma: Optional[float] = None,
) -> OptionGreeks:
    """Finite-difference Delta/Gamma/Vega for limited_risk_lookback_call."""
    pricer = lambda spot, vol: limited_risk_lookback_call(
        spot, K, T, r, vol, barrier, delta=delta, M_0=M_0
    )
    return _finite_difference_greeks(pricer, S0, sigma, dS=dS, dSigma=dSigma)


def limited_risk_lookback_put_greeks(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    barrier: float,
    delta: float = 0.0,
    m_0: Optional[float] = None,
    dS: Optional[float] = None,
    dSigma: Optional[float] = None,
) -> OptionGreeks:
    """Finite-difference Delta/Gamma/Vega for limited_risk_lookback_put."""
    pricer = lambda spot, vol: limited_risk_lookback_put(
        spot, K, T, r, vol, barrier, delta=delta, m_0=m_0
    )
    return _finite_difference_greeks(pricer, S0, sigma, dS=dS, dSigma=dSigma)


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
    barrier_call: float = 120.0   # upper cut-off for call
    barrier_put: float = 80.0    # lower cut-off for put
    delta: float = 0.0


if __name__ == "__main__":
    p = Params()
    c = limited_risk_lookback_call(p.S0, p.K, p.T, p.r, p.sigma, p.barrier_call, delta=p.delta)
    pt = limited_risk_lookback_put(p.S0, p.K, p.T, p.r, p.sigma, p.barrier_put, delta=p.delta)
    print("Limited risk lookback (call = UOC Conze–Viswanathan-style formula)")
    print(f"  S0={p.S0}, K={p.K}, T={p.T}, r={p.r}, σ={p.sigma}, δ={p.delta}")
    print(f"  Call (barrier={p.barrier_call}): {c:.4f}")
    print(f"  Put (barrier={p.barrier_put}):  {pt:.4f}")
    # With dividend
    p_div = Params(delta=0.02)
    c_div = limited_risk_lookback_call(
        p_div.S0, p_div.K, p_div.T, p_div.r, p_div.sigma, p_div.barrier_call, delta=p_div.delta
    )
    pt_div = limited_risk_lookback_put(
        p_div.S0, p_div.K, p_div.T, p_div.r, p_div.sigma, p_div.barrier_put, delta=p_div.delta
    )
    print(f"  Call (δ=0.02): {c_div:.4f}")
    print(f"  Put (δ=0.02):  {pt_div:.4f}")
    # With optional current extremum
    c_M0 = limited_risk_lookback_call(
        p.S0, p.K, p.T, p.r, p.sigma, p.barrier_call, delta=p.delta, M_0=115.0
    )
    pt_m0 = limited_risk_lookback_put(
        p.S0, p.K, p.T, p.r, p.sigma, p.barrier_put, delta=p.delta, m_0=85.0
    )
    print(f"  Call (M_0=115): {c_M0:.4f}")
    print(f"  Put (m_0=85):   {pt_m0:.4f}")
    # Barrier already breached → 0
    c_knock = limited_risk_lookback_call(
        p.S0, p.K, p.T, p.r, p.sigma, p.barrier_call, delta=p.delta, M_0=125.0
    )
    pt_knock = limited_risk_lookback_put(
        p.S0, p.K, p.T, p.r, p.sigma, p.barrier_put, delta=p.delta, m_0=75.0
    )
    print(f"  Call (M_0=125 >= barrier): {c_knock:.4f}")
    print(f"  Put (m_0=75 <= barrier):  {pt_knock:.4f}")
