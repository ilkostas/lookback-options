"""
European Reverse-Strike Lookback Options — closed-form and Monte Carlo.

From the article and lookback.txt:
- Call: payoff max(Y_T - K, 0). If current minimum L < K, option is guaranteed worthless (price 0).
- Put:  payoff max(K - Z_T, 0). If current maximum L > K, option is guaranteed worthless (price 0).
- Pricing via generic lookback decomposition:
  Vc = m(x, y, τ) - m(x, min(y, k), τ),  Vp = M(x, max(z, k), τ) - M(x, z, τ)
  with m = x - C_y - D⁻_y,  M = x + P_z + D⁺_z (European C/P and lookback premium D with dividend).
- Near b = r - δ = 0, generic m/M switch to explicit finite-limit formulas to avoid the singular α/β decomposition.
- Supports optional current extremum L (y for call, z for put) and continuous dividend yield δ.
BGK continuity correction is intentionally not wrapped here: reverse-strike structures
need product-specific discrete-to-continuous handling.
"""

from math import log, sqrt, exp, erf, pi
from typing import Callable, Optional
from dataclasses import dataclass

import numpy as np

_B_ZERO_EPS = 1e-5


@dataclass
class OptionGreeks:
    delta: float
    gamma: float
    vega: float


def norm_cdf(x: float) -> float:
    """Cumulative distribution function of the standard normal N(d)."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Probability density function of the standard normal."""
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


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
# European vanilla call/put (d1, d2 with r - delta; for generic m and M)
# -----------------------------------------------------------------------------

def _d1(S: float, K: float, T: float, r: float, delta: float, sigma: float) -> float:
    return (log(S / K) + (r - delta + 0.5 * sigma**2) * T) / (sigma * sqrt(T))


def _d2(S: float, K: float, T: float, r: float, delta: float, sigma: float) -> float:
    sqrt_T = sqrt(T)
    return (log(S / K) + (r - delta - 0.5 * sigma**2) * T) / (sigma * sqrt_T)


def _d12(S: float, K: float, T: float, r: float, delta: float, sigma: float) -> tuple[float, float]:
    sqrt_T = sqrt(T)
    d1_val = (log(S / K) + (r - delta + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    return d1_val, d1_val - sigma * sqrt_T


def _european_call(S0: float, K: float, T: float, r: float, sigma: float, delta: float = 0.0) -> float:
    if T <= 0:
        return max(S0 - K, 0.0)
    d1_val, d2_val = _d12(S0, K, T, r, delta, sigma)
    return S0 * exp(-delta * T) * norm_cdf(d1_val) - K * exp(-r * T) * norm_cdf(d2_val)


def _european_put(S0: float, K: float, T: float, r: float, sigma: float, delta: float = 0.0) -> float:
    if T <= 0:
        return max(K - S0, 0.0)
    d1_val, d2_val = _d12(S0, K, T, r, delta, sigma)
    return K * exp(-r * T) * norm_cdf(-d2_val) - S0 * exp(-delta * T) * norm_cdf(-d1_val)


# -----------------------------------------------------------------------------
# D building blocks with dividend: drift b = r - delta in d_ξ and in α, β.
# Discount in bond binaries remains e^{-r τ}.
# -----------------------------------------------------------------------------

def _alpha_beta(r: float, delta: float, sigma: float) -> tuple[float, float]:
    """α = σ²/(2b), β = 2b/σ² - 1 with b = r - δ. Guard when |b| small."""
    b = r - delta
    if abs(b) < 1e-12:
        return 0.5 * sigma**2, -1.0
    alpha = sigma**2 / (2 * b)
    beta = (2 * b / sigma**2) - 1
    return alpha, beta


def _d_xi(x: float, xi: float, tau: float, b: float, sigma: float) -> float:
    """d_ξ with drift b = r - δ."""
    if tau <= 0:
        return 0.0
    return (log(x / xi) + (b + 0.5 * sigma**2) * tau) / (sigma * sqrt(tau))


def _d_prime_xi(x: float, xi: float, tau: float, b: float, sigma: float) -> float:
    return _d_xi(x, xi, tau, b, sigma) - sigma * sqrt(tau)


def _bar_d_prime_xi(x: float, xi: float, tau: float, b: float, sigma: float) -> float:
    """bar{d}'_ξ with drift b."""
    if tau <= 0:
        return 0.0
    return (log(xi / x) + (b - 0.5 * sigma**2) * tau) / (sigma * sqrt(tau))


def _A_minus(x: float, xi: float, tau: float, b: float, sigma: float) -> float:
    """A^-_ξ = x * N(-d_ξ)."""
    return x * norm_cdf(-_d_xi(x, xi, tau, b, sigma))


def _B_plus_image(x: float, xi: float, tau: float, r: float, b: float, sigma: float, beta: float) -> float:
    """*B^+_ξ = (ξ/x)^β e^{-rτ} N(bar{d}'_ξ)."""
    if x <= 0 or xi <= 0:
        return 0.0
    return exp(beta * log(xi / x)) * exp(-r * tau) * norm_cdf(_bar_d_prime_xi(x, xi, tau, b, sigma))


def _D_minus(x: float, xi: float, tau: float, r: float, delta: float, sigma: float) -> float:
    """D^-_ξ = -α [ A^-_ξ - ξ * *B^+_ξ ]. Uses b = r - delta."""
    if tau <= 0:
        return 0.0
    b = r - delta
    alpha, beta = _alpha_beta(r, delta, sigma)
    return -alpha * (_A_minus(x, xi, tau, b, sigma) - xi * _B_plus_image(x, xi, tau, r, b, sigma, beta))


def _A_plus(x: float, xi: float, tau: float, b: float, sigma: float) -> float:
    """A^+_ξ = x * N(d_ξ)."""
    return x * norm_cdf(_d_xi(x, xi, tau, b, sigma))


def _B_minus_image(x: float, xi: float, tau: float, r: float, b: float, sigma: float, beta: float) -> float:
    """*B^-_ξ = (ξ/x)^β e^{-rτ} N(-bar{d}'_ξ)."""
    if x <= 0 or xi <= 0:
        return 0.0
    return exp(beta * log(xi / x)) * exp(-r * tau) * norm_cdf(-_bar_d_prime_xi(x, xi, tau, b, sigma))


def _D_plus(x: float, xi: float, tau: float, r: float, delta: float, sigma: float) -> float:
    """D^+_ξ = α [ A^+_ξ - ξ * *B^-_ξ ]. Uses b = r - delta."""
    if tau <= 0:
        return 0.0
    b = r - delta
    alpha, beta = _alpha_beta(r, delta, sigma)
    return alpha * (_A_plus(x, xi, tau, b, sigma) - xi * _B_minus_image(x, xi, tau, r, b, sigma, beta))


# -----------------------------------------------------------------------------
# Generic lookback values: m(x,y,τ) = x - C_y - D⁻_y,  M(x,z,τ) = x + P_z + D⁺_z
# -----------------------------------------------------------------------------

def _generic_min_value(S0: float, y: float, T: float, r: float, delta: float, sigma: float) -> float:
    """Value of contract that pays the minimum at expiry: m = x - C_y - D⁻_y."""
    if T <= 0:
        return min(S0, y)
    b = r - delta
    if abs(b) <= _B_ZERO_EPS:
        # Exact b -> 0 limit (Buchen-Konstandatos / L'Hopital branch):
        # m(x,y,tau) = e^{-r tau}[ y N(d') + x N(-d) + sigma*sqrt(tau)*x*(d*N(-d) - phi(d)) ].
        sqrt_tau = sqrt(T)
        d = (log(S0 / y) / (sigma * sqrt_tau)) + 0.5 * sigma * sqrt_tau
        d_prime = d - sigma * sqrt_tau
        disc = exp(-r * T)
        return disc * (
            y * norm_cdf(d_prime)
            + S0 * norm_cdf(-d)
            + sigma * sqrt_tau * S0 * (d * norm_cdf(-d) - norm_pdf(d))
        )
    C_y = _european_call(S0, y, T, r, sigma, delta)
    D_minus = _D_minus(S0, y, T, r, delta, sigma)
    return S0 - C_y - D_minus


def _generic_max_value(S0: float, z: float, T: float, r: float, delta: float, sigma: float) -> float:
    """Value of contract that pays the maximum at expiry: M = x + P_z + D⁺_z."""
    if T <= 0:
        return max(S0, z)
    b = r - delta
    if abs(b) <= _B_ZERO_EPS:
        # Symmetric exact b -> 0 counterpart for maximum-paying contract.
        sqrt_tau = sqrt(T)
        d = (log(S0 / z) / (sigma * sqrt_tau)) + 0.5 * sigma * sqrt_tau
        d_prime = d - sigma * sqrt_tau
        disc = exp(-r * T)
        return disc * (
            z * norm_cdf(-d_prime)
            + S0 * norm_cdf(d)
            + sigma * sqrt_tau * S0 * (d * norm_cdf(d) + norm_pdf(d))
        )
    P_z = _european_put(S0, z, T, r, sigma, delta)
    D_plus = _D_plus(S0, z, T, r, delta, sigma)
    return S0 + P_z + D_plus


# -----------------------------------------------------------------------------
# Reverse-strike lookback call: Vc = m(x, y, τ) - m(x, min(y, k), τ). Knock-out if y < k.
# -----------------------------------------------------------------------------

def reverse_strike_lookback_call(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    delta: float = 0.0,
    L: Optional[float] = None,
) -> float:
    """
    European reverse-strike lookback call. Payoff max(Y_T - K, 0).
    If current minimum L < K, option is guaranteed worthless (return 0).
    L: current realized minimum from inception to today; if None, pricing at inception (L = S0).
    delta: continuous dividend yield δ (optional, default 0).
    """
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive")
    if L is not None and L < K:
        return 0.0
    if T <= 0:
        cur_min = S0 if L is None else L
        return max(cur_min - K, 0.0)
    y = S0 if L is None else L
    m_y = _generic_min_value(S0, y, T, r, delta, sigma)
    m_min_y_k = _generic_min_value(S0, min(y, K), T, r, delta, sigma)
    return m_y - m_min_y_k


# -----------------------------------------------------------------------------
# Reverse-strike lookback put: Vp = M(x, max(z, k), τ) - M(x, z, τ). Knock-out if z > k.
# -----------------------------------------------------------------------------

def reverse_strike_lookback_put(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    delta: float = 0.0,
    L: Optional[float] = None,
) -> float:
    """
    European reverse-strike lookback put. Payoff max(K - Z_T, 0).
    If current maximum L > K, option is guaranteed worthless (return 0).
    Vp = M(x, max(z, K), τ) - M(x, z, τ): long generic at strike K, short generic at z replicates max(K - Z_T, 0).
    L: current realized maximum from inception to today; if None, pricing at inception (L = S0).
    delta: continuous dividend yield δ (optional, default 0).
    """
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive")
    if L is not None and L > K:
        return 0.0
    if T <= 0:
        cur_max = S0 if L is None else L
        return max(K - cur_max, 0.0)
    z = S0 if L is None else L
    # Vp = M(max(z,k)) - M(z): long generic at K, short at z replicates max(K - Z_T, 0).
    M_z = _generic_max_value(S0, z, T, r, delta, sigma)
    M_max_z_k = _generic_max_value(S0, max(z, K), T, r, delta, sigma)
    return M_max_z_k - M_z


def reverse_strike_lookback_call_greeks(
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
    """Finite-difference Delta/Gamma/Vega for reverse_strike_lookback_call."""
    pricer = lambda spot, vol: reverse_strike_lookback_call(spot, K, T, r, vol, delta=delta, L=L)
    return _finite_difference_greeks(pricer, S0, sigma, dS=dS, dSigma=dSigma)


def reverse_strike_lookback_put_greeks(
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
    """Finite-difference Delta/Gamma/Vega for reverse_strike_lookback_put."""
    pricer = lambda spot, vol: reverse_strike_lookback_put(spot, K, T, r, vol, delta=delta, L=L)
    return _finite_difference_greeks(pricer, S0, sigma, dS=dS, dSigma=dSigma)


# -----------------------------------------------------------------------------
# Monte Carlo: path simulation and pricers
# -----------------------------------------------------------------------------

def _simulate_paths(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    delta: float,
    n_paths: int,
    n_steps: int,
    rng: Optional[np.random.Generator],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    dt = T / n_steps
    drift = r - delta - 0.5 * sigma**2
    Z = rng.standard_normal((n_steps, n_paths))
    log_S = np.zeros(n_paths)
    S = np.full(n_paths, S0)
    S_min = np.full(n_paths, S0)
    S_max = np.full(n_paths, S0)
    # Discrete monitoring of extrema; closed-form lookback formulas assume continuous monitoring.
    for i in range(n_steps):
        log_S += drift * dt + sigma * sqrt(dt) * Z[i]
        S = S0 * np.exp(log_S)
        S_min = np.minimum(S_min, S)
        S_max = np.maximum(S_max, S)
    return S, S_min, S_max


def reverse_strike_lookback_call_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    delta: float = 0.0,
    n_paths: int = 50_000,
    n_steps: int = 252,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Reverse-strike lookback call by Monte Carlo. Payoff max(S_min - K, 0)."""
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive")
    _, S_min, _ = _simulate_paths(S0, T, r, sigma, delta, n_paths, n_steps, rng)
    payoffs = np.maximum(S_min - K, 0.0)
    return exp(-r * T) * float(np.mean(payoffs))


def reverse_strike_lookback_put_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    delta: float = 0.0,
    n_paths: int = 50_000,
    n_steps: int = 252,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Reverse-strike lookback put by Monte Carlo. Payoff max(K - S_max, 0)."""
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive")
    _, _, S_max = _simulate_paths(S0, T, r, sigma, delta, n_paths, n_steps, rng)
    payoffs = np.maximum(K - S_max, 0.0)
    return exp(-r * T) * float(np.mean(payoffs))


# -----------------------------------------------------------------------------
# Payoffs (for Monte Carlo or backtesting)
# -----------------------------------------------------------------------------

def payoff_reverse_strike_call(Y_T: float, K: float) -> float:
    """Payoff at expiry: max(Y_T - K, 0)."""
    return max(Y_T - K, 0.0)


def payoff_reverse_strike_put(K: float, Z_T: float) -> float:
    """Payoff at expiry: max(K - Z_T, 0)."""
    return max(K - Z_T, 0.0)


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
    c = reverse_strike_lookback_call(p.S0, p.K, p.T, p.r, p.sigma, delta=p.delta)
    pt = reverse_strike_lookback_put(p.S0, p.K, p.T, p.r, p.sigma, delta=p.delta)
    print("Reverse-strike lookback (from article formulas)")
    print(f"  S0={p.S0}, K={p.K}, T={p.T}, r={p.r}, σ={p.sigma}, δ={p.delta}")
    print(f"  Call (value): {c:.4f}")
    print(f"  Put (value):  {pt:.4f}")
    # With non-zero dividend
    p_div = Params(delta=0.02)
    c_div = reverse_strike_lookback_call(p_div.S0, p_div.K, p_div.T, p_div.r, p_div.sigma, delta=p_div.delta)
    pt_div = reverse_strike_lookback_put(p_div.S0, p_div.K, p_div.T, p_div.r, p_div.sigma, delta=p_div.delta)
    print(f"  Call (δ=0.02): {c_div:.4f}")
    print(f"  Put (δ=0.02):  {pt_div:.4f}")
    # With optional current extremum
    c_L = reverse_strike_lookback_call(p.S0, p.K, p.T, p.r, p.sigma, delta=p.delta, L=95.0)
    pt_L = reverse_strike_lookback_put(p.S0, p.K, p.T, p.r, p.sigma, delta=p.delta, L=105.0)
    print(f"  Call (L_min=95):  {c_L:.4f}")
    print(f"  Put (L_max=105):  {pt_L:.4f}")
    # Knock-out examples
    c_knock = reverse_strike_lookback_call(p.S0, p.K, p.T, p.r, p.sigma, delta=p.delta, L=90.0)
    pt_knock = reverse_strike_lookback_put(p.S0, p.K, p.T, p.r, p.sigma, delta=p.delta, L=110.0)
    print(f"  Call (L=90 < K, knocked out): {c_knock:.4f}")
    print(f"  Put (L=110 > K, knocked out):  {pt_knock:.4f}")
    # Payoff examples
    print(f"  Payoff call (Y_T=102, K=100): {payoff_reverse_strike_call(102.0, p.K):.4f}")
    print(f"  Payoff put (K=100, Z_T=96):   {payoff_reverse_strike_put(p.K, 96.0):.4f}")
    # Monte Carlo comparison
    rng = np.random.default_rng(42)
    c_mc = reverse_strike_lookback_call_mc(p.S0, p.K, p.T, p.r, p.sigma, delta=p.delta, n_paths=25_000, rng=rng)
    pt_mc = reverse_strike_lookback_put_mc(p.S0, p.K, p.T, p.r, p.sigma, delta=p.delta, n_paths=25_000, rng=rng)
    print("  Monte Carlo (n_paths=25_000):")
    print(f"    Call: {c_mc:.4f}")
    print(f"    Put:  {pt_mc:.4f}")
