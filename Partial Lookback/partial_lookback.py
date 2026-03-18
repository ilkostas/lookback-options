"""
European Partial Lookback Options — partial-price and partial-time, closed-form and Monte Carlo.

Partial-price (multiplier on extremum):
- Call: payoff max(S_T - λ Y_T, 0), λ ≥ 1. Price = C_{λy}(x,τ) + λ^{β+2} D^-_{y/λ}(x,τ).
- Put:  payoff max(μ Z_T - S_T, 0), μ ≤ 1. Price = P_{μz}(x,τ) + μ^{β+2} D^+_{z/μ}(x,τ).
  (Put uses μ^{β+2} so the strike multiplier μ also scales the lookback premium.)

Partial-time (sub-window [0,T_1], expiry T_2):
- Call: min over [0,T_1], payoff max(S_{T_2} - Y_{T_1}, 0). Buchen–Konstandatos (2005) closed form.
- Put:  max over [0,T_1], payoff max(Z_{T_1} - S_{T_2}, 0).

Replication uses Method of Images; α = σ²/(2r), β = 2r/σ² - 1. Dividend: δ=0 only for closed-form; for δ≠0 use Monte Carlo.
Near b = r - δ = 0, closed-form paths use dedicated finite-limit handling on full prices instead of heuristic α/β substitution.
BGK continuity correction is intentionally not wrapped here: partial-time/partial-price products
require product-specific discrete-to-continuous handling.
"""

from math import log, sqrt, exp, erf, pi
from typing import Callable, Optional
from dataclasses import dataclass

import numpy as np


def norm_cdf(x: float) -> float:
    """Cumulative distribution function of the standard normal N(d)."""
    try:
        from scipy.stats import norm
        return float(norm.cdf(x))
    except ImportError:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))


_DREZNER_GL_NODES, _DREZNER_GL_WEIGHTS = np.polynomial.legendre.leggauss(20)
_DREZNER_RHO_EPS = 1e-12
_DREZNER_EXP_CUTOFF = -80.0
_B_ZERO_EPS = 1e-8


@dataclass
class OptionGreeks:
    delta: float
    gamma: float
    vega: float


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


def _sgn(x: float) -> float:
    return 1.0 if x >= 0 else -1.0


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, x))


def _rho_reduced(h: float, k: float, rho: float) -> float:
    denom_sq = h * h - 2.0 * rho * h * k + k * k
    if denom_sq <= 0.0:
        return 0.0
    reduced = ((rho * h - k) * _sgn(h)) / sqrt(denom_sq)
    return max(-1.0 + _DREZNER_RHO_EPS, min(1.0 - _DREZNER_RHO_EPS, reduced))


def _delta_hk(h: float, k: float) -> float:
    return (1.0 - _sgn(h) * _sgn(k)) / 4.0


def _norm_bivariate_cdf_drezner_core(h: float, k: float, rho: float) -> float:
    """
    Gauss-quadrature core for the rho-integral representation of the bivariate normal CDF.
    """
    base = norm_cdf(h) * norm_cdf(k)
    if abs(rho) < 1e-15:
        return _clip01(base)
    a = 0.0
    b = rho
    half = 0.5 * (b - a)
    mid = 0.5 * (b + a)
    acc = 0.0
    for xi, wi in zip(_DREZNER_GL_NODES, _DREZNER_GL_WEIGHTS):
        t = half * float(xi) + mid
        one_minus_t2 = max(1e-16, 1.0 - t * t)
        expo = -(h * h - 2.0 * t * h * k + k * k) / (2.0 * one_minus_t2)
        if expo > _DREZNER_EXP_CUTOFF:
            acc += float(wi) * exp(expo) / sqrt(one_minus_t2)
    integral = (half * acc) / (2.0 * pi)
    return _clip01(base + integral)


def _norm_bivariate_cdf_drezner(h: float, k: float, rho: float) -> float:
    """
    Drezner (1978) routing:
    - Evaluate quadrature directly only on h <= 0, k <= 0, rho <= 0.
    - Use symmetry/decomposition elsewhere for numerical stability.
    """
    if rho >= 1.0 - _DREZNER_RHO_EPS:
        return _clip01(norm_cdf(min(h, k)))
    if rho <= -1.0 + _DREZNER_RHO_EPS:
        return _clip01(max(0.0, norm_cdf(h) - norm_cdf(-k)))
    rho = max(-1.0 + _DREZNER_RHO_EPS, min(1.0 - _DREZNER_RHO_EPS, rho))

    if abs(rho) < 1e-15:
        return _clip01(norm_cdf(h) * norm_cdf(k))

    if h * k * rho <= 0.0:
        if h <= 0.0 and k <= 0.0 and rho <= 0.0:
            return _norm_bivariate_cdf_drezner_core(h, k, rho)
        if h <= 0.0 and k >= 0.0 and rho >= 0.0:
            return _clip01(norm_cdf(h) - _norm_bivariate_cdf_drezner(h, -k, -rho))
        if h >= 0.0 and k <= 0.0 and rho >= 0.0:
            return _clip01(norm_cdf(k) - _norm_bivariate_cdf_drezner(-h, k, -rho))
        if h >= 0.0 and k >= 0.0 and rho <= 0.0:
            return _clip01(norm_cdf(h) + norm_cdf(k) - 1.0 + _norm_bivariate_cdf_drezner(-h, -k, rho))

    rho_hk = _rho_reduced(h, k, rho)
    rho_kh = _rho_reduced(k, h, rho)
    val = (
        _norm_bivariate_cdf_drezner(h, 0.0, rho_hk)
        + _norm_bivariate_cdf_drezner(k, 0.0, rho_kh)
        - _delta_hk(h, k)
    )
    return _clip01(val)


def norm_bivariate_cdf(a: float, b: float, rho: float) -> float:
    """Bivariate standard normal CDF N_2(a, b; rho) = P(Z1 <= a, Z2 <= b) with corr(Z1,Z2)=rho."""
    try:
        from scipy.stats import multivariate_normal
        rho = max(-1 + 1e-10, min(1 - 1e-10, rho))
        return float(multivariate_normal.cdf([a, b], mean=[0, 0], cov=[[1, rho], [rho, 1]]))
    except Exception:
        pass
    return _norm_bivariate_cdf_drezner(a, b, rho)


# -----------------------------------------------------------------------------
# European vanilla call/put (for partial-price and gap in partial-time)
# -----------------------------------------------------------------------------

def _d1(S: float, K: float, T: float, r: float, delta: float, sigma: float) -> float:
    return (log(S / K) + (r - delta + 0.5 * sigma**2) * T) / (sigma * sqrt(T))


def _d2(S: float, K: float, T: float, r: float, delta: float, sigma: float) -> float:
    return _d1(S, K, T, r, delta, sigma) - sigma * sqrt(T)


def _european_call(S0: float, K: float, T: float, r: float, sigma: float, delta: float = 0.0) -> float:
    if T <= 0:
        return max(S0 - K, 0.0)
    d1_val = _d1(S0, K, T, r, delta, sigma)
    d2_val = _d2(S0, K, T, r, delta, sigma)
    return S0 * exp(-delta * T) * norm_cdf(d1_val) - K * exp(-r * T) * norm_cdf(d2_val)


def _european_put(S0: float, K: float, T: float, r: float, sigma: float, delta: float = 0.0) -> float:
    if T <= 0:
        return max(K - S0, 0.0)
    d1_val = _d1(S0, K, T, r, delta, sigma)
    d2_val = _d2(S0, K, T, r, delta, sigma)
    return K * exp(-r * T) * norm_cdf(-d2_val) - S0 * exp(-delta * T) * norm_cdf(-d1_val)


# -----------------------------------------------------------------------------
# Partial-price: binary building blocks (δ=0 in replication)
# d_ξ = [ln(x/ξ) + (r + σ²/2)τ]/(σ√τ), d'_ξ = d_ξ - σ√τ, bar{d}'_ξ = [ln(ξ/x) + (r - σ²/2)τ]/(σ√τ)
# A^-_ξ = x*N(-d_ξ), B^+_ξ = e^{-rτ}N(d'_ξ), *B^+_ξ = (ξ/x)^β e^{-rτ}N(bar{d}'_ξ)
# D^-_ξ = -α [ A^-_ξ - ξ * *B^+_ξ ]
# A^+_ξ = x*N(d_ξ), *B^-_ξ = (x/ξ)^β e^{-rτ}N(bar{d}'_ξ)  [image for max]
# D^+_ξ = α [ A^+_ξ - ξ * *B^-_ξ ]
# -----------------------------------------------------------------------------

def _alpha_beta(r: float, delta: float, sigma: float) -> tuple[float, float]:
    """
    α = σ²/(2b), β = 2b/σ² - 1 with cost-of-carry b = r - δ.

    This helper is only valid away from b=0. The b≈0 limit must be handled
    by dedicated function-level branches on the full pricing expression.
    """
    b = r - delta
    if abs(b) < _B_ZERO_EPS:
        raise ValueError("alpha/beta undefined near b=0; use dedicated b≈0 limit branch")
    alpha = sigma**2 / (2 * b)
    beta = (2 * b / sigma**2) - 1
    return alpha, beta


def _d_xi(x: float, xi: float, tau: float, r: float, delta: float, sigma: float) -> float:
    """Distance d_ξ with cost-of-carry b = r - δ."""
    b = r - delta
    return (log(x / xi) + (b + 0.5 * sigma**2) * tau) / (sigma * sqrt(tau))


def _d_prime_xi(x: float, xi: float, tau: float, r: float, delta: float, sigma: float) -> float:
    """d'_ξ = d_ξ - σ√τ."""
    return _d_xi(x, xi, tau, r, delta, sigma) - sigma * sqrt(tau)


def _bar_d_prime_xi(x: float, xi: float, tau: float, r: float, delta: float, sigma: float) -> float:
    """Reflected distance bar{d}'_ξ with b = r - δ."""
    b = r - delta
    return (log(xi / x) + (b - 0.5 * sigma**2) * tau) / (sigma * sqrt(tau))


def _A_minus(x: float, xi: float, tau: float, r: float, delta: float, sigma: float) -> float:
    """A^-_ξ = x * e^{-δτ} * N(-d_ξ)."""
    return x * exp(-delta * tau) * norm_cdf(-_d_xi(x, xi, tau, r, delta, sigma))


def _B_plus(x: float, xi: float, tau: float, r: float, delta: float, sigma: float) -> float:
    """B^+_ξ = e^{-rτ} * N(d'_ξ)."""
    return exp(-r * tau) * norm_cdf(_d_prime_xi(x, xi, tau, r, delta, sigma))


def _B_plus_image(x: float, xi: float, tau: float, r: float, delta: float, sigma: float, beta: float) -> float:
    """*B^+_ξ = (ξ/x)^β e^{-rτ} * N(bar{d}'_ξ)."""
    if x <= 0 or xi <= 0:
        return 0.0
    return exp(beta * log(xi / x)) * exp(-r * tau) * norm_cdf(_bar_d_prime_xi(x, xi, tau, r, delta, sigma))


def _D_minus(x: float, xi: float, tau: float, r: float, delta: float, sigma: float, alpha: float, beta: float) -> float:
    """D^-_ξ = -α [ A^-_ξ - ξ * *B^+_ξ ]."""
    return -alpha * (_A_minus(x, xi, tau, r, delta, sigma) - xi * _B_plus_image(x, xi, tau, r, delta, sigma, beta))


def _A_plus(x: float, xi: float, tau: float, r: float, delta: float, sigma: float) -> float:
    """A^+_ξ = x * e^{-δτ} * N(d_ξ)."""
    return x * exp(-delta * tau) * norm_cdf(_d_xi(x, xi, tau, r, delta, sigma))


def _B_minus_image(x: float, xi: float, tau: float, r: float, delta: float, sigma: float, beta: float) -> float:
    """*B^-_ξ = (ξ/x)^β e^{-rτ} * N(-bar{d}'_ξ) for maximum (reflected bond binary)."""
    if x <= 0 or xi <= 0:
        return 0.0
    return exp(beta * log(xi / x)) * exp(-r * tau) * norm_cdf(-_bar_d_prime_xi(x, xi, tau, r, delta, sigma))


def _D_plus(x: float, xi: float, tau: float, r: float, delta: float, sigma: float, alpha: float, beta: float) -> float:
    """D^+_ξ = α [ A^+_ξ - ξ * *B^-_ξ ]."""
    return alpha * (_A_plus(x, xi, tau, r, delta, sigma) - xi * _B_minus_image(x, xi, tau, r, delta, sigma, beta))


def _partial_price_lookback_call_core(
    x: float,
    y_use: float,
    tau: float,
    r: float,
    sigma: float,
    lambda_mult: float,
) -> float:
    """Core BK call formula for b != 0."""
    alpha, beta = _alpha_beta(r, 0.0, sigma)
    K_call = lambda_mult * y_use
    C = _european_call(x, K_call, tau, r, sigma, 0.0)
    xi_fractional = y_use / lambda_mult
    if xi_fractional <= 0:
        return C
    D = _D_minus(x, xi_fractional, tau, r, 0.0, sigma, alpha, beta)
    return C + exp((beta + 2) * log(lambda_mult)) * D


def _partial_price_lookback_put_core(
    x: float,
    z_use: float,
    tau: float,
    r: float,
    sigma: float,
    mu_mult: float,
) -> float:
    """Core BK put formula for b != 0."""
    alpha, beta = _alpha_beta(r, 0.0, sigma)
    K_put = mu_mult * z_use
    P = _european_put(x, K_put, tau, r, sigma, 0.0)
    xi_fractional = z_use / mu_mult
    if xi_fractional <= 0:
        return P
    D = _D_plus(x, xi_fractional, tau, r, 0.0, sigma, alpha, beta)
    return P + exp((beta + 2) * log(mu_mult)) * D


# -----------------------------------------------------------------------------
# Partial-price lookback call: V_c = C_{λy}(x,τ) + λ^{β+2} D^-_{y/λ}(x,τ)
# -----------------------------------------------------------------------------

def partial_price_lookback_call(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    lambda_mult: float,
    delta: float = 0.0,
    y: Optional[float] = None,
) -> float:
    """
    Partial-price lookback call. Payoff max(S_T - λ Y_T, 0), λ ≥ 1.
    y: current realized minimum; if None, pricing at inception (y = S0).
    Closed-form valid only for δ=0 (Buchen-Konstandatos 2005); for δ≠0 use the corresponding _mc pricer.
    """
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive")
    if lambda_mult < 1.0:
        raise ValueError("partial_price call requires λ ≥ 1")
    if abs(delta) > 1e-12:
        raise NotImplementedError(
            "Closed-form partial lookback formulas (Buchen-Konstandatos 2005) support only non-dividend-paying underlyings (delta=0). "
            "For dividend yield != 0, use the Monte Carlo pricers (e.g. partial_price_lookback_call_mc, partial_time_lookback_call_mc)."
        )
    if T <= 0:
        strike = lambda_mult * (y if y is not None else S0)
        return max(S0 - strike, 0.0)
    y_use = S0 if y is None else y
    tau = T
    x = S0
    b = r - delta
    if abs(b) < _B_ZERO_EPS:
        # Evaluate the full finite b->0 limit numerically via symmetric perturbation.
        r_plus = delta + _B_ZERO_EPS
        r_minus = delta - _B_ZERO_EPS
        v_plus = _partial_price_lookback_call_core(x, y_use, tau, r_plus, sigma, lambda_mult)
        v_minus = _partial_price_lookback_call_core(x, y_use, tau, r_minus, sigma, lambda_mult)
        return 0.5 * (v_plus + v_minus)
    return _partial_price_lookback_call_core(x, y_use, tau, r, sigma, lambda_mult)


# -----------------------------------------------------------------------------
# Partial-price lookback put: V_p = P_{μz}(x,τ) + μ^{β+2} D^+_{z/μ}(x,τ)
# -----------------------------------------------------------------------------

def partial_price_lookback_put(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    mu_mult: float,
    delta: float = 0.0,
    z: Optional[float] = None,
) -> float:
    """
    Partial-price lookback put. Payoff max(μ Z_T - S_T, 0), 0 < μ ≤ 1.
    z: current realized maximum; if None, pricing at inception (z = S0).
    Put uses μ^{β+2} to scale the lookback premium by the strike multiplier μ.
    Closed-form valid only for δ=0 (Buchen-Konstandatos 2005); for δ≠0 use the corresponding _mc pricer.
    """
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive")
    if mu_mult <= 0 or mu_mult > 1.0:
        raise ValueError("partial_price put requires 0 < μ ≤ 1")
    if abs(delta) > 1e-12:
        raise NotImplementedError(
            "Closed-form partial lookback formulas (Buchen-Konstandatos 2005) support only non-dividend-paying underlyings (delta=0). "
            "For dividend yield != 0, use the Monte Carlo pricers (e.g. partial_price_lookback_call_mc, partial_time_lookback_call_mc)."
        )
    if T <= 0:
        strike = mu_mult * (z if z is not None else S0)
        return max(strike - S0, 0.0)
    z_use = S0 if z is None else z
    tau = T
    x = S0
    b = r - delta
    if abs(b) < _B_ZERO_EPS:
        # Evaluate the full finite b->0 limit numerically via symmetric perturbation.
        r_plus = delta + _B_ZERO_EPS
        r_minus = delta - _B_ZERO_EPS
        v_plus = _partial_price_lookback_put_core(x, z_use, tau, r_plus, sigma, mu_mult)
        v_minus = _partial_price_lookback_put_core(x, z_use, tau, r_minus, sigma, mu_mult)
        return 0.5 * (v_plus + v_minus)
    return _partial_price_lookback_put_core(x, z_use, tau, r, sigma, mu_mult)


# -----------------------------------------------------------------------------
# Partial-time: Buchen–Konstandatos (2005) building blocks
# τ_1 = T_1 - t, τ_2 = T_2 - t, τ = T_2 - T_1
# ρ = sqrt(τ_1/τ_2), d_i, d'_i for ξ_i
# A^{s1 s2}_{ξ1 ξ2} = x * N_2(s1*d1, s2*d2; s1*s2*ρ), B^{s1 s2} = e^{-r τ_2} N_2(s1*d'1, s2*d'2; s1*s2*ρ)
# Q^{s1 s2}_{k1 k2} = A^{s1 s2}_{k1 k2} - k2 * B^{s1 s2}_{k1 k2}
# D^{ss}_{ξξ} = (σ²/(2r)) * [ -A^{ss}_{ξξ} + ξ * B_image^{opposite}_{ξξ} ]
# k(τ) = g(τ) + α*h(τ), k'(τ) = g'(τ) + α*h'(τ)
# -----------------------------------------------------------------------------

def _rho_partial_time(tau1: float, tau2: float) -> float:
    if tau2 <= 0:
        return 0.0
    return sqrt(tau1 / tau2)


def _d1_d2_partial(x: float, xi1: float, xi2: float, tau1: float, tau2: float, r: float, sigma: float) -> tuple[float, float, float, float]:
    """d1, d'1 for ξ1 and τ1; d2, d'2 for ξ2 and τ2."""
    if tau1 <= 0:
        d1_val = 1e10 if x >= xi1 else -1e10
        d1p = d1_val - sigma * sqrt(tau1) if tau1 > 0 else d1_val
    else:
        d1_val = (log(x / xi1) + (r + 0.5 * sigma**2) * tau1) / (sigma * sqrt(tau1))
        d1p = d1_val - sigma * sqrt(tau1)
    if tau2 <= 0:
        d2_val = 1e10 if x >= xi2 else -1e10
        d2p = d2_val - sigma * sqrt(tau2) if tau2 > 0 else d2_val
    else:
        d2_val = (log(x / xi2) + (r + 0.5 * sigma**2) * tau2) / (sigma * sqrt(tau2))
        d2p = d2_val - sigma * sqrt(tau2)
    return d1_val, d1p, d2_val, d2p


def _A_bivariate(x: float, xi1: float, xi2: float, tau1: float, tau2: float, r: float, sigma: float, s1: int, s2: int) -> float:
    """A^{s1 s2}_{ξ1 ξ2} = x * N_2(s1*d1, s2*d2; s1*s2*ρ)."""
    d1_val, _, d2_val, _ = _d1_d2_partial(x, xi1, xi2, tau1, tau2, r, sigma)
    rho = _rho_partial_time(tau1, tau2)
    rho_signed = s1 * s2 * rho
    return x * norm_bivariate_cdf(s1 * d1_val, s2 * d2_val, rho_signed)


def _B_bivariate(x: float, xi1: float, xi2: float, tau1: float, tau2: float, r: float, sigma: float, s1: int, s2: int) -> float:
    """B^{s1 s2}_{ξ1 ξ2} = e^{-r τ_2} N_2(s1*d'1, s2*d'2; s1*s2*ρ)."""
    _, d1p, _, d2p = _d1_d2_partial(x, xi1, xi2, tau1, tau2, r, sigma)
    rho = _rho_partial_time(tau1, tau2)
    rho_signed = s1 * s2 * rho
    return exp(-r * tau2) * norm_bivariate_cdf(s1 * d1p, s2 * d2p, rho_signed)


def _B_image_bivariate(x: float, xi: float, tau1: float, tau2: float, r: float, sigma: float, beta: float, s1: int, s2: int) -> float:
    """Image of bond binary for second-order D; barrier ξ, same for both horizons. N_2 with reflected d'."""
    bar_d1 = _bar_d_prime_xi(x, xi, tau1, r, 0.0, sigma)
    bar_d2 = _bar_d_prime_xi(x, xi, tau2, r, 0.0, sigma)
    rho = _rho_partial_time(tau1, tau2)
    rho_signed = s1 * s2 * rho
    # B_image = (ξ/x)^β e^{-r τ_2} N_2(bar{d}'1, bar{d}'2; ρ) for appropriate signs
    factor = exp(beta * log(xi / x)) * exp(-r * tau2) if xi > 0 and x > 0 else 0.0
    return factor * norm_bivariate_cdf(s1 * bar_d1, s2 * bar_d2, rho_signed)


def _Q_bivariate(x: float, xi1: float, xi2: float, tau1: float, tau2: float, r: float, sigma: float, s1: int, s2: int) -> float:
    """Q^{s1 s2}_{ξ1 ξ2} = A^{s1 s2}_{ξ1 ξ2} - ξ2 * B^{s1 s2}_{ξ1 ξ2}."""
    A = _A_bivariate(x, xi1, xi2, tau1, tau2, r, sigma, s1, s2)
    B = _B_bivariate(x, xi1, xi2, tau1, tau2, r, sigma, s1, s2)
    return A - xi2 * B


def _D_bivariate(x: float, xi: float, tau1: float, tau2: float, r: float, sigma: float, alpha: float, beta: float, s: int) -> float:
    """D^{ss}_{ξξ} = (σ²/(2r)) * [ -A^{ss}_{ξξ} + ξ * B_image^{opposite}_{ξξ} ]. s=+1 or -1."""
    A = _A_bivariate(x, xi, xi, tau1, tau2, r, sigma, s, s)
    opp = -s
    B_im = _B_image_bivariate(x, xi, tau1, tau2, r, sigma, beta, opp, opp)
    return alpha * (-A + xi * B_im)


def _k_tau_gap(tau: float, r: float, sigma: float, alpha: float) -> float:
    """k(τ) = g(τ) + α*h(τ). a = (r + σ²/2)/σ, a' = (r - σ²/2)/σ."""
    if tau <= 0:
        return 0.0
    a = (r + 0.5 * sigma**2) / sigma
    ap = (r - 0.5 * sigma**2) / sigma
    sqrt_tau = sqrt(tau)
    g = norm_cdf(a * sqrt_tau) - exp(-r * tau) * norm_cdf(ap * sqrt_tau)
    h = norm_cdf(-a * sqrt_tau) - exp(-r * tau) * norm_cdf(ap * sqrt_tau)
    return g + alpha * h


def _k_prime_tau_gap(tau: float, r: float, sigma: float, alpha: float) -> float:
    """k'(τ) = g'(τ) + α*h'(τ)."""
    if tau <= 0:
        return 0.0
    a = (r + 0.5 * sigma**2) / sigma
    ap = (r - 0.5 * sigma**2) / sigma
    sqrt_tau = sqrt(tau)
    gp = -norm_cdf(-a * sqrt_tau) + exp(-r * tau) * norm_cdf(-ap * sqrt_tau)
    hp = -norm_cdf(a * sqrt_tau) + exp(-r * tau) * norm_cdf(-ap * sqrt_tau)
    return gp + alpha * hp


def _A_first_order(x: float, xi: float, tau: float, r: float, sigma: float, sign: int) -> float:
    """A^-_ξ = x*N(-d_ξ), A^+_ξ = x*N(d_ξ). sign: -1 for A^-, +1 for A^+."""
    d = _d_xi(x, xi, tau, r, 0.0, sigma)
    return x * norm_cdf(sign * d)


def _partial_time_lookback_call_core(
    x: float,
    y_use: float,
    tau1: float,
    tau2: float,
    tau_gap: float,
    r: float,
    sigma: float,
) -> float:
    """Core BK call formula for r != 0."""
    alpha = sigma**2 / (2 * r)
    beta = (2 * r / sigma**2) - 1
    Q_pp = _Q_bivariate(x, y_use, y_use, tau1, tau2, r, sigma, 1, 1)
    D_mm = _D_bivariate(x, y_use, tau1, tau2, r, sigma, alpha, beta, -1)
    k_val = _k_tau_gap(tau_gap, r, sigma, alpha)
    A_minus_y = _A_first_order(x, y_use, tau1, r, sigma, -1)
    return Q_pp + D_mm + k_val * A_minus_y


def _partial_time_lookback_put_core(
    x: float,
    z_use: float,
    tau1: float,
    tau2: float,
    tau_gap: float,
    r: float,
    sigma: float,
) -> float:
    """Core BK put formula for r != 0."""
    alpha = sigma**2 / (2 * r)
    beta = (2 * r / sigma**2) - 1
    Q_mm = _Q_bivariate(x, z_use, z_use, tau1, tau2, r, sigma, -1, -1)
    D_pp = _D_bivariate(x, z_use, tau1, tau2, r, sigma, alpha, beta, 1)
    kp_val = _k_prime_tau_gap(tau_gap, r, sigma, alpha)
    A_plus_z = _A_first_order(x, z_use, tau1, r, sigma, 1)
    return -Q_mm + D_pp + kp_val * A_plus_z


# -----------------------------------------------------------------------------
# Partial-time lookback call: Vc = Q^{++}_{yy} + D^{--}_{yy} + k(τ)*A^-_y(x,τ_1)
# -----------------------------------------------------------------------------

def partial_time_lookback_call(
    S0: float,
    T1: float,
    T2: float,
    r: float,
    sigma: float,
    t: float = 0.0,
    y: Optional[float] = None,
    delta: float = 0.0,
) -> float:
    """
    Partial-time lookback call: minimum monitored over [0, T_1], payoff at T_2 is max(S_{T_2} - Y_{T_1}, 0).
    t: current time (0 = inception). y: current minimum over [0, t]; if None, y = S0.
    For t >= T_1 the option becomes a European call with strike Y_{T_1} (pass y = locked-in min).
    Closed-form valid only for δ=0 (Buchen-Konstandatos 2005); for δ≠0 use the corresponding _mc pricer.
    """
    if T1 >= T2 or T1 < t:
        raise ValueError("Require 0 <= t < T_1 < T_2")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive")
    if abs(delta) > 1e-12:
        raise NotImplementedError(
            "Closed-form partial lookback formulas (Buchen-Konstandatos 2005) support only non-dividend-paying underlyings (delta=0). "
            "For dividend yield != 0, use the Monte Carlo pricers (e.g. partial_price_lookback_call_mc, partial_time_lookback_call_mc)."
        )
    tau1 = T1 - t
    tau2 = T2 - t
    tau_gap = T2 - T1
    x = S0
    y_use = S0 if y is None else y
    if abs(r) < _B_ZERO_EPS:
        # Avoid heuristic alpha/beta injection by taking a symmetric limit on the full price.
        v_plus = _partial_time_lookback_call_core(x, y_use, tau1, tau2, tau_gap, _B_ZERO_EPS, sigma)
        v_minus = _partial_time_lookback_call_core(x, y_use, tau1, tau2, tau_gap, -_B_ZERO_EPS, sigma)
        return 0.5 * (v_plus + v_minus)
    return _partial_time_lookback_call_core(x, y_use, tau1, tau2, tau_gap, r, sigma)


def partial_time_lookback_put(
    S0: float,
    T1: float,
    T2: float,
    r: float,
    sigma: float,
    t: float = 0.0,
    z: Optional[float] = None,
    delta: float = 0.0,
) -> float:
    """
    Partial-time lookback put: maximum monitored over [0, T_1], payoff at T_2 is max(Z_{T_1} - S_{T_2}, 0).
    t: current time. z: current maximum over [0, t]; if None, z = S0.
    For t >= T_1 the option becomes a European put with strike Z_{T_1}.
    Uses corrected formula (sign on Q, A^+_z) versus published Equation (4.15) in Buchen-Konstandatos (2005).
    Closed-form valid only for δ=0 (Buchen-Konstandatos 2005); for δ≠0 use the corresponding _mc pricer.
    """
    if T1 >= T2 or T1 < t:
        raise ValueError("Require 0 <= t < T_1 < T_2")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive")
    if abs(delta) > 1e-12:
        raise NotImplementedError(
            "Closed-form partial lookback formulas (Buchen-Konstandatos 2005) support only non-dividend-paying underlyings (delta=0). "
            "For dividend yield != 0, use the Monte Carlo pricers (e.g. partial_price_lookback_call_mc, partial_time_lookback_call_mc)."
        )
    tau1 = T1 - t
    tau2 = T2 - t
    tau_gap = T2 - T1
    x = S0
    z_use = S0 if z is None else z
    if abs(r) < _B_ZERO_EPS:
        # Avoid heuristic alpha/beta injection by taking a symmetric limit on the full price.
        v_plus = _partial_time_lookback_put_core(x, z_use, tau1, tau2, tau_gap, _B_ZERO_EPS, sigma)
        v_minus = _partial_time_lookback_put_core(x, z_use, tau1, tau2, tau_gap, -_B_ZERO_EPS, sigma)
        return 0.5 * (v_plus + v_minus)
    return _partial_time_lookback_put_core(x, z_use, tau1, tau2, tau_gap, r, sigma)


def partial_price_lookback_call_greeks(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    lambda_mult: float,
    delta: float = 0.0,
    dS: Optional[float] = None,
    dSigma: Optional[float] = None,
) -> OptionGreeks:
    """Finite-difference Delta/Gamma/Vega for partial_price_lookback_call."""
    pricer = lambda spot, vol: partial_price_lookback_call(
        spot, T, r, vol, lambda_mult, delta=delta
    )
    return _finite_difference_greeks(pricer, S0, sigma, dS=dS, dSigma=dSigma)


def partial_price_lookback_put_greeks(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    mu_mult: float,
    delta: float = 0.0,
    dS: Optional[float] = None,
    dSigma: Optional[float] = None,
) -> OptionGreeks:
    """Finite-difference Delta/Gamma/Vega for partial_price_lookback_put."""
    pricer = lambda spot, vol: partial_price_lookback_put(
        spot, T, r, vol, mu_mult, delta=delta
    )
    return _finite_difference_greeks(pricer, S0, sigma, dS=dS, dSigma=dSigma)


def partial_time_lookback_call_greeks(
    S0: float,
    T1: float,
    T2: float,
    r: float,
    sigma: float,
    t: float = 0.0,
    y: Optional[float] = None,
    delta: float = 0.0,
    dS: Optional[float] = None,
    dSigma: Optional[float] = None,
) -> OptionGreeks:
    """Finite-difference Delta/Gamma/Vega for partial_time_lookback_call."""
    pricer = lambda spot, vol: partial_time_lookback_call(
        spot, T1, T2, r, vol, t=t, y=y, delta=delta
    )
    return _finite_difference_greeks(pricer, S0, sigma, dS=dS, dSigma=dSigma)


def partial_time_lookback_put_greeks(
    S0: float,
    T1: float,
    T2: float,
    r: float,
    sigma: float,
    t: float = 0.0,
    z: Optional[float] = None,
    delta: float = 0.0,
    dS: Optional[float] = None,
    dSigma: Optional[float] = None,
) -> OptionGreeks:
    """Finite-difference Delta/Gamma/Vega for partial_time_lookback_put."""
    pricer = lambda spot, vol: partial_time_lookback_put(
        spot, T1, T2, r, vol, t=t, z=z, delta=delta
    )
    return _finite_difference_greeks(pricer, S0, sigma, dS=dS, dSigma=dSigma)


def partial_time_lookback_call_after_monitoring(
    S0: float,
    Y_T1: float,
    T2: float,
    t: float,
    r: float,
    sigma: float,
    delta: float = 0.0,
) -> float:
    """When T_1 <= t <= T_2: value = European call with strike K = Y_T1, time to expiry T_2 - t."""
    return _european_call(S0, Y_T1, T2 - t, r, sigma, delta)


def partial_time_lookback_put_after_monitoring(
    S0: float,
    Z_T1: float,
    T2: float,
    t: float,
    r: float,
    sigma: float,
    delta: float = 0.0,
) -> float:
    """When T_1 <= t <= T_2: value = European put with strike K = Z_T1."""
    return _european_put(S0, Z_T1, T2 - t, r, sigma, delta)


# -----------------------------------------------------------------------------
# Monte Carlo: partial-price
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


def partial_price_lookback_call_mc(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    lambda_mult: float,
    delta: float = 0.0,
    n_paths: int = 50_000,
    n_steps: int = 252,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Partial-price lookback call by Monte Carlo."""
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive")
    S_T, S_min, _ = _simulate_paths(S0, T, r, sigma, delta, n_paths, n_steps, rng)
    payoffs = np.maximum(S_T - lambda_mult * S_min, 0.0)
    return exp(-r * T) * float(np.mean(payoffs))


def partial_price_lookback_put_mc(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    mu_mult: float,
    delta: float = 0.0,
    n_paths: int = 50_000,
    n_steps: int = 252,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Partial-price lookback put by Monte Carlo."""
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive")
    S_T, _, S_max = _simulate_paths(S0, T, r, sigma, delta, n_paths, n_steps, rng)
    payoffs = np.maximum(mu_mult * S_max - S_T, 0.0)
    return exp(-r * T) * float(np.mean(payoffs))


def partial_time_lookback_call_mc(
    S0: float,
    T1: float,
    T2: float,
    r: float,
    sigma: float,
    delta: float = 0.0,
    n_paths: int = 50_000,
    n_steps: int = 252,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Partial-time lookback call by Monte Carlo (min over [0,T1], payoff at T2)."""
    if T1 >= T2 or T2 <= 0:
        raise ValueError("Require 0 < T_1 < T_2")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive")
    if rng is None:
        rng = np.random.default_rng()
    dt = (T2 - 0) / n_steps
    steps1 = max(1, int(T1 / dt))
    drift = r - delta - 0.5 * sigma**2
    log_S = np.zeros(n_paths)
    S = np.full(n_paths, S0)
    Y_T1 = np.full(n_paths, S0)
    # Discrete monitoring over [0, T1]; this differs from continuous-monitoring closed-form assumptions.
    for i in range(n_steps):
        t = (i + 1) * dt
        log_S += drift * dt + sigma * sqrt(dt) * rng.standard_normal(n_paths)
        S = S0 * np.exp(log_S)
        if t <= T1:
            Y_T1 = np.minimum(Y_T1, S)
    payoffs = np.maximum(S - Y_T1, 0.0)
    return exp(-r * T2) * float(np.mean(payoffs))


def partial_time_lookback_put_mc(
    S0: float,
    T1: float,
    T2: float,
    r: float,
    sigma: float,
    delta: float = 0.0,
    n_paths: int = 50_000,
    n_steps: int = 252,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Partial-time lookback put by Monte Carlo."""
    if T1 >= T2 or T2 <= 0:
        raise ValueError("Require 0 < T_1 < T_2")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive")
    if rng is None:
        rng = np.random.default_rng()
    dt = (T2 - 0) / n_steps
    drift = r - delta - 0.5 * sigma**2
    log_S = np.zeros(n_paths)
    S = np.full(n_paths, S0)
    Z_T1 = np.full(n_paths, S0)
    # Discrete monitoring over [0, T1]; this differs from continuous-monitoring closed-form assumptions.
    for i in range(n_steps):
        t = (i + 1) * dt
        log_S += drift * dt + sigma * sqrt(dt) * rng.standard_normal(n_paths)
        S = S0 * np.exp(log_S)
        if t <= T1:
            Z_T1 = np.maximum(Z_T1, S)
    payoffs = np.maximum(Z_T1 - S, 0.0)
    return exp(-r * T2) * float(np.mean(payoffs))


# -----------------------------------------------------------------------------
# Payoffs
# -----------------------------------------------------------------------------

def payoff_partial_price_call(S_T: float, Y_T: float, lambda_mult: float) -> float:
    """Payoff at expiry: max(S_T - λ Y_T, 0)."""
    return max(S_T - lambda_mult * Y_T, 0.0)


def payoff_partial_price_put(Z_T: float, S_T: float, mu_mult: float) -> float:
    """Payoff at expiry: max(μ Z_T - S_T, 0)."""
    return max(mu_mult * Z_T - S_T, 0.0)


def payoff_partial_time_call(S_T2: float, Y_T1: float) -> float:
    """Payoff at T_2: max(S_{T_2} - Y_{T_1}, 0)."""
    return max(S_T2 - Y_T1, 0.0)


def payoff_partial_time_put(Z_T1: float, S_T2: float) -> float:
    """Payoff at T_2: max(Z_{T_1} - S_{T_2}, 0)."""
    return max(Z_T1 - S_T2, 0.0)


# -----------------------------------------------------------------------------
# Example / CLI
# -----------------------------------------------------------------------------

@dataclass
class PartialPriceParams:
    S0: float = 100.0
    T: float = 1.0
    r: float = 0.05
    sigma: float = 0.2
    lambda_mult: float = 1.05
    mu_mult: float = 0.95
    delta: float = 0.0


@dataclass
class PartialTimeParams:
    S0: float = 100.0
    T1: float = 0.5
    T2: float = 1.0
    r: float = 0.05
    sigma: float = 0.2
    delta: float = 0.0


if __name__ == "__main__":
    print("Partial lookback (partial-price and partial-time, closed-form + MC)")

    # Partial-price
    pp = PartialPriceParams()
    c = partial_price_lookback_call(pp.S0, pp.T, pp.r, pp.sigma, pp.lambda_mult, delta=pp.delta)
    pt = partial_price_lookback_put(pp.S0, pp.T, pp.r, pp.sigma, pp.mu_mult, delta=pp.delta)
    print(f"  S0={pp.S0}, T={pp.T}, r={pp.r}, σ={pp.sigma}, λ={pp.lambda_mult}, μ={pp.mu_mult}")
    print("  Partial-price (closed form):")
    print(f"    Call: {c:.4f}")
    print(f"    Put:  {pt:.4f}")
    rng = np.random.default_rng(42)
    c_mc = partial_price_lookback_call_mc(pp.S0, pp.T, pp.r, pp.sigma, pp.lambda_mult, n_paths=30_000, rng=rng)
    pt_mc = partial_price_lookback_put_mc(pp.S0, pp.T, pp.r, pp.sigma, pp.mu_mult, n_paths=30_000, rng=rng)
    print("  Partial-price (MC):")
    print(f"    Call: {c_mc:.4f}")
    print(f"    Put:  {pt_mc:.4f}")
    print("  Payoffs:")
    print(f"    Partial-price call (S_T=105, Y_T=92, λ=1.05): {payoff_partial_price_call(105.0, 92.0, 1.05):.4f}")
    print(f"    Partial-price put (Z_T=110, S_T=100, μ=0.95): {payoff_partial_price_put(110.0, 100.0, 0.95):.4f}")

    # Partial-time
    pt_params = PartialTimeParams()
    c_pt = partial_time_lookback_call(pt_params.S0, pt_params.T1, pt_params.T2, pt_params.r, pt_params.sigma)
    pt_pt = partial_time_lookback_put(pt_params.S0, pt_params.T1, pt_params.T2, pt_params.r, pt_params.sigma)
    print("  Partial-time (closed form):")
    print(f"    Call: {c_pt:.4f}")
    print(f"    Put:  {pt_pt:.4f}")
    c_pt_mc = partial_time_lookback_call_mc(pt_params.S0, pt_params.T1, pt_params.T2, pt_params.r, pt_params.sigma, n_paths=30_000, rng=rng)
    pt_pt_mc = partial_time_lookback_put_mc(pt_params.S0, pt_params.T1, pt_params.T2, pt_params.r, pt_params.sigma, n_paths=30_000, rng=rng)
    print("  Partial-time (MC):")
    print(f"    Call: {c_pt_mc:.4f}")
    print(f"    Put:  {pt_pt_mc:.4f}")
    print(f"    Payoff partial-time call (S_T2=102, Y_T1=94): {payoff_partial_time_call(102.0, 94.0):.4f}")
    print(f"    Payoff partial-time put (Z_T1=108, S_T2=98): {payoff_partial_time_put(108.0, 98.0):.4f}")
