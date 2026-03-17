"""
European Quanto Lookback Options — closed-form and Monte Carlo.

From lookback.txt (Quanto Lookback, cross-currency drift adjustment):
- Standard fixed-rate: F_c * max(S_max_T - K, 0) or F_c * max(K - S_min_T, 0). Closed form via
  standard lookback formulas with quanto drift (r_effective = r_foreign - rho*sigma_S*sigma_F) and discount at r_domestic.
- Floating-strike quanto: F_c * (S_T - Y_T) or F_c * (Z_T - S_T). Same closed-form trick.
- Max exchange rate quanto call: F_max_T * max(S_T - K, 0). Monte Carlo only.
- Joint quanto fixed-strike call: max(F_c, F_T) * max(S_max_T - K, 0). Monte Carlo only.
- No L (current extremum); pricing from inception only.
"""

from math import log, sqrt, exp
from dataclasses import dataclass
from typing import Optional

import numpy as np


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
    return exp(-x * x / 2) / sqrt(2 * 3.141592653589793)


# -----------------------------------------------------------------------------
# Auxiliary distance functions (same as lookback.txt; r, delta, sigma)
# -----------------------------------------------------------------------------

def _d1(S: float, K: float, T: float, r: float, delta: float, sigma: float) -> float:
    return (log(S / K) + (r - delta + 0.5 * sigma**2) * T) / (sigma * sqrt(T))


def _d2(S: float, K: float, T: float, r: float, delta: float, sigma: float) -> float:
    return _d1(S, K, T, r, delta, sigma) - sigma * sqrt(T)


def _d3(S: float, K: float, T: float, r: float, delta: float, sigma: float) -> float:
    return _d1(S, K, T, r, delta, sigma) - 2 * (r - delta) * sqrt(T) / sigma


def _quanto_r_effective(r_foreign: float, sigma_asset: float, sigma_fx: float, rho: float) -> float:
    """Effective risk-free rate for asset drift under domestic measure: r_f - rho*sigma_S*sigma_F."""
    return r_foreign - rho * sigma_asset * sigma_fx


# -----------------------------------------------------------------------------
# Standard fixed-rate quanto lookback (closed form, no L)
# Payoff: F_c * max(S_max_T - K, 0) or F_c * max(K - S_min_T, 0).
# Use standard formulas with r_effective, delta = dividend yield, discount at r_domestic, multiply by F_c.
# -----------------------------------------------------------------------------

def _fixed_strike_call_std(S0: float, K: float, T: float, r: float, delta: float, sigma: float) -> float:
    """Standard fixed-strike lookback call with single r (for quanto scaling)."""
    if T <= 0:
        return max(S0 - K, 0.0)
    b = r - delta
    d1_val = _d1(S0, K, T, r, delta, sigma)
    d2_val = _d2(S0, K, T, r, delta, sigma)
    d3_val = _d3(S0, K, T, r, delta, sigma)
    term1 = S0 * exp(-delta * T) * norm_cdf(d1_val)
    term2 = K * exp(-r * T) * norm_cdf(d2_val)
    expo = -2 * b / (sigma**2)
    if abs(b) < 1e-12:
        term3 = 0.5 * (sigma**2) * T * S0 * (norm_cdf(-d3_val) - exp(-r * T) * norm_cdf(-d2_val))
    else:
        term3 = (sigma**2 / (2 * b)) * S0 * (
            (S0 / K) ** expo * norm_cdf(-d3_val) - exp(-r * T) * norm_cdf(-d2_val)
        )
    return term1 - term2 + term3


def quanto_fixed_strike_lookback_call(
    S0: float,
    K: float,
    T: float,
    r_domestic: float,
    r_foreign: float,
    sigma_asset: float,
    sigma_fx: float,
    rho: float,
    quanto_factor: float,
    delta: float = 0.0,
) -> float:
    """
    Standard fixed-rate quanto lookback call. Payoff in domestic: F_c * max(S_max_T - K, 0).
    Pricing from inception only (no L). Closed form: V_std with r_eff, then F_c * V_std * exp((r_eff - r_d)*T).
    """
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative")
    if sigma_asset <= 0 or sigma_fx <= 0:
        raise ValueError("Volatility sigma_asset and sigma_fx must be positive")
    if T <= 0:
        return quanto_factor * max(S0 - K, 0.0)
    r_eff = _quanto_r_effective(r_foreign, sigma_asset, sigma_fx, rho)
    v_std = _fixed_strike_call_std(S0, K, T, r_eff, delta, sigma_asset)
    return quanto_factor * v_std * exp((r_eff - r_domestic) * T)


def _fixed_strike_put_std(S0: float, K: float, T: float, r: float, delta: float, sigma: float) -> float:
    """Standard fixed-strike lookback put with single r (for quanto scaling)."""
    if T <= 0:
        return max(K - S0, 0.0)
    b = r - delta
    d1_val = _d1(S0, K, T, r, delta, sigma)
    d2_val = _d2(S0, K, T, r, delta, sigma)
    d3_val = _d3(S0, K, T, r, delta, sigma)
    term1 = K * exp(-r * T) * norm_cdf(-d2_val)
    term2 = S0 * exp(-delta * T) * norm_cdf(-d1_val)
    b_eff = b if abs(b) >= 1e-12 else (1e-12 if b >= 0 else -1e-12)
    expo = -2 * b_eff / (sigma**2)
    term3 = (sigma**2 / (2 * b_eff)) * (
        -S0 * exp(-delta * T) * norm_cdf(-d1_val)
        + K * exp(-r * T) * (S0 / K) ** expo * norm_cdf(-d3_val)
    )
    return term1 - term2 + term3


def quanto_fixed_strike_lookback_put(
    S0: float,
    K: float,
    T: float,
    r_domestic: float,
    r_foreign: float,
    sigma_asset: float,
    sigma_fx: float,
    rho: float,
    quanto_factor: float,
    delta: float = 0.0,
) -> float:
    """
    Standard fixed-rate quanto lookback put. Payoff in domestic: F_c * max(K - S_min_T, 0).
    Pricing from inception only (no L). Closed form: V_std with r_eff, then F_c * V_std * exp((r_eff - r_d)*T).
    """
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative")
    if sigma_asset <= 0 or sigma_fx <= 0:
        raise ValueError("Volatility sigma_asset and sigma_fx must be positive")
    if T <= 0:
        return quanto_factor * max(K - S0, 0.0)
    r_eff = _quanto_r_effective(r_foreign, sigma_asset, sigma_fx, rho)
    v_std = _fixed_strike_put_std(S0, K, T, r_eff, delta, sigma_asset)
    return quanto_factor * v_std * exp((r_eff - r_domestic) * T)


# -----------------------------------------------------------------------------
# Quanto floating-strike lookback (closed form, no L)
# Payoff: F_c * (S_T - Y_T) or F_c * (Z_T - S_T). L_use = S0 (inception only).
# -----------------------------------------------------------------------------

def _floating_strike_call_std(S0: float, T: float, r: float, delta: float, sigma: float) -> float:
    """Standard floating-strike lookback call with L=S0, single r (for quanto scaling)."""
    if T <= 0:
        return 0.0
    L_use = S0
    b = r - delta
    d1_val = _d1(S0, L_use, T, r, delta, sigma)
    d2_val = _d2(S0, L_use, T, r, delta, sigma)
    d3_val = _d3(S0, L_use, T, r, delta, sigma)
    term1 = S0 * exp(-delta * T) * norm_cdf(d1_val)
    term2 = L_use * exp(-r * T) * norm_cdf(d2_val)
    expo = -2 * b / (sigma**2)
    if abs(b) < 1e-12:
        term3 = 0.5 * (sigma**2) * S0 * exp(-r * T) * (
            T * norm_cdf(-d1_val) + sqrt(T) / sigma * norm_pdf(d1_val)
        )
    else:
        term3 = (sigma**2 / (2 * b)) * S0 * (
            exp(-delta * T) * norm_cdf(-d1_val)
            - exp(-r * T) * (S0 / L_use) ** expo * norm_cdf(-d3_val)
        )
    return term1 - term2 + term3


def quanto_floating_strike_lookback_call(
    S0: float,
    T: float,
    r_domestic: float,
    r_foreign: float,
    sigma_asset: float,
    sigma_fx: float,
    rho: float,
    quanto_factor: float,
    delta: float = 0.0,
) -> float:
    """
    Quanto floating-strike lookback call. Payoff in domestic: F_c * (S_T - Y_T).
    Pricing from inception only (no L). Closed form: V_std with r_eff, then F_c * V_std * exp((r_eff - r_d)*T).
    """
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative")
    if sigma_asset <= 0 or sigma_fx <= 0:
        raise ValueError("Volatility sigma_asset and sigma_fx must be positive")
    if T <= 0:
        return 0.0
    r_eff = _quanto_r_effective(r_foreign, sigma_asset, sigma_fx, rho)
    v_std = _floating_strike_call_std(S0, T, r_eff, delta, sigma_asset)
    return quanto_factor * v_std * exp((r_eff - r_domestic) * T)


def _floating_strike_put_std(S0: float, T: float, r: float, delta: float, sigma: float) -> float:
    """Standard floating-strike lookback put with L=S0, single r (for quanto scaling)."""
    if T <= 0:
        return 0.0
    L_use = S0
    b = r - delta
    d1_val = _d1(S0, L_use, T, r, delta, sigma)
    d2_val = _d2(S0, L_use, T, r, delta, sigma)
    d3_val = _d3(S0, L_use, T, r, delta, sigma)
    term1 = -S0 * exp(-delta * T) * norm_cdf(-d1_val)
    term2 = L_use * exp(-r * T) * norm_cdf(-d2_val)
    b_eff = b if abs(b) >= 1e-12 else (1e-12 if b >= 0 else -1e-12)
    expo = -2 * b_eff / (sigma**2)
    term3 = (sigma**2 / (2 * b_eff)) * S0 * (
        -exp(-delta * T) * norm_cdf(d1_val)
        + exp(-r * T) * (S0 / L_use) ** expo * norm_cdf(d3_val)
    )
    return term1 + term2 + term3


def quanto_floating_strike_lookback_put(
    S0: float,
    T: float,
    r_domestic: float,
    r_foreign: float,
    sigma_asset: float,
    sigma_fx: float,
    rho: float,
    quanto_factor: float,
    delta: float = 0.0,
) -> float:
    """
    Quanto floating-strike lookback put. Payoff in domestic: F_c * (Z_T - S_T).
    Pricing from inception only (no L). Closed form: V_std with r_eff, then F_c * V_std * exp((r_eff - r_d)*T).
    """
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative")
    if sigma_asset <= 0 or sigma_fx <= 0:
        raise ValueError("Volatility sigma_asset and sigma_fx must be positive")
    if T <= 0:
        return 0.0
    r_eff = _quanto_r_effective(r_foreign, sigma_asset, sigma_fx, rho)
    v_std = _floating_strike_put_std(S0, T, r_eff, delta, sigma_asset)
    return quanto_factor * v_std * exp((r_eff - r_domestic) * T)


# -----------------------------------------------------------------------------
# Monte Carlo: max exchange rate quanto call and joint quanto fixed-strike call
# SDEs: drift_S = r_foreign - δ - rho*sigma_asset*sigma_fx, drift_F = r_domestic - r_foreign
# -----------------------------------------------------------------------------

def _simulate_quanto_paths(
    S0: float,
    F0: float,
    T: float,
    r_domestic: float,
    r_foreign: float,
    sigma_asset: float,
    sigma_fx: float,
    rho: float,
    delta: float,
    n_paths: int,
    n_steps: int,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate (S_t, F_t) under domestic risk-neutral measure. Returns S_T, S_max, F_T, F_max per path."""
    if rng is None:
        rng = np.random.default_rng()
    dt = T / n_steps
    drift_S = r_foreign - delta - rho * sigma_asset * sigma_fx
    drift_F = r_domestic - r_foreign
    # Correlated normals: Z_S, Z_F with corr(Z_S, Z_F) = rho
    Z1 = rng.standard_normal((n_steps, n_paths))
    Z2 = rng.standard_normal((n_steps, n_paths))
    Z_S = Z1
    Z_F = rho * Z1 + sqrt(1 - rho**2) * Z2
    log_S = np.zeros(n_paths)
    log_F = np.zeros(n_paths)
    S_max = np.full(n_paths, S0)
    F_max = np.full(n_paths, F0)
    S = np.full(n_paths, S0)
    F = np.full(n_paths, F0)
    for i in range(n_steps):
        dW_S = sqrt(dt) * Z_S[i]
        dW_F = sqrt(dt) * Z_F[i]
        log_S += (drift_S - 0.5 * sigma_asset**2) * dt + sigma_asset * dW_S
        log_F += (drift_F - 0.5 * sigma_fx**2) * dt + sigma_fx * dW_F
        S = S0 * np.exp(log_S)
        F = F0 * np.exp(log_F)
        S_max = np.maximum(S_max, S)
        F_max = np.maximum(F_max, F)
    S_T = S
    F_T = F
    return S_T, S_max, F_T, F_max


def max_exchange_rate_quanto_call_mc(
    S0: float,
    F0: float,
    K: float,
    T: float,
    r_domestic: float,
    r_foreign: float,
    sigma_asset: float,
    sigma_fx: float,
    rho: float,
    delta: float = 0.0,
    n_paths: int = 50_000,
    n_steps: int = 252,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Maximum exchange rate quanto call. Payoff: F_max_T * max(S_T - K, 0). Monte Carlo.
    """
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative")
    if sigma_asset <= 0 or sigma_fx <= 0:
        raise ValueError("Volatility sigma_asset and sigma_fx must be positive")
    S_T, _, F_T, F_max = _simulate_quanto_paths(
        S0, F0, T, r_domestic, r_foreign, sigma_asset, sigma_fx, rho, delta, n_paths, n_steps, rng
    )
    payoffs = F_max * np.maximum(S_T - K, 0.0)
    return exp(-r_domestic * T) * float(np.mean(payoffs))


def joint_quanto_fixed_strike_call_mc(
    S0: float,
    F0: float,
    K: float,
    T: float,
    r_domestic: float,
    r_foreign: float,
    sigma_asset: float,
    sigma_fx: float,
    rho: float,
    fixed_fx_rate: float,
    delta: float = 0.0,
    n_paths: int = 50_000,
    n_steps: int = 252,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Joint quanto fixed-strike lookback call. Payoff: max(F_c, F_T) * max(S_max_T - K, 0). Monte Carlo.
    """
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative")
    if sigma_asset <= 0 or sigma_fx <= 0:
        raise ValueError("Volatility sigma_asset and sigma_fx must be positive")
    S_T, S_max, F_T, _ = _simulate_quanto_paths(
        S0, F0, T, r_domestic, r_foreign, sigma_asset, sigma_fx, rho, delta, n_paths, n_steps, rng
    )
    conversion = np.maximum(fixed_fx_rate, F_T)
    payoffs = conversion * np.maximum(S_max - K, 0.0)
    return exp(-r_domestic * T) * float(np.mean(payoffs))


# -----------------------------------------------------------------------------
# Payoffs (for Monte Carlo or backtesting)
# -----------------------------------------------------------------------------

def payoff_quanto_fixed_strike_call(S_max: float, K: float, quanto_factor: float) -> float:
    """Payoff at expiry (domestic): F_c * max(S_max_T - K, 0)."""
    return quanto_factor * max(S_max - K, 0.0)


def payoff_quanto_fixed_strike_put(S_min: float, K: float, quanto_factor: float) -> float:
    """Payoff at expiry (domestic): F_c * max(K - S_min_T, 0)."""
    return quanto_factor * max(K - S_min, 0.0)


def payoff_quanto_floating_strike_call(S_T: float, Y_T: float, quanto_factor: float) -> float:
    """Payoff at expiry (domestic): F_c * (S_T - Y_T)."""
    return quanto_factor * max(S_T - Y_T, 0.0)


def payoff_quanto_floating_strike_put(Z_T: float, S_T: float, quanto_factor: float) -> float:
    """Payoff at expiry (domestic): F_c * (Z_T - S_T)."""
    return quanto_factor * max(Z_T - S_T, 0.0)


def payoff_max_exchange_rate_quanto_call(F_max: float, S_T: float, K: float) -> float:
    """Payoff at expiry: F_max_T * max(S_T - K, 0)."""
    return F_max * max(S_T - K, 0.0)


def payoff_joint_quanto_fixed_strike_call(
    F_c: float, F_T: float, S_max: float, K: float
) -> float:
    """Payoff at expiry: max(F_c, F_T) * max(S_max_T - K, 0)."""
    return max(F_c, F_T) * max(S_max - K, 0.0)


# -----------------------------------------------------------------------------
# Example / CLI
# -----------------------------------------------------------------------------

@dataclass
class QuantoParams:
    S0: float = 100.0
    F0: float = 1.0
    K: float = 100.0
    T: float = 1.0
    r_domestic: float = 0.05
    r_foreign: float = 0.02
    sigma_asset: float = 0.2
    sigma_fx: float = 0.1
    rho: float = 0.3
    quanto_factor: float = 1.0
    delta: float = 0.0


if __name__ == "__main__":
    p = QuantoParams()
    print("Quanto lookback (from lookback.txt formulas)")
    print(f"  S0={p.S0}, K={p.K}, T={p.T}, r_d={p.r_domestic}, r_f={p.r_foreign}")
    print(f"  sigma_S={p.sigma_asset}, sigma_F={p.sigma_fx}, rho={p.rho}, F_c={p.quanto_factor}, δ={p.delta}")

    # Closed form: standard fixed-rate and floating-strike
    c_fix = quanto_fixed_strike_lookback_call(
        p.S0, p.K, p.T, p.r_domestic, p.r_foreign, p.sigma_asset, p.sigma_fx, p.rho, p.quanto_factor, delta=p.delta
    )
    pt_fix = quanto_fixed_strike_lookback_put(
        p.S0, p.K, p.T, p.r_domestic, p.r_foreign, p.sigma_asset, p.sigma_fx, p.rho, p.quanto_factor, delta=p.delta
    )
    c_float = quanto_floating_strike_lookback_call(
        p.S0, p.T, p.r_domestic, p.r_foreign, p.sigma_asset, p.sigma_fx, p.rho, p.quanto_factor, delta=p.delta
    )
    pt_float = quanto_floating_strike_lookback_put(
        p.S0, p.T, p.r_domestic, p.r_foreign, p.sigma_asset, p.sigma_fx, p.rho, p.quanto_factor, delta=p.delta
    )
    print("  Standard fixed-rate quanto lookback (closed form):")
    print(f"    Call on max: {c_fix:.4f}")
    print(f"    Put on min:  {pt_fix:.4f}")
    print("  Quanto floating-strike (closed form):")
    print(f"    Call: {c_float:.4f}")
    print(f"    Put:  {pt_float:.4f}")

    # Monte Carlo
    rng = np.random.default_rng(42)
    mc_max = max_exchange_rate_quanto_call_mc(
        p.S0, p.F0, p.K, p.T, p.r_domestic, p.r_foreign,
        p.sigma_asset, p.sigma_fx, p.rho, delta=p.delta, n_paths=25_000, rng=rng
    )
    mc_joint = joint_quanto_fixed_strike_call_mc(
        p.S0, p.F0, p.K, p.T, p.r_domestic, p.r_foreign,
        p.sigma_asset, p.sigma_fx, p.rho, p.quanto_factor, delta=p.delta, n_paths=25_000, rng=rng
    )
    print("  Max exchange rate quanto call (MC):", f"{mc_max:.4f}")
    print("  Joint quanto fixed-strike call (MC):", f"{mc_joint:.4f}")

    print("  Payoffs (examples):")
    print(f"    Quanto fixed call (S_max=110, K=100): {payoff_quanto_fixed_strike_call(110.0, p.K, p.quanto_factor):.4f}")
    print(f"    Quanto floating call (S_T=105, Y_T=90): {payoff_quanto_floating_strike_call(105.0, 90.0, p.quanto_factor):.4f}")
