# Documentation index

This repository’s main narrative, quickstart, and structure live in the root [`README.md`](../README.md).

## API design — limited-risk Greeks

[`Limited-Risk Lookback/limited_risk_lookback.py`](../Limited-Risk%20Lookback/limited_risk_lookback.py) exposes:

- `limited_risk_lookback_call_greeks(..., dS=..., dSigma=..., *, include_gamma=True, include_vega=True)`
- `limited_risk_lookback_put_greeks(..., dS=..., dSigma=..., *, include_gamma=True, include_vega=True)`

`include_gamma` and `include_vega` are keyword-only. When `False`, that Greek is not computed and is returned as `nan`, and the corresponding pricer evaluations are skipped (fewer calls for large batches). Defaults match the previous behavior: full central-difference Δ, Γ, and 𝒱 (five pricer evaluations).

Closed-form pricing uses `math` for `exp` / `log` / `sqrt`; optional SciPy accelerates `norm_cdf` only.

## Implementation notes

Recent internal refactors keep repeated work local to each product script rather than sharing helpers across product families.

- `Partial Lookback/partial_lookback.py` now precomputes shared partial-time bivariate terms once per call path and reuses them inside `_A_bivariate`, `_B_bivariate`, and `_Q_bivariate`. Its partial-time Monte Carlo call and put pricers also share one internal monitored-extremum simulator.
- `Fixed-Strike Lookback/fixed_strike_lookback.py` and `Quanto Lookback/quanto_lookback.py` now precompute `d1`/`d2`/`d3` and discount factors once per pricer call instead of rebuilding the same distances multiple times.
- `Floating-Strike Lookback/floating_strike_lookback.py`, `Limited-Risk Lookback/limited_risk_lookback.py`, and `Reverse-Strike Lookback/reverse_strike_lookback.py` now use local per-script setup helpers for repeated pricing state while keeping their public APIs unchanged.
