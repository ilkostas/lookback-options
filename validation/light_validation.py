from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def _load_module(relative_path: str, module_name: str) -> ModuleType:
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_finite_greeks(greeks: object, label: str) -> None:
    for attr in ("delta", "gamma", "vega"):
        value = getattr(greeks, attr)
        if not math.isfinite(value):
            raise AssertionError(f"{label} has non-finite {attr}: {value}")


def main() -> None:
    floating = _load_module("Floating-Strike Lookback/floating_strike_lookback.py", "floating")
    fixed = _load_module("Fixed-Strike Lookback/fixed_strike_lookback.py", "fixed")
    limited = _load_module("Limited-Risk Lookback/limited_risk_lookback.py", "limited")
    partial = _load_module("Partial Lookback/partial_lookback.py", "partial")
    quanto = _load_module("Quanto Lookback/quanto_lookback.py", "quanto")
    reverse = _load_module("Reverse-Strike Lookback/reverse_strike_lookback.py", "reverse")

    # BGK convergence sanity: corrected discrete proxy should approach continuous as m grows.
    params_float = dict(S0=100.0, T=1.0, r=0.05, sigma=0.2, delta=0.01)
    v_cont_call = floating.floating_strike_lookback_call(**params_float)
    v_bgk_50 = floating.floating_strike_lookback_call_bgk(**params_float, monitoring_points=50)
    v_bgk_5000 = floating.floating_strike_lookback_call_bgk(**params_float, monitoring_points=5000)
    assert abs(v_bgk_5000 - v_cont_call) < abs(v_bgk_50 - v_cont_call)
    assert abs(v_bgk_5000 - v_cont_call) / max(1.0, abs(v_cont_call)) < 0.08

    v_cont_put = floating.floating_strike_lookback_put(**params_float)
    v_bgk_put_50 = floating.floating_strike_lookback_put_bgk(**params_float, monitoring_points=50)
    v_bgk_put_5000 = floating.floating_strike_lookback_put_bgk(**params_float, monitoring_points=5000)
    assert abs(v_bgk_put_5000 - v_cont_put) < abs(v_bgk_put_50 - v_cont_put)
    assert abs(v_bgk_put_5000 - v_cont_put) / max(1.0, abs(v_cont_put)) < 0.03

    params_fixed = dict(S0=100.0, K=95.0, T=1.0, r=0.05, sigma=0.2, delta=0.01)
    v_fixed_call = fixed.fixed_strike_lookback_call(**params_fixed)
    v_fixed_call_bgk_50 = fixed.fixed_strike_lookback_call_bgk(**params_fixed, monitoring_points=50)
    v_fixed_call_bgk_5000 = fixed.fixed_strike_lookback_call_bgk(**params_fixed, monitoring_points=5000)
    assert abs(v_fixed_call_bgk_5000 - v_fixed_call) < 1e-8
    assert abs(v_fixed_call_bgk_50 - v_fixed_call) < 1e-8

    v_fixed_put = fixed.fixed_strike_lookback_put(**params_fixed)
    v_fixed_put_bgk_50 = fixed.fixed_strike_lookback_put_bgk(**params_fixed, monitoring_points=50)
    v_fixed_put_bgk_5000 = fixed.fixed_strike_lookback_put_bgk(**params_fixed, monitoring_points=5000)
    assert abs(v_fixed_put_bgk_5000 - v_fixed_put) < 1e-8
    assert abs(v_fixed_put_bgk_50 - v_fixed_put) < 1e-8

    # Greeks smoke checks across modules.
    _assert_finite_greeks(
        floating.floating_strike_lookback_call_greeks(100.0, 1.0, 0.05, 0.2, delta=0.01),
        "floating call greeks",
    )
    _assert_finite_greeks(
        floating.floating_strike_lookback_put_greeks(100.0, 1.0, 0.05, 0.2, delta=0.01),
        "floating put greeks",
    )
    _assert_finite_greeks(
        fixed.fixed_strike_lookback_call_greeks(100.0, 95.0, 1.0, 0.05, 0.2, delta=0.01),
        "fixed call greeks",
    )
    _assert_finite_greeks(
        fixed.fixed_strike_lookback_put_greeks(100.0, 95.0, 1.0, 0.05, 0.2, delta=0.01),
        "fixed put greeks",
    )
    _assert_finite_greeks(
        limited.limited_risk_lookback_call_greeks(100.0, 95.0, 1.0, 0.05, 0.2, 120.0, delta=0.01),
        "limited call greeks",
    )
    _assert_finite_greeks(
        limited.limited_risk_lookback_put_greeks(100.0, 95.0, 1.0, 0.05, 0.2, 80.0, delta=0.01),
        "limited put greeks",
    )
    _assert_finite_greeks(
        partial.partial_price_lookback_call_greeks(100.0, 1.0, 0.05, 0.2, 1.05),
        "partial price call greeks",
    )
    _assert_finite_greeks(
        partial.partial_price_lookback_put_greeks(100.0, 1.0, 0.05, 0.2, 0.95),
        "partial price put greeks",
    )
    _assert_finite_greeks(
        partial.partial_time_lookback_call_greeks(100.0, 0.5, 1.0, 0.05, 0.2),
        "partial time call greeks",
    )
    _assert_finite_greeks(
        partial.partial_time_lookback_put_greeks(100.0, 0.5, 1.0, 0.05, 0.2),
        "partial time put greeks",
    )
    _assert_finite_greeks(
        reverse.reverse_strike_lookback_call_greeks(100.0, 95.0, 1.0, 0.05, 0.2, delta=0.01),
        "reverse call greeks",
    )
    _assert_finite_greeks(
        reverse.reverse_strike_lookback_put_greeks(100.0, 105.0, 1.0, 0.05, 0.2, delta=0.01),
        "reverse put greeks",
    )
    _assert_finite_greeks(
        quanto.quanto_fixed_strike_lookback_call_greeks(
            100.0, 95.0, 1.0, 0.05, 0.02, 0.2, 0.1, 0.3, 1.0, delta=0.01
        ),
        "quanto fixed call greeks",
    )
    _assert_finite_greeks(
        quanto.quanto_fixed_strike_lookback_put_greeks(
            100.0, 95.0, 1.0, 0.05, 0.02, 0.2, 0.1, 0.3, 1.0, delta=0.01
        ),
        "quanto fixed put greeks",
    )
    _assert_finite_greeks(
        quanto.quanto_floating_strike_lookback_call_greeks(
            100.0, 1.0, 0.05, 0.02, 0.2, 0.1, 0.3, 1.0, delta=0.01
        ),
        "quanto floating call greeks",
    )
    _assert_finite_greeks(
        quanto.quanto_floating_strike_lookback_put_greeks(
            100.0, 1.0, 0.05, 0.02, 0.2, 0.1, 0.3, 1.0, delta=0.01
        ),
        "quanto floating put greeks",
    )

    print("Light validation passed.")


if __name__ == "__main__":
    main()
