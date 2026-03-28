# Lookback Options

Quantitative finance codebase for pricing multiple families of European lookback options under Black-Scholes dynamics, with a mix of closed-form valuation, Monte Carlo engines, finite-difference Greeks, and discrete-monitoring adjustments where the literature supports them.

This repository is designed to demonstrate breadth across path-dependent exotics and depth in numerical implementation: reflected-path pricing formulas, method-of-images constructions, custom bivariate normal integration, quanto drift adjustment, validation harnesses, and explicit edge-case handling for difficult parameter regimes.

## Highlights

- Multiple lookback families implemented in a single codebase: floating-strike, fixed-strike, reverse-strike, limited-risk, partial, and quanto.
- Closed-form coverage where the product admits tractable pricing, with Monte Carlo engines added for structures that require path simulation.
- Dividend-aware pricing throughout the standard lookback families, plus in-progress extremum support in the modules that expose current running minima or maxima.
- Broadie-Glasserman-Kou continuity correction for discrete monitoring on the classical floating-strike and fixed-strike contracts.
- Central-difference Delta, Gamma, and Vega across product families, with performance-minded optional skips in the limited-risk Greek routines.
- Lightweight validation that checks BGK behavior and finite Greek outputs across the implemented surface.

## Product Coverage

| Family | Core payoffs | Implementation |
|---|---|---|
| `Floating-Strike Lookback` | Call: \(S_T - Y_T\), Put: \(Z_T - S_T\) | Closed form, BGK correction, finite-difference Greeks |
| `Fixed-Strike Lookback` | Call: \(\max(Z_T - K, 0)\), Put: \(\max(K - Y_T, 0)\) | Closed form, BGK correction, finite-difference Greeks |
| `Reverse-Strike Lookback` | Call: \(\max(Y_T - K, 0)\), Put: \(\max(K - Z_T, 0)\) | Closed form, Monte Carlo, finite-difference Greeks |
| `Limited-Risk Lookback` | Up-and-out call and down-and-out put variants linked to running extrema | Closed form, finite-difference Greeks |
| `Partial Lookback` | Partial-price and partial-time structures | Closed form, Monte Carlo, finite-difference Greeks |
| `Quanto Lookback` | Fixed-rate quanto, floating-strike quanto, max-FX quanto, joint quanto | Closed form where available, Monte Carlo for joint/path-dependent FX variants, finite-difference Greeks |

Notation:

- \(Y_T = \min_{0 \le t \le T} S_t\)
- \(Z_T = \max_{0 \le t \le T} S_t\)
- \(K\) is the fixed strike
- \(\lambda \ge 1\) and \(\mu \le 1\) are the partial-price multipliers
- \(T_1 < T_2\) defines the partial-time monitoring window
- \(F_c\) is the contractual quanto conversion factor

## Why This Repository Stands Out

Lookback options are not a single formula copied across variants. Each family changes the role of the path extremum, the strike convention, the monitoring window, or the currency layer, which in turn changes the mathematics and the numerical treatment. This repository reflects that reality directly.

- Standard fixed-strike and floating-strike contracts are implemented with analytical formulas under GBM and extended with BGK-style discrete-monitoring adjustment.
- Reverse-strike and partial structures go beyond the most common textbook examples and incorporate method-of-images style decomposition and simulation support.
- Partial-time pricing includes a dedicated bivariate normal CDF treatment rather than treating the correlation structure as a black box.
- Quanto pricing captures the cross-currency measure change through a dedicated effective-rate adjustment and correlated Monte Carlo for products that require joint simulation of asset and FX paths.
- Near-singular cases such as \(b = r - \delta \to 0\) are handled explicitly rather than left to unstable generic expressions.

## Mathematical And Numerical Design

### Core Model

Closed-form pricing is built on Black-Scholes dynamics under the risk-neutral measure:

$$
dS_t = (r - \delta) S_t dt + \sigma S_t dW_t
$$

with constant rate \(r\), continuous dividend yield \(\delta\), and volatility \(\sigma\).

Across the standard lookback modules, pricing relies on the familiar auxiliary distances

$$
d_{1,K} = \frac{\ln(S_0/K) + (r - \delta + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}},
\qquad
d_{2,K} = d_{1,K} - \sigma\sqrt{T},
\qquad
d_{3,K} = d_{1,K} - \frac{2(r-\delta)\sqrt{T}}{\sigma}
$$

where the reflected-path term \(d_{3,K}\) captures the extremum contribution that distinguishes lookbacks from vanilla options.

### Pricing Architecture

- `Floating-Strike Lookback` and `Fixed-Strike Lookback` implement continuous-monitoring closed forms plus BGK wrappers for discrete-observation approximations.
- `Reverse-Strike Lookback` builds prices through generic minimum/maximum lookback decompositions and adds Monte Carlo engines for pathwise estimation.
- `Limited-Risk Lookback` treats the contracts as barrier-linked lookback structures, using closed-form up-and-out and down-and-out style constructions.
- `Partial Lookback` covers both partial-price and partial-time contracts, combining analytical formulas, method-of-images components, and simulation-based alternatives.
- `Quanto Lookback` combines closed-form lookback valuation with a quanto-adjusted drift and uses correlated asset/FX simulation when closed form is not available.

### Method Of Images And Bivariate Normal Integration

The partial and reverse-strike implementations rely on method-of-images style reasoning to replicate lookback payoffs from simpler building blocks. For partial-time options, the code also implements the bivariate normal distribution needed for correlated monitoring windows. The `Partial Lookback` module includes a custom Drezner-style Gauss-Legendre quadrature routine, with routing for numerical stability across sign and correlation regimes and optional SciPy acceleration when available.

### Quanto Adjustment

For quanto structures, the domestic-currency pricing logic uses the effective foreign-asset drift

$$
r_{\mathrm{eff}} = r_f - \rho_{S,F}\sigma_S\sigma_F
$$

before discounting back under the domestic curve. The Monte Carlo routines extend this setup by simulating correlated asset and FX paths explicitly.

### Greeks

All product families expose Delta, Gamma, and Vega through central finite differences:

$$
\Delta \approx \frac{V(S_0+h)-V(S_0-h)}{2h},
\qquad
\Gamma \approx \frac{V(S_0+h)-2V(S_0)+V(S_0-h)}{h^2},
\qquad
\mathcal{V} \approx \frac{V(\sigma+h_\sigma)-V(\sigma-h_\sigma)}{2h_\sigma}
$$

Default bump sizes are scaled to spot and volatility, with guards against invalid stencils. In the limited-risk module, `include_gamma` and `include_vega` can be disabled to reduce repeated pricer calls in larger batch workflows.

## Validation

The repository includes a lightweight validation harness in `validation/light_validation.py`.

It currently performs two classes of checks:

- BGK checks for the classical continuous-monitoring wrappers, including convergence behavior on floating-strike contracts and regression-style consistency checks on fixed-strike contracts.
- Finite-Greek smoke tests across floating-strike, fixed-strike, limited-risk, partial-price, partial-time, reverse-strike, and quanto contracts to ensure the reported sensitivities remain numerically well-defined.

Run it with:

```bash
python validation/light_validation.py
```

On success, it prints `Light validation passed.`

## Repository Layout

```text
Lookback Options/
├── Floating-Strike Lookback/
│   └── floating_strike_lookback.py
├── Fixed-Strike Lookback/
│   └── fixed_strike_lookback.py
├── Reverse-Strike Lookback/
│   └── reverse_strike_lookback.py
├── Limited-Risk Lookback/
│   └── limited_risk_lookback.py
├── Partial Lookback/
│   └── partial_lookback.py
├── Quanto Lookback/
│   └── quanto_lookback.py
├── validation/
│   └── light_validation.py
├── docs/
│   └── README.md
└── README.md
```

## Setup

This repository does not currently use `uv` or Poetry. The intended workflow is a lightweight local Python environment with direct module execution.

### Requirements

- Python 3.10+
- `numpy`
- `scipy` optional but recommended for faster normal CDF evaluation and bivariate normal support where used

Install the numerical dependencies with:

```bash
pip install numpy scipy
```

### Run Example Modules

Each pricing module includes a `__main__` block for direct execution:

```bash
python "Floating-Strike Lookback/floating_strike_lookback.py"
python "Fixed-Strike Lookback/fixed_strike_lookback.py"
python "Reverse-Strike Lookback/reverse_strike_lookback.py"
python "Limited-Risk Lookback/limited_risk_lookback.py"
python "Partial Lookback/partial_lookback.py"
python "Quanto Lookback/quanto_lookback.py"
```

### Use As A Library

The codebase is organized as standalone modules rather than an installed package. One simple pattern is dynamic import by path:

```python
from importlib.util import module_from_spec, spec_from_file_location

spec = spec_from_file_location(
    "floating",
    "Floating-Strike Lookback/floating_strike_lookback.py",
)
floating = module_from_spec(spec)
spec.loader.exec_module(floating)

price = floating.floating_strike_lookback_call(
    S0=100.0,
    T=1.0,
    r=0.05,
    sigma=0.20,
    delta=0.01,
)
greeks = floating.floating_strike_lookback_call_greeks(
    S0=100.0,
    T=1.0,
    r=0.05,
    sigma=0.20,
    delta=0.01,
)
print(price, greeks)
```

## Assumptions And Scope

- Closed-form prices assume continuous monitoring unless a discrete-monitoring wrapper is explicitly provided.
- All models assume constant parameters over the option life: rate, dividend yield, and volatility are time-homogeneous.
- The repository focuses on European-style exercise.
- The underlying dynamics are geometric Brownian motion without jumps.
- Partial lookback closed-form routines are limited to the documented dividend assumptions in their module; for unsupported dividend cases, Monte Carlo is the intended route.
- Some product families intentionally do not expose BGK-style wrappers because their discrete-to-continuous corrections are product-specific rather than generic.

## References

- Goldman, Sosin and Gatto (1979) for the original floating-strike lookback framework.
- Conze and Viswanathan (1991) for fixed-strike and limited-risk lookback formulations.
- Heynen and Kat (1994) for partial-price lookback structures.
- Buchen and Konstandatos (2005) for method-of-images constructions and partial-time lookbacks.
- Broadie, Glasserman and Kou (1999) for continuity correction under discrete monitoring.
- Drezner (1978) for bivariate normal CDF quadrature.
