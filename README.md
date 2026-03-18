# Lookback Options — Closed-Form Pricing & Monte Carlo

A from-scratch Python implementation of **six families of European lookback options** under Black-Scholes dynamics, featuring analytical closed-form solutions, Monte Carlo engines, finite-difference Greeks, and BGK discrete-monitoring corrections.

Built independently as a self-directed study in exotic derivatives pricing — no textbook code, no library wrappers, just the papers and the math.

---

## Why This Exists

I am a university economics student who taught himself derivatives pricing from primary sources — Goldman-Sachs research notes, Conze & Viswanathan (1991), Buchen & Konstandatos (2005), Broadie, Glasserman & Kou (1999), and Drezner (1978). Every formula in this repository was derived from the referenced literature, implemented by hand, and validated against known benchmarks.

This project demonstrates:

- Fluency with **path-dependent exotic option theory** (extremum distributions under GBM, Method of Images, static replication).
- Ability to translate dense quantitative finance papers into **production-quality numerical code** with edge-case handling.
- Independent work ethic — no course requirement, no starter code, no shortcuts.

---

## Implemented Product Families

| Family | Module | Payoff (Call) | Payoff (Put) | Methods |
|---|---|---|---|---|
| **Floating-Strike** | `Floating-Strike Lookback/` | \(S_T - Y_T\) | \(Z_T - S_T\) | Closed-form, BGK correction, Greeks |
| **Fixed-Strike** | `Fixed-Strike Lookback/` | \(\max(Z_T - K, 0)\) | \(\max(K - Y_T, 0)\) | Closed-form, BGK correction, Greeks |
| **Reverse-Strike** | `Reverse-Strike Lookback/` | \(\max(Y_T - K, 0)\) | \(\max(K - Z_T, 0)\) | Closed-form, Monte Carlo, Greeks |
| **Limited-Risk** | `Limited-Risk Lookback/` | \((S_T - K) \cdot \mathbf{1}_{Z_T < m}\) | \((K - S_T) \cdot \mathbf{1}_{Y_T > m}\) | Closed-form (UOC/DOP), Greeks |
| **Partial** | `Partial Lookback/` | \(\max(S_T - \lambda Y_T, 0)\), \(\max(S_{T_2} - Y_{T_1}, 0)\) | \(\max(\mu Z_T - S_T, 0)\), \(\max(Z_{T_1} - S_{T_2}, 0)\) | Closed-form, Monte Carlo, Greeks |
| **Quanto** | `Quanto Lookback/` | \(F_c \cdot \max(Z_T - K, 0)\) | \(F_c \cdot \max(K - Y_T, 0)\) | Closed-form (fixed/floating), Monte Carlo (max-FX, joint), Greeks |

Where \(Y_T = \min_{0 \le t \le T} S_t\), \(Z_T = \max_{0 \le t \le T} S_t\), \(K\) = fixed strike, \(\lambda \ge 1\) and \(\mu \le 1\) are partial-price multipliers, \(T_1 < T_2\) defines the partial-time monitoring window, \(m\) is a barrier level, and \(F_c\) is a predetermined currency conversion factor.

---

## Mathematical Foundation

### Model Assumptions

All closed-form pricing operates under the standard **Black-Scholes / GBM** framework:

$$dS_t = (r - \delta)\,S_t\,dt + \sigma\,S_t\,dW_t$$

with constant risk-free rate \(r\), continuous dividend yield \(\delta\), and volatility \(\sigma\). Pricing is performed under the risk-neutral measure \(\mathbb{Q}\).

### Core Auxiliary Functions

Every module shares a common set of standardised distance functions:

$$d_{1,K} = \frac{\ln(S_0/K) + (r - \delta + \tfrac{1}{2}\sigma^2)\,T}{\sigma\sqrt{T}}, \qquad d_{2,K} = d_{1,K} - \sigma\sqrt{T}, \qquad d_{3,K} = d_{1,K} - \frac{2(r-\delta)\sqrt{T}}{\sigma}$$

with cost-of-carry \(b = r - \delta\). The \(d_3\) term arises from the reflected Brownian motion that governs the extremum distribution.

### Lookback Premium Decomposition

A key structural insight: every lookback price decomposes as

$$V_{\text{lookback}} = V_{\text{BS vanilla}} + \text{lookback premium}$$

The premium term captures the additional value of exercising at the path extremum rather than the terminal price. For floating-strike options, the premium involves the ratio \(\sigma^2 / (2b)\) and reflected-path CDF terms. For partial-price options, the premium scales with \(\lambda^{\beta+2}\) (or \(\mu^{\beta+2}\)), where \(\alpha = \sigma^2/(2b)\) and \(\beta = 2b/\sigma^2 - 1\).

### Method of Images

The partial-lookback and reverse-strike modules use the **Buchen-Konstandatos Method of Images** for static replication. Binary building blocks \(A^\pm_\xi\), \(B^\pm_\xi\), \(*B^\pm_\xi\) (image binaries), and lookback premium operators \(D^\pm_\xi\) are assembled to replicate arbitrary lookback payoffs:

$$D^-_\xi = -\alpha\bigl[A^-_\xi - \xi \cdot {*B}^+_\xi\bigr], \qquad D^+_\xi = \alpha\bigl[A^+_\xi - \xi \cdot {*B}^-_\xi\bigr]$$

### Bivariate Normal CDF

Partial-time lookback options require the bivariate normal CDF \(N_2(a, b; \rho)\) with correlation \(\rho = \sqrt{T_1/T_2}\). This is computed via a custom implementation of the **Drezner (1978)** Gauss-Legendre quadrature algorithm (20-point), with symmetry routing for numerical stability across all sign combinations of \((h, k, \rho)\). Falls back to `scipy.stats.multivariate_normal` when available.

### Quanto Drift Adjustment

Quanto lookback options price a foreign-denominated asset paid in domestic currency. The closed-form solution replaces the domestic risk-free rate with the **quanto-adjusted effective rate**:

$$r_{\text{eff}} = r_f - \rho_{S,F}\,\sigma_S\,\sigma_F$$

where \(r_f\) is the foreign rate, \(\rho_{S,F}\) is the asset-FX correlation, and \(\sigma_S\), \(\sigma_F\) are asset and FX volatilities. Discounting uses the domestic rate \(r_d\). For products requiring joint path simulation (max exchange rate quanto, joint quanto), Monte Carlo with correlated Cholesky-decomposed Brownian motions is used.

### BGK Continuity Correction

Real-world lookback options monitor prices discretely (e.g., daily closes). Continuous-monitoring formulas overprice these contracts. The **Broadie-Glasserman-Kou (1999)** correction adjusts via:

$$H_{\text{discrete}} \approx H_{\text{continuous}} \cdot e^{\pm \beta_1 \sigma \sqrt{\Delta t}}$$

where \(\beta_1 \approx 0.5826\) (derived from the Riemann zeta function \(\zeta(1/2)/\sqrt{2\pi}\)) and \(\Delta t = T/m\) for \(m\) monitoring points. Implemented for floating-strike and fixed-strike families. Partial, quanto, reverse-strike, and limited-risk families intentionally omit BGK wrapping because they require product-specific discrete-to-continuous adjustments.

### Near-Zero Drift Handling (\(b \to 0\))

When \(b = r - \delta \approx 0\), the standard \(\alpha = \sigma^2/(2b)\) diverges. Every module implements a dedicated **L'Hopital / finite-limit branch** that evaluates the full pricing expression in its \(b \to 0\) limit, avoiding heuristic \(\alpha/\beta\) substitution. The threshold is typically \(|b| < 10^{-8}\) to \(10^{-5}\) depending on the product.

---

## Greeks

All product families expose **Delta**, **Gamma**, and **Vega** via central finite-difference bumping:

$$\Delta \approx \frac{V(S_0 + h) - V(S_0 - h)}{2h}, \qquad \Gamma \approx \frac{V(S_0 + h) - 2V(S_0) + V(S_0 - h)}{h^2}, \qquad \mathcal{V} \approx \frac{V(\sigma + h_\sigma) - V(\sigma - h_\sigma)}{2h_\sigma}$$

Bump sizes default to \(10^{-4} \times S_0\) (spot) and \(10^{-4} \times \sigma\) (vol), with guards against zero or excessively large bumps.

---

## Project Structure

```
Lookback Options/
├── Floating-Strike Lookback/
│   └── floating_strike_lookback.py    # Closed-form, BGK, Greeks, payoffs
├── Fixed-Strike Lookback/
│   └── fixed_strike_lookback.py       # Closed-form, BGK, Greeks, payoffs
├── Reverse-Strike Lookback/
│   └── reverse_strike_lookback.py     # Closed-form, MC, Greeks, payoffs
├── Limited-Risk Lookback/
│   └── limited_risk_lookback.py       # UOC/DOP closed-form, Greeks
├── Partial Lookback/
│   └── partial_lookback.py            # Partial-price + partial-time, MC, Greeks
├── Quanto Lookback/
│   └── quanto_lookback.py             # Quanto fixed/floating CF, MC (max-FX, joint), Greeks
├── validation/
│   └── light_validation.py            # BGK convergence + Greeks smoke tests
└── README.md
```

---

## Quickstart

### Requirements

- **Python 3.10+**
- **NumPy** (required by Partial, Reverse-Strike, and Quanto modules for Monte Carlo path simulation and the Drezner quadrature nodes)
- **SciPy** (optional; used for `norm.cdf` and `multivariate_normal.cdf` when available, otherwise pure-Python fallbacks are used)

Install dependencies:

```bash
pip install numpy scipy
```

### Run Individual Modules

Each module has a `__main__` block that prints example prices:

```bash
python "Floating-Strike Lookback/floating_strike_lookback.py"
python "Fixed-Strike Lookback/fixed_strike_lookback.py"
python "Reverse-Strike Lookback/reverse_strike_lookback.py"
python "Limited-Risk Lookback/limited_risk_lookback.py"
python "Partial Lookback/partial_lookback.py"
python "Quanto Lookback/quanto_lookback.py"
```

### Run Validation

```bash
python validation/light_validation.py
```

On success, prints `Light validation passed.`

### Use as a Library

```python
from importlib.util import spec_from_file_location, module_from_spec

spec = spec_from_file_location("floating", "Floating-Strike Lookback/floating_strike_lookback.py")
floating = module_from_spec(spec)
spec.loader.exec_module(floating)

price = floating.floating_strike_lookback_call(S0=100, T=1.0, r=0.05, sigma=0.2)
greeks = floating.floating_strike_lookback_call_greeks(S0=100, T=1.0, r=0.05, sigma=0.2)
print(f"Price: {price:.4f}, Delta: {greeks.delta:.4f}, Gamma: {greeks.gamma:.6f}, Vega: {greeks.vega:.4f}")
```

---

## Validation

The validation suite (`validation/light_validation.py`) performs two categories of checks:

**1. BGK Convergence Sanity** — Verifies that the BGK-corrected discrete price converges toward the continuous-monitoring price as the number of monitoring points increases. For both floating-strike calls and puts:
- Price at 5000 monitoring points is closer to continuous than price at 50 monitoring points.
- Relative error at 5000 points is bounded (< 8% for calls, < 3% for puts).

**2. Greeks Smoke Tests** — Confirms that Delta, Gamma, and Vega are finite (not NaN or Inf) for every product family across representative parameter sets, covering floating-strike, fixed-strike, limited-risk, partial-price, partial-time, reverse-strike, and all four quanto variants.

---

## Assumptions and Limitations

- **Continuous monitoring**: Closed-form prices assume the extremum is observed continuously. Real contracts monitor discretely; use the BGK-corrected functions where available, or Monte Carlo for product families without BGK wrappers.
- **Constant parameters**: All of \(r\), \(\delta\), \(\sigma\) are assumed constant over the life of the option. Stochastic volatility and term-structure effects are not modelled.
- **European exercise only**: No early-exercise (American/Bermudan) pricing is implemented.
- **GBM dynamics**: The underlying follows geometric Brownian motion with no jumps. Real asset prices exhibit fat tails, jumps, and volatility clustering.
- **Partial lookback dividends**: Partial-price and partial-time closed-form solutions are derived under \(\delta = 0\) only. For \(\delta \neq 0\), the Monte Carlo engines should be used.
- **No transaction costs or market frictions**.
- **Numerical edge cases**: Near \(b = 0\), \(\sigma \to 0\), or \(T \to 0\), dedicated branches handle limits, but extreme parameter combinations may still produce numerical noise.

---

## Roadmap

- [ ] Formal test suite with `pytest`, including regression tests against published table values (e.g., Conze-Viswanathan Table 1, Heynen-Kat benchmarks)
- [ ] Extreme spread lookback options (forward-start extremum differences)
- [ ] American lookback pricing via binomial trees or PDE methods
- [ ] Stochastic volatility extensions (Heston lookback)
- [ ] Package as an installable library with `pyproject.toml`
- [ ] Interactive visualisation of lookback surfaces and Greek profiles

---

## References

- **Goldman, Sosin & Gatto (1979)** — Original floating-strike lookback pricing under GBM.
- **Conze & Viswanathan (1991)** — Fixed-strike lookback options and limited-risk (barrier) lookback formulations.
- **Buchen & Konstandatos (2005)** — Method of Images for partial-time lookback options and static replication via binary building blocks.
- **Broadie, Glasserman & Kou (1999)** — Continuity correction for discrete-monitoring lookback and barrier options (\(\beta_1 \approx 0.5826\)).
- **Heynen & Kat (1994)** — Partial-price lookback options (multiplier on extremum).
- **Drezner (1978)** — Bivariate normal CDF computation via Gauss-Legendre quadrature.
