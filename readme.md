Wavelet‑Driven Brownian Motion & Stochastic Simulation

This repository constructs Brownian motion from a wavelet basis (after Michael J. Steele) and for exploring geometric Brownian motion, Monte‑Carlo convergence, and Euler–Maruyama discretisation.

Key mathematical ideas include:

- **Wavelet series representation** of standard Brownian motion on [0, 1] using integrated Haar functions.
- **Time‑scaling property** giving paths on [0, T] via \( B_t = \sqrt{T}\,B_{t/T} \).
- **Geometric Brownian motion** expressed as \( S_t = S_0 \exp\!\bigl((\mu-\tfrac12\sigma^2)t + \sigma B_t\bigr) \).
- **Monte‑Carlo error analysis** showing \(O(N^{-1/2})\) convergence and CLT variance estimates for expectations of log‑normal functionals.
- **Euler–Maruyama scheme** for the GBM SDE and comparison with the exact distribution.

The core functions are:

| Function | Description |
|----------|-------------|
| `BrownianMotion` | Sample path of \( B_t \) via wavelet expansion |
| `GeometricBrownianMotion` | Exponential transform to obtain GBM |
| `MonteCarloCallSim` | Estimate payoffs of European call options and plot trajectories |
| `GBMEulerMaruyamaApprox` | Single‑step EM update \( Y_{n+1}=Y_n+\mu Y_nΔt+σY_nΔB \) |
| `EulerMonteCarloCallSim` | Monte‑Carlo estimator using Euler-Maruyama method for SDEs |

The wavelet representation used here follows Michael J. Steele, *Stochastic Calculus and Financial Applications*, Section 3.2.

