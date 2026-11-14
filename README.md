# bayesian-insurance-claim-modeling
Bayesian estimation of motor insurance claim frequencies using Poisson-Gamma conjugacy, Importance Sampling, and Gibbs Sampling in R
# Bayesian Inference for Motor Insurance Claim Frequencies

Bayesian modeling of annual motor insurance claim frequencies using the French Motor Third-Party Liability dataset (`freMTPL2freq`, $n = 200$ policies, $\sum X_i = 20$ claims).

## Methods Implemented
1. **Analytical Conjugate Model:** Derived the exact posterior $\theta \mid X_{1:n} \sim \text{Gamma}(23, 240)$ under a weakly informative $\text{Gamma}(3, 40)$ prior and analyzed asymptotic convergence as $n \to \infty$.
2. **Importance Sampling:** Simulated $M = 10,000$ prior draws with log-sum-exp numerical stabilization, achieving an Effective Sample Size (ESS) of $4,361$.
3. **Hierarchical Gibbs Sampling:** Extended the model to treat the prior rate $\beta \sim \text{Gamma}(4, 0.1)$ as unknown and ran a two-step Gibbs sampler ($10,000$ iterations, $2,000$ burn-in).

## Key Results Summary

| Method | Posterior Mean ($\theta$) | Notes |
| :--- | :---: | :--- |
| **Analytical Conjugate** | `0.09583` | Exact $\text{Gamma}(23, 240)$ posterior |
| **Importance Sampling** | `0.09613` | $0.31\%$ relative error; $\text{ESS} = 4,360.62$ |
| **Hierarchical Gibbs Sampler** | `0.09833` | Unknown $\beta$ posterior mean $= 35.79$ |

## Repository Files
* `Group4.R` — Complete R script for data filtering, Poisson diagnostics, Importance Sampling, and Gibbs Sampling.
* `Group4.pdf` — Full mathematical derivations, diagnostics, and trace plots.
* `insurance_analysis_sample_200.csv` — Reproducible 200-policy analysis sample (`set.seed(2026)`).