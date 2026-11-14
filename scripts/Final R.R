# ==============================================================================
# Machine Learning Assignment - Group 4
# Askhat Doschanov, Bibinur Ashirepbay, Metodi Popchevaliev
# Complete R Script: Questions 3, 4, 5(b), 6(b)-(c), and Sensitivity Analysis
# ==============================================================================
# Note on reproducibility: R's sample() algorithm changed in R 3.6.0 (RNGkind).
# Shipping insurance_analysis_sample_200.csv guarantees exact reproducibility
# regardless of the R version used to run this script.


# ==============================================================================
# QUESTION 3: Dataset Selection & Formal Poisson Diagnostics
# ==============================================================================

url <- "https://huggingface.co/datasets/mabilton/fremtpl2/resolve/main/freMTPL2freq.csv"

df <- tryCatch(
  {
    read.csv(url)
  },
  error = function(e) {
    message("Remote URL unavailable; loading local CSV fallback.")
    NULL
  }
)

if (!is.null(df)) {
  # Filter for comparable policies (~1 yr exposure, drivers 30-60, cars <= 10 yrs)
  pool <- subset(
    df,
    Exposure >= 0.99 &
      Exposure <= 1.00 &
      DrivAge >= 30 &
      DrivAge <= 60 &
      VehAge <= 10
  )
  
  pool_mean <- mean(pool$ClaimNb)
  pool_var  <- var(pool$ClaimNb)
  pool_disp <- pool_var / pool_mean
  
  cat("=== Q3: Full Filtered Eligible Pool ===\n")
  cat("Number of eligible policies:", nrow(pool), "\n")
  print(table(pool$ClaimNb))
  cat(sprintf("Pool mean: %.5f | Pool var: %.5f | Pool dispersion ratio: %.4f\n\n",
              pool_mean, pool_var, pool_disp))
  
  # Reproducible 200-policy analysis sample
  set.seed(2026)
  analysis_sample <- pool[sample(nrow(pool), 200), ]
  write.csv(analysis_sample, "insurance_analysis_sample_200.csv", row.names = FALSE)
} else {
  analysis_sample <- read.csv("insurance_analysis_sample_200.csv")
}

x         <- analysis_sample$ClaimNb
n         <- length(x)
theta_hat <- mean(x)
s2        <- var(x)
disp_ratio <- s2 / theta_hat

# Formal dispersion test: (n - 1) * s^2 / theta_hat ~ Chi-sq(n - 1) under Poisson
disp_stat <- (n - 1) * s2 / theta_hat
disp_pval <- pchisq(disp_stat, df = n - 1, lower.tail = FALSE)

cat("=== Q3: 200-Policy Analysis Sample & Formal Tests ===\n")
cat(sprintf("Sample size (n): %d | Total claims: %d\n", n, sum(x)))
cat(sprintf("Sample mean (theta_hat): %.4f | Sample variance (s^2): %.5f\n", theta_hat, s2))
cat(sprintf("Dispersion ratio (s^2 / theta_hat): %.4f\n", disp_ratio))
cat(sprintf("Dispersion test stat: %.2f (df = %d, upper-tail p = %.4f)\n\n",
            disp_stat, n - 1, disp_pval))

# Observed vs Poisson-expected frequencies (0, 1, 2, 3+)
observed <- c(sum(x == 0), sum(x == 1), sum(x == 2), sum(x >= 3))
expected <- n * c(dpois(0:2, theta_hat), 1 - ppois(2, theta_hat))

comparison <- data.frame(
  Claims   = c("0", "1", "2", "3+"),
  Observed = observed,
  Poisson  = round(expected, 2)
)
print(comparison)

# Chi-square goodness-of-fit test pooling 2+ cell (0, 1, 2+)
obs_pooled <- c(sum(x == 0), sum(x == 1), sum(x >= 2))
exp_pooled <- n * c(dpois(0, theta_hat), dpois(1, theta_hat), 1 - ppois(1, theta_hat))
chisq_gof  <- sum((obs_pooled - exp_pooled)^2 / exp_pooled)
chisq_pval <- pchisq(chisq_gof, df = 1, lower.tail = FALSE)

cat(sprintf("\nChi-square GOF (pooled 2+ cell, exp = %.2f < 5): stat = %.4f, df = 1, p = %.4f\n\n",
            exp_pooled[3], chisq_gof, chisq_pval))

# Save Figure 1: Observed vs Poisson Claim Frequencies
png("Q3_Poisson_Comparison.png", width = 900, height = 560, res = 130)
barplot(
  t(as.matrix(comparison[, 2:3])),
  beside      = TRUE,
  names.arg   = comparison$Claims,
  col         = c("steelblue", "orange"),
  legend.text = c("Observed", "Poisson"),
  main        = "Observed vs Poisson Claim Frequencies (n = 200)",
  xlab        = "Number of claims per policy",
  ylab        = "Number of policies"
)
dev.off()


# ==============================================================================
# QUESTION 4: Gamma Prior Selection & Exact Conjugate Posterior
# ==============================================================================

if (!is.null(df)) {
  reference  <- pool[!(pool$IDpol %in% analysis_sample$IDpol), ]
  n_ref      <- nrow(reference)
  claims_ref <- sum(reference$ClaimNb)
  mean_ref   <- mean(reference$ClaimNb)
  cat("=== Q4: Excluded Reference Policies ===\n")
  cat(sprintf("Reference policies: %d | Claims: %d | Mean: %.5f\n",
              n_ref, claims_ref, mean_ref))
}

# Weakly informative prior: theta ~ Gamma(alpha = 3, beta_prior = 40)
alpha      <- 3
beta_prior <- 40

prior_mean <- alpha / beta_prior
prior_sd   <- sqrt(alpha) / beta_prior

total_claims    <- sum(x)
post_alpha      <- alpha + total_claims
post_beta       <- beta_prior + n
analytical_mean <- post_alpha / post_beta
analytical_sd   <- sqrt(post_alpha) / post_beta
analytical_ci   <- qgamma(c(0.025, 0.975), shape = post_alpha, rate = post_beta)

cat(sprintf("Prior: Gamma(%d, %d) -> Mean = %.4f, SD = %.5f\n",
            alpha, beta_prior, prior_mean, prior_sd))
cat(sprintf("Posterior: Gamma(%d, %d) -> Mean = %.6f, SD = %.6f, 95%% CI = [%.5f, %.5f]\n\n",
            post_alpha, post_beta, analytical_mean, analytical_sd,
            analytical_ci[1], analytical_ci[2]))


# ==============================================================================
# QUESTION 5(b): Importance Sampling, MCSE, and Weighted Quantiles
# ==============================================================================

weighted_quantile <- function(vals, w, probs) {
  ord <- order(vals)
  vals <- vals[ord]
  w    <- w[ord] / sum(w)
  cw   <- cumsum(w)
  sapply(probs, function(p) vals[which(cw >= p)[1]])
}

set.seed(2026)
M <- 10000

theta_draws_is <- rgamma(M, shape = alpha, rate = beta_prior)
log_weights    <- total_claims * log(theta_draws_is) - n * theta_draws_is
weights        <- exp(log_weights - max(log_weights))
weights        <- weights / sum(weights)

IS_mean <- sum(weights * theta_draws_is)
IS_var  <- sum(weights * (theta_draws_is - IS_mean)^2)
IS_sd   <- sqrt(IS_var)
ESS_is  <- 1 / sum(weights^2)
MCSE_is <- IS_sd / sqrt(ESS_is)
IS_ci   <- weighted_quantile(theta_draws_is, weights, c(0.025, 0.975))

cat("=== Q5(b): Importance Sampling Results (M = 10,000) ===\n")
cat(sprintf("IS Posterior Mean:   %.6f (Exact: %.6f)\n", IS_mean, analytical_mean))
cat(sprintf("Absolute Difference: %.6f (MCSE: %.6f)\n", abs(IS_mean - analytical_mean), MCSE_is))
cat(sprintf("Relative Difference: %.2f%%\n", 100 * abs(IS_mean - analytical_mean) / analytical_mean))
cat(sprintf("Effective Sample Size (ESS): %.2f\n", ESS_is))
cat(sprintf("IS 95%% CI: [%.5f, %.5f] vs Exact Gamma(23, 240) CI: [%.5f, %.5f]\n\n",
            IS_ci[1], IS_ci[2], analytical_ci[1], analytical_ci[2]))


# ==============================================================================
# QUESTION 6(b) & 6(c): Two-Chain Hierarchical Gibbs Sampler & Diagnostics
# ==============================================================================

run_gibbs_chain <- function(a_hyp, b_hyp, S = 10000, burn_in = 2000,
                            beta_init = 40, seed = 2026) {
  set.seed(seed)
  th <- numeric(S)
  be <- numeric(S)
  bc <- beta_init
  for (s in 1:S) {
    tc <- rgamma(1, shape = alpha + total_claims, rate = bc + n)
    bc <- rgamma(1, shape = alpha + a_hyp, rate = b_hyp + tc)
    th[s] <- tc
    be[s] <- bc
  }
  list(
    theta_all  = th,
    beta_all   = be,
    theta_post = th[(burn_in + 1):S],
    beta_post  = be[(burn_in + 1):S]
  )
}

# ESS helper (uses coda::effectiveSize if installed, else spectral/IPS estimator)
compute_ess <- function(chain) {
  if (requireNamespace("coda", quietly = TRUE)) {
    return(as.numeric(coda::effectiveSize(coda::as.mcmc(chain))))
  }
  n_s <- length(chain)
  acf_vals <- acf(chain, lag.max = floor(n_s / 2), plot = FALSE)$acf[-1]
  tau <- 1
  for (k in seq_len(floor(length(acf_vals) / 2))) {
    pair_sum <- acf_vals[2 * k - 1] + acf_vals[2 * k]
    if (is.na(pair_sum) || pair_sum < 0) break
    tau <- tau + 2 * pair_sum
  }
  n_s / tau
}

# Gelman-Rubin R-hat across two chains
gelman_rubin_rhat <- function(c1, c2) {
  M_len <- length(c1)
  W <- 0.5 * (var(c1) + var(c2))
  B <- M_len * var(c(mean(c1), mean(c2)))
  var_hat <- ((M_len - 1) / M_len) * W + (1 / M_len) * B
  sqrt(var_hat / W)
}

# Hyperparameters: beta ~ Gamma(a = 4, b = 0.1) -> E[beta] = 40, E[theta] = 0.10
a_hyp   <- 4
b_hyp   <- 0.1
S       <- 10000
burn_in <- 2000

chain1 <- run_gibbs_chain(a_hyp, b_hyp, S = S, burn_in = burn_in, beta_init = 40, seed = 2026)
chain2 <- run_gibbs_chain(a_hyp, b_hyp, S = S, burn_in = burn_in, beta_init = 5,  seed = 2027)

theta_post <- chain1$theta_post
beta_post  <- chain1$beta_post

theta_mean <- mean(theta_post)
theta_sd   <- sd(theta_post)
theta_ci   <- quantile(theta_post, c(0.025, 0.975))
theta_ess  <- compute_ess(theta_post)
theta_mcse <- theta_sd / sqrt(theta_ess)
theta_rhat <- gelman_rubin_rhat(chain1$theta_post, chain2$theta_post)

beta_mean  <- mean(beta_post)
beta_sd    <- sd(beta_post)
beta_ci    <- quantile(beta_post, c(0.025, 0.975))
beta_ess   <- compute_ess(beta_post)
beta_mcse  <- beta_sd / sqrt(beta_ess)
beta_rhat  <- gelman_rubin_rhat(chain1$beta_post, chain2$beta_post)

cat("=== Q6(c): Hierarchical Gibbs Posterior & Convergence Diagnostics ===\n")
cat(sprintf("theta | Mean: %.5f | SD: %.5f | 95%% CI: [%.5f, %.5f] | ESS: %.1f | MCSE: %.6f | R-hat: %.5f\n",
            theta_mean, theta_sd, theta_ci[1], theta_ci[2], theta_ess, theta_mcse, theta_rhat))
cat(sprintf("beta  | Mean: %.2f   | SD: %.2f   | 95%% CI: [%.2f, %.2f]   | ESS: %.1f | MCSE: %.4f   | R-hat: %.5f\n\n",
            beta_mean, beta_sd, beta_ci[1], beta_ci[2], beta_ess, beta_mcse, beta_rhat))

# Save Figure 2: Full Trace Plots + First 500 Iterations Zoom
png("Q6_Gibbs_Trace_Plots.png", width = 1100, height = 720, res = 130)
par(mfrow = c(2, 2), mar = c(4, 4.5, 2.5, 1.2))

# Full trace theta
plot(1:S, chain1$theta_all, type = "l", col = "steelblue", lwd = 0.5,
     main = "Trace of theta (10,000 Iterations)", xlab = "Iteration", ylab = expression(theta))
abline(v = burn_in, col = "red", lty = 2, lwd = 1.4)
abline(h = theta_mean, col = "darkorange", lwd = 1.4)

# Zoom first 500 iterations theta
plot(1:500, chain1$theta_all[1:500], type = "l", col = "steelblue", lwd = 0.9,
     main = "Early Convergence Zoom: theta (First 500)", xlab = "Iteration", ylab = expression(theta))
lines(1:500, chain2$theta_all[1:500], col = rgb(0.85, 0.2, 0.2, 0.65), lwd = 0.8)
abline(h = theta_mean, col = "darkorange", lwd = 1.4)
legend("topright", legend = c("Start beta=40", "Start beta=5"),
       col = c("steelblue", rgb(0.85, 0.2, 0.2, 0.8)), lty = 1, lwd = 1.2, bty = "n", cex = 0.85)

# Full trace beta
plot(1:S, chain1$beta_all, type = "l", col = "seagreen", lwd = 0.5,
     main = "Trace of beta (10,000 Iterations)", xlab = "Iteration", ylab = expression(beta))
abline(v = burn_in, col = "red", lty = 2, lwd = 1.4)
abline(h = beta_mean, col = "darkorange", lwd = 1.4)

# Zoom first 500 iterations beta
plot(1:500, chain1$beta_all[1:500], type = "l", col = "seagreen", lwd = 0.9,
     main = "Early Convergence Zoom: beta (First 500)", xlab = "Iteration", ylab = expression(beta))
lines(1:500, chain2$beta_all[1:500], col = rgb(0.85, 0.2, 0.2, 0.65), lwd = 0.8)
abline(h = beta_mean, col = "darkorange", lwd = 1.4)
legend("topright", legend = c("Start beta=40", "Start beta=5"),
       col = c("seagreen", rgb(0.85, 0.2, 0.2, 0.8)), lty = 1, lwd = 1.2, bty = "n", cex = 0.85)

dev.off()

# Save Figure 3: ACF Plots + Posterior Histograms with Prior Densities Overlaid
png("Q6_Gibbs_Diagnostics.png", width = 1100, height = 720, res = 130)
par(mfrow = c(2, 2), mar = c(4, 4.5, 2.5, 1.2))

# ACF theta
acf(theta_post, lag.max = 30, main = "Autocorrelation Function: theta", col = "steelblue", lwd = 2)

# Posterior histogram theta + Marginal prior density
hist(theta_post, breaks = 40, freq = FALSE, col = "lightblue", border = "white",
     main = "Posterior vs Marginal Prior: theta", xlab = expression(theta), xlim = c(0, 0.22))
grid_th <- seq(0.001, 0.22, length.out = 300)
# Marginal prior of theta under Gamma(alpha, beta) & beta ~ Gamma(a, b) is Gamma-Prime / Beta-Prime:
# p(theta) = (b^a / B(alpha, a)) * theta^(alpha - 1) * (b + theta)^(-(alpha + a))
prior_th_dens <- (b_hyp^a_hyp / base::beta(alpha, a_hyp)) *
  grid_th^(alpha - 1) * (b_hyp + grid_th)^(-(alpha + a_hyp))
lines(grid_th, prior_th_dens, col = "red", lwd = 2, lty = 2)
abline(v = theta_mean, col = "darkorange", lwd = 2)
legend("topright", legend = c("Posterior", "Marginal Prior", "Post. Mean"),
       col = c("lightblue", "red", "darkorange"), lty = c(1, 2, 1), lwd = c(6, 2, 2), bty = "n", cex = 0.85)

# ACF beta
acf(beta_post, lag.max = 30, main = "Autocorrelation Function: beta", col = "seagreen", lwd = 2)

# Posterior histogram beta + Gamma(4, 0.1) prior density
hist(beta_post, breaks = 40, freq = FALSE, col = "darkseagreen2", border = "white",
     main = "Posterior vs Prior: beta", xlab = expression(beta), xlim = c(0, 110))
grid_be <- seq(0.1, 110, length.out = 300)
lines(grid_be, dgamma(grid_be, shape = a_hyp, rate = b_hyp), col = "red", lwd = 2, lty = 2)
abline(v = beta_mean, col = "darkorange", lwd = 2)
legend("topright", legend = c("Posterior", "Gamma(4, 0.1) Prior", "Post. Mean"),
       col = c("darkseagreen2", "red", "darkorange"), lty = c(1, 2, 1), lwd = c(6, 2, 2), bty = "n", cex = 0.85)

dev.off()
par(mfrow = c(1, 1))


# ==============================================================================
# SENSITIVITY ANALYSIS: Fixed Priors & Alternative Hierarchical Hyperparameters
# ==============================================================================

cat("=== Sensitivity Analysis ===\n")
# 1. Fixed-beta conjugate models keeping prior mean = 0.075 (alpha = 0.075 * beta_prior)
for (bp in c(10, 40, 200)) {
  ap <- 0.075 * bp
  pm <- (ap + total_claims) / (bp + n)
  ps <- sqrt(ap + total_claims) / (bp + n)
  ci <- qgamma(c(0.025, 0.975), shape = ap + total_claims, rate = bp + n)
  cat(sprintf("Fixed Gamma(%.2f, %d) [Prior Mean 0.075] -> Post Mean: %.5f | SD: %.5f | 95%% CI: [%.5f, %.5f]\n",
              ap, bp, pm, ps, ci[1], ci[2]))
}

# 2. Alternative hierarchical (a, b) priors centered at E[beta] = 40
for (pair in list(c(2, 0.05), c(4, 0.1), c(16, 0.4))) {
  ch <- run_gibbs_chain(pair[1], pair[2], S = S, burn_in = burn_in, beta_init = 40, seed = 2026)
  pm <- mean(ch$theta_post)
  ps <- sd(ch$theta_post)
  ci <- quantile(ch$theta_post, c(0.025, 0.975))
  cat(sprintf("Hierarchical (a=%.0f, b=%.2f) [Prior E[theta]=%.3f] -> Post Mean theta: %.5f | SD: %.5f | 95%% CI: [%.5f, %.5f] | Post Mean beta: %.2f\n",
              pair[1], pair[2], alpha * pair[2] / (pair[1] - 1),
              pm, ps, ci[1], ci[2], mean(ch$beta_post)))
}

cat("\n=== Session Information ===\n")
print(sessionInfo())