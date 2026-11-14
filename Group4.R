# ==============================================================================
# Machine Learning Assignment - Complete R Script (Questions 3, 4, 5b, and 6)
# ==============================================================================


# ==============================================================================
# QUESTION 3: Dataset Selection & Suitability of the Poisson Model
# ==============================================================================

# 1. Load the French Motor Third-Party Liability dataset
url <- "https://huggingface.co/datasets/mabilton/fremtpl2/resolve/main/freMTPL2freq.csv"
df  <- read.csv(url)

# 2. Filter for comparable policies (~1 year exposure, drivers 30-60, cars <= 10 yrs)
pool <- subset(
  df,
  Exposure >= 0.99 &
    Exposure <= 1.00 &
    DrivAge >= 30 &
    DrivAge <= 60 &
    VehAge <= 10
)

cat("=== Q3: Filtered Eligible Pool ===\n")
cat("Number of eligible policies:", nrow(pool), "\n")
print(table(pool$ClaimNb))
cat("Pool mean claims:", mean(pool$ClaimNb), "\n")
cat("Pool variance:   ", var(pool$ClaimNb), "\n\n")

# 3. Draw reproducible 200-policy analysis sample
set.seed(2026)
analysis_sample <- pool[sample(nrow(pool), 200), ]

# Save analysis sample CSV (if not already saved)
write.csv(analysis_sample, "insurance_analysis_sample_200.csv", row.names = FALSE)

x      <- analysis_sample$ClaimNb
n      <- length(x)
lambda <- mean(x)

cat("=== Q3: 200-Policy Analysis Sample ===\n")
cat("Sample size (n):   ", n, "\n")
cat("Total claims:      ", sum(x), "\n")
cat("Sample mean:       ", lambda, "\n")
cat("Sample variance:   ", var(x), "\n")
cat("Dispersion ratio:  ", var(x) / lambda, "\n\n")

# 4. Compare observed claim counts with Poisson expectations
observed <- c(
  sum(x == 0),
  sum(x == 1),
  sum(x == 2),
  sum(x >= 3)
)

expected <- n * c(
  dpois(0:2, lambda),
  1 - ppois(2, lambda)
)

comparison <- data.frame(
  Claims   = c("0", "1", "2", "3+"),
  Observed = observed,
  Poisson  = round(expected, 2)
)

print(comparison)

# Plot Observed vs Poisson Claim Frequencies
barplot(
  t(as.matrix(comparison[, 2:3])),
  beside      = TRUE,
  names.arg   = comparison$Claims,
  col         = c("steelblue", "orange"),
  legend.text = c("Observed", "Poisson"),
  main        = "Observed vs Poisson Claim Frequencies",
  xlab        = "Number of claims per policy",
  ylab        = "Number of policies"
)


# ==============================================================================
# QUESTION 4: Gamma Prior Selection & Analytical Conjugate Posterior
# ==============================================================================

# Exclude the 200 analysis policies to form an independent reference set
reference  <- pool[!(pool$IDpol %in% analysis_sample$IDpol), ]
n_ref      <- nrow(reference)
claims_ref <- sum(reference$ClaimNb)
mean_ref   <- mean(reference$ClaimNb)

cat("\n=== Q4: Reference Policies & Prior Specification ===\n")
cat("Reference policies:", n_ref, "\n")
cat("Reference claims:  ", claims_ref, "\n")
cat("Reference mean:    ", mean_ref, "\n")

# Weakly informative prior: theta ~ Gamma(alpha = 3, beta = 40)
alpha <- 3
beta  <- 40

prior_mean <- alpha / beta
prior_sd   <- sqrt(alpha) / beta

# Conjugate posterior update: theta | x ~ Gamma(alpha + sum(x), beta + n)
total_claims    <- sum(x)
analytical_mean <- (alpha + total_claims) / (beta + n)

cat("Prior mean (alpha/beta):    ", prior_mean, "\n")
cat("Prior SD (sqrt(alpha)/beta):", prior_sd, "\n")
cat("Analytical posterior mean:  ", analytical_mean, "\n\n")


# ==============================================================================
# QUESTION 5(b): Importance Sampling
# ==============================================================================

set.seed(2026)
M <- 10000

# Draw M = 10,000 independent values from the Gamma(3, 40) prior
theta_draws_is <- rgamma(M, shape = alpha, rate = beta)

# Compute log-likelihood weights for numerical stability
log_weights <- total_claims * log(theta_draws_is) - n * theta_draws_is
weights     <- exp(log_weights - max(log_weights))
weights     <- weights / sum(weights)

# Self-normalized importance sampling estimate & Effective Sample Size (ESS)
IS_mean <- sum(weights * theta_draws_is)
ESS     <- 1 / sum(weights^2)

cat("=== Q5(b): Importance Sampling Results (M = 10,000) ===\n")
cat("Importance Sampling Mean: ", IS_mean, "\n")
cat("Analytical Posterior Mean:", analytical_mean, "\n")
cat("Absolute Difference:      ", abs(IS_mean - analytical_mean), "\n")
cat("Relative Difference (%):  ", 100 * abs(IS_mean - analytical_mean) / analytical_mean, "\n")
cat("Effective Sample Size:    ", ESS, "\n\n")


# ==============================================================================
# QUESTION 6(b) & 6(c): Hierarchical Model & Gibbs Sampling
# ==============================================================================

# Hyperparameters for unknown rate beta ~ Gamma(a, b)
# Centered at E[beta] = a / b = 40 (matching Q4's fixed beta) with SD = 20
a <- 4
b <- 0.1

S       <- 10000   # Total MCMC iterations
burn_in <- 2000    # Burn-in draws to discard

theta_draws <- numeric(S)
beta_draws  <- numeric(S)

# Initialize beta at its prior mean
beta_curr <- 40

set.seed(2026)

for (s in 1:S) {
  # Step 1: Sample theta | beta, X ~ Gamma(alpha + total_claims, beta + n)
  theta_curr <- rgamma(1, shape = alpha + total_claims, rate = beta_curr + n)
  
  # Step 2: Sample beta | theta, X ~ Gamma(alpha + a, b + theta)
  beta_curr  <- rgamma(1, shape = alpha + a, rate = b + theta_curr)
  
  theta_draws[s] <- theta_curr
  beta_draws[s]  <- beta_curr
}

# Discard first 2,000 burn-in iterations
theta_post <- theta_draws[(burn_in + 1):S]
beta_post  <- beta_draws[(burn_in + 1):S]

theta_mean <- mean(theta_post)
theta_sd   <- sd(theta_post)
theta_ci   <- quantile(theta_post, probs = c(0.025, 0.975))

beta_mean  <- mean(beta_post)
beta_sd    <- sd(beta_post)
beta_ci    <- quantile(beta_post, probs = c(0.025, 0.975))

cat("=== Q6(c): Gibbs Sampling Posterior Summary (8,000 Retained Draws) ===\n")
cat(sprintf("theta | Mean: %.6f | SD: %.6f | 95%% CI: [%.6f, %.6f]\n",
            theta_mean, theta_sd, theta_ci[1], theta_ci[2]))
cat(sprintf("beta  | Mean: %.4f   | SD: %.4f   | 95%% CI: [%.4f, %.4f]\n",
            beta_mean, beta_sd, beta_ci[1], beta_ci[2]))

# Generate Trace Plots for theta and beta
par(mfrow = c(2, 1), mar = c(4, 4.5, 2.5, 1.5))

# Trace plot for theta
plot(1:S, theta_draws, type = "l", col = "steelblue", lwd = 0.6,
     main = "Gibbs Sampler Trace Plots (10,000 Iterations, 2,000 Burn-in)",
     xlab = "", ylab = expression(theta))
abline(v = burn_in, col = "red", lty = 2, lwd = 1.5)
abline(h = theta_mean, col = "darkorange", lwd = 1.5)
legend("topright",
       legend = c("Burn-in (2,000)", sprintf("Post. Mean = %.5f", theta_mean)),
       col = c("red", "darkorange"), lty = c(2, 1), lwd = 1.5, bty = "n")

# Trace plot for beta
plot(1:S, beta_draws, type = "l", col = "seagreen", lwd = 0.6,
     xlab = "Iteration", ylab = expression(beta))
abline(v = burn_in, col = "red", lty = 2, lwd = 1.5)
abline(h = beta_mean, col = "darkorange", lwd = 1.5)
legend("topright",
       legend = c("Burn-in (2,000)", sprintf("Post. Mean = %.2f", beta_mean)),
       col = c("red", "darkorange"), lty = c(2, 1), lwd = 1.5, bty = "n")

# Reset plot layout
par(mfrow = c(1, 1))