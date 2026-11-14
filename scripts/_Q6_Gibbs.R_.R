# ==============================================================================
# Question 6: Hierarchical Poisson-Gamma Model & Gibbs Sampling
# ==============================================================================

# 1. Load the 200-policy analysis sample
analysis_sample <- read.csv("data/insurance_analysis_sample_200.csv")

n     <- nrow(analysis_sample)          # 200 policies
sum_x <- sum(analysis_sample$ClaimNb)   # 20 total observed claims
alpha <- 3                              # Fixed shape parameter from Q4

# 2. Select hyperparameters a and b for beta ~ Gamma(a, b)
# Centered at E[beta] = a / b = 40 (matching Q4's fixed beta = 40) with SD = 20
a <- 4
b <- 0.1

# 3. Gibbs Sampler Settings (Q6b)
S       <- 10000   # Total iterations
burn_in <- 2000    # Discard first 2,000 draws

theta_draws <- numeric(S)
beta_draws  <- numeric(S)

# Initialize beta at its prior mean
beta_curr <- 40

set.seed(2026)

for (s in 1:S) {
  # Step 1: Sample theta | beta, X ~ Gamma(alpha + sum_x, beta + n)
  theta_curr <- rgamma(1, shape = alpha + sum_x, rate = beta_curr + n)
  
  # Step 2: Sample beta | theta, X ~ Gamma(alpha + a, b + theta)
  beta_curr  <- rgamma(1, shape = alpha + a, rate = b + theta_curr)
  
  theta_draws[s] <- theta_curr
  beta_draws[s]  <- beta_curr
}

# 4. Discard burn-in and compute posterior summaries (Q6c)
theta_post <- theta_draws[(burn_in + 1):S]
beta_post  <- beta_draws[(burn_in + 1):S]

theta_mean <- mean(theta_post)
theta_sd   <- sd(theta_post)
theta_ci   <- quantile(theta_post, probs = c(0.025, 0.975))

beta_mean  <- mean(beta_post)
beta_sd    <- sd(beta_post)
beta_ci    <- quantile(beta_post, probs = c(0.025, 0.975))

cat("=== Gibbs Sampling Posterior Summary (8,000 Retained Draws) ===\n")
cat(sprintf("theta | Mean: %.6f | SD: %.6f | 95%% CI: [%.6f, %.6f]\n",
            theta_mean, theta_sd, theta_ci[1], theta_ci[2]))
cat(sprintf("beta  | Mean: %.4f   | SD: %.4f   | 95%% CI: [%.4f, %.4f]\n",
            beta_mean, beta_sd, beta_ci[1], beta_ci[2]))

# 5. Generate and save trace plots (Q6c)
png("figures/Q6_Gibbs_Trace_Plots.png", width = 1000, height = 650, res = 130)
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

dev.off()