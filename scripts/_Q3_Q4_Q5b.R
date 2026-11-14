
# STEP 1: Load the insurance dataset

url <- "https://huggingface.co/datasets/mabilton/fremtpl2/resolve/main/freMTPL2freq.csv"

df <- read.csv(url)

# Check the dataset
dim(df)
head(df)

# STEP 2: Select comparable policies
# Approximately one year of insurance
# Drivers aged 30-60, vehicles aged up to 10 years

pool <- subset(
  df,
  Exposure >= 0.99 &
    Exposure <= 1.00 &
    DrivAge >= 30 &
    DrivAge <= 60 &
    VehAge <= 10
)

# STEP 3: Examine the observations

cat("Number of selected policies:", nrow(pool), "\n")

cat("Distribution of claim counts:\n")
print(table(pool$ClaimNb))

cat("Average claims:", mean(pool$ClaimNb), "\n")
cat("Variance:", var(pool$ClaimNb), "\n")


# STEP 4: Select our final analysis sample

set.seed(2026)

analysis_sample <- pool[
  sample(nrow(pool), 200),
]

# Check our final sample
cat("Sample size:", nrow(analysis_sample), "\n")

cat("Claims distribution:\n")
print(table(analysis_sample$ClaimNb))

cat("Mean:", mean(analysis_sample$ClaimNb), "\n")
cat("Variance:", var(analysis_sample$ClaimNb), "\n")


# Compare actual claim counts with Poisson predictions

x <- analysis_sample$ClaimNb
lambda <- mean(x)
n <- length(x)

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
  Claims = c("0", "1", "2", "3+"),
  Observed = observed,
  Poisson = round(expected, 2)
)

print(comparison)

# Create the graph
barplot(
  t(as.matrix(comparison[, 2:3])),
  beside = TRUE,
  names.arg = comparison$Claims,
  col = c("steelblue", "orange"),
  legend.text = c("Observed", "Poisson"),
  main = "Observed vs Poisson Claim Frequencies",
  xlab = "Number of claims per policy",
  ylab = "Number of policies"
)


write.csv(
  analysis_sample,
  "insurance_analysis_sample_200.csv",
  row.names = FALSE
)


# Q4: Selecting the Gamma prior

# Reference data: exclude our 200 analysis policies
reference <- pool[
  !(pool$IDpol %in% analysis_sample$IDpol),
]

# Calculate the reference claim frequency
n_ref <- nrow(reference)
claims_ref <- sum(reference$ClaimNb)
mean_ref <- mean(reference$ClaimNb)

cat("Reference policies:", n_ref, "\n")
cat("Reference claims:", claims_ref, "\n")
cat("Reference average:", mean_ref, "\n")

# Choose a weakly informative Gamma prior
alpha <- 3
beta <- 40

# Gamma(shape, rate) prior statistics
prior_mean <- alpha / beta
prior_sd <- sqrt(alpha) / beta

cat("Alpha:", alpha, "\n")
cat("Beta:", beta, "\n")
cat("Prior mean:", prior_mean, "\n")
cat("Prior standard deviation:", prior_sd, "\n")

# Analytical posterior mean for comparison
n <- nrow(analysis_sample)
total_claims <- sum(analysis_sample$ClaimNb)

posterior_mean <- (alpha + total_claims) / (beta + n)

cat("Posterior mean:", posterior_mean, "\n")



# Q5(b): Importance Sampling

set.seed(2026)

# Number of simulations
M <- 10000

# Draw 10,000 values from our Gamma prior
theta_draws <- rgamma(
  M,
  shape = alpha,
  rate = beta
)

# Calculate log-likelihood for each draw
# Constants not depending on theta are omitted
log_weights <- total_claims * log(theta_draws) -
  n * theta_draws

# Convert log-weights into normalized weights
# Subtracting the maximum improves numerical stability
weights <- exp(log_weights - max(log_weights))
weights <- weights / sum(weights)

# Importance sampling estimate
IS_mean <- sum(weights * theta_draws)

# Analytical posterior mean from Q1
analytical_mean <- (alpha + total_claims) /
  (beta + n)

# Effective sample size
ESS <- 1 / sum(weights^2)

# Display results
cat("Importance Sampling Mean:", IS_mean, "\n")
cat("Analytical Posterior Mean:", analytical_mean, "\n")
cat("Absolute Difference:",
    abs(IS_mean - analytical_mean), "\n")
cat("Effective Sample Size:", ESS, "\n")
