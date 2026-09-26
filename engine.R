# engine.R
# Master Inference Module

run_t_test <- function(data_vector, mu_val = 0, conf_level = 0.95, alt = "two.sided") {
    # Execute the exact base R statistical test
    result <- t.test(data_vector, mu = mu_val, conf.level = conf_level, alternative = alt)
    
    # Return a structured list so Python can parse the exact values
    return(list(
        statistic = as.numeric(result$statistic),
        p_value = as.numeric(result$p.value),
        df = as.numeric(result$parameter),
        mean_est = as.numeric(result$estimate),
        conf_lower = as.numeric(result$conf.int[1]),
        conf_upper = as.numeric(result$conf.int[2])
    ))
}
# engine.R
# Master Inference Module
run_t_test <- function(data_vector, mu_val = 0, conf_level = 0.95, alt = "two.sided") {
    result <- t.test(data_vector, mu = mu_val, conf.level = conf_level, alternative = alt)
    
    return(list(
        statistic = as.numeric(result$statistic),
        p_value = as.numeric(result$p.value),
        df = as.numeric(result$parameter),
        mean_est = as.numeric(result$estimate),
        conf_lower = as.numeric(result$conf.int[1]),
        conf_upper = as.numeric(result$conf.int[2])
    ))
}

# The Distribution Module
calc_binom_prob <- function(k, n, p, lower_tail = TRUE) {
    # Computes P(X <= k) or P(X > k)
    prob <- pbinom(k, size = n, prob = p, lower.tail = lower_tail)
    return(as.numeric(prob))
}

calc_norm_prob <- function(x, mu, sigma, lower_tail = TRUE) {
    # Computes standard continuous normal probabilities
    prob <- pnorm(x, mean = mu, sd = sigma, lower.tail = lower_tail)
    return(as.numeric(prob))
}
# The Regression Module
run_simple_regression <- function(x_vector, y_vector) {
    model <- lm(y_vector ~ x_vector)
    s <- summary(model)
    
    return(list(
        intercept = as.numeric(coef(model)[1]),
        slope = as.numeric(coef(model)[2]),
        r_squared = as.numeric(s$r.squared),
        p_value = as.numeric(coef(s)[2, 4]) # p-value for the slope
    ))
}