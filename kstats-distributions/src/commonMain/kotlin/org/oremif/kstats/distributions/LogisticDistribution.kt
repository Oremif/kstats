package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.*
import kotlin.random.Random

/**
 * The logistic distribution, defined by its location [mu] and [scale] parameter.
 *
 * The logistic distribution is a symmetric, bell-shaped continuous distribution similar
 * to the normal distribution but with heavier tails. It is widely used in logistic
 * regression, modeling growth curves, and as the distribution of the log-odds in
 * binary classification. Its CDF is the sigmoid function.
 *
 * ### Example:
 * ```kotlin
 * val dist = LogisticDistribution(mu = 0.0, scale = 1.0)
 * dist.pdf(0.0)      // 0.25 (peak density at the mean)
 * dist.cdf(0.0)      // 0.5
 * dist.quantile(0.75) // 1.0986... (log-odds)
 * ```
 *
 * @param mu the location (center) of the distribution. Defaults to `0.0`.
 * @param scale the scale (spread) of the distribution. Must be positive. Defaults to `1.0`.
 */
public class LogisticDistribution(
    public val mu: Double = 0.0,
    public val scale: Double = 1.0,
) : ContinuousDistribution {

    init {
        if (scale <= 0.0) throw InvalidParameterException("scale must be positive, got $scale")
    }

    /**
     * Returns the probability density at [x] for this logistic distribution.
     *
     * Uses the absolute-value form of the standardized variable to avoid overflow
     * from large exponentials.
     *
     * @param x the point at which to evaluate the density.
     * @return the probability density at [x]. Always non-negative.
     */
    override fun pdf(x: Double): Double {
        val z = abs((x - mu) / scale)
        val expNegZ = exp(-z)
        val denom = 1.0 + expNegZ
        return expNegZ / (scale * denom * denom)
    }

    /**
     * Returns the natural logarithm of the probability density at [x] for this logistic distribution.
     *
     * Computed directly in log-space for improved numerical precision with extreme values.
     *
     * @param x the point at which to evaluate the log-density.
     * @return the natural log of the probability density at [x].
     */
    override fun logPdf(x: Double): Double {
        val z = abs((x - mu) / scale)
        return -z - ln(scale) - 2.0 * ln(1.0 + exp(-z))
    }

    /**
     * Returns the cumulative distribution function value at [x] for this logistic distribution.
     *
     * The CDF is the standard sigmoid function applied to the standardized variable.
     * Uses a two-branch formulation to avoid overflow for both large positive and
     * large negative inputs.
     *
     * @param x the point at which to evaluate the cumulative probability.
     * @return the probability that a value is less than or equal to [x], in the range `[0, 1]`.
     */
    override fun cdf(x: Double): Double {
        val z = (x - mu) / scale
        return if (z >= 0.0) 1.0 / (1.0 + exp(-z)) else exp(z) / (1.0 + exp(z))
    }

    /**
     * Returns the survival function value at [x] for this logistic distribution.
     *
     * Computed as the complement of [cdf] using a numerically stable two-branch formula
     * that avoids catastrophic cancellation.
     *
     * @param x the point at which to evaluate the survival probability.
     * @return the probability that a value is greater than [x], in the range `[0, 1]`.
     */
    override fun sf(x: Double): Double {
        val z = (x - mu) / scale
        return if (z >= 0.0) exp(-z) / (1.0 + exp(-z)) else 1.0 / (1.0 + exp(z))
    }

    /**
     * Returns the quantile (inverse CDF) for the given probability [p].
     *
     * The quantile is computed as the log-odds of [p], scaled and shifted by the
     * distribution parameters. Returns negative infinity for p=0 and positive infinity for p=1.
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the value x at which `cdf(x) = p`.
     */
    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return Double.NEGATIVE_INFINITY
        if (p == 1.0) return Double.POSITIVE_INFINITY
        return mu + scale * ln(p / (1.0 - p))
    }

    /** The mean of this distribution, equal to [mu]. */
    override val mean: Double get() = mu

    /** The variance of this distribution, equal to (pi * scale)^2 / 3. */
    override val variance: Double get() = PI * PI * scale * scale / 3.0

    /** The standard deviation of this distribution, equal to pi * scale / sqrt(3). */
    override val standardDeviation: Double get() = PI * scale / sqrt(3.0)

    /** The skewness of this distribution, always zero (symmetric). */
    override val skewness: Double get() = 0.0

    /** The excess kurtosis of this distribution, always 6/5 = 1.2. */
    override val kurtosis: Double get() = 1.2

    /** The Shannon entropy of this distribution in nats, equal to ln(scale) + 2. */
    override val entropy: Double = ln(scale) + 2.0

    /**
     * Draws a single random value from this logistic distribution using inverse CDF sampling.
     *
     * @param random the source of randomness.
     * @return a random value drawn from this distribution.
     */
    override fun sample(random: Random): Double {
        val u = random.nextDouble().coerceIn(1e-15, 1.0 - 1e-15)
        return mu + scale * ln(u / (1.0 - u))
    }

    public companion object {
        /** Standard logistic distribution with mu=0 and scale=1. */
        public val STANDARD: LogisticDistribution = LogisticDistribution(0.0, 1.0)
    }
}
