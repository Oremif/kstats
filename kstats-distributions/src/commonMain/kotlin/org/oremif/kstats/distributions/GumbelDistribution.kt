package org.oremif.kstats.distributions

import org.oremif.kstats.core.EULER_MASCHERONI
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.*
import kotlin.random.Random

/**
 * The Gumbel distribution (extreme value type I, right-skewed), defined by its
 * location [mu] and scale [beta] parameters.
 *
 * The Gumbel distribution models the distribution of the maximum of a number of
 * samples of various distributions. It is widely used in hydrology (flood analysis),
 * meteorology (extreme weather events), and reliability engineering (failure analysis).
 * Unlike symmetric distributions, the Gumbel has a longer right tail, reflecting the
 * tendency for extreme maxima to be farther from the center than extreme minima.
 *
 * This implementation corresponds to `scipy.stats.gumbel_r` (right-skewed Gumbel).
 *
 * ### Example:
 * ```kotlin
 * val dist = GumbelDistribution(mu = 0.0, beta = 1.0)
 * dist.pdf(0.0)      // 0.3679... (peak density at the mode)
 * dist.cdf(0.0)      // 0.3679...
 * dist.quantile(0.5) // 0.3665... (median)
 * dist.mean           // 0.5772... (Euler-Mascheroni constant)
 * ```
 *
 * @param mu the location parameter (mode) of the distribution. Defaults to `0.0`.
 * @param beta the scale parameter (spread) of the distribution. Must be positive. Defaults to `1.0`.
 */
public class GumbelDistribution(
    public val mu: Double = 0.0,
    public val beta: Double = 1.0,
) : ContinuousDistribution {

    init {
        if (beta <= 0.0) throw InvalidParameterException("beta must be positive, got $beta")
    }

    /**
     * Returns the probability density at [x] for this Gumbel distribution.
     *
     * The density is computed using the standardized variable z = (x - mu) / beta as
     * exp(-(z + exp(-z))) / beta. The left tail decays doubly-exponentially (much faster
     * than the right tail), so density values near the left extreme may underflow to zero.
     *
     * @param x the point at which to evaluate the density.
     * @return the probability density at [x]. Always non-negative.
     */
    override fun pdf(x: Double): Double {
        val z = (x - mu) / beta
        return exp(-(z + exp(-z))) / beta
    }

    /**
     * Returns the natural logarithm of the probability density at [x] for this Gumbel distribution.
     *
     * Computed directly in log-space as -ln(beta) - z - exp(-z), avoiding the intermediate
     * exponentiation and providing better precision for extreme values where the density is
     * very small.
     *
     * @param x the point at which to evaluate the log-density.
     * @return the natural log of the probability density at [x].
     */
    override fun logPdf(x: Double): Double {
        val z = (x - mu) / beta
        return -ln(beta) - z - exp(-z)
    }

    /**
     * Returns the cumulative distribution function value at [x] for this Gumbel distribution.
     *
     * The CDF has the double-exponential form exp(-exp(-z)), where z = (x - mu) / beta.
     * This gives the probability that a random value from this distribution is less than
     * or equal to [x].
     *
     * @param x the point at which to evaluate the cumulative probability.
     * @return the probability that a value is less than or equal to [x], in the range `[0, 1]`.
     */
    override fun cdf(x: Double): Double {
        val z = (x - mu) / beta
        return exp(-exp(-z))
    }

    /**
     * Returns the survival function value at [x] for this Gumbel distribution.
     *
     * Computed as -expm1(-exp(-z)) to avoid catastrophic cancellation when the CDF is
     * close to 1 (i.e., in the right tail).
     *
     * @param x the point at which to evaluate the survival probability.
     * @return the probability that a value is greater than [x], in the range `[0, 1]`.
     */
    override fun sf(x: Double): Double {
        val z = (x - mu) / beta
        return -expm1(-exp(-z))
    }

    /**
     * Returns the quantile (inverse CDF) for the given probability [p].
     *
     * The quantile is computed as mu - beta * ln(-ln(p)). Returns negative infinity for
     * p = 0 and positive infinity for p = 1.
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the value x at which `cdf(x) = p`.
     */
    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return Double.NEGATIVE_INFINITY
        if (p == 1.0) return Double.POSITIVE_INFINITY
        return mu - beta * ln(-ln(p))
    }

    /** The mean of this distribution, equal to mu + beta * gamma (Euler-Mascheroni constant). */
    override val mean: Double get() = mu + beta * EULER_MASCHERONI

    /** The variance of this distribution, equal to (pi * beta)^2 / 6. */
    override val variance: Double get() = PI * PI * beta * beta / 6.0

    /** The standard deviation of this distribution, equal to pi * beta / sqrt(6). */
    override val standardDeviation: Double get() = PI * beta / sqrt(6.0)

    /** The skewness of this distribution, a positive constant approximately equal to 1.1395. */
    override val skewness: Double get() = 1.1395470994717452

    /** The excess kurtosis of this distribution, always 12/5 = 2.4. */
    override val kurtosis: Double get() = 2.4

    /** The Shannon entropy of this distribution in nats, equal to ln(beta) + 1 + gamma. */
    override val entropy: Double = ln(beta) + 1.0 + EULER_MASCHERONI

    /**
     * Draws a single random value from this Gumbel distribution using inverse CDF sampling.
     *
     * @param random the source of randomness.
     * @return a random value drawn from this distribution.
     */
    override fun sample(random: Random): Double {
        val u = random.nextDouble().coerceIn(Double.MIN_VALUE, 1.0 - Double.MIN_VALUE)
        return mu - beta * ln(-ln(u))
    }

    public companion object {
        /** Standard Gumbel distribution with mu = 0 and beta = 1. */
        public val STANDARD: GumbelDistribution = GumbelDistribution(0.0, 1.0)
    }
}
