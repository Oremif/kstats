package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.*
import kotlin.random.Random

/**
 * Represents the Laplace distribution (also known as the double exponential distribution),
 * defined by its location [mu] and [scale] parameters.
 *
 * The Laplace distribution is a symmetric, peaked distribution centered at [mu] with
 * exponentially decaying tails on both sides. Compared to the normal distribution with
 * the same variance, the Laplace has a sharper peak at the center and heavier tails,
 * meaning extreme values are more likely. This makes it useful in robust statistics
 * where outliers are common, in signal processing for modeling Laplacian noise, and
 * in Bayesian inference as a sparsity-promoting prior (the Lasso penalty in regression
 * corresponds to a Laplace prior). The support is the entire real line.
 *
 * ### Example:
 * ```kotlin
 * val dist = LaplaceDistribution(mu = 0.0, scale = 1.0)
 * dist.pdf(0.0)      // 0.5 (peak density at the center)
 * dist.cdf(0.0)      // 0.5 (symmetric around mu)
 * dist.quantile(0.75) // 0.6931... (third quartile)
 * dist.mean           // 0.0
 * dist.variance       // 2.0
 * ```
 *
 * @param mu the location parameter (center and mode) of the distribution. Defaults to `0.0`.
 * @param scale the scale parameter controlling the spread. Must be positive. Defaults to `1.0`.
 *   Larger values produce wider, flatter distributions.
 */
public class LaplaceDistribution(
    public val mu: Double = 0.0,
    public val scale: Double = 1.0,
) : ContinuousDistribution {

    init {
        if (!mu.isFinite()) throw InvalidParameterException("mu must be finite, got $mu")
        if (!(scale > 0.0)) throw InvalidParameterException("scale must be positive, got $scale")
    }

    public companion object {
        /** Standard Laplace distribution with mu = 0 and scale = 1. */
        public val STANDARD: LaplaceDistribution = LaplaceDistribution(0.0, 1.0)
    }

    /**
     * Returns the probability density at [x] for this Laplace distribution.
     *
     * The density is highest at [mu] (where it equals 1 / (2 * scale)) and decays
     * exponentially on both sides. Unlike the normal distribution, the decay is purely
     * exponential rather than Gaussian, which produces the characteristic sharp peak.
     *
     * @param x the point at which to evaluate the density.
     * @return the probability density at [x]. Always non-negative.
     */
    override fun pdf(x: Double): Double {
        return exp(-abs(x - mu) / scale) / (2.0 * scale)
    }

    /**
     * Returns the natural logarithm of the probability density at [x] for this Laplace distribution.
     *
     * Computed directly in log-space as -|x - mu| / scale - ln(2 * scale), avoiding
     * intermediate exponentiation and providing better precision for extreme values
     * where the density is very small.
     *
     * @param x the point at which to evaluate the log-density.
     * @return the natural log of the probability density at [x].
     */
    override fun logPdf(x: Double): Double {
        return -abs(x - mu) / scale - ln(2.0 * scale)
    }

    /**
     * Returns the cumulative distribution function value at [x] for this Laplace distribution.
     *
     * The CDF has a piecewise form: for values at or below [mu] it equals 0.5 * exp(z),
     * and for values above [mu] it equals 1 - 0.5 * exp(-z), where z = (x - mu) / scale.
     * This gives the probability that a random value from this distribution is less than
     * or equal to [x].
     *
     * @param x the point at which to evaluate the cumulative probability.
     * @return the probability that a value is less than or equal to [x], in the range `[0, 1]`.
     */
    override fun cdf(x: Double): Double {
        val z = (x - mu) / scale
        return if (z <= 0.0) 0.5 * exp(z) else 1.0 - 0.5 * exp(-z)
    }

    /**
     * Returns the survival function value at [x] for this Laplace distribution.
     *
     * The survival function is 1 - cdf(x), computed using the complementary piecewise
     * formula to avoid catastrophic cancellation when the CDF is close to 1.
     *
     * @param x the point at which to evaluate the survival probability.
     * @return the probability that a value is greater than [x], in the range `[0, 1]`.
     */
    override fun sf(x: Double): Double {
        val z = (x - mu) / scale
        return if (z <= 0.0) 1.0 - 0.5 * exp(z) else 0.5 * exp(-z)
    }

    /**
     * Returns the quantile (inverse CDF) for the given probability [p].
     *
     * The quantile function inverts the piecewise CDF. For probabilities up to 0.5,
     * the result lies at or below [mu]; for probabilities above 0.5, the result lies
     * above [mu]. Returns negative infinity for p = 0 and positive infinity for p = 1.
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the value x at which `cdf(x) = p`.
     */
    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return Double.NEGATIVE_INFINITY
        if (p == 1.0) return Double.POSITIVE_INFINITY
        return if (p <= 0.5) mu + scale * ln(2.0 * p) else mu - scale * ln(2.0 * (1.0 - p))
    }

    /** The mean of this distribution, equal to [mu]. */
    override val mean: Double get() = mu

    /** The variance of this distribution, equal to 2 * scale^2. */
    override val variance: Double get() = 2.0 * scale * scale

    /** The standard deviation of this distribution, equal to scale * sqrt(2). */
    override val standardDeviation: Double get() = scale * sqrt(2.0)

    /** The skewness of this distribution, always zero due to symmetry around [mu]. */
    override val skewness: Double get() = 0.0

    /** The excess kurtosis of this distribution, always 3.0 (leptokurtic, heavier tails than normal). */
    override val kurtosis: Double get() = 3.0 // excess kurtosis

    /** The Shannon entropy of this distribution in nats, equal to 1 + ln(2 * scale). */
    override val entropy: Double = 1.0 + ln(2.0 * scale)

    /**
     * Draws a single random value from this Laplace distribution using inverse CDF sampling.
     *
     * Generates a uniform random variable shifted to the range (-0.5, 0.5) and applies
     * the inverse CDF transform to produce a Laplace-distributed value.
     *
     * @param random the source of randomness.
     * @return a random value drawn from this distribution.
     */
    override fun sample(random: Random): Double {
        val u = random.nextDouble().coerceAtLeast(Double.MIN_VALUE) - 0.5
        return mu - scale * sign(u) * ln1p(-2.0 * abs(u))
    }
}
