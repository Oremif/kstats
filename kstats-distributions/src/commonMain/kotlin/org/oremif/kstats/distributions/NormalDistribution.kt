package org.oremif.kstats.distributions

import org.oremif.kstats.core.erf
import org.oremif.kstats.core.erfInv
import org.oremif.kstats.core.erfc
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.*
import kotlin.random.Random

/**
 * Represents the normal (Gaussian) distribution, the most widely used continuous probability
 * distribution in statistics.
 *
 * The normal distribution is symmetric and bell-shaped, centered at the mean [mu] with spread
 * controlled by the standard deviation [sigma]. It arises naturally as the limiting distribution
 * of sums of independent random variables (central limit theorem) and is the foundation for
 * many statistical methods including confidence intervals, hypothesis tests, and regression.
 *
 * About 68% of values fall within one standard deviation of the mean, about 95% within two,
 * and about 99.7% within three. The distribution is fully supported on the entire real line,
 * meaning any real number is a possible outcome, though values far from the mean are
 * exceedingly rare.
 *
 * The CDF and survival function use the error function for numerical accuracy, and the quantile
 * function uses the inverse error function. Random sampling uses the Box-Muller transform, which
 * converts two uniform random draws into a normally distributed value.
 *
 * ### Example:
 * ```kotlin
 * val dist = NormalDistribution(mu = 100.0, sigma = 15.0)
 * dist.mean                // 100.0
 * dist.variance            // 225.0
 * dist.pdf(100.0)          // 0.0266 (peak density at the mean)
 * dist.cdf(115.0)          // 0.8413 (about 84% of values are below one SD above the mean)
 * dist.quantile(0.975)     // 129.39 (the 97.5th percentile)
 * dist.sample(Random(42))  // a single random draw from N(100, 15)
 *
 * // Standard normal (mu=0, sigma=1)
 * val z = NormalDistribution.STANDARD
 * z.cdf(1.96)              // 0.975
 * ```
 *
 * @param mu the mean (location parameter) of the distribution. Default is `0.0`, which centers
 * the distribution at the origin.
 * @param sigma the standard deviation (scale parameter) of the distribution. Must be positive.
 * Default is `1.0`, which gives the standard normal distribution.
 */
public class NormalDistribution(
    public val mu: Double = 0.0,
    public val sigma: Double = 1.0
) : ContinuousDistribution {

    init {
        if (!mu.isFinite()) throw InvalidParameterException("mu must be finite, got $mu")
        if (!sigma.isFinite() || sigma <= 0.0) throw InvalidParameterException("sigma must be finite and positive, got $sigma")
    }

    /** Provides the pre-built standard normal distribution constant. */
    public companion object {
        /** The standard normal distribution with mean 0 and standard deviation 1. */
        public val STANDARD: NormalDistribution = NormalDistribution(0.0, 1.0)
    }

    /**
     * Computes the probability density at [x].
     *
     * @param x the point at which to evaluate the density.
     * @return the probability density at [x], always non-negative.
     */
    override fun pdf(x: Double): Double {
        val z = (x - mu) / sigma
        return exp(-0.5 * z * z) / (sigma * sqrt(2.0 * PI))
    }

    /**
     * Computes the natural logarithm of the probability density at [x].
     *
     * @param x the point at which to evaluate the log-density.
     * @return the natural log of the density at [x].
     */
    override fun logPdf(x: Double): Double {
        val z = (x - mu) / sigma
        return -0.5 * z * z - ln(sigma) - 0.5 * ln(2.0 * PI)
    }

    /**
     * Computes the cumulative distribution function at [x] using the error function.
     *
     * @param x the point at which to evaluate the cumulative probability.
     * @return the probability that a value drawn from this distribution is less than or equal to [x].
     */
    override fun cdf(x: Double): Double {
        return 0.5 * (1.0 + erf((x - mu) / (sigma * sqrt(2.0))))
    }

    /**
     * Computes the survival function at [x] using the complementary error function.
     *
     * This override uses `erfc` directly rather than `1 - cdf(x)`, which provides better
     * numerical accuracy in the upper tail where the CDF is close to 1.
     *
     * @param x the point at which to evaluate the survival probability.
     * @return the probability that a value drawn from this distribution is greater than [x].
     */
    override fun sf(x: Double): Double {
        return 0.5 * erfc((x - mu) / (sigma * sqrt(2.0)))
    }

    /**
     * Computes the quantile (inverse CDF) for the given probability [p] using the inverse
     * error function.
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the value x at which `cdf(x) = p`. Returns [Double.NEGATIVE_INFINITY] for
     * `p = 0` and [Double.POSITIVE_INFINITY] for `p = 1`.
     */
    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return Double.NEGATIVE_INFINITY
        if (p == 1.0) return Double.POSITIVE_INFINITY
        return mu + sigma * sqrt(2.0) * erfInv(2.0 * p - 1.0)
    }

    /** The mean of this distribution, equal to [mu]. */
    override val mean: Double get() = mu

    /** The variance of this distribution, equal to [sigma] squared. */
    override val variance: Double get() = sigma * sigma

    /** The standard deviation of this distribution, equal to [sigma]. */
    override val standardDeviation: Double get() = sigma

    /** The skewness of this distribution, always zero because the normal distribution is symmetric. */
    override val skewness: Double get() = 0.0

    /** The excess kurtosis of this distribution, always zero by definition (the normal distribution is the reference). */
    override val kurtosis: Double get() = 0.0 // excess kurtosis

    /** The differential entropy of this distribution in nats, computed from [sigma]. */
    override val entropy: Double = 0.5 * ln(2.0 * PI * E * sigma * sigma)

    /**
     * Draws a single random value from this normal distribution using the Box-Muller transform.
     *
     * The Box-Muller transform converts two independent uniform random numbers into a pair of
     * independent standard normal random values. One of those values is then scaled by [sigma]
     * and shifted by [mu].
     *
     * @param random the source of randomness.
     * @return a random value drawn from this distribution.
     */
    override fun sample(random: Random): Double {
        // Box-Muller transform (guard against u1=0 which would produce ln(0)=-∞)
        val u1 = random.nextDouble().coerceAtLeast(Double.MIN_VALUE)
        val u2 = random.nextDouble()
        return mu + sigma * sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)
    }
}
