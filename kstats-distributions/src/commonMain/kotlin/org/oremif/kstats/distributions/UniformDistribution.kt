package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.ln
import kotlin.random.Random

/**
 * Represents the continuous uniform distribution, where all values in the interval `[min, max]`
 * are equally likely.
 *
 * The uniform distribution assigns constant probability density to every point within its
 * support and zero density outside it. It is the simplest continuous distribution and is often
 * used as a baseline or "uninformative" prior in Bayesian statistics, for random number
 * generation, and in simulation when every outcome in a range should be equally probable.
 *
 * The distribution is parameterized by [min] and [max], which define the lower and upper
 * bounds of the support. The probability density is constant and equal to `1 / (max - min)`
 * throughout the interval. The CDF increases linearly from 0 at [min] to 1 at [max].
 *
 * ### Example:
 * ```kotlin
 * // Model a random arrival time uniformly distributed between 0 and 60 minutes
 * val dist = UniformDistribution(min = 0.0, max = 60.0)
 * dist.mean                // 30.0
 * dist.variance            // 300.0
 * dist.pdf(30.0)           // 0.0167 (constant density across the interval)
 * dist.cdf(15.0)           // 0.25 (25% chance of arriving in the first quarter)
 * dist.quantile(0.5)       // 30.0 (the median equals the mean for a symmetric distribution)
 * dist.sample(Random(42))  // a single random arrival time in [0, 60]
 *
 * // Standard uniform on [0, 1]
 * val standard = UniformDistribution.STANDARD
 * standard.cdf(0.5)        // 0.5
 * ```
 *
 * @param min the lower bound of the distribution's support. Default is `0.0`.
 * @param max the upper bound of the distribution's support. Must be strictly greater than [min].
 * Default is `1.0`, which gives the standard uniform distribution on `[0, 1]`.
 */
public data class UniformDistribution(
    val min: Double = 0.0,
    val max: Double = 1.0
) : ContinuousDistribution {

    init {
        if (min >= max) throw InvalidParameterException("min must be less than max, got min=$min, max=$max")
    }

    private val range = max - min

    /**
     * Computes the probability density at [x].
     *
     * Returns the constant density `1 / (max - min)` if [x] is within the support
     * `[min, max]`, or `0.0` otherwise.
     *
     * @param x the point at which to evaluate the density.
     * @return the probability density at [x].
     */
    override fun pdf(x: Double): Double = if (x in min..max) 1.0 / range else 0.0

    /**
     * Computes the natural logarithm of the probability density at [x].
     *
     * @param x the point at which to evaluate the log-density.
     * @return the natural log of the density at [x], or [Double.NEGATIVE_INFINITY] if [x] is
     * outside the support.
     */
    override fun logPdf(x: Double): Double = if (x in min..max) -ln(range) else Double.NEGATIVE_INFINITY

    /**
     * Computes the cumulative distribution function at [x].
     *
     * The CDF increases linearly from 0 at [min] to 1 at [max].
     *
     * @param x the point at which to evaluate the cumulative probability.
     * @return the probability that a value drawn from this distribution is less than or equal to [x].
     */
    override fun cdf(x: Double): Double = when {
        x <= min -> 0.0
        x >= max -> 1.0
        else -> (x - min) / range
    }

    /**
     * Computes the quantile (inverse CDF) for the given probability [p].
     *
     * Because the CDF is linear, the quantile is a simple linear interpolation between
     * [min] and [max].
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the value x at which `cdf(x) = p`.
     */
    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        return min + p * range
    }

    /** The mean of this distribution, equal to the midpoint of [min] and [max]. */
    override val mean: Double get() = (min + max) / 2.0

    /** The variance of this distribution, equal to the squared range divided by 12. */
    override val variance: Double get() = range * range / 12.0

    /** The skewness of this distribution, always zero because the uniform distribution is symmetric. */
    override val skewness: Double get() = 0.0

    /** The excess kurtosis of this distribution, always -1.2 for any continuous uniform distribution. */
    override val kurtosis: Double get() = -6.0 / 5.0 // excess

    /** The differential entropy of this distribution in nats, equal to the natural log of the range. */
    override val entropy: Double = ln(range)

    /**
     * Draws a single random value uniformly from the interval `[min, max]`.
     *
     * @param random the source of randomness.
     * @return a random value in `[min, max]`.
     */
    override fun sample(random: Random): Double = min + random.nextDouble() * range

    /** Provides the pre-built standard uniform distribution constant. */
    public companion object {
        /** The standard uniform distribution on the interval `[0, 1]`. */
        public val STANDARD: UniformDistribution = UniformDistribution(0.0, 1.0)
    }
}
