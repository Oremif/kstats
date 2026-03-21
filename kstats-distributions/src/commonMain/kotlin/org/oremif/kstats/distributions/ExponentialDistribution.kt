package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.ln1p
import kotlin.random.Random

/**
 * Represents the exponential distribution, a continuous probability distribution that models
 * the time between events in a Poisson process.
 *
 * The exponential distribution is commonly used to model waiting times, such as the time until
 * the next customer arrives, the time until a component fails, or the time between radioactive
 * decays. It is the continuous analog of the geometric distribution and is the only continuous
 * distribution with the memoryless property: the probability of waiting an additional amount of
 * time is independent of how long you have already waited.
 *
 * The distribution is parameterized by [rate] (often written as lambda), which is the average
 * number of events per unit time. A higher rate means events happen more frequently and the
 * distribution is concentrated closer to zero. The support is `[0, +infinity)` -- only
 * non-negative values have positive density.
 *
 * Random sampling uses the inverse CDF method, which transforms a single uniform random draw
 * into an exponentially distributed value.
 *
 * ### Example:
 * ```kotlin
 * // Model a server that handles 2 requests per second on average
 * val dist = ExponentialDistribution(rate = 2.0)
 * dist.mean                // 0.5 (average wait is 1/rate = 0.5 seconds)
 * dist.variance            // 0.25
 * dist.pdf(0.0)            // 2.0 (density is highest at zero)
 * dist.cdf(1.0)            // 0.8647 (about 86% chance the next request arrives within 1 second)
 * dist.quantile(0.5)       // 0.3466 (the median wait time)
 * dist.sample(Random(42))  // a single random wait time
 *
 * // Standard exponential (rate=1)
 * val standard = ExponentialDistribution.STANDARD
 * standard.mean             // 1.0
 * ```
 *
 * @param rate the rate parameter (lambda), representing the average number of events per unit
 * time. Must be positive. Default is `1.0`, which gives the standard exponential distribution.
 */
public class ExponentialDistribution(
    public val rate: Double = 1.0
) : ContinuousDistribution {

    init {
        if (rate <= 0.0) throw InvalidParameterException("rate must be positive, got $rate")
    }

    /**
     * Computes the probability density at [x].
     *
     * @param x the point at which to evaluate the density.
     * @return the probability density at [x], or `0.0` if [x] is negative.
     */
    override fun pdf(x: Double): Double = if (x >= 0.0) rate * exp(-rate * x) else 0.0

    /**
     * Computes the natural logarithm of the probability density at [x].
     *
     * @param x the point at which to evaluate the log-density.
     * @return the natural log of the density at [x], or [Double.NEGATIVE_INFINITY] if [x] is negative.
     */
    override fun logPdf(x: Double): Double = if (x >= 0.0) ln(rate) - rate * x else Double.NEGATIVE_INFINITY

    /**
     * Computes the cumulative distribution function at [x].
     *
     * @param x the point at which to evaluate the cumulative probability.
     * @return the probability that a value drawn from this distribution is less than or equal to [x],
     * or `0.0` if [x] is negative.
     */
    override fun cdf(x: Double): Double = if (x >= 0.0) 1.0 - exp(-rate * x) else 0.0

    /**
     * Computes the survival function at [x].
     *
     * This override computes the survival probability directly as an exponential decay rather
     * than as `1 - cdf(x)`, which avoids catastrophic cancellation when [x] is large and the
     * CDF is close to 1.
     *
     * @param x the point at which to evaluate the survival probability.
     * @return the probability that a value drawn from this distribution is greater than [x],
     * or `1.0` if [x] is negative.
     */
    override fun sf(x: Double): Double = if (x >= 0.0) exp(-rate * x) else 1.0

    /**
     * Computes the quantile (inverse CDF) for the given probability [p].
     *
     * Uses `ln1p` for numerical stability when [p] is close to zero.
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the value x at which `cdf(x) = p`. Returns [Double.POSITIVE_INFINITY] for `p = 1`.
     */
    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 1.0) return Double.POSITIVE_INFINITY
        return -ln1p(-p) / rate
    }

    /** The mean of this distribution, equal to the reciprocal of [rate]. */
    override val mean: Double get() = 1.0 / rate

    /** The variance of this distribution, equal to the reciprocal of [rate] squared. */
    override val variance: Double get() = 1.0 / (rate * rate)

    /** The skewness of this distribution, always 2 for any exponential distribution. */
    override val skewness: Double get() = 2.0

    /** The excess kurtosis of this distribution, always 6 for any exponential distribution. */
    override val kurtosis: Double get() = 6.0 // excess

    /** The differential entropy of this distribution in nats, computed from [rate]. */
    override val entropy: Double = 1.0 - ln(rate)

    /**
     * Draws a single random value from this exponential distribution using the inverse CDF method.
     *
     * @param random the source of randomness.
     * @return a non-negative random value drawn from this distribution.
     */
    override fun sample(random: Random): Double = -ln(random.nextDouble()) / rate

    /** Provides the pre-built standard exponential distribution constant. */
    public companion object {
        /** The standard exponential distribution with rate 1. */
        public val STANDARD: ExponentialDistribution = ExponentialDistribution(1.0)
    }
}
