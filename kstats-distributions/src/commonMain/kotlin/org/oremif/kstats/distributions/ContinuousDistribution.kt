package org.oremif.kstats.distributions

import kotlin.random.Random

/**
 * Common interface for continuous probability distributions.
 *
 * A continuous distribution assigns probabilities to intervals of real numbers via a
 * probability density function (PDF). Implementations provide methods to evaluate the
 * density, log-density, cumulative probability, quantiles, and random sampling.
 *
 * Extends [Distribution], which provides shared statistical properties such as [mean],
 * [variance], [standardDeviation], [skewness], [kurtosis], and the [sf] survival function.
 *
 * ### Example:
 * ```kotlin
 * val dist: ContinuousDistribution = NormalDistribution(mu = 0.0, sigma = 1.0)
 * dist.pdf(0.0)              // 0.3989... (peak density at the mean)
 * dist.logPdf(0.0)           // -0.9189... (log of the density)
 * dist.cdf(1.96)             // 0.975... (area under the curve up to 1.96)
 * dist.quantile(0.975)       // 1.96 (inverse of cdf)
 * dist.sample(Random(42))    // a single random draw
 * dist.sample(5, Random(42)) // five random draws
 * ```
 *
 * @see Distribution for inherited statistical properties and survival function.
 */
public interface ContinuousDistribution : Distribution {

    /**
     * Returns the probability density at [x].
     *
     * The probability density indicates how likely values near [x] are relative to other
     * values. Higher density means more probable. The density can exceed 1.0 for narrow
     * distributions, but integrates to 1.0 over the entire support.
     *
     * @param x the point at which to evaluate the density.
     * @return the probability density at [x]. Always non-negative; zero for values outside the support.
     */
    public fun pdf(x: Double): Double

    /**
     * Returns the natural logarithm of the probability density at [x].
     *
     * The log-density is useful when working with very small density values that would
     * underflow to zero in regular floating-point arithmetic. Every implementation provides
     * a direct formula rather than computing `ln(pdf(x))`, which avoids unnecessary
     * precision loss.
     *
     * @param x the point at which to evaluate the log-density.
     * @return the natural log of the probability density at [x]. Returns [Double.NEGATIVE_INFINITY]
     * when the density is zero.
     */
    public fun logPdf(x: Double): Double

    /**
     * Returns the cumulative distribution function (CDF) value at [x].
     *
     * The CDF gives the probability that a random variable drawn from this distribution
     * is less than or equal to [x]. Equivalently, it is the area under the probability
     * density curve from negative infinity up to [x].
     *
     * @param x the point at which to evaluate the cumulative probability.
     * @return the probability that a value is less than or equal to [x], in the range `[0, 1]`.
     */
    override fun cdf(x: Double): Double

    /**
     * Returns the quantile (inverse CDF) for the given probability [p].
     *
     * The quantile function returns the smallest value [x] such that `cdf(x) >= p`.
     * It is the inverse of the cumulative distribution function.
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the value [x] at which `cdf(x) = p`.
     */
    override fun quantile(p: Double): Double

    /**
     * Returns the Shannon entropy of this distribution in nats (natural logarithm units).
     *
     * Entropy measures the uncertainty or "spread" of a distribution. Higher entropy means
     * more uncertainty. For continuous distributions, this is the differential entropy,
     * which can be negative.
     *
     * Returns [Double.NaN] by default. Implementations override this with the exact formula
     * for each distribution.
     *
     * @return the differential entropy in nats, or [Double.NaN] if not yet implemented.
     */
    override val entropy: Double get() = Double.NaN

    /**
     * Draws a single random value from this distribution.
     *
     * @param random the source of randomness.
     * @return a random value drawn from this distribution.
     */
    public fun sample(random: Random): Double

    /**
     * Draws [n] independent random values from this distribution.
     *
     * Each element in the returned array is drawn independently using [sample].
     *
     * @param n the number of values to draw. Must be non-negative.
     * @param random the source of randomness.
     * @return a [DoubleArray] of [n] independent random draws.
     */
    public fun sample(n: Int, random: Random): DoubleArray = DoubleArray(n) { sample(random) }
}
