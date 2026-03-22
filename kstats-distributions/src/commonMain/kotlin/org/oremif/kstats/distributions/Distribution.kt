package org.oremif.kstats.distributions

import kotlin.math.sqrt

/**
 * Represents the common sealed interface for all probability distributions in kstats.
 *
 * Every probability distribution — whether continuous or discrete — extends this interface,
 * which provides shared statistical properties such as the mean, variance, standard deviation,
 * skewness, kurtosis, and entropy. It also declares the cumulative distribution function (CDF),
 * the survival function (SF), and the quantile (inverse CDF) function.
 *
 * Concrete distributions do not implement this interface directly. Instead, they implement
 * either [ContinuousDistribution] or [DiscreteDistribution], both of which extend
 * [Distribution] with type-specific methods like `pdf`/`pmf` and `sample`.
 *
 * ### Example:
 * ```kotlin
 * val dist: Distribution = NormalDistribution(mu = 0.0, sigma = 1.0)
 * dist.mean                // 0.0
 * dist.variance            // 1.0
 * dist.standardDeviation   // 1.0
 * dist.cdf(0.0)            // 0.5 (half the probability mass is below the mean)
 * dist.sf(0.0)             // 0.5 (survival function, complement of cdf)
 * dist.quantile(0.975)     // 1.96 (the value below which 97.5% of the mass lies)
 * ```
 *
 * @see ContinuousDistribution for distributions defined over real-valued intervals.
 * @see DiscreteDistribution for distributions defined over integer-valued outcomes.
 */
public sealed interface Distribution {

    /** The expected value (first moment) of this distribution. */
    public val mean: Double

    /** The variance (second central moment) of this distribution, measuring how spread out values are around the [mean]. */
    public val variance: Double

    /** The standard deviation of this distribution, equal to the square root of [variance]. */
    public val standardDeviation: Double get() = sqrt(variance)

    /** The skewness (third standardized moment) of this distribution, measuring asymmetry around the [mean]. Zero for symmetric distributions, positive for right-skewed, negative for left-skewed. */
    public val skewness: Double

    /** The excess kurtosis (fourth standardized moment minus 3) of this distribution, measuring the heaviness of the tails relative to a normal distribution. Zero for normal, positive for heavier tails. */
    public val kurtosis: Double

    /** The Shannon entropy of this distribution in nats (natural logarithm units), measuring the uncertainty or spread of the distribution. */
    public val entropy: Double

    /**
     * Computes the cumulative distribution function (CDF) at the given point.
     *
     * The CDF returns the probability that a random variable drawn from this distribution
     * takes a value less than or equal to [x].
     *
     * @param x the point at which to evaluate the cumulative probability.
     * @return the probability that a value is less than or equal to [x], in the range `[0, 1]`.
     */
    public fun cdf(x: Double): Double

    /**
     * Computes the survival function (SF) at the given point.
     *
     * The survival function returns the probability that a random variable drawn from this
     * distribution takes a value strictly greater than [x]. It is the complement of the CDF,
     * that is, `sf(x) = 1 - cdf(x)`. Some distributions override this default to provide a
     * numerically more accurate result in the upper tail.
     *
     * @param x the point at which to evaluate the survival probability.
     * @return the probability that a value is greater than [x], in the range `[0, 1]`.
     */
    public fun sf(x: Double): Double = 1.0 - cdf(x)

    /**
     * Computes the quantile (inverse CDF) for the given cumulative probability.
     *
     * Returns the smallest value x such that `cdf(x) >= p`. This is the inverse of the
     * cumulative distribution function: given a probability, it returns the corresponding
     * threshold value.
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the value x at which `cdf(x) = p`.
     * @throws org.oremif.kstats.core.exceptions.InvalidParameterException if [p] is not in `[0, 1]`.
     */
    public fun quantile(p: Double): Double
}
