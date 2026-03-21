package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.floor
import kotlin.random.Random

/**
 * Common interface for discrete probability distributions.
 *
 * A discrete distribution assigns probabilities to individual integer outcomes via a
 * probability mass function (PMF). Implementations provide methods to evaluate the
 * mass, log-mass, cumulative probability, quantiles, and random sampling.
 *
 * Extends [Distribution], which provides shared statistical properties such as [mean],
 * [variance], [standardDeviation], and the [sf] survival function. The [cdf] and [sf]
 * methods are available in both `Int` and `Double` overloads — the `Double` variants
 * delegate to the `Int` versions via truncation.
 *
 * ### Example:
 * ```kotlin
 * val dist: DiscreteDistribution = BinomialDistribution(trials = 10, probability = 0.3)
 * dist.pmf(3)                  // 0.2668... (probability of exactly 3 successes)
 * dist.logPmf(3)               // -1.3218... (log of the mass)
 * dist.cdf(3)                  // 0.6496... (probability of 3 or fewer successes)
 * dist.quantileInt(0.5)        // 3 (median as Int)
 * dist.quantile(0.5)           // 3.0 (median as Double, for Distribution compatibility)
 * dist.sample(Random(42))      // a single random draw
 * dist.sample(5, Random(42))   // five random draws
 * ```
 *
 * @see Distribution for inherited statistical properties and survival function.
 * @see ContinuousDistribution for the continuous counterpart.
 */
public interface DiscreteDistribution : Distribution {

    /**
     * Returns the probability mass at [k].
     *
     * The probability mass is the exact probability that the random variable equals [k].
     * Returns zero for values outside the distribution's support.
     *
     * @param k the integer outcome at which to evaluate the mass.
     * @return the probability that the random variable equals [k], in the range `[0, 1]`.
     */
    public fun pmf(k: Int): Double

    /**
     * Returns the natural logarithm of the probability mass at [k].
     *
     * The log-mass is useful when working with very small probabilities that would
     * underflow to zero in regular floating-point arithmetic. Every implementation provides
     * a direct formula rather than computing `ln(pmf(k))`, which avoids unnecessary
     * precision loss.
     *
     * @param k the integer outcome at which to evaluate the log-mass.
     * @return the natural log of the probability mass at [k]. Returns [Double.NEGATIVE_INFINITY]
     * when the mass is zero.
     */
    public fun logPmf(k: Int): Double

    /**
     * Returns the cumulative distribution function (CDF) value at integer [k].
     *
     * The CDF gives the probability that a random variable drawn from this distribution
     * is less than or equal to [k]. This is the sum of [pmf] values from the smallest
     * supported value up to [k].
     *
     * @param k the integer point at which to evaluate the cumulative probability.
     * @return the probability that a value is less than or equal to [k], in the range `[0, 1]`.
     */
    public fun cdf(k: Int): Double

    /**
     * Returns the survival function value at integer [k].
     *
     * The survival function is the probability that the random variable is strictly
     * greater than [k], equal to `1 - cdf(k)`.
     *
     * @param k the integer point at which to evaluate the survival probability.
     * @return the probability that a value is greater than [k], in the range `[0, 1]`.
     */
    public fun sf(k: Int): Double = 1.0 - cdf(k)

    /**
     * Returns the SF value at [x], bridging to the integer [sf] overload.
     *
     * Floors [x] to an integer via [floor] and delegates to [sf].
     * Returns [Double.NaN] if [x] is NaN.
     *
     * @param x the point at which to evaluate the survival probability.
     * @return the probability that a value is greater than the floor of [x].
     */
    override fun sf(x: Double): Double {
        if (x.isNaN()) return Double.NaN
        return sf(floor(x).toInt())
    }

    /**
     * Returns the CDF value at [x], bridging to the integer [cdf] overload.
     *
     * Floors [x] to an integer via [floor] and delegates to [cdf].
     * This override satisfies the [Distribution] interface contract, which declares
     * `cdf(x: Double)`.
     *
     * @param x the point at which to evaluate the cumulative probability.
     * @return the probability that a value is less than or equal to the floor of [x].
     */
    override fun cdf(x: Double): Double {
        if (x.isNaN()) return Double.NaN
        return cdf(floor(x).toInt())
    }

    /**
     * Returns the quantile (inverse CDF) for the given probability [p] as an [Int].
     *
     * The quantile function returns the smallest integer k such that `cdf(k) >= p`.
     * This is the type-safe integer variant — use [quantile] for [Double] compatibility
     * with the [Distribution] interface.
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the smallest integer k at which `cdf(k) >= p`.
     */
    public fun quantileInt(p: Double): Int

    /**
     * Returns the quantile (inverse CDF) for the given probability [p] as a [Double].
     *
     * Delegates to [quantileInt] and converts the result to [Double]. This override
     * satisfies the [Distribution] interface contract. Prefer [quantileInt] when an
     * integer result is needed.
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the smallest value at which `cdf(x) >= p`, as a [Double].
     */
    override fun quantile(p: Double): Double = quantileInt(p).toDouble()

    /**
     * Returns the skewness of this distribution.
     *
     * Returns [Double.NaN] by default. Implementations override this with the exact formula
     * for each distribution.
     *
     * @return the skewness, or [Double.NaN] if not yet implemented.
     */
    override val skewness: Double get() = Double.NaN

    /**
     * Returns the excess kurtosis (Fisher definition) of this distribution.
     *
     * Returns [Double.NaN] by default. Implementations override this with the exact formula
     * for each distribution.
     *
     * @return the excess kurtosis, or [Double.NaN] if not yet implemented.
     */
    override val kurtosis: Double get() = Double.NaN

    /**
     * Returns the Shannon entropy of this distribution in nats (natural logarithm units).
     *
     * Entropy measures the uncertainty or "spread" of a distribution. Higher entropy means
     * more uncertainty.
     *
     * Returns [Double.NaN] by default. Implementations override this with the exact formula
     * for each distribution.
     *
     * @return the entropy in nats, or [Double.NaN] if not yet implemented.
     */
    override val entropy: Double get() = Double.NaN

    /**
     * Draws a single random integer value from this distribution.
     *
     * @param random the source of randomness.
     * @return a random integer drawn from this distribution.
     */
    public fun sample(random: Random): Int

    /**
     * Draws [n] independent random integer values from this distribution.
     *
     * Each element in the returned array is drawn independently using [sample].
     *
     * @param n the number of values to draw. Must be non-negative.
     * @param random the source of randomness.
     * @return an [IntArray] of [n] independent random draws.
     */
    public fun sample(n: Int, random: Random): IntArray {
        if (n < 0) throw InvalidParameterException("n must be non-negative, got $n")
        return IntArray(n) { sample(random) }
    }
}
