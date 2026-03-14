package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.*
import kotlin.random.Random

/**
 * Represents the geometric distribution, which models the number of failures before the first
 * success in a series of independent Bernoulli trials.
 *
 * This distribution uses the 0-indexed convention: `k = 0` means success occurred on the very
 * first trial (zero failures), `k = 1` means one failure before the first success, and so on.
 * The support is `{0, 1, 2, ...}` (all non-negative integers).
 *
 * The geometric distribution is memoryless -- the probability of needing at least `m` more
 * failures is the same regardless of how many failures have already occurred. It is also a
 * special case of the negative binomial distribution with `successes = 1`.
 *
 * Common applications include modeling the number of defective items inspected before finding
 * a good one, the number of unsuccessful sales calls before a sale, or the number of coin
 * flips before landing heads.
 *
 * ### Example:
 * ```kotlin
 * val dist = GeometricDistribution(probability = 0.3)
 * dist.pmf(0)               // 0.3 (success on first trial)
 * dist.pmf(2)               // 0.147 (two failures, then success)
 * dist.cdf(3)               // 0.7599 (at most 3 failures)
 * dist.mean                 // 2.3333 (expected number of failures)
 * dist.quantileInt(0.5)     // 2 (median number of failures)
 * dist.sample(Random(42))   // a single random draw
 * ```
 *
 * @param probability the probability of success on each trial. Must be in `(0, 1]`.
 */
public data class GeometricDistribution(
    val probability: Double
) : DiscreteDistribution {

    init {
        if (probability <= 0.0 || probability > 1.0) throw InvalidParameterException("probability must be in (0, 1], got $probability")
    }

    private val p = probability
    private val q = 1.0 - p

    /**
     * Returns the probability mass at [k], the probability of exactly [k] failures before
     * the first success.
     *
     * @param k the number of failures before the first success.
     * @return the probability of exactly [k] failures, or zero if [k] is negative.
     */
    override fun pmf(k: Int): Double {
        if (k < 0) return 0.0
        return p * q.pow(k)
    }

    /**
     * Returns the natural logarithm of the probability mass at [k].
     *
     * @param k the number of failures before the first success.
     * @return the natural log of the probability mass at [k]. Returns [Double.NEGATIVE_INFINITY]
     * when [k] is negative.
     */
    override fun logPmf(k: Int): Double {
        if (k < 0) return Double.NEGATIVE_INFINITY
        return ln(p) + k * ln(q)
    }

    /**
     * Returns the cumulative distribution function value at [k].
     *
     * @param k the integer point at which to evaluate the cumulative probability.
     * @return the probability of at most [k] failures before the first success.
     */
    override fun cdf(k: Int): Double {
        if (k < 0) return 0.0
        return 1.0 - q.pow(k + 1)
    }

    /**
     * Returns the quantile (inverse CDF) for the given probability [p] as an [Int].
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the smallest integer k at which `cdf(k) >= p`. Returns [Int.MAX_VALUE] when
     * `p = 1.0`.
     */
    override fun quantileInt(p: Double): Int {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return 0
        if (p == 1.0) return Int.MAX_VALUE
        return ceil(ln(1.0 - p) / ln(q) - 1.0).toInt().coerceAtLeast(0)
    }

    /** The mean (expected number of failures before the first success). */
    override val mean: Double get() = q / p

    /** The variance of the number of failures before the first success. */
    override val variance: Double get() = q / (p * p)

    /** The skewness of this distribution. */
    override val skewness: Double get() = (2.0 - p) / sqrt(q)

    /** The excess kurtosis of this distribution. */
    override val kurtosis: Double get() = 6.0 + p * p / q

    /** The Shannon entropy of this distribution in nats. Returns zero when [probability] is 1.0 (degenerate case). */
    override val entropy: Double get() {
        if (p == 1.0) return 0.0
        return (-q * ln(q) - p * ln(p)) / p
    }

    /**
     * Returns the survival function value at [k].
     *
     * Computed directly as `(1 - probability)^(k+1)` rather than `1 - cdf(k)`, which avoids
     * catastrophic cancellation in the upper tail.
     *
     * @param k the integer point at which to evaluate the survival probability.
     * @return the probability that the number of failures is strictly greater than [k].
     */
    override fun sf(k: Int): Double {
        if (k < 0) return 1.0
        return q.pow(k + 1)
    }

    /**
     * Draws a single random value from this geometric distribution using inverse transform
     * sampling.
     *
     * @param random the source of randomness.
     * @return a random non-negative integer representing the number of failures before success.
     */
    override fun sample(random: Random): Int {
        return floor(ln(random.nextDouble()) / ln(q)).toInt()
    }
}
