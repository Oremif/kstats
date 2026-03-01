package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Represents the Bernoulli distribution, the simplest discrete probability distribution.
 *
 * The Bernoulli distribution models a single trial with exactly two outcomes: success (1) with
 * probability [probability], or failure (0) with the complementary probability. It is the
 * building block for many other discrete distributions -- a coin flip is the classic example,
 * and a sequence of independent Bernoulli trials gives rise to the binomial distribution.
 * In fact, the Bernoulli distribution is a special case of the binomial with `trials = 1`.
 *
 * The support of this distribution is `{0, 1}`. The PMF returns [probability] at `k = 1` and
 * `1 - probability` at `k = 0`, and zero for any other value.
 *
 * ### Example:
 * ```kotlin
 * val coin = BernoulliDistribution(probability = 0.6)
 * coin.pmf(1)               // 0.6 (probability of success)
 * coin.pmf(0)               // 0.4 (probability of failure)
 * coin.cdf(0)               // 0.4 (probability of 0 or fewer)
 * coin.mean                 // 0.6
 * coin.variance             // 0.24
 * coin.quantileInt(0.5)     // 1 (median)
 * coin.sample(Random(42))   // 0 or 1
 * ```
 *
 * @param probability the probability of success (outcome = 1). Must be in `[0, 1]`.
 */
public class BernoulliDistribution(
    public val probability: Double
) : DiscreteDistribution {

    init {
        if (probability !in 0.0..1.0) throw InvalidParameterException("probability must be in [0, 1], got $probability")
    }

    private val q = 1.0 - probability

    /**
     * Returns the probability mass at [k].
     *
     * Returns [probability] when `k = 1`, `1 - probability` when `k = 0`, and zero otherwise.
     *
     * @param k the integer outcome at which to evaluate the mass.
     * @return the probability that the random variable equals [k].
     */
    override fun pmf(k: Int): Double = when (k) {
        0 -> q
        1 -> probability
        else -> 0.0
    }

    /**
     * Returns the natural logarithm of the probability mass at [k].
     *
     * @param k the integer outcome at which to evaluate the log-mass.
     * @return the natural log of the probability mass at [k]. Returns [Double.NEGATIVE_INFINITY]
     * when [k] is outside the support `{0, 1}`.
     */
    override fun logPmf(k: Int): Double = when (k) {
        0 -> ln(q)
        1 -> ln(probability)
        else -> Double.NEGATIVE_INFINITY
    }

    /**
     * Returns the cumulative distribution function value at [k].
     *
     * @param k the integer point at which to evaluate the cumulative probability.
     * @return the probability that a value is less than or equal to [k].
     */
    override fun cdf(k: Int): Double = when {
        k < 0 -> 0.0
        k < 1 -> q
        else -> 1.0
    }

    /**
     * Returns the quantile (inverse CDF) for the given probability [p] as an [Int].
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the smallest outcome `k` in `{0, 1}` such that `cdf(k) >= p`.
     *   Returns `0` when `p <= 1 - probability` (including the boundary), and `1` otherwise.
     */
    override fun quantileInt(p: Double): Int {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        return if (p <= q) 0 else 1
    }

    /** The mean of this distribution, equal to [probability]. */
    override val mean: Double get() = probability

    /** The variance of this distribution, equal to `probability * (1 - probability)`. */
    override val variance: Double get() = probability * q

    /** The skewness of this distribution. Returns [Double.NaN] when [probability] is 0 or 1. */
    override val skewness: Double
        get() =
            if (probability == 0.0 || probability == 1.0) Double.NaN
            else (1.0 - 2.0 * probability) / sqrt(probability * q)

    /** The excess kurtosis of this distribution. Returns [Double.NaN] when [probability] is 0 or 1. */
    override val kurtosis: Double
        get() =
            if (probability == 0.0 || probability == 1.0) Double.NaN
            else (1.0 - 6.0 * probability * q) / (probability * q)

    /** The Shannon entropy of this distribution in nats. Returns zero for degenerate cases where [probability] is 0 or 1. */
    override val entropy: Double
        get() = when {
            probability == 0.0 || probability == 1.0 -> 0.0
            else -> -probability * ln(probability) - q * ln(q)
        }

    /**
     * Draws a single random value from this Bernoulli distribution.
     *
     * Returns `1` (success) with probability [probability] and `0` (failure) otherwise.
     *
     * @param random the source of randomness.
     * @return `0` or `1`.
     */
    override fun sample(random: Random): Int = if (random.nextDouble() < probability) 1 else 0
}
