package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.ceil
import kotlin.math.ln
import kotlin.random.Random

/**
 * Represents the discrete uniform distribution, where all integer outcomes in a finite range
 * are equally likely.
 *
 * Every integer from [min] to [max] (inclusive) has the same probability of being observed.
 * This is the discrete analogue of the continuous uniform distribution and models situations
 * where each outcome is equally probable, such as rolling a fair die (`min = 1`, `max = 6`),
 * selecting a random integer from a range, or choosing a random index in an array.
 *
 * The support is `{min, min + 1, ..., max}`, giving `max - min + 1` equally likely outcomes.
 * The distribution is always symmetric, so the skewness is zero.
 *
 * ### Example:
 * ```kotlin
 * // Fair six-sided die
 * val die = UniformDiscreteDistribution(min = 1, max = 6)
 * die.pmf(3)               // 0.1667 (each face has probability 1/6)
 * die.cdf(3)               // 0.5 (probability of rolling 3 or less)
 * die.mean                 // 3.5
 * die.variance             // 2.9167
 * die.quantileInt(0.5)     // 3 (median)
 * die.sample(Random(42))   // a single random roll
 *
 * // Random array index
 * val idx = UniformDiscreteDistribution(min = 0, max = 99)
 * idx.pmf(50)              // 0.01 (each index equally likely)
 * ```
 *
 * @param min the smallest value in the support (inclusive).
 * @param max the largest value in the support (inclusive). Must be greater than or equal to [min].
 */
public class UniformDiscreteDistribution(
    public val min: Int,
    public val max: Int
) : DiscreteDistribution {

    init {
        if (min > max) throw InvalidParameterException("min must be <= max, got min=$min, max=$max")
    }

    private val n: Long = max.toLong() - min.toLong() + 1

    /**
     * Returns the probability mass at [k].
     *
     * Returns `1 / (max - min + 1)` for values within the support and zero otherwise.
     *
     * @param k the integer outcome at which to evaluate the mass.
     * @return the probability that the random variable equals [k].
     */
    override fun pmf(k: Int): Double = if (k in min..max) 1.0 / n else 0.0

    /**
     * Returns the natural logarithm of the probability mass at [k].
     *
     * @param k the integer outcome at which to evaluate the log-mass.
     * @return the natural log of the probability mass at [k]. Returns [Double.NEGATIVE_INFINITY]
     * when [k] is outside the support.
     */
    override fun logPmf(k: Int): Double = if (k in min..max) -ln(n.toDouble()) else Double.NEGATIVE_INFINITY

    /**
     * Returns the cumulative distribution function value at [k].
     *
     * @param k the integer point at which to evaluate the cumulative probability.
     * @return the probability that a value is less than or equal to [k].
     */
    override fun cdf(k: Int): Double = when {
        k < min -> 0.0
        k >= max -> 1.0
        else -> (k.toLong() - min.toLong() + 1).toDouble() / n.toDouble()
    }

    /**
     * Returns the quantile (inverse CDF) for the given probability [p] as an [Int].
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the smallest integer k in the support at which `cdf(k) >= p`.
     */
    override fun quantileInt(p: Double): Int {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return min
        return (min.toLong() + ceil(p * n.toDouble()).toLong() - 1).coerceIn(min.toLong(), max.toLong()).toInt()
    }

    /** The mean of this distribution, equal to the midpoint of [min] and [max]. */
    override val mean: Double get() = min / 2.0 + max / 2.0

    /** The variance of this distribution. */
    override val variance: Double get() = (n.toDouble() * n - 1.0) / 12.0

    /** The skewness of this distribution, always zero because the discrete uniform distribution is symmetric. */
    override val skewness: Double get() = 0.0

    /** The excess kurtosis of this distribution. Returns [Double.NaN] when there is only one outcome. */
    override val kurtosis: Double
        get() {
            if (n == 1L) return Double.NaN
            val nd = n.toDouble()
            return -6.0 * (nd * nd + 1.0) / (5.0 * (nd * nd - 1.0))
        }

    /** The Shannon entropy of this distribution in nats, equal to the natural log of the number of outcomes. */
    override val entropy: Double get() = ln(n.toDouble())

    /**
     * Draws a single random value from this uniform discrete distribution.
     *
     * @param random the source of randomness.
     * @return a random integer uniformly chosen from `[min, max]`.
     */
    override fun sample(random: Random): Int = (min.toLong() + random.nextLong(n)).toInt()
}
