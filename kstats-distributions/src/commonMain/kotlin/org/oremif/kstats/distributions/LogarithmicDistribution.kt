package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Represents the logarithmic (log-series) distribution, a discrete power-series distribution
 * on the positive integers {1, 2, 3, ...}.
 *
 * The probability of observing k is proportional to p^k / k. This distribution arises as
 * the conditional distribution of the number of occurrences given at least one occurrence,
 * and commonly models species abundance in ecology (Fisher's logarithmic series), word
 * frequency distributions, and cascade failures in networks.
 *
 * The PMF is `f(k) = -p^k / (k * ln(1 - p))` for k = 1, 2, 3, ...
 *
 * The normalization constant is `-1 / ln(1 - p)`, derived from the Maclaurin series
 * `-ln(1 - p) = Σ_{k=1}^{∞} p^k / k`.
 *
 * ### Example:
 * ```kotlin
 * val dist = LogarithmicDistribution(probability = 0.5)
 * dist.pmf(1)               // 0.7213 (most probable value)
 * dist.pmf(3)               // 0.0601
 * dist.cdf(2)               // 0.9017
 * dist.mean                 // 1.4427
 * dist.quantileInt(0.5)     // 1 (median)
 * dist.sample(Random(42))   // a single random draw
 * ```
 *
 * @param probability the probability parameter p of the distribution. Must be in `(0, 1)`.
 */
public class LogarithmicDistribution(
    public val probability: Double
) : DiscreteDistribution {

    init {
        if (probability <= 0.0 || probability >= 1.0) throw InvalidParameterException(
            "probability must be in (0, 1), got $probability"
        )
    }

    private val logOneMinusP = ln(1.0 - probability)
    private val a = -1.0 / logOneMinusP
    private val lnA = ln(a)
    private val lnP = ln(probability)

    /**
     * Returns the probability mass at [k].
     *
     * @param k the positive integer at which to evaluate the mass.
     * @return the probability of exactly [k], or zero if [k] is outside the support.
     */
    override fun pmf(k: Int): Double {
        if (k < 1) return 0.0
        return exp(logPmf(k))
    }

    /**
     * Returns the natural logarithm of the probability mass at [k].
     *
     * Computed directly in log-space as `k * ln(p) - ln(k) + ln(a)` where `a = -1 / ln(1 - p)`.
     *
     * @param k the positive integer at which to evaluate the log-mass.
     * @return the natural log of the probability mass at [k]. Returns [Double.NEGATIVE_INFINITY]
     * when [k] is outside the support.
     */
    override fun logPmf(k: Int): Double {
        if (k < 1) return Double.NEGATIVE_INFINITY
        return k * lnP - ln(k.toDouble()) + lnA
    }

    /**
     * Returns the cumulative distribution function value at [k].
     *
     * Computed by accumulating PMF values from k=1 using the recurrence
     * `PMF(k) = PMF(k-1) * p * (k-1) / k`.
     *
     * @param k the integer point at which to evaluate the cumulative probability.
     * @return the probability of observing a value less than or equal to [k].
     */
    override fun cdf(k: Int): Double {
        if (k < 1) return 0.0
        var sum = 0.0
        var term = a * probability // PMF(1)
        sum += term
        for (i in 2..k) {
            term *= probability * (i - 1).toDouble() / i.toDouble()
            sum += term
        }
        return sum.coerceIn(0.0, 1.0)
    }

    /**
     * Returns the survival function value at [k], equal to `1 - cdf(k)`.
     *
     * @param k the integer point at which to evaluate the survival probability.
     * @return the probability of observing a value strictly greater than [k].
     */
    override fun sf(k: Int): Double {
        if (k < 1) return 1.0
        // Compute upper tail directly by subtracting lower tail sum from 1
        // Use CDF for small k, then 1 - cdf for large k to maintain precision
        val cdfVal = cdf(k)
        return 1.0 - cdfVal
    }

    /**
     * Returns the quantile (inverse CDF) for the given probability [p] as an [Int].
     *
     * Uses a linear search accumulating PMF values from k=1.
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the smallest integer k in the support at which `cdf(k) >= p`.
     * Returns [Int.MAX_VALUE] when `p = 1.0` (infinite support).
     */
    override fun quantileInt(p: Double): Int {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return 1
        if (p == 1.0) return Int.MAX_VALUE
        var cumulative = 0.0
        var term = a * probability // PMF(1)
        var k = 1
        while (true) {
            cumulative += term
            if (cumulative >= p) return k
            k++
            term *= probability * (k - 1).toDouble() / k.toDouble()
            if (k > 1_000_000) return k
        }
    }

    /**
     * The mean of this distribution: `-p / ((1-p) * ln(1-p))`.
     */
    override val mean: Double get() = a * probability / (1.0 - probability)

    /** The variance of this distribution. */
    override val variance: Double get() {
        val q = 1.0 - probability
        val mu1 = a * probability / q
        val mu2p = a * probability / (q * q)
        return mu2p - mu1 * mu1
    }

    /**
     * The skewness of this distribution.
     *
     * Computed from closed-form raw moments E[X^r] = a * p * S_r / (1-p)^r
     * where S_1=1, S_2=1, S_3=(1+p), S_4=(1+4p+p²).
     */
    override val skewness: Double get() {
        val q = 1.0 - probability
        val mu1 = a * probability / q
        val mu2p = a * probability / (q * q)
        val mu3p = a * probability * (1.0 + probability) / (q * q * q)
        val v = mu2p - mu1 * mu1
        val m3 = mu3p - 3.0 * mu1 * mu2p + 2.0 * mu1 * mu1 * mu1
        return m3 / (v * sqrt(v))
    }

    /**
     * The excess kurtosis (Fisher definition) of this distribution.
     *
     * Computed from closed-form raw moments.
     */
    override val kurtosis: Double get() {
        val q = 1.0 - probability
        val mu1 = a * probability / q
        val mu2p = a * probability / (q * q)
        val mu3p = a * probability * (1.0 + probability) / (q * q * q)
        val mu4p = a * probability * (1.0 + 4.0 * probability + probability * probability) / (q * q * q * q)
        val v = mu2p - mu1 * mu1
        val m4 = mu4p - 4.0 * mu1 * mu3p + 6.0 * mu1 * mu1 * mu2p -
            3.0 * mu1 * mu1 * mu1 * mu1
        return m4 / (v * v) - 3.0
    }

    /**
     * The Shannon entropy of this distribution in nats, computed by summing over the support
     * until cumulative probability is within 1e-15 of 1.0.
     */
    override val entropy: Double get() {
        var h = 0.0
        var cumP = 0.0
        var term = a * probability // PMF(1)
        var k = 1
        while (cumP < 1.0 - 1e-15) {
            if (term > 0.0) {
                h -= term * ln(term)
                cumP += term
            }
            k++
            term *= probability * (k - 1).toDouble() / k.toDouble()
            if (k > 100_000) break
        }
        return h
    }

    /**
     * Draws a single random value from this logarithmic distribution using inverse transform
     * sampling.
     *
     * @param random the source of randomness.
     * @return a random positive integer drawn from this distribution.
     */
    override fun sample(random: Random): Int = quantileInt(random.nextDouble())
}
