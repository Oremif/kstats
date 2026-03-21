package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.core.generalizedHarmonic
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Represents the Zipf distribution (finite support variant), a discrete power-law distribution
 * over the integers 1, 2, ..., [numberOfElements].
 *
 * The probability of observing rank k is proportional to 1/k^[exponent]. This distribution
 * models phenomena where a few items are very frequent and the rest are increasingly rare,
 * such as word frequencies in natural language (Zipf's law), city population sizes, and
 * website traffic distributions.
 *
 * This is the finite-support parameterization matching Apache Commons Math: all moments are
 * finite and the CDF is an exact finite sum. The normalization constant is the generalized
 * harmonic number H([numberOfElements], [exponent]).
 *
 * ### Example:
 * ```kotlin
 * val dist = ZipfDistribution(numberOfElements = 10, exponent = 1.0)
 * dist.pmf(1)               // 0.3414 (most probable rank)
 * dist.pmf(10)              // 0.0341 (least probable rank)
 * dist.cdf(5)               // 0.7796
 * dist.mean                 // 3.4142
 * dist.quantileInt(0.5)     // 2 (median)
 * dist.sample(Random(42))   // a single random draw
 * ```
 *
 * @param numberOfElements the number of elements (upper bound of support). Must be at least 1.
 * @param exponent the exponent characterizing the distribution. Must be positive.
 */
public class ZipfDistribution(
    public val numberOfElements: Int,
    public val exponent: Double
) : DiscreteDistribution {

    init {
        if (numberOfElements < 1) throw InvalidParameterException(
            "numberOfElements must be >= 1, got $numberOfElements"
        )
        if (exponent <= 0.0) throw InvalidParameterException(
            "exponent must be > 0, got $exponent"
        )
    }

    private val n = numberOfElements
    private val s = exponent

    /** Normalization constant: H(n, s). */
    private val hns = generalizedHarmonic(n, s)

    /** Log of normalization constant. */
    private val logHns = ln(hns)

    /**
     * Returns the probability mass at [k], the probability of observing rank [k].
     *
     * @param k the rank at which to evaluate the mass.
     * @return the probability of rank [k], or zero if [k] is outside the support `[1, n]`.
     */
    override fun pmf(k: Int): Double {
        if (k !in 1..n) return 0.0
        return exp(logPmf(k))
    }

    /**
     * Returns the natural logarithm of the probability mass at [k].
     *
     * Computed directly in log-space as `-s * ln(k) - ln(H(n, s))`.
     *
     * @param k the rank at which to evaluate the log-mass.
     * @return the natural log of the probability mass at [k]. Returns [Double.NEGATIVE_INFINITY]
     * when [k] is outside the support.
     */
    override fun logPmf(k: Int): Double {
        if (k !in 1..n) return Double.NEGATIVE_INFINITY
        return -s * ln(k.toDouble()) - logHns
    }

    /**
     * Returns the cumulative distribution function value at [k].
     *
     * Computed as the exact finite sum H([k], [exponent]) / H([numberOfElements], [exponent]).
     *
     * @param k the integer point at which to evaluate the cumulative probability.
     * @return the probability of observing a rank less than or equal to [k].
     */
    override fun cdf(k: Int): Double {
        if (k < 1) return 0.0
        if (k >= n) return 1.0
        return generalizedHarmonic(k, s) / hns
    }

    /**
     * Returns the survival function value at [k], equal to `1 - cdf(k)`.
     *
     * @param k the integer point at which to evaluate the survival probability.
     * @return the probability of observing a rank strictly greater than [k].
     */
    override fun sf(k: Int): Double {
        if (k < 1) return 1.0
        if (k >= n) return 0.0
        return (hns - generalizedHarmonic(k, s)) / hns
    }

    /**
     * Returns the quantile (inverse CDF) for the given probability [p] as an [Int].
     *
     * Uses a linear search accumulating PMF values from k=1.
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the smallest integer k in `[1, n]` at which `cdf(k) >= p`.
     */
    override fun quantileInt(p: Double): Int {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        var cumulative = 0.0
        for (k in 1..n) {
            cumulative += pmf(k)
            if (cumulative >= p) return k
        }
        return n
    }

    /** The mean of this distribution: H(n, s-1) / H(n, s). */
    override val mean: Double get() = generalizedHarmonic(n, s - 1.0) / hns

    /** The variance of this distribution. */
    override val variance: Double
        get() {
            val hns1 = generalizedHarmonic(n, s - 1.0)
            val hns2 = generalizedHarmonic(n, s - 2.0)
            return (hns2 * hns - hns1 * hns1) / (hns * hns)
        }

    /**
     * The skewness of this distribution.
     *
     * Computed from raw moments μ'_r = H(n, s-r) / H(n, s) via the third central moment.
     * Returns [Double.NaN] when [numberOfElements] is 1 (degenerate case with zero variance).
     */
    override val skewness: Double
        get() {
            if (n == 1) return Double.NaN
            val mu1 = generalizedHarmonic(n, s - 1.0) / hns
            val mu2 = generalizedHarmonic(n, s - 2.0) / hns
            val mu3 = generalizedHarmonic(n, s - 3.0) / hns
            val v = mu2 - mu1 * mu1
            if (v == 0.0) return Double.NaN
            val m3 = mu3 - 3.0 * mu1 * mu2 + 2.0 * mu1 * mu1 * mu1
            return m3 / (v * sqrt(v))
        }

    /**
     * The excess kurtosis (Fisher definition) of this distribution.
     *
     * Computed from raw moments μ'_r = H(n, s-r) / H(n, s) via the fourth central moment.
     * Returns [Double.NaN] when [numberOfElements] is 1 (degenerate case with zero variance).
     */
    override val kurtosis: Double
        get() {
            if (n == 1) return Double.NaN
            val mu1 = generalizedHarmonic(n, s - 1.0) / hns
            val mu2 = generalizedHarmonic(n, s - 2.0) / hns
            val mu3 = generalizedHarmonic(n, s - 3.0) / hns
            val mu4 = generalizedHarmonic(n, s - 4.0) / hns
            val v = mu2 - mu1 * mu1
            if (v == 0.0) return Double.NaN
            val m4 = mu4 - 4.0 * mu1 * mu3 + 6.0 * mu1 * mu1 * mu2 - 3.0 * mu1 * mu1 * mu1 * mu1
            return m4 / (v * v) - 3.0
        }

    /**
     * The Shannon entropy of this distribution in nats, computed by summing over the entire support.
     */
    override val entropy: Double
        get() {
            var h = 0.0
            for (k in 1..n) {
                val pk = pmf(k)
                if (pk > 0.0) h -= pk * ln(pk)
            }
            return h
        }

    /**
     * Draws a single random value from this Zipf distribution using inverse transform sampling.
     *
     * @param random the source of randomness.
     * @return a random integer in `[1, n]`.
     */
    override fun sample(random: Random): Int = quantileInt(random.nextDouble())
}
