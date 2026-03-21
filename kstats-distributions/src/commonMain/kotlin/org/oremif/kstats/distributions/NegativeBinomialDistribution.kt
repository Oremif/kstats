package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.core.lnCombination
import org.oremif.kstats.core.regularizedBeta
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Represents the negative binomial distribution, which models the number of failures before
 * achieving a specified number of successes in a sequence of independent Bernoulli trials.
 *
 * Each trial has the same success probability [probability]. The random variable counts how
 * many failures occur before accumulating [successes] successes. For example, with
 * `successes = 3` and `probability = 0.5`, this distribution gives the probability of
 * observing `k` tails before getting 3 heads in a series of fair coin flips.
 *
 * The negative binomial generalizes the geometric distribution: setting `successes = 1` yields
 * a geometric distribution. The support is `{0, 1, 2, ...}` (all non-negative integers), where
 * `k = 0` means all required successes occurred with no failures.
 *
 * The CDF and survival function are computed using the regularized incomplete beta function,
 * which provides high accuracy without summing many individual PMF terms. Sampling is performed
 * by summing [successes] independent geometric random variables.
 *
 * ### Example:
 * ```kotlin
 * // Number of failures before 5 successes, with p=0.4 per trial
 * val dist = NegativeBinomialDistribution(successes = 5, probability = 0.4)
 * dist.pmf(0)               // 0.01024 (all 5 successes, no failures)
 * dist.pmf(5)               // 0.1003 (5 failures before the 5th success)
 * dist.cdf(7)               // 0.5765 (at most 7 failures)
 * dist.mean                 // 7.5 (expected number of failures)
 * dist.variance             // 18.75
 * dist.quantileInt(0.5)     // 7 (median number of failures)
 * dist.sample(Random(42))   // a single random draw
 * ```
 *
 * @param successes the required number of successes (the "stopping count"). Must be positive.
 * @param probability the probability of success on each trial. Must be in `(0, 1]`.
 */
public class NegativeBinomialDistribution(
    public val successes: Int,
    public val probability: Double
) : DiscreteDistribution {

    init {
        if (successes <= 0) throw InvalidParameterException("successes must be positive, got $successes")
        if (probability <= 0.0 || probability > 1.0) throw InvalidParameterException("probability must be in (0, 1], got $probability")
    }

    private val r = successes
    private val p = probability
    private val q = 1.0 - p

    /**
     * Returns the probability mass at [k], the probability of exactly [k] failures before
     * the [successes]-th success.
     *
     * When [probability] is 1.0 (every trial succeeds), returns 1.0 for `k = 0` and zero
     * otherwise.
     *
     * @param k the number of failures before achieving the required successes.
     * @return the probability of exactly [k] failures, or zero if [k] is negative.
     */
    override fun pmf(k: Int): Double {
        if (k < 0) return 0.0
        if (p == 1.0) return if (k == 0) 1.0 else 0.0
        return exp(logPmf(k))
    }

    /**
     * Returns the natural logarithm of the probability mass at [k].
     *
     * Computed using log-combinations and log-probabilities to maintain numerical stability
     * for large values of [k] or [successes].
     *
     * @param k the number of failures before achieving the required successes.
     * @return the natural log of the probability mass at [k]. Returns [Double.NEGATIVE_INFINITY]
     * when [k] is negative or has zero probability.
     */
    override fun logPmf(k: Int): Double {
        if (k < 0) return Double.NEGATIVE_INFINITY
        if (p == 1.0) return if (k == 0) 0.0 else Double.NEGATIVE_INFINITY
        return lnCombination(k + r - 1, k) + r * ln(p) + k * ln(q)
    }

    /**
     * Returns the cumulative distribution function value at [k] using the regularized incomplete
     * beta function.
     *
     * @param k the integer point at which to evaluate the cumulative probability.
     * @return the probability of at most [k] failures before the [successes]-th success.
     */
    override fun cdf(k: Int): Double {
        if (k < 0) return 0.0
        if (p == 1.0) return 1.0
        return regularizedBeta(p, r.toDouble(), (k + 1).toDouble())
    }

    /**
     * Returns the survival function value at [k] using the regularized incomplete beta function.
     *
     * Computed directly via the beta function rather than `1 - cdf(k)`, which avoids
     * catastrophic cancellation in the upper tail.
     *
     * @param k the integer point at which to evaluate the survival probability.
     * @return the probability of strictly more than [k] failures before the [successes]-th success.
     */
    override fun sf(k: Int): Double {
        if (k < 0) return 1.0
        if (p == 1.0) return 0.0
        return regularizedBeta(q, (k + 1).toDouble(), r.toDouble())
    }

    /**
     * Returns the quantile (inverse CDF) for the given probability [p] as an [Int].
     *
     * Uses a linear search from zero, accumulating PMF values until the cumulative probability
     * meets or exceeds [p]. Includes a safety bound to prevent infinite loops.
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the smallest integer k at which `cdf(k) >= p`.
     */
    override fun quantileInt(p: Double): Int {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return 0
        // Binary search using the CDF (which uses regularizedBeta, so it's O(1) per call)
        val mu = mean
        var lo = 0
        var hi = maxOf(ceil(mu + 20.0 * sqrt(variance)).toInt(), 100)
        // Expand upper bound if needed
        while (cdf(hi) < p) hi *= 2
        while (lo < hi) {
            val mid = lo + (hi - lo) / 2
            if (cdf(mid) < p) lo = mid + 1 else hi = mid
        }
        return lo
    }

    /** The mean (expected number of failures before achieving [successes] successes). */
    override val mean: Double get() = r * q / p

    /** The variance of the number of failures before achieving [successes] successes. */
    override val variance: Double get() = r * q / (p * p)

    /** The skewness of this distribution. */
    override val skewness: Double get() = (2.0 - p) / sqrt(r.toDouble() * q)

    /** The excess kurtosis of this distribution. */
    override val kurtosis: Double get() = 6.0 / r + p * p / (r * q)

    /** The Shannon entropy of this distribution in nats, computed by summing over the support until convergence. */
    override val entropy: Double get() {
        var h = 0.0
        var cumP = 0.0
        var k = 0
        while (cumP < 1.0 - 1e-15) {
            val pk = pmf(k)
            if (pk > 0.0) {
                h -= pk * ln(pk)
                cumP += pk
            }
            k++
            if (k > 100_000) break
        }
        return h
    }

    /**
     * Draws a single random value from this negative binomial distribution.
     *
     * Generates [successes] independent geometric random variables and returns their sum,
     * exploiting the fact that a negative binomial random variable is the sum of independent
     * geometric random variables.
     *
     * @param random the source of randomness.
     * @return a random non-negative integer representing the number of failures before the
     * required successes.
     */
    override fun sample(random: Random): Int {
        // Sum of r geometric(p) random variables
        val geo = GeometricDistribution(p)
        var sum = 0
        for (i in 0 until r) {
            sum += geo.sample(random)
        }
        return sum
    }
}
