package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.core.lnCombination
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Represents the hypergeometric distribution, which models the number of successes when drawing
 * without replacement from a finite population.
 *
 * Consider an urn containing [population] balls, of which [successes] are "success" balls (e.g.,
 * red) and the rest are "failure" balls (e.g., blue). You draw [draws] balls without putting any
 * back. The hypergeometric distribution gives the probability of observing exactly `k` success
 * balls among the drawn items.
 *
 * Unlike the binomial distribution, the hypergeometric distribution accounts for the changing
 * composition of the population as items are drawn. This makes it the correct model when
 * sampling without replacement, such as quality control inspections from a finite lot, card
 * games (e.g., the probability of being dealt a certain number of aces), or Fisher's exact test
 * in statistics.
 *
 * The support is `{max(0, draws + successes - population), ..., min(draws, successes)}`, which
 * reflects the physical constraint that you cannot draw more success balls than exist or more
 * than the total number of draws.
 *
 * The CDF and survival function are computed using a numerically stable log-sum-exp technique
 * over the PMF values. Sampling uses direct simulation of the drawing process.
 *
 * ### Example:
 * ```kotlin
 * // Urn with 50 balls: 15 red, 35 blue. Draw 10 without replacement.
 * val dist = HypergeometricDistribution(population = 50, successes = 15, draws = 10)
 * dist.pmf(3)               // 0.2786 (probability of exactly 3 red balls)
 * dist.cdf(3)               // 0.6749 (probability of 3 or fewer red balls)
 * dist.mean                 // 3.0 (expected number of red balls)
 * dist.quantileInt(0.5)     // 3 (median)
 * dist.sample(Random(42))   // a single random draw
 * ```
 *
 * @param population the total number of items in the population. Must be non-negative.
 * @param successes the number of success items in the population. Must be in `[0, population]`.
 * @param draws the number of items drawn without replacement. Must be in `[0, population]`.
 */
public class HypergeometricDistribution(
    public val population: Int,
    public val successes: Int,
    public val draws: Int
) : DiscreteDistribution {

    init {
        if (population < 0) throw InvalidParameterException("population must be non-negative, got $population")
        if (successes !in 0..population) throw InvalidParameterException("successes must be in [0, population], got $successes")
        if (draws !in 0..population) throw InvalidParameterException("draws must be in [0, population], got $draws")
    }

    private val bigN = population
    private val bigK = successes
    private val n = draws

    private val kMin = maxOf(0, n + bigK - bigN)
    private val kMax = minOf(n, bigK)

    /**
     * Returns the probability mass at [k], the probability of drawing exactly [k] success items.
     *
     * @param k the number of success items in the draw.
     * @return the probability of exactly [k] successes, or zero if [k] is outside the support.
     */
    override fun pmf(k: Int): Double {
        if (k !in kMin..kMax) return 0.0
        return exp(logPmf(k))
    }

    /**
     * Returns the natural logarithm of the probability mass at [k].
     *
     * Computed using log-combinations to avoid overflow for large population sizes.
     *
     * @param k the number of success items in the draw.
     * @return the natural log of the probability mass at [k]. Returns [Double.NEGATIVE_INFINITY]
     * when [k] is outside the support.
     */
    override fun logPmf(k: Int): Double {
        if (k !in kMin..kMax) return Double.NEGATIVE_INFINITY
        return lnCombination(bigK, k) + lnCombination(bigN - bigK, n - k) - lnCombination(bigN, n)
    }

    /**
     * Computes the log of a sum of PMF values over a range using the log-sum-exp trick for
     * numerical stability.
     */
    private fun logSumPmf(range: IntRange): Double {
        var maxLog = Double.NEGATIVE_INFINITY
        var sumExp = 0.0
        for (i in range) {
            val lp = logPmf(i)
            if (lp > maxLog) {
                sumExp = sumExp * exp(maxLog - lp) + 1.0
                maxLog = lp
            } else {
                sumExp += exp(lp - maxLog)
            }
        }
        return if (maxLog == Double.NEGATIVE_INFINITY) Double.NEGATIVE_INFINITY
        else maxLog + ln(sumExp)
    }

    /**
     * Returns the cumulative distribution function value at [k].
     *
     * @param k the integer point at which to evaluate the cumulative probability.
     * @return the probability of drawing [k] or fewer success items.
     */
    override fun cdf(k: Int): Double {
        if (k < kMin) return 0.0
        if (k >= kMax) return 1.0
        return exp(logSumPmf(kMin..k)).coerceIn(0.0, 1.0)
    }

    /**
     * Returns the survival function value at [k].
     *
     * Computed directly by summing PMF values above [k] using the log-sum-exp technique,
     * rather than `1 - cdf(k)`, for better numerical accuracy.
     *
     * @param k the integer point at which to evaluate the survival probability.
     * @return the probability of drawing strictly more than [k] success items.
     */
    override fun sf(k: Int): Double {
        if (k < kMin) return 1.0
        if (k >= kMax) return 0.0
        return exp(logSumPmf((k + 1)..kMax)).coerceIn(0.0, 1.0)
    }

    /**
     * Returns the quantile (inverse CDF) for the given probability [p] as an [Int].
     *
     * Uses a linear search over the support, accumulating PMF values until the cumulative
     * probability meets or exceeds [p].
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the smallest integer k in the support at which `cdf(k) >= p`.
     */
    override fun quantileInt(p: Double): Int {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        var cumulative = 0.0
        for (k in kMin..kMax) {
            cumulative += pmf(k)
            if (cumulative >= p) return k
        }
        return kMax
    }

    /** The mean (expected number of success items drawn). */
    override val mean: Double get() = n.toDouble() * bigK / bigN

    /** The variance of the number of success items drawn. */
    override val variance: Double get() {
        val nd = n.toDouble()
        val bigNd = bigN.toDouble()
        val bigKd = bigK.toDouble()
        return nd * bigKd * (bigNd - bigKd) * (bigNd - nd) / (bigNd * bigNd * (bigNd - 1.0))
    }

    /** The skewness of this distribution. Returns [Double.NaN] when [population] is less than 3. */
    override val skewness: Double get() {
        if (bigN < 3) return Double.NaN
        val bigNd = bigN.toDouble()
        val bigKd = bigK.toDouble()
        val nd = n.toDouble()
        return (bigNd - 2.0 * bigKd) * (bigNd - 2.0 * nd) * sqrt(bigNd - 1.0) /
            ((bigNd - 2.0) * sqrt(nd * bigKd * (bigNd - bigKd) * (bigNd - nd)))
    }

    /** The excess kurtosis of this distribution. Returns [Double.NaN] when [population] is less than 4. */
    override val kurtosis: Double get() {
        if (bigN < 4) return Double.NaN
        val bigNd = bigN.toDouble()
        val bigKd = bigK.toDouble()
        val nd = n.toDouble()
        val num = (bigNd - 1.0) * bigNd * bigNd * (bigNd * (bigNd + 1.0) - 6.0 * bigKd * (bigNd - bigKd) - 6.0 * nd * (bigNd - nd)) +
            6.0 * nd * bigKd * (bigNd - bigKd) * (bigNd - nd) * (5.0 * bigNd - 6.0)
        val den = nd * bigKd * (bigNd - bigKd) * (bigNd - nd) * (bigNd - 2.0) * (bigNd - 3.0)
        return num / den
    }

    /** The Shannon entropy of this distribution in nats, computed by summing over the entire support. */
    override val entropy: Double get() {
        var h = 0.0
        for (k in kMin..kMax) {
            val pk = pmf(k)
            if (pk > 0.0) h -= pk * ln(pk)
        }
        return h
    }

    /**
     * Draws a single random value from this hypergeometric distribution using direct simulation.
     *
     * Simulates the drawing process by maintaining a pool of success and failure items and
     * drawing one at a time with probabilities proportional to the remaining pool sizes.
     *
     * @param random the source of randomness.
     * @return a random integer representing the number of success items drawn.
     */
    override fun sample(random: Random): Int {
        // Direct simulation
        var succPool = bigK
        var failPool = bigN - bigK
        var result = 0
        for (i in 0 until n) {
            val total = succPool + failPool
            if (random.nextDouble() < succPool.toDouble() / total) {
                result++
                succPool--
            } else {
                failPool--
            }
        }
        return result
    }
}
