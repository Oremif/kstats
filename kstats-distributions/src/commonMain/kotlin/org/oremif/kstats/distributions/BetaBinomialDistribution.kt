package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.core.lnBeta
import org.oremif.kstats.core.lnCombination
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Represents the beta-binomial distribution, a compound distribution arising when the success
 * probability of a binomial distribution is itself random and follows a beta distribution.
 *
 * The beta-binomial distribution models overdispersed count data where the assumption of a fixed
 * success probability is too restrictive. Instead of a single probability, each observation draws
 * its own success probability from a Beta([alpha], [beta]) distribution, and the count of successes
 * in [trials] independent trials then follows a binomial. This two-stage model captures extra
 * variability beyond what a simple binomial allows, making it suitable for scenarios such as the
 * number of defective items in batches with varying quality, or the number of positive responses
 * in surveys where response rates vary across groups.
 *
 * The support is `{0, 1, ..., trials}`. Sampling works by first drawing a random probability from
 * Beta([alpha], [beta]), then drawing from Binomial([trials], p). The PMF, CDF, and survival
 * function are computed in log-space using the log-beta function and the log-sum-exp technique
 * for numerical stability.
 *
 * ### Example:
 * ```kotlin
 * val dist = BetaBinomialDistribution(trials = 10, alpha = 2.0, beta = 5.0)
 * dist.pmf(3)               // probability of exactly 3 successes
 * dist.cdf(3)               // probability of 3 or fewer successes
 * dist.quantileInt(0.5)     // median number of successes
 * dist.mean                 // 2.857... (10 * 2 / (2 + 5))
 * dist.variance             // 3.316... (overdispersed relative to binomial)
 * dist.sample(Random(42))   // a single random draw
 * ```
 *
 * @property trials the number of independent trials. Must be non-negative.
 * @property alpha the first shape parameter of the underlying Beta distribution. Must be positive.
 * @property beta the second shape parameter of the underlying Beta distribution. Must be positive.
 */
public class BetaBinomialDistribution(
    public val trials: Int,
    public val alpha: Double,
    public val beta: Double,
) : DiscreteDistribution {

    init {
        if (trials < 0) throw InvalidParameterException("trials must be non-negative, got $trials")
        if (!alpha.isFinite() || alpha <= 0.0) throw InvalidParameterException("alpha must be finite and positive, got $alpha")
        if (!beta.isFinite() || beta <= 0.0) throw InvalidParameterException("beta must be finite and positive, got $beta")
    }

    private val n = trials
    private val a = alpha
    private val b = beta
    private val lnBetaAB = lnBeta(a, b)

    private val betaDelegate: BetaDistribution by lazy { BetaDistribution(a, b) }

    /**
     * Returns the probability of exactly [k] successes in this beta-binomial distribution.
     *
     * Computes the probability mass by exponentiating the [logPmf] value. Returns zero for
     * values outside the support (k < 0 or k > trials). When trials is zero, the only possible
     * outcome is k = 0 with probability 1.
     *
     * @param k the number of successes at which to evaluate the probability.
     * @return the probability of exactly [k] successes, in the range `[0, 1]`.
     */
    override fun pmf(k: Int): Double {
        if (k !in 0..n) return 0.0
        if (n == 0) return if (k == 0) 1.0 else 0.0
        return exp(logPmf(k))
    }

    /**
     * Returns the natural logarithm of the probability mass at [k] for this beta-binomial distribution.
     *
     * Computed directly in log-space using the log-binomial-coefficient and the log-beta function:
     * `ln C(n, k) + ln B(k + alpha, n - k + beta) - ln B(alpha, beta)`. This avoids overflow for
     * large trial counts where intermediate products would exceed floating-point range.
     *
     * @param k the number of successes at which to evaluate the log-probability.
     * @return the natural log of the probability mass at [k]. Returns [Double.NEGATIVE_INFINITY]
     *   when the mass is zero.
     */
    override fun logPmf(k: Int): Double {
        if (k !in 0..n) return Double.NEGATIVE_INFINITY
        if (n == 0) return if (k == 0) 0.0 else Double.NEGATIVE_INFINITY
        return lnCombination(n, k) + lnBeta(k + a, n - k + b) - lnBetaAB
    }

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
     * Returns the cumulative distribution function value at [k] for this beta-binomial distribution.
     *
     * Gives the probability of observing [k] or fewer successes. Computed by summing PMF values
     * from 0 to [k] using the log-sum-exp technique for numerical stability, then exponentiating
     * the result.
     *
     * @param k the number of successes at which to evaluate the cumulative probability.
     * @return the probability of [k] or fewer successes, in the range `[0, 1]`.
     */
    override fun cdf(k: Int): Double {
        if (k < 0) return 0.0
        if (k >= n) return 1.0
        return exp(logSumPmf(0..k)).coerceIn(0.0, 1.0)
    }

    /**
     * Returns the survival function value at [k] for this beta-binomial distribution.
     *
     * Gives the probability of observing strictly more than [k] successes. Computed directly
     * by summing PMF values from k + 1 to trials using the log-sum-exp technique, rather than
     * `1 - cdf(k)`, for better numerical accuracy in the upper tail.
     *
     * @param k the number of successes at which to evaluate the survival probability.
     * @return the probability of more than [k] successes, in the range `[0, 1]`.
     */
    override fun sf(k: Int): Double {
        if (k < 0) return 1.0
        if (k >= n) return 0.0
        return exp(logSumPmf((k + 1)..n)).coerceIn(0.0, 1.0)
    }

    /**
     * Returns the quantile (inverse CDF) for the given probability [p] as an integer.
     *
     * Finds the smallest number of successes k such that the cumulative probability of k or
     * fewer successes is at least [p]. Uses a linear search over the support from 0 to trials,
     * accumulating PMF values.
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the smallest integer k at which `cdf(k) >= p`.
     */
    override fun quantileInt(p: Double): Int {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (n == 0) return 0
        var cumulative = 0.0
        for (k in 0..n) {
            cumulative += pmf(k)
            if (cumulative >= p) return k
        }
        return n
    }

    /** The mean of this distribution, equal to n * alpha / (alpha + beta). */
    override val mean: Double get() = n * a / (a + b)

    /** The variance of this distribution, overdispersed relative to the binomial by a factor of (alpha + beta + n) / (alpha + beta + 1). */
    override val variance: Double
        get() {
            if (n == 0) return 0.0
            val ab = a + b
            return n * a * b * (ab + n) / (ab * ab * (ab + 1.0))
        }

    /** The skewness of this distribution, computed from the shape parameters and trial count. Returns [Double.NaN] when trials is zero. */
    override val skewness: Double
        get() {
            if (n == 0) return Double.NaN
            val ab = a + b
            return (ab + 2.0 * n) * (b - a) / (ab + 2.0) * sqrt((1.0 + ab) / (n * a * b * (ab + n)))
        }

    /**
     * Returns the excess kurtosis (Fisher definition) of this beta-binomial distribution.
     *
     * Computed from factorial moments of the distribution, which are converted to raw moments
     * and then to central moments. Returns [Double.NaN] when trials is zero or the variance
     * is zero.
     *
     * @return the excess kurtosis, or [Double.NaN] for degenerate cases.
     */
    override val kurtosis: Double
        get() {
            if (n == 0) return Double.NaN
            val nd = n.toDouble()
            val ab = a + b
            // Falling factorial moments: e_k = [n]_k * prod_{i=0}^{k-1} (a+i)/(ab+i)
            val e1 = nd * a / ab
            val e2 = nd * (nd - 1) * a * (a + 1) / (ab * (ab + 1))
            val e3 = nd * (nd - 1) * (nd - 2) * a * (a + 1) * (a + 2) / (ab * (ab + 1) * (ab + 2))
            val e4 = nd * (nd - 1) * (nd - 2) * (nd - 3) * a * (a + 1) * (a + 2) * (a + 3) /
                (ab * (ab + 1) * (ab + 2) * (ab + 3))
            // Raw moments from factorial moments
            val ex2 = e2 + e1
            val ex3 = e3 + 3 * e2 + e1
            val ex4 = e4 + 6 * e3 + 7 * e2 + e1
            val mu = e1
            val mu2 = ex2 - mu * mu
            if (mu2 == 0.0) return Double.NaN
            val mu4 = ex4 - 4 * mu * ex3 + 6 * mu * mu * ex2 - 3 * mu * mu * mu * mu
            return mu4 / (mu2 * mu2) - 3.0
        }

    /**
     * Returns the Shannon entropy of this beta-binomial distribution in nats.
     *
     * Computed by summing -pmf(k) * ln(pmf(k)) over all supported values k from 0 to [trials].
     * Returns zero when trials is zero (a degenerate distribution with no uncertainty).
     *
     * @return the entropy in nats. Always non-negative.
     */
    override val entropy: Double
        get() {
            if (n == 0) return 0.0
            var h = 0.0
            for (k in 0..n) {
                val lp = logPmf(k)
                if (lp > Double.NEGATIVE_INFINITY) h -= exp(lp) * lp
            }
            return h
        }

    /**
     * Draws a single random integer from this beta-binomial distribution using compound sampling.
     *
     * First draws a random success probability from Beta([alpha], [beta]), then draws a count of
     * successes from Binomial([trials], p). This two-stage process faithfully reproduces the
     * beta-binomial distribution.
     *
     * @param random the source of randomness.
     * @return a random number of successes drawn from this distribution, in `[0, trials]`.
     */
    override fun sample(random: Random): Int {
        var p = betaDelegate.sample(random)
        if (p.isNaN()) p = a / (a + b)
        p = p.coerceIn(0.0, 1.0)
        if (p == 0.0) return 0
        if (p == 1.0) return n
        // Inline binomial sampling to avoid per-call allocation
        var successes = 0
        for (i in 0 until n) {
            if (random.nextDouble() < p) successes++
        }
        return successes
    }
}
