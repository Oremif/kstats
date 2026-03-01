package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.core.lnCombination
import org.oremif.kstats.core.regularizedBeta
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Represents the binomial distribution, defined by the number of [trials] and the
 * [probability] of success on each trial.
 *
 * The binomial distribution models the number of successes in a fixed number of
 * independent trials, where each trial has the same probability of success. The classic
 * example is counting the number of heads in n coin flips: if you flip a fair coin 10
 * times, the number of heads follows a binomial distribution with trials = 10 and
 * probability = 0.5. More generally, it applies to any repeated yes/no experiment
 * such as defective items in a batch, patients responding to a treatment, or
 * successful network requests. The support is the set of integers from 0 to [trials].
 *
 * The CDF and survival function use the regularized incomplete beta function for
 * numerical stability. Sampling uses direct simulation for small trial counts and a
 * normal approximation for large trial counts.
 *
 * ### Example:
 * ```kotlin
 * val dist = BinomialDistribution(trials = 10, probability = 0.3)
 * dist.pmf(3)           // 0.2668... (probability of exactly 3 successes)
 * dist.cdf(3)           // 0.6496... (probability of 3 or fewer successes)
 * dist.quantileInt(0.5) // 3 (median number of successes)
 * dist.mean             // 3.0
 * dist.variance         // 2.1
 * ```
 *
 * @param trials the number of independent trials. Must be non-negative.
 * @param probability the probability of success on each trial. Must be in `[0, 1]`.
 */
public class BinomialDistribution(
    public val trials: Int,
    public val probability: Double
) : DiscreteDistribution {

    init {
        if (trials < 0) throw InvalidParameterException("trials must be non-negative, got $trials")
        if (probability !in 0.0..1.0) throw InvalidParameterException("probability must be in [0, 1], got $probability")
    }

    private val n = trials
    private val p = probability

    /**
     * Returns the probability mass at [k] for this binomial distribution.
     *
     * Computes the exact probability of observing exactly [k] successes in [trials]
     * independent trials. Returns zero for values outside the support (k < 0 or k > trials).
     * Handles the degenerate cases where the success probability is 0 or 1 directly.
     *
     * @param k the number of successes at which to evaluate the probability.
     * @return the probability of exactly [k] successes, in the range `[0, 1]`.
     */
    override fun pmf(k: Int): Double {
        if (k !in 0..n) return 0.0
        if (p == 0.0) return if (k == 0) 1.0 else 0.0
        if (p == 1.0) return if (k == n) 1.0 else 0.0
        return exp(logPmf(k))
    }

    /**
     * Returns the natural logarithm of the probability mass at [k] for this binomial distribution.
     *
     * Computed directly in log-space using the log-binomial-coefficient plus weighted
     * log-probabilities, avoiding overflow for large trial counts where the binomial
     * coefficient would exceed floating-point range.
     *
     * @param k the number of successes at which to evaluate the log-probability.
     * @return the natural log of the probability mass at [k]. Returns [Double.NEGATIVE_INFINITY]
     *   when the mass is zero.
     */
    override fun logPmf(k: Int): Double {
        if (k !in 0..n) return Double.NEGATIVE_INFINITY
        if (p == 0.0) return if (k == 0) 0.0 else Double.NEGATIVE_INFINITY
        if (p == 1.0) return if (k == n) 0.0 else Double.NEGATIVE_INFINITY
        return lnCombination(n, k) + k * ln(p) + (n - k) * ln(1.0 - p)
    }

    /**
     * Returns the cumulative distribution function value at [k] for this binomial distribution.
     *
     * Gives the probability of observing [k] or fewer successes. Uses the regularized
     * incomplete beta function for numerical stability rather than summing individual
     * probability masses, which would be slow and imprecise for large trial counts.
     *
     * @param k the number of successes at which to evaluate the cumulative probability.
     * @return the probability of [k] or fewer successes, in the range `[0, 1]`.
     */
    override fun cdf(k: Int): Double {
        if (k < 0) return 0.0
        if (k >= n) return 1.0
        // I_{1-p}(n-k, k+1) using regularized beta
        return regularizedBeta(1.0 - p, (n - k).toDouble(), (k + 1).toDouble())
    }

    /**
     * Returns the quantile (inverse CDF) for the given probability [p] as an integer.
     *
     * Finds the smallest number of successes k such that the cumulative probability
     * of k or fewer successes is at least [p]. Uses a binary search over `[0, trials]`.
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the smallest integer k at which `cdf(k) >= p`.
     */
    override fun quantileInt(p: Double): Int {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return 0
        // Binary search over [0, n]
        var lo = 0
        var hi = n
        while (lo < hi) {
            val mid = lo + (hi - lo) / 2
            if (cdf(mid) < p) lo = mid + 1 else hi = mid
        }
        return lo
    }

    /** The mean of this distribution, equal to trials * probability. */
    override val mean: Double get() = n * p

    /** The variance of this distribution, equal to trials * probability * (1 - probability). */
    override val variance: Double get() = n * p * (1.0 - p)

    /**
     * Returns the skewness of this binomial distribution.
     *
     * Measures the asymmetry of the distribution. When the success probability is
     * below 0.5 the distribution is right-skewed; when above 0.5, left-skewed; at
     * exactly 0.5 the distribution is symmetric and the skewness is zero. Returns
     * [Double.NaN] for degenerate cases (zero trials, or probability of 0 or 1).
     *
     * @return the skewness, or [Double.NaN] for degenerate cases.
     */
    override val skewness: Double
        get() {
            val q = 1.0 - p
            if (n == 0 || p == 0.0 || p == 1.0) return Double.NaN
            return (1.0 - 2.0 * p) / sqrt(n.toDouble() * p * q)
        }

    /**
     * Returns the excess kurtosis (Fisher definition) of this binomial distribution.
     *
     * Measures how heavy the tails are compared to a normal distribution. The binomial
     * distribution is always platykurtic (negative excess kurtosis) or mesokurtic,
     * meaning its tails are lighter than or equal to those of the normal. Returns
     * [Double.NaN] for degenerate cases (zero trials, or probability of 0 or 1).
     *
     * @return the excess kurtosis, or [Double.NaN] for degenerate cases.
     */
    override val kurtosis: Double
        get() {
            val q = 1.0 - p
            if (n == 0 || p == 0.0 || p == 1.0) return Double.NaN
            return (1.0 - 6.0 * p * q) / (n.toDouble() * p * q)
        }

    /**
     * Returns the Shannon entropy of this binomial distribution in nats.
     *
     * Computed by summing -pmf(k) * ln(pmf(k)) over all supported values k from 0
     * to [trials]. Returns zero when trials is zero (a degenerate distribution with
     * no uncertainty).
     *
     * @return the entropy in nats. Always non-negative.
     */
    override val entropy: Double
        get() {
            if (n == 0) return 0.0
            if (p == 0.0 || p == 1.0) return 0.0
            val q = 1.0 - p
            var h = 0.0
            var pk = q.pow(n.toDouble()) // pmf(0) = (1-p)^n
            if (pk > 0.0) h -= pk * ln(pk)
            for (k in 0 until n) {
                pk *= (n - k).toDouble() / (k + 1).toDouble() * p / q
                if (pk > 0.0) h -= pk * ln(pk)
            }
            return h
        }

    /**
     * Returns the survival function value at [k] for this binomial distribution.
     *
     * Gives the probability of observing strictly more than [k] successes. Uses the
     * regularized incomplete beta function for numerical stability.
     *
     * @param k the number of successes at which to evaluate the survival probability.
     * @return the probability of more than [k] successes, in the range `[0, 1]`.
     */
    override fun sf(k: Int): Double {
        if (k < 0) return 1.0
        if (k >= n) return 0.0
        return regularizedBeta(p, (k + 1).toDouble(), (n - k).toDouble())
    }

    /**
     * Draws a single random integer from this binomial distribution.
     *
     * For small trial counts (fewer than 25), uses direct simulation by running each
     * trial independently. For large trial counts, uses a normal approximation with
     * the result clamped to the valid range [0, trials].
     *
     * @param random the source of randomness.
     * @return a random number of successes drawn from this distribution, in `[0, trials]`.
     */
    override fun sample(random: Random): Int {
        if (p == 0.0) return 0
        if (p == 1.0) return n
        // For small n or when normal approximation is unreliable, direct simulation
        val np = n * p
        val nq = n * (1.0 - p)
        if (n < 25 || np < 5.0 || nq < 5.0) {
            var successes = 0
            for (i in 0 until n) {
                if (random.nextDouble() < p) successes++
            }
            return successes
        }
        // For large n with reliable normal approximation
        val normal = NormalDistribution(mean, sqrt(variance))
        return normal.sample(random).roundToInt().coerceIn(0, n)
    }
}
