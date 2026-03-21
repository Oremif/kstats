package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.core.lnFactorial
import org.oremif.kstats.core.regularizedGammaP
import org.oremif.kstats.core.regularizedGammaQ
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.roundToInt
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Represents the Poisson distribution, defined by its average [rate] of occurrence.
 *
 * The Poisson distribution models the number of events occurring in a fixed interval
 * of time or space, assuming events happen independently and at a constant average
 * rate. Classic examples include the number of emails received per hour, the number
 * of typos on a page, or the number of customers arriving at a store in an hour.
 * It is often used as an approximation to the binomial distribution when the number
 * of trials is large and the probability of success is small. The support is the set
 * of all non-negative integers {0, 1, 2, ...}.
 *
 * The CDF and survival function use the regularized incomplete gamma function for
 * numerical stability. Sampling uses Knuth's algorithm for small rates and a normal
 * approximation for large rates.
 *
 * ### Example:
 * ```kotlin
 * val dist = PoissonDistribution(rate = 4.0)
 * dist.pmf(3)           // 0.1954... (probability of exactly 3 events)
 * dist.cdf(3)           // 0.4335... (probability of 3 or fewer events)
 * dist.quantileInt(0.5) // 4 (median)
 * dist.mean             // 4.0
 * dist.variance         // 4.0 (mean and variance are equal)
 * ```
 *
 * @param rate the average number of events per interval (lambda). Must be positive.
 */
public class PoissonDistribution(
    public val rate: Double
) : DiscreteDistribution {

    init {
        if (rate <= 0.0) throw InvalidParameterException("rate must be positive, got $rate")
    }

    private val lambda = rate

    /**
     * Returns the probability mass at [k] for this Poisson distribution.
     *
     * Computes the exact probability of observing exactly [k] events. Returns zero
     * for negative values of [k]. The computation delegates to [logPmf] and exponentiates
     * the result to avoid intermediate overflow of factorial terms.
     *
     * @param k the number of events at which to evaluate the probability.
     * @return the probability of exactly [k] events, in the range `[0, 1]`.
     */
    override fun pmf(k: Int): Double {
        if (k < 0) return 0.0
        return exp(logPmf(k))
    }

    /**
     * Returns the natural logarithm of the probability mass at [k] for this Poisson distribution.
     *
     * Computed directly in log-space as k * ln(lambda) - lambda - ln(k!), avoiding
     * overflow that would occur when computing k! or lambda^k directly for large values.
     *
     * @param k the number of events at which to evaluate the log-probability.
     * @return the natural log of the probability mass at [k]. Returns [Double.NEGATIVE_INFINITY]
     *   when the mass is zero.
     */
    override fun logPmf(k: Int): Double {
        if (k < 0) return Double.NEGATIVE_INFINITY
        return k * ln(lambda) - lambda - lnFactorial(k)
    }

    /**
     * Returns the cumulative distribution function value at [k] for this Poisson distribution.
     *
     * Gives the probability of observing [k] or fewer events. Uses the regularized
     * upper incomplete gamma function for numerical stability rather than summing
     * individual probability masses.
     *
     * @param k the number of events at which to evaluate the cumulative probability.
     * @return the probability of [k] or fewer events, in the range `[0, 1]`.
     */
    override fun cdf(k: Int): Double {
        if (k < 0) return 0.0
        // P(X <= k) = Q(k+1, lambda) = regularizedGammaQ(k+1, lambda)
        return regularizedGammaQ((k + 1).toDouble(), lambda)
    }

    /**
     * Returns the quantile (inverse CDF) for the given probability [p] as an integer.
     *
     * Finds the smallest number of events k such that the cumulative probability of
     * k or fewer events is at least [p]. Uses a linear search starting from the
     * expected value (lambda) for efficiency.
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the smallest integer k at which `cdf(k) >= p`.
     */
    override fun quantileInt(p: Double): Int {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return 0
        // Search from the mean
        var k = lambda.toInt()
        while (k > 0 && cdf(k - 1) >= p) k--
        while (cdf(k) < p) k++
        return k
    }

    /** The mean of this distribution, equal to [rate] (lambda). */
    override val mean: Double get() = lambda

    /** The variance of this distribution, equal to [rate] (lambda). For the Poisson, mean and variance are equal. */
    override val variance: Double get() = lambda

    /** The skewness of this distribution, equal to 1 / sqrt(lambda). Always positive (right-skewed). */
    override val skewness: Double get() = 1.0 / sqrt(lambda)

    /** The excess kurtosis of this distribution, equal to 1 / lambda. Always positive (leptokurtic). */
    override val kurtosis: Double get() = 1.0 / lambda

    /**
     * Returns the Shannon entropy of this Poisson distribution in nats.
     *
     * Computed by summing -pmf(k) * ln(pmf(k)) over all non-negative integers k until
     * the cumulative probability is within 1e-15 of 1.0, or a safety limit of 100,000
     * terms is reached.
     *
     * @return the entropy in nats. Always non-negative.
     */
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
     * Returns the survival function value at [k] for this Poisson distribution.
     *
     * Gives the probability of observing strictly more than [k] events. Uses the
     * regularized lower incomplete gamma function for numerical stability.
     *
     * @param k the number of events at which to evaluate the survival probability.
     * @return the probability of more than [k] events, in the range `[0, 1]`.
     */
    override fun sf(k: Int): Double {
        if (k < 0) return 1.0
        return regularizedGammaP((k + 1).toDouble(), lambda)
    }

    /**
     * Draws a single random integer from this Poisson distribution.
     *
     * For small rates (lambda < 30), uses Knuth's algorithm which multiplies uniform
     * random numbers until their product falls below exp(-lambda). For large rates,
     * uses a normal approximation with the result clamped to non-negative values.
     *
     * @param random the source of randomness.
     * @return a random non-negative integer drawn from this distribution.
     */
    override fun sample(random: Random): Int {
        // Knuth's algorithm for small lambda
        if (lambda < 30) {
            val l = exp(-lambda)
            var k = 0
            var p = 1.0
            do {
                k++
                p *= random.nextDouble()
            } while (p > l)
            return k - 1
        }
        // For large lambda, use normal approximation
        return NormalDistribution(lambda, sqrt(lambda)).sample(random).roundToInt().coerceAtLeast(0)
    }
}
