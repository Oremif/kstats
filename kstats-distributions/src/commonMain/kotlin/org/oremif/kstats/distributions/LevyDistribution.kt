package org.oremif.kstats.distributions

import org.oremif.kstats.core.EULER_MASCHERONI
import org.oremif.kstats.core.erf
import org.oremif.kstats.core.erfc
import org.oremif.kstats.core.erfcInv
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Represents the Levy distribution, a heavy-tailed, right-skewed continuous probability
 * distribution supported on `[mu, +infinity)`.
 *
 * The Levy distribution is one of only three stable distributions whose probability density
 * has a closed-form expression (the other two being the normal and Cauchy distributions).
 * It arises naturally in physics as the distribution of first-passage times of Brownian
 * motion and is used in finance to model heavy-tailed phenomena such as extreme market
 * movements.
 *
 * The distribution is parameterized by a location parameter [mu] and a scale parameter [c].
 * Because the tails are extremely heavy, the mean and variance are both infinite, and the
 * skewness and excess kurtosis are undefined (returned as [Double.NaN]).
 *
 * The CDF and survival function are expressed in terms of the complementary error function
 * and error function respectively, and the quantile function uses the inverse complementary
 * error function, so no iterative root-finding is needed.
 *
 * ### Example:
 * ```kotlin
 * val levy = LevyDistribution(mu = 0.0, c = 1.0)
 * levy.mean                // Infinity
 * levy.variance            // Infinity
 * levy.pdf(1.0)            // 0.2420 (density at x = 1)
 * levy.cdf(2.0)            // 0.4795
 * levy.quantile(0.5)       // 2.1981 (the median)
 * levy.sample(Random(42))  // a single random draw from Levy(0, 1)
 *
 * // Standard Levy distribution (mu=0, c=1)
 * val std = LevyDistribution.STANDARD
 * std.cdf(1.0)             // 0.3173
 * ```
 *
 * @param mu the location parameter, defining the left endpoint of the support. Default is `0.0`.
 * @param c the scale parameter, controlling the spread of the distribution. Must be positive.
 * Default is `1.0`.
 */
public class LevyDistribution(
    public val mu: Double = 0.0,
    public val c: Double = 1.0,
) : ContinuousDistribution {

    init {
        if (c <= 0.0) throw InvalidParameterException("c must be positive, got $c")
    }

    /**
     * Computes the probability density at [x] using the Levy density formula.
     *
     * Returns zero for any [x] less than or equal to [mu], since the distribution is
     * supported only on `(mu, +infinity)`.
     *
     * @param x the point at which to evaluate the density.
     * @return the probability density at [x], zero for `x <= mu` and positive otherwise.
     */
    override fun pdf(x: Double): Double {
        if (x <= mu) return 0.0
        val diff = x - mu
        return sqrt(c / (2.0 * PI)) * exp(-c / (2.0 * diff)) / (diff * sqrt(diff))
    }

    /**
     * Computes the natural logarithm of the probability density at [x].
     *
     * Returns [Double.NEGATIVE_INFINITY] for any [x] less than or equal to [mu].
     * Computed directly rather than as `ln(pdf(x))` to avoid precision loss.
     *
     * @param x the point at which to evaluate the log-density.
     * @return the natural log of the density at [x], or [Double.NEGATIVE_INFINITY] for `x <= mu`.
     */
    override fun logPdf(x: Double): Double {
        if (x <= mu) return Double.NEGATIVE_INFINITY
        val diff = x - mu
        return 0.5 * (ln(c) - ln(2.0 * PI)) - c / (2.0 * diff) - 1.5 * ln(diff)
    }

    /**
     * Computes the cumulative distribution function at [x] using the complementary error function.
     *
     * Returns zero for any [x] less than or equal to [mu].
     *
     * @param x the point at which to evaluate the cumulative probability.
     * @return the probability that a value drawn from this distribution is less than or equal to [x].
     */
    override fun cdf(x: Double): Double {
        if (x <= mu) return 0.0
        return erfc(sqrt(c / (2.0 * (x - mu))))
    }

    /**
     * Computes the survival function at [x] using the error function directly.
     *
     * Returns one for any [x] less than or equal to [mu]. This override uses `erf` directly
     * rather than `1 - cdf(x)`, which provides better numerical accuracy in the upper tail.
     *
     * @param x the point at which to evaluate the survival probability.
     * @return the probability that a value drawn from this distribution is greater than [x].
     */
    override fun sf(x: Double): Double {
        if (x <= mu) return 1.0
        return erf(sqrt(c / (2.0 * (x - mu))))
    }

    /**
     * Computes the quantile (inverse CDF) for the given probability [p] using the inverse
     * complementary error function.
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the value x at which `cdf(x) = p`. Returns [mu] for `p = 0` and
     * [Double.POSITIVE_INFINITY] for `p = 1`.
     */
    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return mu
        if (p == 1.0) return Double.POSITIVE_INFINITY
        val z = erfcInv(p)
        return mu + c / (2.0 * z * z)
    }

    /** Returns [Double.POSITIVE_INFINITY] because the Levy distribution has infinite mean. */
    override val mean: Double get() = Double.POSITIVE_INFINITY

    /** Returns [Double.POSITIVE_INFINITY] because the Levy distribution has infinite variance. */
    override val variance: Double get() = Double.POSITIVE_INFINITY

    /** Returns [Double.NaN] because the skewness of the Levy distribution is undefined. */
    override val skewness: Double get() = Double.NaN

    /** Returns [Double.NaN] because the excess kurtosis of the Levy distribution is undefined. */
    override val kurtosis: Double get() = Double.NaN

    /** The differential entropy of this distribution in nats, computed from the scale parameter [c] and the Euler-Mascheroni constant. */
    override val entropy: Double = 0.5 * (1.0 + 3.0 * EULER_MASCHERONI + ln(16.0 * PI * c * c))

    /**
     * Draws a single random value from this Levy distribution using inverse CDF sampling.
     *
     * @param random the source of randomness.
     * @return a random value drawn from this distribution, always greater than [mu].
     */
    override fun sample(random: Random): Double = quantile(random.nextDouble())

    /** Provides the standard Levy distribution instance. */
    public companion object {
        /** The standard Levy distribution with location 0 and scale 1. */
        public val STANDARD: LevyDistribution = LevyDistribution(0.0, 1.0)
    }
}
