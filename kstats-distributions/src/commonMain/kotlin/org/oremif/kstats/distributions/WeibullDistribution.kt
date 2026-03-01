package org.oremif.kstats.distributions

import org.oremif.kstats.core.EULER_MASCHERONI
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.core.gamma
import kotlin.math.*
import kotlin.random.Random

/**
 * Represents the Weibull distribution.
 *
 * The Weibull distribution is widely used in reliability engineering, survival analysis, and
 * failure modeling. It generalizes the exponential distribution: when [shape] equals 1 the
 * Weibull reduces to an exponential distribution with rate `1 / scale`. A shape less than 1
 * indicates a decreasing failure rate (early-life failures), a shape of 1 indicates a constant
 * failure rate (random failures), and a shape greater than 1 indicates an increasing failure
 * rate (wear-out failures).
 *
 * The distribution is supported on the interval from zero to positive infinity. The [scale]
 * parameter stretches or compresses the distribution along the x-axis, while the [shape]
 * parameter controls how steeply the density rises and falls.
 *
 * Statistical properties (mean, variance, skewness, kurtosis) are computed using the gamma
 * function. The quantile function has a closed-form expression, so sampling is performed via
 * inverse CDF (no iterative root-finding is needed).
 *
 * ### Example:
 * ```kotlin
 * val weibull = WeibullDistribution(shape = 2.0, scale = 1.0)
 * weibull.pdf(0.5)           // 0.7788... (density at x = 0.5)
 * weibull.cdf(1.0)           // 0.6321... (probability that X <= 1)
 * weibull.quantile(0.5)      // 0.8326... (median)
 * weibull.mean               // 0.8862... (gamma(1.5) for shape=2, scale=1)
 * weibull.sample(Random(42)) // a single random draw from the distribution
 * ```
 *
 * @param shape the shape parameter (often denoted k), controlling the failure rate behavior. Must be positive.
 * @param scale the scale parameter (often denoted lambda), stretching the distribution along the x-axis. Must be positive. Defaults to 1.0.
 */
public class WeibullDistribution(
    public val shape: Double,
    public val scale: Double = 1.0
) : ContinuousDistribution {

    init {
        if (!shape.isFinite() || shape <= 0.0) throw InvalidParameterException("shape must be finite and positive, got $shape")
        if (!scale.isFinite() || scale <= 0.0) throw InvalidParameterException("scale must be finite and positive, got $scale")
    }

    private val k = shape
    private val lambda = scale

    /**
     * Returns the probability density at [x].
     *
     * The density is zero for negative values. At zero the behavior depends on [shape]: the
     * density is infinite when shape is less than 1, equals `1 / scale` when shape is exactly 1,
     * and is zero when shape exceeds 1.
     *
     * @param x the point at which to evaluate the density.
     * @return the probability density at [x]. Always non-negative.
     */
    override fun pdf(x: Double): Double {
        if (x < 0.0) return 0.0
        if (x == 0.0) return if (k == 1.0) 1.0 / lambda else if (k < 1.0) Double.POSITIVE_INFINITY else 0.0
        return exp(logPdf(x))
    }

    /**
     * Returns the natural logarithm of the probability density at [x].
     *
     * Computed directly rather than as `ln(pdf(x))` to avoid precision loss with very small
     * density values.
     *
     * @param x the point at which to evaluate the log-density.
     * @return the natural log of the probability density, or [Double.NEGATIVE_INFINITY] when [x] is negative.
     */
    override fun logPdf(x: Double): Double {
        if (x < 0.0 || x == Double.POSITIVE_INFINITY) return Double.NEGATIVE_INFINITY
        if (x == 0.0) return when {
            k < 1.0 -> Double.POSITIVE_INFINITY
            k == 1.0 -> -ln(lambda)
            else -> Double.NEGATIVE_INFINITY
        }
        val xNorm = x / lambda
        return ln(k / lambda) + (k - 1.0) * ln(xNorm) - xNorm.pow(k)
    }

    /**
     * Returns the cumulative distribution function value at [x].
     *
     * Gives the probability that a random variable drawn from this Weibull distribution is
     * less than or equal to [x]. Uses a closed-form expression.
     *
     * @param x the point at which to evaluate the cumulative probability.
     * @return the probability that a value is less than or equal to [x], in the range `[0, 1]`.
     */
    override fun cdf(x: Double): Double {
        if (x <= 0.0) return 0.0
        return -expm1(-(x / lambda).pow(k))
    }

    /**
     * Returns the survival function value at [x], equal to `1 - cdf(x)`.
     *
     * Computed directly as an exponential to maintain precision when the CDF is close to 1.
     *
     * @param x the point at which to evaluate the survival probability.
     * @return the probability that a value exceeds [x], in the range `[0, 1]`.
     */
    override fun sf(x: Double): Double {
        if (x <= 0.0) return 1.0
        return exp(-(x / lambda).pow(k))
    }

    /**
     * Returns the quantile (inverse CDF) for the given probability [p].
     *
     * Uses a closed-form expression based on the inverse of the CDF. No iterative
     * root-finding is needed.
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the value x at which `cdf(x) = p`. Returns 0.0 for `p = 0` and positive infinity for `p = 1`.
     */
    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return 0.0
        if (p == 1.0) return Double.POSITIVE_INFINITY
        return lambda * (-ln1p(-p)).pow(1.0 / k)
    }

    /** Returns the mean, computed using the gamma function evaluated at `1 + 1/shape`. */
    override val mean: Double get() = lambda * gamma(1.0 + 1.0 / k)

    /** Returns the variance, computed using the gamma function. */
    override val variance: Double
        get() {
            val g1 = gamma(1.0 + 1.0 / k)
            val g2 = gamma(1.0 + 2.0 / k)
            return lambda * lambda * (g2 - g1 * g1)
        }

    /** Returns the skewness, computed using the gamma function. */
    override val skewness: Double
        get() {
            val g1 = gamma(1.0 + 1.0 / k)
            val g3 = gamma(1.0 + 3.0 / k)
            val mu = lambda * g1
            val sigma = sqrt(variance)
            return (lambda * lambda * lambda * g3 - 3.0 * mu * sigma * sigma - mu * mu * mu) / (sigma * sigma * sigma)
        }

    /** Returns the excess kurtosis, computed using the gamma function. */
    override val kurtosis: Double
        get() {
            val g1 = gamma(1.0 + 1.0 / k)
            val g2 = gamma(1.0 + 2.0 / k)
            val g3 = gamma(1.0 + 3.0 / k)
            val g4 = gamma(1.0 + 4.0 / k)
            val mu2 = g2 - g1 * g1
            return (-6.0 * g1 * g1 * g1 * g1 + 12.0 * g1 * g1 * g2 - 3.0 * g2 * g2 - 4.0 * g1 * g3 + g4) / (mu2 * mu2) - 3.0
        }

    /** Returns the Shannon entropy of this distribution in nats, using the Euler-Mascheroni constant. */
    override val entropy: Double = EULER_MASCHERONI * (1.0 - 1.0 / k) + ln(lambda / k) + 1.0

    /**
     * Draws a single random value from this Weibull distribution.
     *
     * Uses the inverse CDF method, which is exact because the quantile function has a
     * closed-form expression.
     *
     * @param random the source of randomness.
     * @return a random value drawn from this distribution.
     */
    override fun sample(random: Random): Double = quantile(random.nextDouble())
}
