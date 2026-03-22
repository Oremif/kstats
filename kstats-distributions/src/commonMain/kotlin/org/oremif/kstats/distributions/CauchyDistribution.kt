package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.PI
import kotlin.math.atan
import kotlin.math.ln
import kotlin.math.tan
import kotlin.random.Random

/**
 * Represents the Cauchy distribution (also known as the Lorentz distribution).
 *
 * The Cauchy distribution is a symmetric, heavy-tailed continuous distribution centered at
 * [location]. It is famous for having no defined mean, variance, or higher moments -- all of
 * these properties return [Double.NaN]. This happens because the tails are so heavy that the
 * integrals defining these moments diverge.
 *
 * A common way the Cauchy distribution arises in practice is as the ratio of two independent
 * standard normal random variables. It also describes the distribution of the tangent of a
 * uniformly distributed angle and appears in physics as the line shape of spectral lines
 * (where it is called a Lorentzian).
 *
 * The distribution is supported on the entire real line, from negative infinity to positive
 * infinity. The [scale] parameter controls how spread out the distribution is around the
 * [location] (analogous to standard deviation in a normal distribution, but the Cauchy
 * distribution decays much more slowly in the tails).
 *
 * The quantile function has a closed-form expression using the tangent function, so sampling
 * is performed via inverse CDF (no iterative root-finding is needed).
 *
 * ### Example:
 * ```kotlin
 * val cauchy = CauchyDistribution(location = 0.0, scale = 1.0)
 * cauchy.pdf(0.0)           // 0.3183... (peak density at the location)
 * cauchy.cdf(0.0)           // 0.5 (symmetric around the location)
 * cauchy.quantile(0.75)     // 1.0 (third quartile of the standard Cauchy)
 * cauchy.mean               // NaN (the mean is undefined)
 * cauchy.entropy             // 2.5310... (entropy is defined even though moments are not)
 * cauchy.sample(Random(42)) // a single random draw
 * ```
 *
 * @param location the center of the distribution, where the density peaks. Defaults to 0.0.
 * @param scale the half-width at half-maximum, controlling the spread of the distribution. Must be positive. Defaults to 1.0.
 */
public class CauchyDistribution(
    public val location: Double = 0.0,
    public val scale: Double = 1.0
) : ContinuousDistribution {

    init {
        if (scale.isNaN() || scale <= 0.0) throw InvalidParameterException("scale must be positive, got $scale")
    }

    /**
     * Provides the standard Cauchy distribution instance.
     */
    public companion object {
        private val LOG_PI = ln(PI)

        /** The standard Cauchy distribution with location 0 and scale 1. */
        public val STANDARD: CauchyDistribution = CauchyDistribution(0.0, 1.0)
    }

    /**
     * Returns the probability density at [x].
     *
     * The Cauchy density has its peak at [location] and decreases slowly in the tails,
     * much more slowly than a normal distribution.
     *
     * @param x the point at which to evaluate the density.
     * @return the probability density at [x]. Always positive.
     */
    override fun pdf(x: Double): Double {
        val z = (x - location) / scale
        return 1.0 / (PI * scale * (1.0 + z * z))
    }

    /**
     * Returns the natural logarithm of the probability density at [x].
     *
     * Computed directly rather than as `ln(pdf(x))` to avoid precision loss.
     *
     * @param x the point at which to evaluate the log-density.
     * @return the natural log of the probability density at [x].
     */
    override fun logPdf(x: Double): Double {
        val z = (x - location) / scale
        return -LOG_PI - ln(scale) - ln(1.0 + z * z)
    }

    /**
     * Returns the cumulative distribution function value at [x].
     *
     * Gives the probability that a random variable drawn from this Cauchy distribution is less
     * than or equal to [x]. Computed using the arctangent function.
     *
     * @param x the point at which to evaluate the cumulative probability.
     * @return the probability that a value is less than or equal to [x], in the range `[0, 1]`.
     */
    override fun cdf(x: Double): Double = 0.5 + atan((x - location) / scale) / PI

    /**
     * Returns the survival function value at [x], equal to `1 - cdf(x)`.
     *
     * Computed directly using arctangent to maintain precision when the CDF is close to 1.
     *
     * @param x the point at which to evaluate the survival probability.
     * @return the probability that a value exceeds [x], in the range `[0, 1]`.
     */
    override fun sf(x: Double): Double {
        val z = (x - location) / scale
        // Use atan(1/z) for z > 0 to avoid catastrophic cancellation in 0.5 - atan(z)/π when atan(z) → π/2
        return if (z > 0) atan(1.0 / z) / PI else 0.5 - atan(z) / PI
    }

    /** Returns the Shannon entropy of this distribution in nats. The Cauchy entropy is always defined even though the moments are not. */
    override val entropy: Double get() = ln(4.0 * PI * scale)

    /**
     * Returns the quantile (inverse CDF) for the given probability [p].
     *
     * Uses the closed-form expression based on the tangent function. No iterative
     * root-finding is needed.
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the value x at which `cdf(x) = p`. Returns negative infinity for `p = 0` and positive infinity for `p = 1`.
     */
    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return Double.NEGATIVE_INFINITY
        if (p == 1.0) return Double.POSITIVE_INFINITY
        return location + scale * tan(PI * (p - 0.5))
    }

    /** Returns [Double.NaN] because the Cauchy distribution has no defined mean. */
    override val mean: Double get() = Double.NaN

    /** Returns [Double.NaN] because the Cauchy distribution has no defined variance. */
    override val variance: Double get() = Double.NaN

    /** Returns [Double.NaN] because the Cauchy distribution has no defined standard deviation. */
    override val standardDeviation: Double get() = Double.NaN

    /** Returns [Double.NaN] because the Cauchy distribution has no defined skewness. */
    override val skewness: Double get() = Double.NaN

    /** Returns [Double.NaN] because the Cauchy distribution has no defined kurtosis. */
    override val kurtosis: Double get() = Double.NaN

    /**
     * Draws a single random value from this Cauchy distribution.
     *
     * Uses the inverse CDF method, which is exact because the quantile function has a
     * closed-form expression.
     *
     * @param random the source of randomness.
     * @return a random value drawn from this distribution.
     */
    override fun sample(random: Random): Double {
        val u = random.nextDouble().coerceAtLeast(Double.MIN_VALUE)
        return location + scale * tan(PI * (u - 0.5))
    }
}
