package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Represents the triangular distribution, a continuous probability distribution defined by a
 * lower limit [a], an upper limit [b], and a mode (peak) [c].
 *
 * The triangular distribution has a piecewise linear density that rises from zero at [a] to a
 * peak at [c], then falls back to zero at [b]. It is commonly used in simulation and risk
 * analysis when only the minimum, maximum, and most likely value of a quantity are known, making
 * it a practical alternative to the beta distribution when limited information is available.
 *
 * The PDF consists of two linear segments joined at the mode [c], the CDF is piecewise quadratic,
 * and the quantile function uses the inverse of those quadratic segments. Sampling is performed
 * via the inverse CDF method.
 *
 * ### Example:
 * ```kotlin
 * // Model project duration: min 2 weeks, most likely 5 weeks, max 10 weeks
 * val dist = TriangularDistribution(a = 2.0, b = 10.0, c = 5.0)
 * dist.mean                // 5.6667 (shifted toward the longer tail)
 * dist.variance            // 2.7222
 * dist.pdf(5.0)            // 0.25 (peak density at the mode)
 * dist.cdf(5.0)            // 0.375
 * dist.quantile(0.5)       // 5.2915 (the median)
 * dist.sample(Random(42))  // a single random draw from Triangular(2, 10, 5)
 *
 * // Symmetric triangular distribution
 * val sym = TriangularDistribution(a = 0.0, b = 1.0, c = 0.5)
 * sym.skewness             // 0.0
 * ```
 *
 * @param a the lower limit (minimum) of the distribution's support. Must be strictly less than [b].
 * @param b the upper limit (maximum) of the distribution's support. Must be strictly greater than [a].
 * @param c the mode (peak) of the distribution. Must satisfy `a <= c <= b`.
 */
public class TriangularDistribution(
    public val a: Double,
    public val b: Double,
    public val c: Double,
) : ContinuousDistribution {

    init {
        if (a >= b) throw InvalidParameterException("a must be less than b, got a=$a, b=$b")
        if (c !in a..b) throw InvalidParameterException("c must be in [a, b], got a=$a, b=$b, c=$c")
    }

    private val ba = b - a
    private val ca = c - a
    private val bc = b - c
    private val pc = ca / ba

    private companion object {
        private val SQRT2 = sqrt(2.0)
    }

    /**
     * Computes the probability density at [x] using the piecewise linear density function.
     *
     * Returns `2(x - a) / ((b - a)(c - a))` for `a <= x < c`, `2 / (b - a)` at `x = c`,
     * `2(b - x) / ((b - a)(b - c))` for `c < x <= b`, and `0.0` outside the support.
     *
     * @param x the point at which to evaluate the density.
     * @return the probability density at [x], always non-negative.
     */
    override fun pdf(x: Double): Double = when {
        x.isNaN() -> Double.NaN
        x !in a..b -> 0.0
        x < c -> 2.0 * (x - a) / (ba * ca)
        x > c -> 2.0 * (b - x) / (ba * bc)
        else -> 2.0 / ba // x == c
    }

    /**
     * Computes the natural logarithm of the probability density at [x].
     *
     * Uses the logarithmic form of the piecewise density to avoid computing the density and
     * then taking its log, which improves numerical stability for values near the boundaries.
     *
     * @param x the point at which to evaluate the log-density.
     * @return the natural log of the density at [x], or [Double.NEGATIVE_INFINITY] if [x] is
     * outside the support.
     */
    override fun logPdf(x: Double): Double = when {
        x.isNaN() -> Double.NaN
        x !in a..b -> Double.NEGATIVE_INFINITY
        x < c -> ln(2.0) + ln(x - a) - ln(ba) - ln(ca)
        x > c -> ln(2.0) + ln(b - x) - ln(ba) - ln(bc)
        else -> ln(2.0) - ln(ba) // x == c
    }

    /**
     * Computes the cumulative distribution function at [x] using the piecewise quadratic CDF.
     *
     * Returns `(x - a)^2 / ((b - a)(c - a))` for `a < x <= c` and
     * `1 - (b - x)^2 / ((b - a)(b - c))` for `c < x < b`.
     *
     * @param x the point at which to evaluate the cumulative probability.
     * @return the probability that a value drawn from this distribution is less than or equal to [x].
     */
    override fun cdf(x: Double): Double = when {
        x.isNaN() -> Double.NaN
        x <= a -> 0.0
        x <= c -> (x - a) * (x - a) / (ba * ca)
        x < b -> 1.0 - (b - x) * (b - x) / (ba * bc)
        else -> 1.0
    }

    /**
     * Computes the survival function (1 - CDF) at [x].
     *
     * This is computed directly from the piecewise quadratic form rather than as `1 - cdf(x)`,
     * which avoids catastrophic cancellation in the upper tail.
     *
     * @param x the point at which to evaluate the survival probability.
     * @return the probability that a value drawn from this distribution is greater than [x].
     */
    override fun sf(x: Double): Double = when {
        x.isNaN() -> Double.NaN
        x <= a -> 1.0
        x <= c -> 1.0 - (x - a) * (x - a) / (ba * ca)
        x < b -> (b - x) * (b - x) / (ba * bc)
        else -> 0.0
    }

    /**
     * Computes the quantile (inverse CDF) for the given probability [p].
     *
     * Uses the inverse of the piecewise quadratic CDF. For `p <= (c - a) / (b - a)`, the
     * result is computed from the left segment; otherwise, from the right segment.
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the value x at which `cdf(x) = p`. Returns [a] for `p = 0` and [b] for `p = 1`.
     */
    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        return when {
            p == 0.0 -> a
            p == 1.0 -> b
            p <= pc -> a + sqrt(p * ba * ca)
            else -> b - sqrt((1.0 - p) * ba * bc)
        }
    }

    /** The mean of this distribution, equal to `(a + b + c) / 3`. */
    override val mean: Double get() = (a + b + c) / 3.0

    /** The variance of this distribution, equal to `(a^2 + b^2 + c^2 - ab - ac - bc) / 18`. */
    override val variance: Double get() = (a * a + b * b + c * c - a * b - a * c - b * c) / 18.0

    /** The skewness of this distribution, computed from [a], [b], and [c]. Zero when the distribution is symmetric (`c = (a + b) / 2`). */
    override val skewness: Double
        get() {
            val num = SQRT2 * (a + b - 2.0 * c) * (2.0 * a - b - c) * (a - 2.0 * b + c)
            val den = 5.0 * (a * a + b * b + c * c - a * b - a * c - b * c).let { it * sqrt(it) }
            return num / den
        }

    /** The excess kurtosis of this distribution, always `-0.6` for any triangular distribution regardless of parameters. */
    override val kurtosis: Double get() = -0.6

    /** The differential entropy of this distribution in nats, equal to `0.5 + ln((b - a) / 2)`. */
    override val entropy: Double get() = 0.5 + ln(ba / 2.0)

    /**
     * Draws a single random value from this triangular distribution using inverse CDF sampling.
     *
     * @param random the source of randomness.
     * @return a random value in `[a, b]` drawn from this distribution.
     */
    override fun sample(random: Random): Double = quantile(random.nextDouble())
}
