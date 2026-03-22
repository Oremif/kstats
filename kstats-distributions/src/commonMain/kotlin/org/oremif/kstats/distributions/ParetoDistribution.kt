package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.expm1
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * The Pareto Type I distribution, defined by its [shape] (α) and [scale] (xm) parameters.
 *
 * The Pareto distribution is a heavy-tailed, right-skewed continuous distribution
 * originally used to model the distribution of wealth. It is widely used in
 * actuarial science, reliability engineering, and modeling phenomena that follow
 * power laws. The distribution has support on `[scale, +∞)`.
 *
 * ### Example:
 * ```kotlin
 * val dist = ParetoDistribution(shape = 2.0, scale = 1.0)
 * dist.pdf(1.0)       // 2.0 (peak density at the scale)
 * dist.cdf(2.0)       // 0.75
 * dist.quantile(0.5)  // 1.4142... (√2)
 * ```
 *
 * @param shape the tail index (α). Must be positive. Defaults to `1.0`.
 * @param scale the minimum value / location (xm). Must be positive. Defaults to `1.0`.
 */
public class ParetoDistribution(
    public val shape: Double = 1.0,
    public val scale: Double = 1.0,
) : ContinuousDistribution {

    init {
        if (shape <= 0.0) throw InvalidParameterException("shape must be positive, got $shape")
        if (scale <= 0.0) throw InvalidParameterException("scale must be positive, got $scale")
    }

    public companion object {
        /** Standard Pareto distribution with shape=1 and scale=1. */
        public val STANDARD: ParetoDistribution = ParetoDistribution(1.0, 1.0)
    }

    /**
     * Returns the probability density at [x] for this Pareto distribution.
     *
     * For x ≥ scale: `α * xm^α / x^(α+1)`. Returns 0 for x < scale.
     *
     * @param x the point at which to evaluate the density.
     * @return the probability density at [x]. Always non-negative.
     */
    override fun pdf(x: Double): Double {
        if (x < scale) return 0.0
        return shape * scale.pow(shape) / x.pow(shape + 1.0)
    }

    /**
     * Returns the natural logarithm of the probability density at [x] for this Pareto distribution.
     *
     * Computed directly in log-space: `ln(α) + α·ln(xm) - (α+1)·ln(x)`.
     *
     * @param x the point at which to evaluate the log-density.
     * @return the natural log of the probability density at [x].
     */
    override fun logPdf(x: Double): Double {
        if (x < scale) return Double.NEGATIVE_INFINITY
        return ln(shape) + shape * ln(scale) - (shape + 1.0) * ln(x)
    }

    /**
     * Returns the cumulative distribution function value at [x] for this Pareto distribution.
     *
     * For x ≥ scale: `1 - (xm/x)^α`. Returns 0 for x < scale.
     * Computed via `−expm1(α·ln(xm/x))` to avoid catastrophic cancellation when x ≈ scale.
     *
     * @param x the point at which to evaluate the cumulative probability.
     * @return the probability that a value is less than or equal to [x], in the range `[0, 1]`.
     */
    override fun cdf(x: Double): Double {
        if (x < scale) return 0.0
        return -expm1(shape * ln(scale / x))
    }

    /**
     * Returns the survival function value at [x] for this Pareto distribution.
     *
     * Computed directly as `(xm/x)^α` for improved numerical precision.
     *
     * @param x the point at which to evaluate the survival probability.
     * @return the probability that a value is greater than [x], in the range `[0, 1]`.
     */
    override fun sf(x: Double): Double {
        if (x < scale) return 1.0
        return (scale / x).pow(shape)
    }

    /**
     * Returns the quantile (inverse CDF) for the given probability [p].
     *
     * Computed as `xm * (1-p)^(-1/α)`. Returns scale for p=0 and +∞ for p=1.
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the value x at which `cdf(x) = p`.
     */
    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return scale
        if (p == 1.0) return Double.POSITIVE_INFINITY
        return scale * (1.0 - p).pow(-1.0 / shape)
    }

    /** The mean of this distribution. Equal to `α*xm/(α-1)` if α > 1, else +∞. */
    override val mean: Double get() = if (shape > 1.0) shape * scale / (shape - 1.0) else Double.POSITIVE_INFINITY

    /** The variance of this distribution. Equal to `xm²·α / ((α-1)²·(α-2))` if α > 2, else +∞. */
    override val variance: Double
        get() = if (shape > 2.0) {
            scale * scale * shape / ((shape - 1.0) * (shape - 1.0) * (shape - 2.0))
        } else {
            Double.POSITIVE_INFINITY
        }

    /** The skewness of this distribution. Defined only for α > 3. */
    override val skewness: Double
        get() = if (shape > 3.0) {
            2.0 * (1.0 + shape) / (shape - 3.0) * sqrt((shape - 2.0) / shape)
        } else {
            Double.NaN
        }

    /** The excess (Fisher) kurtosis of this distribution. Defined only for α > 4. */
    override val kurtosis: Double
        get() = if (shape > 4.0) {
            6.0 * (shape * shape * shape + shape * shape - 6.0 * shape - 2.0) /
                (shape * (shape - 3.0) * (shape - 4.0))
        } else {
            Double.NaN
        }

    /** The Shannon entropy of this distribution in nats. Equal to `ln(xm/α) + 1/α + 1`. */
    override val entropy: Double = ln(scale / shape) + 1.0 / shape + 1.0

    /**
     * Draws a single random value from this Pareto distribution using inverse CDF sampling.
     *
     * @param random the source of randomness.
     * @return a random value drawn from this distribution (always ≥ scale).
     */
    override fun sample(random: Random): Double {
        val u = random.nextDouble().coerceIn(Double.MIN_VALUE, 1.0 - Double.MIN_VALUE)
        return scale * u.pow(-1.0 / shape)
    }
}
