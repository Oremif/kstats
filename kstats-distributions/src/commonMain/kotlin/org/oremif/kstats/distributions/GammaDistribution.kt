package org.oremif.kstats.distributions

import org.oremif.kstats.core.digamma
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.core.findQuantile
import org.oremif.kstats.core.lnGamma
import org.oremif.kstats.core.regularizedGammaP
import org.oremif.kstats.core.regularizedGammaQ
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Represents the Gamma distribution, a continuous probability distribution defined on the
 * interval [0, +infinity).
 *
 * The Gamma distribution generalizes the exponential distribution to allow for a variable
 * number of waiting periods. It is commonly used to model waiting times, rainfall amounts,
 * insurance claims, and other non-negative continuous quantities. When the [shape] parameter
 * is 1, the distribution reduces to an exponential distribution with the given [rate].
 *
 * The CDF is computed via the regularized incomplete gamma function. Quantiles are found
 * using Newton's method with a Wilson-Hilferty normal approximation as the initial guess.
 * Random samples are generated using the Marsaglia-Tsang method for shape >= 1, with a
 * transformation trick for shape < 1.
 *
 * ### Example:
 * ```kotlin
 * val dist = GammaDistribution(shape = 3.0, rate = 2.0)
 * dist.mean      // 1.5 (shape / rate)
 * dist.variance  // 0.75 (shape / rate^2)
 * dist.pdf(1.0)  // 0.5413... (density at x = 1)
 * dist.cdf(2.0)  // 0.8008... (probability of being at most 2)
 *
 * // Exponential distribution as a special case
 * val expo = GammaDistribution(shape = 1.0, rate = 0.5)
 * expo.mean // 2.0
 * ```
 *
 * @param shape the shape parameter (sometimes called k or alpha). Must be positive.
 * @param rate the rate parameter (inverse of scale). Defaults to 1.0, meaning the scale is 1. Must be positive.
 * @see ContinuousDistribution
 * @see ChiSquaredDistribution
 */
public class GammaDistribution(
    public val shape: Double,
    public val rate: Double = 1.0
) : ContinuousDistribution {

    init {
        if (shape <= 0.0) throw InvalidParameterException("shape must be positive, got $shape")
        if (rate <= 0.0) throw InvalidParameterException("rate must be positive, got $rate")
    }

    private val scale = 1.0 / rate

    /**
     * Computes the probability density at [x].
     *
     * Returns zero for negative values. At x = 0, the density depends on the shape
     * parameter: it equals [rate] when shape is 1, is infinite when shape is less than 1,
     * and is zero when shape is greater than 1.
     *
     * @param x the point at which to evaluate the density.
     * @return the probability density at [x]. Always non-negative.
     */
    override fun pdf(x: Double): Double {
        if (x < 0.0) return 0.0
        if (x == 0.0) return when {
            shape == 1.0 -> rate
            shape < 1.0 -> Double.POSITIVE_INFINITY
            else -> 0.0
        }
        return exp(logPdf(x))
    }

    /**
     * Computes the natural logarithm of the probability density at [x].
     *
     * Returns [Double.NEGATIVE_INFINITY] for non-positive values.
     *
     * @param x the point at which to evaluate the log-density.
     * @return the natural log of the density at [x].
     */
    override fun logPdf(x: Double): Double {
        if (x <= 0.0) return Double.NEGATIVE_INFINITY
        return (shape - 1.0) * ln(x) - x * rate + shape * ln(rate) - lnGamma(shape)
    }

    /**
     * Computes the cumulative distribution function at [x].
     *
     * Returns the probability that a Gamma-distributed random variable is less than or
     * equal to [x], evaluated via the regularized lower incomplete gamma function.
     *
     * @param x the point at which to evaluate the CDF.
     * @return the cumulative probability at [x], in the range [0, 1].
     */
    override fun cdf(x: Double): Double {
        if (x <= 0.0) return 0.0
        return regularizedGammaP(shape, x * rate)
    }

    /**
     * Computes the survival function (one minus the CDF) at [x].
     *
     * Uses the regularized upper incomplete gamma function for improved numerical accuracy
     * in the upper tail.
     *
     * @param x the point at which to evaluate the survival function.
     * @return the probability that a value exceeds [x], in the range [0, 1].
     */
    override fun sf(x: Double): Double {
        if (x <= 0.0) return 1.0
        return regularizedGammaQ(shape, x * rate)
    }

    /**
     * Computes the quantile (inverse CDF) for the given probability [p].
     *
     * Uses Newton's method seeded with a Wilson-Hilferty normal approximation for shape >= 1,
     * or a power-law approximation for shape < 1.
     *
     * @param p the cumulative probability, must be in [0, 1].
     * @return the value x >= 0 such that `cdf(x) = p`.
     */
    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return 0.0
        if (p == 1.0) return Double.POSITIVE_INFINITY
        val guess = if (shape >= 1.0) {
            val z = NormalDistribution.STANDARD.quantile(p)
            val w = 2.0 / (9.0 * shape)
            (shape * (1.0 - w + z * sqrt(w)).pow(3.0)).coerceAtLeast(0.001) / rate
        } else {
            (shape * p.pow(1.0 / shape)) / rate
        }
        return findQuantile(p, ::cdf, ::pdf, guess, lowerBound = 1e-15)
    }

    /** The differential entropy of this distribution. */
    override val entropy: Double get() =
        shape - ln(rate) + lnGamma(shape) + (1.0 - shape) * digamma(shape)

    /** The mean of this distribution, equal to shape / rate. */
    override val mean: Double get() = shape / rate

    /** The variance of this distribution, equal to shape / rate squared. */
    override val variance: Double get() = shape / (rate * rate)

    /** The skewness of this distribution, which decreases as shape increases. */
    override val skewness: Double get() = 2.0 / sqrt(shape)

    /** The excess kurtosis of this distribution, which decreases as shape increases. */
    override val kurtosis: Double get() = 6.0 / shape // excess

    /**
     * Draws a single random value from this Gamma distribution.
     *
     * Uses the Marsaglia-Tsang method for shape >= 1. For shape < 1, draws from
     * Gamma(shape + 1) and applies a power transformation to correct the distribution.
     *
     * @param random the source of randomness.
     * @return a random non-negative value drawn from this distribution.
     */
    override fun sample(random: Random): Double {
        // Marsaglia and Tsang's method for shape >= 1
        // For shape < 1: use Gamma(shape+1)*U^(1/shape)
        if (shape < 1.0) {
            val g1 = GammaDistribution(shape + 1.0, 1.0).sample(random)
            return g1 * random.nextDouble().pow(1.0 / shape) / rate
        }

        val d = shape - 1.0 / 3.0
        val c = 1.0 / sqrt(9.0 * d)

        while (true) {
            var x: Double
            var v: Double
            do {
                x = NormalDistribution.STANDARD.sample(random)
                v = 1.0 + c * x
            } while (v <= 0.0)

            v = v * v * v
            val u = random.nextDouble()

            if (u < 1.0 - 0.0331 * x * x * x * x) {
                return d * v / rate
            }

            if (ln(u) < 0.5 * x * x + d * (1.0 - v + ln(v))) {
                return d * v / rate
            }
        }
    }
}
