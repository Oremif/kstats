package org.oremif.kstats.distributions

import org.oremif.kstats.core.*
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Represents the chi-squared distribution, a continuous probability distribution defined on
 * the interval [0, +infinity).
 *
 * The chi-squared distribution describes the distribution of a sum of squared standard
 * normal random variables. It is one of the most widely used distributions in statistical
 * inference, appearing in chi-squared goodness-of-fit tests, tests of independence in
 * contingency tables, and confidence intervals for population variance. The single parameter
 * [degreesOfFreedom] determines the shape: higher values shift the distribution to the right
 * and make it more symmetric.
 *
 * Internally, this distribution is implemented as a special case of the Gamma distribution
 * with shape equal to half the degrees of freedom and rate equal to 0.5. The CDF, quantile,
 * and sampling methods all delegate to this underlying Gamma parameterization.
 *
 * ### Example:
 * ```kotlin
 * val dist = ChiSquaredDistribution(degreesOfFreedom = 5.0)
 * dist.mean      // 5.0 (equal to the degrees of freedom)
 * dist.variance  // 10.0 (twice the degrees of freedom)
 * dist.pdf(3.0)  // 0.1542... (density at x = 3)
 * dist.cdf(11.07) // 0.95 (approximately)
 *
 * // Quantile for a chi-squared test critical value
 * dist.quantile(0.95) // 11.07... (95th percentile with 5 df)
 * ```
 *
 * @param degreesOfFreedom the number of degrees of freedom. Must be positive.
 * @see ContinuousDistribution
 * @see GammaDistribution
 */
public class ChiSquaredDistribution(
    public val degreesOfFreedom: Double
) : ContinuousDistribution {

    init {
        if (degreesOfFreedom <= 0.0) throw InvalidParameterException("Degrees of freedom must be positive, got $degreesOfFreedom")
    }

    private val df = degreesOfFreedom
    private val halfDf = df / 2.0

    private val gammaDelegate: GammaDistribution by lazy { GammaDistribution(halfDf, 0.5) }

    /**
     * Computes the probability density at [x].
     *
     * Returns zero for negative values. At x = 0, the density is 0.5 when degrees of freedom
     * equals 2, infinite when less than 2, and zero when greater than 2.
     *
     * @param x the point at which to evaluate the density.
     * @return the probability density at [x]. Always non-negative.
     */
    override fun pdf(x: Double): Double {
        if (x < 0.0) return 0.0
        if (x == 0.0) return if (df == 2.0) 0.5 else if (df < 2.0) Double.POSITIVE_INFINITY else 0.0
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
        if (x < 0.0 || x == Double.POSITIVE_INFINITY) return Double.NEGATIVE_INFINITY
        if (x == 0.0) return when {
            df == 2.0 -> -ln(2.0)
            df < 2.0 -> Double.POSITIVE_INFINITY
            else -> Double.NEGATIVE_INFINITY
        }
        return (halfDf - 1.0) * ln(x) - x / 2.0 - halfDf * ln(2.0) - lnGamma(halfDf)
    }

    /**
     * Computes the cumulative distribution function at [x].
     *
     * Returns the probability that a chi-squared-distributed random variable is less than
     * or equal to [x], evaluated via the regularized lower incomplete gamma function.
     *
     * @param x the point at which to evaluate the CDF.
     * @return the cumulative probability at [x], in the range [0, 1].
     */
    override fun cdf(x: Double): Double {
        if (x <= 0.0) return 0.0
        return regularizedGammaP(halfDf, x / 2.0)
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
        return regularizedGammaQ(halfDf, x / 2.0)
    }

    /**
     * Computes the quantile (inverse CDF) for the given probability [p].
     *
     * Uses Newton's method seeded with a Wilson-Hilferty normal approximation for degrees
     * of freedom greater than 2, or the midpoint of the distribution for smaller values.
     *
     * @param p the cumulative probability, must be in [0, 1].
     * @return the value x >= 0 such that `cdf(x) = p`.
     */
    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return 0.0
        if (p == 1.0) return Double.POSITIVE_INFINITY
        val guess = if (df > 2) {
            val z = NormalDistribution.STANDARD.quantile(p)
            val w = 2.0 / (9.0 * df)
            df * (1.0 - w + z * sqrt(w)).pow(3.0).coerceAtLeast(0.001)
        } else {
            df * 0.5
        }
        return findQuantile(p, ::cdf, ::pdf, guess, lowerBound = 1e-15)
    }

    /** The differential entropy of this distribution. */
    override val entropy: Double
        get() =
            halfDf + ln(2.0) + lnGamma(halfDf) + (1.0 - halfDf) * digamma(halfDf)

    /** The mean of this distribution, equal to the degrees of freedom. */
    override val mean: Double get() = df

    /** The variance of this distribution, equal to twice the degrees of freedom. */
    override val variance: Double get() = 2.0 * df

    /** The skewness of this distribution, which decreases as degrees of freedom increases. */
    override val skewness: Double get() = sqrt(8.0 / df)

    /** The excess kurtosis of this distribution, which decreases as degrees of freedom increases. */
    override val kurtosis: Double get() = 12.0 / df // excess

    /**
     * Draws a single random value from this chi-squared distribution.
     *
     * Delegates to the Gamma distribution with shape = df/2 and rate = 0.5, which is
     * mathematically equivalent.
     *
     * @param random the source of randomness.
     * @return a random non-negative value drawn from this distribution.
     */
    override fun sample(random: Random): Double {
        return gammaDelegate.sample(random)
    }
}
