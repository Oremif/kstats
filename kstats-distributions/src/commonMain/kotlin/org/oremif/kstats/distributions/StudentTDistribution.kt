package org.oremif.kstats.distributions

import org.oremif.kstats.core.*
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.*
import kotlin.random.Random

/**
 * Represents Student's t-distribution, a continuous probability distribution defined on
 * the entire real line.
 *
 * Student's t-distribution arises when estimating the mean of a normally distributed
 * population whose standard deviation is unknown and must be estimated from the data.
 * It is the foundation of t-tests and confidence intervals for means. The distribution
 * resembles the standard normal distribution but has heavier tails, which accounts for
 * the additional uncertainty from estimating the standard deviation. As [degreesOfFreedom]
 * increases, the t-distribution converges to the standard normal distribution.
 *
 * The CDF is computed via the regularized incomplete beta function. Quantiles are found
 * using Newton's method seeded with the corresponding standard normal quantile. Random
 * samples are generated as the ratio of a standard normal draw to the square root of an
 * independent chi-squared draw divided by its degrees of freedom.
 *
 * ### Example:
 * ```kotlin
 * val dist = StudentTDistribution(degreesOfFreedom = 10.0)
 * dist.mean      // 0.0 (symmetric about zero)
 * dist.variance  // 1.25 (greater than 1, reflecting heavier tails than normal)
 * dist.pdf(0.0)  // 0.3891... (slightly less than normal's 0.3989)
 * dist.cdf(2.228) // 0.975 (approximately; critical value for 97.5th percentile)
 *
 * // Critical value for a two-sided 95% confidence interval with 10 df
 * dist.quantile(0.975) // 2.228...
 * ```
 *
 * @param degreesOfFreedom the number of degrees of freedom. Must be positive.
 * @see ContinuousDistribution
 * @see ChiSquaredDistribution
 * @see NormalDistribution
 */
public class StudentTDistribution(
    public val degreesOfFreedom: Double
) : ContinuousDistribution {

    init {
        if (!degreesOfFreedom.isFinite() || degreesOfFreedom <= 0.0) throw InvalidParameterException("Degrees of freedom must be finite and positive, got $degreesOfFreedom")
    }

    private val df = degreesOfFreedom

    private val logPdfPrefix = lnGamma((df + 1) / 2) - lnGamma(df / 2) - 0.5 * ln(df * PI)
    private val pdfCoeff = exp(logPdfPrefix)
    private val halfDfPlus1 = (df + 1) / 2

    private val chi2Delegate by lazy { ChiSquaredDistribution(df) }

    /**
     * Computes the probability density at [x].
     *
     * The density is symmetric about zero and approaches the standard normal density
     * as degrees of freedom increases.
     *
     * @param x the point at which to evaluate the density.
     * @return the probability density at [x]. Always non-negative.
     */
    override fun pdf(x: Double): Double {
        if (x.isInfinite()) return 0.0
        return pdfCoeff * (1.0 + x * x / df).pow(-halfDfPlus1)
    }

    /**
     * Computes the natural logarithm of the probability density at [x].
     *
     * @param x the point at which to evaluate the log-density.
     * @return the natural log of the density at [x].
     */
    override fun logPdf(x: Double): Double {
        if (x.isInfinite()) return Double.NEGATIVE_INFINITY
        return logPdfPrefix - halfDfPlus1 * ln(1.0 + x * x / df)
    }

    /**
     * Computes the cumulative distribution function at [x].
     *
     * Returns the probability that a t-distributed random variable is less than or equal
     * to [x], evaluated via the regularized incomplete beta function.
     *
     * @param x the point at which to evaluate the CDF.
     * @return the cumulative probability at [x], in the range [0, 1].
     */
    override fun cdf(x: Double): Double {
        if (x == Double.NEGATIVE_INFINITY) return 0.0
        if (x == Double.POSITIVE_INFINITY) return 1.0
        val t2 = x * x
        val ib = regularizedBeta(df / (df + t2), df / 2, 0.5)
        return if (x >= 0) 1.0 - 0.5 * ib else 0.5 * ib
    }

    /**
     * Computes the survival function (one minus the CDF) at [x].
     *
     * Uses the symmetry of the t-distribution through the regularized incomplete beta
     * function for improved numerical accuracy in each tail.
     *
     * @param x the point at which to evaluate the survival function.
     * @return the probability that a value exceeds [x], in the range [0, 1].
     */
    override fun sf(x: Double): Double {
        if (x == Double.NEGATIVE_INFINITY) return 1.0
        if (x == Double.POSITIVE_INFINITY) return 0.0
        val t2 = x * x
        val ib = regularizedBeta(df / (df + t2), df / 2, 0.5)
        return if (x >= 0) 0.5 * ib else 1.0 - 0.5 * ib
    }

    /**
     * Computes the quantile (inverse CDF) for the given probability [p].
     *
     * Uses Newton's method seeded with the corresponding standard normal quantile. The
     * distribution is symmetric, so `quantile(0.5)` returns exactly 0.
     *
     * @param p the cumulative probability, must be in [0, 1].
     * @return the value x such that `cdf(x) = p`.
     */
    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return Double.NEGATIVE_INFINITY
        if (p == 1.0) return Double.POSITIVE_INFINITY
        if (p == 0.5) return 0.0
        return findQuantile(p, ::cdf, ::pdf, NormalDistribution.STANDARD.quantile(p))
    }

    /** The differential entropy of this distribution. */
    override val entropy: Double =
        (df + 1.0) / 2.0 * (digamma((df + 1.0) / 2.0) - digamma(df / 2.0)) +
            0.5 * ln(df) + lnBeta(df / 2.0, 0.5)

    /** The mean of this distribution, which is 0 when degrees of freedom exceeds 1, or NaN otherwise. */
    override val mean: Double get() = if (df > 1) 0.0 else Double.NaN

    /** The variance of this distribution. Finite when df > 2, infinite when 1 < df <= 2, or NaN when df <= 1. */
    override val variance: Double
        get() = when {
            df > 2 -> df / (df - 2)
            df > 1 -> Double.POSITIVE_INFINITY
            else -> Double.NaN
        }

    /** The skewness of this distribution, which is 0 when degrees of freedom exceeds 3, or NaN otherwise. */
    override val skewness: Double get() = if (df > 3) 0.0 else Double.NaN

    /** The excess kurtosis of this distribution. Finite when df > 4, infinite when 2 < df <= 4, or NaN when df <= 2. */
    override val kurtosis: Double
        get() = when {
            df > 4 -> 6.0 / (df - 4)
            df > 2 -> Double.POSITIVE_INFINITY
            else -> Double.NaN
        }

    /**
     * Draws a single random value from this Student's t-distribution.
     *
     * Generates an independent standard normal sample and a chi-squared sample, then
     * returns their ratio scaled by the degrees of freedom. This is the definition-based
     * sampling method.
     *
     * @param random the source of randomness.
     * @return a random value drawn from this distribution.
     */
    override fun sample(random: Random): Double {
        // Ratio of normal to chi-squared
        val normal = NormalDistribution.STANDARD.sample(random)
        val chi2 = chi2Delegate.sample(random)
        return normal / sqrt(chi2 / df)
    }
}
