package org.oremif.kstats.distributions

import org.oremif.kstats.core.digamma
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.core.findQuantile
import org.oremif.kstats.core.lnBeta
import org.oremif.kstats.core.regularizedBeta
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Represents the F-distribution (also known as the Fisher-Snedecor distribution).
 *
 * The F-distribution arises as the ratio of two independent chi-squared random variables,
 * each divided by its degrees of freedom. It is the workhorse distribution behind F-tests,
 * which appear in analysis of variance (ANOVA), regression significance testing, and
 * comparing the variances of two populations.
 *
 * The distribution is right-skewed and supported on the interval from zero to positive
 * infinity. As both degrees-of-freedom parameters grow, it converges toward a normal
 * distribution. The mean exists only when the denominator degrees of freedom exceeds 2,
 * and the variance exists only when it exceeds 4.
 *
 * Internally, the CDF and survival function use the regularized incomplete beta function,
 * and the quantile function uses Newton's method. Random sampling generates two independent
 * chi-squared variates and computes their ratio.
 *
 * ### Example:
 * ```kotlin
 * val f = FDistribution(dfNumerator = 5.0, dfDenominator = 10.0)
 * f.pdf(1.0)           // 0.6092... (density at x = 1)
 * f.cdf(3.33)          // ~0.95 (probability that F <= 3.33)
 * f.quantile(0.95)     // ~3.33 (critical value for a 5% upper-tail test)
 * f.mean               // 1.25 (10 / (10 - 2))
 * f.sample(Random(42)) // a single random draw from the distribution
 * ```
 *
 * @param dfNumerator the degrees of freedom for the numerator (first) chi-squared variable. Must be positive.
 * @param dfDenominator the degrees of freedom for the denominator (second) chi-squared variable. Must be positive.
 */
public class FDistribution(
    public val dfNumerator: Double,
    public val dfDenominator: Double
) : ContinuousDistribution {

    init {
        if (dfNumerator <= 0.0) throw InvalidParameterException("dfNumerator must be positive, got $dfNumerator")
        if (dfDenominator <= 0.0) throw InvalidParameterException("dfDenominator must be positive, got $dfDenominator")
    }

    private val d1 = dfNumerator
    private val d2 = dfDenominator

    /**
     * Returns the probability density at [x].
     *
     * The density is zero for negative values. At zero it depends on the numerator degrees of
     * freedom: it is infinite when the numerator df is less than 2, exactly 1.0 when equal to 2,
     * and zero when greater than 2. For positive values the density is computed via exponentiation
     * of the log-density.
     *
     * @param x the point at which to evaluate the density.
     * @return the probability density at [x]. Always non-negative.
     */
    override fun pdf(x: Double): Double {
        if (x < 0.0) return 0.0
        if (x == 0.0) return if (d1 == 2.0) 1.0 else if (d1 < 2.0) Double.POSITIVE_INFINITY else 0.0
        return exp(logPdf(x))
    }

    /**
     * Returns the natural logarithm of the probability density at [x].
     *
     * Computed directly rather than as `ln(pdf(x))` to avoid precision loss with very small
     * density values. Returns negative infinity for non-positive inputs.
     *
     * @param x the point at which to evaluate the log-density.
     * @return the natural log of the probability density, or [Double.NEGATIVE_INFINITY] when [x] is non-positive.
     */
    override fun logPdf(x: Double): Double {
        if (x <= 0.0) return Double.NEGATIVE_INFINITY
        return 0.5 * (d1 * ln(d1) + d2 * ln(d2) + (d1 - 2.0) * ln(x)) -
            lnBeta(d1 / 2, d2 / 2) -
            (d1 + d2) / 2.0 * ln(d1 * x + d2)
    }

    /**
     * Returns the cumulative distribution function value at [x].
     *
     * Gives the probability that a random variable drawn from this F-distribution is less than
     * or equal to [x]. Computed using the regularized incomplete beta function for numerical
     * stability.
     *
     * @param x the point at which to evaluate the cumulative probability.
     * @return the probability that a value is less than or equal to [x], in the range `[0, 1]`.
     */
    override fun cdf(x: Double): Double {
        if (x <= 0.0) return 0.0
        return regularizedBeta(d1 * x / (d1 * x + d2), d1 / 2, d2 / 2)
    }

    /**
     * Returns the survival function value at [x], equal to `1 - cdf(x)`.
     *
     * Computed directly using the regularized incomplete beta function with swapped parameters
     * to avoid catastrophic cancellation when the CDF is close to 1.
     *
     * @param x the point at which to evaluate the survival probability.
     * @return the probability that a value exceeds [x], in the range `[0, 1]`.
     */
    override fun sf(x: Double): Double {
        if (x <= 0.0) return 1.0
        return regularizedBeta(d2 / (d1 * x + d2), d2 / 2, d1 / 2)
    }

    /**
     * Returns the quantile (inverse CDF) for the given probability [p].
     *
     * Finds the value x such that `cdf(x) = p` using Newton's method with an initial guess
     * based on the distribution mean.
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the value x at which `cdf(x) = p`. Returns 0.0 for `p = 0` and positive infinity for `p = 1`.
     */
    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return 0.0
        if (p == 1.0) return Double.POSITIVE_INFINITY
        return findQuantile(p, ::cdf, ::pdf, d2 / (d2 - 2.0).coerceAtLeast(0.1), lowerBound = 1e-15)
    }

    /** The differential entropy of this distribution. */
    override val entropy: Double get() =
        ln(d2 / d1) + lnBeta(d1 / 2.0, d2 / 2.0) +
            (1.0 - d1 / 2.0) * digamma(d1 / 2.0) - (1.0 + d2 / 2.0) * digamma(d2 / 2.0) +
            (d1 + d2) / 2.0 * digamma((d1 + d2) / 2.0)

    /** Returns the mean, defined only when the denominator degrees of freedom exceeds 2. Returns [Double.NaN] otherwise. */
    override val mean: Double get() = if (d2 > 2) d2 / (d2 - 2) else Double.NaN

    /** Returns the variance, defined only when the denominator degrees of freedom exceeds 4. Returns [Double.NaN] otherwise. */
    override val variance: Double get() = if (d2 > 4) {
        2.0 * d2 * d2 * (d1 + d2 - 2) / (d1 * (d2 - 2) * (d2 - 2) * (d2 - 4))
    } else Double.NaN

    /** Returns the skewness, defined only when the denominator degrees of freedom exceeds 6. Returns [Double.NaN] otherwise. */
    override val skewness: Double get() = if (d2 > 6) {
        (2 * d1 + d2 - 2) * sqrt(8.0 * (d2 - 4)) / ((d2 - 6) * sqrt(d1 * (d1 + d2 - 2)))
    } else Double.NaN

    /** Returns the excess kurtosis, defined only when the denominator degrees of freedom exceeds 8. Returns [Double.NaN] otherwise. */
    override val kurtosis: Double get() = if (d2 > 8) {
        12.0 * (d1 * (5 * d2 - 22) * (d1 + d2 - 2) + (d2 - 4) * (d2 - 2) * (d2 - 2)) /
            (d1 * (d2 - 6) * (d2 - 8) * (d1 + d2 - 2))
    } else Double.NaN

    /**
     * Draws a single random value from this F-distribution.
     *
     * Generates two independent chi-squared variates (one for each degrees-of-freedom parameter),
     * divides each by its degrees of freedom, and returns their ratio.
     *
     * @param random the source of randomness.
     * @return a random value drawn from this distribution.
     */
    override fun sample(random: Random): Double {
        val chi1 = ChiSquaredDistribution(d1).sample(random) / d1
        val chi2 = ChiSquaredDistribution(d2).sample(random) / d2
        return chi1 / chi2
    }
}
