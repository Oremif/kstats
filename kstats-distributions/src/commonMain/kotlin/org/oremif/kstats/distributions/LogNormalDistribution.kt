package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Represents the log-normal distribution.
 *
 * A random variable follows a log-normal distribution when its natural logarithm is normally
 * distributed. Equivalently, if X is normal then exp(X) is log-normal. This makes the
 * log-normal distribution a natural model for multiplicative processes -- situations where
 * many small random factors combine by multiplication rather than addition. Common examples
 * include stock prices, personal incomes, city populations, and body weights.
 *
 * The distribution is supported on the interval from zero (exclusive) to positive infinity.
 * It is always right-skewed: small values are common while very large values occur rarely but
 * are not negligible. The parameter [mu] is the mean of the underlying normal distribution
 * (not the mean of the log-normal itself), and [sigma] is the standard deviation of the
 * underlying normal distribution (not the standard deviation of the log-normal).
 *
 * Internally, the CDF, survival function, and quantile function delegate to a
 * [NormalDistribution] applied to the log-transformed input. Random sampling generates a
 * normal variate and exponentiates it.
 *
 * ### Example:
 * ```kotlin
 * val ln = LogNormalDistribution(mu = 0.0, sigma = 1.0)
 * ln.pdf(1.0)           // 0.3989... (density at x = 1)
 * ln.cdf(1.0)           // 0.5 (median of the standard log-normal)
 * ln.quantile(0.5)      // 1.0 (the median equals exp(mu) when sigma > 0)
 * ln.mean               // 1.6487... (exp(0 + 1/2) for mu=0, sigma=1)
 * ln.sample(Random(42)) // a single random draw from the distribution
 * ```
 *
 * @property mu the mean of the underlying normal distribution (log-scale location). Defaults to 0.0.
 * @property sigma the standard deviation of the underlying normal distribution (log-scale spread). Must be positive. Defaults to 1.0.
 */
public class LogNormalDistribution(
    public val mu: Double = 0.0,
    public val sigma: Double = 1.0
) : ContinuousDistribution {

    init {
        if (!mu.isFinite()) throw InvalidParameterException("mu must be finite, got $mu")
        if (!sigma.isFinite() || sigma <= 0.0) throw InvalidParameterException("sigma must be finite and positive, got $sigma")
    }

    private val normal = NormalDistribution(mu, sigma)

    /**
     * Returns the probability density at [x].
     *
     * The density is zero for non-positive values. For positive values, the density is
     * computed from the normal density applied to the natural logarithm of [x], divided
     * by [x] to account for the change of variable.
     *
     * @param x the point at which to evaluate the density.
     * @return the probability density at [x]. Always non-negative; zero for non-positive values.
     */
    override fun pdf(x: Double): Double {
        if (x <= 0.0) return 0.0
        val lnX = ln(x)
        val z = (lnX - mu) / sigma
        return exp(-0.5 * z * z) / (x * sigma * sqrt(2.0 * PI))
    }

    /**
     * Returns the natural logarithm of the probability density at [x].
     *
     * Computed directly rather than as `ln(pdf(x))` to avoid precision loss with very small
     * density values.
     *
     * @param x the point at which to evaluate the log-density.
     * @return the natural log of the probability density, or [Double.NEGATIVE_INFINITY] when [x] is non-positive.
     */
    override fun logPdf(x: Double): Double {
        if (x <= 0.0) return Double.NEGATIVE_INFINITY
        val lnX = ln(x)
        val z = (lnX - mu) / sigma
        return -0.5 * z * z - lnX - ln(sigma) - 0.5 * ln(2.0 * PI)
    }

    /**
     * Returns the cumulative distribution function value at [x].
     *
     * Gives the probability that a random variable drawn from this log-normal distribution
     * is less than or equal to [x]. Delegates to the CDF of the underlying normal distribution
     * evaluated at `ln(x)`.
     *
     * @param x the point at which to evaluate the cumulative probability.
     * @return the probability that a value is less than or equal to [x], in the range `[0, 1]`.
     */
    override fun cdf(x: Double): Double {
        if (x <= 0.0) return 0.0
        return normal.cdf(ln(x))
    }

    /**
     * Returns the survival function value at [x], equal to `1 - cdf(x)`.
     *
     * Delegates to the survival function of the underlying normal distribution evaluated
     * at `ln(x)` to maintain precision when the CDF is close to 1.
     *
     * @param x the point at which to evaluate the survival probability.
     * @return the probability that a value exceeds [x], in the range `[0, 1]`.
     */
    override fun sf(x: Double): Double {
        if (x <= 0.0) return 1.0
        return normal.sf(ln(x))
    }

    /**
     * Returns the quantile (inverse CDF) for the given probability [p].
     *
     * Delegates to the quantile function of the underlying normal distribution and
     * exponentiates the result.
     *
     * @param p the cumulative probability, must be in `[0, 1]`.
     * @return the value x at which `cdf(x) = p`. Returns 0.0 for `p = 0` and positive infinity for `p = 1`.
     */
    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return 0.0
        if (p == 1.0) return Double.POSITIVE_INFINITY
        return exp(normal.quantile(p))
    }

    /** Returns the mean of the log-normal distribution, equal to `exp(mu + sigma^2 / 2)`. */
    override val mean: Double get() = exp(mu + sigma * sigma / 2.0)

    /** Returns the variance of the log-normal distribution. */
    override val variance: Double
        get() {
            val s2 = sigma * sigma
            return (exp(s2) - 1.0) * exp(2.0 * mu + s2)
        }

    /** Returns the skewness, which is always positive (right-skewed). */
    override val skewness: Double
        get() {
            val s2 = sigma * sigma
            return (exp(s2) + 2.0) * sqrt(exp(s2) - 1.0)
        }

    /** Returns the excess kurtosis, which is always positive (heavier tails than a normal distribution). */
    override val kurtosis: Double
        get() { // excess
            val s2 = sigma * sigma
            return exp(4.0 * s2) + 2.0 * exp(3.0 * s2) + 3.0 * exp(2.0 * s2) - 6.0
        }

    /** Returns the Shannon entropy of this distribution in nats. */
    override val entropy: Double = mu + 0.5 + ln(sigma) + 0.5 * ln(2.0 * PI)

    /**
     * Draws a single random value from this log-normal distribution.
     *
     * Generates a normal variate from the underlying normal distribution and exponentiates it.
     *
     * @param random the source of randomness.
     * @return a random value drawn from this distribution. Always positive.
     */
    override fun sample(random: Random): Double = exp(normal.sample(random))
}
