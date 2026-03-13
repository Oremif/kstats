package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.*
import kotlin.random.Random

public data class LogNormalDistribution(
    val mu: Double = 0.0,
    val sigma: Double = 1.0
) : ContinuousDistribution {

    init {
        if (sigma <= 0.0) throw InvalidParameterException("sigma must be positive, got $sigma")
    }

    private val normal = NormalDistribution(mu, sigma)

    override fun pdf(x: Double): Double {
        if (x <= 0.0) return 0.0
        val lnX = ln(x)
        val z = (lnX - mu) / sigma
        return exp(-0.5 * z * z) / (x * sigma * sqrt(2.0 * PI))
    }

    override fun logPdf(x: Double): Double {
        if (x <= 0.0) return Double.NEGATIVE_INFINITY
        val lnX = ln(x)
        val z = (lnX - mu) / sigma
        return -0.5 * z * z - lnX - ln(sigma) - 0.5 * ln(2.0 * PI)
    }

    override fun cdf(x: Double): Double {
        if (x <= 0.0) return 0.0
        return normal.cdf(ln(x))
    }

    override fun sf(x: Double): Double {
        if (x <= 0.0) return 1.0
        return normal.sf(ln(x))
    }

    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return 0.0
        if (p == 1.0) return Double.POSITIVE_INFINITY
        return exp(normal.quantile(p))
    }

    override val mean: Double get() = exp(mu + sigma * sigma / 2.0)
    override val variance: Double get() {
        val s2 = sigma * sigma
        return (exp(s2) - 1.0) * exp(2.0 * mu + s2)
    }
    override val skewness: Double get() {
        val s2 = sigma * sigma
        return (exp(s2) + 2.0) * sqrt(exp(s2) - 1.0)
    }
    override val kurtosis: Double get() { // excess
        val s2 = sigma * sigma
        return exp(4.0 * s2) + 2.0 * exp(3.0 * s2) + 3.0 * exp(2.0 * s2) - 6.0
    }

    override val entropy: Double get() = mu + 0.5 + ln(sigma) + 0.5 * ln(2.0 * PI)

    override fun sample(random: Random): Double = exp(normal.sample(random))
}
