package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.*
import kotlin.random.Random

public data class LaplaceDistribution(
    val mu: Double = 0.0,
    val scale: Double = 1.0,
) : ContinuousDistribution {

    init {
        if (scale <= 0.0) throw InvalidParameterException("scale must be positive, got $scale")
    }

    override fun pdf(x: Double): Double {
        return exp(-abs(x - mu) / scale) / (2.0 * scale)
    }

    override fun logPdf(x: Double): Double {
        return -abs(x - mu) / scale - ln(2.0 * scale)
    }

    override fun cdf(x: Double): Double {
        val z = (x - mu) / scale
        return if (z <= 0.0) 0.5 * exp(z) else 1.0 - 0.5 * exp(-z)
    }

    override fun sf(x: Double): Double {
        val z = (x - mu) / scale
        return if (z <= 0.0) 1.0 - 0.5 * exp(z) else 0.5 * exp(-z)
    }

    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return Double.NEGATIVE_INFINITY
        if (p == 1.0) return Double.POSITIVE_INFINITY
        return if (p <= 0.5) mu + scale * ln(2.0 * p) else mu - scale * ln(2.0 * (1.0 - p))
    }

    override val mean: Double get() = mu
    override val variance: Double get() = 2.0 * scale * scale
    override val standardDeviation: Double get() = scale * sqrt(2.0)
    override val skewness: Double get() = 0.0
    override val kurtosis: Double get() = 3.0 // excess kurtosis
    override val entropy: Double = 1.0 + ln(2.0 * scale)

    override fun sample(random: Random): Double {
        val u = random.nextDouble() - 0.5
        return mu - scale * sign(u) * ln(1.0 - 2.0 * abs(u))
    }

    public companion object {
        public val STANDARD: LaplaceDistribution = LaplaceDistribution(0.0, 1.0)
    }
}
