package org.oremif.kstats.distributions

import org.oremif.kstats.core.gamma
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.*
import kotlin.random.Random

public data class WeibullDistribution(
    val shape: Double,
    val scale: Double = 1.0
) : ContinuousDistribution {

    init {
        if (shape <= 0.0) throw InvalidParameterException("shape must be positive, got $shape")
        if (scale <= 0.0) throw InvalidParameterException("scale must be positive, got $scale")
    }

    private val k = shape
    private val lambda = scale

    override fun pdf(x: Double): Double {
        if (x < 0.0) return 0.0
        if (x == 0.0) return if (k == 1.0) 1.0 / lambda else if (k < 1.0) Double.POSITIVE_INFINITY else 0.0
        return (k / lambda) * (x / lambda).pow(k - 1.0) * exp(-(x / lambda).pow(k))
    }

    override fun cdf(x: Double): Double {
        if (x <= 0.0) return 0.0
        return 1.0 - exp(-(x / lambda).pow(k))
    }

    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return 0.0
        if (p == 1.0) return Double.POSITIVE_INFINITY
        return lambda * (-ln(1.0 - p)).pow(1.0 / k)
    }

    override val mean: Double get() = lambda * gamma(1.0 + 1.0 / k)
    override val variance: Double get() {
        val g1 = gamma(1.0 + 1.0 / k)
        val g2 = gamma(1.0 + 2.0 / k)
        return lambda * lambda * (g2 - g1 * g1)
    }
    override val skewness: Double get() {
        val g1 = gamma(1.0 + 1.0 / k)
        val g2 = gamma(1.0 + 2.0 / k)
        val g3 = gamma(1.0 + 3.0 / k)
        val mu = lambda * g1
        val sigma = sqrt(variance)
        return (lambda * lambda * lambda * g3 - 3.0 * mu * sigma * sigma - mu * mu * mu) / (sigma * sigma * sigma)
    }
    override val kurtosis: Double get() {
        val g1 = gamma(1.0 + 1.0 / k)
        val g2 = gamma(1.0 + 2.0 / k)
        val g3 = gamma(1.0 + 3.0 / k)
        val g4 = gamma(1.0 + 4.0 / k)
        val mu2 = g2 - g1 * g1
        return (-6.0 * g1 * g1 * g1 * g1 + 12.0 * g1 * g1 * g2 - 3.0 * g2 * g2 - 4.0 * g1 * g3 + g4) / (mu2 * mu2) - 3.0
    }

    override fun sample(random: Random): Double = quantile(random.nextDouble())
}
