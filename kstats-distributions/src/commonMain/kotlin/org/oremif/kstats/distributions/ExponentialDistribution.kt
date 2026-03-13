package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.*
import kotlin.random.Random

public data class ExponentialDistribution(
    val rate: Double = 1.0
) : ContinuousDistribution {

    init {
        if (rate <= 0.0) throw InvalidParameterException("rate must be positive, got $rate")
    }

    override fun pdf(x: Double): Double = if (x >= 0.0) rate * exp(-rate * x) else 0.0

    override fun logPdf(x: Double): Double = if (x >= 0.0) ln(rate) - rate * x else Double.NEGATIVE_INFINITY

    override fun cdf(x: Double): Double = if (x >= 0.0) 1.0 - exp(-rate * x) else 0.0

    override fun sf(x: Double): Double = if (x >= 0.0) exp(-rate * x) else 1.0

    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 1.0) return Double.POSITIVE_INFINITY
        return -ln(1.0 - p) / rate
    }

    override val mean: Double get() = 1.0 / rate
    override val variance: Double get() = 1.0 / (rate * rate)
    override val skewness: Double get() = 2.0
    override val kurtosis: Double get() = 6.0 // excess

    override fun sample(random: Random): Double = -ln(random.nextDouble()) / rate

    public companion object {
        public val STANDARD: ExponentialDistribution = ExponentialDistribution(1.0)
    }
}
