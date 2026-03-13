package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.*
import kotlin.random.Random

public data class CauchyDistribution(
    val location: Double = 0.0,
    val scale: Double = 1.0
) : ContinuousDistribution {

    init {
        if (scale <= 0.0) throw InvalidParameterException("scale must be positive, got $scale")
    }

    override fun pdf(x: Double): Double {
        val z = (x - location) / scale
        return 1.0 / (PI * scale * (1.0 + z * z))
    }

    override fun logPdf(x: Double): Double {
        val z = (x - location) / scale
        return -ln(PI) - ln(scale) - ln(1.0 + z * z)
    }

    override fun cdf(x: Double): Double = 0.5 + atan((x - location) / scale) / PI

    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return Double.NEGATIVE_INFINITY
        if (p == 1.0) return Double.POSITIVE_INFINITY
        return location + scale * tan(PI * (p - 0.5))
    }

    override val mean: Double get() = Double.NaN
    override val variance: Double get() = Double.NaN
    override val standardDeviation: Double get() = Double.NaN
    override val skewness: Double get() = Double.NaN
    override val kurtosis: Double get() = Double.NaN

    override fun sample(random: Random): Double = quantile(random.nextDouble())

    public companion object {
        public val STANDARD: CauchyDistribution = CauchyDistribution(0.0, 1.0)
    }
}
