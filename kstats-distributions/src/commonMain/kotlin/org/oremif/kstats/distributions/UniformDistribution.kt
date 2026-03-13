package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

public data class UniformDistribution(
    val min: Double = 0.0,
    val max: Double = 1.0
) : ContinuousDistribution {

    init {
        if (min >= max) throw InvalidParameterException("min must be less than max, got min=$min, max=$max")
    }

    private val range = max - min

    override fun pdf(x: Double): Double = if (x in min..max) 1.0 / range else 0.0

    override fun logPdf(x: Double): Double = if (x in min..max) -ln(range) else Double.NEGATIVE_INFINITY

    override fun cdf(x: Double): Double = when {
        x <= min -> 0.0
        x >= max -> 1.0
        else -> (x - min) / range
    }

    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        return min + p * range
    }

    override val mean: Double get() = (min + max) / 2.0
    override val variance: Double get() = range * range / 12.0
    override val skewness: Double get() = 0.0
    override val kurtosis: Double get() = -6.0 / 5.0 // excess

    override fun sample(random: Random): Double = min + random.nextDouble() * range

    public companion object {
        public val STANDARD: UniformDistribution = UniformDistribution(0.0, 1.0)
    }
}
