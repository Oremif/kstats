package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.ceil
import kotlin.math.ln
import kotlin.random.Random

public data class UniformDiscreteDistribution(
    val min: Int,
    val max: Int
) : DiscreteDistribution {

    init {
        if (min > max) throw InvalidParameterException("min must be <= max, got min=$min, max=$max")
    }

    private val n = max - min + 1

    override fun pmf(k: Int): Double = if (k in min..max) 1.0 / n else 0.0

    override fun logPmf(k: Int): Double = if (k in min..max) -ln(n.toDouble()) else Double.NEGATIVE_INFINITY

    override fun cdf(k: Int): Double = when {
        k < min -> 0.0
        k >= max -> 1.0
        else -> (k - min + 1).toDouble() / n
    }

    override fun quantileInt(p: Double): Int {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return min
        return (min + ceil(p * n).toInt() - 1).coerceIn(min, max)
    }

    override val mean: Double get() = (min + max) / 2.0
    override val variance: Double get() = (n.toDouble() * n - 1.0) / 12.0

    override val skewness: Double get() = 0.0
    override val kurtosis: Double get() {
        if (n == 1) return Double.NaN
        val nd = n.toDouble()
        return -6.0 * (nd * nd + 1.0) / (5.0 * (nd * nd - 1.0))
    }
    override val entropy: Double get() = ln(n.toDouble())

    override fun sample(random: Random): Int = random.nextInt(min, max + 1)
}
