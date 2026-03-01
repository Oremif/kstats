package org.oremif.kstats.distributions

import kotlin.math.ln
import kotlin.random.Random

public data class UniformDiscreteDistribution(
    val min: Int,
    val max: Int
) : DiscreteDistribution {

    init {
        require(min <= max) { "min must be <= max, got min=$min, max=$max" }
    }

    private val n = max - min + 1

    override fun pmf(k: Int): Double = if (k in min..max) 1.0 / n else 0.0

    override fun logPmf(k: Int): Double = if (k in min..max) -ln(n.toDouble()) else Double.NEGATIVE_INFINITY

    override fun cdf(k: Int): Double = when {
        k < min -> 0.0
        k >= max -> 1.0
        else -> (k - min + 1).toDouble() / n
    }

    override fun quantile(p: Double): Int {
        require(p in 0.0..1.0) { "p must be in [0, 1], got $p" }
        return (min + (p * n).toInt()).coerceAtMost(max)
    }

    override val mean: Double get() = (min + max) / 2.0
    override val variance: Double get() = (n.toDouble() * n - 1.0) / 12.0

    override fun sample(random: Random): Int = random.nextInt(min, max + 1)
}
