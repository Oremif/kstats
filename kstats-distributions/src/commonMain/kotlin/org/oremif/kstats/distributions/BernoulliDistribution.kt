package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

public data class BernoulliDistribution(
    val probability: Double
) : DiscreteDistribution {

    init {
        if (probability !in 0.0..1.0) throw InvalidParameterException("probability must be in [0, 1], got $probability")
    }

    private val p = probability
    private val q = 1.0 - p

    override fun pmf(k: Int): Double = when (k) {
        0 -> q
        1 -> p
        else -> 0.0
    }

    override fun logPmf(k: Int): Double = when (k) {
        0 -> ln(q)
        1 -> ln(p)
        else -> Double.NEGATIVE_INFINITY
    }

    override fun cdf(k: Int): Double = when {
        k < 0 -> 0.0
        k < 1 -> q
        else -> 1.0
    }

    override fun quantile(p: Double): Int {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        return if (p <= q) 0 else 1
    }

    override val mean: Double get() = p
    override val variance: Double get() = p * q

    override fun sample(random: Random): Int = if (random.nextDouble() < p) 1 else 0
}
