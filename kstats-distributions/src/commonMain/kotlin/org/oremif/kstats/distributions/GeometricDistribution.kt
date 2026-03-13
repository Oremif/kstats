package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.*
import kotlin.random.Random

public data class GeometricDistribution(
    val probability: Double
) : DiscreteDistribution {

    init {
        if (probability <= 0.0 || probability > 1.0) throw InvalidParameterException("probability must be in (0, 1], got $probability")
    }

    private val p = probability
    private val q = 1.0 - p

    // k = number of failures before first success (0-indexed)
    override fun pmf(k: Int): Double {
        if (k < 0) return 0.0
        return p * q.pow(k)
    }

    override fun logPmf(k: Int): Double {
        if (k < 0) return Double.NEGATIVE_INFINITY
        return ln(p) + k * ln(q)
    }

    override fun cdf(k: Int): Double {
        if (k < 0) return 0.0
        return 1.0 - q.pow(k + 1)
    }

    override fun quantileInt(p: Double): Int {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return 0
        if (p == 1.0) return Int.MAX_VALUE
        return ceil(ln(1.0 - p) / ln(q) - 1.0).toInt().coerceAtLeast(0)
    }

    override val mean: Double get() = q / p
    override val variance: Double get() = q / (p * p)

    override val skewness: Double get() = (2.0 - p) / sqrt(q)
    override val kurtosis: Double get() = 6.0 + p * p / q
    override val entropy: Double get() {
        if (p == 1.0) return 0.0
        return (-q * ln(q) - p * ln(p)) / p
    }

    override fun sf(k: Int): Double {
        if (k < 0) return 1.0
        return q.pow(k + 1)
    }

    override fun sample(random: Random): Int {
        return floor(ln(random.nextDouble()) / ln(q)).toInt()
    }
}
