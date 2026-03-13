package org.oremif.kstats.distributions

import org.oremif.kstats.core.erf
import org.oremif.kstats.core.erfc
import org.oremif.kstats.core.erfInv
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.*
import kotlin.random.Random

public data class NormalDistribution(
    val mu: Double = 0.0,
    val sigma: Double = 1.0
) : ContinuousDistribution {

    init {
        if (sigma <= 0.0) throw InvalidParameterException("sigma must be positive, got $sigma")
    }

    override fun pdf(x: Double): Double {
        val z = (x - mu) / sigma
        return exp(-0.5 * z * z) / (sigma * sqrt(2.0 * PI))
    }

    override fun logPdf(x: Double): Double {
        val z = (x - mu) / sigma
        return -0.5 * z * z - ln(sigma) - 0.5 * ln(2.0 * PI)
    }

    override fun cdf(x: Double): Double {
        return 0.5 * (1.0 + erf((x - mu) / (sigma * sqrt(2.0))))
    }

    override fun sf(x: Double): Double {
        return 0.5 * erfc((x - mu) / (sigma * sqrt(2.0)))
    }

    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return Double.NEGATIVE_INFINITY
        if (p == 1.0) return Double.POSITIVE_INFINITY
        return mu + sigma * sqrt(2.0) * erfInv(2.0 * p - 1.0)
    }

    override val mean: Double get() = mu
    override val variance: Double get() = sigma * sigma
    override val standardDeviation: Double get() = sigma
    override val skewness: Double get() = 0.0
    override val kurtosis: Double get() = 0.0 // excess kurtosis

    override fun sample(random: Random): Double {
        // Box-Muller transform
        val u1 = random.nextDouble()
        val u2 = random.nextDouble()
        return mu + sigma * sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)
    }

    public companion object {
        public val STANDARD: NormalDistribution = NormalDistribution(0.0, 1.0)
    }
}
