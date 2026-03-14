package org.oremif.kstats.distributions

import org.oremif.kstats.core.EULER_MASCHERONI
import org.oremif.kstats.core.erf
import org.oremif.kstats.core.erfc
import org.oremif.kstats.core.erfInv
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

public data class LevyDistribution(
    val mu: Double = 0.0,
    val c: Double = 1.0,
) : ContinuousDistribution {

    init {
        if (c <= 0.0) throw InvalidParameterException("c must be positive, got $c")
    }

    override fun pdf(x: Double): Double {
        if (x <= mu) return 0.0
        val diff = x - mu
        return sqrt(c / (2.0 * PI)) * exp(-c / (2.0 * diff)) / (diff * sqrt(diff))
    }

    override fun logPdf(x: Double): Double {
        if (x <= mu) return Double.NEGATIVE_INFINITY
        val diff = x - mu
        return 0.5 * (ln(c) - ln(2.0 * PI)) - c / (2.0 * diff) - 1.5 * ln(diff)
    }

    override fun cdf(x: Double): Double {
        if (x <= mu) return 0.0
        return erfc(sqrt(c / (2.0 * (x - mu))))
    }

    override fun sf(x: Double): Double {
        if (x <= mu) return 1.0
        return erf(sqrt(c / (2.0 * (x - mu))))
    }

    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return mu
        if (p == 1.0) return Double.POSITIVE_INFINITY
        val z = erfInv(1.0 - p)
        return mu + c / (2.0 * z * z)
    }

    override val mean: Double get() = Double.POSITIVE_INFINITY

    override val variance: Double get() = Double.POSITIVE_INFINITY

    override val skewness: Double get() = Double.NaN

    override val kurtosis: Double get() = Double.NaN

    override val entropy: Double = 0.5 * (1.0 + 3.0 * EULER_MASCHERONI + ln(16.0 * PI * c * c))

    override fun sample(random: Random): Double = quantile(random.nextDouble())

    public companion object {
        public val STANDARD: LevyDistribution = LevyDistribution(0.0, 1.0)
    }
}
