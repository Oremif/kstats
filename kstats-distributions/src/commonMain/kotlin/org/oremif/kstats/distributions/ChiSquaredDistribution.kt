package org.oremif.kstats.distributions

import org.oremif.kstats.core.*
import kotlin.math.*
import kotlin.random.Random

public data class ChiSquaredDistribution(
    val degreesOfFreedom: Double
) : ContinuousDistribution {

    init {
        require(degreesOfFreedom > 0.0) { "Degrees of freedom must be positive, got $degreesOfFreedom" }
    }

    private val df = degreesOfFreedom
    private val halfDf = df / 2.0

    override fun pdf(x: Double): Double {
        if (x < 0.0) return 0.0
        if (x == 0.0) return if (df == 2.0) 0.5 else if (df < 2.0) Double.POSITIVE_INFINITY else 0.0
        return exp(logPdf(x))
    }

    override fun logPdf(x: Double): Double {
        if (x <= 0.0) return Double.NEGATIVE_INFINITY
        return (halfDf - 1.0) * ln(x) - x / 2.0 - halfDf * ln(2.0) - lnGamma(halfDf)
    }

    override fun cdf(x: Double): Double {
        if (x <= 0.0) return 0.0
        return regularizedGammaP(halfDf, x / 2.0)
    }

    override fun sf(x: Double): Double {
        if (x <= 0.0) return 1.0
        return regularizedGammaQ(halfDf, x / 2.0)
    }

    override fun quantile(p: Double): Double {
        require(p in 0.0..1.0) { "p must be in [0, 1], got $p" }
        if (p == 0.0) return 0.0
        if (p == 1.0) return Double.POSITIVE_INFINITY

        // Wilson-Hilferty initial approximation
        var x = if (df > 2) {
            val z = NormalDistribution.STANDARD.quantile(p)
            val w = 2.0 / (9.0 * df)
            df * (1.0 - w + z * sqrt(w)).pow(3.0).coerceAtLeast(0.001)
        } else {
            df * 0.5
        }

        // Newton's method
        for (i in 0..49) {
            val cdfVal = cdf(x)
            val pdfVal = pdf(x)
            if (pdfVal == 0.0) break
            val delta = (cdfVal - p) / pdfVal
            x = (x - delta).coerceAtLeast(1e-15)
            if (abs(delta) < 1e-12 * x) break
        }

        return x
    }

    override val mean: Double get() = df
    override val variance: Double get() = 2.0 * df
    override val skewness: Double get() = sqrt(8.0 / df)
    override val kurtosis: Double get() = 12.0 / df // excess

    override fun sample(random: Random): Double {
        // Sum of df standard normal squared (for integer df) or Gamma(df/2, 2) general
        return GammaDistribution(halfDf, 0.5).sample(random)
    }
}
