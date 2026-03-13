package org.oremif.kstats.distributions

import org.oremif.kstats.core.*
import org.oremif.kstats.core.exceptions.ConvergenceException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.*
import kotlin.random.Random

public data class FDistribution(
    val dfNumerator: Double,
    val dfDenominator: Double
) : ContinuousDistribution {

    init {
        if (dfNumerator <= 0.0) throw InvalidParameterException("dfNumerator must be positive, got $dfNumerator")
        if (dfDenominator <= 0.0) throw InvalidParameterException("dfDenominator must be positive, got $dfDenominator")
    }

    private val d1 = dfNumerator
    private val d2 = dfDenominator

    override fun pdf(x: Double): Double {
        if (x < 0.0) return 0.0
        if (x == 0.0) return if (d1 == 2.0) 1.0 else if (d1 < 2.0) Double.POSITIVE_INFINITY else 0.0
        return exp(logPdf(x))
    }

    override fun logPdf(x: Double): Double {
        if (x <= 0.0) return Double.NEGATIVE_INFINITY
        return 0.5 * (d1 * ln(d1) + d2 * ln(d2) + (d1 - 2.0) * ln(x)) -
            lnBeta(d1 / 2, d2 / 2) -
            (d1 + d2) / 2.0 * ln(d1 * x + d2)
    }

    override fun cdf(x: Double): Double {
        if (x <= 0.0) return 0.0
        return regularizedBeta(d1 * x / (d1 * x + d2), d1 / 2, d2 / 2)
    }

    override fun sf(x: Double): Double {
        if (x <= 0.0) return 1.0
        return regularizedBeta(d2 / (d1 * x + d2), d2 / 2, d1 / 2)
    }

    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return 0.0
        if (p == 1.0) return Double.POSITIVE_INFINITY

        // Initial guess
        var x = d2 / (d2 - 2.0).coerceAtLeast(0.1) // near mean for df2 > 2

        // Newton's method
        var converged = false
        for (i in 0..49) {
            val cdfVal = cdf(x)
            val pdfVal = pdf(x)
            if (pdfVal == 0.0) { converged = true; break }
            val delta = (cdfVal - p) / pdfVal
            x = (x - delta).coerceAtLeast(1e-15)
            if (abs(delta) < 1e-12 * x) { converged = true; break }
        }
        if (!converged) throw ConvergenceException(
            "F quantile did not converge for p=$p after 50 iterations",
            iterations = 50,
            lastEstimate = x
        )

        return x
    }

    override val mean: Double get() = if (d2 > 2) d2 / (d2 - 2) else Double.NaN
    override val variance: Double get() = if (d2 > 4) {
        2.0 * d2 * d2 * (d1 + d2 - 2) / (d1 * (d2 - 2) * (d2 - 2) * (d2 - 4))
    } else Double.NaN
    override val skewness: Double get() = if (d2 > 6) {
        (2 * d1 + d2 - 2) * sqrt(8.0 * (d2 - 4)) / ((d2 - 6) * sqrt(d1 * (d1 + d2 - 2)))
    } else Double.NaN
    override val kurtosis: Double get() = if (d2 > 8) {
        12.0 * (d1 * (5 * d2 - 22) * (d1 + d2 - 2) + (d2 - 4) * (d2 - 2) * (d2 - 2)) /
            (d1 * (d2 - 6) * (d2 - 8) * (d1 + d2 - 2))
    } else Double.NaN

    override fun sample(random: Random): Double {
        val chi1 = ChiSquaredDistribution(d1).sample(random) / d1
        val chi2 = ChiSquaredDistribution(d2).sample(random) / d2
        return chi1 / chi2
    }
}
