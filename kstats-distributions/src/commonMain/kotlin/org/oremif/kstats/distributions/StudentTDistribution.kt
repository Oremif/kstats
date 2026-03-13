package org.oremif.kstats.distributions

import org.oremif.kstats.core.*
import org.oremif.kstats.core.exceptions.ConvergenceException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.*
import kotlin.random.Random

public data class StudentTDistribution(
    val degreesOfFreedom: Double
) : ContinuousDistribution {

    init {
        if (degreesOfFreedom <= 0.0) throw InvalidParameterException("Degrees of freedom must be positive, got $degreesOfFreedom")
    }

    private val df = degreesOfFreedom

    override fun pdf(x: Double): Double {
        val coeff = exp(lnGamma((df + 1) / 2) - lnGamma(df / 2)) / sqrt(df * PI)
        return coeff * (1.0 + x * x / df).pow(-(df + 1) / 2)
    }

    override fun logPdf(x: Double): Double {
        return lnGamma((df + 1) / 2) - lnGamma(df / 2) - 0.5 * ln(df * PI) -
            (df + 1) / 2 * ln(1.0 + x * x / df)
    }

    override fun cdf(x: Double): Double {
        val t2 = x * x
        val ib = regularizedBeta(df / (df + t2), df / 2, 0.5)
        return if (x >= 0) 1.0 - 0.5 * ib else 0.5 * ib
    }

    override fun sf(x: Double): Double {
        val t2 = x * x
        val ib = regularizedBeta(df / (df + t2), df / 2, 0.5)
        return if (x >= 0) 0.5 * ib else 1.0 - 0.5 * ib
    }

    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return Double.NEGATIVE_INFINITY
        if (p == 1.0) return Double.POSITIVE_INFINITY
        if (p == 0.5) return 0.0

        // Newton's method using normal quantile as initial guess
        val normal = NormalDistribution.STANDARD
        var t = normal.quantile(p)

        var converged = false
        for (i in 0..29) {
            val cdfVal = cdf(t)
            val pdfVal = pdf(t)
            if (pdfVal == 0.0) { converged = true; break }
            val delta = (cdfVal - p) / pdfVal
            t -= delta
            if (abs(delta) < 1e-12 * abs(t).coerceAtLeast(1.0)) { converged = true; break }
        }
        if (!converged) throw ConvergenceException(
            "StudentT quantile did not converge for p=$p after 30 iterations",
            iterations = 30,
            lastEstimate = t
        )

        return t
    }

    override val mean: Double get() = if (df > 1) 0.0 else Double.NaN
    override val variance: Double get() = when {
        df > 2 -> df / (df - 2)
        df > 1 -> Double.POSITIVE_INFINITY
        else -> Double.NaN
    }
    override val skewness: Double get() = if (df > 3) 0.0 else Double.NaN
    override val kurtosis: Double get() = when {
        df > 4 -> 6.0 / (df - 4)
        df > 2 -> Double.POSITIVE_INFINITY
        else -> Double.NaN
    }

    override fun sample(random: Random): Double {
        // Ratio of normal to chi-squared
        val normal = NormalDistribution.STANDARD.sample(random)
        val chi2 = ChiSquaredDistribution(df).sample(random)
        return normal / sqrt(chi2 / df)
    }
}
