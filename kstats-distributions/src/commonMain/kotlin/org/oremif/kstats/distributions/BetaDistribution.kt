package org.oremif.kstats.distributions

import org.oremif.kstats.core.*
import kotlin.math.*
import kotlin.random.Random

public data class BetaDistribution(
    val alpha: Double,
    val beta: Double
) : ContinuousDistribution {

    init {
        require(alpha > 0.0) { "alpha must be positive, got $alpha" }
        require(beta > 0.0) { "beta must be positive, got $beta" }
    }

    override fun pdf(x: Double): Double {
        if (x < 0.0 || x > 1.0) return 0.0
        if (x == 0.0) return when {
            alpha == 1.0 -> beta
            alpha < 1.0 -> Double.POSITIVE_INFINITY
            else -> 0.0
        }
        if (x == 1.0) return when {
            beta == 1.0 -> alpha
            beta < 1.0 -> Double.POSITIVE_INFINITY
            else -> 0.0
        }
        return exp(logPdf(x))
    }

    override fun logPdf(x: Double): Double {
        if (x <= 0.0 || x >= 1.0) return Double.NEGATIVE_INFINITY
        return (alpha - 1.0) * ln(x) + (beta - 1.0) * ln(1.0 - x) - lnBeta(alpha, beta)
    }

    override fun cdf(x: Double): Double {
        if (x <= 0.0) return 0.0
        if (x >= 1.0) return 1.0
        return regularizedBeta(x, alpha, beta)
    }

    override fun quantile(p: Double): Double {
        require(p in 0.0..1.0) { "p must be in [0, 1], got $p" }
        if (p == 0.0) return 0.0
        if (p == 1.0) return 1.0

        // Newton's method with initial guess
        var x = alpha / (alpha + beta) // start at the mean

        for (i in 0..49) {
            val cdfVal = cdf(x)
            val pdfVal = pdf(x)
            if (pdfVal == 0.0) break
            val delta = (cdfVal - p) / pdfVal
            x = (x - delta).coerceIn(1e-15, 1.0 - 1e-15)
            if (abs(delta) < 1e-12) break
        }

        return x
    }

    override val mean: Double get() = alpha / (alpha + beta)
    override val variance: Double get() {
        val ab = alpha + beta
        return (alpha * beta) / (ab * ab * (ab + 1.0))
    }
    override val skewness: Double get() {
        val ab = alpha + beta
        return 2.0 * (beta - alpha) * sqrt(ab + 1.0) / ((ab + 2.0) * sqrt(alpha * beta))
    }
    override val kurtosis: Double get() { // excess
        val ab = alpha + beta
        return 6.0 * (alpha * alpha * alpha - alpha * alpha * (2.0 * beta - 1.0) +
            beta * beta * (beta + 1.0) - 2.0 * alpha * beta * (beta + 2.0)) /
            (alpha * beta * (ab + 2.0) * (ab + 3.0))
    }

    override fun sample(random: Random): Double {
        val x = GammaDistribution(alpha, 1.0).sample(random)
        val y = GammaDistribution(beta, 1.0).sample(random)
        return x / (x + y)
    }

    public companion object {
        public val STANDARD: BetaDistribution = BetaDistribution(1.0, 1.0)
    }
}
