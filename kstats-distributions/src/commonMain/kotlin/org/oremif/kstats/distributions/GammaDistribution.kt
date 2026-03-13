package org.oremif.kstats.distributions

import org.oremif.kstats.core.*
import org.oremif.kstats.core.exceptions.ConvergenceException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.*
import kotlin.random.Random

public data class GammaDistribution(
    val shape: Double,
    val rate: Double = 1.0
) : ContinuousDistribution {

    init {
        if (shape <= 0.0) throw InvalidParameterException("shape must be positive, got $shape")
        if (rate <= 0.0) throw InvalidParameterException("rate must be positive, got $rate")
    }

    private val scale = 1.0 / rate

    override fun pdf(x: Double): Double {
        if (x < 0.0) return 0.0
        if (x == 0.0) return when {
            shape == 1.0 -> rate
            shape < 1.0 -> Double.POSITIVE_INFINITY
            else -> 0.0
        }
        return exp(logPdf(x))
    }

    override fun logPdf(x: Double): Double {
        if (x <= 0.0) return Double.NEGATIVE_INFINITY
        return (shape - 1.0) * ln(x) - x * rate + shape * ln(rate) - lnGamma(shape)
    }

    override fun cdf(x: Double): Double {
        if (x <= 0.0) return 0.0
        return regularizedGammaP(shape, x * rate)
    }

    override fun sf(x: Double): Double {
        if (x <= 0.0) return 1.0
        return regularizedGammaQ(shape, x * rate)
    }

    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return 0.0
        if (p == 1.0) return Double.POSITIVE_INFINITY

        // Initial guess using Wilson-Hilferty or chi-squared approximation
        var x = if (shape >= 1.0) {
            val z = NormalDistribution.STANDARD.quantile(p)
            val w = 2.0 / (9.0 * shape)
            (shape * (1.0 - w + z * sqrt(w)).pow(3.0)).coerceAtLeast(0.001) / rate
        } else {
            // For small shape, use a simple guess
            (shape * p.pow(1.0 / shape)) / rate
        }

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
            "Gamma quantile did not converge for p=$p after 50 iterations",
            iterations = 50,
            lastEstimate = x
        )

        return x
    }

    override val mean: Double get() = shape / rate
    override val variance: Double get() = shape / (rate * rate)
    override val skewness: Double get() = 2.0 / sqrt(shape)
    override val kurtosis: Double get() = 6.0 / shape // excess

    override fun sample(random: Random): Double {
        // Marsaglia and Tsang's method for shape >= 1
        // For shape < 1: use Gamma(shape+1)*U^(1/shape)
        if (shape < 1.0) {
            val g1 = GammaDistribution(shape + 1.0, 1.0).sample(random)
            return g1 * random.nextDouble().pow(1.0 / shape) / rate
        }

        val d = shape - 1.0 / 3.0
        val c = 1.0 / sqrt(9.0 * d)

        while (true) {
            var x: Double
            var v: Double
            do {
                x = NormalDistribution.STANDARD.sample(random)
                v = 1.0 + c * x
            } while (v <= 0.0)

            v = v * v * v
            val u = random.nextDouble()

            if (u < 1.0 - 0.0331 * x * x * x * x) {
                return d * v / rate
            }

            if (ln(u) < 0.5 * x * x + d * (1.0 - v + ln(v))) {
                return d * v / rate
            }
        }
    }
}
