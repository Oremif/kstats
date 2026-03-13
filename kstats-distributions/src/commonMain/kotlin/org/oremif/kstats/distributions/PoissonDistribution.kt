package org.oremif.kstats.distributions

import org.oremif.kstats.core.*
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.*
import kotlin.random.Random

public data class PoissonDistribution(
    val rate: Double
) : DiscreteDistribution {

    init {
        if (rate <= 0.0) throw InvalidParameterException("rate must be positive, got $rate")
    }

    private val lambda = rate

    override fun pmf(k: Int): Double {
        if (k < 0) return 0.0
        return exp(logPmf(k))
    }

    override fun logPmf(k: Int): Double {
        if (k < 0) return Double.NEGATIVE_INFINITY
        return k * ln(lambda) - lambda - lnFactorial(k)
    }

    override fun cdf(k: Int): Double {
        if (k < 0) return 0.0
        // P(X <= k) = Q(k+1, lambda) = regularizedGammaQ(k+1, lambda)
        return regularizedGammaQ((k + 1).toDouble(), lambda)
    }

    override fun quantile(p: Double): Int {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return 0
        // Search from the mean
        var k = lambda.toInt()
        while (k > 0 && cdf(k - 1) >= p) k--
        while (cdf(k) < p) k++
        return k
    }

    override val mean: Double get() = lambda
    override val variance: Double get() = lambda

    override fun sample(random: Random): Int {
        // Knuth's algorithm for small lambda
        if (lambda < 30) {
            val l = exp(-lambda)
            var k = 0
            var p = 1.0
            do {
                k++
                p *= random.nextDouble()
            } while (p > l)
            return k - 1
        }
        // For large lambda, use normal approximation
        return NormalDistribution(lambda, sqrt(lambda)).sample(random).roundToInt().coerceAtLeast(0)
    }
}
