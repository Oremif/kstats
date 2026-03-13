package org.oremif.kstats.distributions

import org.oremif.kstats.core.lnCombination
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.*
import kotlin.random.Random

public data class HypergeometricDistribution(
    val population: Int,
    val successes: Int,
    val draws: Int
) : DiscreteDistribution {

    init {
        if (population < 0) throw InvalidParameterException("population must be non-negative, got $population")
        if (successes !in 0..population) throw InvalidParameterException("successes must be in [0, population], got $successes")
        if (draws !in 0..population) throw InvalidParameterException("draws must be in [0, population], got $draws")
    }

    private val bigN = population
    private val bigK = successes
    private val n = draws

    private val kMin = maxOf(0, n + bigK - bigN)
    private val kMax = minOf(n, bigK)

    override fun pmf(k: Int): Double {
        if (k < kMin || k > kMax) return 0.0
        return exp(logPmf(k))
    }

    override fun logPmf(k: Int): Double {
        if (k < kMin || k > kMax) return Double.NEGATIVE_INFINITY
        return lnCombination(bigK, k) + lnCombination(bigN - bigK, n - k) - lnCombination(bigN, n)
    }

    override fun cdf(k: Int): Double {
        if (k < kMin) return 0.0
        if (k >= kMax) return 1.0
        var sum = 0.0
        for (i in kMin..k) {
            sum += pmf(i)
        }
        return sum.coerceAtMost(1.0)
    }

    override fun quantile(p: Double): Int {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        var cumulative = 0.0
        for (k in kMin..kMax) {
            cumulative += pmf(k)
            if (cumulative >= p) return k
        }
        return kMax
    }

    override val mean: Double get() = n.toDouble() * bigK / bigN
    override val variance: Double get() {
        val nd = n.toDouble()
        val Nd = bigN.toDouble()
        val Kd = bigK.toDouble()
        return nd * Kd * (Nd - Kd) * (Nd - nd) / (Nd * Nd * (Nd - 1.0))
    }

    override fun sample(random: Random): Int {
        // Direct simulation
        var succPool = bigK
        var failPool = bigN - bigK
        var result = 0
        for (i in 0 until n) {
            val total = succPool + failPool
            if (random.nextDouble() < succPool.toDouble() / total) {
                result++
                succPool--
            } else {
                failPool--
            }
        }
        return result
    }
}
