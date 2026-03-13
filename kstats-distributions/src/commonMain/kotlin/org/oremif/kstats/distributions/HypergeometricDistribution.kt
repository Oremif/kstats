package org.oremif.kstats.distributions

import org.oremif.kstats.core.lnCombination
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
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

    private fun logSumPmf(range: IntRange): Double {
        var maxLog = Double.NEGATIVE_INFINITY
        var sumExp = 0.0
        for (i in range) {
            val lp = logPmf(i)
            if (lp > maxLog) {
                sumExp = sumExp * exp(maxLog - lp) + 1.0
                maxLog = lp
            } else {
                sumExp += exp(lp - maxLog)
            }
        }
        return if (maxLog == Double.NEGATIVE_INFINITY) Double.NEGATIVE_INFINITY
        else maxLog + ln(sumExp)
    }

    override fun cdf(k: Int): Double {
        if (k < kMin) return 0.0
        if (k >= kMax) return 1.0
        return exp(logSumPmf(kMin..k)).coerceIn(0.0, 1.0)
    }

    override fun sf(k: Int): Double {
        if (k < kMin) return 1.0
        if (k >= kMax) return 0.0
        return exp(logSumPmf((k + 1)..kMax)).coerceIn(0.0, 1.0)
    }

    override fun quantileInt(p: Double): Int {
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
        val bigNd = bigN.toDouble()
        val bigKd = bigK.toDouble()
        return nd * bigKd * (bigNd - bigKd) * (bigNd - nd) / (bigNd * bigNd * (bigNd - 1.0))
    }

    override val skewness: Double get() {
        if (bigN < 3) return Double.NaN
        val bigNd = bigN.toDouble()
        val bigKd = bigK.toDouble()
        val nd = n.toDouble()
        return (bigNd - 2.0 * bigKd) * (bigNd - 2.0 * nd) * sqrt(bigNd - 1.0) /
            ((bigNd - 2.0) * sqrt(nd * bigKd * (bigNd - bigKd) * (bigNd - nd)))
    }

    override val kurtosis: Double get() {
        if (bigN < 4) return Double.NaN
        val bigNd = bigN.toDouble()
        val bigKd = bigK.toDouble()
        val nd = n.toDouble()
        val num = (bigNd - 1.0) * bigNd * bigNd * (bigNd * (bigNd + 1.0) - 6.0 * bigKd * (bigNd - bigKd) - 6.0 * nd * (bigNd - nd)) +
            6.0 * nd * bigKd * (bigNd - bigKd) * (bigNd - nd) * (5.0 * bigNd - 6.0)
        val den = nd * bigKd * (bigNd - bigKd) * (bigNd - nd) * (bigNd - 2.0) * (bigNd - 3.0)
        return num / den
    }

    override val entropy: Double get() {
        var h = 0.0
        for (k in kMin..kMax) {
            val pk = pmf(k)
            if (pk > 0.0) h -= pk * ln(pk)
        }
        return h
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
