package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.core.lnBeta
import org.oremif.kstats.core.lnCombination
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

public data class BetaBinomialDistribution(
    val trials: Int,
    val alpha: Double,
    val beta: Double,
) : DiscreteDistribution {

    init {
        if (trials < 0) throw InvalidParameterException("trials must be non-negative, got $trials")
        if (alpha <= 0.0) throw InvalidParameterException("alpha must be positive, got $alpha")
        if (beta <= 0.0) throw InvalidParameterException("beta must be positive, got $beta")
    }

    private val n = trials
    private val a = alpha
    private val b = beta
    private val lnBetaAB = lnBeta(a, b)

    override fun pmf(k: Int): Double {
        if (k < 0 || k > n) return 0.0
        if (n == 0) return if (k == 0) 1.0 else 0.0
        return exp(logPmf(k))
    }

    override fun logPmf(k: Int): Double {
        if (k < 0 || k > n) return Double.NEGATIVE_INFINITY
        if (n == 0) return if (k == 0) 0.0 else Double.NEGATIVE_INFINITY
        return lnCombination(n, k) + lnBeta(k + a, n - k + b) - lnBetaAB
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
        if (k < 0) return 0.0
        if (k >= n) return 1.0
        return exp(logSumPmf(0..k)).coerceIn(0.0, 1.0)
    }

    override fun sf(k: Int): Double {
        if (k < 0) return 1.0
        if (k >= n) return 0.0
        return exp(logSumPmf((k + 1)..n)).coerceIn(0.0, 1.0)
    }

    override fun quantileInt(p: Double): Int {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (n == 0) return 0
        var cumulative = 0.0
        for (k in 0..n) {
            cumulative += pmf(k)
            if (cumulative >= p) return k
        }
        return n
    }

    override val mean: Double get() = n * a / (a + b)

    override val variance: Double get() {
        if (n == 0) return 0.0
        val ab = a + b
        return n * a * b * (ab + n) / (ab * ab * (ab + 1.0))
    }

    override val skewness: Double get() {
        if (n == 0) return Double.NaN
        val ab = a + b
        return (ab + 2.0 * n) * (b - a) / (ab + 2.0) * sqrt((1.0 + ab) / (n * a * b * (ab + n)))
    }

    override val kurtosis: Double get() {
        if (n == 0) return Double.NaN
        val nd = n.toDouble()
        val ab = a + b
        // Falling factorial moments: e_k = [n]_k * prod_{i=0}^{k-1} (a+i)/(ab+i)
        val e1 = nd * a / ab
        val e2 = nd * (nd - 1) * a * (a + 1) / (ab * (ab + 1))
        val e3 = nd * (nd - 1) * (nd - 2) * a * (a + 1) * (a + 2) / (ab * (ab + 1) * (ab + 2))
        val e4 = nd * (nd - 1) * (nd - 2) * (nd - 3) * a * (a + 1) * (a + 2) * (a + 3) /
            (ab * (ab + 1) * (ab + 2) * (ab + 3))
        // Raw moments from factorial moments
        val ex2 = e2 + e1
        val ex3 = e3 + 3 * e2 + e1
        val ex4 = e4 + 6 * e3 + 7 * e2 + e1
        val mu = e1
        val mu2 = ex2 - mu * mu
        if (mu2 == 0.0) return Double.NaN
        val mu4 = ex4 - 4 * mu * ex3 + 6 * mu * mu * ex2 - 3 * mu * mu * mu * mu
        return mu4 / (mu2 * mu2) - 3.0
    }

    override val entropy: Double get() {
        if (n == 0) return 0.0
        var h = 0.0
        for (k in 0..n) {
            val pk = pmf(k)
            if (pk > 0.0) h -= pk * ln(pk)
        }
        return h
    }

    override fun sample(random: Random): Int {
        val p = BetaDistribution(a, b).sample(random).coerceIn(0.0, 1.0)
        return BinomialDistribution(n, p).sample(random)
    }
}
