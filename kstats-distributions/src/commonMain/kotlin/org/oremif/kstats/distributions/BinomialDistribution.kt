package org.oremif.kstats.distributions

import org.oremif.kstats.core.*
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.*
import kotlin.random.Random

public data class BinomialDistribution(
    val trials: Int,
    val probability: Double
) : DiscreteDistribution {

    init {
        if (trials < 0) throw InvalidParameterException("trials must be non-negative, got $trials")
        if (probability !in 0.0..1.0) throw InvalidParameterException("probability must be in [0, 1], got $probability")
    }

    private val n = trials
    private val p = probability

    override fun pmf(k: Int): Double {
        if (k < 0 || k > n) return 0.0
        if (p == 0.0) return if (k == 0) 1.0 else 0.0
        if (p == 1.0) return if (k == n) 1.0 else 0.0
        return exp(logPmf(k))
    }

    override fun logPmf(k: Int): Double {
        if (k < 0 || k > n) return Double.NEGATIVE_INFINITY
        return lnCombination(n, k) + k * ln(p) + (n - k) * ln(1.0 - p)
    }

    override fun cdf(k: Int): Double {
        if (k < 0) return 0.0
        if (k >= n) return 1.0
        // I_{1-p}(n-k, k+1) using regularized beta
        return regularizedBeta(1.0 - p, (n - k).toDouble(), (k + 1).toDouble())
    }

    override fun quantile(p: Double): Int {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        if (p == 0.0) return 0
        if (p == 1.0) return n
        // Linear search from the mean
        var k = (n * this.p).toInt()
        while (k > 0 && cdf(k - 1) >= p) k--
        while (cdf(k) < p) k++
        return k
    }

    override val mean: Double get() = n * p
    override val variance: Double get() = n * p * (1.0 - p)

    override fun sample(random: Random): Int {
        // For small n, direct simulation
        if (n < 25) {
            var successes = 0
            for (i in 0 until n) {
                if (random.nextDouble() < p) successes++
            }
            return successes
        }
        // For large n, use normal approximation with correction
        val normal = NormalDistribution(mean, sqrt(variance))
        return normal.sample(random).roundToInt().coerceIn(0, n)
    }
}
