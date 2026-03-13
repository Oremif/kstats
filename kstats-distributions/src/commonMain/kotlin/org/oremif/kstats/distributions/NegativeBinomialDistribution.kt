package org.oremif.kstats.distributions

import org.oremif.kstats.core.lnCombination
import kotlin.math.*
import kotlin.random.Random

public data class NegativeBinomialDistribution(
    val successes: Int,
    val probability: Double
) : DiscreteDistribution {

    init {
        require(successes > 0) { "successes must be positive, got $successes" }
        require(probability > 0.0 && probability <= 1.0) { "probability must be in (0, 1], got $probability" }
    }

    private val r = successes
    private val p = probability
    private val q = 1.0 - p

    // k = number of failures before r-th success
    override fun pmf(k: Int): Double {
        if (k < 0) return 0.0
        return exp(logPmf(k))
    }

    override fun logPmf(k: Int): Double {
        if (k < 0) return Double.NEGATIVE_INFINITY
        return lnCombination(k + r - 1, k) + r * ln(p) + k * ln(q)
    }

    override fun cdf(k: Int): Double {
        if (k < 0) return 0.0
        var sum = 0.0
        for (i in 0..k) {
            sum += pmf(i)
        }
        return sum.coerceAtMost(1.0)
    }

    override fun quantile(p: Double): Int {
        require(p in 0.0..1.0) { "p must be in [0, 1], got $p" }
        var cumulative = 0.0
        var k = 0
        while (cumulative < p) {
            cumulative += pmf(k)
            if (cumulative >= p) return k
            k++
            if (k > 10000) break // safety
        }
        return k
    }

    override val mean: Double get() = r * q / p
    override val variance: Double get() = r * q / (p * p)

    override fun sample(random: Random): Int {
        // Sum of r geometric(p) random variables
        val geo = GeometricDistribution(p)
        var sum = 0
        for (i in 0 until r) {
            sum += geo.sample(random)
        }
        return sum
    }
}
