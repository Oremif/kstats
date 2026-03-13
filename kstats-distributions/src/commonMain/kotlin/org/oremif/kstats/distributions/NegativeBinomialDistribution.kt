package org.oremif.kstats.distributions

import org.oremif.kstats.core.lnCombination
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

public data class NegativeBinomialDistribution(
    val successes: Int,
    val probability: Double
) : DiscreteDistribution {

    init {
        if (successes <= 0) throw InvalidParameterException("successes must be positive, got $successes")
        if (probability <= 0.0 || probability > 1.0) throw InvalidParameterException("probability must be in (0, 1], got $probability")
    }

    private val r = successes
    private val p = probability
    private val q = 1.0 - p

    // k = number of failures before r-th success
    override fun pmf(k: Int): Double {
        if (k < 0) return 0.0
        if (p == 1.0) return if (k == 0) 1.0 else 0.0
        return exp(logPmf(k))
    }

    override fun logPmf(k: Int): Double {
        if (k < 0) return Double.NEGATIVE_INFINITY
        if (p == 1.0) return if (k == 0) 0.0 else Double.NEGATIVE_INFINITY
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

    override fun quantileInt(p: Double): Int {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
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

    override val skewness: Double get() = (2.0 - p) / sqrt(r.toDouble() * q)
    override val kurtosis: Double get() = 6.0 / r + p * p / (r * q)
    override val entropy: Double get() {
        var h = 0.0
        var cumP = 0.0
        var k = 0
        while (cumP < 1.0 - 1e-15) {
            val pk = pmf(k)
            if (pk > 0.0) {
                h -= pk * ln(pk)
                cumP += pk
            }
            k++
            if (k > 100_000) break
        }
        return h
    }

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
