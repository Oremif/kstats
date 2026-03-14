package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

public data class TriangularDistribution(
    val a: Double,
    val b: Double,
    val c: Double,
) : ContinuousDistribution {

    init {
        if (a >= b) throw InvalidParameterException("a must be less than b, got a=$a, b=$b")
        if (c < a || c > b) throw InvalidParameterException("c must be in [a, b], got a=$a, b=$b, c=$c")
    }

    private val ba = b - a
    private val ca = c - a
    private val bc = b - c
    private val pc = ca / ba

    override fun pdf(x: Double): Double = when {
        x < a || x > b -> 0.0
        x < c -> 2.0 * (x - a) / (ba * ca)
        x > c -> 2.0 * (b - x) / (ba * bc)
        else -> 2.0 / ba // x == c
    }

    override fun logPdf(x: Double): Double = when {
        x < a || x > b -> Double.NEGATIVE_INFINITY
        x < c -> ln(2.0) + ln(x - a) - ln(ba) - ln(ca)
        x > c -> ln(2.0) + ln(b - x) - ln(ba) - ln(bc)
        else -> ln(2.0) - ln(ba) // x == c
    }

    override fun cdf(x: Double): Double = when {
        x <= a -> 0.0
        x <= c -> (x - a) * (x - a) / (ba * ca)
        x < b -> 1.0 - (b - x) * (b - x) / (ba * bc)
        else -> 1.0
    }

    override fun sf(x: Double): Double = when {
        x <= a -> 1.0
        x <= c -> 1.0 - (x - a) * (x - a) / (ba * ca)
        x < b -> (b - x) * (b - x) / (ba * bc)
        else -> 0.0
    }

    override fun quantile(p: Double): Double {
        if (p !in 0.0..1.0) throw InvalidParameterException("p must be in [0, 1], got $p")
        return when {
            p == 0.0 -> a
            p == 1.0 -> b
            p <= pc -> a + sqrt(p * ba * ca)
            else -> b - sqrt((1.0 - p) * ba * bc)
        }
    }

    override val mean: Double get() = (a + b + c) / 3.0

    override val variance: Double get() = (a * a + b * b + c * c - a * b - a * c - b * c) / 18.0

    override val skewness: Double get() {
        val num = SQRT2 * (a + b - 2.0 * c) * (2.0 * a - b - c) * (a - 2.0 * b + c)
        val den = 5.0 * (a * a + b * b + c * c - a * b - a * c - b * c).let { it * sqrt(it) }
        return num / den
    }

    override val kurtosis: Double get() = -0.6

    override val entropy: Double get() = 0.5 + ln(ba / 2.0)

    override fun sample(random: Random): Double = quantile(random.nextDouble())

    private companion object {
        private val SQRT2 = sqrt(2.0)
    }
}
