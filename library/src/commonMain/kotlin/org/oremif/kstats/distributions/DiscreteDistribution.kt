package org.oremif.kstats.distributions

import kotlin.random.Random

public interface DiscreteDistribution {
    public fun pmf(k: Int): Double
    public fun logPmf(k: Int): Double = kotlin.math.ln(pmf(k))
    public fun cdf(k: Int): Double
    public fun sf(k: Int): Double = 1.0 - cdf(k)
    public fun quantile(p: Double): Int
    public val mean: Double
    public val variance: Double
    public fun sample(random: Random = Random): Int
    public fun sample(n: Int, random: Random = Random): IntArray = IntArray(n) { sample(random) }
}
