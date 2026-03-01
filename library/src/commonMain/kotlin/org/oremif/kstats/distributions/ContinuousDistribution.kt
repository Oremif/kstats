package org.oremif.kstats.distributions

import kotlin.random.Random

public interface ContinuousDistribution {
    public fun pdf(x: Double): Double
    public fun logPdf(x: Double): Double = kotlin.math.ln(pdf(x))
    public fun cdf(x: Double): Double
    public fun sf(x: Double): Double = 1.0 - cdf(x)
    public fun quantile(p: Double): Double
    public val mean: Double
    public val variance: Double
    public val standardDeviation: Double get() = kotlin.math.sqrt(variance)
    public val skewness: Double
    public val kurtosis: Double
    public fun sample(random: Random = Random): Double
    public fun sample(n: Int, random: Random = Random): DoubleArray = DoubleArray(n) { sample(random) }
}
