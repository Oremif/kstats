package org.oremif.kstats.distributions

import kotlin.math.sqrt

public sealed interface Distribution {
    public val mean: Double
    public val variance: Double
    public val standardDeviation: Double get() = sqrt(variance)
    public val skewness: Double
    public val kurtosis: Double
    public val entropy: Double
    public fun cdf(x: Double): Double
    public fun sf(x: Double): Double = 1.0 - cdf(x)
    public fun quantile(p: Double): Double
}
