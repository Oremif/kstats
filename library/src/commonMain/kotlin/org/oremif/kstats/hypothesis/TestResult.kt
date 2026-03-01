package org.oremif.kstats.hypothesis

public data class TestResult(
    val testName: String,
    val statistic: Double,
    val pValue: Double,
    val degreesOfFreedom: Double = Double.NaN,
    val alternative: Alternative = Alternative.TWO_SIDED,
    val confidenceInterval: Pair<Double, Double>? = null,
    val additionalInfo: Map<String, Double> = emptyMap()
) {
    public fun isSignificant(alpha: Double = 0.05): Boolean = pValue < alpha
}
