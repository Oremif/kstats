package org.oremif.kstats.descriptive

import kotlin.math.sqrt

public data class DescriptiveStatistics(
    val count: Int,
    val mean: Double,
    val standardDeviation: Double,
    val min: Double,
    val q1: Double,
    val median: Double,
    val q3: Double,
    val max: Double,
    val variance: Double,
    val skewness: Double,
    val kurtosis: Double,
    val sum: Double,
    val range: Double,
    val interquartileRange: Double,
    val standardError: Double
)

public fun Iterable<Double>.describe(): DescriptiveStatistics {
    val list = toList()
    require(list.size >= 4) { "describe() requires at least 4 elements" }

    val sorted = list.sorted()
    val n = sorted.size
    val sum = sorted.sum()
    val m = sum / n
    val minVal = sorted.first()
    val maxVal = sorted.last()
    val (q1, med, q3) = sorted.quartiles()
    val variance = list.variance(PopulationKind.SAMPLE)
    val sd = sqrt(variance)

    return DescriptiveStatistics(
        count = n,
        mean = m,
        standardDeviation = sd,
        min = minVal,
        q1 = q1,
        median = med,
        q3 = q3,
        max = maxVal,
        variance = variance,
        skewness = list.skewness(PopulationKind.SAMPLE),
        kurtosis = list.kurtosis(PopulationKind.SAMPLE),
        sum = sum,
        range = maxVal - minVal,
        interquartileRange = q3 - q1,
        standardError = sd / sqrt(n.toDouble())
    )
}

public fun DoubleArray.describe(): DescriptiveStatistics = asIterable().describe()
