package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InsufficientDataException
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
    if (list.isEmpty()) throw InsufficientDataException("describe() requires at least 1 element")

    // Single sort — order statistics
    val sorted = list.sorted()
    val n = sorted.size
    val minVal = sorted.first()
    val maxVal = sorted.last()
    val q1 = sortedQuantile(sorted, 0.25)
    val med = sortedQuantile(sorted, 0.50)
    val q3 = sortedQuantile(sorted, 0.75)

    // Single Welford pass — mean + M2
    var count = 0
    var mean = 0.0
    var m2 = 0.0
    for (x in list) {
        count++
        val delta = x - mean
        mean += delta / count
        val delta2 = x - mean
        m2 += delta * delta2
    }

    val sum = mean * n

    // Variance-dependent fields: need n >= 2
    val popVariance = m2 / n
    val variance = if (n >= 2) m2 / (n - 1) else Double.NaN
    val sd = if (n >= 2) sqrt(variance) else Double.NaN
    val se = if (n >= 2) sd / sqrt(n.toDouble()) else Double.NaN

    // Single z-pass — skewness (z³) + kurtosis (z⁴)
    var skew = Double.NaN
    var kurt = Double.NaN
    if (n >= 3 && popVariance != 0.0) {
        val popSd = sqrt(popVariance)
        var sumZ3 = 0.0
        var sumZ4 = 0.0
        for (x in list) {
            val z = (x - mean) / popSd
            val z2 = z * z
            sumZ3 += z2 * z
            sumZ4 += z2 * z2
        }
        val g1 = sumZ3 / n
        skew = sqrt(n.toDouble() * (n - 1)) / (n - 2) * g1

        if (n >= 4) {
            val g2 = sumZ4 / n
            val nd = n.toDouble()
            kurt = (nd - 1.0) / ((nd - 2.0) * (nd - 3.0)) * ((nd + 1.0) * g2 - 3.0 * (nd - 1.0))
        }
    }

    return DescriptiveStatistics(
        count = n,
        mean = mean,
        standardDeviation = sd,
        min = minVal,
        q1 = q1,
        median = med,
        q3 = q3,
        max = maxVal,
        variance = variance,
        skewness = skew,
        kurtosis = kurt,
        sum = sum,
        range = maxVal - minVal,
        interquartileRange = q3 - q1,
        standardError = se
    )
}

/** Compute quantile from an already-sorted list using linear interpolation. */
private fun sortedQuantile(sorted: List<Double>, q: Double): Double {
    if (sorted.size == 1) return sorted[0]
    val pos = q * (sorted.size - 1)
    val lo = kotlin.math.floor(pos).toInt()
    val hi = kotlin.math.ceil(pos).toInt()
    val frac = pos - lo
    return sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

public fun DoubleArray.describe(): DescriptiveStatistics = asIterable().describe()
