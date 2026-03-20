package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InsufficientDataException
import kotlin.math.sqrt

/**
 * A snapshot of common descriptive statistics for a dataset.
 *
 * Returned by the [describe] function. Contains measures of central tendency, dispersion,
 * shape, and position. Fields that require a minimum number of data points are set to
 * [Double.NaN] when there is insufficient data (e.g. variance needs at least 2, skewness
 * needs at least 3, kurtosis needs at least 4).
 *
 * ### Example:
 * ```kotlin
 * val stats = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0).describe()
 * stats.mean     // 3.0
 * stats.median   // 3.0
 * stats.variance // 2.5
 * ```
 *
 * @property count the number of observations.
 * @property mean the arithmetic mean.
 * @property standardDeviation the sample standard deviation (divides by n-1), or NaN if n < 2.
 * @property min the smallest value in the dataset.
 * @property q1 the first quartile (25th percentile).
 * @property median the second quartile (50th percentile).
 * @property q3 the third quartile (75th percentile).
 * @property max the largest value in the dataset.
 * @property variance the sample variance (divides by n-1), or NaN if n < 2.
 * @property skewness the sample-adjusted Fisher-Pearson skewness, or NaN if n < 3.
 * @property kurtosis the sample-adjusted excess kurtosis, or NaN if n < 4.
 * @property sum the sum of all values.
 * @property range the difference between the maximum and minimum values.
 * @property interquartileRange the difference between Q3 and Q1.
 * @property standardError the standard error of the mean (stddev / sqrt(n)), or NaN if n < 2.
 */
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

/**
 * Computes a comprehensive descriptive statistics summary of the values in this iterable.
 *
 * Performs a single sort for order statistics (min, max, quartiles, median) and a single
 * Welford pass for mean, variance, skewness, and kurtosis. This is more efficient than
 * computing each statistic individually.
 *
 * ### Example:
 * ```kotlin
 * val stats = listOf(1.0, 2.0, 3.0, 4.0, 5.0).describe()
 * stats.count  // 5
 * stats.mean   // 3.0
 * stats.median // 3.0
 * ```
 *
 * @return a [DescriptiveStatistics] containing all computed statistics.
 */
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

    // Single Welford pass — mean + M2, with compensated sum
    var count = 0
    var mean = 0.0
    var m2 = 0.0
    var sum = 0.0
    var sumCompensation = 0.0
    for (x in list) {
        count++
        val delta = x - mean
        mean += delta / count
        val delta2 = x - mean
        m2 += delta * delta2
        val t = sum + x
        sumCompensation += if (kotlin.math.abs(sum) >= kotlin.math.abs(x)) (sum - t) + x else (x - t) + sum
        sum = t
    }
    sum += sumCompensation

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

/**
 * Computes a comprehensive descriptive statistics summary of the values in this array.
 *
 * Performs a single sort for order statistics and a single Welford pass for mean, variance,
 * skewness, and kurtosis.
 *
 * ### Example:
 * ```kotlin
 * val stats = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0).describe()
 * stats.count  // 5
 * stats.mean   // 3.0
 * stats.median // 3.0
 * ```
 *
 * @return a [DescriptiveStatistics] containing all computed statistics.
 */
public fun DoubleArray.describe(): DescriptiveStatistics = asIterable().describe()
