@file:JvmName("LongStatistics")

package org.oremif.kstats.descriptive

import kotlin.jvm.JvmName

// ── Iterable<Long> extensions ───────────────────────────────────────────────
//
// All Long extensions convert values to Double via DoubleArray to avoid
// intermediate boxing. Double has 53 bits of mantissa, so Long values with
// absolute value > 2^53 (9_007_199_254_740_992) may lose precision in the
// least-significant digits. For data within that range the conversion is exact.

private fun Iterable<Long>.toStatArray(): DoubleArray {
    val list = toList()
    return DoubleArray(list.size) { list[it].toDouble() }
}

/**
 * Computes the arithmetic mean of the Long values.
 *
 * Values are converted to Double internally. Long values whose absolute value exceeds
 * 2^53 (9,007,199,254,740,992) may lose precision in the least-significant digits.
 *
 * ### Example:
 * ```kotlin
 * listOf(1L, 2L, 3L).mean() // 2.0
 * ```
 *
 * @return the arithmetic mean of the Long values as a Double.
 */
@JvmName("meanOfLong")
public fun Iterable<Long>.mean(): Double = toStatArray().mean()

/**
 * Computes the median of the Long values.
 *
 * Values are converted to Double internally. Long values whose absolute value exceeds
 * 2^53 (9,007,199,254,740,992) may lose precision in the least-significant digits.
 *
 * ### Example:
 * ```kotlin
 * listOf(1L, 2L, 3L, 4L).median() // 2.5
 * ```
 *
 * @return the median of the Long values as a Double.
 */
@JvmName("medianOfLong")
public fun Iterable<Long>.median(): Double = toStatArray().median()

/**
 * Computes the variance of the Long values.
 *
 * Values are converted to Double internally. Long values whose absolute value exceeds
 * 2^53 (9,007,199,254,740,992) may lose precision in the least-significant digits.
 *
 * ### Example:
 * ```kotlin
 * listOf(2L, 4L, 4L, 4L, 5L, 5L, 7L, 9L).variance() // 4.5714...
 * ```
 *
 * @param kind whether to compute sample or population variance. Defaults to [PopulationKind.SAMPLE].
 * @return the variance of the Long values as a Double.
 */
@JvmName("varianceOfLong")
public fun Iterable<Long>.variance(kind: PopulationKind = PopulationKind.SAMPLE): Double =
    toStatArray().variance(kind)

/**
 * Computes the standard deviation of the Long values.
 *
 * Values are converted to Double internally. Long values whose absolute value exceeds
 * 2^53 (9,007,199,254,740,992) may lose precision in the least-significant digits.
 *
 * ### Example:
 * ```kotlin
 * listOf(2L, 4L, 4L, 4L, 5L, 5L, 7L, 9L).standardDeviation() // 2.1380...
 * ```
 *
 * @param kind whether to compute sample or population standard deviation. Defaults to [PopulationKind.SAMPLE].
 * @return the standard deviation of the Long values as a Double.
 */
@JvmName("standardDeviationOfLong")
public fun Iterable<Long>.standardDeviation(kind: PopulationKind = PopulationKind.SAMPLE): Double =
    toStatArray().standardDeviation(kind)

/**
 * Computes the p-th percentile of the Long values.
 *
 * Values are converted to Double internally. Long values whose absolute value exceeds
 * 2^53 (9,007,199,254,740,992) may lose precision in the least-significant digits.
 *
 * ### Example:
 * ```kotlin
 * listOf(1L, 2L, 3L, 4L, 5L).percentile(50.0) // 3.0
 * ```
 *
 * @param p the percentile to compute, in [0, 100].
 * @param interpolation how to interpolate between data points. Defaults to [QuantileInterpolation.LINEAR].
 * @return the p-th percentile of the Long values as a Double.
 */
@JvmName("percentileOfLong")
public fun Iterable<Long>.percentile(
    p: Double,
    interpolation: QuantileInterpolation = QuantileInterpolation.LINEAR
): Double = toStatArray().percentile(p, interpolation)

/**
 * Computes a descriptive statistics summary of the Long values.
 *
 * Values are converted to Double internally. Long values whose absolute value exceeds
 * 2^53 (9,007,199,254,740,992) may lose precision in the least-significant digits.
 *
 * ### Example:
 * ```kotlin
 * val stats = listOf(1L, 2L, 3L, 4L, 5L).describe()
 * stats.mean   // 3.0
 * stats.median // 3.0
 * ```
 *
 * @return a [DescriptiveStatistics] summary of the values.
 */
@JvmName("describeOfLong")
public fun Iterable<Long>.describe(): DescriptiveStatistics = toStatArray().describe()
