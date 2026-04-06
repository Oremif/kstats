@file:JvmName("IntStatistics")

package org.oremif.kstats.descriptive

import kotlin.jvm.JvmName

// ── Iterable<Int> extensions ────────────────────────────────────────────────
//
// Int values are converted to Double internally via DoubleArray to avoid
// intermediate boxing. Since Int fits within Double's 53-bit mantissa,
// the conversion is always exact.

private fun Iterable<Int>.toStatArray(): DoubleArray {
    val list = toList()
    return DoubleArray(list.size) { list[it].toDouble() }
}

/**
 * Computes the arithmetic mean of the Int values.
 *
 * Values are converted to Double internally. The conversion is exact for all Int values.
 *
 * ### Example:
 * ```kotlin
 * listOf(1, 2, 3).mean() // 2.0
 * ```
 *
 * @return the arithmetic mean of the Int values as a Double.
 */
@JvmName("meanOfInt")
public fun Iterable<Int>.mean(): Double = toStatArray().mean()

/**
 * Computes the median of the Int values.
 *
 * Values are converted to Double internally. The conversion is exact for all Int values.
 *
 * ### Example:
 * ```kotlin
 * listOf(1, 2, 3, 4).median() // 2.5
 * ```
 *
 * @return the median of the Int values as a Double.
 */
@JvmName("medianOfInt")
public fun Iterable<Int>.median(): Double = toStatArray().median()

/**
 * Computes the variance of the Int values.
 *
 * Values are converted to Double internally. The conversion is exact for all Int values.
 *
 * ### Example:
 * ```kotlin
 * listOf(2, 4, 4, 4, 5, 5, 7, 9).variance() // 4.5714...
 * ```
 *
 * @param kind whether to compute sample or population variance. Defaults to [PopulationKind.SAMPLE].
 * @return the variance of the Int values as a Double.
 */
@JvmName("varianceOfInt")
public fun Iterable<Int>.variance(kind: PopulationKind = PopulationKind.SAMPLE): Double =
    toStatArray().variance(kind)

/**
 * Computes the standard deviation of the Int values.
 *
 * Values are converted to Double internally. The conversion is exact for all Int values.
 *
 * ### Example:
 * ```kotlin
 * listOf(2, 4, 4, 4, 5, 5, 7, 9).standardDeviation() // 2.1380...
 * ```
 *
 * @param kind whether to compute sample or population standard deviation. Defaults to [PopulationKind.SAMPLE].
 * @return the standard deviation of the Int values as a Double.
 */
@JvmName("standardDeviationOfInt")
public fun Iterable<Int>.standardDeviation(kind: PopulationKind = PopulationKind.SAMPLE): Double =
    toStatArray().standardDeviation(kind)

/**
 * Computes the p-th percentile of the Int values.
 *
 * Values are converted to Double internally. The conversion is exact for all Int values.
 *
 * ### Example:
 * ```kotlin
 * listOf(1, 2, 3, 4, 5).percentile(50.0) // 3.0
 * ```
 *
 * @param p the percentile to compute, in [0, 100].
 * @param method the quantile estimation method. Defaults to [QuantileMethod.LINEAR] (HF7).
 * @return the p-th percentile of the Int values as a Double.
 */
@JvmName("percentileOfInt")
public fun Iterable<Int>.percentile(
    p: Double,
    method: QuantileMethod = QuantileMethod.LINEAR,
): Double = toStatArray().percentile(p, method)

/**
 * Computes the p-th percentile of the Int values.
 *
 * Values are converted to Double internally. The conversion is exact for all Int values.
 *
 * @param p the percentile to compute, in [0, 100].
 * @param interpolation the interpolation mode.
 * @return the p-th percentile of the Int values as a Double.
 */
@Suppress("DEPRECATION")
@Deprecated(
    "Use the overload with QuantileMethod instead.",
    ReplaceWith("percentile(p, interpolation.toQuantileMethod())"),
    DeprecationLevel.WARNING,
)
@JvmName("percentileOfIntDeprecated")
public fun Iterable<Int>.percentile(
    p: Double,
    interpolation: QuantileInterpolation,
): Double = toStatArray().percentile(p, interpolation.toQuantileMethod())

/**
 * Computes a descriptive statistics summary of the Int values.
 *
 * Values are converted to Double internally. The conversion is exact for all Int values.
 *
 * ### Example:
 * ```kotlin
 * val stats = listOf(1, 2, 3, 4, 5).describe()
 * stats.mean   // 3.0
 * stats.median // 3.0
 * ```
 *
 * @return a [DescriptiveStatistics] summary of the values.
 */
@JvmName("describeOfInt")
public fun Iterable<Int>.describe(): DescriptiveStatistics = toStatArray().describe()
