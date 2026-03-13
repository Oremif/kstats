@file:JvmName("LongStatistics")

package org.oremif.kstats.descriptive

import kotlin.jvm.JvmName

// ── Iterable<Long> extensions ───────────────────────────────────────────────
//
// All Long extensions convert values to Double via Long.toDouble().
// Double has 53 bits of mantissa, so Long values with absolute value > 2^53
// (9_007_199_254_740_992) may lose precision in the least-significant digits.
// For data within that range the conversion is exact.

/**
 * Arithmetic mean of Long values.
 *
 * **Precision:** Long values whose absolute value exceeds 2^53 (9,007,199,254,740,992)
 * may lose precision when converted to Double.
 */
@JvmName("meanOfLong")
public fun Iterable<Long>.mean(): Double = map { it.toDouble() }.mean()

/**
 * Median of Long values.
 *
 * **Precision:** Long values whose absolute value exceeds 2^53 (9,007,199,254,740,992)
 * may lose precision when converted to Double.
 */
@JvmName("medianOfLong")
public fun Iterable<Long>.median(): Double = map { it.toDouble() }.median()

/**
 * Variance of Long values.
 *
 * **Precision:** Long values whose absolute value exceeds 2^53 (9,007,199,254,740,992)
 * may lose precision when converted to Double.
 */
@JvmName("varianceOfLong")
public fun Iterable<Long>.variance(kind: PopulationKind = PopulationKind.SAMPLE): Double =
    map { it.toDouble() }.variance(kind)

/**
 * Standard deviation of Long values.
 *
 * **Precision:** Long values whose absolute value exceeds 2^53 (9,007,199,254,740,992)
 * may lose precision when converted to Double.
 */
@JvmName("standardDeviationOfLong")
public fun Iterable<Long>.standardDeviation(kind: PopulationKind = PopulationKind.SAMPLE): Double =
    map { it.toDouble() }.standardDeviation(kind)

/**
 * Percentile of Long values.
 *
 * **Precision:** Long values whose absolute value exceeds 2^53 (9,007,199,254,740,992)
 * may lose precision when converted to Double.
 */
@JvmName("percentileOfLong")
public fun Iterable<Long>.percentile(
    p: Double,
    interpolation: QuantileInterpolation = QuantileInterpolation.LINEAR
): Double = map { it.toDouble() }.percentile(p, interpolation)

/**
 * Descriptive statistics summary of Long values.
 *
 * **Precision:** Long values whose absolute value exceeds 2^53 (9,007,199,254,740,992)
 * may lose precision when converted to Double.
 */
@JvmName("describeOfLong")
public fun Iterable<Long>.describe(): DescriptiveStatistics = map { it.toDouble() }.describe()
