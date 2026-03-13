@file:JvmName("LongStatistics")

package org.oremif.kstats.descriptive

import kotlin.jvm.JvmName

// ── Iterable<Long> extensions ───────────────────────────────────────────────

@JvmName("meanOfLong")
public fun Iterable<Long>.mean(): Double = map { it.toDouble() }.mean()

@JvmName("medianOfLong")
public fun Iterable<Long>.median(): Double = map { it.toDouble() }.median()

@JvmName("varianceOfLong")
public fun Iterable<Long>.variance(kind: PopulationKind = PopulationKind.SAMPLE): Double =
    map { it.toDouble() }.variance(kind)

@JvmName("standardDeviationOfLong")
public fun Iterable<Long>.standardDeviation(kind: PopulationKind = PopulationKind.SAMPLE): Double =
    map { it.toDouble() }.standardDeviation(kind)

@JvmName("percentileOfLong")
public fun Iterable<Long>.percentile(
    p: Double,
    interpolation: QuantileInterpolation = QuantileInterpolation.LINEAR
): Double = map { it.toDouble() }.percentile(p, interpolation)

@JvmName("describeOfLong")
public fun Iterable<Long>.describe(): DescriptiveStatistics = map { it.toDouble() }.describe()
