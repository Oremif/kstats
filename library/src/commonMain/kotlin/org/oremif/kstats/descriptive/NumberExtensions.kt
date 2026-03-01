@file:JvmName("IntStatistics")

package org.oremif.kstats.descriptive

import kotlin.jvm.JvmName

// ── Iterable<Int> extensions ────────────────────────────────────────────────

@JvmName("meanOfInt")
public fun Iterable<Int>.mean(): Double = map { it.toDouble() }.mean()

@JvmName("medianOfInt")
public fun Iterable<Int>.median(): Double = map { it.toDouble() }.median()

@JvmName("varianceOfInt")
public fun Iterable<Int>.variance(kind: PopulationKind = PopulationKind.SAMPLE): Double =
    map { it.toDouble() }.variance(kind)

@JvmName("standardDeviationOfInt")
public fun Iterable<Int>.standardDeviation(kind: PopulationKind = PopulationKind.SAMPLE): Double =
    map { it.toDouble() }.standardDeviation(kind)

@JvmName("percentileOfInt")
public fun Iterable<Int>.percentile(
    p: Double,
    interpolation: QuantileInterpolation = QuantileInterpolation.LINEAR
): Double = map { it.toDouble() }.percentile(p, interpolation)

@JvmName("describeOfInt")
public fun Iterable<Int>.describe(): DescriptiveStatistics = map { it.toDouble() }.describe()
