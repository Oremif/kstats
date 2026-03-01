package org.oremif.kstats.descriptive

import org.oremif.kstats.descriptive.PopulationKind.SAMPLE
import kotlin.math.abs
import kotlin.math.sqrt

// ── variance (Welford's online algorithm) ───────────────────────────────────

public fun Iterable<Double>.variance(kind: PopulationKind = SAMPLE): Double {
    var count = 0
    var mean = 0.0
    var m2 = 0.0
    for (x in this) {
        count++
        val delta = x - mean
        mean += delta / count
        val delta2 = x - mean
        m2 += delta * delta2
    }
    require(count > 0) { "Collection must not be empty" }
    val divisor = if (kind == SAMPLE) {
        require(count > 1) { "Sample variance requires at least 2 elements" }
        count - 1
    } else {
        count
    }
    return m2 / divisor
}

public fun DoubleArray.variance(kind: PopulationKind = SAMPLE): Double = asIterable().variance(kind)

// ── standardDeviation ───────────────────────────────────────────────────────

public fun Iterable<Double>.standardDeviation(kind: PopulationKind = SAMPLE): Double =
    sqrt(variance(kind))

public fun DoubleArray.standardDeviation(kind: PopulationKind = SAMPLE): Double =
    sqrt(variance(kind))

// ── range ───────────────────────────────────────────────────────────────────

public fun Iterable<Double>.range(): Double {
    val list = toList()
    require(list.isNotEmpty()) { "Collection must not be empty" }
    return list.max() - list.min()
}

public fun DoubleArray.range(): Double {
    require(isNotEmpty()) { "Array must not be empty" }
    return max() - min()
}

// ── interquartileRange ──────────────────────────────────────────────────────

public fun Iterable<Double>.interquartileRange(): Double {
    val q = quartiles()
    return q.third - q.first
}

public fun DoubleArray.interquartileRange(): Double = asIterable().interquartileRange()

// ── meanAbsoluteDeviation ───────────────────────────────────────────────────

public fun Iterable<Double>.meanAbsoluteDeviation(): Double {
    val list = toList()
    require(list.isNotEmpty()) { "Collection must not be empty" }
    val m = list.mean()
    return list.map { abs(it - m) }.mean()
}

public fun DoubleArray.meanAbsoluteDeviation(): Double = asIterable().meanAbsoluteDeviation()

// ── medianAbsoluteDeviation ─────────────────────────────────────────────────

public fun Iterable<Double>.medianAbsoluteDeviation(): Double {
    val list = toList()
    require(list.isNotEmpty()) { "Collection must not be empty" }
    val med = list.median()
    return list.map { abs(it - med) }.median()
}

public fun DoubleArray.medianAbsoluteDeviation(): Double = asIterable().medianAbsoluteDeviation()

// ── standardError ───────────────────────────────────────────────────────────

public fun Iterable<Double>.standardError(): Double {
    val list = toList()
    require(list.size > 1) { "Standard error requires at least 2 elements" }
    return list.standardDeviation() / sqrt(list.size.toDouble())
}

public fun DoubleArray.standardError(): Double {
    require(size > 1) { "Standard error requires at least 2 elements" }
    return standardDeviation() / sqrt(size.toDouble())
}

// ── coefficientOfVariation ──────────────────────────────────────────────────

public fun Iterable<Double>.coefficientOfVariation(kind: PopulationKind = SAMPLE): Double {
    val m = mean()
    require(m != 0.0) { "Coefficient of variation is undefined when mean is zero" }
    return standardDeviation(kind) / m
}

public fun DoubleArray.coefficientOfVariation(kind: PopulationKind = SAMPLE): Double =
    asIterable().coefficientOfVariation(kind)
