package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.DegenerateDataException
import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.descriptive.PopulationKind.SAMPLE
import kotlin.math.abs
import kotlin.math.floor
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
    if (count == 0) throw InsufficientDataException("Collection must not be empty")
    val divisor = if (kind == SAMPLE) {
        if (count <= 1) throw InsufficientDataException("Sample variance requires at least 2 elements")
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
    if (list.isEmpty()) throw InsufficientDataException("Collection must not be empty")
    return list.max() - list.min()
}

public fun DoubleArray.range(): Double {
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")
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
    if (list.isEmpty()) throw InsufficientDataException("Collection must not be empty")
    val m = list.mean()
    return list.map { abs(it - m) }.mean()
}

public fun DoubleArray.meanAbsoluteDeviation(): Double = asIterable().meanAbsoluteDeviation()

// ── medianAbsoluteDeviation ─────────────────────────────────────────────────

public fun Iterable<Double>.medianAbsoluteDeviation(): Double {
    val list = toList()
    if (list.isEmpty()) throw InsufficientDataException("Collection must not be empty")
    val med = list.median()
    return list.map { abs(it - med) }.median()
}

public fun DoubleArray.medianAbsoluteDeviation(): Double = asIterable().medianAbsoluteDeviation()

// ── standardError ───────────────────────────────────────────────────────────

public fun Iterable<Double>.standardError(): Double {
    val list = toList()
    if (list.size <= 1) throw InsufficientDataException("Standard error requires at least 2 elements")
    return list.standardDeviation() / sqrt(list.size.toDouble())
}

public fun DoubleArray.standardError(): Double {
    if (size <= 1) throw InsufficientDataException("Standard error requires at least 2 elements")
    return standardDeviation() / sqrt(size.toDouble())
}

// ── coefficientOfVariation ──────────────────────────────────────────────────

public fun Iterable<Double>.coefficientOfVariation(kind: PopulationKind = SAMPLE): Double {
    val m = mean()
    if (m == 0.0) throw DegenerateDataException("Coefficient of variation is undefined when mean is zero")
    return standardDeviation(kind) / m
}

public fun DoubleArray.coefficientOfVariation(kind: PopulationKind = SAMPLE): Double =
    asIterable().coefficientOfVariation(kind)

// ── trimmedVariance ─────────────────────────────────────────────────────────

/**
 * Computes the variance of the values after removing a fraction from each tail.
 *
 * The trimmed variance sorts the data, discards the lowest and highest [proportion] of
 * values, and computes the variance of the remaining middle portion using Welford's
 * numerically stable algorithm. This makes it more robust to outliers than the regular
 * variance. A proportion of 0.0 gives the ordinary variance.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0).trimmedVariance(0.1) // 6.0
 * ```
 *
 * @param proportion the fraction of values to remove from each tail, in [0.0, 0.5).
 * For example, 0.1 removes the lowest 10% and highest 10%.
 * @param kind whether to compute sample or population variance. Defaults to [PopulationKind.SAMPLE],
 * which divides by n-1 (Bessel's correction) where n is the count after trimming.
 * @return the variance of the remaining values after trimming.
 * @see trimmedMean
 * @see trimmedStd
 */
public fun DoubleArray.trimmedVariance(proportion: Double, kind: PopulationKind = SAMPLE): Double {
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")
    if (proportion.isNaN() || proportion < 0.0 || proportion >= 0.5)
        throw InvalidParameterException("proportion must be in [0.0, 0.5), got $proportion")
    val sorted = sortedArray()
    val k = floor(size * proportion).toInt()
    val m = size - 2 * k
    val divisor = if (kind == SAMPLE) {
        if (m <= 1) throw InsufficientDataException(
            "Sample variance of trimmed data requires at least 2 remaining elements, got $m"
        )
        m - 1
    } else {
        m
    }
    var mean = 0.0
    var m2 = 0.0
    for (i in k until size - k) {
        val j = i - k + 1
        val delta = sorted[i] - mean
        mean += delta / j
        val delta2 = sorted[i] - mean
        m2 += delta * delta2
    }
    return m2 / divisor
}

/**
 * Computes the variance of the values after removing a fraction from each tail.
 *
 * The trimmed variance sorts the data, discards the lowest and highest [proportion] of
 * values, and computes the variance of the remaining middle portion. This makes it more
 * robust to outliers than the regular variance. A proportion of 0.0 gives the ordinary variance.
 *
 * ### Example:
 * ```kotlin
 * listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0).trimmedVariance(0.1) // 6.0
 * ```
 *
 * @param proportion the fraction of values to remove from each tail, in [0.0, 0.5).
 * For example, 0.1 removes the lowest 10% and highest 10%.
 * @param kind whether to compute sample or population variance. Defaults to [PopulationKind.SAMPLE],
 * which divides by n-1 (Bessel's correction) where n is the count after trimming.
 * @return the variance of the remaining values after trimming.
 * @see trimmedMean
 * @see trimmedStd
 */
public fun Iterable<Double>.trimmedVariance(proportion: Double, kind: PopulationKind = SAMPLE): Double =
    toList().toDoubleArray().trimmedVariance(proportion, kind)

/**
 * Computes the variance of the values after removing a fraction from each tail.
 *
 * The trimmed variance sorts the data, discards the lowest and highest [proportion] of
 * values, and computes the variance of the remaining middle portion. This makes it more
 * robust to outliers than the regular variance. A proportion of 0.0 gives the ordinary variance.
 *
 * ### Example:
 * ```kotlin
 * sequenceOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0).trimmedVariance(0.1) // 6.0
 * ```
 *
 * @param proportion the fraction of values to remove from each tail, in [0.0, 0.5).
 * For example, 0.1 removes the lowest 10% and highest 10%.
 * @param kind whether to compute sample or population variance. Defaults to [PopulationKind.SAMPLE],
 * which divides by n-1 (Bessel's correction) where n is the count after trimming.
 * @return the variance of the remaining values after trimming.
 * @see trimmedMean
 * @see trimmedStd
 */
public fun Sequence<Double>.trimmedVariance(proportion: Double, kind: PopulationKind = SAMPLE): Double =
    toList().toDoubleArray().trimmedVariance(proportion, kind)

// ── trimmedStd ──────────────────────────────────────────────────────────────

/**
 * Computes the standard deviation of the values after removing a fraction from each tail.
 *
 * This is the square root of the trimmed variance. See [trimmedVariance] for details on how
 * trimming works. A proportion of 0.0 gives the ordinary standard deviation.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0).trimmedStd(0.1) // 2.449...
 * ```
 *
 * @param proportion the fraction of values to remove from each tail, in [0.0, 0.5).
 * For example, 0.1 removes the lowest 10% and highest 10%.
 * @param kind whether to compute sample or population standard deviation. Defaults to
 * [PopulationKind.SAMPLE], which divides by n-1 (Bessel's correction) where n is the count
 * after trimming.
 * @return the standard deviation of the remaining values after trimming.
 * @see trimmedVariance
 * @see trimmedMean
 */
public fun DoubleArray.trimmedStd(proportion: Double, kind: PopulationKind = SAMPLE): Double =
    sqrt(trimmedVariance(proportion, kind))

/**
 * Computes the standard deviation of the values after removing a fraction from each tail.
 *
 * This is the square root of the trimmed variance. See [trimmedVariance] for details on how
 * trimming works. A proportion of 0.0 gives the ordinary standard deviation.
 *
 * ### Example:
 * ```kotlin
 * listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0).trimmedStd(0.1) // 2.449...
 * ```
 *
 * @param proportion the fraction of values to remove from each tail, in [0.0, 0.5).
 * For example, 0.1 removes the lowest 10% and highest 10%.
 * @param kind whether to compute sample or population standard deviation. Defaults to
 * [PopulationKind.SAMPLE], which divides by n-1 (Bessel's correction) where n is the count
 * after trimming.
 * @return the standard deviation of the remaining values after trimming.
 * @see trimmedVariance
 * @see trimmedMean
 */
public fun Iterable<Double>.trimmedStd(proportion: Double, kind: PopulationKind = SAMPLE): Double =
    sqrt(trimmedVariance(proportion, kind))

/**
 * Computes the standard deviation of the values after removing a fraction from each tail.
 *
 * This is the square root of the trimmed variance. See [trimmedVariance] for details on how
 * trimming works. A proportion of 0.0 gives the ordinary standard deviation.
 *
 * ### Example:
 * ```kotlin
 * sequenceOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0).trimmedStd(0.1) // 2.449...
 * ```
 *
 * @param proportion the fraction of values to remove from each tail, in [0.0, 0.5).
 * For example, 0.1 removes the lowest 10% and highest 10%.
 * @param kind whether to compute sample or population standard deviation. Defaults to
 * [PopulationKind.SAMPLE], which divides by n-1 (Bessel's correction) where n is the count
 * after trimming.
 * @return the standard deviation of the remaining values after trimming.
 * @see trimmedVariance
 * @see trimmedMean
 */
public fun Sequence<Double>.trimmedStd(proportion: Double, kind: PopulationKind = SAMPLE): Double =
    sqrt(trimmedVariance(proportion, kind))
