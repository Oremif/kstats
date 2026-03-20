package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.DegenerateDataException
import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.descriptive.PopulationKind.SAMPLE
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.sqrt

// ── variance (Welford's online algorithm) ───────────────────────────────────

/**
 * Computes the variance of the values in this iterable.
 *
 * Variance measures how far values spread from their mean. Uses Welford's numerically
 * stable single-pass algorithm. Sample variance (default) divides by n-1 (Bessel's
 * correction) for an unbiased estimate; population variance divides by n.
 *
 * ### Example:
 * ```kotlin
 * listOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0).variance()                         // 4.5714...
 * listOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0).variance(PopulationKind.POPULATION) // 4.0
 * ```
 *
 * @param kind whether to compute sample or population variance. Defaults to [PopulationKind.SAMPLE].
 * @return the variance of the elements.
 */
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

/**
 * Computes the variance of the values in this array.
 *
 * Variance measures how far values spread from their mean. Uses Welford's numerically
 * stable single-pass algorithm.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0).variance() // 4.5714...
 * ```
 *
 * @param kind whether to compute sample or population variance. Defaults to [PopulationKind.SAMPLE].
 * @return the variance of the array elements.
 */
public fun DoubleArray.variance(kind: PopulationKind = SAMPLE): Double = asIterable().variance(kind)

// ── standardDeviation ───────────────────────────────────────────────────────

/**
 * Computes the standard deviation of the values in this iterable.
 *
 * The standard deviation is the square root of the [variance]. It has the same unit as the
 * data, making it easier to interpret than variance.
 *
 * ### Example:
 * ```kotlin
 * listOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0).standardDeviation() // 2.1380...
 * ```
 *
 * @param kind whether to compute sample or population standard deviation. Defaults to [PopulationKind.SAMPLE].
 * @return the standard deviation of the elements.
 */
public fun Iterable<Double>.standardDeviation(kind: PopulationKind = SAMPLE): Double =
    sqrt(variance(kind))

/**
 * Computes the standard deviation of the values in this array.
 *
 * The standard deviation is the square root of the [variance].
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0).standardDeviation() // 2.1380...
 * ```
 *
 * @param kind whether to compute sample or population standard deviation. Defaults to [PopulationKind.SAMPLE].
 * @return the standard deviation of the array elements.
 */
public fun DoubleArray.standardDeviation(kind: PopulationKind = SAMPLE): Double =
    sqrt(variance(kind))

// ── range ───────────────────────────────────────────────────────────────────

/**
 * Computes the range of the values in this iterable.
 *
 * The range is the difference between the maximum and minimum values. It is the simplest
 * measure of spread but is sensitive to outliers.
 *
 * ### Example:
 * ```kotlin
 * listOf(1.0, 5.0, 3.0).range() // 4.0
 * ```
 *
 * @return the range (max - min) of the elements.
 */
public fun Iterable<Double>.range(): Double {
    val list = toList()
    if (list.isEmpty()) throw InsufficientDataException("Collection must not be empty")
    return list.max() - list.min()
}

/**
 * Computes the range of the values in this array.
 *
 * The range is the difference between the maximum and minimum values.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(1.0, 5.0, 3.0).range() // 4.0
 * ```
 *
 * @return the range (max - min) of the array elements.
 */
public fun DoubleArray.range(): Double {
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")
    return max() - min()
}

// ── interquartileRange ──────────────────────────────────────────────────────

/**
 * Computes the interquartile range (IQR) of the values in this iterable.
 *
 * The IQR is the difference between the third quartile (Q3, 75th percentile) and the first
 * quartile (Q1, 25th percentile). It measures the spread of the middle 50% of the data and
 * is robust to outliers.
 *
 * ### Example:
 * ```kotlin
 * listOf(1.0, 2.0, 3.0, 4.0, 5.0).interquartileRange() // 2.0
 * ```
 *
 * @return the interquartile range (Q3 - Q1).
 */
public fun Iterable<Double>.interquartileRange(): Double {
    val q = quartiles()
    return q.third - q.first
}

/**
 * Computes the interquartile range (IQR) of the values in this array.
 *
 * The IQR is the difference between Q3 (75th percentile) and Q1 (25th percentile).
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0).interquartileRange() // 2.0
 * ```
 *
 * @return the interquartile range (Q3 - Q1).
 */
public fun DoubleArray.interquartileRange(): Double = asIterable().interquartileRange()

// ── meanAbsoluteDeviation ───────────────────────────────────────────────────

/**
 * Computes the mean absolute deviation (MAD) of the values in this iterable.
 *
 * The MAD is the average of the absolute deviations from the mean. It is a robust measure
 * of spread that is less sensitive to outliers than standard deviation.
 *
 * ### Example:
 * ```kotlin
 * listOf(1.0, 2.0, 3.0, 4.0, 5.0).meanAbsoluteDeviation() // 1.2
 * ```
 *
 * @return the mean absolute deviation from the mean.
 */
public fun Iterable<Double>.meanAbsoluteDeviation(): Double {
    val list = toList()
    if (list.isEmpty()) throw InsufficientDataException("Collection must not be empty")
    val m = list.mean()
    return list.map { abs(it - m) }.mean()
}

/**
 * Computes the mean absolute deviation (MAD) of the values in this array.
 *
 * The MAD is the average of the absolute deviations from the mean.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0).meanAbsoluteDeviation() // 1.2
 * ```
 *
 * @return the mean absolute deviation from the mean.
 */
public fun DoubleArray.meanAbsoluteDeviation(): Double = asIterable().meanAbsoluteDeviation()

// ── medianAbsoluteDeviation ─────────────────────────────────────────────────

/**
 * Computes the median absolute deviation (median AD) of the values in this iterable.
 *
 * This is the median of the absolute deviations from the median of the data. It is an
 * extremely robust measure of spread — even more resistant to outliers than the mean
 * absolute deviation — since both the center and spread use the median.
 *
 * ### Example:
 * ```kotlin
 * listOf(1.0, 2.0, 3.0, 4.0, 5.0).medianAbsoluteDeviation() // 1.0
 * ```
 *
 * @return the median absolute deviation from the median.
 */
public fun Iterable<Double>.medianAbsoluteDeviation(): Double {
    val list = toList()
    if (list.isEmpty()) throw InsufficientDataException("Collection must not be empty")
    val med = list.median()
    return list.map { abs(it - med) }.median()
}

/**
 * Computes the median absolute deviation (median AD) of the values in this array.
 *
 * This is the median of the absolute deviations from the median of the data.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0).medianAbsoluteDeviation() // 1.0
 * ```
 *
 * @return the median absolute deviation from the median.
 */
public fun DoubleArray.medianAbsoluteDeviation(): Double = asIterable().medianAbsoluteDeviation()

// ── standardError ───────────────────────────────────────────────────────────

/**
 * Computes the standard error of the mean for the values in this iterable.
 *
 * The standard error estimates how much the sample mean is expected to vary from the true
 * population mean. It equals the sample standard deviation divided by the square root of
 * the sample size. Smaller values indicate a more precise estimate of the population mean.
 *
 * ### Example:
 * ```kotlin
 * listOf(1.0, 2.0, 3.0, 4.0, 5.0).standardError() // 0.7071...
 * ```
 *
 * @return the standard error of the mean.
 */
public fun Iterable<Double>.standardError(): Double {
    val list = toList()
    if (list.size <= 1) throw InsufficientDataException("Standard error requires at least 2 elements")
    return list.standardDeviation() / sqrt(list.size.toDouble())
}

/**
 * Computes the standard error of the mean for the values in this array.
 *
 * The standard error equals the sample standard deviation divided by the square root of
 * the sample size.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0).standardError() // 0.7071...
 * ```
 *
 * @return the standard error of the mean.
 */
public fun DoubleArray.standardError(): Double {
    if (size <= 1) throw InsufficientDataException("Standard error requires at least 2 elements")
    return standardDeviation() / sqrt(size.toDouble())
}

// ── coefficientOfVariation ──────────────────────────────────────────────────

/**
 * Computes the coefficient of variation (CV) of the values in this iterable.
 *
 * The CV is the ratio of the standard deviation to the mean. It expresses variability as
 * a proportion of the mean, which is useful for comparing the spread of datasets with
 * different units or scales. The mean must not be zero.
 *
 * ### Example:
 * ```kotlin
 * listOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0).coefficientOfVariation() // 0.4276...
 * ```
 *
 * @param kind whether to use sample or population standard deviation. Defaults to [PopulationKind.SAMPLE].
 * @return the coefficient of variation (standard deviation / mean).
 * @throws DegenerateDataException if the mean is zero.
 */
public fun Iterable<Double>.coefficientOfVariation(kind: PopulationKind = SAMPLE): Double {
    val m = mean()
    if (m == 0.0) throw DegenerateDataException("Coefficient of variation is undefined when mean is zero")
    return standardDeviation(kind) / m
}

/**
 * Computes the coefficient of variation (CV) of the values in this array.
 *
 * The CV is the ratio of the standard deviation to the mean. The mean must not be zero.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0).coefficientOfVariation() // 0.4276...
 * ```
 *
 * @param kind whether to use sample or population standard deviation. Defaults to [PopulationKind.SAMPLE].
 * @return the coefficient of variation (standard deviation / mean).
 * @throws DegenerateDataException if the mean is zero.
 */
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

// ── semiVariance ────────────────────────────────────────────────────────────

/**
 * Computes the semi-variance of the values on one side of a threshold.
 *
 * Semi-variance measures variability on only one side of a threshold, ignoring values on the
 * other side. It is commonly used in finance to quantify downside risk separately from upside
 * potential. The divisor uses the total number of elements (n-1 for sample, n for population),
 * not just the count on the measured side, matching the Apache Commons Math convention.
 * When the threshold equals the mean, the sum of downside and upside semi-variance equals
 * the full variance. Uses Neumaier compensated summation for numerical stability.
 *
 * ### Example:
 * ```kotlin
 * val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
 * data.semiVariance() // 1.7143 (downside, sample, threshold = mean)
 * data.semiVariance(direction = SemiVarianceDirection.UPSIDE) // 2.8571
 * ```
 *
 * @param threshold the reference point that separates downside from upside. Defaults to
 * the [mean] of the values.
 * @param direction which side of the threshold to measure. Defaults to
 * [SemiVarianceDirection.DOWNSIDE], measuring downside risk.
 * @param kind whether to compute sample or population semi-variance. Defaults to
 * [PopulationKind.SAMPLE], which divides by n-1 (Bessel's correction).
 * @return the semi-variance on the selected side of the threshold.
 * @see variance
 */
public fun DoubleArray.semiVariance(
    threshold: Double = mean(),
    direction: SemiVarianceDirection = SemiVarianceDirection.DOWNSIDE,
    kind: PopulationKind = SAMPLE,
): Double {
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")
    val divisor = if (kind == SAMPLE) {
        if (size <= 1) throw InsufficientDataException("Sample semi-variance requires at least 2 elements")
        size - 1
    } else {
        size
    }
    var sum = 0.0
    var compensation = 0.0
    for (x in this) {
        val diff = x - threshold
        // NaN propagation (IEEE 754 / numpy default): include NaN diffs so the result becomes NaN
        val include = diff.isNaN() || when (direction) {
            SemiVarianceDirection.DOWNSIDE -> diff < 0.0
            SemiVarianceDirection.UPSIDE -> diff > 0.0
        }
        if (include) {
            val sq = diff * diff
            val t = sum + sq
            compensation += if (abs(sum) >= abs(sq)) (sum - t) + sq else (sq - t) + sum
            sum = t
        }
    }
    return (sum + compensation) / divisor
}

/**
 * Computes the semi-variance of the values on one side of a threshold.
 *
 * Semi-variance measures variability on only one side of a threshold, ignoring values on the
 * other side. It is commonly used in finance to quantify downside risk separately from upside
 * potential. The divisor uses the total number of elements (n-1 for sample, n for population),
 * not just the count on the measured side. When the threshold equals the mean, the sum of
 * downside and upside semi-variance equals the full variance.
 *
 * ### Example:
 * ```kotlin
 * listOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0).semiVariance() // 1.7143 (downside, sample)
 * ```
 *
 * @param threshold the reference point that separates downside from upside. Defaults to
 * the [mean] of the values.
 * @param direction which side of the threshold to measure. Defaults to
 * [SemiVarianceDirection.DOWNSIDE], measuring downside risk.
 * @param kind whether to compute sample or population semi-variance. Defaults to
 * [PopulationKind.SAMPLE], which divides by n-1 (Bessel's correction).
 * @return the semi-variance on the selected side of the threshold.
 * @see variance
 */
public fun Iterable<Double>.semiVariance(
    threshold: Double = mean(),
    direction: SemiVarianceDirection = SemiVarianceDirection.DOWNSIDE,
    kind: PopulationKind = SAMPLE,
): Double = toList().toDoubleArray().semiVariance(threshold, direction, kind)

/**
 * Computes the semi-variance of the values on one side of a threshold.
 *
 * Semi-variance measures variability on only one side of a threshold, ignoring values on the
 * other side. It is commonly used in finance to quantify downside risk separately from upside
 * potential. The divisor uses the total number of elements (n-1 for sample, n for population),
 * not just the count on the measured side. When the threshold equals the mean, the sum of
 * downside and upside semi-variance equals the full variance.
 *
 * Since sequences can only be consumed once, the threshold cannot default to the mean directly
 * (unlike [DoubleArray.semiVariance] and [Iterable.semiVariance][Iterable.semiVariance] where
 * the default is `mean()`). Instead, pass `Double.NaN` (the default) to use the mean of the
 * materialized data — the behavior is identical, only the default parameter encoding differs.
 *
 * ### Example:
 * ```kotlin
 * sequenceOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0).semiVariance() // 1.7143 (downside, sample)
 * ```
 *
 * @param threshold the reference point that separates downside from upside. Defaults to
 * `Double.NaN`, which uses the mean of the values.
 * @param direction which side of the threshold to measure. Defaults to
 * [SemiVarianceDirection.DOWNSIDE], measuring downside risk.
 * @param kind whether to compute sample or population semi-variance. Defaults to
 * [PopulationKind.SAMPLE], which divides by n-1 (Bessel's correction).
 * @return the semi-variance on the selected side of the threshold.
 * @see variance
 */
public fun Sequence<Double>.semiVariance(
    threshold: Double = Double.NaN,
    direction: SemiVarianceDirection = SemiVarianceDirection.DOWNSIDE,
    kind: PopulationKind = SAMPLE,
): Double {
    val array = toList().toDoubleArray()
    val t = if (threshold.isNaN()) array.mean() else threshold
    return array.semiVariance(t, direction, kind)
}
