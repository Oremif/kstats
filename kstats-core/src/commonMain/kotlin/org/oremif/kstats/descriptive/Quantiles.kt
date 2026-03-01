package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.core.introSelect
import kotlin.math.ceil
import kotlin.math.floor

/**
 * Controls how values between two data points are interpolated when computing quantiles
 * and percentiles.
 *
 * When the desired quantile falls between two adjacent sorted values, this enum determines
 * how the result is computed from those two neighbors.
 */
public enum class QuantileInterpolation {
    /** Linearly interpolate between the two nearest data points. This is the most common method. */
    LINEAR,

    /** Return the lower of the two nearest data points (floor). */
    LOWER,

    /** Return the higher of the two nearest data points (ceiling). */
    HIGHER,

    /** Return whichever of the two nearest data points is closer to the exact position. Ties (frac = 0.5) round up to the higher value. */
    NEAREST,

    /** Return the average of the two nearest data points. */
    MIDPOINT
}

// ── percentile ──────────────────────────────────────────────────────────────

/**
 * Computes the p-th percentile of the values in this iterable.
 *
 * The percentile indicates the value below which a given percentage of observations fall.
 * For example, the 50th percentile is the median. Delegates to [quantile] after converting
 * the percentile (0-100 scale) to a quantile (0-1 scale).
 *
 * ### Example:
 * ```kotlin
 * listOf(1.0, 2.0, 3.0, 4.0, 5.0).percentile(50.0) // 3.0
 * listOf(1.0, 2.0, 3.0, 4.0, 5.0).percentile(25.0) // 2.0
 * ```
 *
 * @param p the percentile to compute, in [0, 100].
 * @param interpolation how to interpolate between data points. Defaults to [QuantileInterpolation.LINEAR].
 * @return the p-th percentile value.
 */
public fun Iterable<Double>.percentile(
    p: Double,
    interpolation: QuantileInterpolation = QuantileInterpolation.LINEAR
): Double {
    if (p !in 0.0..100.0) throw InvalidParameterException("Percentile must be in [0, 100], got $p")
    return quantile(p / 100.0, interpolation)
}

/**
 * Computes the p-th percentile of the values in this array.
 *
 * The percentile indicates the value below which a given percentage of observations fall.
 * For example, the 50th percentile is the median.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0).percentile(50.0) // 3.0
 * doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0).percentile(25.0) // 2.0
 * ```
 *
 * @param p the percentile to compute, in [0, 100].
 * @param interpolation how to interpolate between data points. Defaults to [QuantileInterpolation.LINEAR].
 * @return the p-th percentile value.
 */
public fun DoubleArray.percentile(
    p: Double,
    interpolation: QuantileInterpolation = QuantileInterpolation.LINEAR
): Double {
    if (p !in 0.0..100.0) throw InvalidParameterException("Percentile must be in [0, 100], got $p")
    return quantile(p / 100.0, interpolation)
}

// ── quantile ────────────────────────────────────────────────────────────────

/**
 * Computes the q-th quantile of the values in this iterable.
 *
 * The quantile at q is the value below which a fraction q of the data falls. For example,
 * quantile(0.5) is the median. The data is sorted internally, and values between data points
 * are computed according to the chosen [interpolation] method.
 *
 * ### Example:
 * ```kotlin
 * listOf(1.0, 2.0, 3.0, 4.0, 5.0).quantile(0.5)  // 3.0
 * listOf(1.0, 2.0, 3.0, 4.0, 5.0).quantile(0.25) // 2.0
 * ```
 *
 * @param q the quantile to compute, in [0, 1].
 * @param interpolation how to interpolate between data points. Defaults to [QuantileInterpolation.LINEAR].
 * @return the q-th quantile value.
 */
public fun Iterable<Double>.quantile(
    q: Double,
    interpolation: QuantileInterpolation = QuantileInterpolation.LINEAR
): Double = toList().toDoubleArray().quantile(q, interpolation)

/**
 * Computes the q-th quantile of the values in this array.
 *
 * The quantile at q is the value below which a fraction q of the data falls. For example,
 * quantile(0.5) is the median. Uses introselect (O(n) expected time) instead of a full sort.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0).quantile(0.5)  // 3.0
 * doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0).quantile(0.25) // 2.0
 * ```
 *
 * @param q the quantile to compute, in [0, 1].
 * @param interpolation how to interpolate between data points. Defaults to [QuantileInterpolation.LINEAR].
 * @return the q-th quantile value.
 */
public fun DoubleArray.quantile(
    q: Double,
    interpolation: QuantileInterpolation = QuantileInterpolation.LINEAR
): Double {
    if (q !in 0.0..1.0) throw InvalidParameterException("Quantile must be in [0, 1], got $q")
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")
    if (size == 1) return this[0]

    val work = copyOf()
    val pos = q * (work.size - 1)
    val lo = floor(pos).toInt()
    val hi = ceil(pos).toInt()

    work.introSelect(lo)
    val loVal = work[lo]
    val hiVal = if (hi == lo) loVal else {
        // After introSelect(lo), elements in [lo+1..n-1] are >= work[lo].
        // Find the minimum of that partition to get the hi element.
        var minRight = work[lo + 1]
        for (i in lo + 2 until work.size) {
            if (work[i].compareTo(minRight) < 0) minRight = work[i]
        }
        minRight
    }

    val frac = pos - lo
    return when (interpolation) {
        QuantileInterpolation.LINEAR -> loVal + frac * (hiVal - loVal)
        QuantileInterpolation.LOWER -> loVal
        QuantileInterpolation.HIGHER -> hiVal
        QuantileInterpolation.NEAREST -> if (frac < 0.5) loVal else hiVal
        QuantileInterpolation.MIDPOINT -> (loVal + hiVal) / 2.0
    }
}

// ── sortedQuantile (internal helper) ─────────────────────────────────────────

/** Compute quantile from an already-sorted array using linear interpolation. */
internal fun sortedQuantile(sorted: DoubleArray, q: Double): Double {
    if (sorted.size == 1) return sorted[0]
    val pos = q * (sorted.size - 1)
    val lo = floor(pos).toInt()
    val hi = ceil(pos).toInt()
    val frac = pos - lo
    return sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/** Compute quantile from an already-sorted list using linear interpolation. */
internal fun sortedQuantile(sorted: List<Double>, q: Double): Double {
    if (sorted.size == 1) return sorted[0]
    val pos = q * (sorted.size - 1)
    val lo = floor(pos).toInt()
    val hi = ceil(pos).toInt()
    val frac = pos - lo
    return sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

// ── quartiles ───────────────────────────────────────────────────────────────

/**
 * Computes the three quartiles (Q1, Q2, Q3) of the values in this iterable.
 *
 * Q1 (25th percentile), Q2 (median, 50th percentile), and Q3 (75th percentile) divide the
 * data into four equal-frequency groups. Uses linear interpolation. The data is sorted once
 * and reused for all three quartile computations.
 *
 * ### Example:
 * ```kotlin
 * val (q1, q2, q3) = listOf(1.0, 2.0, 3.0, 4.0, 5.0).quartiles()
 * // q1 = 2.0, q2 = 3.0, q3 = 4.0
 * ```
 *
 * @return a [Triple] of (Q1, Q2, Q3).
 */
public fun Iterable<Double>.quartiles(): Triple<Double, Double, Double> {
    val sorted = toList().sorted()
    if (sorted.isEmpty()) throw InsufficientDataException("Collection must not be empty")
    return Triple(
        sortedQuantile(sorted, 0.25),
        sortedQuantile(sorted, 0.50),
        sortedQuantile(sorted, 0.75),
    )
}

/**
 * Computes the three quartiles (Q1, Q2, Q3) of the values in this array.
 *
 * Q1 (25th percentile), Q2 (median, 50th percentile), and Q3 (75th percentile) divide the
 * data into four equal-frequency groups. Uses linear interpolation.
 *
 * ### Example:
 * ```kotlin
 * val (q1, q2, q3) = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0).quartiles()
 * // q1 = 2.0, q2 = 3.0, q3 = 4.0
 * ```
 *
 * @return a [Triple] of (Q1, Q2, Q3).
 */
public fun DoubleArray.quartiles(): Triple<Double, Double, Double> {
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")
    val sorted = sortedArray()
    return Triple(
        sortedQuantile(sorted, 0.25),
        sortedQuantile(sorted, 0.50),
        sortedQuantile(sorted, 0.75),
    )
}

// ── Sequence overloads ──────────────────────────────────────────────────────

/**
 * Computes the p-th percentile of the values in this sequence.
 *
 * The sequence is materialized internally. See [DoubleArray.percentile] for details.
 *
 * @param p the percentile to compute, in [0, 100].
 * @param interpolation how to interpolate between data points. Defaults to [QuantileInterpolation.LINEAR].
 * @return the p-th percentile value.
 */
public fun Sequence<Double>.percentile(
    p: Double,
    interpolation: QuantileInterpolation = QuantileInterpolation.LINEAR
): Double = toList().toDoubleArray().percentile(p, interpolation)

/**
 * Computes the q-th quantile of the values in this sequence.
 *
 * The sequence is materialized internally. See [DoubleArray.quantile] for details.
 *
 * @param q the quantile to compute, in [0, 1].
 * @param interpolation how to interpolate between data points. Defaults to [QuantileInterpolation.LINEAR].
 * @return the q-th quantile value.
 */
public fun Sequence<Double>.quantile(
    q: Double,
    interpolation: QuantileInterpolation = QuantileInterpolation.LINEAR
): Double = toList().toDoubleArray().quantile(q, interpolation)

/**
 * Computes the three quartiles (Q1, Q2, Q3) of the values in this sequence.
 *
 * The sequence is materialized internally. See [DoubleArray.quartiles] for details.
 *
 * @return a [Triple] of (Q1, Q2, Q3).
 */
public fun Sequence<Double>.quartiles(): Triple<Double, Double, Double> =
    toList().toDoubleArray().quartiles()
