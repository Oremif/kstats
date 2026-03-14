package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.roundToInt

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

    /** Return whichever of the two nearest data points is closer to the exact position. */
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
): Double = asIterable().percentile(p, interpolation)

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
): Double {
    if (q !in 0.0..1.0) throw InvalidParameterException("Quantile must be in [0, 1], got $q")
    val sorted = toList().sorted()
    if (sorted.isEmpty()) throw InsufficientDataException("Collection must not be empty")

    if (sorted.size == 1) return sorted[0]

    val pos = q * (sorted.size - 1)
    val lo = floor(pos).toInt()
    val hi = ceil(pos).toInt()
    val frac = pos - lo

    return when (interpolation) {
        QuantileInterpolation.LINEAR -> sorted[lo] + frac * (sorted[hi] - sorted[lo])
        QuantileInterpolation.LOWER -> sorted[lo]
        QuantileInterpolation.HIGHER -> sorted[hi]
        QuantileInterpolation.NEAREST -> sorted[pos.roundToInt()]
        QuantileInterpolation.MIDPOINT -> (sorted[lo] + sorted[hi]) / 2.0
    }
}

/**
 * Computes the q-th quantile of the values in this array.
 *
 * The quantile at q is the value below which a fraction q of the data falls. For example,
 * quantile(0.5) is the median.
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
): Double = asIterable().quantile(q, interpolation)

// ── quartiles ───────────────────────────────────────────────────────────────

/**
 * Computes the three quartiles (Q1, Q2, Q3) of the values in this iterable.
 *
 * Q1 (25th percentile), Q2 (median, 50th percentile), and Q3 (75th percentile) divide the
 * data into four equal-frequency groups. Uses linear interpolation.
 *
 * ### Example:
 * ```kotlin
 * val (q1, q2, q3) = listOf(1.0, 2.0, 3.0, 4.0, 5.0).quartiles()
 * // q1 = 2.0, q2 = 3.0, q3 = 4.0
 * ```
 *
 * @return a [Triple] of (Q1, Q2, Q3).
 */
public fun Iterable<Double>.quartiles(): Triple<Double, Double, Double> = Triple(
    quantile(0.25),
    quantile(0.50),
    quantile(0.75)
)

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
public fun DoubleArray.quartiles(): Triple<Double, Double, Double> = asIterable().quartiles()
