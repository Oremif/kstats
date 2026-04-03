package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.core.introSelect
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.round

/**
 * Controls how values between two data points are interpolated when computing quantiles
 * and percentiles.
 *
 * @deprecated Use [QuantileMethod] instead, which provides the nine standard Hyndman and Fan
 * estimation methods in addition to the interpolation modes available here.
 */
@Suppress("DEPRECATION")
@Deprecated(
    "Use QuantileMethod instead.",
    ReplaceWith("QuantileMethod", "org.oremif.kstats.descriptive.QuantileMethod"),
    DeprecationLevel.WARNING,
)
public enum class QuantileInterpolation {
    LINEAR,
    LOWER,
    HIGHER,
    NEAREST,
    MIDPOINT;

    public fun toQuantileMethod(): QuantileMethod = when (this) {
        LINEAR -> QuantileMethod.LINEAR
        LOWER -> QuantileMethod.LOWER
        HIGHER -> QuantileMethod.HIGHER
        NEAREST -> QuantileMethod.NEAREST
        MIDPOINT -> QuantileMethod.MIDPOINT
    }
}

// ── percentile ──────────────────────────────────────────────────────────────

/**
 * Computes the p-th percentile of the values in this iterable.
 *
 * The percentile indicates the value below which a given percentage of observations fall.
 * For example, the 50th percentile is the median. Delegates to [quantile] after converting
 * the percentile (0–100 scale) to a quantile (0–1 scale).
 *
 * ### Example:
 * ```kotlin
 * listOf(1.0, 2.0, 3.0, 4.0, 5.0).percentile(50.0) // 3.0
 * listOf(1.0, 2.0, 3.0, 4.0, 5.0).percentile(25.0, QuantileMethod.WEIBULL) // 1.5
 * ```
 *
 * @param p the percentile to compute, in [0, 100].
 * @param method the quantile estimation method. Defaults to [QuantileMethod.LINEAR] (HF7).
 * @return the p-th percentile value.
 */
public fun Iterable<Double>.percentile(
    p: Double,
    method: QuantileMethod = QuantileMethod.LINEAR,
): Double {
    if (p !in 0.0..100.0) throw InvalidParameterException("Percentile must be in [0, 100], got $p")
    return quantile(p / 100.0, method)
}

/**
 * Computes the p-th percentile of the values in this array.
 *
 * The percentile indicates the value below which a given percentage of observations fall.
 * For example, the 50th percentile is the median. Uses introselect (expected O(n) time)
 * instead of a full sort.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0).percentile(50.0) // 3.0
 * doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0).percentile(25.0, QuantileMethod.WEIBULL) // 1.5
 * ```
 *
 * @param p the percentile to compute, in [0, 100].
 * @param method the quantile estimation method. Defaults to [QuantileMethod.LINEAR] (HF7).
 * @return the p-th percentile value.
 */
public fun DoubleArray.percentile(
    p: Double,
    method: QuantileMethod = QuantileMethod.LINEAR,
): Double {
    if (p !in 0.0..100.0) throw InvalidParameterException("Percentile must be in [0, 100], got $p")
    return quantile(p / 100.0, method)
}

@Suppress("DEPRECATION")
@Deprecated(
    "Use the overload with QuantileMethod instead.",
    ReplaceWith("percentile(p, interpolation.toQuantileMethod())"),
    DeprecationLevel.WARNING,
)
public fun Iterable<Double>.percentile(
    p: Double,
    interpolation: QuantileInterpolation,
): Double = percentile(p, interpolation.toQuantileMethod())

@Suppress("DEPRECATION")
@Deprecated(
    "Use the overload with QuantileMethod instead.",
    ReplaceWith("percentile(p, interpolation.toQuantileMethod())"),
)
public fun DoubleArray.percentile(
    p: Double,
    interpolation: QuantileInterpolation,
): Double = percentile(p, interpolation.toQuantileMethod())

// ── quantile ────────────────────────────────────────────────────────────────

/**
 * Computes the q-th quantile of the values in this iterable.
 *
 * The quantile at q is the value below which a fraction q of the data falls. For example,
 * quantile(0.5) is the median. The estimation method determines how positions between data
 * points are handled — see [QuantileMethod] for the nine standard Hyndman and Fan methods.
 *
 * ### Example:
 * ```kotlin
 * listOf(1.0, 2.0, 3.0, 4.0, 5.0).quantile(0.5)  // 3.0
 * listOf(1.0, 2.0, 3.0, 4.0, 5.0).quantile(0.25, QuantileMethod.MEDIAN_UNBIASED) // 1.75
 * ```
 *
 * @param q the quantile to compute, in [0, 1].
 * @param method the quantile estimation method. Defaults to [QuantileMethod.LINEAR] (HF7).
 * @return the q-th quantile value.
 */
public fun Iterable<Double>.quantile(
    q: Double,
    method: QuantileMethod = QuantileMethod.LINEAR,
): Double = toList().toDoubleArray().quantile(q, method)

/**
 * Computes the q-th quantile of the values in this array.
 *
 * The quantile at q is the value below which a fraction q of the data falls. For example,
 * quantile(0.5) is the median. Uses introselect (expected O(n) time) instead of a full sort.
 * The estimation method determines how positions between data points are handled — see
 * [QuantileMethod] for the nine standard Hyndman and Fan methods.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0).quantile(0.5)  // 3.0
 * doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0).quantile(0.25, QuantileMethod.MEDIAN_UNBIASED) // 1.75
 * ```
 *
 * @param q the quantile to compute, in [0, 1].
 * @param method the quantile estimation method. Defaults to [QuantileMethod.LINEAR] (HF7).
 * @return the q-th quantile value.
 */
public fun DoubleArray.quantile(
    q: Double,
    method: QuantileMethod = QuantileMethod.LINEAR,
): Double {
    if (q !in 0.0..1.0) throw InvalidParameterException("Quantile must be in [0, 1], got $q")
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")
    if (size == 1) return this[0]

    val work = copyOf()
    return computeQuantile(work, q, method, useIntroSelect = true)
}

@Suppress("DEPRECATION")
@Deprecated(
    "Use the overload with QuantileMethod instead.",
    ReplaceWith("quantile(q, interpolation.toQuantileMethod())"),
)
public fun Iterable<Double>.quantile(
    q: Double,
    interpolation: QuantileInterpolation,
): Double = quantile(q, interpolation.toQuantileMethod())

@Suppress("DEPRECATION")
@Deprecated(
    "Use the overload with QuantileMethod instead.",
    ReplaceWith("quantile(q, interpolation.toQuantileMethod())"),
)
public fun DoubleArray.quantile(
    q: Double,
    interpolation: QuantileInterpolation,
): Double = quantile(q, interpolation.toQuantileMethod())

// ── core algorithm ──────────────────────────────────────────────────────────

// Hyndman & Fan (1996) position formulas and interpolation.
// All position formulas use 1-based indexing: h ∈ [1, n].
// For linear methods (HF4-9): result = (1 - g) * x_(j) + g * x_(j+1), j = floor(h), g = h - j.
// For discontinuous methods (HF1-3): result = x_(selected index).
// Reference: Hyndman, R.J. and Fan, Y. (1996) Sample Quantiles in Statistical Packages.
// The American Statistician, 50, 361-365.

/** Computes the 1-based position h(q, n) for the given Hyndman-Fan method. */
private fun computePosition(n: Int, q: Double, method: QuantileMethod): Double = when (method) {
    QuantileMethod.INVERTED_CDF,
    QuantileMethod.AVERAGED_INVERTED_CDF,
    QuantileMethod.HAZEN -> n * q + 0.5

    QuantileMethod.CLOSEST_OBSERVATION,
    QuantileMethod.INTERPOLATED_INVERTED_CDF -> n * q

    QuantileMethod.WEIBULL -> (n + 1) * q

    QuantileMethod.LINEAR,
    QuantileMethod.LOWER,
    QuantileMethod.HIGHER,
    QuantileMethod.NEAREST,
    QuantileMethod.MIDPOINT -> (n - 1) * q + 1.0

    QuantileMethod.MEDIAN_UNBIASED -> (n + 1.0 / 3.0) * q + 1.0 / 3.0
    QuantileMethod.NORMAL_UNBIASED -> (n + 0.25) * q + 3.0 / 8.0
}

/** Computes a 0-based discrete index for non-interpolating methods from the 1-based position [h]. */
private fun discreteIndex(h: Double, n: Int, method: QuantileMethod): Int = when (method) {
    // x_(ceil(h - 0.5)), clamped to [1, n]
    QuantileMethod.INVERTED_CDF,
    QuantileMethod.AVERAGED_INVERTED_CDF -> ceil(h - 0.5).toInt().coerceIn(1, n) - 1
    // x_(round(h)), banker's rounding (ties to even), clamped to [1, n]
    QuantileMethod.CLOSEST_OBSERVATION -> bankersRound(h.coerceIn(1.0, n.toDouble())).coerceIn(1, n) - 1
    QuantileMethod.LOWER -> floor(h).toInt().coerceIn(1, n) - 1
    QuantileMethod.HIGHER -> ceil(h).toInt().coerceIn(1, n) - 1
    // Round half up (not banker's rounding)
    QuantileMethod.NEAREST -> floor(h + 0.5).toInt().coerceIn(1, n) - 1
    else -> error("$method is not a discrete selector")
}

private fun computeQuantile(
    work: DoubleArray,
    q: Double,
    method: QuantileMethod,
    useIntroSelect: Boolean,
): Double {
    val n = work.size

    // All methods agree: q=0 -> min, q=1 -> max.
    // Exception: AVERAGED_INVERTED_CDF averages at discontinuities, including q=0.
    if (q == 0.0 && method != QuantileMethod.AVERAGED_INVERTED_CDF) {
        return selectElement(work, 0, useIntroSelect)
    }
    if (q == 1.0) return selectElement(work, n - 1, useIntroSelect)

    val h = computePosition(n, q, method)

    return when (method) {
        QuantileMethod.INVERTED_CDF,
        QuantileMethod.CLOSEST_OBSERVATION,
        QuantileMethod.LOWER,
        QuantileMethod.HIGHER,
        QuantileMethod.NEAREST -> {
            selectElement(work, discreteIndex(h, n, method), useIntroSelect)
        }

        QuantileMethod.AVERAGED_INVERTED_CDF -> {
            // Like HF1, but average at discontinuities (when h - 0.5 is integer)
            val idx = discreteIndex(h, n, method)
            val hMinusHalf = h - 0.5
            val isDiscontinuity = hMinusHalf == floor(hMinusHalf) && idx + 1 < n
            if (isDiscontinuity) {
                val lo = selectElement(work, idx, useIntroSelect)
                val hi = findNextElement(work, idx, useIntroSelect)
                (lo + hi) / 2.0
            } else {
                selectElement(work, idx, useIntroSelect)
            }
        }

        QuantileMethod.INTERPOLATED_INVERTED_CDF,
        QuantileMethod.HAZEN,
        QuantileMethod.WEIBULL,
        QuantileMethod.LINEAR,
        QuantileMethod.MEDIAN_UNBIASED,
        QuantileMethod.NORMAL_UNBIASED -> {
            linearInterpolate(work, h, n, useIntroSelect)
        }

        QuantileMethod.MIDPOINT -> {
            val lo = floor(h).toInt().coerceIn(1, n) - 1
            val hi = ceil(h).toInt().coerceIn(1, n) - 1
            if (lo == hi) {
                selectElement(work, lo, useIntroSelect)
            } else {
                val loVal = selectElement(work, lo, useIntroSelect)
                val hiVal = findNextElement(work, lo, useIntroSelect)
                (loVal + hiVal) / 2.0
            }
        }
    }
}

// Linear interpolation: result = (1-g)*x_(j) + g*x_(j+1), h 1-based, j = floor(h)
private fun linearInterpolate(
    work: DoubleArray,
    h: Double,
    n: Int,
    useIntroSelect: Boolean,
): Double {
    // Clamp h to [1, n]
    val hClamped = h.coerceIn(1.0, n.toDouble())
    val j = floor(hClamped).toInt() // 1-based lower index
    val g = hClamped - j

    val idx = (j - 1).coerceIn(0, n - 1) // 0-based
    if (g == 0.0 || idx == n - 1) {
        return selectElement(work, idx, useIntroSelect)
    }

    val loVal = selectElement(work, idx, useIntroSelect)
    val hiVal = findNextElement(work, idx, useIntroSelect)
    return loVal + g * (hiVal - loVal)
}

private fun selectElement(work: DoubleArray, idx: Int, useIntroSelect: Boolean): Double {
    if (useIntroSelect) work.introSelect(idx)
    return work[idx]
}

// After introSelect(idx), elements in [idx+1..n-1] are >= work[idx].
// Find the minimum of that right partition (i.e., the next order statistic).
private fun findNextElement(work: DoubleArray, idx: Int, useIntroSelect: Boolean): Double {
    if (!useIntroSelect) return work[idx + 1] // already sorted
    var minRight = work[idx + 1]
    for (i in idx + 2 until work.size) {
        if (work[i].compareTo(minRight) < 0) minRight = work[i]
    }
    return minRight
}

// Banker's rounding (round half to even), matching IEEE 754 / R's round().
// kotlin.math.round() uses ties-to-even; Double.roundToInt() uses ties-towards-positive-infinity.
private fun bankersRound(x: Double): Int = round(x).toInt()

// ── sortedQuantile (internal helper) ─────────────────────────────────────────

/** Computes a quantile from an already-sorted array without copying. */
internal fun sortedQuantile(sorted: DoubleArray, q: Double, method: QuantileMethod = QuantileMethod.LINEAR): Double {
    if (sorted.size == 1) return sorted[0]
    return computeQuantile(sorted, q, method, useIntroSelect = false)
}

/** Computes a quantile from an already-sorted list. */
internal fun sortedQuantile(sorted: List<Double>, q: Double, method: QuantileMethod = QuantileMethod.LINEAR): Double {
    if (sorted.size == 1) return sorted[0]
    val arr = sorted.toDoubleArray()
    return computeQuantile(arr, q, method, useIntroSelect = false)
}

// ── quartiles ───────────────────────────────────────────────────────────────

/**
 * Computes the three quartiles (Q1, Q2, Q3) of the values in this iterable.
 *
 * Q1 (25th percentile), Q2 (median, 50th percentile), and Q3 (75th percentile) divide the
 * data into four equal-frequency groups. The data is sorted once and reused for all three
 * quartile computations.
 *
 * ### Example:
 * ```kotlin
 * val (q1, q2, q3) = listOf(1.0, 2.0, 3.0, 4.0, 5.0).quartiles()
 * // q1 = 2.0, q2 = 3.0, q3 = 4.0
 * ```
 *
 * @param method the quantile estimation method. Defaults to [QuantileMethod.LINEAR] (HF7).
 * @return a [Triple] of (Q1, Q2, Q3).
 */
public fun Iterable<Double>.quartiles(method: QuantileMethod = QuantileMethod.LINEAR): Triple<Double, Double, Double> {
    val sorted = toList().sorted()
    if (sorted.isEmpty()) throw InsufficientDataException("Collection must not be empty")
    return Triple(
        sortedQuantile(sorted, 0.25, method),
        sortedQuantile(sorted, 0.50, method),
        sortedQuantile(sorted, 0.75, method),
    )
}

/**
 * Computes the three quartiles (Q1, Q2, Q3) of the values in this array.
 *
 * Q1 (25th percentile), Q2 (median, 50th percentile), and Q3 (75th percentile) divide the
 * data into four equal-frequency groups.
 *
 * ### Example:
 * ```kotlin
 * val (q1, q2, q3) = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0).quartiles()
 * // q1 = 2.0, q2 = 3.0, q3 = 4.0
 * ```
 *
 * @param method the quantile estimation method. Defaults to [QuantileMethod.LINEAR] (HF7).
 * @return a [Triple] of (Q1, Q2, Q3).
 */
public fun DoubleArray.quartiles(method: QuantileMethod = QuantileMethod.LINEAR): Triple<Double, Double, Double> {
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")
    val sorted = sortedArray()
    return Triple(
        sortedQuantile(sorted, 0.25, method),
        sortedQuantile(sorted, 0.50, method),
        sortedQuantile(sorted, 0.75, method),
    )
}

// ── quantileSelect / percentileSelect ───────────────────────────────────────

/**
 * Selects the element at the q-th quantile position from this sorted list.
 *
 * Unlike [quantile], this function works with any [Comparable] type (not just Double) and
 * always returns an actual element from the list rather than an interpolated value. This is
 * useful when you need, for example, the 20th percentile from an ordered list of strings.
 *
 * Only non-interpolating methods are supported: [QuantileMethod.INVERTED_CDF],
 * [QuantileMethod.CLOSEST_OBSERVATION], [QuantileMethod.LOWER], [QuantileMethod.HIGHER],
 * and [QuantileMethod.NEAREST].
 *
 * ### Example:
 * ```kotlin
 * listOf("a", "b", "c", "d", "e").quantileSelect(0.5) // "c"
 * listOf(10, 20, 30, 40, 50).quantileSelect(0.25, QuantileMethod.LOWER) // 20
 * ```
 *
 * Note: the default method is [QuantileMethod.NEAREST], not [QuantileMethod.LINEAR] (which is
 * the default for [quantile]), because `quantileSelect` only supports non-interpolating methods.
 *
 * @param q the quantile to compute, in [0, 1].
 * @param method the non-interpolating quantile method. Defaults to [QuantileMethod.NEAREST].
 * @return the element at the q-th quantile position.
 */
public fun <T : Comparable<T>> List<T>.quantileSelect(
    q: Double,
    method: QuantileMethod = QuantileMethod.NEAREST,
): T {
    if (q !in 0.0..1.0) throw InvalidParameterException("Quantile must be in [0, 1], got $q")
    if (isEmpty()) throw InsufficientDataException("List must not be empty")
    requireNonInterpolating(method)
    if (size == 1) return this[0]

    val sorted = sorted()
    val n = sorted.size

    if (q == 0.0) return sorted.first()
    if (q == 1.0) return sorted.last()

    return sorted[discreteIndex(computePosition(n, q, method), n, method)]
}

/**
 * Selects the element at the p-th percentile position from this sorted list.
 *
 * This is a convenience wrapper around [quantileSelect] that accepts a percentile (0–100)
 * instead of a quantile (0–1). See [quantileSelect] for details on supported methods and
 * behavior.
 *
 * ### Example:
 * ```kotlin
 * listOf("a", "b", "c", "d", "e").percentileSelect(50.0) // "c"
 * ```
 *
 * @param p the percentile to compute, in [0, 100].
 * @param method the non-interpolating quantile method. Defaults to [QuantileMethod.NEAREST].
 * @return the element at the p-th percentile position.
 */
public fun <T : Comparable<T>> List<T>.percentileSelect(
    p: Double,
    method: QuantileMethod = QuantileMethod.NEAREST,
): T {
    if (p !in 0.0..100.0) throw InvalidParameterException("Percentile must be in [0, 100], got $p")
    return quantileSelect(p / 100.0, method)
}

private val NON_INTERPOLATING_METHODS: Set<QuantileMethod> = setOf(
    QuantileMethod.INVERTED_CDF,
    QuantileMethod.CLOSEST_OBSERVATION,
    QuantileMethod.LOWER,
    QuantileMethod.HIGHER,
    QuantileMethod.NEAREST,
)

private fun requireNonInterpolating(method: QuantileMethod) {
    if (method !in NON_INTERPOLATING_METHODS) {
        throw InvalidParameterException(
            "quantileSelect requires a non-interpolating method " +
                "(INVERTED_CDF, CLOSEST_OBSERVATION, LOWER, HIGHER, NEAREST), got $method"
        )
    }
}

// ── Sequence overloads ──────────────────────────────────────────────────────

/**
 * Computes the p-th percentile of the values in this sequence.
 *
 * The sequence is materialized internally. See [DoubleArray.percentile] for details.
 *
 * @param p the percentile to compute, in [0, 100].
 * @param method the quantile estimation method. Defaults to [QuantileMethod.LINEAR] (HF7).
 * @return the p-th percentile value.
 */
public fun Sequence<Double>.percentile(
    p: Double,
    method: QuantileMethod = QuantileMethod.LINEAR,
): Double = toList().toDoubleArray().percentile(p, method)

/**
 * Computes the q-th quantile of the values in this sequence.
 *
 * The sequence is materialized internally. See [DoubleArray.quantile] for details.
 *
 * @param q the quantile to compute, in [0, 1].
 * @param method the quantile estimation method. Defaults to [QuantileMethod.LINEAR] (HF7).
 * @return the q-th quantile value.
 */
public fun Sequence<Double>.quantile(
    q: Double,
    method: QuantileMethod = QuantileMethod.LINEAR,
): Double = toList().toDoubleArray().quantile(q, method)

/**
 * Computes the three quartiles (Q1, Q2, Q3) of the values in this sequence.
 *
 * The sequence is materialized internally. See [DoubleArray.quartiles] for details.
 *
 * @param method the quantile estimation method. Defaults to [QuantileMethod.LINEAR] (HF7).
 * @return a [Triple] of (Q1, Q2, Q3).
 */
public fun Sequence<Double>.quartiles(method: QuantileMethod = QuantileMethod.LINEAR): Triple<Double, Double, Double> =
    toList().toDoubleArray().quartiles(method)

@Suppress("DEPRECATION")
@Deprecated(
    "Use the overload with QuantileMethod instead.",
    ReplaceWith("percentile(p, interpolation.toQuantileMethod())"),
)
public fun Sequence<Double>.percentile(
    p: Double,
    interpolation: QuantileInterpolation,
): Double = percentile(p, interpolation.toQuantileMethod())

@Suppress("DEPRECATION")
@Deprecated(
    "Use the overload with QuantileMethod instead.",
    ReplaceWith("quantile(q, interpolation.toQuantileMethod())"),
)
public fun Sequence<Double>.quantile(
    q: Double,
    interpolation: QuantileInterpolation,
): Double = quantile(q, interpolation.toQuantileMethod())
