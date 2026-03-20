package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.descriptive.PopulationKind.SAMPLE
import kotlin.math.sqrt

// ── skewness ────────────────────────────────────────────────────────────────

/**
 * Computes the skewness of the values in this iterable.
 *
 * Skewness measures the asymmetry of a distribution. A positive value indicates a longer
 * right tail, a negative value indicates a longer left tail, and zero indicates symmetry.
 * Uses a two-pass algorithm with z-normalization for numerical stability: first computes
 * the mean and variance via Welford's method, then accumulates normalized cubed deviations.
 *
 * NaN values propagate through the computation (IEEE 754 semantics): if any element is NaN,
 * the result is NaN. Filter NaN values before calling this function if that is not desired.
 *
 * ### Example:
 * ```kotlin
 * listOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0).skewness() // 0.656...
 * ```
 *
 * @param kind whether to compute sample-adjusted (Fisher-Pearson) or population skewness.
 * Defaults to [PopulationKind.SAMPLE], which applies the bias correction factor
 * sqrt(n*(n-1)) / (n-2).
 * @return the skewness, or 0.0 if variance is zero (constant data).
 */
public fun Iterable<Double>.skewness(kind: PopulationKind = SAMPLE): Double {
    val list = toList()
    val n = list.size
    if (n < 3) throw InsufficientDataException("Skewness requires at least 3 elements")

    return list.welford { _, mean, m2 ->
        val variance = m2 / n
        if (variance == 0.0) return 0.0
        val sd = sqrt(variance)

        var sumZ3 = 0.0
        for (x in list) {
            val z = (x - mean) / sd
            sumZ3 += z * z * z
        }
        val g1 = sumZ3 / n

        if (kind == SAMPLE) {
            val adj = sqrt(n.toDouble() * (n - 1)) / (n - 2)
            adj * g1
        } else {
            g1
        }
    }
}

/**
 * Computes the skewness of the values in this array.
 *
 * Skewness measures the asymmetry of a distribution. A positive value indicates a longer
 * right tail, a negative value indicates a longer left tail, and zero indicates symmetry.
 *
 * NaN values propagate through the computation (IEEE 754 semantics): if any element is NaN,
 * the result is NaN. Filter NaN values before calling this function if that is not desired.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0).skewness() // 0.656...
 * ```
 *
 * @param kind whether to compute sample-adjusted (Fisher-Pearson) or population skewness.
 * Defaults to [PopulationKind.SAMPLE], which applies the bias correction factor
 * sqrt(n*(n-1)) / (n-2).
 * @return the skewness, or 0.0 if variance is zero (constant data).
 */
public fun DoubleArray.skewness(kind: PopulationKind = SAMPLE): Double {
    val n = size
    if (n < 3) throw InsufficientDataException("Skewness requires at least 3 elements")

    return welford { mean, m2 ->
        val variance = m2 / n
        if (variance == 0.0) return 0.0
        val sd = sqrt(variance)

        var sumZ3 = 0.0
        for (x in this) {
            val z = (x - mean) / sd
            sumZ3 += z * z * z
        }
        val g1 = sumZ3 / n

        if (kind == SAMPLE) {
            val adj = sqrt(n.toDouble() * (n - 1)) / (n - 2)
            adj * g1
        } else {
            g1
        }
    }
}

// ── kurtosis ────────────────────────────────────────────────────────────────

/**
 * Computes the kurtosis of the values in this iterable.
 *
 * Kurtosis measures the "tailedness" of a distribution relative to a normal distribution.
 * Higher kurtosis indicates heavier tails and a sharper peak. By default, computes excess
 * kurtosis (subtracting 3 so that a normal distribution has kurtosis 0). Uses a two-pass
 * algorithm with z-normalization for numerical stability.
 *
 * NaN values propagate through the computation (IEEE 754 semantics): if any element is NaN,
 * the result is NaN. Filter NaN values before calling this function if that is not desired.
 *
 * ### Example:
 * ```kotlin
 * listOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0).kurtosis() // -0.151...
 * listOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0).kurtosis(excess = false) // 2.848...
 * ```
 *
 * @param kind whether to compute sample-adjusted or population kurtosis. Defaults to
 * [PopulationKind.SAMPLE], which applies bias correction.
 * @param excess whether to subtract 3 (the kurtosis of a normal distribution). Defaults to
 * `true`. Set to `false` for raw kurtosis.
 * @return the kurtosis. Returns -3.0 (excess) or 0.0 (non-excess) if variance is zero.
 */
public fun Iterable<Double>.kurtosis(kind: PopulationKind = SAMPLE, excess: Boolean = true): Double {
    val list = toList()
    val n = list.size
    if (n < 4) throw InsufficientDataException("Kurtosis requires at least 4 elements")

    return list.welford { _, mean, m2 ->
        val variance = m2 / n
        if (variance == 0.0) return if (excess) -3.0 else 0.0
        val sd = sqrt(variance)

        var sumZ4 = 0.0
        for (x in list) {
            val z = (x - mean) / sd
            val z2 = z * z
            sumZ4 += z2 * z2
        }
        val g2 = sumZ4 / n

        if (kind == SAMPLE) {
            val nd = n.toDouble()
            val adj = (nd - 1.0) / ((nd - 2.0) * (nd - 3.0)) * ((nd + 1.0) * g2 - 3.0 * (nd - 1.0))
            if (excess) adj else adj + 3.0
        } else {
            if (excess) g2 - 3.0 else g2
        }
    }
}

/**
 * Computes the kurtosis of the values in this array.
 *
 * Kurtosis measures the "tailedness" of a distribution relative to a normal distribution.
 * Higher kurtosis indicates heavier tails and a sharper peak.
 *
 * NaN values propagate through the computation (IEEE 754 semantics): if any element is NaN,
 * the result is NaN. Filter NaN values before calling this function if that is not desired.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0).kurtosis() // -0.151...
 * ```
 *
 * @param kind whether to compute sample-adjusted or population kurtosis. Defaults to
 * [PopulationKind.SAMPLE], which applies bias correction.
 * @param excess whether to subtract 3 (the kurtosis of a normal distribution). Defaults to
 * `true`. Set to `false` for raw kurtosis.
 * @return the kurtosis. Returns -3.0 (excess) or 0.0 (non-excess) if variance is zero.
 */
public fun DoubleArray.kurtosis(kind: PopulationKind = SAMPLE, excess: Boolean = true): Double {
    val n = size
    if (n < 4) throw InsufficientDataException("Kurtosis requires at least 4 elements")

    return welford { mean, m2 ->
        val variance = m2 / n
        if (variance == 0.0) return if (excess) -3.0 else 0.0
        val sd = sqrt(variance)

        var sumZ4 = 0.0
        for (x in this) {
            val z = (x - mean) / sd
            val z2 = z * z
            sumZ4 += z2 * z2
        }
        val g2 = sumZ4 / n

        if (kind == SAMPLE) {
            val nd = n.toDouble()
            val adj = (nd - 1.0) / ((nd - 2.0) * (nd - 3.0)) * ((nd + 1.0) * g2 - 3.0 * (nd - 1.0))
            if (excess) adj else adj + 3.0
        } else {
            if (excess) g2 - 3.0 else g2
        }
    }
}
