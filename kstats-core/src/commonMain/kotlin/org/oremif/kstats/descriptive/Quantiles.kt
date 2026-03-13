package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.roundToInt

public enum class QuantileInterpolation {
    LINEAR, LOWER, HIGHER, NEAREST, MIDPOINT
}

// ── percentile ──────────────────────────────────────────────────────────────

public fun Iterable<Double>.percentile(
    p: Double,
    interpolation: QuantileInterpolation = QuantileInterpolation.LINEAR
): Double {
    if (p !in 0.0..100.0) throw InvalidParameterException("Percentile must be in [0, 100], got $p")
    return quantile(p / 100.0, interpolation)
}

public fun DoubleArray.percentile(
    p: Double,
    interpolation: QuantileInterpolation = QuantileInterpolation.LINEAR
): Double = asIterable().percentile(p, interpolation)

// ── quantile ────────────────────────────────────────────────────────────────

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

public fun DoubleArray.quantile(
    q: Double,
    interpolation: QuantileInterpolation = QuantileInterpolation.LINEAR
): Double = asIterable().quantile(q, interpolation)

// ── quartiles ───────────────────────────────────────────────────────────────

public fun Iterable<Double>.quartiles(): Triple<Double, Double, Double> = Triple(
    quantile(0.25),
    quantile(0.50),
    quantile(0.75)
)

public fun DoubleArray.quartiles(): Triple<Double, Double, Double> = asIterable().quartiles()
