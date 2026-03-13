package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.descriptive.PopulationKind.SAMPLE
import kotlin.math.sqrt

// ── skewness ────────────────────────────────────────────────────────────────

public fun Iterable<Double>.skewness(kind: PopulationKind = SAMPLE): Double {
    val list = toList()
    val n = list.size
    if (n < 3) throw InsufficientDataException("Skewness requires at least 3 elements")

    // Pass 1 — Welford mean + M2
    var count = 0
    var mean = 0.0
    var m2 = 0.0
    for (x in list) {
        count++
        val delta = x - mean
        mean += delta / count
        val delta2 = x - mean
        m2 += delta * delta2
    }

    val variance = m2 / n
    if (variance == 0.0) return 0.0
    val sd = sqrt(variance)

    // Pass 2 — normalized z³ accumulation (overflow-safe)
    var sumZ3 = 0.0
    for (x in list) {
        val z = (x - mean) / sd
        sumZ3 += z * z * z
    }
    val g1 = sumZ3 / n

    return if (kind == SAMPLE) {
        // Adjusted Fisher-Pearson standardized moment coefficient
        val adj = sqrt(n.toDouble() * (n - 1)) / (n - 2)
        adj * g1
    } else {
        g1
    }
}

public fun DoubleArray.skewness(kind: PopulationKind = SAMPLE): Double = asIterable().skewness(kind)

// ── kurtosis ────────────────────────────────────────────────────────────────

public fun Iterable<Double>.kurtosis(kind: PopulationKind = SAMPLE, excess: Boolean = true): Double {
    val list = toList()
    val n = list.size
    if (n < 4) throw InsufficientDataException("Kurtosis requires at least 4 elements")

    // Pass 1 — Welford mean + M2
    var count = 0
    var mean = 0.0
    var m2 = 0.0
    for (x in list) {
        count++
        val delta = x - mean
        mean += delta / count
        val delta2 = x - mean
        m2 += delta * delta2
    }

    val variance = m2 / n
    if (variance == 0.0) return if (excess) -3.0 else 0.0

    val sd = sqrt(variance)

    // Pass 2 — normalized z⁴ accumulation (overflow-safe)
    var sumZ4 = 0.0
    for (x in list) {
        val z = (x - mean) / sd
        val z2 = z * z
        sumZ4 += z2 * z2
    }
    val g2 = sumZ4 / n

    val result = if (kind == SAMPLE) {
        // Excess kurtosis with sample correction
        val nd = n.toDouble()
        val adj = (nd - 1.0) / ((nd - 2.0) * (nd - 3.0)) * ((nd + 1.0) * g2 - 3.0 * (nd - 1.0))
        if (excess) adj else adj + 3.0
    } else {
        if (excess) g2 - 3.0 else g2
    }

    return result
}

public fun DoubleArray.kurtosis(kind: PopulationKind = SAMPLE, excess: Boolean = true): Double =
    asIterable().kurtosis(kind, excess)
