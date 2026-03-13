package org.oremif.kstats.descriptive

import org.oremif.kstats.descriptive.PopulationKind.SAMPLE
import kotlin.math.sqrt

// ── skewness ────────────────────────────────────────────────────────────────

public fun Iterable<Double>.skewness(kind: PopulationKind = SAMPLE): Double {
    val list = toList()
    val n = list.size
    require(n >= 3) { "Skewness requires at least 3 elements" }

    val m = list.mean()
    var m2 = 0.0
    var m3 = 0.0
    for (x in list) {
        val d = x - m
        m2 += d * d
        m3 += d * d * d
    }

    val variance = m2 / n
    if (variance == 0.0) return 0.0
    val sd = sqrt(variance)

    val g1 = (m3 / n) / (sd * sd * sd)

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
    require(n >= 4) { "Kurtosis requires at least 4 elements" }

    val m = list.mean()
    var m2 = 0.0
    var m4 = 0.0
    for (x in list) {
        val d = x - m
        val d2 = d * d
        m2 += d2
        m4 += d2 * d2
    }

    val variance = m2 / n
    if (variance == 0.0) return if (excess) -3.0 else 0.0

    val g2 = (m4 / n) / (variance * variance)

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
