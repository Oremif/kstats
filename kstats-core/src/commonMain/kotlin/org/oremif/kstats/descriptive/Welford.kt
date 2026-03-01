package org.oremif.kstats.descriptive

/**
 * Performs a single Welford pass over this array, computing the running mean
 * and the sum of squared deviations (M2), then passes them to [block].
 */
internal inline fun <R> DoubleArray.welford(block: (mean: Double, m2: Double) -> R): R {
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
    return block(mean, m2)
}

/**
 * Performs a single Welford pass over this iterable, computing the element count,
 * running mean, and the sum of squared deviations (M2), then passes them to [block].
 */
internal inline fun <R> Iterable<Double>.welford(block: (count: Int, mean: Double, m2: Double) -> R): R {
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
    return block(count, mean, m2)
}
