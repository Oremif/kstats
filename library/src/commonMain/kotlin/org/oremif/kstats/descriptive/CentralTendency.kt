package org.oremif.kstats.descriptive

import kotlin.math.ln
import kotlin.math.exp

// ── mean ────────────────────────────────────────────────────────────────────

public fun Iterable<Double>.mean(): Double {
    var sum = 0.0
    var count = 0
    for (element in this) {
        sum += element
        count++
    }
    require(count > 0) { "Collection must not be empty" }
    return sum / count
}

public fun DoubleArray.mean(): Double {
    require(isNotEmpty()) { "Array must not be empty" }
    return sum() / size
}

public fun Sequence<Double>.mean(): Double {
    var sum = 0.0
    var count = 0
    for (element in this) {
        sum += element
        count++
    }
    require(count > 0) { "Sequence must not be empty" }
    return sum / count
}

// ── geometricMean ───────────────────────────────────────────────────────────

public fun Iterable<Double>.geometricMean(): Double {
    var sumLn = 0.0
    var count = 0
    for (element in this) {
        require(element > 0.0) { "All elements must be positive for geometric mean" }
        sumLn += ln(element)
        count++
    }
    require(count > 0) { "Collection must not be empty" }
    return exp(sumLn / count)
}

public fun DoubleArray.geometricMean(): Double {
    require(isNotEmpty()) { "Array must not be empty" }
    var sumLn = 0.0
    for (element in this) {
        require(element > 0.0) { "All elements must be positive for geometric mean" }
        sumLn += ln(element)
    }
    return exp(sumLn / size)
}

// ── harmonicMean ────────────────────────────────────────────────────────────

public fun Iterable<Double>.harmonicMean(): Double {
    var sumReciprocal = 0.0
    var count = 0
    for (element in this) {
        require(element > 0.0) { "All elements must be positive for harmonic mean" }
        sumReciprocal += 1.0 / element
        count++
    }
    require(count > 0) { "Collection must not be empty" }
    return count.toDouble() / sumReciprocal
}

public fun DoubleArray.harmonicMean(): Double {
    require(isNotEmpty()) { "Array must not be empty" }
    var sumReciprocal = 0.0
    for (element in this) {
        require(element > 0.0) { "All elements must be positive for harmonic mean" }
        sumReciprocal += 1.0 / element
    }
    return size.toDouble() / sumReciprocal
}

// ── weightedMean ────────────────────────────────────────────────────────────

public fun Iterable<Double>.weightedMean(weights: Iterable<Double>): Double {
    val valueIter = this.iterator()
    val weightIter = weights.iterator()
    var weightedSum = 0.0
    var totalWeight = 0.0
    var count = 0
    while (valueIter.hasNext() && weightIter.hasNext()) {
        val v = valueIter.next()
        val w = weightIter.next()
        require(w >= 0.0) { "Weights must be non-negative" }
        weightedSum += v * w
        totalWeight += w
        count++
    }
    require(count > 0) { "Collections must not be empty" }
    require(!valueIter.hasNext() && !weightIter.hasNext()) { "Values and weights must have the same size" }
    require(totalWeight > 0.0) { "Total weight must be positive" }
    return weightedSum / totalWeight
}

public fun DoubleArray.weightedMean(weights: DoubleArray): Double {
    require(size == weights.size) { "Values and weights must have the same size" }
    require(isNotEmpty()) { "Arrays must not be empty" }
    var weightedSum = 0.0
    var totalWeight = 0.0
    for (i in indices) {
        require(weights[i] >= 0.0) { "Weights must be non-negative" }
        weightedSum += this[i] * weights[i]
        totalWeight += weights[i]
    }
    require(totalWeight > 0.0) { "Total weight must be positive" }
    return weightedSum / totalWeight
}

// ── median ──────────────────────────────────────────────────────────────────

public fun Iterable<Double>.median(): Double {
    val sorted = this.toList().sorted()
    require(sorted.isNotEmpty()) { "Collection must not be empty" }
    val n = sorted.size
    return if (n % 2 == 1) {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

public fun DoubleArray.median(): Double {
    require(isNotEmpty()) { "Array must not be empty" }
    val sorted = this.sortedArray()
    val n = sorted.size
    return if (n % 2 == 1) {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

// ── mode ────────────────────────────────────────────────────────────────────

public fun <T : Comparable<T>> Iterable<T>.mode(): Set<T> {
    val counts = mutableMapOf<T, Int>()
    for (element in this) {
        counts[element] = (counts[element] ?: 0) + 1
    }
    require(counts.isNotEmpty()) { "Collection must not be empty" }
    val maxCount = counts.values.max()
    return counts.filter { it.value == maxCount }.keys
}
