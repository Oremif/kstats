package org.oremif.kstats.descriptive

import org.oremif.kstats.core.compensatedSum
import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.exp

// ── mean ────────────────────────────────────────────────────────────────────

public fun Iterable<Double>.mean(): Double {
    var sum = 0.0
    var compensation = 0.0
    var count = 0
    for (element in this) {
        val t = sum + element
        compensation += if (abs(sum) >= abs(element)) (sum - t) + element else (element - t) + sum
        sum = t
        count++
    }
    if (count == 0) throw InsufficientDataException("Collection must not be empty")
    return (sum + compensation) / count
}

public fun DoubleArray.mean(): Double {
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")
    return compensatedSum() / size
}

public fun Sequence<Double>.mean(): Double {
    var sum = 0.0
    var compensation = 0.0
    var count = 0
    for (element in this) {
        val t = sum + element
        compensation += if (abs(sum) >= abs(element)) (sum - t) + element else (element - t) + sum
        sum = t
        count++
    }
    if (count == 0) throw InsufficientDataException("Sequence must not be empty")
    return (sum + compensation) / count
}

// ── geometricMean ───────────────────────────────────────────────────────────

public fun Iterable<Double>.geometricMean(): Double {
    var sumLn = 0.0
    var count = 0
    for (element in this) {
        if (element <= 0.0) throw InvalidParameterException("All elements must be positive for geometric mean")
        sumLn += ln(element)
        count++
    }
    if (count == 0) throw InsufficientDataException("Collection must not be empty")
    return exp(sumLn / count)
}

public fun DoubleArray.geometricMean(): Double {
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")
    var sumLn = 0.0
    for (element in this) {
        if (element <= 0.0) throw InvalidParameterException("All elements must be positive for geometric mean")
        sumLn += ln(element)
    }
    return exp(sumLn / size)
}

// ── harmonicMean ────────────────────────────────────────────────────────────

public fun Iterable<Double>.harmonicMean(): Double {
    var sumReciprocal = 0.0
    var compensation = 0.0
    var count = 0
    for (element in this) {
        if (element <= 0.0) throw InvalidParameterException("All elements must be positive for harmonic mean")
        val reciprocal = 1.0 / element
        val t = sumReciprocal + reciprocal
        compensation += if (abs(sumReciprocal) >= abs(reciprocal)) (sumReciprocal - t) + reciprocal else (reciprocal - t) + sumReciprocal
        sumReciprocal = t
        count++
    }
    if (count == 0) throw InsufficientDataException("Collection must not be empty")
    return count.toDouble() / (sumReciprocal + compensation)
}

public fun DoubleArray.harmonicMean(): Double {
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")
    var sumReciprocal = 0.0
    var compensation = 0.0
    for (element in this) {
        if (element <= 0.0) throw InvalidParameterException("All elements must be positive for harmonic mean")
        val reciprocal = 1.0 / element
        val t = sumReciprocal + reciprocal
        compensation += if (abs(sumReciprocal) >= abs(reciprocal)) (sumReciprocal - t) + reciprocal else (reciprocal - t) + sumReciprocal
        sumReciprocal = t
    }
    return size.toDouble() / (sumReciprocal + compensation)
}

// ── weightedMean ────────────────────────────────────────────────────────────

public fun Iterable<Double>.weightedMean(weights: Iterable<Double>): Double {
    val valueIter = this.iterator()
    val weightIter = weights.iterator()
    var weightedSum = 0.0
    var wsCompensation = 0.0
    var totalWeight = 0.0
    var twCompensation = 0.0
    var count = 0
    while (valueIter.hasNext() && weightIter.hasNext()) {
        val v = valueIter.next()
        val w = weightIter.next()
        if (w < 0.0) throw InvalidParameterException("Weights must be non-negative")
        val vw = v * w
        val t1 = weightedSum + vw
        wsCompensation += if (abs(weightedSum) >= abs(vw)) (weightedSum - t1) + vw else (vw - t1) + weightedSum
        weightedSum = t1
        val t2 = totalWeight + w
        twCompensation += if (abs(totalWeight) >= abs(w)) (totalWeight - t2) + w else (w - t2) + totalWeight
        totalWeight = t2
        count++
    }
    if (count == 0) throw InsufficientDataException("Collections must not be empty")
    if (valueIter.hasNext() || weightIter.hasNext()) throw InvalidParameterException("Values and weights must have the same size")
    val finalWeight = totalWeight + twCompensation
    if (finalWeight <= 0.0) throw InvalidParameterException("Total weight must be positive")
    return (weightedSum + wsCompensation) / finalWeight
}

public fun DoubleArray.weightedMean(weights: DoubleArray): Double {
    if (size != weights.size) throw InvalidParameterException("Values and weights must have the same size")
    if (isEmpty()) throw InsufficientDataException("Arrays must not be empty")
    var weightedSum = 0.0
    var wsCompensation = 0.0
    var totalWeight = 0.0
    var twCompensation = 0.0
    for (i in indices) {
        if (weights[i] < 0.0) throw InvalidParameterException("Weights must be non-negative")
        val vw = this[i] * weights[i]
        val t1 = weightedSum + vw
        wsCompensation += if (abs(weightedSum) >= abs(vw)) (weightedSum - t1) + vw else (vw - t1) + weightedSum
        weightedSum = t1
        val w = weights[i]
        val t2 = totalWeight + w
        twCompensation += if (abs(totalWeight) >= abs(w)) (totalWeight - t2) + w else (w - t2) + totalWeight
        totalWeight = t2
    }
    val finalWeight = totalWeight + twCompensation
    if (finalWeight <= 0.0) throw InvalidParameterException("Total weight must be positive")
    return (weightedSum + wsCompensation) / finalWeight
}

// ── median ──────────────────────────────────────────────────────────────────

public fun Iterable<Double>.median(): Double {
    val sorted = this.toList().sorted()
    if (sorted.isEmpty()) throw InsufficientDataException("Collection must not be empty")
    val n = sorted.size
    return if (n % 2 == 1) {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

public fun DoubleArray.median(): Double {
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")
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
    if (counts.isEmpty()) throw InsufficientDataException("Collection must not be empty")
    val maxCount = counts.values.max()
    return counts.filter { it.value == maxCount }.keys
}
