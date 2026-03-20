package org.oremif.kstats.descriptive

import org.oremif.kstats.core.compensatedSum
import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.core.introSelect
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.ln

// ── mean ────────────────────────────────────────────────────────────────────

/**
 * Computes the arithmetic mean of the values in this iterable.
 *
 * The arithmetic mean is the sum of all values divided by the count. Uses compensated
 * (Neumaier) summation for improved numerical precision.
 *
 * ### Example:
 * ```kotlin
 * listOf(1.0, 2.0, 3.0).mean() // 2.0
 * ```
 *
 * @return the arithmetic mean of the elements.
 */
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

/**
 * Computes the arithmetic mean of the values in this array.
 *
 * The arithmetic mean is the sum of all values divided by the count. Uses compensated
 * (Neumaier) summation for improved numerical precision with large arrays.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(1.0, 2.0, 3.0).mean() // 2.0
 * ```
 *
 * @return the arithmetic mean of the array elements.
 */
public fun DoubleArray.mean(): Double {
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")
    return compensatedSum() / size
}

/**
 * Computes the arithmetic mean of the values in this sequence.
 *
 * The arithmetic mean is the sum of all values divided by the count. Uses compensated
 * (Neumaier) summation for improved numerical precision. The sequence is consumed once.
 *
 * ### Example:
 * ```kotlin
 * sequenceOf(1.0, 2.0, 3.0).mean() // 2.0
 * ```
 *
 * @return the arithmetic mean of the elements.
 */
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

/**
 * Computes the geometric mean of the values in this iterable.
 *
 * The geometric mean is the nth root of the product of n values. It is useful for data that
 * spans several orders of magnitude or for computing average growth rates. All values must
 * be positive. Computed via logarithms to avoid overflow.
 *
 * ### Example:
 * ```kotlin
 * listOf(1.0, 2.0, 4.0, 8.0).geometricMean() // 2.8284...
 * ```
 *
 * @return the geometric mean of the elements.
 */
public fun Iterable<Double>.geometricMean(): Double {
    var sumLn = 0.0
    var compensation = 0.0
    var count = 0
    for (element in this) {
        if (element <= 0.0) throw InvalidParameterException("All elements must be positive for geometric mean")
        val lnVal = ln(element)
        val t = sumLn + lnVal
        compensation += if (abs(sumLn) >= abs(lnVal)) (sumLn - t) + lnVal else (lnVal - t) + sumLn
        sumLn = t
        count++
    }
    if (count == 0) throw InsufficientDataException("Collection must not be empty")
    return exp((sumLn + compensation) / count)
}

/**
 * Computes the geometric mean of the values in this array.
 *
 * The geometric mean is the nth root of the product of n values. It is useful for data that
 * spans several orders of magnitude or for computing average growth rates. All values must
 * be positive. Computed via logarithms to avoid overflow.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(1.0, 2.0, 4.0, 8.0).geometricMean() // 2.8284...
 * ```
 *
 * @return the geometric mean of the array elements.
 */
public fun DoubleArray.geometricMean(): Double {
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")
    var sumLn = 0.0
    var compensation = 0.0
    for (element in this) {
        if (element <= 0.0) throw InvalidParameterException("All elements must be positive for geometric mean")
        val lnVal = ln(element)
        val t = sumLn + lnVal
        compensation += if (abs(sumLn) >= abs(lnVal)) (sumLn - t) + lnVal else (lnVal - t) + sumLn
        sumLn = t
    }
    return exp((sumLn + compensation) / size)
}

// ── harmonicMean ────────────────────────────────────────────────────────────

/**
 * Computes the harmonic mean of the values in this iterable.
 *
 * The harmonic mean is the reciprocal of the arithmetic mean of the reciprocals. It is
 * appropriate for averaging rates or ratios (e.g. speeds, P/E ratios). All values must
 * be positive. Uses compensated summation for the reciprocals.
 *
 * ### Example:
 * ```kotlin
 * listOf(1.0, 2.0, 4.0).harmonicMean() // 1.7142...
 * ```
 *
 * @return the harmonic mean of the elements.
 */
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

/**
 * Computes the harmonic mean of the values in this array.
 *
 * The harmonic mean is the reciprocal of the arithmetic mean of the reciprocals. It is
 * appropriate for averaging rates or ratios. All values must be positive. Uses compensated
 * summation for the reciprocals.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(1.0, 2.0, 4.0).harmonicMean() // 1.7142...
 * ```
 *
 * @return the harmonic mean of the array elements.
 */
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

/**
 * Computes the weighted arithmetic mean of the values using the given [weights].
 *
 * Each value is multiplied by its corresponding weight, the products are summed, and the
 * result is divided by the total weight. Weights must be non-negative and their sum must
 * be positive. The values and weights iterables must have the same number of elements.
 * Uses compensated summation for both the weighted sum and the total weight.
 *
 * ### Example:
 * ```kotlin
 * listOf(1.0, 2.0, 3.0).weightedMean(listOf(3.0, 1.0, 1.0)) // 1.6
 * ```
 *
 * @param weights the weights corresponding to each value. Must be non-negative.
 * @return the weighted arithmetic mean.
 */
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

/**
 * Computes the weighted arithmetic mean of the values using the given [weights].
 *
 * Each value is multiplied by its corresponding weight, the products are summed, and the
 * result is divided by the total weight. Weights must be non-negative and their sum must
 * be positive. The arrays must have the same size. Uses compensated summation.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(1.0, 2.0, 3.0).weightedMean(doubleArrayOf(3.0, 1.0, 1.0)) // 1.6
 * ```
 *
 * @param weights the weights corresponding to each value. Must be non-negative.
 * @return the weighted arithmetic mean.
 */
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

/**
 * Computes the median of the values in this iterable.
 *
 * The median is the middle value when the data is sorted. For an even number of elements,
 * it is the average of the two middle values. Unlike the mean, the median is robust to
 * outliers.
 *
 * ### Example:
 * ```kotlin
 * listOf(3.0, 1.0, 2.0).median()      // 2.0
 * listOf(1.0, 2.0, 3.0, 4.0).median() // 2.5
 * ```
 *
 * @return the median of the elements.
 */
public fun Iterable<Double>.median(): Double =
    medianInPlace(toList().toDoubleArray())

/**
 * Computes the median of the values in this array.
 *
 * The median is the middle value when the data is sorted. For an even number of elements,
 * it is the average of the two middle values. Uses introselect (O(n) expected time) instead
 * of a full sort for efficiency.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(3.0, 1.0, 2.0).median()      // 2.0
 * doubleArrayOf(1.0, 2.0, 3.0, 4.0).median() // 2.5
 * ```
 *
 * @return the median of the array elements.
 */
public fun DoubleArray.median(): Double = medianInPlace(copyOf())

/** Computes the median, mutating [work] in place via introselect. */
private fun medianInPlace(work: DoubleArray): Double {
    if (work.isEmpty()) throw InsufficientDataException("Array must not be empty")
    val n = work.size
    val mid = n / 2
    work.introSelect(mid)
    return if (n % 2 == 1) {
        work[mid]
    } else {
        // introSelect guarantees: work[0..mid-1] <= work[mid]
        // Find max of left half in O(n/2)
        var left = work[0]
        for (i in 1 until mid) {
            if (work[i].compareTo(left) > 0) left = work[i]
        }
        (left + work[mid]) / 2.0
    }
}

// ── mode ────────────────────────────────────────────────────────────────────

/**
 * Returns the mode (most frequently occurring values) of this iterable.
 *
 * The mode is the set of values that appear most often. If multiple values share the highest
 * frequency, all of them are returned (multimodal). Works with any comparable type, not just
 * Double.
 *
 * ### Example:
 * ```kotlin
 * listOf(1.0, 2.0, 2.0, 3.0).mode()       // setOf(2.0)
 * listOf(1.0, 1.0, 2.0, 2.0).mode()       // setOf(1.0, 2.0)
 * listOf("a", "b", "b", "c").mode()        // setOf("b")
 * ```
 *
 * @return a [Set] containing all values with the highest frequency.
 */
public fun <T : Comparable<T>> Iterable<T>.mode(): Set<T> {
    val counts = mutableMapOf<T, Int>()
    for (element in this) {
        counts[element] = (counts[element] ?: 0) + 1
    }
    if (counts.isEmpty()) throw InsufficientDataException("Collection must not be empty")
    val maxCount = counts.values.max()
    return counts.filter { it.value == maxCount }.keys
}

// ── trimmedMean ─────────────────────────────────────────────────────────────

/**
 * Computes the trimmed (truncated) mean by removing a fraction of values from each tail.
 *
 * The trimmed mean sorts the data, discards the lowest and highest [proportion] of values,
 * and computes the arithmetic mean of the remaining middle portion. This makes it more
 * robust to outliers than the regular mean. A proportion of 0.0 gives the ordinary mean;
 * a proportion approaching 0.5 converges toward the median.
 *
 * Uses compensated (Neumaier) summation for improved numerical precision with large values.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0).trimmedMean(0.1) // 5.5
 * ```
 *
 * @param proportion the fraction of values to remove from each tail, in [0.0, 0.5).
 * For example, 0.1 removes the lowest 10% and highest 10%.
 * @return the mean of the remaining values after trimming.
 * @see trimmedVariance
 */
public fun DoubleArray.trimmedMean(proportion: Double): Double {
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")
    if (proportion.isNaN() || proportion < 0.0 || proportion >= 0.5)
        throw InvalidParameterException("proportion must be in [0.0, 0.5), got $proportion")
    val sorted = sortedArray()
    val k = floor(size * proportion).toInt()
    val m = size - 2 * k
    var sum = 0.0
    var compensation = 0.0
    for (i in k until size - k) {
        val t = sum + sorted[i]
        compensation += if (abs(sum) >= abs(sorted[i])) (sum - t) + sorted[i] else (sorted[i] - t) + sum
        sum = t
    }
    return (sum + compensation) / m
}

/**
 * Computes the trimmed (truncated) mean by removing a fraction of values from each tail.
 *
 * The trimmed mean sorts the data, discards the lowest and highest [proportion] of values,
 * and computes the arithmetic mean of the remaining middle portion. This makes it more
 * robust to outliers than the regular mean. A proportion of 0.0 gives the ordinary mean;
 * a proportion approaching 0.5 converges toward the median.
 *
 * ### Example:
 * ```kotlin
 * listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0).trimmedMean(0.1) // 5.5
 * ```
 *
 * @param proportion the fraction of values to remove from each tail, in [0.0, 0.5).
 * For example, 0.1 removes the lowest 10% and highest 10%.
 * @return the mean of the remaining values after trimming.
 * @see trimmedVariance
 */
public fun Iterable<Double>.trimmedMean(proportion: Double): Double =
    toList().toDoubleArray().trimmedMean(proportion)

/**
 * Computes the trimmed (truncated) mean by removing a fraction of values from each tail.
 *
 * The trimmed mean sorts the data, discards the lowest and highest [proportion] of values,
 * and computes the arithmetic mean of the remaining middle portion. This makes it more
 * robust to outliers than the regular mean. A proportion of 0.0 gives the ordinary mean;
 * a proportion approaching 0.5 converges toward the median.
 *
 * ### Example:
 * ```kotlin
 * sequenceOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0).trimmedMean(0.1) // 5.5
 * ```
 *
 * @param proportion the fraction of values to remove from each tail, in [0.0, 0.5).
 * For example, 0.1 removes the lowest 10% and highest 10%.
 * @return the mean of the remaining values after trimming.
 * @see trimmedVariance
 */
public fun Sequence<Double>.trimmedMean(proportion: Double): Double =
    toList().toDoubleArray().trimmedMean(proportion)
