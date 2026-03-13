package org.oremif.kstats.sampling

import org.oremif.kstats.core.exceptions.DegenerateDataException
import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.descriptive.mean
import org.oremif.kstats.descriptive.standardDeviation

/**
 * Z-score normalization: (x - mean) / sd
 */
public fun DoubleArray.zScore(): DoubleArray {
    if (size < 2) throw InsufficientDataException("Need at least 2 elements for z-score")
    val m = mean()
    val sd = standardDeviation()
    if (sd <= 0.0) throw DegenerateDataException("Standard deviation is zero, cannot compute z-scores")
    return DoubleArray(size) { (this[it] - m) / sd }
}

public fun Iterable<Double>.zScore(): List<Double> {
    val list = toList()
    if (list.size < 2) throw InsufficientDataException("Need at least 2 elements for z-score")
    val m = list.mean()
    val sd = list.standardDeviation()
    if (sd <= 0.0) throw DegenerateDataException("Standard deviation is zero, cannot compute z-scores")
    return list.map { (it - m) / sd }
}

/**
 * Min-max normalization to [0, 1].
 */
public fun DoubleArray.minMaxNormalize(): DoubleArray {
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")
    val minVal = min()
    val maxVal = max()
    val range = maxVal - minVal
    if (range == 0.0) return DoubleArray(size) { 0.0 }
    return DoubleArray(size) { (this[it] - minVal) / range }
}

/**
 * Min-max normalization to [newMin, newMax].
 */
public fun DoubleArray.minMaxNormalize(newMin: Double, newMax: Double): DoubleArray {
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")
    if (newMin >= newMax) throw InvalidParameterException("newMin must be less than newMax")
    val minVal = min()
    val maxVal = max()
    val range = maxVal - minVal
    val newRange = newMax - newMin
    if (range == 0.0) return DoubleArray(size) { newMin }
    return DoubleArray(size) { (this[it] - minVal) / range * newRange + newMin }
}
