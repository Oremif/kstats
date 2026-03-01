package org.oremif.kstats.sampling

import org.oremif.kstats.descriptive.mean
import org.oremif.kstats.descriptive.standardDeviation

/**
 * Z-score normalization: (x - mean) / sd
 */
public fun DoubleArray.zScore(): DoubleArray {
    require(size >= 2) { "Need at least 2 elements for z-score" }
    val m = mean()
    val sd = standardDeviation()
    require(sd > 0.0) { "Standard deviation is zero, cannot compute z-scores" }
    return DoubleArray(size) { (this[it] - m) / sd }
}

public fun Iterable<Double>.zScore(): List<Double> {
    val list = toList()
    require(list.size >= 2) { "Need at least 2 elements for z-score" }
    val m = list.mean()
    val sd = list.standardDeviation()
    require(sd > 0.0) { "Standard deviation is zero, cannot compute z-scores" }
    return list.map { (it - m) / sd }
}

/**
 * Min-max normalization to [0, 1].
 */
public fun DoubleArray.minMaxNormalize(): DoubleArray {
    require(isNotEmpty()) { "Array must not be empty" }
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
    require(isNotEmpty()) { "Array must not be empty" }
    require(newMin < newMax) { "newMin must be less than newMax" }
    val minVal = min()
    val maxVal = max()
    val range = maxVal - minVal
    val newRange = newMax - newMin
    if (range == 0.0) return DoubleArray(size) { newMin }
    return DoubleArray(size) { (this[it] - minVal) / range * newRange + newMin }
}
