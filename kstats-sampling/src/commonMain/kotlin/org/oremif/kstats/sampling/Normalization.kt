package org.oremif.kstats.sampling

import org.oremif.kstats.core.exceptions.DegenerateDataException
import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.descriptive.mean
import org.oremif.kstats.descriptive.standardDeviation

/**
 * Computes the z-score (standard score) of each element.
 *
 * The z-score expresses how many standard deviations an element is from the mean.
 * Each value is transformed by subtracting the sample mean and dividing by the
 * sample standard deviation. The resulting array has a mean of approximately 0
 * and a standard deviation of approximately 1.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0).zScore()
 * // [-1.2649, -0.6325, 0.0, 0.6325, 1.2649] (approximately)
 * ```
 *
 * @return an array of z-scores in the same order as the input.
 * @throws DegenerateDataException if the standard deviation is zero (all values are identical).
 */
public fun DoubleArray.zScore(): DoubleArray {
    if (size < 2) throw InsufficientDataException("Need at least 2 elements for z-score")
    val m = mean()
    val sd = standardDeviation()
    if (sd <= 0.0) throw DegenerateDataException("Standard deviation is zero, cannot compute z-scores")
    return DoubleArray(size) { (this[it] - m) / sd }
}

/**
 * Computes the z-score (standard score) of each element in this iterable.
 *
 * The z-score expresses how many standard deviations an element is from the mean.
 * Each value is transformed by subtracting the sample mean and dividing by the
 * sample standard deviation. The resulting list has a mean of approximately 0
 * and a standard deviation of approximately 1.
 *
 * ### Example:
 * ```kotlin
 * listOf(1.0, 2.0, 3.0, 4.0, 5.0).zScore()
 * // [-1.2649, -0.6325, 0.0, 0.6325, 1.2649] (approximately)
 * ```
 *
 * @return a list of z-scores in the same order as the input.
 * @throws DegenerateDataException if the standard deviation is zero (all values are identical).
 */
public fun Iterable<Double>.zScore(): List<Double> {
    val list = toList()
    if (list.size < 2) throw InsufficientDataException("Need at least 2 elements for z-score")
    val m = list.mean()
    val sd = list.standardDeviation()
    if (sd <= 0.0) throw DegenerateDataException("Standard deviation is zero, cannot compute z-scores")
    return list.map { (it - m) / sd }
}

/**
 * Scales each element to the range [0, 1] using min-max normalization.
 *
 * The minimum value maps to 0 and the maximum maps to 1, with all other values
 * linearly interpolated between them. If all values are identical (range is zero),
 * every element maps to 0.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0).minMaxNormalize()
 * // [0.0, 0.25, 0.5, 0.75, 1.0]
 * ```
 *
 * @return an array of normalized values in [0, 1].
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
 * Scales each element to the range [[newMin], [newMax]] using min-max normalization.
 *
 * The minimum value maps to [newMin] and the maximum maps to [newMax], with all other
 * values linearly interpolated between them. If all values are identical (range is zero),
 * every element maps to [newMin].
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(0.0, 5.0, 10.0).minMaxNormalize(-1.0, 1.0) // [-1.0, 0.0, 1.0]
 * ```
 *
 * @param newMin the lower bound of the target range.
 * @param newMax the upper bound of the target range. Must be greater than [newMin].
 * @return an array of normalized values in [[newMin], [newMax]].
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
