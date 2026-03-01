package org.oremif.kstats.sampling

import org.oremif.kstats.core.exceptions.DegenerateDataException
import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.descriptive.mean
import org.oremif.kstats.descriptive.standardDeviation

/**
 * Threshold below which the standard deviation is considered effectively zero.
 * Chosen to be well above machine epsilon (~2.2e-16) to avoid division
 * by near-zero values that would produce unreliable z-scores.
 */
private const val NEAR_ZERO_SD_THRESHOLD = 1e-15

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
 * @throws InsufficientDataException if the array has fewer than 2 elements.
 * @throws InvalidParameterException if the array contains NaN or Infinity.
 * @throws DegenerateDataException if the standard deviation is zero (all values are identical).
 */
public fun DoubleArray.zScore(): DoubleArray {
    if (size < 2) throw InsufficientDataException("Need at least 2 elements for z-score")
    for (v in this) {
        if (!v.isFinite()) throw InvalidParameterException("Array contains non-finite value: $v")
    }
    val m = mean()
    val sd = standardDeviation()
    if (sd < NEAR_ZERO_SD_THRESHOLD) throw DegenerateDataException("Standard deviation is near-zero, cannot compute z-scores")
    return DoubleArray(size) { (this[it] - m) / sd }
}

/**
 * Computes the z-score (standard score) of each element in this iterable.
 *
 * This is a convenience overload that accepts any [Iterable]. The collection is
 * materialized to a [DoubleArray] internally.
 *
 * ### Example:
 * ```kotlin
 * listOf(1.0, 2.0, 3.0, 4.0, 5.0).zScore()
 * // [-1.2649, -0.6325, 0.0, 0.6325, 1.2649] (approximately)
 * ```
 *
 * @return a list of z-scores in the same order as the input.
 * @throws InsufficientDataException if the collection has fewer than 2 elements.
 * @throws InvalidParameterException if the collection contains NaN or Infinity.
 * @throws DegenerateDataException if the standard deviation is zero (all values are identical).
 * @see DoubleArray.zScore
 */
public fun Iterable<Double>.zScore(): List<Double> =
    toList().toDoubleArray().zScore().toList()

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
 * @throws InsufficientDataException if the array is empty.
 * @throws InvalidParameterException if the array contains NaN or Infinity.
 */
public fun DoubleArray.minMaxNormalize(): DoubleArray {
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")
    for (v in this) {
        if (!v.isFinite()) throw InvalidParameterException("Array contains non-finite value: $v")
    }
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
 * @param newMin the lower bound of the target range. Must be finite.
 * @param newMax the upper bound of the target range. Must be finite and greater than [newMin].
 * @return an array of normalized values in [[newMin], [newMax]].
 * @throws InsufficientDataException if the array is empty.
 * @throws InvalidParameterException if [newMin] >= [newMax], parameters are non-finite,
 * or the array contains NaN or Infinity.
 */
public fun DoubleArray.minMaxNormalize(newMin: Double, newMax: Double): DoubleArray {
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")
    if (!newMin.isFinite()) throw InvalidParameterException("newMin must be finite, got $newMin")
    if (!newMax.isFinite()) throw InvalidParameterException("newMax must be finite, got $newMax")
    if (newMin >= newMax) throw InvalidParameterException("newMin must be less than newMax")
    for (v in this) {
        if (!v.isFinite()) throw InvalidParameterException("Array contains non-finite value: $v")
    }
    val minVal = min()
    val maxVal = max()
    val range = maxVal - minVal
    val newRange = newMax - newMin
    if (range == 0.0) return DoubleArray(size) { newMin }
    return DoubleArray(size) { (this[it] - minVal) / range * newRange + newMin }
}

/**
 * Scales each element to the range [0, 1] using min-max normalization.
 *
 * This is a convenience overload that accepts any [Iterable]. The collection is
 * materialized to a [DoubleArray] internally.
 *
 * @return a list of normalized values in [0, 1].
 * @throws InsufficientDataException if the collection is empty.
 * @throws InvalidParameterException if the collection contains NaN or Infinity.
 * @see DoubleArray.minMaxNormalize
 */
public fun Iterable<Double>.minMaxNormalize(): List<Double> =
    toList().toDoubleArray().minMaxNormalize().toList()

/**
 * Scales each element to the range [[newMin], [newMax]] using min-max normalization.
 *
 * This is a convenience overload that accepts any [Iterable]. The collection is
 * materialized to a [DoubleArray] internally.
 *
 * @param newMin the lower bound of the target range. Must be finite.
 * @param newMax the upper bound of the target range. Must be finite and greater than [newMin].
 * @return a list of normalized values in [[newMin], [newMax]].
 * @throws InsufficientDataException if the collection is empty.
 * @throws InvalidParameterException if [newMin] >= [newMax], parameters are non-finite,
 * or the collection contains NaN or Infinity.
 * @see DoubleArray.minMaxNormalize
 */
public fun Iterable<Double>.minMaxNormalize(newMin: Double, newMax: Double): List<Double> =
    toList().toDoubleArray().minMaxNormalize(newMin, newMax).toList()
