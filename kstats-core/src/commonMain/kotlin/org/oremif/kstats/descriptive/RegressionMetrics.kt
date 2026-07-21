package org.oremif.kstats.descriptive

import org.oremif.kstats.core.compensatedSum
import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.abs
import kotlin.math.sqrt

// ── rmse ────────────────────────────────────────────────────────────────────
/**
 * Computes the Root Mean Square Error (RMSE) between actual and predicted values.
 *
 * RMSE squares each error before averaging, then takes the square root.
 *
 * Uses compensated (Neumaier) summation for improved numerical precision.
 *
 * NaN values propagate through the computation (IEEE 754 semantics): if any element is NaN,
 * the result is NaN. Filter NaN values before calling this function if that is not desired.
 *
 * ### Example:
 * ```kotlin
 * val actual = doubleArrayOf(3.0, 5.0, 2.5, 7.0)
 * val predicted = doubleArrayOf(2.8, 5.2, 2.1, 6.8)
 * rmse(actual, predicted) // 0.2449...
 * ```
 *
 * @return the root mean square error between actual and predicted.
 * Version 1: DoubleArray
 */
public fun rmse(actual: DoubleArray, predicted: DoubleArray): Double {
    if (actual.size != predicted.size)
        throw InvalidParameterException("actual and predicted must have the same size")
    if (actual.isEmpty()) throw InsufficientDataException("Arrays must not be empty")

    val squaredDiffs = DoubleArray(actual.size) {i ->
        val diff = actual[i] - predicted[i]
        diff * diff
    }
    return sqrt(squaredDiffs.compensatedSum() / actual.size)
}

/**
 * Computes the Root Mean Square Error (RMSE) between actual and predicted values.
 * Version 2: Iterable<Double> (for list, sets, etc)
 */
 
public fun rmse(actual: Iterable<Double>, predicted: Iterable<Double>): Double {
    val actualIter = actual.iterator()
    val predictedIter = predicted.iterator()
    var sum = 0.0
    var compensation = 0.0
    var count = 0
    while (actualIter.hasNext() && predictedIter.hasNext()) {
        val diff = actualIter.next() - predictedIter.next()
        val squared = diff * diff
        val t = sum + squared
        compensation += if (abs(sum) >= abs(squared)) (sum - t) + squared else (squared - t) + sum
        sum = t
        count++
    }
    if (actualIter.hasNext() || predictedIter.hasNext())
        throw InvalidParameterException("actual and predicted must have the same size")
    if (count == 0) throw InsufficientDataException("Collections must not be empty")
    return sqrt((sum + compensation) / count)
}

/**
 * Computes the Root Mean Square Error (RMSE) between actual and predicted values.
 * Version 3: Sequence<Double>
 */
 
public fun rmse(actual: Sequence<Double>, predicted: Sequence<Double>): Double =
    rmse(actual.asIterable(), predicted.asIterable())

// ── mae ─────────────────────────────────────────────────────────────────────

/**
 * Computes the Mean Absolute Error (MAE) between actual and predicted values.
 *
 * MAE averages the absolute value of each error.
 *
 * Uses compensated (Neumaier) summation for improved numerical precision.
 *
 * NaN values propagate through the computation (IEEE 754 semantics): if any element is NaN,
 * the result is NaN. Filter NaN values before calling this function if that is not desired.
 *
 * ### Example:
 * ```kotlin
 * val actual = doubleArrayOf(3.0, 5.0, 2.5, 7.0)
 * val predicted = doubleArrayOf(2.8, 5.2, 2.1, 6.8)
 * mae(actual, predicted) // 0.225
 * ```
 *
 * @return the mean absolute error between actual and predicted.
 * Version 1: DoubleArray
 */
public fun mae(actual: DoubleArray, predicted: DoubleArray): Double {
    if (actual.size != predicted.size)
        throw InvalidParameterException("actual and predicted must have the same size")
    if (actual.isEmpty()) throw InsufficientDataException("Arrays must not be empty")

    val absDiffs = DoubleArray(actual.size) { i -> abs(actual[i] - predicted[i]) }
    return absDiffs.compensatedSum() / actual.size
}

/**
 * Computes the Mean Absolute Error (MAE) between actual and predicted values.
 * Version 2: Iterable<Double> (for list, sets, etc)
 */
public fun mae(actual: Iterable<Double>, predicted: Iterable<Double>): Double {
    val actualIter = actual.iterator()
    val predictedIter = predicted.iterator()
    var sum = 0.0
    var compensation = 0.0
    var count = 0
    while (actualIter.hasNext() && predictedIter.hasNext()) {
        val absDiff = abs(actualIter.next() - predictedIter.next())
        val t = sum + absDiff
        compensation += if (abs(sum) >= abs(absDiff)) (sum - t) + absDiff else (absDiff - t) + sum
        sum = t
        count++
    }
    if (actualIter.hasNext() || predictedIter.hasNext())
        throw InvalidParameterException("actual and predicted must have the same size")
    if (count == 0) throw InsufficientDataException("Collections must not be empty")
    return (sum + compensation) / count
}

/**
 * Computes the Mean Absolute Error (MAE) between actual and predicted values.
 * Version 3: Sequence<Double>
 */
public fun mae(actual: Sequence<Double>, predicted: Sequence<Double>): Double =
    mae(actual.asIterable(), predicted.asIterable())
