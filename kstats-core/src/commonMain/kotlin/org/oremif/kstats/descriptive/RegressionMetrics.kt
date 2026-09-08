package org.oremif.kstats.descriptive

import kotlin.math.abs
import kotlin.math.sqrt
import org.oremif.kstats.core.compensatedSum
import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.core.neumaierTotal

/**
 * Mean of the error terms produced by [errorTerm] over paired elements of the two iterators.
 *
 * Uses the same Neumaier compensated summation as [compensatedSum], so the streaming overloads
 * agree with the array ones to the last bit. The iterators are consumed in lockstep and a length
 * mismatch is reported only after the shorter input is exhausted, so a NaN or infinite error term
 * can never mask the size check.
 */
private inline fun errorMean(
    actual: Iterator<Double>,
    predicted: Iterator<Double>,
    errorTerm: (Double) -> Double,
): Double {
    var sum = 0.0
    var compensation = 0.0
    var count = 0
    while (actual.hasNext() && predicted.hasNext()) {
        val term = errorTerm(actual.next() - predicted.next())
        val t = sum + term
        compensation += if (abs(sum) >= abs(term)) (sum - t) + term else (term - t) + sum
        sum = t
        count++
    }
    if (actual.hasNext() || predicted.hasNext())
        throw InvalidParameterException("actual and predicted must have the same size")
    if (count == 0) throw InsufficientDataException("Collections must not be empty")
    return neumaierTotal(sum, compensation) / count
}

/**
 * Computes the Root Mean Square Error (RMSE) between actual and predicted values.
 *
 * RMSE squares each error before averaging, then takes the square root, so large errors weigh more
 * heavily than they do in [mae]. Uses compensated (Neumaier) summation for improved numerical
 * precision.
 *
 * NaN values propagate through the computation (IEEE 754 semantics): if any element is NaN, the
 * result is NaN. Filter NaN values before calling this function if that is not desired. An error
 * that overflows to infinity yields [Double.POSITIVE_INFINITY].
 *
 * ### Example:
 * ```kotlin
 * val actual = doubleArrayOf(3.0, 5.0, 2.5, 7.0)
 * val predicted = doubleArrayOf(2.8, 5.2, 2.1, 6.8)
 * rmse(actual, predicted) // 0.2646...
 * ```
 *
 * @param actual the actual observed values.
 * @param predicted the predicted values corresponding to the actual values.
 * @return the root mean square error between actual and predicted.
 * @throws InvalidParameterException if the arrays have different sizes.
 * @throws InsufficientDataException if the arrays are empty.
 */
public fun rmse(actual: DoubleArray, predicted: DoubleArray): Double {
    if (actual.size != predicted.size)
        throw InvalidParameterException("actual and predicted must have the same size")
    if (actual.isEmpty()) throw InsufficientDataException("Arrays must not be empty")

    val squaredDiffs =
        DoubleArray(actual.size) { i ->
            val diff = actual[i] - predicted[i]
            diff * diff
        }
    return sqrt(squaredDiffs.compensatedSum() / actual.size)
}

/**
 * Computes the Root Mean Square Error (RMSE) between actual and predicted values.
 *
 * RMSE squares each error before averaging, then takes the square root, so large errors weigh more
 * heavily than they do in [mae]. Both iterables are consumed once, in lockstep, and must yield the
 * same number of elements. Uses compensated (Neumaier) summation for improved numerical precision.
 *
 * NaN values propagate through the computation (IEEE 754 semantics): if any element is NaN, the
 * result is NaN. Filter NaN values before calling this function if that is not desired. An error
 * that overflows to infinity yields [Double.POSITIVE_INFINITY].
 *
 * ### Example:
 * ```kotlin
 * rmse(listOf(3.0, 5.0, 2.5, 7.0), listOf(2.8, 5.2, 2.1, 6.8)) // 0.2646...
 * ```
 *
 * @param actual the actual observed values.
 * @param predicted the predicted values corresponding to the actual values.
 * @return the root mean square error between actual and predicted.
 * @throws InvalidParameterException if the iterables yield different numbers of elements.
 * @throws InsufficientDataException if the iterables are empty.
 */
public fun rmse(actual: Iterable<Double>, predicted: Iterable<Double>): Double =
    sqrt(errorMean(actual.iterator(), predicted.iterator()) { diff -> diff * diff })

/**
 * Computes the Root Mean Square Error (RMSE) between actual and predicted values.
 *
 * RMSE squares each error before averaging, then takes the square root, so large errors weigh more
 * heavily than they do in [mae]. Both sequences are consumed once, in lockstep, and must yield the
 * same number of elements. Uses compensated (Neumaier) summation for improved numerical precision.
 *
 * NaN values propagate through the computation (IEEE 754 semantics): if any element is NaN, the
 * result is NaN. Filter NaN values before calling this function if that is not desired. An error
 * that overflows to infinity yields [Double.POSITIVE_INFINITY].
 *
 * ### Example:
 * ```kotlin
 * rmse(sequenceOf(3.0, 5.0, 2.5, 7.0), sequenceOf(2.8, 5.2, 2.1, 6.8)) // 0.2646...
 * ```
 *
 * @param actual the actual observed values.
 * @param predicted the predicted values corresponding to the actual values.
 * @return the root mean square error between actual and predicted.
 * @throws InvalidParameterException if the sequences yield different numbers of elements.
 * @throws InsufficientDataException if the sequences are empty.
 */
public fun rmse(actual: Sequence<Double>, predicted: Sequence<Double>): Double =
    sqrt(errorMean(actual.iterator(), predicted.iterator()) { diff -> diff * diff })

// ── mae ─────────────────────────────────────────────────────────────────────

/**
 * Computes the Mean Absolute Error (MAE) between actual and predicted values.
 *
 * MAE averages the absolute value of each error, so every error contributes in proportion to its
 * magnitude — unlike [rmse], which penalises large errors disproportionately. Uses compensated
 * (Neumaier) summation for improved numerical precision.
 *
 * NaN values propagate through the computation (IEEE 754 semantics): if any element is NaN, the
 * result is NaN. Filter NaN values before calling this function if that is not desired. An error
 * that overflows to infinity yields [Double.POSITIVE_INFINITY].
 *
 * ### Example:
 * ```kotlin
 * val actual = doubleArrayOf(3.0, 5.0, 2.5, 7.0)
 * val predicted = doubleArrayOf(2.8, 5.2, 2.1, 6.8)
 * mae(actual, predicted) // 0.25
 * ```
 *
 * @param actual the actual observed values.
 * @param predicted the predicted values corresponding to the actual values.
 * @return the mean absolute error between actual and predicted.
 * @throws InvalidParameterException if the arrays have different sizes.
 * @throws InsufficientDataException if the arrays are empty.
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
 *
 * MAE averages the absolute value of each error, so every error contributes in proportion to its
 * magnitude — unlike [rmse], which penalises large errors disproportionately. Both iterables are
 * consumed once, in lockstep, and must yield the same number of elements. Uses compensated
 * (Neumaier) summation for improved numerical precision.
 *
 * NaN values propagate through the computation (IEEE 754 semantics): if any element is NaN, the
 * result is NaN. Filter NaN values before calling this function if that is not desired. An error
 * that overflows to infinity yields [Double.POSITIVE_INFINITY].
 *
 * ### Example:
 * ```kotlin
 * mae(listOf(3.0, 5.0, 2.5, 7.0), listOf(2.8, 5.2, 2.1, 6.8)) // 0.25
 * ```
 *
 * @param actual the actual observed values.
 * @param predicted the predicted values corresponding to the actual values.
 * @return the mean absolute error between actual and predicted.
 * @throws InvalidParameterException if the iterables yield different numbers of elements.
 * @throws InsufficientDataException if the iterables are empty.
 */
public fun mae(actual: Iterable<Double>, predicted: Iterable<Double>): Double =
    errorMean(actual.iterator(), predicted.iterator()) { diff -> abs(diff) }

/**
 * Computes the Mean Absolute Error (MAE) between actual and predicted values.
 *
 * MAE averages the absolute value of each error, so every error contributes in proportion to its
 * magnitude — unlike [rmse], which penalises large errors disproportionately. Both sequences are
 * consumed once, in lockstep, and must yield the same number of elements. Uses compensated
 * (Neumaier) summation for improved numerical precision.
 *
 * NaN values propagate through the computation (IEEE 754 semantics): if any element is NaN, the
 * result is NaN. Filter NaN values before calling this function if that is not desired. An error
 * that overflows to infinity yields [Double.POSITIVE_INFINITY].
 *
 * ### Example:
 * ```kotlin
 * mae(sequenceOf(3.0, 5.0, 2.5, 7.0), sequenceOf(2.8, 5.2, 2.1, 6.8)) // 0.25
 * ```
 *
 * @param actual the actual observed values.
 * @param predicted the predicted values corresponding to the actual values.
 * @return the mean absolute error between actual and predicted.
 * @throws InvalidParameterException if the sequences yield different numbers of elements.
 * @throws InsufficientDataException if the sequences are empty.
 */
public fun mae(actual: Sequence<Double>, predicted: Sequence<Double>): Double =
    errorMean(actual.iterator(), predicted.iterator()) { diff -> abs(diff) }
