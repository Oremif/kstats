package org.oremif.kstats.core

import org.oremif.kstats.core.exceptions.ConvergenceException
import org.oremif.kstats.core.exceptions.DegenerateDataException
import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException

/** Throws [InsufficientDataException] if this collection is empty. Returns the collection for chaining. */
internal fun <T> Collection<T>.requireNonEmpty(name: String = "collection"): Collection<T> {
    if (isEmpty()) throw InsufficientDataException("$name must not be empty")
    return this
}

/** Throws [InsufficientDataException] if this array is empty. Returns the array for chaining. */
internal fun DoubleArray.requireNonEmpty(name: String = "array"): DoubleArray {
    if (isEmpty()) throw InsufficientDataException("$name must not be empty")
    return this
}

/** Throws [InsufficientDataException] if this collection has fewer than [minSize] elements. Returns the collection for chaining. */
internal fun <T> Collection<T>.requireMinSize(minSize: Int, name: String = "collection"): Collection<T> {
    if (size < minSize) throw InsufficientDataException("$name must have at least $minSize elements, got $size")
    return this
}

/** Throws [InsufficientDataException] if this array has fewer than [minSize] elements. Returns the array for chaining. */
internal fun DoubleArray.requireMinSize(minSize: Int, name: String = "array"): DoubleArray {
    if (size < minSize) throw InsufficientDataException("$name must have at least $minSize elements, got $size")
    return this
}

/**
 * Throws [InvalidParameterException] if [value] is not strictly positive (> 0).
 *
 * NaN passes validation intentionally (`NaN <= 0.0` is `false` per IEEE 754).
 * This allows NaN to propagate through subsequent computation rather than being
 * rejected at the validation boundary.
 */
internal fun requirePositive(value: Double, name: String = "value") {
    if (value <= 0.0) throw InvalidParameterException("$name must be positive, got $value")
}

/**
 * Throws [InvalidParameterException] if [value] is negative (< 0).
 *
 * NaN passes validation intentionally (`NaN < 0.0` is `false` per IEEE 754).
 * This allows NaN to propagate through subsequent computation rather than being
 * rejected at the validation boundary.
 */
internal fun requireNonNegative(value: Double, name: String = "value") {
    if (value < 0.0) throw InvalidParameterException("$name must be non-negative, got $value")
}

/** Throws [InvalidParameterException] if [value] is outside the closed interval \[[min], [max]\]. */
internal fun requireInRange(value: Double, min: Double, max: Double, name: String = "value") {
    if (value !in min..max) throw InvalidParameterException("$name must be in [$min, $max], got $value")
}

/** Throws [InvalidParameterException] if [value] is outside \[0, 1\]. Delegates to [requireInRange]. */
internal fun requireProbability(value: Double, name: String = "probability") {
    requireInRange(value, 0.0, 1.0, name)
}

/** Throws [DegenerateDataException] with the lazy message if [condition] is false. */
internal fun requireNotDegenerate(condition: Boolean, lazyMessage: () -> String) {
    if (!condition) throw DegenerateDataException(lazyMessage())
}

/**
 * Throws [ConvergenceException] if [converged] is false.
 *
 * Captures the iteration count and last estimate for diagnostic purposes in the exception.
 */
internal fun checkConvergence(converged: Boolean, iterations: Int, lastEstimate: Double, lazyMessage: () -> String) {
    if (!converged) throw ConvergenceException(lazyMessage(), iterations, lastEstimate)
}
