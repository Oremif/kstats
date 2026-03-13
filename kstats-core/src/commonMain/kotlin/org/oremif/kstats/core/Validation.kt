package org.oremif.kstats.core

import org.oremif.kstats.core.exceptions.*

internal fun <T> Collection<T>.requireNonEmpty(name: String = "collection"): Collection<T> {
    if (isEmpty()) throw InsufficientDataException("$name must not be empty")
    return this
}

internal fun DoubleArray.requireNonEmpty(name: String = "array"): DoubleArray {
    if (isEmpty()) throw InsufficientDataException("$name must not be empty")
    return this
}

internal fun <T> Collection<T>.requireMinSize(minSize: Int, name: String = "collection"): Collection<T> {
    if (size < minSize) throw InsufficientDataException("$name must have at least $minSize elements, got $size")
    return this
}

internal fun DoubleArray.requireMinSize(minSize: Int, name: String = "array"): DoubleArray {
    if (size < minSize) throw InsufficientDataException("$name must have at least $minSize elements, got $size")
    return this
}

internal fun requirePositive(value: Double, name: String = "value") {
    if (value <= 0.0) throw InvalidParameterException("$name must be positive, got $value")
}

internal fun requireNonNegative(value: Double, name: String = "value") {
    if (value < 0.0) throw InvalidParameterException("$name must be non-negative, got $value")
}

internal fun requireInRange(value: Double, min: Double, max: Double, name: String = "value") {
    if (value !in min..max) throw InvalidParameterException("$name must be in [$min, $max], got $value")
}

internal fun requireProbability(value: Double, name: String = "probability") {
    requireInRange(value, 0.0, 1.0, name)
}

internal fun requireNotDegenerate(condition: Boolean, lazyMessage: () -> String) {
    if (!condition) throw DegenerateDataException(lazyMessage())
}

internal fun checkConvergence(converged: Boolean, iterations: Int, lastEstimate: Double, lazyMessage: () -> String) {
    if (!converged) throw ConvergenceException(lazyMessage(), iterations, lastEstimate)
}
