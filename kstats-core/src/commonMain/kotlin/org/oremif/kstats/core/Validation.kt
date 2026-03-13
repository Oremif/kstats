package org.oremif.kstats.core

internal fun <T> Collection<T>.requireNonEmpty(name: String = "collection"): Collection<T> {
    require(isNotEmpty()) { "$name must not be empty" }
    return this
}

internal fun DoubleArray.requireNonEmpty(name: String = "array"): DoubleArray {
    require(isNotEmpty()) { "$name must not be empty" }
    return this
}

internal fun <T> Collection<T>.requireMinSize(minSize: Int, name: String = "collection"): Collection<T> {
    require(size >= minSize) { "$name must have at least $minSize elements, got $size" }
    return this
}

internal fun DoubleArray.requireMinSize(minSize: Int, name: String = "array"): DoubleArray {
    require(size >= minSize) { "$name must have at least $minSize elements, got $size" }
    return this
}

internal fun requirePositive(value: Double, name: String = "value") {
    require(value > 0.0) { "$name must be positive, got $value" }
}

internal fun requireNonNegative(value: Double, name: String = "value") {
    require(value >= 0.0) { "$name must be non-negative, got $value" }
}

internal fun requireInRange(value: Double, min: Double, max: Double, name: String = "value") {
    require(value in min..max) { "$name must be in [$min, $max], got $value" }
}

internal fun requireProbability(value: Double, name: String = "probability") {
    requireInRange(value, 0.0, 1.0, name)
}
