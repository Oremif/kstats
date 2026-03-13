package org.oremif.kstats.sampling

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.ceil
import kotlin.math.floor

public data class Bin<T>(
    val range: ClosedRange<Double>,
    val items: List<T>
) {
    public val count: Int get() = items.size
}

public data class FrequencyBin(
    val range: ClosedRange<Double>,
    val count: Int,
    val relativeFrequency: Double,
    val cumulativeFrequency: Double
)

/**
 * Bin items by a double value selector with a fixed bin size.
 */
public fun <T> Iterable<T>.binByDouble(
    valueSelector: (T) -> Double,
    binSize: Double,
    rangeStart: Double? = null
): List<Bin<T>> {
    if (binSize <= 0.0) throw InvalidParameterException("binSize must be positive")
    val items = toList()
    if (items.isEmpty()) return emptyList()

    val values = items.map(valueSelector)
    val minVal = rangeStart ?: values.min()
    val maxVal = values.max()

    val numBins = ceil((maxVal - minVal) / binSize).toInt().coerceAtLeast(1)
    val bins = Array(numBins) { i ->
        val start = minVal + i * binSize
        val end = start + binSize
        start..end to mutableListOf<T>()
    }

    for (item in items) {
        val v = valueSelector(item)
        val idx = floor((v - minVal) / binSize).toInt().coerceIn(0, numBins - 1)
        bins[idx].second.add(item)
    }

    return bins.map { (range, binItems) -> Bin(range, binItems) }
}

/**
 * Bin items by a double value selector with a fixed number of bins.
 */
public fun <T> Iterable<T>.binByDouble(
    valueSelector: (T) -> Double,
    binCount: Int
): List<Bin<T>> {
    if (binCount <= 0) throw InvalidParameterException("binCount must be positive")
    val items = toList()
    if (items.isEmpty()) return emptyList()

    val values = items.map(valueSelector)
    val minVal = values.min()
    val maxVal = values.max()
    val range = maxVal - minVal
    val binSize = if (range == 0.0) 1.0 else range / binCount

    return binByDouble(valueSelector, binSize, minVal)
}

/**
 * Direct binning of double values with fixed bin size.
 */
public fun Iterable<Double>.bin(binSize: Double): List<Bin<Double>> =
    binByDouble({ it }, binSize)

/**
 * Direct binning of double values with fixed number of bins.
 */
public fun Iterable<Double>.bin(binCount: Int): List<Bin<Double>> =
    binByDouble({ it }, binCount)

/**
 * Frequency table with fixed number of bins.
 */
public fun Iterable<Double>.frequencyTable(binCount: Int): List<FrequencyBin> {
    val bins = bin(binCount)
    val total = bins.sumOf { it.count }.toDouble()
    if (total == 0.0) return emptyList()

    var cumulative = 0.0
    return bins.map { bin ->
        val relative = bin.count / total
        cumulative += relative
        FrequencyBin(bin.range, bin.count, relative, cumulative)
    }
}

/**
 * Frequency table with fixed bin size.
 */
public fun Iterable<Double>.frequencyTable(binSize: Double): List<FrequencyBin> {
    val bins = bin(binSize)
    val total = bins.sumOf { it.count }.toDouble()
    if (total == 0.0) return emptyList()

    var cumulative = 0.0
    return bins.map { bin ->
        val relative = bin.count / total
        cumulative += relative
        FrequencyBin(bin.range, bin.count, relative, cumulative)
    }
}
