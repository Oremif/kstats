package org.oremif.kstats.sampling

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.ceil
import kotlin.math.floor

/**
 * A histogram bin containing the items that fall within a value range.
 *
 * Bin boundaries follow the half-open interval convention: items are assigned to the bin
 * whose range contains them as `[start, end)`, except for the last bin which is `[start, end]`.
 * Boundary values (values that fall exactly on an interior bin edge) are assigned to the
 * higher bin via `floor(index)` arithmetic. As a result, [range] (a [ClosedRange]) may
 * report `contains(value) == true` for boundary values that are actually assigned to the
 * adjacent bin.
 *
 * This is a stable value type: its property set ([range], [items]) is fixed and will not
 * change in future versions.
 *
 * ### Example:
 * ```kotlin
 * val bins = listOf(1.0, 2.0, 3.0, 4.0, 5.0).bin(2.5)
 * bins[0].range // 1.0..3.5
 * bins[0].count // number of items in this bin
 * bins[0].items // the actual items
 * ```
 *
 * @param T the type of items in the bin.
 * @property range the closed interval of values that this bin covers.
 * @property items the elements whose values fall within [range].
 */
public data class Bin<T>(
    val range: ClosedRange<Double>,
    val items: List<T>
) {
    /**
     * Returns the number of items in this bin.
     */
    public val count: Int get() = items.size
}

/**
 * A histogram bin with frequency statistics but without the original items.
 *
 * This is a stable value type: its property set ([range], [count], [relativeFrequency],
 * [cumulativeFrequency]) is fixed and will not change in future versions.
 *
 * ### Example:
 * ```kotlin
 * val freq = listOf(1.0, 2.0, 3.0, 4.0, 5.0).frequencyTable(2)
 * freq[0].range               // bin boundaries
 * freq[0].count               // how many values in this bin
 * freq[0].relativeFrequency   // proportion of total
 * freq[0].cumulativeFrequency // running proportion
 * ```
 *
 * @property range the closed interval of values that this bin covers.
 * @property count the number of values that fall within [range].
 * @property relativeFrequency the proportion of total values in this bin (between 0 and 1).
 * @property cumulativeFrequency the running total of relative frequencies up to and including
 * this bin (between 0 and 1, with the last bin always equal to 1).
 */
public data class FrequencyBin(
    val range: ClosedRange<Double>,
    val count: Int,
    val relativeFrequency: Double,
    val cumulativeFrequency: Double
)

/**
 * Groups items into equal-width bins based on a numeric value extracted by [valueSelector].
 *
 * Each bin covers a range of width [binSize]. The first bin starts at [rangeStart] (or the
 * minimum extracted value if not specified). The number of bins is determined automatically
 * to cover all values.
 *
 * Items whose extracted value falls exactly on an interior bin boundary are assigned to the
 * higher bin via `floor(index)` arithmetic. The last bin includes its upper boundary
 * (i.e., bins are `[start, end)` except the last which is `[start, end]`).
 *
 * ### Example:
 * ```kotlin
 * data class Point(val x: Double, val label: String)
 * val points = listOf(Point(1.0, "a"), Point(3.5, "b"), Point(7.0, "c"))
 * val bins = points.binByDouble({ it.x }, binSize = 5.0)
 * bins.size         // 2
 * bins[0].count     // 2 (points at 1.0 and 3.5)
 * bins[1].count     // 1 (point at 7.0)
 * ```
 *
 * @param T the type of items being binned.
 * @param valueSelector extracts the numeric value to bin on from each item.
 * @param binSize the width of each bin. Must be a positive finite number.
 * @param rangeStart the lower bound of the first bin. Must be finite and less than or equal
 * to the minimum extracted value. Defaults to the minimum value in the collection.
 * @return a list of [Bin] objects ordered by range, each containing the items that fall
 * within that range. Returns an empty list if the collection is empty.
 * @throws InvalidParameterException if [binSize] is not positive or not finite, if
 * [rangeStart] is non-finite or exceeds the minimum value, or if [valueSelector]
 * produces non-finite values.
 */
public fun <T> Iterable<T>.binByDouble(
    valueSelector: (T) -> Double,
    binSize: Double,
    rangeStart: Double? = null
): List<Bin<T>> {
    if (!binSize.isFinite() || binSize <= 0.0) {
        throw InvalidParameterException("binSize must be a positive finite number, got $binSize")
    }
    val items = toList()
    if (items.isEmpty()) return emptyList()

    val values = DoubleArray(items.size)
    var minActual = Double.POSITIVE_INFINITY
    var maxActual = Double.NEGATIVE_INFINITY
    for (i in items.indices) {
        val v = valueSelector(items[i])
        if (!v.isFinite()) throw InvalidParameterException("valueSelector produced non-finite value: $v")
        values[i] = v
        if (v < minActual) minActual = v
        if (v > maxActual) maxActual = v
    }

    if (rangeStart != null) {
        if (!rangeStart.isFinite()) {
            throw InvalidParameterException("rangeStart must be finite, got $rangeStart")
        }
        if (rangeStart > minActual) {
            throw InvalidParameterException(
                "rangeStart ($rangeStart) must not exceed the minimum value ($minActual)"
            )
        }
    }
    val minVal = rangeStart ?: minActual
    val maxVal = maxActual

    val numBins = ceil((maxVal - minVal) / binSize).toInt().coerceAtLeast(1)
    val bins = Array(numBins) { i ->
        val start = minVal + i * binSize
        val end = if (i == numBins - 1) maxOf(start + binSize, maxVal) else start + binSize
        start..end to mutableListOf<T>()
    }

    for ((index, item) in items.withIndex()) {
        val v = values[index]
        val idx = floor((v - minVal) / binSize).toInt().coerceIn(0, numBins - 1)
        bins[idx].second.add(item)
    }

    return bins.map { (range, binItems) -> Bin(range, binItems) }
}

/**
 * Groups items into a fixed number of equal-width bins based on a numeric value extracted
 * by [valueSelector].
 *
 * The bin width is computed automatically by dividing the data range by [binCount].
 * If all values are identical, a single bin is created regardless of [binCount].
 *
 * Items on a bin boundary are assigned to the higher bin, except for items in the last bin
 * which includes its upper boundary.
 *
 * ### Example:
 * ```kotlin
 * val items = listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
 * val bins = items.binByDouble({ it }, binCount = 3)
 * bins.size // 3
 * ```
 *
 * @param T the type of items being binned.
 * @param valueSelector extracts the numeric value to bin on from each item.
 * @param binCount the desired number of bins. Must be positive.
 * @return a list of [Bin] objects ordered by range. Returns an empty list if the
 * collection is empty.
 * @throws InvalidParameterException if [binCount] is not positive or if [valueSelector]
 * produces non-finite values.
 */
public fun <T> Iterable<T>.binByDouble(
    valueSelector: (T) -> Double,
    binCount: Int
): List<Bin<T>> {
    if (binCount <= 0) throw InvalidParameterException("binCount must be positive")
    val items = toList()
    if (items.isEmpty()) return emptyList()

    val values = DoubleArray(items.size)
    var computedMin = Double.POSITIVE_INFINITY
    var computedMax = Double.NEGATIVE_INFINITY
    for (i in items.indices) {
        val v = valueSelector(items[i])
        if (!v.isFinite()) throw InvalidParameterException("valueSelector produced non-finite value: $v")
        values[i] = v
        if (v < computedMin) computedMin = v
        if (v > computedMax) computedMax = v
    }

    val minVal = computedMin
    val maxVal = computedMax
    val range = maxVal - minVal
    val effectiveCount = if (range == 0.0) 1 else binCount
    val binSize = if (range == 0.0) 1.0 else range / binCount

    val bins = Array(effectiveCount) { i ->
        val start = minVal + i * binSize
        val end = if (i == effectiveCount - 1) maxOf(start + binSize, maxVal) else start + binSize
        start..end to mutableListOf<T>()
    }

    for ((index, item) in items.withIndex()) {
        val v = values[index]
        val idx = if (range == 0.0) {
            0
        } else {
            // Use floor() consistent with binByDouble(binSize) for boundary assignment
            floor((v - minVal) / range * binCount).toInt().coerceIn(0, effectiveCount - 1)
        }
        bins[idx].second.add(item)
    }

    return bins.map { (r, binItems) -> Bin(r, binItems) }
}

/**
 * Groups the values into equal-width bins of the given [binSize].
 *
 * This is a convenience wrapper around [binByDouble] for collections of raw Double values.
 *
 * ### Example:
 * ```kotlin
 * listOf(1.0, 2.0, 3.0, 4.0, 5.0).bin(2.5)
 * // 2 bins: [1.0..3.5] and [3.5..6.0]
 * ```
 *
 * @param binSize the width of each bin. Must be positive.
 * @return a list of [Bin] objects containing the values that fall within each range.
 */
public fun Iterable<Double>.bin(binSize: Double): List<Bin<Double>> =
    binByDouble({ it }, binSize)

/**
 * Groups the values into a fixed number of equal-width bins.
 *
 * This is a convenience wrapper around [binByDouble] for collections of raw Double values.
 *
 * ### Example:
 * ```kotlin
 * listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0).bin(3)
 * // 3 bins spanning the range [1.0, 10.0]
 * ```
 *
 * @param binCount the desired number of bins. Must be positive.
 * @return a list of [Bin] objects containing the values that fall within each range.
 */
public fun Iterable<Double>.bin(binCount: Int): List<Bin<Double>> =
    binByDouble({ it }, binCount)

/**
 * Builds a frequency table by dividing the values into a fixed number of equal-width bins.
 *
 * Each bin includes its count, relative frequency (proportion of total), and cumulative
 * frequency (running proportion). The cumulative frequency of the last bin is always 1.0.
 *
 * ### Example:
 * ```kotlin
 * val freq = listOf(1.0, 2.0, 3.0, 4.0, 5.0).frequencyTable(2)
 * freq[0].relativeFrequency     // proportion of values in the first bin
 * freq.last().cumulativeFrequency // 1.0
 * ```
 *
 * @param binCount the desired number of bins. Must be positive.
 * @return a list of [FrequencyBin] objects ordered by range. Returns an empty list if the
 * collection is empty.
 */
public fun Iterable<Double>.frequencyTable(binCount: Int): List<FrequencyBin> =
    bin(binCount).toFrequencyBins()

/**
 * Builds a frequency table by dividing the values into equal-width bins of the given size.
 *
 * Each bin includes its count, relative frequency (proportion of total), and cumulative
 * frequency (running proportion). The cumulative frequency of the last bin is always 1.0.
 *
 * ### Example:
 * ```kotlin
 * val freq = listOf(1.0, 2.0, 3.0, 4.0, 5.0).frequencyTable(2.5)
 * freq[0].count                  // number of values in [1.0, 3.5]
 * freq.last().cumulativeFrequency // 1.0
 * ```
 *
 * @param binSize the width of each bin. Must be positive.
 * @return a list of [FrequencyBin] objects ordered by range. Returns an empty list if the
 * collection is empty.
 */
public fun Iterable<Double>.frequencyTable(binSize: Double): List<FrequencyBin> =
    bin(binSize).toFrequencyBins()

/**
 * Groups the values into equal-width bins of the given [binSize].
 *
 * This is a convenience overload that accepts a [DoubleArray]. The array is
 * converted to a list internally.
 *
 * @param binSize the width of each bin. Must be positive.
 * @return a list of [Bin] objects containing the values that fall within each range.
 * @see [Iterable.bin]
 */
public fun DoubleArray.bin(binSize: Double): List<Bin<Double>> =
    asList().bin(binSize)

/**
 * Groups the values into a fixed number of equal-width bins.
 *
 * This is a convenience overload that accepts a [DoubleArray]. The array is
 * converted to a list internally.
 *
 * @param binCount the desired number of bins. Must be positive.
 * @return a list of [Bin] objects containing the values that fall within each range.
 * @see [Iterable.bin]
 */
public fun DoubleArray.bin(binCount: Int): List<Bin<Double>> =
    asList().bin(binCount)

/**
 * Builds a frequency table by dividing the values into a fixed number of equal-width bins.
 *
 * This is a convenience overload that accepts a [DoubleArray]. The array is
 * converted to a list internally.
 *
 * @param binCount the desired number of bins. Must be positive.
 * @return a list of [FrequencyBin] objects ordered by range. Returns an empty list if the
 * array is empty.
 * @see [Iterable.frequencyTable]
 */
public fun DoubleArray.frequencyTable(binCount: Int): List<FrequencyBin> =
    asList().frequencyTable(binCount)

/**
 * Builds a frequency table by dividing the values into equal-width bins of the given size.
 *
 * This is a convenience overload that accepts a [DoubleArray]. The array is
 * converted to a list internally.
 *
 * @param binSize the width of each bin. Must be positive.
 * @return a list of [FrequencyBin] objects ordered by range. Returns an empty list if the
 * array is empty.
 * @see [Iterable.frequencyTable]
 */
public fun DoubleArray.frequencyTable(binSize: Double): List<FrequencyBin> =
    asList().frequencyTable(binSize)

private fun List<Bin<Double>>.toFrequencyBins(): List<FrequencyBin> {
    val total = sumOf { it.count }.toDouble()
    if (total == 0.0) return emptyList()

    var cumulative = 0.0
    return mapIndexed { index, bin ->
        val relative = bin.count / total
        cumulative += relative
        val cumulativeFrequency = if (index == lastIndex) 1.0 else cumulative
        FrequencyBin(bin.range, bin.count, relative, cumulativeFrequency)
    }
}
