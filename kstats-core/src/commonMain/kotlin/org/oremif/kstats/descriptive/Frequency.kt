package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InvalidParameterException

/**
 * A data holder representing a single value and its observed frequency.
 *
 * @param T the type of the observed value.
 * @property value the observed value.
 * @property count the number of times [value] has been observed.
 */
public data class FrequencyEntry<T>(val value: T, val count: Long)

/**
 * A mutable accumulator that counts the frequency of discrete values.
 *
 * Maintains exact counts for each distinct value of type [T], and provides
 * cumulative counts, proportions, and mode queries. This is the discrete
 * counterpart of histogram binning — it counts exact matches rather than
 * grouping values into intervals.
 *
 * Analogous to Apache Commons Math `Frequency`.
 *
 * ### Example:
 * ```kotlin
 * val freq = Frequency<String>()
 * freq.addAll(listOf("a", "a", "b", "b", "b", "c"))
 * freq.count("b")            // 3
 * freq.proportion("b")       // 0.5
 * freq.cumulativeCount("b")  // 5  (a=2 + b=3)
 * freq.mode                  // setOf("b")
 * ```
 *
 * @param T the type of observed values; must be [Comparable] for cumulative queries and sorted output.
 */
public class Frequency<T : Comparable<T>> {

    private val counts: MutableMap<T, Long> = mutableMapOf()
    private var cachedTotal: Long = 0L
    private var sortedKeysCache: List<T>? = null

    private fun sortedKeys(): List<T> =
        sortedKeysCache ?: counts.keys.sorted().also { sortedKeysCache = it }

    private fun invalidateCache() {
        sortedKeysCache = null
    }

    // ── Mutators ─────────────────────────────────────────────────────────────

    /**
     * Adds a single observation of [value] (increments its count by 1).
     *
     * @param value the value to observe.
     */
    public fun add(value: T) {
        counts[value] = (counts[value] ?: 0L) + 1L
        cachedTotal++
        invalidateCache()
    }

    /**
     * Adds [count] observations of [value].
     *
     * @param value the value to observe.
     * @param count the number of observations to add; must be positive.
     * @throws InvalidParameterException if [count] is not positive.
     */
    public fun add(value: T, count: Long) {
        if (count <= 0L) throw InvalidParameterException("count must be positive, got $count")
        counts[value] = (counts[value] ?: 0L) + count
        cachedTotal += count
        invalidateCache()
    }

    /**
     * Adds a single observation for each element in [values].
     *
     * @param values the values to observe.
     */
    public fun addAll(values: Iterable<T>) {
        for (v in values) add(v)
    }

    /**
     * Resets the accumulator to its initial empty state.
     */
    public fun clear() {
        counts.clear()
        cachedTotal = 0L
        invalidateCache()
    }

    // ── Counts ───────────────────────────────────────────────────────────────

    /**
     * The total number of observations added so far.
     *
     * Uses [Long] to support streams with more than 2^31 values.
     */
    public val totalCount: Long get() = cachedTotal

    /**
     * The number of distinct values observed.
     */
    public val uniqueCount: Int get() = counts.size

    /**
     * Returns the number of times [value] has been observed, or 0 if it has never been observed.
     *
     * @param value the value to query.
     * @return the observed count for [value].
     */
    public fun count(value: T): Long = counts[value] ?: 0L

    /**
     * Returns the cumulative count for [value]: the sum of counts for all observed values
     * less than or equal to [value].
     *
     * @param value the upper bound (inclusive).
     * @return the cumulative count.
     */
    public fun cumulativeCount(value: T): Long {
        var cum = 0L
        for (key in sortedKeys()) {
            if (key > value) break
            cum += counts[key]!!
        }
        return cum
    }

    // ── Proportions ──────────────────────────────────────────────────────────

    /**
     * Returns the proportion of observations equal to [value] (`count / totalCount`).
     *
     * Returns 0.0 if no observations have been added.
     *
     * @param value the value to query.
     * @return the proportion in [0.0, 1.0], or 0.0 if empty.
     */
    public fun proportion(value: T): Double {
        if (cachedTotal == 0L) return 0.0
        return count(value).toDouble() / cachedTotal.toDouble()
    }

    /**
     * Returns the cumulative proportion for [value] (`cumulativeCount / totalCount`).
     *
     * Returns [Double.NaN] if no observations have been added (0/0).
     *
     * @param value the upper bound (inclusive).
     * @return the cumulative proportion, or [Double.NaN] if empty.
     */
    public fun cumulativeProportion(value: T): Double {
        if (cachedTotal == 0L) return Double.NaN
        return cumulativeCount(value).toDouble() / cachedTotal.toDouble()
    }

    // ── Query ────────────────────────────────────────────────────────────────

    /**
     * The set of values with the highest observed frequency (the statistical mode).
     *
     * Returns an empty set if no observations have been added.
     * Returns multiple values in case of a tie.
     */
    public val mode: Set<T>
        get() {
            if (counts.isEmpty()) return emptySet()
            val maxCount = counts.values.max()
            return counts.entries
                .filter { it.value == maxCount }
                .map { it.key }
                .toSet()
        }

    /**
     * All distinct observed values, sorted in ascending order.
     */
    public val values: List<T> get() = sortedKeys()

    /**
     * All observed values with their counts, sorted by value in ascending order.
     */
    public val entries: List<FrequencyEntry<T>>
        get() = sortedKeys().map { FrequencyEntry(it, counts[it]!!) }
}

// ── Factory ──────────────────────────────────────────────────────────────────

/**
 * Creates a [Frequency] accumulator from the elements of this iterable.
 *
 * ### Example:
 * ```kotlin
 * val freq = listOf("a", "b", "b", "c").toFrequency()
 * freq.count("b") // 2
 * ```
 *
 * @return a [Frequency] containing one observation per element.
 */
public fun <T : Comparable<T>> Iterable<T>.toFrequency(): Frequency<T> {
    val freq = Frequency<T>()
    freq.addAll(this)
    return freq
}
