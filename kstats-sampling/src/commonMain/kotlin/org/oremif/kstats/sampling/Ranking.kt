package org.oremif.kstats.sampling

import org.oremif.kstats.core.exceptions.InsufficientDataException

/**
 * Controls how tied (equal) values are assigned ranks.
 *
 * When multiple elements share the same value, they occupy consecutive rank positions.
 * This enum determines which rank each tied element receives.
 */
public enum class TieMethod {
    /**
     * Assigns each tied element the average of the positions they occupy.
     *
     * For example, if two elements tie for positions 3 and 4, both receive rank 3.5.
     * This is the most common method and is used by default in [rank].
     */
    AVERAGE,

    /**
     * Assigns each tied element the lowest position in the tie group.
     *
     * For example, if two elements tie for positions 3 and 4, both receive rank 3.
     */
    MIN,

    /**
     * Assigns each tied element the highest position in the tie group.
     *
     * For example, if two elements tie for positions 3 and 4, both receive rank 4.
     */
    MAX,

    /**
     * Assigns consecutive integer ranks to distinct values, ignoring gaps.
     *
     * If the sorted unique values are A < B < C, they receive ranks 1, 2, 3 regardless
     * of how many elements share each value. The maximum rank equals the number of
     * distinct values.
     */
    DENSE,

    /**
     * Assigns each element a unique rank based on its position after sorting.
     *
     * Tied elements receive different ranks depending on their original order. This is
     * the only method that guarantees all ranks are distinct.
     */
    ORDINAL
}

/**
 * Computes ranks for each element based on its relative position when sorted.
 *
 * Ranking assigns a 1-based position to each value according to its magnitude.
 * The smallest value gets rank 1 (or a tie-adjusted rank), and ranks increase
 * with value. The [tieMethod] parameter controls how equal values share ranks.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(3.0, 1.0, 4.0, 1.0, 5.0).rank() // [3.0, 1.5, 4.0, 1.5, 5.0]
 * doubleArrayOf(3.0, 1.0, 1.0, 5.0).rank(TieMethod.DENSE) // [2.0, 1.0, 1.0, 3.0]
 * ```
 *
 * @param tieMethod how to handle tied (equal) values. Defaults to [TieMethod.AVERAGE],
 * which assigns each tied element the mean of the positions they occupy.
 * @return an array of ranks in the same order as the input, where each element's rank
 * reflects its position in the sorted order.
 */
public fun DoubleArray.rank(tieMethod: TieMethod = TieMethod.AVERAGE): DoubleArray {
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")

    val n = size
    val indexed = Array(n) { IndexedValue(it, this[it]) }
    indexed.sortBy { it.value }

    val ranks = DoubleArray(n)

    var i = 0
    var denseRank = 0
    while (i < n) {
        var j = i
        while (j < n - 1 && indexed[j + 1].value == indexed[i].value) {
            j++
        }
        denseRank++

        // Positions i..j are tied
        for (k in i..j) {
            ranks[indexed[k].index] = when (tieMethod) {
                TieMethod.AVERAGE -> (i + j + 2.0) / 2.0 // 1-based average
                TieMethod.MIN -> (i + 1).toDouble()
                TieMethod.MAX -> (j + 1).toDouble()
                TieMethod.DENSE -> denseRank.toDouble()
                TieMethod.ORDINAL -> (k + 1).toDouble()
            }
        }
        i = j + 1
    }

    return ranks
}

/**
 * Computes the percentile rank of each element, indicating what percentage of values
 * fall at or below it.
 *
 * Percentile ranks are derived from average-method ranks and scaled to the range
 * 0 to 100. The smallest value receives 0 and the largest receives 100. For a
 * single-element array, the result is 0.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(10.0, 20.0, 30.0, 40.0, 50.0).percentileRank()
 * // [0.0, 25.0, 50.0, 75.0, 100.0]
 * ```
 *
 * @return an array of percentile ranks in the range [0, 100], preserving the original
 * element order.
 */
public fun DoubleArray.percentileRank(): DoubleArray {
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")
    val n = size
    val ranks = rank(TieMethod.AVERAGE)
    return DoubleArray(n) { (ranks[it] - 1.0) / (n - 1).coerceAtLeast(1) * 100.0 }
}
