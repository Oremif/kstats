package org.oremif.kstats.sampling

public enum class TieMethod { AVERAGE, MIN, MAX, DENSE, ORDINAL }

/**
 * Compute ranks for the elements of this array.
 */
public fun DoubleArray.rank(tieMethod: TieMethod = TieMethod.AVERAGE): DoubleArray {
    require(isNotEmpty()) { "Array must not be empty" }

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
 * Compute percentile ranks for each element.
 * Returns values in [0, 100].
 */
public fun DoubleArray.percentileRank(): DoubleArray {
    require(isNotEmpty()) { "Array must not be empty" }
    val n = size
    val ranks = rank(TieMethod.AVERAGE)
    return DoubleArray(n) { (ranks[it] - 1.0) / (n - 1).coerceAtLeast(1) * 100.0 }
}
