package org.oremif.kstats.sampling

import kotlin.random.Random

public class WeightedCoin(public val p: Double, private val random: Random = Random) {
    init {
        require(p in 0.0..1.0) { "probability must be in [0, 1], got $p" }
    }

    public fun flip(): Boolean = random.nextDouble() < p
}

public class WeightedDice<T>(weights: Map<T, Double>, private val random: Random = Random) {
    private val outcomes: List<T>
    private val cumulativeWeights: DoubleArray

    init {
        require(weights.isNotEmpty()) { "weights must not be empty" }
        require(weights.values.all { it >= 0.0 }) { "weights must be non-negative" }
        val totalWeight = weights.values.sum()
        require(totalWeight > 0.0) { "total weight must be positive" }

        outcomes = weights.keys.toList()
        val normalized = weights.values.map { it / totalWeight }
        cumulativeWeights = DoubleArray(normalized.size)
        var cumulative = 0.0
        for (i in normalized.indices) {
            cumulative += normalized[i]
            cumulativeWeights[i] = cumulative
        }
    }

    public fun roll(): T {
        val u = random.nextDouble()
        for (i in cumulativeWeights.indices) {
            if (u <= cumulativeWeights[i]) return outcomes[i]
        }
        return outcomes.last()
    }
}

/**
 * Random sample without replacement.
 */
public fun <T> Iterable<T>.randomSample(n: Int, random: Random = Random): List<T> {
    val list = toMutableList()
    require(n >= 0) { "n must be non-negative" }
    require(n <= list.size) { "n ($n) cannot exceed collection size (${list.size})" }

    // Fisher-Yates shuffle for first n elements
    for (i in 0 until n) {
        val j = i + random.nextInt(list.size - i)
        val temp = list[i]
        list[i] = list[j]
        list[j] = temp
    }
    return list.subList(0, n).toList()
}

/**
 * Bootstrap sample with replacement.
 */
public fun <T> List<T>.bootstrapSample(n: Int, random: Random = Random): List<T> {
    require(isNotEmpty()) { "List must not be empty" }
    require(n >= 0) { "n must be non-negative" }
    return List(n) { this[random.nextInt(size)] }
}
