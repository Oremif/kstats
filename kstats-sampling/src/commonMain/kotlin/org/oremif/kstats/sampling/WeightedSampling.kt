package org.oremif.kstats.sampling

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.random.Random

public class WeightedCoin(public val p: Double, private val random: Random = Random) {
    init {
        if (p !in 0.0..1.0) throw InvalidParameterException("probability must be in [0, 1], got $p")
    }

    public fun flip(): Boolean = random.nextDouble() < p
}

public class WeightedDice<T>(weights: Map<T, Double>, private val random: Random = Random) {
    private val outcomes: List<T>
    private val cumulativeWeights: DoubleArray

    init {
        if (weights.isEmpty()) throw InsufficientDataException("weights must not be empty")
        if (!weights.values.all { it >= 0.0 }) throw InvalidParameterException("weights must be non-negative")
        val totalWeight = weights.values.sum()
        if (totalWeight <= 0.0) throw InvalidParameterException("total weight must be positive")

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
    if (n < 0) throw InvalidParameterException("n must be non-negative")
    if (n > list.size) throw InvalidParameterException("n ($n) cannot exceed collection size (${list.size})")

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
    if (isEmpty()) throw InsufficientDataException("List must not be empty")
    if (n < 0) throw InvalidParameterException("n must be non-negative")
    return List(n) { this[random.nextInt(size)] }
}
