package org.oremif.kstats.sampling

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.abs
import kotlin.random.Random

/**
 * A biased coin that lands heads (true) with the given [probability].
 *
 * Useful for simulating Bernoulli trials with a specific success probability.
 *
 * ### Example:
 * ```kotlin
 * val coin = WeightedCoin(0.7)
 * coin.flip() // true with 70% probability
 * ```
 *
 * @param probability the probability of heads (true). Must be in [0, 1].
 * @param random the random number generator. Defaults to [Random].
 */
public class WeightedCoin(public val probability: Double, private val random: Random = Random) {
    init {
        if (probability !in 0.0..1.0) throw InvalidParameterException("probability must be in [0, 1], got $probability")
    }

    /**
     * Flips the coin and returns the result.
     *
     * @return `true` (heads) with the configured [probability], `false` (tails) otherwise.
     */
    public fun flip(): Boolean = random.nextDouble() < probability
}

/**
 * A weighted die that produces outcomes with probabilities proportional to their weights.
 *
 * Weights are normalized internally so they do not need to sum to 1. Uses a cumulative
 * weight lookup with binary search for O(log n) roll time, where n is the number of outcomes.
 *
 * ### Example:
 * ```kotlin
 * val die = WeightedDice(mapOf("A" to 3.0, "B" to 1.0))
 * die.roll() // "A" with 75% probability, "B" with 25%
 * ```
 *
 * @param T the type of outcomes.
 * @param weights a map from each outcome to its non-negative finite weight. At least one
 * weight must be positive. The iteration order of the map determines the internal ordering
 * of outcomes; use an insertion-ordered map (e.g. `linkedMapOf`) for reproducible results
 * with a seeded [random].
 * @param random the random number generator. Defaults to [Random].
 * @throws InsufficientDataException if [weights] is empty.
 * @throws InvalidParameterException if any weight is negative, non-finite, or all weights
 * are zero.
 */
public class WeightedDice<T>(weights: Map<T, Double>, private val random: Random = Random) {
    private val outcomes: List<T>
    private val cumulativeWeights: DoubleArray

    init {
        if (weights.isEmpty()) throw InsufficientDataException("weights must not be empty")
        // Neumaier compensated summation for numerical precision with many outcomes
        var totalWeight = 0.0
        var compensation = 0.0
        for (w in weights.values) {
            if (!w.isFinite()) throw InvalidParameterException("weights must be finite")
            if (w < 0.0) throw InvalidParameterException("weights must be non-negative")
            val t = totalWeight + w
            compensation += if (abs(totalWeight) >= abs(w)) (totalWeight - t) + w else (w - t) + totalWeight
            totalWeight = t
        }
        totalWeight += compensation
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

    /**
     * Rolls the die and returns one outcome, selected with probability proportional
     * to its weight.
     *
     * @return a randomly selected outcome of type [T].
     */
    public fun roll(): T {
        val u = random.nextDouble()
        val idx = cumulativeBinarySearch(cumulativeWeights, u)
        return outcomes[idx.coerceIn(outcomes.indices)]
    }
}

/**
 * Finds the leftmost index in a sorted [array] where `array[index] > value`.
 * Uses strict `>` so that zero-weight outcomes (whose cumulative weight equals
 * the previous one) are never selected.
 */
private fun cumulativeBinarySearch(array: DoubleArray, value: Double): Int {
    var low = 0
    var high = array.size - 1
    while (low < high) {
        val mid = (low + high) ushr 1
        if (array[mid] <= value) {
            low = mid + 1
        } else {
            high = mid
        }
    }
    return low
}

/**
 * Draws a random sample of [n] elements without replacement.
 *
 * Uses a partial Fisher-Yates shuffle to select [n] elements in O(n) time.
 * Each element can appear at most once in the result. The collection is materialized
 * to a mutable list internally.
 *
 * ### Example:
 * ```kotlin
 * listOf(1, 2, 3, 4, 5).randomSample(3) // e.g. [4, 1, 5]
 * ```
 *
 * @param T the type of elements.
 * @param n the number of elements to draw. Must be between 0 and the collection size.
 * @param random the random number generator. Defaults to [Random].
 * @return a list of [n] randomly selected elements.
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
 * Draws a bootstrap sample of [n] elements with replacement.
 *
 * Bootstrap sampling randomly picks elements from the list, allowing the same element
 * to be chosen multiple times. This is commonly used for estimating the sampling
 * distribution of a statistic (bootstrap method).
 *
 * ### Example:
 * ```kotlin
 * listOf(1, 2, 3).bootstrapSample(5) // e.g. [2, 1, 3, 1, 2]
 * ```
 *
 * @param T the type of elements.
 * @param n the number of elements to draw. Must be non-negative.
 * @param random the random number generator. Defaults to [Random].
 * @return a list of [n] randomly drawn elements, potentially with duplicates.
 */
public fun <T> List<T>.bootstrapSample(n: Int, random: Random = Random): List<T> {
    if (n < 0) throw InvalidParameterException("n must be non-negative")
    if (isEmpty() && n > 0) throw InsufficientDataException("List must not be empty")
    return List(n) { this[random.nextInt(size)] }
}

/**
 * Draws a bootstrap sample of [n] elements with replacement.
 *
 * This is a convenience overload that accepts any [Iterable]. The collection is
 * materialized to a list internally.
 *
 * @param T the type of elements.
 * @param n the number of elements to draw. Must be non-negative.
 * @param random the random number generator. Defaults to [Random].
 * @return a list of [n] randomly drawn elements, potentially with duplicates.
 */
public fun <T> Iterable<T>.bootstrapSample(n: Int, random: Random = Random): List<T> =
    toList().bootstrapSample(n, random)
