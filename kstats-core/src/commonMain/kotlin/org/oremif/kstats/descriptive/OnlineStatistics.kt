package org.oremif.kstats.descriptive

import org.oremif.kstats.descriptive.PopulationKind.SAMPLE
import kotlin.math.min as kMin
import kotlin.math.max as kMax
import kotlin.math.sqrt

/**
 * A streaming one-pass accumulator for descriptive statistics.
 *
 * Computes mean, variance, standard deviation, skewness, and kurtosis without
 * storing individual data points. Uses the Terriberry (2008) extension of Welford's
 * online algorithm for numerically stable single-pass updates of central moments
 * M2, M3, and M4.
 *
 * This is analogous to Apache Commons Math `SummaryStatistics`.
 *
 * ### Example:
 * ```kotlin
 * val stats = OnlineStatistics()
 * stats.addAll(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0))
 * stats.mean      // 3.0
 * stats.variance() // 2.5
 * stats.count      // 5
 * ```
 *
 * @see DescriptiveStatistics
 */
public class OnlineStatistics {

    private var n: Long = 0L
    private var m1: Double = 0.0
    private var m2: Double = 0.0
    private var m3: Double = 0.0
    private var m4: Double = 0.0
    private var minVal: Double = Double.NaN
    private var maxVal: Double = Double.NaN

    /**
     * Adds a single observation to the accumulator.
     *
     * Updates all internal moments using the Terriberry (2008) single-pass algorithm.
     * The update order is critical: M4 → M3 → M2 → mean, as each formula uses the
     * previous values of the lower moments.
     *
     * @param x the value to add.
     */
    public fun add(x: Double) {
        val n1 = n
        n++
        val delta = x - m1
        val deltaN = delta / n
        val deltaN2 = deltaN * deltaN
        val term1 = delta * deltaN * n1

        m4 += term1 * deltaN2 * (n * n - 3 * n + 3) + 6.0 * deltaN2 * m2 - 4.0 * deltaN * m3
        m3 += term1 * deltaN * (n - 2) - 3.0 * deltaN * m2
        m2 += term1
        m1 += deltaN

        minVal = if (n1 == 0L) x else kMin(minVal, x)
        maxVal = if (n1 == 0L) x else kMax(maxVal, x)
    }

    /**
     * Adds all values from a [DoubleArray] to the accumulator.
     *
     * @param values the values to add.
     */
    public fun addAll(values: DoubleArray) {
        for (x in values) add(x)
    }

    /**
     * Adds all values from an [Iterable] to the accumulator.
     *
     * @param values the values to add.
     */
    public fun addAll(values: Iterable<Double>) {
        for (x in values) add(x)
    }

    /**
     * The number of observations added so far.
     *
     * Uses [Long] to support streams with more than 2^31 values.
     */
    public val count: Long get() = n

    /**
     * The arithmetic mean of all observations, or [Double.NaN] if no observations have been added.
     */
    public val mean: Double get() = if (n == 0L) Double.NaN else m1

    /**
     * The sum of all observations (`mean * count`), or [Double.NaN] if no observations have been added.
     */
    public val sum: Double get() = if (n == 0L) Double.NaN else m1 * n

    /**
     * The minimum observed value, or [Double.NaN] if no observations have been added.
     */
    public val min: Double get() = minVal

    /**
     * The maximum observed value, or [Double.NaN] if no observations have been added.
     */
    public val max: Double get() = maxVal

    /**
     * Computes the variance of all observations.
     *
     * Returns [Double.NaN] if there are fewer observations than required (at least 2 for
     * [PopulationKind.SAMPLE], at least 1 for [PopulationKind.POPULATION]).
     *
     * @param kind whether to compute sample variance (divides by n-1) or population variance
     * (divides by n). Defaults to [PopulationKind.SAMPLE].
     * @return the variance, or [Double.NaN] if insufficient data.
     */
    public fun variance(kind: PopulationKind = SAMPLE): Double {
        if (n == 0L) return Double.NaN
        return if (kind == SAMPLE) {
            if (n < 2L) Double.NaN else m2 / (n - 1)
        } else {
            m2 / n
        }
    }

    /**
     * Computes the standard deviation of all observations.
     *
     * This is the square root of [variance].
     *
     * @param kind whether to compute sample or population standard deviation.
     * Defaults to [PopulationKind.SAMPLE].
     * @return the standard deviation, or [Double.NaN] if insufficient data.
     */
    public fun standardDeviation(kind: PopulationKind = SAMPLE): Double = sqrt(variance(kind))

    /**
     * Computes the skewness of all observations.
     *
     * Returns [Double.NaN] if fewer than 3 observations have been added.
     * Returns 0.0 if the variance is zero (constant data).
     *
     * @param kind whether to compute sample-adjusted (Fisher-Pearson) or population skewness.
     * Defaults to [PopulationKind.SAMPLE].
     * @return the skewness, or [Double.NaN] if insufficient data.
     */
    public fun skewness(kind: PopulationKind = SAMPLE): Double {
        if (n < 3L) return Double.NaN
        if (m2 == 0.0) return 0.0
        val nd = n.toDouble()
        val g1 = sqrt(nd) * m3 / (m2 * sqrt(m2))
        return if (kind == SAMPLE) {
            sqrt(nd * (nd - 1.0)) / (nd - 2.0) * g1
        } else {
            g1
        }
    }

    /**
     * Computes the kurtosis of all observations.
     *
     * Returns [Double.NaN] if fewer than 4 observations have been added.
     * Returns -3.0 (excess) or 0.0 (non-excess) if the variance is zero (constant data).
     *
     * @param kind whether to compute sample-adjusted or population kurtosis.
     * Defaults to [PopulationKind.SAMPLE].
     * @param excess whether to subtract 3 (the kurtosis of a normal distribution).
     * Defaults to `true`.
     * @return the kurtosis, or [Double.NaN] if insufficient data.
     */
    public fun kurtosis(kind: PopulationKind = SAMPLE, excess: Boolean = true): Double {
        if (n < 4L) return Double.NaN
        if (m2 == 0.0) return if (excess) -3.0 else 0.0
        val nd = n.toDouble()
        val g2 = nd * m4 / (m2 * m2)
        return if (kind == SAMPLE) {
            val adj = (nd - 1.0) / ((nd - 2.0) * (nd - 3.0)) * ((nd + 1.0) * g2 - 3.0 * (nd - 1.0))
            if (excess) adj else adj + 3.0
        } else {
            if (excess) g2 - 3.0 else g2
        }
    }

    /**
     * Resets the accumulator to its initial empty state.
     */
    public fun clear() {
        n = 0L
        m1 = 0.0
        m2 = 0.0
        m3 = 0.0
        m4 = 0.0
        minVal = Double.NaN
        maxVal = Double.NaN
    }
}
