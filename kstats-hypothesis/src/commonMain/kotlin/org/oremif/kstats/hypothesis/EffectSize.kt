package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.descriptive.mean
import org.oremif.kstats.descriptive.variance
import kotlin.math.asin
import kotlin.math.sqrt

/**
 * Computes Cohen's d effect size for the standardized mean difference between two groups.
 *
 * Cohen's d measures how far apart two group means are in units of standard deviation.
 * A value of 0 means identical means; values around 0.2, 0.5, and 0.8 are conventionally
 * considered small, medium, and large effects respectively (Cohen, 1988). The sign indicates
 * direction: positive when the mean of [x] exceeds the mean of [y].
 *
 * ### Example:
 * ```kotlin
 * val group1 = doubleArrayOf(2.0, 4.0, 6.0, 8.0)
 * val group2 = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
 * cohensD(group1, group2)                // ~1.15 (large effect)
 * cohensD(group1, group2, pooled = false) // same when n1 = n2
 * ```
 *
 * @param x the first sample. Must contain at least 2 elements.
 * @param y the second sample. Must contain at least 2 elements.
 * @param pooled whether to use the pooled standard deviation, which weights each group's
 * variance by its degrees of freedom. Defaults to `true`. When `false`, uses the unweighted
 * root-mean-square of the two group standard deviations, which treats both groups equally
 * regardless of sample size. The two variants give identical results when the sample sizes
 * are equal.
 * @return the Cohen's d effect size.
 */
public fun cohensD(
    x: DoubleArray,
    y: DoubleArray,
    pooled: Boolean = true,
): Double {
    if (x.size < 2) throw InsufficientDataException("Sample x must have at least 2 elements")
    if (y.size < 2) throw InsufficientDataException("Sample y must have at least 2 elements")

    val n1 = x.size.toDouble()
    val n2 = y.size.toDouble()
    val mean1 = x.mean()
    val mean2 = y.mean()
    val var1 = x.variance()
    val var2 = y.variance()

    val sd = if (pooled) {
        // Pooled standard deviation, weighted by degrees of freedom
        sqrt(((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0))
    } else {
        // Root-mean-square of the two standard deviations (unweighted)
        sqrt((var1 + var2) / 2.0)
    }

    return (mean1 - mean2) / sd
}

/**
 * Computes Cohen's h effect size for the difference between two proportions.
 *
 * Cohen's h measures how far apart two proportions are on a scale that accounts for the
 * non-linear nature of proportions. It applies an arcsine transformation to each proportion
 * before computing the difference, which stabilizes comparisons across the full range from
 * 0 to 1. Values around 0.2, 0.5, and 0.8 are conventionally considered small, medium,
 * and large effects respectively (Cohen, 1988). The sign indicates direction: positive when
 * [p1] exceeds [p2]. The result is bounded between -pi and pi.
 *
 * ### Example:
 * ```kotlin
 * cohensH(p1 = 0.65, p2 = 0.50) // ~0.31 (small-medium effect)
 * cohensH(p1 = 0.80, p2 = 0.20) // ~1.29 (large effect)
 * cohensH(p1 = 0.50, p2 = 0.50) // 0.0 (no effect)
 * ```
 *
 * @param p1 the first proportion. Must be in [0, 1].
 * @param p2 the second proportion. Must be in [0, 1].
 * @return the Cohen's h effect size, in the range [-pi, pi].
 */
public fun cohensH(
    p1: Double,
    p2: Double,
): Double {
    if (p1 < 0.0 || p1 > 1.0) throw InvalidParameterException("p1 must be in [0, 1], got $p1")
    if (p2 < 0.0 || p2 > 1.0) throw InvalidParameterException("p2 must be in [0, 1], got $p2")

    return 2.0 * asin(sqrt(p1)) - 2.0 * asin(sqrt(p2))
}
