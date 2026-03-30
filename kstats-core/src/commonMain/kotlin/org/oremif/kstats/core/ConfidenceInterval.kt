package org.oremif.kstats.core

/**
 * A confidence interval with a lower and upper bound.
 *
 * A confidence interval gives a range of plausible values for a population parameter.
 * If you compute a 95% confidence interval from many independent samples, approximately
 * 95% of those intervals will contain the true parameter value.
 *
 * Supports destructuring into `(lower, upper)`:
 * ```kotlin
 * val (lo, hi) = confidenceInterval
 * ```
 *
 * @property lower the lower bound of the interval.
 * @property upper the upper bound of the interval. Always greater than or equal to [lower]
 * for well-behaved statistics with sufficient data.
 */
public data class ConfidenceInterval(
    public val lower: Double,
    public val upper: Double,
)
