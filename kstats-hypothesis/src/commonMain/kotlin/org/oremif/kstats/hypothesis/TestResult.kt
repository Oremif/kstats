package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.ConfidenceInterval

/**
 * The result of a statistical hypothesis test.
 *
 * Contains the test statistic, p-value, and optional additional information such as
 * degrees of freedom and a confidence interval. Use [isSignificant] to quickly check
 * whether the result is statistically significant at a given significance level.
 *
 * ### Example:
 * ```kotlin
 * val result = tTest(sample = doubleArrayOf(2.1, 2.5, 2.3, 2.8), mu = 2.0)
 * result.statistic          // t-statistic value
 * result.pValue             // p-value
 * result.isSignificant()    // true/false at alpha = 0.05
 * result.confidenceInterval // (lower, upper) bounds for the estimated mean
 * ```
 *
 * @property testName the name of the test that was performed (e.g. "One-Sample t-Test").
 * @property statistic the computed test statistic (e.g. t-value, chi-squared value, U statistic).
 * @property pValue the probability of observing a test statistic at least as extreme as the
 * computed value, assuming the null hypothesis is true. Smaller values indicate stronger
 * evidence against the null hypothesis.
 * @property degreesOfFreedom the degrees of freedom for the test distribution, or [Double.NaN]
 * if not applicable. Defaults to [Double.NaN].
 * @property alternative the direction of the alternative hypothesis that was tested.
 * Defaults to [Alternative.TWO_SIDED].
 * @property confidenceInterval the confidence interval for the estimated parameter,
 * or `null` if not computed. Defaults to `null`.
 * @property additionalInfo a map of supplementary statistics (e.g. "mean", "standardError",
 * "oddsRatio") that vary by test type. Defaults to an empty map.
 */
public data class TestResult(
    val testName: String,
    val statistic: Double,
    val pValue: Double,
    val degreesOfFreedom: Double = Double.NaN,
    val alternative: Alternative = Alternative.TWO_SIDED,
    val confidenceInterval: ConfidenceInterval? = null,
    val additionalInfo: Map<String, Double> = emptyMap()
) {
    /**
     * Returns `true` if the test result is statistically significant at the given significance level.
     *
     * A result is significant when the p-value is less than [alpha], meaning there is sufficient
     * evidence to reject the null hypothesis.
     *
     * ### Example:
     * ```kotlin
     * val result = tTest(doubleArrayOf(5.0, 6.0, 7.0), mu = 0.0)
     * result.isSignificant()      // true at default alpha = 0.05
     * result.isSignificant(0.01)  // check at stricter 1% level
     * ```
     *
     * @param alpha the significance level (Type I error rate). Defaults to `0.05` (5%).
     * @return `true` if [pValue] < [alpha].
     */
    public fun isSignificant(alpha: Double = 0.05): Boolean = pValue < alpha
}
