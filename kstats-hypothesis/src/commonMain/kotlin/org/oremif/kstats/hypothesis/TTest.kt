package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.descriptive.mean
import org.oremif.kstats.descriptive.standardDeviation
import org.oremif.kstats.distributions.StudentTDistribution
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Performs a one-sample t-test for whether the population mean equals [mu].
 *
 * The null hypothesis is that the true mean of the population from which [sample] was
 * drawn equals [mu]. The test uses the Student's t-distribution with n-1 degrees of freedom,
 * where n is the sample size.
 *
 * ### Example:
 * ```kotlin
 * val sample = doubleArrayOf(5.0, 6.0, 7.0, 5.5, 6.5)
 * val result = tTest(sample, mu = 5.0)
 * result.statistic          // t-statistic
 * result.pValue             // p-value
 * result.confidenceInterval // 95% CI for the sample mean
 * result.isSignificant()    // true if p < 0.05
 * ```
 *
 * @param sample the observed values. Must contain at least 2 elements.
 * @param mu the hypothesized population mean. Defaults to `0.0`.
 * @param alternative the direction of the alternative hypothesis. Defaults to [Alternative.TWO_SIDED],
 * which tests whether the true mean differs from [mu] in either direction.
 * @param confidenceLevel the confidence level for the confidence interval. Defaults to `0.95` (95%).
 * @return a [TestResult] containing the t-statistic, p-value, degrees of freedom (n-1),
 * a confidence interval for the mean, and additional info with "mean" and "standardError".
 */
public fun tTest(
    sample: DoubleArray,
    mu: Double = 0.0,
    alternative: Alternative = Alternative.TWO_SIDED,
    confidenceLevel: Double = 0.95
): TestResult {
    if (sample.size < 2) throw InsufficientDataException("Sample must have at least 2 elements")
    if (confidenceLevel <= 0.0 || confidenceLevel >= 1.0) throw InvalidParameterException(
        "confidenceLevel must be in (0, 1), got $confidenceLevel"
    )

    val n = sample.size
    val mean = sample.mean()
    val sd = sample.standardDeviation()
    val se = sd / sqrt(n.toDouble())
    val t = (mean - mu) / se
    val df = (n - 1).toDouble()

    val dist = StudentTDistribution(df)
    val pValue = when (alternative) {
        Alternative.TWO_SIDED -> 2.0 * dist.sf(abs(t))
        Alternative.LESS -> dist.cdf(t)
        Alternative.GREATER -> dist.sf(t)
    }

    val alpha = 1.0 - confidenceLevel
    val ci = when (alternative) {
        Alternative.TWO_SIDED -> {
            val tCrit = dist.quantile(1.0 - alpha / 2.0)
            Pair(mean - tCrit * se, mean + tCrit * se)
        }
        Alternative.LESS -> {
            val tCrit = dist.quantile(1.0 - alpha)
            Pair(Double.NEGATIVE_INFINITY, mean + tCrit * se)
        }
        Alternative.GREATER -> {
            val tCrit = dist.quantile(1.0 - alpha)
            Pair(mean - tCrit * se, Double.POSITIVE_INFINITY)
        }
    }

    return TestResult(
        testName = "One-Sample t-Test",
        statistic = t,
        pValue = pValue.coerceIn(0.0, 1.0),
        degreesOfFreedom = df,
        alternative = alternative,
        confidenceInterval = ci,
        additionalInfo = mapOf("mean" to mean, "standardError" to se)
    )
}

/**
 * Performs a two-sample t-test for whether two populations have the same mean.
 *
 * The null hypothesis is that the two populations from which [sample1] and [sample2] were
 * drawn have equal means. By default, uses Welch's t-test which does not assume equal
 * variances. Set [equalVariances] to `true` for the pooled (Student's) variant when the
 * populations are known to have similar variances.
 *
 * ### Example:
 * ```kotlin
 * val control = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
 * val treatment = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
 * val result = tTest(control, treatment)
 * result.statistic          // t-statistic
 * result.pValue             // p-value
 * result.confidenceInterval // 95% CI for the difference in means
 * ```
 *
 * @param sample1 the first sample. Must contain at least 2 elements.
 * @param sample2 the second sample. Must contain at least 2 elements.
 * @param alternative the direction of the alternative hypothesis. Defaults to [Alternative.TWO_SIDED],
 * which tests whether the means differ in either direction.
 * @param equalVariances whether to assume equal variances in both populations. Defaults to `false`
 * (Welch's t-test). Set to `true` for the pooled (Student's) t-test.
 * @param confidenceLevel the confidence level for the confidence interval. Defaults to `0.95` (95%).
 * @return a [TestResult] containing the t-statistic, p-value, degrees of freedom, a confidence
 * interval for the difference in means, and additional info with "mean1", "mean2", and "meanDifference".
 */
public fun tTest(
    sample1: DoubleArray,
    sample2: DoubleArray,
    alternative: Alternative = Alternative.TWO_SIDED,
    equalVariances: Boolean = false,
    confidenceLevel: Double = 0.95
): TestResult {
    if (sample1.size < 2) throw InsufficientDataException("Sample 1 must have at least 2 elements")
    if (sample2.size < 2) throw InsufficientDataException("Sample 2 must have at least 2 elements")
    if (confidenceLevel <= 0.0 || confidenceLevel >= 1.0) throw InvalidParameterException(
        "confidenceLevel must be in (0, 1), got $confidenceLevel"
    )

    val n1 = sample1.size.toDouble()
    val n2 = sample2.size.toDouble()
    val mean1 = sample1.mean()
    val mean2 = sample2.mean()
    val var1 = sample1.sumOf { (it - mean1) * (it - mean1) } / (n1 - 1.0)
    val var2 = sample2.sumOf { (it - mean2) * (it - mean2) } / (n2 - 1.0)

    val t: Double
    val df: Double
    val se: Double

    if (equalVariances) {
        // Pooled t-test
        val sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        se = sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
        t = (mean1 - mean2) / se
        df = n1 + n2 - 2.0
    } else {
        // Welch's t-test
        se = sqrt(var1 / n1 + var2 / n2)
        t = (mean1 - mean2) / se
        val num = (var1 / n1 + var2 / n2) * (var1 / n1 + var2 / n2)
        val den = (var1 / n1) * (var1 / n1) / (n1 - 1) + (var2 / n2) * (var2 / n2) / (n2 - 1)
        df = num / den
    }

    val dist = StudentTDistribution(df)
    val pValue = when (alternative) {
        Alternative.TWO_SIDED -> 2.0 * dist.sf(abs(t))
        Alternative.LESS -> dist.cdf(t)
        Alternative.GREATER -> dist.sf(t)
    }

    val diff = mean1 - mean2
    val alpha = 1.0 - confidenceLevel
    val ci = when (alternative) {
        Alternative.TWO_SIDED -> {
            val tCrit = dist.quantile(1.0 - alpha / 2.0)
            Pair(diff - tCrit * se, diff + tCrit * se)
        }
        Alternative.LESS -> {
            val tCrit = dist.quantile(1.0 - alpha)
            Pair(Double.NEGATIVE_INFINITY, diff + tCrit * se)
        }
        Alternative.GREATER -> {
            val tCrit = dist.quantile(1.0 - alpha)
            Pair(diff - tCrit * se, Double.POSITIVE_INFINITY)
        }
    }

    return TestResult(
        testName = if (equalVariances) "Two-Sample t-Test (Equal Variances)" else "Welch's t-Test",
        statistic = t,
        pValue = pValue.coerceIn(0.0, 1.0),
        degreesOfFreedom = df,
        alternative = alternative,
        confidenceInterval = ci,
        additionalInfo = mapOf("mean1" to mean1, "mean2" to mean2, "meanDifference" to diff)
    )
}

/**
 * Performs a paired t-test for whether the mean difference between matched observations is zero.
 *
 * The null hypothesis is that the true mean difference between the paired observations is zero.
 * Internally computes the element-wise differences and delegates to a one-sample t-test on
 * those differences. This test is appropriate when the two samples are not independent — for
 * example, before/after measurements on the same subjects.
 *
 * ### Example:
 * ```kotlin
 * val before = doubleArrayOf(200.0, 190.0, 210.0, 180.0, 195.0)
 * val after = doubleArrayOf(190.0, 180.0, 195.0, 170.0, 185.0)
 * val result = pairedTTest(before, after)
 * result.statistic          // t-statistic
 * result.pValue             // p-value
 * result.confidenceInterval // 95% CI for the mean difference
 * ```
 *
 * @param sample1 the first set of observations (e.g. "before"). Must have at least 2 elements.
 * @param sample2 the second set of observations (e.g. "after"). Must have the same size as [sample1].
 * @param alternative the direction of the alternative hypothesis. Defaults to [Alternative.TWO_SIDED],
 * which tests whether the mean difference differs from zero in either direction.
 * @param confidenceLevel the confidence level for the confidence interval. Defaults to `0.95` (95%).
 * @return a [TestResult] containing the t-statistic, p-value, degrees of freedom (n-1),
 * and a confidence interval for the mean difference.
 */
public fun pairedTTest(
    sample1: DoubleArray,
    sample2: DoubleArray,
    alternative: Alternative = Alternative.TWO_SIDED,
    confidenceLevel: Double = 0.95
): TestResult {
    if (sample1.size != sample2.size) throw InvalidParameterException("Samples must have the same size")
    if (sample1.size < 2) throw InsufficientDataException("Samples must have at least 2 elements")
    if (confidenceLevel <= 0.0 || confidenceLevel >= 1.0) throw InvalidParameterException(
        "confidenceLevel must be in (0, 1), got $confidenceLevel"
    )

    val differences = DoubleArray(sample1.size) { sample1[it] - sample2[it] }
    val result = tTest(differences, mu = 0.0, alternative = alternative, confidenceLevel = confidenceLevel)

    return result.copy(testName = "Paired t-Test")
}
