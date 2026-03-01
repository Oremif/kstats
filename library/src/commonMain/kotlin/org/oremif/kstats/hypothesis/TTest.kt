package org.oremif.kstats.hypothesis

import org.oremif.kstats.descriptive.mean
import org.oremif.kstats.descriptive.standardDeviation
import org.oremif.kstats.distributions.StudentTDistribution
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * One-sample t-test: tests whether the population mean equals [mu].
 */
public fun tTest(
    sample: DoubleArray,
    mu: Double = 0.0,
    alternative: Alternative = Alternative.TWO_SIDED,
    confidenceLevel: Double = 0.95
): TestResult {
    require(sample.size >= 2) { "Sample must have at least 2 elements" }

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
    val tCrit = dist.quantile(1.0 - alpha / 2.0)
    val ci = Pair(mean - tCrit * se, mean + tCrit * se)

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
 * Two-sample t-test. By default uses Welch's t-test (unequal variances).
 */
public fun tTest(
    sample1: DoubleArray,
    sample2: DoubleArray,
    alternative: Alternative = Alternative.TWO_SIDED,
    equalVariances: Boolean = false,
    confidenceLevel: Double = 0.95
): TestResult {
    require(sample1.size >= 2) { "Sample 1 must have at least 2 elements" }
    require(sample2.size >= 2) { "Sample 2 must have at least 2 elements" }

    val n1 = sample1.size.toDouble()
    val n2 = sample2.size.toDouble()
    val mean1 = sample1.mean()
    val mean2 = sample2.mean()
    val var1 = sample1.map { (it - mean1) * (it - mean1) }.sum() / (n1 - 1.0)
    val var2 = sample2.map { (it - mean2) * (it - mean2) }.sum() / (n2 - 1.0)

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

    val alpha = 1.0 - confidenceLevel
    val tCrit = dist.quantile(1.0 - alpha / 2.0)
    val diff = mean1 - mean2
    val ci = Pair(diff - tCrit * se, diff + tCrit * se)

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
 * Paired t-test.
 */
public fun pairedTTest(
    sample1: DoubleArray,
    sample2: DoubleArray,
    alternative: Alternative = Alternative.TWO_SIDED,
    confidenceLevel: Double = 0.95
): TestResult {
    require(sample1.size == sample2.size) { "Samples must have the same size" }
    require(sample1.size >= 2) { "Samples must have at least 2 elements" }

    val differences = DoubleArray(sample1.size) { sample1[it] - sample2[it] }
    val result = tTest(differences, mu = 0.0, alternative = alternative, confidenceLevel = confidenceLevel)

    return result.copy(testName = "Paired t-Test")
}
