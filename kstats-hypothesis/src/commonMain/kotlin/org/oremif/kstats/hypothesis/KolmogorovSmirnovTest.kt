package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.distributions.ContinuousDistribution
import kotlin.math.*

/**
 * Performs a one-sample Kolmogorov-Smirnov test against a reference distribution.
 *
 * The null hypothesis is that [sample] was drawn from [distribution]. The test computes the
 * maximum absolute difference between the empirical cumulative distribution function (ECDF)
 * of the sample and the theoretical CDF of the reference distribution. Uses Kolmogorov's
 * asymptotic formula for the p-value approximation.
 *
 * ### Example:
 * ```kotlin
 * val sample = doubleArrayOf(-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5, -0.3, 0.3, 0.8)
 * val result = kolmogorovSmirnovTest(sample, NormalDistribution.STANDARD)
 * result.statistic                // D statistic (max ECDF-CDF deviation)
 * result.pValue                   // p-value
 * result.additionalInfo["dPlus"]  // max(ECDF - CDF)
 * result.additionalInfo["dMinus"] // max(CDF - ECDF)
 * ```
 *
 * @param sample the observed values. Must not be empty.
 * @param distribution the reference continuous distribution to test against.
 * @return a [TestResult] containing the D statistic, p-value, and additional info
 * with "dPlus" and "dMinus".
 */
public fun kolmogorovSmirnovTest(
    sample: DoubleArray,
    distribution: ContinuousDistribution
): TestResult {
    if (sample.isEmpty()) throw InsufficientDataException("Sample must not be empty")

    if (sample.any { !it.isFinite() }) {
        return TestResult(
            testName = "Kolmogorov-Smirnov Test (One-Sample)",
            statistic = Double.NaN,
            pValue = Double.NaN,
            additionalInfo = mapOf("dPlus" to Double.NaN, "dMinus" to Double.NaN)
        )
    }

    val n = sample.size
    val sorted = sample.sortedArray()

    var dPlus = 0.0
    var dMinus = 0.0
    for (i in sorted.indices) {
        val cdf = distribution.cdf(sorted[i])
        val eUp = (i + 1).toDouble() / n
        val eDown = i.toDouble() / n
        dPlus = max(dPlus, eUp - cdf)
        dMinus = max(dMinus, cdf - eDown)
    }
    val d = max(dPlus, dMinus)

    val pValue = kolmogorovSmirnovPValue(d, n.toDouble())

    return TestResult(
        testName = "Kolmogorov-Smirnov Test (One-Sample)",
        statistic = d,
        pValue = pValue.coerceIn(0.0, 1.0),
        additionalInfo = mapOf("dPlus" to dPlus, "dMinus" to dMinus)
    )
}

/**
 * Performs a two-sample Kolmogorov-Smirnov test.
 *
 * The null hypothesis is that [sample1] and [sample2] are drawn from the same distribution.
 * The test computes the maximum absolute difference between the two empirical cumulative
 * distribution functions. Uses Kolmogorov's asymptotic formula for the p-value approximation
 * with an effective sample size derived from both sample sizes.
 *
 * ### Example:
 * ```kotlin
 * val s1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
 * val s2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
 * val result = kolmogorovSmirnovTest(s1, s2)
 * result.statistic // D statistic (max ECDF difference)
 * result.pValue    // p-value
 * ```
 *
 * @param sample1 the first sample. Must not be empty.
 * @param sample2 the second sample. Must not be empty.
 * @return a [TestResult] containing the D statistic and p-value.
 */
public fun kolmogorovSmirnovTest(
    sample1: DoubleArray,
    sample2: DoubleArray
): TestResult {
    if (sample1.isEmpty() || sample2.isEmpty()) throw InsufficientDataException("Samples must not be empty")

    if (sample1.any { !it.isFinite() } || sample2.any { !it.isFinite() }) {
        return TestResult(
            testName = "Kolmogorov-Smirnov Test (Two-Sample)",
            statistic = Double.NaN,
            pValue = Double.NaN
        )
    }

    val n1 = sample1.size
    val n2 = sample2.size
    val sorted1 = sample1.sortedArray()
    val sorted2 = sample2.sortedArray()

    var d = 0.0
    var i = 0
    var j = 0
    while (i < n1 || j < n2) {
        val v1 = if (i < n1) sorted1[i] else Double.POSITIVE_INFINITY
        val v2 = if (j < n2) sorted2[j] else Double.POSITIVE_INFINITY
        val v = min(v1, v2)
        // Advance past all values equal to v in both samples
        while (i < n1 && sorted1[i] == v) i++
        while (j < n2 && sorted2[j] == v) j++
        d = max(d, abs(i.toDouble() / n1 - j.toDouble() / n2))
    }

    val ne = n1.toDouble() * n2 / (n1 + n2)
    val pValue = kolmogorovSmirnovPValue(d, ne)

    return TestResult(
        testName = "Kolmogorov-Smirnov Test (Two-Sample)",
        statistic = d,
        pValue = pValue.coerceIn(0.0, 1.0)
    )
}

/**
 * Approximates the Kolmogorov-Smirnov p-value using Kolmogorov's asymptotic series.
 *
 * Applies a continuity correction to the D statistic and evaluates the alternating
 * series until convergence (term < 1e-12) or 100 terms.
 */
private fun kolmogorovSmirnovPValue(d: Double, n: Double): Double {
    val sqrtN = sqrt(n)
    val z = (sqrtN + 0.12 + 0.11 / sqrtN) * d

    if (z < 0.27) return 1.0
    if (z > 3.1) return 0.0

    // Kolmogorov's asymptotic formula
    var sum = 0.0
    for (k in 1..100) {
        val sign = if (k % 2 == 1) 1.0 else -1.0
        val term = sign * exp(-2.0 * k * k * z * z)
        sum += term
        if (abs(term) < 1e-12) break
    }
    return (2.0 * sum).coerceIn(0.0, 1.0)
}
