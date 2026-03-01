package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.descriptive.median
import org.oremif.kstats.distributions.ChiSquaredDistribution
import org.oremif.kstats.distributions.NormalDistribution
import org.oremif.kstats.sampling.TieMethod
import org.oremif.kstats.sampling.rank
import kotlin.math.abs

/**
 * Performs the Fligner-Killeen test for equality of variances across two or more groups.
 *
 * The null hypothesis is that all groups have equal variances (homoscedasticity).
 * This is a non-parametric test that computes absolute deviations from group medians,
 * ranks them across all groups, transforms the ranks to normal scores, and then tests
 * whether the group mean scores differ. It is the most robust of the three variance
 * homogeneity tests in kstats (along with [leveneTest] and [bartlettTest]), especially
 * for data from heavy-tailed distributions. The test statistic follows a chi-squared
 * distribution with k - 1 degrees of freedom under the null hypothesis.
 *
 * ### Example:
 * ```kotlin
 * val g1 = doubleArrayOf(10.0, 11.0, 12.0, 9.0, 10.0)
 * val g2 = doubleArrayOf(5.0, 15.0, 10.0, 20.0, 0.0)
 * val result = flignerKilleenTest(g1, g2)
 * result.statistic        // chi-squared test statistic
 * result.pValue           // p-value from chi-squared distribution
 * result.degreesOfFreedom // k - 1
 * ```
 *
 * @param groups two or more groups of observations, each with at least 2 elements.
 * @return a [TestResult] containing the chi-squared statistic, p-value, and degrees of freedom (k - 1).
 */
public fun flignerKilleenTest(vararg groups: DoubleArray): TestResult {
    if (groups.size < 2) throw InsufficientDataException(
        "Fligner-Killeen test requires at least 2 groups, got ${groups.size}"
    )
    for (i in groups.indices) {
        if (groups[i].size < 2) throw InsufficientDataException(
            "Each group must have at least 2 elements, group $i has ${groups[i].size}"
        )
    }

    val k = groups.size
    val df = k - 1

    // Step 1: Compute absolute deviations from group medians
    val deviations = Array(k) { i ->
        val group = groups[i]
        val med = group.median()
        DoubleArray(group.size) { j -> abs(group[j] - med) }
    }

    // Step 2: Combine all deviations into a single array
    val groupSizes = IntArray(k) { deviations[it].size }
    val totalN = groupSizes.sum()

    val allDeviations = DoubleArray(totalN)
    var offset = 0
    for (i in 0 until k) {
        deviations[i].copyInto(allDeviations, offset)
        offset += deviations[i].size
    }

    // Non-finite check: NaN or Infinity in input propagates through deviations
    if (allDeviations.any { !it.isFinite() }) {
        return TestResult(
            testName = "Fligner-Killeen Test",
            statistic = Double.NaN,
            pValue = Double.NaN,
            degreesOfFreedom = df.toDouble()
        )
    }

    // Degenerate case: all deviations identical → all scores equal → stat=0, p=1
    // Check before ranking to avoid floating-point noise in score computation
    val allSame = allDeviations.all { it == allDeviations[0] }
    if (allSame) {
        return TestResult(
            testName = "Fligner-Killeen Test",
            statistic = 0.0,
            pValue = 1.0,
            degreesOfFreedom = df.toDouble()
        )
    }

    // Step 3: Rank and transform to normal scores
    val ranks = allDeviations.rank(TieMethod.AVERAGE)
    val normal = NormalDistribution.STANDARD
    val scores = DoubleArray(totalN) { j ->
        normal.quantile(ranks[j] / (2.0 * (totalN + 1)) + 0.5)
    }

    // Step 4: Compute group means of scores and grand mean
    var grandSum = 0.0
    for (s in scores) grandSum += s
    val grandMean = grandSum / totalN

    offset = 0
    val groupMeans = DoubleArray(k) { i ->
        var sum = 0.0
        for (j in 0 until groupSizes[i]) {
            sum += scores[offset + j]
        }
        offset += groupSizes[i]
        sum / groupSizes[i]
    }

    // Step 5: Sample variance of all scores (ddof=1, matching scipy)
    var sumSqDev = 0.0
    for (s in scores) {
        val d = s - grandMean
        sumSqDev += d * d
    }
    val variance = sumSqDev / (totalN - 1)

    // Chi-squared statistic
    var numerator = 0.0
    for (i in 0 until k) {
        val d = groupMeans[i] - grandMean
        numerator += groupSizes[i] * d * d
    }
    val stat = numerator / variance

    // Non-finite check
    if (stat.isNaN() || stat.isInfinite()) {
        return TestResult(
            testName = "Fligner-Killeen Test",
            statistic = stat,
            pValue = if (stat.isInfinite() && stat > 0) 0.0 else Double.NaN,
            degreesOfFreedom = df.toDouble()
        )
    }

    val pValue = ChiSquaredDistribution(df.toDouble()).sf(stat)

    return TestResult(
        testName = "Fligner-Killeen Test",
        statistic = stat,
        pValue = pValue.coerceIn(0.0, 1.0),
        degreesOfFreedom = df.toDouble()
    )
}
