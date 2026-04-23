package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.DegenerateDataException
import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.distributions.NormalDistribution
import org.oremif.kstats.sampling.TieMethod
import org.oremif.kstats.sampling.rank
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Performs the Mann-Whitney U test (also known as the Wilcoxon rank-sum test).
 *
 * The null hypothesis is that the two samples are drawn from the same distribution.
 * This is a non-parametric test that does not assume normality — it compares the ranks
 * of the combined samples rather than the raw values. Uses a normal approximation for
 * computing the p-value.
 *
 * ### Example:
 * ```kotlin
 * val control = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
 * val treatment = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
 * val result = mannWhitneyUTest(control, treatment)
 * result.statistic              // U statistic (minimum of U1 and U2)
 * result.pValue                 // p-value
 * result.additionalInfo["U1"]   // U statistic for sample 1
 * result.additionalInfo["U2"]   // U statistic for sample 2
 * result.additionalInfo["z"]    // z-score from normal approximation
 * ```
 *
 * @param sample1 the first sample. Must not be empty.
 * @param sample2 the second sample. Must not be empty.
 * @param alternative the direction of the alternative hypothesis. Defaults to [Alternative.TWO_SIDED].
 * @return a [TestResult] containing the U statistic (minimum of U1 and U2), p-value,
 * and additional info with "U1", "U2", and "z".
 */
public fun mannWhitneyUTest(
    sample1: DoubleArray,
    sample2: DoubleArray,
    alternative: Alternative = Alternative.TWO_SIDED
): TestResult {
    if (sample1.isEmpty() || sample2.isEmpty()) throw InsufficientDataException("Samples must not be empty")

    val n1 = sample1.size
    val n2 = sample2.size

    val n = n1 + n2

    // Combine and rank
    val combined = sample1 + sample2
    val ranks = combined.rank(TieMethod.AVERAGE)

    // Sum of ranks for sample 1
    var r1 = 0.0
    for (i in 0 until n1) r1 += ranks[i]
    val u1 = r1 - n1 * (n1 + 1.0) / 2.0
    val u2 = n1.toDouble() * n2 - u1

    // Normal approximation with tie correction
    val mu = n1.toDouble() * n2 / 2.0

    // Compute tie correction: sum(t^3 - t) for each group of tied values
    val sorted = combined.copyOf().also { it.sort() }
    var tieCorrection = 0.0
    var ti = 0
    while (ti < n) {
        var tj = ti + 1
        while (tj < n && sorted[tj] == sorted[ti]) tj++
        val t = (tj - ti).toDouble()
        tieCorrection += t * t * t - t
        ti = tj
    }
    val sigma = sqrt(n1.toDouble() * n2 / 12.0 * ((n + 1) - tieCorrection / (n.toDouble() * (n - 1))))
    val z = if (sigma == 0.0) 0.0 else (u1 - mu) / sigma

    // Continuity correction (matches scipy default use_continuity=True)
    val normal = NormalDistribution.STANDARD
    val pValue = when (alternative) {
        Alternative.TWO_SIDED -> {
            val zc = if (sigma == 0.0) 0.0 else (abs(u1 - mu) - 0.5).coerceAtLeast(0.0) / sigma
            2.0 * normal.sf(zc)
        }
        Alternative.LESS -> {
            val zc = if (sigma == 0.0) 0.0 else (u1 - mu + 0.5) / sigma
            normal.cdf(zc)
        }
        Alternative.GREATER -> {
            val zc = if (sigma == 0.0) 0.0 else (u1 - mu - 0.5) / sigma
            normal.sf(zc)
        }
    }

    return TestResult(
        testName = "Mann-Whitney U Test",
        statistic = u1,
        pValue = pValue.coerceIn(0.0, 1.0),
        alternative = alternative,
        additionalInfo = mapOf("U1" to u1, "U2" to u2, "z" to z)
    )
}

/**
 * Performs the Wilcoxon signed-rank test.
 *
 * In one-sample mode (when [sample2] is `null`), tests whether the median of [sample1]
 * differs from zero. In paired mode (when [sample2] is provided), tests whether the median
 * of the paired differences is zero. This is a non-parametric alternative to the paired
 * t-test that does not assume normality. Uses a normal approximation for computing the p-value.
 *
 * Zero differences are removed before ranking. The test statistic W is the sum of positive
 * signed ranks.
 *
 * ### Example:
 * ```kotlin
 * val before = doubleArrayOf(10.0, 12.0, 14.0, 16.0, 18.0)
 * val after = doubleArrayOf(8.0, 9.0, 11.0, 12.0, 13.0)
 * val result = wilcoxonSignedRankTest(before, after)
 * result.statistic                 // W+ (sum of positive ranks)
 * result.pValue                    // p-value
 * result.additionalInfo["wPlus"]   // sum of positive signed ranks
 * result.additionalInfo["wMinus"]  // sum of negative signed ranks
 * result.additionalInfo["z"]       // z-score from normal approximation
 * ```
 *
 * @param sample1 the first sample, or the only sample in one-sample mode.
 * @param sample2 the second sample for paired mode. Must have the same size as [sample1]
 * if provided. Defaults to `null` (one-sample mode).
 * @param alternative the direction of the alternative hypothesis. Defaults to [Alternative.TWO_SIDED].
 * @return a [TestResult] containing the W+ statistic, p-value, and additional info
 * with "wPlus", "wMinus", and "z".
 * @throws DegenerateDataException if all differences are zero after pairing or in one-sample mode.
 */
public fun wilcoxonSignedRankTest(
    sample1: DoubleArray,
    sample2: DoubleArray? = null,
    alternative: Alternative = Alternative.TWO_SIDED
): TestResult {
    val diffs = if (sample2 != null) {
        if (sample1.size != sample2.size) throw InvalidParameterException("Samples must have the same size")
        DoubleArray(sample1.size) { sample1[it] - sample2[it] }
    } else {
        sample1
    }

    // Remove zeros and extract absolute values + signs in one pass
    var nonZeroCount = 0
    for (d in diffs) {
        if (d != 0.0) nonZeroCount++
    }
    if (nonZeroCount == 0) throw DegenerateDataException("All differences are zero")
    val n = nonZeroCount

    val absDiffs = DoubleArray(n)
    val positive = BooleanArray(n)
    var idx = 0
    for (d in diffs) {
        if (d != 0.0) {
            absDiffs[idx] = abs(d)
            positive[idx] = d > 0
            idx++
        }
    }

    // Rank absolute values
    val ranks = absDiffs.rank(TieMethod.AVERAGE)

    // Signed ranks
    var wPlus = 0.0
    var wMinus = 0.0
    for (i in 0 until n) {
        if (positive[i]) wPlus += ranks[i] else wMinus += ranks[i]
    }

    val w = wPlus

    // Normal approximation with tie correction
    val mu = n * (n + 1.0) / 4.0

    // Compute tie correction: sum(t*(t-1)*(2t+1)) for each group of tied absolute differences
    val sortedAbs = absDiffs.copyOf().also { it.sort() }
    var tieCorrection = 0.0
    var ti = 0
    while (ti < n) {
        var tj = ti + 1
        while (tj < n && sortedAbs[tj] == sortedAbs[ti]) tj++
        val t = (tj - ti).toDouble()
        tieCorrection += t * (t - 1.0) * (2.0 * t + 1.0)
        ti = tj
    }
    val sigma = sqrt(n * (n + 1.0) * (2.0 * n + 1.0) / 24.0 - tieCorrection / 48.0)
    val z = if (sigma == 0.0) 0.0 else (w - mu) / sigma

    // Continuity correction (matches scipy default correction=True)
    val normal = NormalDistribution.STANDARD
    val pValue = when (alternative) {
        Alternative.TWO_SIDED -> {
            val zc = if (sigma == 0.0) 0.0 else (abs(w - mu) - 0.5).coerceAtLeast(0.0) / sigma
            2.0 * normal.sf(zc)
        }
        Alternative.LESS -> {
            val zc = if (sigma == 0.0) 0.0 else (w - mu + 0.5) / sigma
            normal.cdf(zc)
        }
        Alternative.GREATER -> {
            val zc = if (sigma == 0.0) 0.0 else (w - mu - 0.5) / sigma
            normal.sf(zc)
        }
    }

    return TestResult(
        testName = "Wilcoxon Signed-Rank Test",
        statistic = w,
        pValue = pValue.coerceIn(0.0, 1.0),
        alternative = alternative,
        additionalInfo = mapOf("wPlus" to wPlus, "wMinus" to wMinus, "z" to z)
    )
}
