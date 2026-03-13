package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.DegenerateDataException
import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.distributions.ContinuousDistribution
import org.oremif.kstats.distributions.NormalDistribution
import org.oremif.kstats.sampling.rank
import org.oremif.kstats.sampling.TieMethod
import kotlin.math.*

/**
 * Mann-Whitney U test (Wilcoxon rank-sum test).
 */
public fun mannWhitneyUTest(
    sample1: DoubleArray,
    sample2: DoubleArray,
    alternative: Alternative = Alternative.TWO_SIDED
): TestResult {
    if (sample1.isEmpty() || sample2.isEmpty()) throw InsufficientDataException("Samples must not be empty")

    val n1 = sample1.size
    val n2 = sample2.size

    // Combine and rank
    val combined = sample1 + sample2
    val ranks = combined.rank(TieMethod.AVERAGE)

    // Sum of ranks for sample 1
    val r1 = ranks.take(n1).sum()
    val u1 = r1 - n1 * (n1 + 1.0) / 2.0
    val u2 = n1.toDouble() * n2 - u1
    val u = minOf(u1, u2)

    // Normal approximation (for n > 10)
    val mu = n1.toDouble() * n2 / 2.0
    val sigma = sqrt(n1.toDouble() * n2 * (n1 + n2 + 1) / 12.0)
    val z = (u1 - mu) / sigma

    val normal = NormalDistribution.STANDARD
    val pValue = when (alternative) {
        Alternative.TWO_SIDED -> 2.0 * normal.sf(abs(z))
        Alternative.LESS -> normal.cdf(z)
        Alternative.GREATER -> normal.sf(z)
    }

    return TestResult(
        testName = "Mann-Whitney U Test",
        statistic = u,
        pValue = pValue.coerceIn(0.0, 1.0),
        alternative = alternative,
        additionalInfo = mapOf("U1" to u1, "U2" to u2, "z" to z)
    )
}

/**
 * Wilcoxon signed-rank test.
 * One-sample: tests whether the median differs from zero.
 * Two-sample (paired): tests whether the median difference is zero.
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

    // Remove zeros
    val nonZero = diffs.filter { it != 0.0 }
    if (nonZero.isEmpty()) throw DegenerateDataException("All differences are zero")
    val n = nonZero.size

    // Rank absolute values
    val absDiffs = nonZero.map { abs(it) }.toDoubleArray()
    val ranks = absDiffs.rank(TieMethod.AVERAGE)

    // Signed ranks
    var wPlus = 0.0
    var wMinus = 0.0
    for (i in nonZero.indices) {
        if (nonZero[i] > 0) wPlus += ranks[i] else wMinus += ranks[i]
    }

    val w = wPlus

    // Normal approximation
    val mu = n * (n + 1.0) / 4.0
    val sigma = sqrt(n * (n + 1.0) * (2.0 * n + 1.0) / 24.0)
    val z = (w - mu) / sigma

    val normal = NormalDistribution.STANDARD
    val pValue = when (alternative) {
        Alternative.TWO_SIDED -> 2.0 * normal.sf(abs(z))
        Alternative.LESS -> normal.cdf(z)
        Alternative.GREATER -> normal.sf(z)
    }

    return TestResult(
        testName = "Wilcoxon Signed-Rank Test",
        statistic = w,
        pValue = pValue.coerceIn(0.0, 1.0),
        alternative = alternative,
        additionalInfo = mapOf("wPlus" to wPlus, "wMinus" to wMinus, "z" to z)
    )
}

/**
 * One-sample Kolmogorov-Smirnov test: tests if sample comes from the given distribution.
 */
public fun kolmogorovSmirnovTest(
    sample: DoubleArray,
    distribution: ContinuousDistribution
): TestResult {
    if (sample.isEmpty()) throw InsufficientDataException("Sample must not be empty")

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

    val pValue = kolmogorovSmirnovPValue(d, n)

    return TestResult(
        testName = "Kolmogorov-Smirnov Test (One-Sample)",
        statistic = d,
        pValue = pValue.coerceIn(0.0, 1.0),
        additionalInfo = mapOf("dPlus" to dPlus, "dMinus" to dMinus)
    )
}

/**
 * Two-sample Kolmogorov-Smirnov test.
 */
public fun kolmogorovSmirnovTest(
    sample1: DoubleArray,
    sample2: DoubleArray
): TestResult {
    if (sample1.isEmpty() || sample2.isEmpty()) throw InsufficientDataException("Samples must not be empty")

    val n1 = sample1.size
    val n2 = sample2.size
    val sorted1 = sample1.sortedArray()
    val sorted2 = sample2.sortedArray()

    var d = 0.0
    var i = 0
    var j = 0
    while (i < n1 && j < n2) {
        val cdf1 = (i + 1).toDouble() / n1
        val cdf2 = (j + 1).toDouble() / n2
        if (sorted1[i] <= sorted2[j]) {
            d = max(d, abs(cdf1 - j.toDouble() / n2))
            i++
        } else {
            d = max(d, abs(i.toDouble() / n1 - cdf2))
            j++
        }
    }
    while (i < n1) {
        d = max(d, abs((i + 1).toDouble() / n1 - 1.0))
        i++
    }
    while (j < n2) {
        d = max(d, abs(1.0 - (j + 1).toDouble() / n2))
        j++
    }

    val en = sqrt(n1.toDouble() * n2 / (n1 + n2))
    val pValue = kolmogorovSmirnovPValue(d, en.toInt().coerceAtLeast(1))

    return TestResult(
        testName = "Kolmogorov-Smirnov Test (Two-Sample)",
        statistic = d,
        pValue = pValue.coerceIn(0.0, 1.0)
    )
}

/**
 * Shapiro-Wilk test for normality (simplified version for n <= 5000).
 */
public fun shapiroWilkTest(sample: DoubleArray): TestResult {
    val n = sample.size
    if (n < 3) throw InsufficientDataException("Shapiro-Wilk test requires at least 3 elements")

    val sorted = sample.sortedArray()
    val mean = sorted.average()

    // Compute S^2
    var s2 = 0.0
    for (x in sorted) {
        s2 += (x - mean) * (x - mean)
    }

    if (s2 == 0.0) {
        return TestResult(
            testName = "Shapiro-Wilk Test",
            statistic = 1.0,
            pValue = 1.0
        )
    }

    // Compute the Shapiro-Wilk coefficients using normal order statistics approximation
    val normal = NormalDistribution.STANDARD
    val m = DoubleArray(n) { i -> normal.quantile((i + 1.0 - 0.375) / (n + 0.25)) }
    var mSumSq = 0.0
    for (mi in m) mSumSq += mi * mi
    val c = 1.0 / sqrt(mSumSq)
    val a = DoubleArray(n) { i -> c * m[i] }

    // Compute W
    var numerator = 0.0
    for (i in 0 until n) {
        numerator += a[i] * sorted[i]
    }
    val w = (numerator * numerator) / s2

    // Approximate p-value using normal transformation (Royston's approximation, simplified)
    val lnN = kotlin.math.ln(n.toDouble())
    val mu1 = -1.2725 + 1.0521 * lnN
    val sigma1 = 1.0308 - 0.26758 * lnN
    val z = (kotlin.math.ln(1.0 - w) - mu1) / sigma1
    val pValue = NormalDistribution.STANDARD.sf(z)

    return TestResult(
        testName = "Shapiro-Wilk Test",
        statistic = w,
        pValue = pValue.coerceIn(0.0, 1.0)
    )
}

/**
 * Approximation of Kolmogorov-Smirnov p-value.
 */
private fun kolmogorovSmirnovPValue(d: Double, n: Int): Double {
    val sqrtN = sqrt(n.toDouble())
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
