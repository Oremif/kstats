package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.DegenerateDataException
import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.distributions.ChiSquaredDistribution
import org.oremif.kstats.distributions.ContinuousDistribution
import org.oremif.kstats.distributions.NormalDistribution
import org.oremif.kstats.sampling.TieMethod
import org.oremif.kstats.sampling.rank
import kotlin.math.*

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

// Royston AS R94 polynomial coefficients for Shapiro-Wilk test.
// Source: Royston P. (1995) "Remark AS R94", Applied Statistics 44(4), pp.547-551.
// Matching R's swilk.c implementation (GPL-2+, based on AS181/R94).

/** Polynomial for a[n-1] correction, n >= 4, evaluated at 1/sqrt(n). */
private val SW_C1 = doubleArrayOf(0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056)

/** Polynomial for a[n-2] correction, n >= 6, evaluated at 1/sqrt(n). */
private val SW_C2 = doubleArrayOf(0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633)

/** P-value mean polynomial for 4 <= n <= 11, evaluated at n. */
private val SW_C3 = doubleArrayOf(0.544, -0.39978, 0.025054, -6.714e-4)

/** P-value log-sigma polynomial for 4 <= n <= 11, evaluated at n. */
private val SW_C4 = doubleArrayOf(1.3822, -0.77857, 0.062767, -0.0020322)

/** P-value mean polynomial for n >= 12, evaluated at ln(n). */
private val SW_C5 = doubleArrayOf(-1.5861, -0.31082, -0.083751, 0.0038915)

/** P-value log-sigma polynomial for n >= 12, evaluated at ln(n). */
private val SW_C6 = doubleArrayOf(-0.4803, -0.082676, 0.0030302)

/** Gamma polynomial for p-value transform, 4 <= n <= 11, evaluated at n. */
private val SW_G = doubleArrayOf(-2.273, 0.459)

/**
 * Evaluates a polynomial with coefficients in ascending power order using Horner's method.
 *
 * Used internally by the Shapiro-Wilk implementation for Royston AS R94 polynomial
 * approximations.
 */
private fun swPoly(coeffs: DoubleArray, x: Double): Double {
    var result = coeffs[0]
    if (coeffs.size > 1) {
        var p = x * coeffs[coeffs.size - 1]
        for (j in coeffs.size - 2 downTo 1) {
            p = (p + coeffs[j]) * x
        }
        result += p
    }
    return result
}

/**
 * Computes Shapiro-Wilk coefficients using the Royston AS R94 algorithm.
 *
 * Returns a full antisymmetric coefficient array of size n where a[i] = -a[n-1-i].
 * The extreme coefficients are corrected via polynomial approximations; middle
 * coefficients are normalized expected normal order statistics.
 */
private fun shapiroWilkCoefficients(n: Int): DoubleArray {
    val nn2 = n / 2
    val a = DoubleArray(n)

    if (n == 3) {
        a[2] = 1.0 / sqrt(2.0)
        a[0] = -a[2]
        return a
    }

    val normal = NormalDistribution.STANDARD
    val an = n.toDouble()
    val an25 = an + 0.25

    // Expected normal order statistics (first half only, since m is antisymmetric)
    val m = DoubleArray(nn2) { i -> normal.quantile((i + 1.0 - 0.375) / an25) }
    val summ2 = 2.0 * m.sumOf { it * it }
    val ssumm2 = sqrt(summ2)
    val rsn = 1.0 / sqrt(an)

    // Corrected extreme coefficient from polynomial approximation
    val a1 = swPoly(SW_C1, rsn) - m[0] / ssumm2

    val fac: Double
    val i1: Int
    if (n > 5) {
        // Second extreme coefficient correction
        val a2 = -m[1] / ssumm2 + swPoly(SW_C2, rsn)
        fac = sqrt(
            (summ2 - 2.0 * m[0] * m[0] - 2.0 * m[1] * m[1]) /
                (1.0 - 2.0 * a1 * a1 - 2.0 * a2 * a2)
        )
        a[n - 2] = a2
        a[1] = -a2
        i1 = 2
    } else {
        // n = 4 or 5: only the most extreme coefficient is corrected
        fac = sqrt((summ2 - 2.0 * m[0] * m[0]) / (1.0 - 2.0 * a1 * a1))
        i1 = 1
    }

    // Set extreme coefficients
    a[n - 1] = a1
    a[0] = -a1

    // Middle coefficients: normalized expected order statistics
    for (i in i1 until nn2) {
        a[n - 1 - i] = m[i] / (-fac)
        a[i] = -a[n - 1 - i]
    }

    return a
}

/**
 * Computes the Shapiro-Wilk p-value using the Royston AS R94 approximation.
 *
 * Uses three different transforms depending on the sample size:
 * n=3 (exact via arcsine), 4<=n<=11 (gamma-log transform), n>=12 (log-normal transform).
 */
private fun shapiroWilkPValue(w: Double, n: Int): Double {
    val normal = NormalDistribution.STANDARD

    if (n == 3) {
        val pi6 = 6.0 / PI
        val stqr = asin(sqrt(0.75)) // pi/3
        return maxOf(0.0, pi6 * (asin(sqrt(w)) - stqr))
    }

    val w1 = 1.0 - w

    val y: Double
    val mu: Double
    val sigma: Double

    if (n <= 11) {
        val gamma = swPoly(SW_G, n.toDouble())
        val logW1 = ln(w1)
        if (logW1 >= gamma) return 0.0
        y = -ln(gamma - logW1)
        mu = swPoly(SW_C3, n.toDouble())
        sigma = exp(swPoly(SW_C4, n.toDouble()))
    } else {
        y = ln(w1)
        val lnN = ln(n.toDouble())
        mu = swPoly(SW_C5, lnN)
        sigma = exp(swPoly(SW_C6, lnN))
    }

    val z = (y - mu) / sigma
    return normal.sf(z)
}

/**
 * Performs the Shapiro-Wilk test for normality.
 *
 * The null hypothesis is that [sample] was drawn from a normal distribution. The W statistic
 * measures how well the ordered sample values match the expected normal order statistics —
 * values close to 1.0 indicate normality, while values significantly below 1.0 suggest
 * non-normality. Uses Royston's AS R94 algorithm for both the W statistic and p-value
 * approximation. Valid for sample sizes from 3 to 5000.
 *
 * If all values are identical (zero variance), returns W = 1.0 and p-value = 1.0.
 *
 * ### Example:
 * ```kotlin
 * val data = doubleArrayOf(-1.2, -0.5, 0.1, 0.3, 0.7, 1.0, 1.5)
 * val result = shapiroWilkTest(data)
 * result.statistic       // W statistic (~0.984 for this data)
 * result.pValue          // p-value (~0.978, fails to reject normality)
 * result.isSignificant() // false (data is consistent with normality)
 * ```
 *
 * @param sample the observed values. Must have between 3 and 5000 elements.
 * @return a [TestResult] containing the W statistic and p-value.
 */
public fun shapiroWilkTest(sample: DoubleArray): TestResult {
    val n = sample.size
    if (n < 3) throw InsufficientDataException("Shapiro-Wilk test requires at least 3 elements")
    if (n > 5000) throw InvalidParameterException("Shapiro-Wilk test requires at most 5000 elements")

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

    val a = shapiroWilkCoefficients(n)

    // Compute W statistic
    var numerator = 0.0
    for (i in 0 until n) {
        numerator += a[i] * sorted[i]
    }
    val w = (numerator * numerator / s2).coerceIn(0.0, 1.0)

    val pValue = shapiroWilkPValue(w, n)

    return TestResult(
        testName = "Shapiro-Wilk Test",
        statistic = w,
        pValue = pValue.coerceIn(0.0, 1.0)
    )
}

public fun andersonDarlingTest(sample: DoubleArray): TestResult {
    val n = sample.size
    if (n < 3) throw InsufficientDataException("Anderson-Darling test requires at least 3 elements")

    val sorted = sample.sortedArray()
    val mean = sorted.average()

    // Compute sum of squared deviations
    var ss = 0.0
    for (x in sorted) {
        ss += (x - mean) * (x - mean)
    }

    // Constant data: zero variance means normality cannot be assessed,
    // but constant data is trivially consistent with any location-scale family
    if (ss == 0.0) {
        return TestResult(
            testName = "Anderson-Darling Test",
            statistic = 0.0,
            pValue = 1.0,
            additionalInfo = mapOf("modifiedStatistic" to 0.0)
        )
    }

    val sd = sqrt(ss / (n - 1))
    val normal = NormalDistribution.STANDARD

    // Compute A² statistic:
    // A² = -n - (1/n) * Σ_{i=1}^{n} (2i-1) * [ln(Φ(z_i)) + ln(1 - Φ(z_{n+1-i}))]
    // Source: D'Agostino & Stephens (1986), "Goodness-of-Fit Techniques"
    var s = 0.0
    for (i in 0 until n) {
        val zi = (sorted[i] - mean) / sd
        val zni = (sorted[n - 1 - i] - mean) / sd
        // Use cdf for lower tail and sf (via erfc) for upper tail to preserve precision
        val logCdf = ln(normal.cdf(zi).coerceAtLeast(1e-308))
        val logSf = ln(normal.sf(zni).coerceAtLeast(1e-308))
        s += (2.0 * (i + 1) - 1.0) * (logCdf + logSf)
    }

    val a2 = -n.toDouble() - s / n

    // Modified statistic for finite sample correction
    val a2Star = a2 * (1.0 + 0.75 / n + 2.25 / (n.toDouble() * n))

    // P-value approximation: D'Agostino & Stephens (1986)
    val pValue = andersonDarlingPValue(a2Star)

    return TestResult(
        testName = "Anderson-Darling Test",
        statistic = a2,
        pValue = pValue.coerceIn(0.0, 1.0),
        additionalInfo = mapOf("modifiedStatistic" to a2Star)
    )
}

// P-value approximation for the modified Anderson-Darling statistic A²*.
// Source: D'Agostino & Stephens (1986), Table 4.9, case 3 (parameters estimated from data).
private fun andersonDarlingPValue(a2Star: Double): Double {
    if (a2Star.isNaN()) return Double.NaN
    return when {
        a2Star >= 0.6 -> exp(1.2937 - 5.709 * a2Star + 0.0186 * a2Star * a2Star)
        a2Star >= 0.34 -> exp(0.9177 - 4.279 * a2Star - 1.38 * a2Star * a2Star)
        a2Star >= 0.2 -> 1.0 - exp(-8.318 + 42.796 * a2Star - 59.938 * a2Star * a2Star)
        else -> 1.0 - exp(-13.436 + 101.14 * a2Star - 223.73 * a2Star * a2Star)
    }
}

/**
 * Approximates the Kolmogorov-Smirnov p-value using Kolmogorov's asymptotic series.
 *
 * Applies a continuity correction to the D statistic and evaluates the alternating
 * series until convergence (term < 1e-12) or 100 terms.
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

/**
 * Performs the D'Agostino-Pearson omnibus normality test.
 *
 * The null hypothesis is that [sample] was drawn from a normal distribution. The test
 * combines D'Agostino's skewness test (1970) and Anscombe & Glynn's kurtosis test (1983)
 * into a single chi-squared statistic K² = Z₁² + Z₂² with 2 degrees of freedom. This is
 * equivalent to scipy's `normaltest()`. Complements [shapiroWilkTest] and
 * [andersonDarlingTest] for assessing normality.
 *
 * If all values are identical (zero variance), returns K² = 0.0 and p-value = 1.0.
 *
 * ### Example:
 * ```kotlin
 * val data = doubleArrayOf(-1.2, -0.5, 0.1, 0.3, 0.7, 1.0, 1.5,
 *     0.2, -0.3, 0.8, -0.7, 0.4, 1.1, -0.1, 0.6, -0.9, 0.3, -0.2, 0.5, 1.3)
 * val result = dagostinoPearsonTest(data)
 * result.statistic                // K² statistic
 * result.pValue                   // p-value from chi-squared distribution with df=2
 * result.degreesOfFreedom         // 2.0
 * result.additionalInfo["z1"]     // skewness z-score
 * result.additionalInfo["z2"]     // kurtosis z-score
 * result.additionalInfo["skewness"]  // sample skewness (population)
 * result.additionalInfo["kurtosis"]  // sample kurtosis (population, non-excess)
 * result.isSignificant()          // true if data deviates significantly from normality
 * ```
 *
 * @param sample the observed values. Must have at least 20 elements.
 * @return a [TestResult] containing the K² statistic, p-value, degrees of freedom (2.0),
 * and additional info with "z1", "z2", "skewness", and "kurtosis".
 */
public fun dagostinoPearsonTest(sample: DoubleArray): TestResult {
    val n = sample.size
    if (n < 20) throw InsufficientDataException(
        "D'Agostino-Pearson test requires at least 20 elements, got $n"
    )

    // Compute population moments (two-pass)
    val mean = sample.average()
    var m2 = 0.0
    var m3 = 0.0
    var m4 = 0.0
    for (x in sample) {
        val d = x - mean
        val d2 = d * d
        m2 += d2
        m3 += d2 * d
        m4 += d2 * d2
    }
    m2 /= n
    m3 /= n
    m4 /= n

    // Constant data: zero variance
    if (m2 == 0.0) {
        return TestResult(
            testName = "D'Agostino-Pearson Test",
            statistic = 0.0,
            pValue = 1.0,
            degreesOfFreedom = 2.0,
            additionalInfo = mapOf("z1" to 0.0, "z2" to 0.0, "skewness" to 0.0, "kurtosis" to 0.0)
        )
    }

    val b1 = m3 / (m2 * sqrt(m2))   // population skewness
    val b2 = m4 / (m2 * m2)         // population kurtosis (non-excess)

    val z1 = skewTestZScore(b1, n)
    val z2 = kurtosisTestZScore(b2, n)

    val k2 = z1 * z1 + z2 * z2
    if (k2.isNaN() || k2.isInfinite()) {
        return TestResult(
            testName = "D'Agostino-Pearson Test",
            statistic = k2,
            pValue = Double.NaN,
            degreesOfFreedom = 2.0,
            additionalInfo = mapOf("z1" to z1, "z2" to z2, "skewness" to b1, "kurtosis" to b2)
        )
    }
    val pValue = ChiSquaredDistribution(2.0).sf(k2)

    return TestResult(
        testName = "D'Agostino-Pearson Test",
        statistic = k2,
        pValue = pValue.coerceIn(0.0, 1.0),
        degreesOfFreedom = 2.0,
        additionalInfo = mapOf("z1" to z1, "z2" to z2, "skewness" to b1, "kurtosis" to b2)
    )
}

/**
 * Computes the skewness z-score using D'Agostino's (1970) transformation.
 *
 * Transforms the sample skewness [b1] into a standard normal deviate Z₁ via
 * a log-sinh transformation that stabilizes the distribution for moderate n.
 */
private fun skewTestZScore(b1: Double, n: Int): Double {
    val an = n.toDouble()
    val y = b1 * sqrt((an + 1.0) * (an + 3.0) / (6.0 * (an - 2.0)))
    val beta2 = 3.0 * (an * an + 27.0 * an - 70.0) * (an + 1.0) * (an + 3.0) /
        ((an - 2.0) * (an + 5.0) * (an + 7.0) * (an + 9.0))
    val w2 = -1.0 + sqrt(2.0 * (beta2 - 1.0))
    val delta = 1.0 / sqrt(0.5 * ln(w2))
    val alpha = sqrt(2.0 / (w2 - 1.0))
    return delta * asinh(y / alpha)
}

/**
 * Computes the kurtosis z-score using Anscombe & Glynn's (1983) transformation.
 *
 * Transforms the sample kurtosis [b2] (non-excess, population) into a standard normal
 * deviate Z₂ via a cube-root transformation.
 */
private fun kurtosisTestZScore(b2: Double, n: Int): Double {
    val an = n.toDouble()
    val e = 3.0 * (an - 1.0) / (an + 1.0)
    val varB2 = 24.0 * an * (an - 2.0) * (an - 3.0) /
        ((an + 1.0) * (an + 1.0) * (an + 3.0) * (an + 5.0))
    val x = (b2 - e) / sqrt(varB2)
    val sqrtBeta1 = 6.0 * (an * an - 5.0 * an + 2.0) /
        ((an + 7.0) * (an + 9.0)) *
        sqrt(6.0 * (an + 3.0) * (an + 5.0) / (an * (an - 2.0) * (an - 3.0)))
    val a = 6.0 + 8.0 / sqrtBeta1 * (2.0 / sqrtBeta1 + sqrt(1.0 + 4.0 / (sqrtBeta1 * sqrtBeta1)))
    val denom = 1.0 + x * sqrt(2.0 / (a - 4.0))
    val term2 = denom.sign * ((1.0 - 2.0 / a) / abs(denom)).pow(1.0 / 3.0)
    return (1.0 - 2.0 / (9.0 * a) - term2) / sqrt(2.0 / (9.0 * a))
}
