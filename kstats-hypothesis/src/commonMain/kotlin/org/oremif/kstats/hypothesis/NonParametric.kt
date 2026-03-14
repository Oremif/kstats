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

// Royston AS R94 polynomial coefficients for Shapiro-Wilk test.
// Source: Royston P. (1995) "Remark AS R94", Applied Statistics 44(4), pp.547-551.
// Matching R's swilk.c implementation (GPL-2+, based on AS181/R94).

// Polynomial for a[n-1] correction, n >= 4, evaluated at 1/sqrt(n)
private val SW_C1 = doubleArrayOf(0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056)

// Polynomial for a[n-2] correction, n >= 6, evaluated at 1/sqrt(n)
private val SW_C2 = doubleArrayOf(0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633)

// P-value mean polynomial for 4 <= n <= 11, evaluated at n
private val SW_C3 = doubleArrayOf(0.544, -0.39978, 0.025054, -6.714e-4)

// P-value log-sigma polynomial for 4 <= n <= 11, evaluated at n
private val SW_C4 = doubleArrayOf(1.3822, -0.77857, 0.062767, -0.0020322)

// P-value mean polynomial for n >= 12, evaluated at ln(n)
private val SW_C5 = doubleArrayOf(-1.5861, -0.31082, -0.083751, 0.0038915)

// P-value log-sigma polynomial for n >= 12, evaluated at ln(n)
private val SW_C6 = doubleArrayOf(-0.4803, -0.082676, 0.0030302)

// Gamma polynomial for p-value transform, 4 <= n <= 11, evaluated at n
private val SW_G = doubleArrayOf(-2.273, 0.459)

/**
 * Evaluate polynomial with coefficients in ascending power order using Horner's method.
 * Computes: coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ... + coeffs[n-1]*x^(n-1)
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
 * Compute Shapiro-Wilk coefficients using Royston AS R94 algorithm.
 * Returns a full antisymmetric coefficient array of size n: `a[i] = -a[n-1-i]`.
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
 * Compute Shapiro-Wilk p-value using Royston AS R94 approximation.
 * Three different transforms depending on n: n=3 (exact), 4<=n<=11, n>=12.
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
 * Shapiro-Wilk test for normality using Royston's AS R94 algorithm.
 *
 * Tests the null hypothesis that the data was drawn from a normal distribution.
 * Valid for sample sizes 3 <= n <= 5000.
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
