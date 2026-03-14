package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.distributions.ChiSquaredDistribution
import kotlin.math.ln

/**
 * Performs Bartlett's test for equality of variances across two or more groups.
 *
 * The null hypothesis is that all groups have equal variances (homoscedasticity).
 * The test compares the pooled variance to the individual group variances using a
 * log-likelihood ratio with a correction factor for small samples. The test statistic
 * follows a chi-squared distribution with k - 1 degrees of freedom. Assumes that the
 * data in each group are normally distributed. For non-normal data, consider using
 * [leveneTest] or [flignerKilleenTest] instead, which are more robust to departures
 * from normality.
 *
 * ### Example:
 * ```kotlin
 * val g1 = doubleArrayOf(10.0, 11.0, 12.0, 9.0, 10.0)
 * val g2 = doubleArrayOf(5.0, 15.0, 10.0, 20.0, 0.0)
 * val result = bartlettTest(g1, g2)
 * result.statistic                        // chi-squared test statistic
 * result.pValue                           // p-value from chi-squared distribution
 * result.degreesOfFreedom                 // k - 1
 * result.additionalInfo["pooledVariance"] // pooled sample variance
 * ```
 *
 * @param groups two or more groups of observations, each with at least 2 elements.
 * @return a [TestResult] containing the test statistic, p-value, degrees of freedom (k - 1),
 * and additional info with "pooledVariance".
 */
public fun bartlettTest(vararg groups: DoubleArray): TestResult {
    if (groups.size < 2) throw InsufficientDataException(
        "Bartlett's test requires at least 2 groups, got ${groups.size}"
    )
    for (i in groups.indices) {
        if (groups[i].size < 2) throw InsufficientDataException(
            "Each group must have at least 2 elements, group $i has ${groups[i].size}"
        )
    }

    val k = groups.size
    val df = k - 1

    // Compute group sizes and total N
    val sizes = IntArray(k) { groups[it].size }
    val totalN = sizes.sum()

    // Compute group sample variances (Bessel-corrected)
    val variances = DoubleArray(k) { i ->
        val group = groups[i]
        val n = group.size
        var sum = 0.0
        for (j in 0 until n) sum += group[j]
        val mean = sum / n
        var ss = 0.0
        for (j in 0 until n) {
            val diff = group[j] - mean
            ss += diff * diff
        }
        ss / (n - 1)
    }

    // Degenerate case: all variances are zero (all groups constant)
    if (variances.all { it == 0.0 }) {
        return TestResult(
            testName = "Bartlett's Test",
            statistic = 0.0,
            pValue = 1.0,
            degreesOfFreedom = df.toDouble(),
            additionalInfo = mapOf("pooledVariance" to 0.0)
        )
    }

    // Pooled variance: Sp² = Σ(n_i - 1) * S_i² / (N - k)
    var pooledNum = 0.0
    for (i in 0 until k) {
        pooledNum += (sizes[i] - 1) * variances[i]
    }
    val pooledVariance = pooledNum / (totalN - k)

    // Numerator B = (N - k) * ln(Sp²) - Σ(n_i - 1) * ln(S_i²)
    var sumLogVar = 0.0
    for (i in 0 until k) {
        sumLogVar += (sizes[i] - 1) * ln(variances[i])
    }
    val b = (totalN - k) * ln(pooledVariance) - sumLogVar

    // Correction factor C = 1 + (1 / (3(k-1))) * (Σ 1/(n_i - 1) - 1/(N - k))
    var sumReciprocal = 0.0
    for (i in 0 until k) {
        sumReciprocal += 1.0 / (sizes[i] - 1)
    }
    val c = 1.0 + (1.0 / (3.0 * df)) * (sumReciprocal - 1.0 / (totalN - k))

    // Test statistic T = B / C
    val t = b / c

    // Non-finite check
    if (t.isNaN() || t.isInfinite()) {
        return TestResult(
            testName = "Bartlett's Test",
            statistic = t,
            pValue = if (t.isInfinite() && t > 0) 0.0 else Double.NaN,
            degreesOfFreedom = df.toDouble(),
            additionalInfo = mapOf("pooledVariance" to pooledVariance)
        )
    }

    // P-value from chi-squared distribution with (k-1) degrees of freedom
    val pValue = ChiSquaredDistribution(df.toDouble()).sf(t)

    return TestResult(
        testName = "Bartlett's Test",
        statistic = t,
        pValue = pValue.coerceIn(0.0, 1.0),
        degreesOfFreedom = df.toDouble(),
        additionalInfo = mapOf("pooledVariance" to pooledVariance)
    )
}
