package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.descriptive.mean
import org.oremif.kstats.descriptive.median
import org.oremif.kstats.distributions.FDistribution
import kotlin.math.abs

/**
 * Specifies which center statistic to use when computing deviations in [leveneTest].
 *
 * The choice of center affects the test's sensitivity to non-normality:
 * - [MEAN] gives the classic Levene's test, which is more powerful under normality.
 * - [MEDIAN] gives the Brown-Forsythe variant, which is more robust to non-normal data
 *   and is the recommended default (matching scipy's default).
 */
public enum class LeveneCenter {
    /** Classic Levene's test: deviations from the group mean. */
    MEAN,
    /** Brown-Forsythe variant: deviations from the group median (more robust). */
    MEDIAN
}

/**
 * Performs Levene's test for equality of variances across two or more groups.
 *
 * The null hypothesis is that all groups have equal variances (homoscedasticity).
 * The test transforms each observation to its absolute deviation from the group center
 * (mean or median), then performs a one-way ANOVA on the transformed values.
 *
 * When [center] is [LeveneCenter.MEDIAN], this is the Brown-Forsythe variant, which is
 * more robust against non-normal data. When [center] is [LeveneCenter.MEAN], this is
 * the classic Levene's test, which is more powerful under normality.
 *
 * ### Example:
 * ```kotlin
 * val g1 = doubleArrayOf(10.0, 11.0, 12.0, 9.0, 10.0)
 * val g2 = doubleArrayOf(5.0, 15.0, 10.0, 20.0, 0.0)
 * val result = leveneTest(g1, g2)
 * result.statistic                    // W (F-statistic on transformed data)
 * result.pValue                       // p-value from F-distribution
 * result.additionalInfo["dfBetween"]  // between-group degrees of freedom
 * result.additionalInfo["dfWithin"]   // within-group degrees of freedom
 * ```
 *
 * @param groups two or more groups of observations, each with at least 2 elements.
 * @param center the group center statistic used for computing deviations.
 * Defaults to [LeveneCenter.MEDIAN] (Brown-Forsythe variant).
 * @return a [TestResult] containing the W statistic, p-value, and additional info
 * with "dfBetween" and "dfWithin".
 * @throws InsufficientDataException if fewer than 2 groups are provided or any group
 * has fewer than 2 elements.
 */
public fun leveneTest(
    vararg groups: DoubleArray,
    center: LeveneCenter = LeveneCenter.MEDIAN
): TestResult {
    if (groups.size < 2) throw InsufficientDataException(
        "Levene's test requires at least 2 groups, got ${groups.size}"
    )
    for (i in groups.indices) {
        if (groups[i].size < 2) throw InsufficientDataException(
            "Each group must have at least 2 elements, group $i has ${groups[i].size}"
        )
    }

    val k = groups.size

    // Step 1: Transform to absolute deviations from group center
    val z = Array(k) { i ->
        val group = groups[i]
        val c = when (center) {
            LeveneCenter.MEAN -> group.mean()
            LeveneCenter.MEDIAN -> group.median()
        }
        DoubleArray(group.size) { j -> abs(group[j] - c) }
    }

    // Step 2: Compute ANOVA on transformed values
    val groupSizes = IntArray(k) { z[it].size }
    val totalN = groupSizes.sum()

    // Group means of Z
    val zGroupMeans = DoubleArray(k) { z[it].mean() }

    // Grand mean of Z
    var grandSum = 0.0
    for (i in 0 until k) {
        for (j in 0 until z[i].size) {
            grandSum += z[i][j]
        }
    }
    val grandMean = grandSum / totalN

    // SS between
    var ssBetween = 0.0
    for (i in 0 until k) {
        val diff = zGroupMeans[i] - grandMean
        ssBetween += groupSizes[i] * diff * diff
    }

    // SS within
    var ssWithin = 0.0
    for (i in 0 until k) {
        for (j in 0 until z[i].size) {
            val diff = z[i][j] - zGroupMeans[i]
            ssWithin += diff * diff
        }
    }

    val dfBetween = k - 1
    val dfWithin = totalN - k
    val testName = if (center == LeveneCenter.MEDIAN) "Brown-Forsythe Test" else "Levene's Test"

    // Degenerate case: all groups constant → all Z values zero
    if (ssWithin == 0.0 && ssBetween == 0.0) {
        return TestResult(
            testName = testName,
            statistic = 0.0,
            pValue = 1.0,
            degreesOfFreedom = dfBetween.toDouble(),
            additionalInfo = mapOf(
                "dfBetween" to dfBetween.toDouble(),
                "dfWithin" to dfWithin.toDouble()
            )
        )
    }

    val w = (ssBetween / dfBetween) / (ssWithin / dfWithin)

    // Non-finite check (from non-finite input or ssWithin=0 with ssBetween>0)
    if (w.isNaN() || w.isInfinite()) {
        return TestResult(
            testName = testName,
            statistic = w,
            pValue = if (w.isInfinite() && w > 0) 0.0 else Double.NaN,
            degreesOfFreedom = dfBetween.toDouble(),
            additionalInfo = mapOf(
                "dfBetween" to dfBetween.toDouble(),
                "dfWithin" to dfWithin.toDouble()
            )
        )
    }

    val pValue = FDistribution(dfBetween.toDouble(), dfWithin.toDouble()).sf(w)

    return TestResult(
        testName = testName,
        statistic = w,
        pValue = pValue.coerceIn(0.0, 1.0),
        degreesOfFreedom = dfBetween.toDouble(),
        additionalInfo = mapOf(
            "dfBetween" to dfBetween.toDouble(),
            "dfWithin" to dfWithin.toDouble()
        )
    )
}
