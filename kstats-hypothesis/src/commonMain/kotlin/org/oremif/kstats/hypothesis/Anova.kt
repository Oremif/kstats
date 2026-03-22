package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.distributions.FDistribution

/**
 * The result of a one-way ANOVA test.
 *
 * Contains the F-statistic, p-value, and the full ANOVA decomposition into between-group
 * and within-group components.
 *
 * ### Example:
 * ```kotlin
 * val result = oneWayAnova(group1, group2, group3)
 * result.fStatistic // ratio of between-group to within-group variance
 * result.pValue     // p-value from F-distribution
 * result.dfBetween  // number of groups minus one
 * result.dfWithin   // total observations minus number of groups
 * ```
 *
 * @property fStatistic the F-statistic, computed as the ratio of between-group variance to
 * within-group variance. Larger values indicate greater differences between groups.
 * @property pValue the probability of observing an F-statistic at least as extreme as the
 * computed value, assuming all group means are equal. Smaller values indicate stronger
 * evidence that at least one group mean differs.
 * @property dfBetween the between-group degrees of freedom, equal to the number of groups minus one.
 * @property dfWithin the within-group degrees of freedom, equal to the total number of
 * observations minus the number of groups.
 * @property ssBetween the sum of squares between groups, measuring the variation due to
 * differences between group means.
 * @property ssWithin the sum of squares within groups, measuring the variation within each group.
 * @property msBetween the mean square between groups (ssBetween / dfBetween).
 * @property msWithin the mean square within groups (ssWithin / dfWithin).
 */
public data class AnovaResult(
    val fStatistic: Double,
    val pValue: Double,
    val dfBetween: Int,
    val dfWithin: Int,
    val ssBetween: Double,
    val ssWithin: Double,
    val msBetween: Double,
    val msWithin: Double
)

/**
 * Performs a one-way analysis of variance (ANOVA) test.
 *
 * The null hypothesis is that all group means are equal. ANOVA partitions the total
 * variation in the data into variation between groups and variation within groups,
 * then compares these using an F-test. A significant result indicates that at least
 * one group mean differs from the others, but does not identify which one.
 *
 * ### Example:
 * ```kotlin
 * val group1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
 * val group2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
 * val group3 = doubleArrayOf(11.0, 12.0, 13.0, 14.0, 15.0)
 * val result = oneWayAnova(group1, group2, group3)
 * result.fStatistic // F-statistic
 * result.pValue     // p-value (< 0.001 for clearly different groups)
 * result.dfBetween  // 2 (three groups minus one)
 * result.dfWithin   // 12 (fifteen observations minus three groups)
 * ```
 *
 * @param groups two or more groups of observations, each with at least 2 elements.
 * @return an [AnovaResult] containing the F-statistic, p-value, and the full ANOVA table
 * decomposition (degrees of freedom, sums of squares, mean squares).
 */
public fun oneWayAnova(vararg groups: DoubleArray): AnovaResult {
    if (groups.size < 2) throw InsufficientDataException("ANOVA requires at least 2 groups")
    if (!groups.all { it.size >= 2 }) throw InsufficientDataException("Each group must have at least 2 elements")

    val k = groups.size
    val groupMeans = groups.map { it.average() }
    val groupSizes = groups.map { it.size }
    val totalN = groupSizes.sum()
    var grandSum = 0.0
    for (group in groups) for (x in group) grandSum += x
    val grandMean = grandSum / totalN

    // SS between
    var ssBetween = 0.0
    for (i in groups.indices) {
        ssBetween += groupSizes[i] * (groupMeans[i] - grandMean) * (groupMeans[i] - grandMean)
    }

    // SS within
    var ssWithin = 0.0
    for (i in groups.indices) {
        for (x in groups[i]) {
            ssWithin += (x - groupMeans[i]) * (x - groupMeans[i])
        }
    }

    val dfBetween = k - 1
    val dfWithin = totalN - k
    val msBetween = ssBetween / dfBetween
    val msWithin = ssWithin / dfWithin
    val f = msBetween / msWithin

    val dist = FDistribution(dfBetween.toDouble(), dfWithin.toDouble())
    val pValue = dist.sf(f).coerceIn(0.0, 1.0)

    return AnovaResult(
        fStatistic = f,
        pValue = pValue,
        dfBetween = dfBetween,
        dfWithin = dfWithin,
        ssBetween = ssBetween,
        ssWithin = ssWithin,
        msBetween = msBetween,
        msWithin = msWithin
    )
}
