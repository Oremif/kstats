package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.distributions.FDistribution

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
 * One-way ANOVA test.
 */
public fun oneWayAnova(vararg groups: DoubleArray): AnovaResult {
    if (groups.size < 2) throw InsufficientDataException("ANOVA requires at least 2 groups")
    if (!groups.all { it.size >= 2 }) throw InsufficientDataException("Each group must have at least 2 elements")

    val k = groups.size
    val groupMeans = groups.map { it.average() }
    val groupSizes = groups.map { it.size }
    val totalN = groupSizes.sum()
    val grandMean = groups.flatMap { it.toList() }.average()

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
