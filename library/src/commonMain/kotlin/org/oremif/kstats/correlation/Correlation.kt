package org.oremif.kstats.correlation

import org.oremif.kstats.descriptive.PopulationKind
import org.oremif.kstats.descriptive.mean
import org.oremif.kstats.distributions.NormalDistribution
import org.oremif.kstats.distributions.StudentTDistribution
import org.oremif.kstats.sampling.TieMethod
import org.oremif.kstats.sampling.rank
import kotlin.math.abs
import kotlin.math.sqrt

public data class CorrelationResult(
    val coefficient: Double,
    val pValue: Double,
    val n: Int
)

/**
 * Pearson product-moment correlation coefficient with p-value.
 */
public fun pearsonCorrelation(x: DoubleArray, y: DoubleArray): CorrelationResult {
    require(x.size == y.size) { "Arrays must have the same size" }
    val n = x.size
    require(n >= 3) { "Need at least 3 observations" }

    val mx = x.mean()
    val my = y.mean()

    var sxy = 0.0
    var sxx = 0.0
    var syy = 0.0
    for (i in 0 until n) {
        val dx = x[i] - mx
        val dy = y[i] - my
        sxy += dx * dy
        sxx += dx * dx
        syy += dy * dy
    }

    if (sxx == 0.0 || syy == 0.0) {
        return CorrelationResult(Double.NaN, Double.NaN, n)
    }

    val r = sxy / sqrt(sxx * syy)

    // t-test for correlation significance
    val t = r * sqrt((n - 2).toDouble() / (1.0 - r * r))
    val dist = StudentTDistribution((n - 2).toDouble())
    val pValue = 2.0 * dist.sf(abs(t))

    return CorrelationResult(r, pValue.coerceIn(0.0, 1.0), n)
}

/**
 * Spearman rank correlation coefficient.
 */
public fun spearmanCorrelation(x: DoubleArray, y: DoubleArray): CorrelationResult {
    require(x.size == y.size) { "Arrays must have the same size" }
    val n = x.size
    require(n >= 3) { "Need at least 3 observations" }

    val xRanks = x.rank(TieMethod.AVERAGE)
    val yRanks = y.rank(TieMethod.AVERAGE)

    return pearsonCorrelation(xRanks, yRanks)
}

/**
 * Kendall's tau-b correlation coefficient.
 */
public fun kendallTau(x: DoubleArray, y: DoubleArray): CorrelationResult {
    require(x.size == y.size) { "Arrays must have the same size" }
    val n = x.size
    require(n >= 3) { "Need at least 3 observations" }

    var concordant = 0
    var discordant = 0
    var tiedX = 0
    var tiedY = 0

    for (i in 0 until n) {
        for (j in i + 1 until n) {
            val xDiff = x[i].compareTo(x[j])
            val yDiff = y[i].compareTo(y[j])

            if (xDiff == 0 && yDiff == 0) {
                tiedX++
                tiedY++
            } else if (xDiff == 0) {
                tiedX++
            } else if (yDiff == 0) {
                tiedY++
            } else if (xDiff * yDiff > 0) {
                concordant++
            } else {
                discordant++
            }
        }
    }

    val nPairs = n.toLong() * (n - 1) / 2
    val denominator = sqrt((nPairs - tiedX).toDouble() * (nPairs - tiedY).toDouble())

    if (denominator == 0.0) {
        return CorrelationResult(Double.NaN, Double.NaN, n)
    }

    val tau = (concordant - discordant).toDouble() / denominator

    // Normal approximation for p-value
    val sigma = sqrt(2.0 * (2.0 * n + 5.0) / (9.0 * n * (n - 1.0)))
    val z = tau / sigma
    val pValue = 2.0 * NormalDistribution.STANDARD.sf(abs(z))

    return CorrelationResult(tau, pValue.coerceIn(0.0, 1.0), n)
}

/**
 * Covariance between two arrays.
 */
public fun covariance(
    x: DoubleArray,
    y: DoubleArray,
    kind: PopulationKind = PopulationKind.SAMPLE
): Double {
    require(x.size == y.size) { "Arrays must have the same size" }
    val n = x.size
    require(n >= 2) { "Need at least 2 observations" }

    val mx = x.mean()
    val my = y.mean()

    var sum = 0.0
    for (i in 0 until n) {
        sum += (x[i] - mx) * (y[i] - my)
    }

    return sum / (if (kind == PopulationKind.SAMPLE) n - 1 else n)
}

/**
 * Correlation matrix for multiple variables.
 */
public fun correlationMatrix(vararg variables: DoubleArray): Array<DoubleArray> {
    val k = variables.size
    require(k >= 2) { "Need at least 2 variables" }
    val n = variables[0].size
    require(variables.all { it.size == n }) { "All variables must have the same size" }

    return Array(k) { i ->
        DoubleArray(k) { j ->
            if (i == j) 1.0 else pearsonCorrelation(variables[i], variables[j]).coefficient
        }
    }
}

/**
 * Covariance matrix for multiple variables.
 */
public fun covarianceMatrix(
    vararg variables: DoubleArray,
    kind: PopulationKind = PopulationKind.SAMPLE
): Array<DoubleArray> {
    val k = variables.size
    require(k >= 2) { "Need at least 2 variables" }
    val n = variables[0].size
    require(variables.all { it.size == n }) { "All variables must have the same size" }

    return Array(k) { i ->
        DoubleArray(k) { j ->
            covariance(variables[i], variables[j], kind)
        }
    }
}
